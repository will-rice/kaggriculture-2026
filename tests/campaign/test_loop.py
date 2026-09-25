"""The whole loop end to end on real operators, with only codex stood in for.

Everything here is the campaign's own code: real validation, real games
through the real harness and its process pools, a real database, pool and
gate, real ``state.json`` and ``champion.json`` under ``tmp_path``. The one
substitution is the codex call itself -- a ``FakeMutator``, a ``Recorder``,
or a ``CodexMutator`` whose ``COMMAND`` is an ordinary shell command --
because a real call costs quota. Concurrency, the champion and the gate are
therefore observed the way an operator would: through the files the run
wrote, the database it filled, and the metrics it logged.

Three of these are seam tests and play real games all the way through: the
end-to-end one, the one that a promotion changes what the next session
starts from, and the one that the first round is sent the loop's own verdict
on the program it starts from. The rest are about scheduling -- which
program the gate fires on, what a round is handed, what a restart resumes --
and take their numbers from ``stub_evaluator``, because playing 720 turns to
produce a fitness those assertions never read is cost without coverage.
Everything else in them is the real thing, validation included.

Where a test needs the floor to stay where it is, it sets one no candidate
can beat (``strong_champion``) rather than arranging for the gate to fail:
the promotion rule is then the real one, deciding on the numbers it is given.
"""

import asyncio
import inspect
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Coroutine, Sequence
from pathlib import Path
from typing import NamedTuple

import pytest
import wandb
from git import Actor, Repo

from kaggriculture.campaign import (
    archive,
    config,
    dataset,
    evaluator,
    gate,
    harness,
    loop,
    mutate,
    pool,
    prompt,
)

PASS = (
    "def agent(observation, configuration=None):\n"
    "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
)
# Wheat costs 10 a seed, waters up to six units between its second and
# fourth day, and sells at 25 a unit: buy five, work one tile, and the
# episode ends four hundred coins above a PASS agent's untouched opening
# money, on every seed and in both seats.
SELLER = """
def agent(observation, configuration=None):
    day = observation["day"]
    farm = observation["farms"][observation["player"]]
    private = observation["private"]
    x, y = farm["farmer"]
    tile = farm["tiles"][y][x]
    market = [["BUY_SEED", "WHEAT", 5]] if observation["step"] == 0 else []
    wheat = private["shed"].get("WHEAT", 0)
    if wheat:
        market.append(["SELL", "WHEAT", wheat])
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        age = day - tile["planted_day"]
        if age >= 4 and tile["yield_units"]:
            farmer = ["HARVEST"]
        elif not tile["watered_today"]:
            farmer = ["WATER"]
        else:
            farmer = ["PASS"]
    elif tile is None and private["seeds"].get("WHEAT", 0):
        farmer = ["PLANT", "WHEAT"]
    else:
        farmer = ["PASS"]
    return {"farmer": farmer, "hands": [], "market": market}
"""

# A pool opponent that fails in its own seat: not the candidate's failure.
CRASHER = """
def agent(observation, configuration=None):
    raise ZeroDivisionError("the pool is broken")
"""

# Eight sessions can be in flight at once and each may be evaluating, so a
# share has to stay small enough that the smallest box still fits them.
WORKERS = max(1, min(2, config.CORE_BUDGET // config.SESSIONS))


@pytest.fixture
def records() -> list[tuple[float, dict]]:
    """Every metrics dict the loop hands to ``log.log``, with when it did."""
    return []


@pytest.fixture
def log(
    records: list[tuple[float, dict]], monkeypatch: pytest.MonkeyPatch
) -> wandb.Run:
    """A wandb run that records nothing, with its `log` calls kept in ``records``."""
    run = wandb.init(mode="disabled")
    monkeypatch.setattr(
        run, "log", lambda record: records.append((time.time(), record))
    )
    return run


def sessions_of(records: list[tuple[float, dict]]) -> list[dict]:
    """The per-session records, in the order the loop logged them."""
    return [record for _, record in records if "sessions/rounds" in record]


def calls_of(records: list[tuple[float, dict]]) -> list[dict]:
    """The per-round records, in the order the loop logged them."""
    return [record for _, record in records if "calls/ok" in record]


def gates_of(records: list[tuple[float, dict]]) -> list[dict]:
    """The per-gate records, in the order the loop logged them."""
    return [record for _, record in records if "gate/promoted" in record]


def tiny_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rounds: int = 1
) -> config.Run:
    """A campaign of its own under ``tmp_path``, shrunk to one seed.

    Returns the run rather than moving the module's paths to it. Those were
    constants once and a test isolated itself by swapping them, which works
    only while every reader looks them up at call time -- and one did not, so
    a dry run wrote its champions into the live campaign's pairings and the
    real `field.json` ended up holding `champion_1` through `champion_9`.
    There is nothing global left to swap now.

    The one-opponent test pool stands in for the vendored field: these tests
    measure the pipeline, not a field.

    ``rounds`` is one by default, because a test that is not about rounds
    should cost one call and one evaluation; the tests that are about rounds
    ask for more. It is the only constant here that decides behaviour rather
    than cost, which is why it is a parameter and not a line in the body.
    """
    monkeypatch.setattr(config, "ROUNDS_PER_OPPONENT", rounds)
    monkeypatch.setattr(config, "GATE_SEEDS", 1)
    monkeypatch.setattr(evaluator, "VENDORED", ["pass"])
    # The copy check reads every opponent the machine holds, and the campaign
    # now harvests new ones every hour -- so a test that leaves this alone is
    # measured against a corpus that changes underneath it. These fixtures are
    # a few lines each, and a small program's shingle set is small enough that
    # Jaccard against anything at all runs high: on 2026-09-11 `SELLER` came
    # out 0.071 similar to a kernel harvested that morning, against a 0.03
    # bar, and four tests that had passed for weeks began failing on a corpus
    # nobody had touched them with. Pointed at an empty directory, they test
    # the pipeline rather than today's ladder.
    corpus = tmp_path / "no-opponents"
    corpus.mkdir(parents=True, exist_ok=True)
    root = tmp_path / "run"
    return config.Run(root=root, pool=root / "pool.json")


UNVENDORED = Path("/nonexistent-opponents-directory")
"""What these tests pass as the opponents directory.

They build the pool they mean, agent by agent, so there is nothing on disk for
them to adopt. Pointing this at a path that does not exist says that, and keeps
the machine's data box out of a unit test.
"""


def pass_pool(tmp_path: Path, paths: config.Run) -> pool.Pool:
    """A saved one-opponent pool whose opponent is a PASS agent."""
    opponents = pool.Pool(opponents={"pass": str(_write(tmp_path / "pass.py", PASS))})
    opponents.save(paths.pool)
    return opponents


def strong_champion(tmp_path: Path, paths: config.Run) -> gate.Champion:
    """A champion on the floor that nothing can beat, so nothing needs docker.

    The promotion rule asks for a lower bound above the champion's score; a
    champion recorded at 1.0 has no bound above it, so every candidate in
    these tests is measured, compared and turned down without the gate ever
    reaching the image.
    """
    paths.champions.mkdir(parents=True, exist_ok=True)
    kept = _write(paths.champions / "champion_1.py", SELLER)
    opponents = pool.Pool(
        opponents={
            "pass": str(_write(tmp_path / "pass.py", PASS)),
            "champion_1": str(kept),
        }
    )
    opponents.save(paths.pool)
    champion = gate.Champion(
        name="champion_1",
        path=str(kept),
        tarball=str(tmp_path / "champion_1.tar.gz"),
        result=_gate_result("champion_1", {"pass": 1.0}, score=1.0),
    )
    paths.champion.parent.mkdir(parents=True, exist_ok=True)
    paths.champion.write_text(champion.model_dump_json(), encoding="utf-8")
    return champion


def stub_evaluator(monkeypatch: pytest.MonkeyPatch, crashes: bool = False) -> list[str]:
    """Answer the evaluation from the source itself, without playing a game.

    A scheduling test needs a fitness to rank on, not a game to produce one.
    It is patched on the module rather than injected, because the alternative
    is a parameter on ``run`` that exists only for tests. Score is the
    source's length, so a child ranks above the shorter program it was edited
    from by construction rather than by luck, and the rates are keyed by the
    pool the evaluation was handed, reduced through the real
    `evaluator.opponents` so a champion here does not play itself either.

    Args:
        monkeypatch: The test's patcher.
        crashes: Whether every evaluation raises, as one does when the
            candidate fails in its own seat.

    Returns:
        The program ids handed to the evaluation, in order.
    """
    scored: list[str] = []

    def score(agent: Path) -> float:
        """A stand-in fitness: longer source, higher score, never a full 1.0."""
        return min(0.99, len(agent.read_text(encoding="utf-8")) / 1000)

    real = inspect.signature(evaluator.score)

    def measure(*args: object, **given: object) -> evaluator.Result:
        """The evaluation's shape, without its games.

        No day table: a scheduling test asserts on what the loop did with a
        number, never on the states behind it, and the only game that could
        produce one is the game this stands in for.

        Bound against the real signature rather than redeclaring it. A stub
        that names its own parameters drifts the moment the function it stands
        in for gains one, and does it silently: adding `seeds` between `rng`
        and `workers` shifted every later argument by one here, so the pool
        file arrived as the standings and nineteen tests failed a long way
        from the change. Binding makes that a `TypeError` naming the
        parameter, and usually makes it nothing at all.
        """
        bound = real.bind(*args, **given)
        bound.apply_defaults()
        agent = bound.arguments["agent"]
        program_id = bound.arguments["program_id"]
        opponents = bound.arguments["pool"]
        rng = bound.arguments["rng"]
        standings = bound.arguments["standings"]
        always = bound.arguments["always"]
        scored.append(program_id)
        if crashes:
            raise RuntimeError("the candidate raised in its own seat")
        measured = evaluator.opponents(opponents, program_id, agent)
        rate = score(agent)
        # The draw the real evaluation would make, so a test sees the same
        # opponents the gate would: a sample, not the whole pool.
        names = measured.sample(standings or {}, rng, program_id, always)
        return evaluator.Result(
            program_id=program_id,
            fitness=rate,
            field=0.5,
            rates=dict.fromkeys(names, rate),
            # A measured margin, because promotion now asks whether beating
            # the champion was shown rather than by how much: a mean inside
            # its own error refuses, and a zero error reads as unmeasured.
            margins=dict.fromkeys(
                names,
                harness.Margin(mean=5000.0, worst=0.0, best=9000.0, error=500.0),
            ),
            intervals=dict.fromkeys(names, (0.0, 1.0)),
            # Every game decided. A stub that left this empty would be a
            # candidate indistinguishable from the floor, which the gate
            # refuses -- correctly, and not what any of these tests is about.
            decisive=dict.fromkeys(names, config.DECISIVE_GAMES),
            games=2,
            seeds=[1],
            hardest=names[0],
            states=dict.fromkeys(names, []),
        )

    monkeypatch.setattr(evaluator, "score", measure)
    return scored


class Handed(NamedTuple):
    """What one round was given: the program, the message, and the directory.

    Attributes:
        child: The source found in ``child.py``.
        message: The whole prompt, on standard input.
        held: Everything in the directory, by name.
        where: The directory itself, so a test can watch it go away.
    """

    child: str
    message: str
    held: list[str]
    where: Path


class Recorder:
    """A stand-in call that notes what it was handed, then edits ``child.py``.

    The loop removes a round's directory once its program is in the database,
    so what a call was given has to be read while it is running. This is the
    same contract ``FakeMutator`` meets, and nothing but the driving process is
    stood in for -- including where that process reads its skills, which the
    loop asks the mutator for rather than deciding itself.
    """

    SKILLS_DIRS: tuple[Path, ...] = (Path(".agents") / "skills",)
    TRANSCRIPTS: tuple[str, ...] = ("agy*.jsonl",)

    def __init__(self, edit: Callable[[str], str]) -> None:
        """Initializes the recorder.

        Args:
            edit: Turns what ``child.py`` holds into what the round writes.
        """
        self.edit = edit
        self.seen: list[Handed] = []

    async def __call__(
        self, workspace: Path, message: str, program_id: str
    ) -> mutate.Mutation:
        """Note what the round was handed, then write the child.

        Args:
            workspace: The directory built for this round.
            message: The composed prompt.
            program_id: The child program id.

        Returns:
            A `Mutation` with status "ok".
        """
        child = workspace / "child.py"
        source = child.read_text(encoding="utf-8")
        self.seen.append(
            Handed(
                source,
                message,
                sorted(item.name for item in workspace.iterdir()),
                workspace,
            )
        )
        child.write_text(self.edit(source), encoding="utf-8")
        # Every driver leaves one of these, and the loop copies it out before
        # the workspace goes. Written after the round has noted what it was
        # handed, because a transcript is what the round leaves rather than
        # something it was given.
        (workspace / "agy.jsonl").write_text(
            f'{{"round": "{program_id}"}}\n', encoding="utf-8"
        )
        return mutate.Mutation(
            program_id=program_id,
            child=child,
            status="ok",
            reason="",
            seconds=0.0,
            input_tokens=0,
            output_tokens=0,
            model="recorder",
        )


@pytest.mark.slow
# `test_the_pool_plays_itself_before_anything_is_judged_against_it` stood here
# until 2026-09-13. It asserted that a launch fills `paths.field` with the
# pool's own pairings, so that a Bradley-Terry fit over the field is a
# tournament rather than one candidate's row.
#
# The rule it protected is gone. Ratings decided promotion then, and a champion
# that joined the pool with no pairings was rated almost entirely from the row
# of whoever was being judged against it -- beat it, and its rating fell far
# enough that each promotion bought the next one cheaply. Promotion is now a
# win rate over the field and a head-to-head against the champion, both
# measured inside the candidate's own evaluation, and neither reads a fit. The
# refresh that kept the field current was removed with it: it cost 8.5 minutes
# of the promotions lock per promotion, measured 2026-09-13, with three
# candidates queued behind the first one of that run.
#
# So nothing writes the field and nothing downstream of it decides anything.
# What remains of that machinery -- `gate.refresh`, `evaluator.score`'s unused
# `standings` argument, and the fit `measure` makes to feed it -- is dead and
# wants deleting, which is a change to `src/` rather than to a test.


@pytest.mark.slow
def test_a_better_child_is_promoted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    log: wandb.Run,
    records: list[tuple[float, dict]],
) -> None:
    """Two sessions, no schedule: the better child is confirmed and shipped.

    A seam test: real validation, real games through the real harness, the
    real gate, and only the codex session stood in for.

    Six seeds rather than `tiny_run`'s one, because the gate stopped promoting
    on a Bradley-Terry rank and started asking for evidence. At one seed a
    pairing is two games, and neither bar can be met at that depth however much
    better the child is: the field bar is twice the error of the difference,
    which at one shared opponent and two games is 0.707 against a maximum
    possible gap of 0.5, and the head-to-head needs `DECISIVE_GAMES` decided
    games where two exist. Twelve games a pairing puts the field bar at 0.289
    and the Wilson lower bound on a clean sweep at 0.676, and leaves room for a
    draw or two without dropping under the decisive floor.
    """
    paths = tiny_run(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "GATE_SEEDS", 6)
    pass_pool(tmp_path, paths)

    state = loop.run(
        sessions=2,
        mutator=mutate.FakeMutator(edit=lambda _: SELLER),
        workers=WORKERS,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        log=log,
        paths=paths,
        opponents=UNVENDORED,
    )

    assert state.sessions == 2
    champion = state.champion
    assert champion is not None and champion.result.fitness > 0.5
    assert Path(champion.path).read_text() == SELLER
    assert (paths.floor / "main.py").read_text() == SELLER
    assert Path(champion.tarball).exists()
    assert champion.name in pool.Pool.load(paths.pool).names()
    assert gate.load_champion(paths) == champion
    promotions = [record for record in gates_of(records) if record["gate/promoted"]]
    assert promotions and all(record["gate/score"] > 0.5 for record in promotions)
    # The champion carries the result it was promoted on: a program the
    # database holds, measured on the exam block, and what a session is then
    # shown of the program it starts from.
    database = archive.Database(paths.archive, paths.programs)
    assert champion.result.program_id in {p.id for p in database.programs}
    assert [record["sessions"] for record in sessions_of(records)] == [1, 2]


@pytest.mark.slow
def test_a_promotion_leaves_a_tree_the_next_launch_can_start_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """A campaign that promotes must still be a campaign that can restart.

    Every write a promotion makes is under ``run/campaign``. One that landed
    in ``src/`` -- ``served/main.py`` was such a write -- would dirty a
    tracked file, and ``_open_run`` refuses to start a run whose ``src/`` has
    uncommitted changes: the first promotion would be the last thing that
    campaign ever did.
    """
    paths = tiny_run(tmp_path, monkeypatch)
    repo = _repository(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "SERVED", tmp_path / "src" / "served" / "main.py")
    pass_pool(tmp_path, paths)
    stub_evaluator(monkeypatch)

    state = loop.run(
        sessions=1,
        mutator=mutate.FakeMutator(edit=lambda _: SELLER),
        workers=WORKERS,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        log=log,
        paths=paths,
        opponents=UNVENDORED,
    )

    # `config.ROOT` is the real checkout until here, because packaging a
    # champion reads the plumbing modules out of it. From here it is the
    # repository the promotion could have dirtied.
    assert state.champion is not None
    monkeypatch.setattr(config, "ROOT", tmp_path)
    assert repo.git.status("--porcelain", "--", "src") == ""
    loop._open_run(dry_run=True).finish()


def test_the_seed_is_champion_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """The seed is enthroned at startup, so there is no pre-champion regime.

    It was forbidden until 2026-09-12, on the grounds that `champion_1` would be
    "an opponent every candidate already beats: a constant added to every score
    that separates none of them". That objection was about the pool win rate,
    which no longer decides a promotion -- the bar is the head-to-head -- and it
    is one slot in a pool of forty-one.

    What enthroning it buys is the absence of a second regime. Every session
    starts from a champion and every candidate plays one, so the bar is the same
    single condition from the first round instead of a branch for having nothing
    to beat. Each champion after it beats the one before by 22 of 32, which is a
    ratchet whether the first rung is high or low.
    """
    paths = tiny_run(tmp_path, monkeypatch)
    pass_pool(tmp_path, paths)
    scored = stub_evaluator(monkeypatch)

    state = loop.run(
        sessions=1,
        # A child that ranks *below* the seed, so the seed is the top program.
        mutator=mutate.FakeMutator(edit=lambda _: PASS),
        workers=WORKERS,
        seed_agent=_write(tmp_path / "seed.py", SELLER),
        rng=random.Random(0),
        log=log,
        paths=paths,
        opponents=UNVENDORED,
    )

    # Measured at the cold start, then enthroned: in the pool, on the floor,
    # and the thing the first candidate is asked to beat.
    assert config.SEED_ID in scored
    assert state.champion is not None
    # The pool key is fixed; the file carries the number.
    assert state.champion.name == config.POOL_CHAMPION
    assert Path(state.champion.path).name == "champion_1.py"
    assert (paths.floor / "main.py").exists()
    # No tarball: nothing would ever submit the seed, and packaging needs the
    # licence and the served skeleton that a bare run directory has not got.
    assert state.champion.tarball == ""


@pytest.mark.slow
def test_eight_workers_run_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """Every worker's session is in flight together, not one after another.

    A barrier rather than two timestamps compared afterwards. Each session
    blocks in its edit until all ``SESSIONS`` of them have arrived, so the run
    can only finish if they genuinely overlapped -- if any pair were
    serialised the barrier would never fill and the wait would time out.

    The timestamp version of this asserted that the latest start preceded the
    earliest stop, which is the same claim measured badly: it held only while
    the runner scheduled eight threads promptly, and it failed on CI by seven
    tenths of a millisecond. Widening the sleep would have bought margin and
    weakened the claim; the barrier removes the timing from it altogether.

    Run as if on a one-core machine, because that is the condition that found
    the defect underneath. `asyncio.to_thread` borrows a default executor
    sized `cpu_count + 4`, and every thread here waits on a subprocess or a
    file rather than computing, so the core count is the wrong basis: on CI's
    four cores that pool was eight threads against eight sessions and two deep
    evaluations, and the sessions that could not get one simply queued. The
    loop now brings a pool sized by its own concurrency, and pinning
    `cpu_count` at one proves it on any machine -- without the pinning this
    passes on a large box whatever the loop does.
    """
    monkeypatch.setattr(os, "cpu_count", lambda: 1)
    paths = tiny_run(tmp_path, monkeypatch)
    strong_champion(tmp_path, paths)
    stub_evaluator(monkeypatch)
    # Not a measurement: the sessions reach this in well under a second when
    # they run at all, so thirty seconds is sixty times the margin the old
    # timestamps asked for. It exists only so a real serialisation fails the
    # suite instead of hanging it.
    gate = threading.Barrier(config.SESSIONS, timeout=30)
    # A list, because `append` is atomic and `+= 1` from eight threads is not.
    arrived: list[int] = []

    def edit(source: str) -> str:
        """Block until every other session has reached this point too."""
        index = gate.wait()
        arrived.append(index)
        return f"{source}\n# session {index}\n"

    state = loop.run(
        sessions=config.SESSIONS,
        mutator=mutate.FakeMutator(edit=edit),
        workers=WORKERS,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        log=log,
        paths=paths,
        opponents=UNVENDORED,
    )

    assert state.sessions == config.SESSIONS
    assert sorted(arrived) == list(range(config.SESSIONS))
    assert not gate.broken


@pytest.mark.slow
def test_a_promotion_changes_what_the_next_session_starts_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """The champion a promotion wrote is the file the next session edits.

    A seam test: the champion here is one a real promotion produced, out of
    real games, and the second run reads it back off disk.

    The file is `champion_2.py`, not `champion_1.py`: the seed is enthroned as
    champion zero at startup and takes the first number, so the first program a
    round wins with is the second champion. The numbering is the file's and the
    pool key is fixed, which is why the second run's message names
    `config.POOL_CHAMPION` -- see `gate.promote`.

    Six seeds for the same reason as `test_a_better_child_is_promoted`: the
    two-condition gate cannot be satisfied at `tiny_run`'s one.
    """
    paths = tiny_run(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "GATE_SEEDS", 6)
    pass_pool(tmp_path, paths)
    seed = _write(tmp_path / "seed.py", PASS)
    mutator = Recorder(edit=lambda _: SELLER)

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log, paths, UNVENDORED)

    assert [handed.child for handed in mutator.seen] == [PASS]
    assert (paths.champions / "champion_2.py").read_text() == SELLER

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log, paths, UNVENDORED)

    assert mutator.seen[1].child == SELLER
    assert f"`{config.POOL_CHAMPION}m" in mutator.seen[1].message


@pytest.mark.slow
def test_a_program_is_gated_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """A program a round wrote is scored once and never scored again.

    Four restarts, so every session's gates have finished before the next
    session opens: a database that forgot which programs carry a result would
    spend the block on them over and over.

    The champion is the exception and is excluded here. Until 2026-09-13 this
    asserted that *nothing* was scored twice, which was true of the two-stage
    gate: a program was ranked cheaply, then a shortlist went to a sealed block,
    and the memo existed so the block was never spent twice. There is one
    evaluation now, and a session opens by measuring the program it starts from
    -- that measurement is the baseline the round's edits are compared against
    and the game the round is shown -- so the champion is measured once per
    session by design, on whatever seed block is current.
    """
    paths = tiny_run(tmp_path, monkeypatch)
    champion = strong_champion(tmp_path, paths)
    scored = stub_evaluator(monkeypatch)
    seed = _write(tmp_path / "seed.py", PASS)
    for session in range(4):
        mutator = mutate.FakeMutator(edit=lambda s, n=session: f"{s}\n# child {n}\n")
        loop.run(
            1, mutator, WORKERS, seed, random.Random(session), log, paths, UNVENDORED
        )

    written = [name for name in scored if name not in {champion.name, config.SEED_ID}]
    assert len(written) > 2
    assert sorted(written) == sorted(set(written))
    # And the champion is the only thing measured more than once.
    assert scored.count(champion.name) == 4


def test_the_pool_is_changed_on_the_loop_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """A promotion's file work goes to a thread; the pool it joins does not.

    ``gate.promote`` blocks for as long as packaging takes, so it runs in a
    thread. The pool is a different matter: it is the one object every other
    session is reading -- `session` for the names it shows, `evaluate` for
    the copy an evaluation keeps -- and adding a champion rebuilds that dict.
    Off the loop that is a dictionary changing size while a coroutine walks
    it, which is a run that dies or a score against a pool that never was.
    """
    paths = tiny_run(tmp_path, monkeypatch)
    pass_pool(tmp_path, paths)
    stub_evaluator(monkeypatch)
    threads: list[str] = []
    add_champion = pool.Pool.add_champion

    def watched(self: pool.Pool, path: str) -> None:
        """The real pool change, with a note of the thread that made it."""
        threads.append(threading.current_thread().name)
        add_champion(self, path)

    monkeypatch.setattr(pool.Pool, "add_champion", watched)

    state = loop.run(
        sessions=1,
        mutator=mutate.FakeMutator(edit=lambda _: SELLER),
        workers=WORKERS,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        log=log,
        paths=paths,
        opponents=UNVENDORED,
    )

    assert state.champion is not None
    # Every change, not one: champion zero enrols at startup and a promotion
    # enrols again. What matters is that no other thread ever appears.
    assert threads
    assert set(threads) == {threading.main_thread().name}


def test_a_promotion_logs_the_tarball_a_cut_uploads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """The artifact carries the file a cut uploads, not the champion's source.

    A cut is one command -- upload the champion's tarball -- so the run has
    to have kept that file. Logging the ``.py`` beside it would look right on
    the page and leave nothing to upload.
    """
    paths = tiny_run(tmp_path, monkeypatch)
    pass_pool(tmp_path, paths)
    stub_evaluator(monkeypatch)
    artifacts: list[wandb.Artifact] = []
    monkeypatch.setattr(log, "log_artifact", artifacts.append)

    loop.run(
        sessions=1,
        mutator=mutate.FakeMutator(edit=lambda _: SELLER),
        workers=WORKERS,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        log=log,
        paths=paths,
        opponents=UNVENDORED,
    )

    # One artifact: champion zero is not packaged, so only the promotion that
    # followed it is logged, under the fixed pool key with the numbered tarball.
    assert [artifact.name for artifact in artifacts] == [config.POOL_CHAMPION]
    assert list(artifacts[0].manifest.entries) == ["champion_2.tar.gz"]
    # champion_2: champion zero took the first name and is not packaged.
    assert (paths.champions / "champion_2.tar.gz").exists()


def test_a_provider_failure_is_not_the_lineages_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    log: wandb.Run,
    records: list[tuple[float, dict]],
) -> None:
    """A session that never ran to a verdict leaves no failure on the lineage.

    A failure becomes the lineage's feedback in the next prompt; a model at
    capacity is not something the next child can fix. So this is the one
    outcome that ends the session outright -- there is nothing to tell a next
    round -- and the two rounds it had left go to a fresh session instead.
    """
    paths = tiny_run(tmp_path, monkeypatch, rounds=3)
    pass_pool(tmp_path, paths)
    stub_evaluator(monkeypatch)
    monkeypatch.setattr(
        mutate.CodexMutator,
        "COMMAND",
        [
            "bash",
            "-c",
            'printf \'%s\\n\' \'{"type":"turn.failed","error":'
            '{"message":"Selected model is at capacity."}}\'; exit 1',
        ],
    )

    state = loop.run(
        sessions=1,
        mutator=mutate.CodexMutator(model="a", fallback="b"),
        workers=WORKERS,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        log=log,
        paths=paths,
        opponents=UNVENDORED,
    )

    assert state.sessions == 1
    assert '"event": "failure"' not in paths.archive.read_text()
    assert [record["calls/ok"] for record in calls_of(records)] == [0]
    assert [record["calls/fallback"] for record in calls_of(records)] == [1]
    assert [record["sessions/rounds"] for record in sessions_of(records)] == [1]


def test_a_broken_pool_opponent_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """An opponent crashing in its own seat halts the loop rather than the lineage."""
    paths = tiny_run(tmp_path, monkeypatch)
    crasher = pool.Pool(
        opponents={"crasher": str(_write(tmp_path / "crasher.py", CRASHER))}
    )
    crasher.save(paths.pool)
    # A cold start would evaluate the seed against this pool before the loop
    # ever began, so the database is seeded by hand.
    seed = _write(tmp_path / "seed.py", PASS)
    database = archive.Database(paths.archive, paths.programs)
    stored = database.store(PASS, "seed")
    database.add(
        archive.Program(
            id="seed",
            source_path=str(stored),
            started_from="",
            instruction="seed",
            model="",
            fitness=0.5,
            field=0.5,
            created=time.time(),
        )
    )

    with pytest.raises(harness.OpponentCrash):
        loop.run(
            sessions=1,
            mutator=mutate.FakeMutator(edit=lambda _: SELLER),
            workers=WORKERS,
            seed_agent=seed,
            rng=random.Random(0),
            log=log,
            paths=paths,
            opponents=UNVENDORED,
        )

    assert '"event": "failure"' not in paths.archive.read_text()


def test_cancellation_kills_the_session_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """A signal stops the run and takes the session's whole process group with it.

    A session that outlived the loop would keep a core and write into a
    sandbox nothing owns any more.
    """
    paths = tiny_run(tmp_path, monkeypatch)
    pass_pool(tmp_path, paths)
    stub_evaluator(monkeypatch)
    pgid_file = tmp_path / "pgid"
    monkeypatch.setattr(
        mutate.CodexMutator,
        "COMMAND",
        ["bash", "-c", f"ps -o pgid= -p $$ | tr -d ' ' > {pgid_file}; sleep 30"],
    )

    def interrupt() -> None:
        """Signal the loop once the session it started is running."""
        while not pgid_file.exists():
            time.sleep(0.05)
        time.sleep(0.2)
        os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=interrupt, daemon=True).start()
    state = loop.run(
        sessions=1,
        mutator=mutate.CodexMutator(),
        workers=WORKERS,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        log=log,
        paths=paths,
        opponents=UNVENDORED,
    )

    assert state.sessions == 0
    pgid = int(pgid_file.read_text().strip())
    for _ in range(40):
        survivors = subprocess.run(
            ["pgrep", "-g", str(pgid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if survivors.returncode != 0:
            break
        time.sleep(0.1)
    else:
        pytest.fail(f"process group {pgid} outlived the loop that started it")


@pytest.mark.slow
def test_stagnation_switches_the_starting_program(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """With no promotion in sight, a session starts from the top ten and is told so."""
    paths = tiny_run(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "STAGNATION_SESSIONS", 1)
    strong_champion(tmp_path, paths)
    stub_evaluator(monkeypatch)
    seed = _write(tmp_path / "seed.py", PASS)
    marked = f"{SELLER}\n# a child of the champion\n"
    mutator = Recorder(edit=lambda _: marked)

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log, paths, UNVENDORED)
    loop.run(1, mutator, WORKERS, seed, random.Random(0), log, paths, UNVENDORED)

    first, second = mutator.seen
    assert first.child == SELLER and "no promotion" not in first.message
    assert second.child in {PASS, marked} and second.child != SELLER
    assert "no promotion" in second.message and "top ten" in second.message


def test_a_dirty_src_refuses_to_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run named by a revision has to be that revision.

    ``tiny_run`` first, though nothing here should reach a runtime path: this
    is one of two tests that call ``main``, and a `main` that stopped refusing
    would otherwise reach for files under ``tmp_path`` rather than stopping at
    the guard.
    """
    paths = tiny_run(tmp_path, monkeypatch)
    _repository(tmp_path, monkeypatch)
    (tmp_path / "src" / "agent.py").write_text("VERSION = 2\n", encoding="utf-8")
    monkeypatch.setattr(config, "ROOT", tmp_path)

    with pytest.raises(SystemExit, match="src/agent.py"):
        loop.main(["--sessions", "0", "--dry-run", "--run-root", str(paths.root)])


def test_a_bad_model_name_refuses_to_start_before_opening_a_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A typo'd model is caught before wandb opens or a call is ever made.

    Real ``--dry-run`` never reaches ``validate_model`` -- a fake mutator
    spends no codex call -- so this is the other of the two tests that call
    ``main``, and it never runs with ``--dry-run``: a dirty-``src`` check or
    a live wandb run would otherwise have to be arranged just to reach the
    check this test is about.

    The typo goes in an ``.env`` of this test's own, because that file is
    where the slug is chosen: `mutate.model` reloads it with ``override=True``
    on every read, so a slug set any other way -- a patched constant, an
    exported variable -- is overwritten by the host's real file before
    ``main`` ever sees it.
    """
    env = tmp_path / ".env"
    # And the driver, because the campaign runs `agy` by default and each
    # program's models are checked against its own catalog: without this the
    # typo is never looked at, which is exactly what happened when the default
    # moved.
    env.write_text(
        "CAMPAIGN_MUTATOR=codex\nCAMPAIGN_CODEX_MODEL=gpt-5.6-astra\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mutate, "ENV", env)
    monkeypatch.setattr(
        mutate,
        "MODEL_CATALOG_COMMAND",
        ["printf", "%s", '{"models":[{"slug":"gpt-6-astra"}]}'],
    )

    with pytest.raises(SystemExit, match="gpt-5.6-astra"):
        loop.main(["--sessions", "0"])


def test_a_clean_tree_names_the_run_and_a_second_launch_resumes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run id is the revision alone, so a restart lands on the same run.

    The model was in the name until it became a thing `.env` chooses at the
    call. A wandb id is fixed for the life of the run, so a name carrying a
    value that can move without a restart is wrong from the first round that
    moves it, and there is no correcting it afterwards.
    """
    repo = _repository(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "ROOT", tmp_path)
    expected = repo.head.commit.hexsha[:7]

    first = loop._open_run(dry_run=True)
    first.finish()
    second = loop._open_run(dry_run=True)
    second.finish()

    assert first.name == expected and first.id == expected
    assert second.id == first.id
    assert config.CODEX_MODEL not in first.id


def test_a_restart_resumes_state_json_and_champion_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """`state.json` carries the counters, `champion.json` the floor it never saw.

    `state.json` is written after every session and `champion.json` before
    `gate.promote` returns, so a kill in between leaves the first stale: the
    second wins, or the gate would restart with no baseline and promote a
    second time.
    """
    paths = tiny_run(tmp_path, monkeypatch)
    champion = strong_champion(tmp_path, paths)
    stub_evaluator(monkeypatch)
    state_file = paths.archive.with_name("state.json")
    state_file.parent.mkdir(parents=True, exist_ok=True)
    # A champion in `state.json` too, and an older one: this is the kill the
    # precedence exists for, and with the field left empty it would be enough
    # for `champion.json` merely to be read rather than to win.
    stale = gate.Champion(
        name="champion_0",
        path=str(tmp_path / "champion_0.py"),
        tarball=str(tmp_path / "champion_0.tar.gz"),
        result=_gate_result("champion_0", {"pass": 0.6}),
    )
    state_file.write_text(
        loop.State(
            sessions=7, sessions_since_promotion=3, champion=stale
        ).model_dump_json(),
        encoding="utf-8",
    )

    state = loop.run(
        sessions=0,
        mutator=mutate.FakeMutator(edit=lambda source: source),
        workers=WORKERS,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        log=log,
        paths=paths,
        opponents=UNVENDORED,
    )

    assert state.sessions == 7 and state.sessions_since_promotion == 3
    assert state.champion == champion
    assert state.champion != stale


@pytest.mark.slow
def test_a_round_is_told_a_name_and_never_a_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """What a round starts from is interpolated raw, so it is a name, not a path.

    The program a round is editing reaches the message as its pool name, in the
    episode key of the game the round is shown. A path there would point at a
    file it must not read. Everything a model is given is this one string, so
    this is the whole exposure.

    The evaluation is the real one rather than the stub, because the stub
    reports no recorded games and the name is interpolated into the game
    section: stubbed, this asserted on a message that had no section to put a
    name in, and held whatever `compose` did.

    It is the name of the program being edited and not of any opponent. The
    champion is itself a pool opponent, and no opponent is named anywhere in the
    message -- see `test_no_opponent_is_named_anywhere_in_the_message` for the
    measurement behind that.
    """
    paths = tiny_run(tmp_path, monkeypatch)
    champion = strong_champion(tmp_path, paths)
    mutator = Recorder(edit=lambda _: SELLER)
    seed = _write(tmp_path / "seed.py", PASS)

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log, paths, UNVENDORED)

    handed = mutator.seen[0]
    # The episode key is `<name>m<matchup>s<season>`, so the name is a prefix
    # inside the quotes rather than the whole of what they hold.
    assert f"`{champion.name}m" in handed.message
    assert str(tmp_path) not in handed.message
    assert "/data/kaggriculture/campaign" not in handed.message
    # The doctrine lives in `round_prompt.md` now, so this asserts on what
    # was actually delivered rather than on a constant that could drift.
    assert "keep the notice" in handed.message, "the licence obligation travels"


def test_a_round_drawn_from_the_database_is_told_an_id_and_never_a_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """`start`'s other branch, which is the one a campaign opens in.

    Until the first promotion there is no champion to start from, so every
    session draws from the database's top ten, and under stagnation a
    campaign that has one comes back here. The draw holds a `Program`, whose
    id and whose source path are two fields of the same record, so the
    doctrine binds this branch exactly as it binds the champion's -- and
    testing only the champion's left the branch that runs first uncovered.

    It used to assert the id reached the round as well as the path not
    reaching it. The id no longer travels either -- a round is not told what
    the program it is editing is called, because a name is a handle for
    fitting -- and that half is asserted in `test_prompt`, against the
    composed message, where "seed" is not also a word in the game's rules.
    What is left here is the branch: this session drew from the database, and
    no path into the campaign's own tree reached the call. The opponents'
    directory is a different matter since 2026-09-15 and is named on purpose;
    the archive, the champions and the pool are not.
    """
    paths = tiny_run(tmp_path, monkeypatch)
    pass_pool(tmp_path, paths)
    stub_evaluator(monkeypatch)
    mutator = Recorder(edit=lambda source: source + "\n# edited\n")
    seed = _write(tmp_path / "seed.py", PASS)

    state = loop.run(
        1, mutator, WORKERS, seed, random.Random(0), log, paths, UNVENDORED
    )

    # Champion zero, so a session is handed a champion's name rather than an id.
    assert state.champion is not None
    handed = mutator.seen[0]
    assert str(tmp_path) not in handed.message
    assert "/data/kaggriculture/campaign" not in handed.message
    # The doctrine lives in `round_prompt.md` now, so this asserts on what
    # was actually delivered rather than on a constant that could drift.
    assert "keep the notice" in handed.message, "the licence obligation travels"


def _repository(root: Path, monkeypatch: pytest.MonkeyPatch) -> Repo:
    """A repository of its own, with one committed file under ``src/``.

    Every ``GIT_*`` variable is cleared first: pre-commit runs the suite with
    ``GIT_DIR`` and ``GIT_INDEX_FILE`` pointing at the repository being
    committed to, and an init under those would build this test's repository
    there rather than here.
    """
    for name in [name for name in os.environ if name.startswith("GIT_")]:
        monkeypatch.delenv(name)
    repo = Repo.init(root)
    (root / "src").mkdir()
    (root / "src" / "agent.py").write_text("VERSION = 1\n", encoding="utf-8")
    repo.index.add(["src/agent.py"])
    author = Actor("Campaign Tests", "tests@example.com")
    repo.index.commit("seed", author=author, committer=author)
    return repo


def _write(path: Path, source: str) -> Path:
    """Write ``source`` to ``path`` and return it."""
    path.write_text(source, encoding="utf-8")
    return path


def _gate_result(
    program_id: str,
    rates: dict[str, float],
    score: float | None = None,
    days: int = 0,
    seasons: int = 1,
) -> evaluator.Result:
    """A hand-built result standing in for one the gate measured.

    ``days`` and ``seasons`` record that many seasons against each opponent,
    which is what the round writes to ``seasons.csv``. No days by default:
    most of these tests are about what the loop does with a verdict, not about
    the games behind it.
    """
    point = sum(rates.values()) / len(rates) if score is None else score
    return evaluator.Result(
        program_id=program_id,
        fitness=point,
        field=point,
        rates=rates,
        margins={n: harness.Margin(mean=0.0, worst=0.0, best=0.0) for n in rates},
        intervals={
            n: (max(0.0, r - 0.05), min(1.0, r + 0.05)) for n, r in rates.items()
        },
        decisive=dict.fromkeys(rates, config.DECISIVE_GAMES),
        games=4,
        seeds=[1],
        hardest=min(rates, default=""),
        states={
            name: [
                _played(
                    [
                        harness.Day(
                            day=n,
                            ours_bank=100.0 + n,
                            theirs_bank=200.0,
                            ours_plants={"WHEAT": 4},
                            theirs_plants={"MELON": 2},
                            ours_animals={},
                            theirs_animals={"COW": 1},
                            ours_weeds=0,
                            theirs_weeds=3,
                            ours_seeds={"WHEAT": 5},
                            ours_shed={"WHEAT": 12},
                            theirs_shed={"EGG": 3},
                            ours_hands=2,
                            theirs_hands=1,
                            ours=dict.fromkeys(dataset.COLUMNS, 0.0)
                            | {"bank": 100.0 + n},
                            theirs=dict.fromkeys(dataset.COLUMNS, 0.0)
                            | {"bank": 200.0},
                            prices={"WHEAT": 25},
                        )
                        for n in range(days)
                    ]
                )
                for _ in range(seasons)
            ]
            for name in rates
        },
    )


def _played(days: list[harness.Day]) -> harness.Game:
    """One game carrying those days, as the evaluator now records them.

    `Result.states` holds whole games rather than day tables alone: a `Game`
    knows the seat it was played from, and the seat is what lets the days be
    written out the way the public corpus writes them.
    """
    return harness.Game(
        opponent="v54",
        seed=101,
        seat=0,
        ours=days[-1].ours_bank if days else 0.0,
        theirs=days[-1].theirs_bank if days else 0.0,
        worst_step_seconds=0.0,
        days=days,
    )


@pytest.mark.slow
def test_a_session_is_rounds_and_each_continues_from_the_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    log: wandb.Run,
    records: list[tuple[float, dict]],
) -> None:
    """Rounds go deeper on one line, and every one of them is kept.

    A round continues from its own previous program -- that is the whole
    difference between a round and a session, which starts again from the
    champion -- and every round's program is scored and inserted, so three
    rounds leave three programs in the database beside the seed.

    Each edit here lengthens the source and the stub scores on length, so every
    round genuinely improves on the one before it. That is the condition:
    `test_a_round_that_lost_ground_is_not_what_the_next_one_builds_on` is the
    same session when it does not hold.
    """
    paths = tiny_run(tmp_path, monkeypatch, rounds=3)
    pass_pool(tmp_path, paths)
    stub_evaluator(monkeypatch)
    seed = _write(tmp_path / "seed.py", PASS)
    mutator = Recorder(edit=lambda source: source + "# a round\n")

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log, paths, UNVENDORED)

    handed = [given.child for given in mutator.seen]
    assert handed == [PASS, PASS + "# a round\n", PASS + "# a round\n" * 2]
    database = archive.Database(paths.archive, paths.programs)
    written = [p for p in database.programs if p.id != config.SEED_ID]
    assert len(written) == 3
    assert all(program.rates for program in written)
    assert len(calls_of(records)) == 3
    assert [record["sessions/rounds"] for record in sessions_of(records)] == [3]


def test_a_round_that_lost_ground_is_not_what_the_next_one_builds_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """A session climbs from its best program, not from its latest one.

    Measured 2026-09-13: one session wrote a program scoring 0.000, then spent
    three more rounds editing that, while the program it had started from
    scored 0.316. Four of the run's thirteen rounds went on a lineage already
    known to be broken, because a session continued from whatever its last
    round happened to write. That is a random walk; taking the better of the
    two is a climb.

    The second round here shortens the source, which the stub scores lower, so
    the third round is handed the first round's program again -- and is told
    that the discarded one exists and what it cost.
    """
    paths = tiny_run(tmp_path, monkeypatch, rounds=3)
    pass_pool(tmp_path, paths)
    stub_evaluator(monkeypatch)
    seed = _write(tmp_path / "seed.py", PASS)
    better = PASS + "# a longer program\n"
    written = iter([better, PASS, better + "# and further\n"])
    mutator = Recorder(edit=lambda _: next(written))

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log, paths, UNVENDORED)

    first, second, third = mutator.seen
    assert first.child == PASS
    assert second.child == better
    # Not `PASS`, which is what the second round wrote and scored worse for.
    assert third.child == better
    # The regression is still in the database -- it was played, and what it
    # cost is worth telling the next round -- and it reaches that round as a
    # file it can diff rather than as prose.
    assert "## Edits already tried on" in third.message
    assert "`tried_1.py` scored" in third.message
    assert "tried_1.py" in third.held
    database = archive.Database(paths.archive, paths.programs)
    assert len([p for p in database.programs if p.id != config.SEED_ID]) == 3


def test_the_database_records_which_model_wrote_each_program(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """A program remembers which model wrote it.

    That is what lets a block of quota be judged after the fact, even once
    ``config.CODEX_MODEL`` has moved on to another value. The seed carries no
    model: no session wrote it.
    """
    paths = tiny_run(tmp_path, monkeypatch)
    pass_pool(tmp_path, paths)
    stub_evaluator(monkeypatch)
    seed = _write(tmp_path / "seed.py", PASS)
    mutator = Recorder(edit=lambda source: source + "# a round\n")

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log, paths, UNVENDORED)

    database = archive.Database(paths.archive, paths.programs)
    assert database.get(config.SEED_ID).model == ""
    written = [p for p in database.programs if p.id != config.SEED_ID]
    assert len(written) == 1 and written[0].model == "recorder"


def test_a_round_that_clears_the_bar_ends_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """There is nothing left to ask for once a program clears the bar.

    The condition is `gate.promotion` on the round's own rates -- the same
    function, on the same reading of better, that the message told the model
    it had to clear. The bar is a rating margin above the floor now rather
    than a rank, so a program can lead the field and still be asked to go
    again; what ends the session is clearing, and it ends it immediately.
    """
    paths = tiny_run(tmp_path, monkeypatch, rounds=3)
    pass_pool(tmp_path, paths)
    stub_evaluator(monkeypatch)
    seed = _write(tmp_path / "seed.py", PASS)
    # Long enough that the stand-in fitness puts it above half against every
    # opponent, which is what beating them all means.
    mutator = Recorder(edit=lambda _: SELLER)

    state = loop.run(
        1, mutator, WORKERS, seed, random.Random(0), log, paths, UNVENDORED
    )

    # Ended before the round cap, and ended because the floor moved.
    assert 0 < len(mutator.seen) < 3
    assert state.champion is not None


def test_a_round_that_writes_nothing_feeds_the_next_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    log: wandb.Run,
    records: list[tuple[float, dict]],
) -> None:
    """A rejected round spends its own budget and nothing more.

    Every round here is refused, so nothing reaches the database; the session
    still runs the rounds it was given, because the next one starts from the
    same program with this one's reason in front of it.
    """
    paths = tiny_run(tmp_path, monkeypatch, rounds=3)
    pass_pool(tmp_path, paths)
    stub_evaluator(monkeypatch)
    monkeypatch.setattr(mutate.CodexMutator, "COMMAND", ["true"])

    loop.run(
        sessions=1,
        mutator=mutate.CodexMutator(model="a", fallback=""),
        workers=WORKERS,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        log=log,
        paths=paths,
        opponents=UNVENDORED,
    )

    assert [record["sessions/rounds"] for record in sessions_of(records)] == [3]
    assert paths.archive.read_text().count('"no_output') == 3
    database = archive.Database(paths.archive, paths.programs)
    assert [program.id for program in database.programs] == [config.SEED_ID]


def test_a_rejected_round_is_the_next_rounds_feedback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """A program that did not parse is exactly what a next attempt can fix.

    The first round writes something that will not compile. Validation is the
    real one, so the reason in the second round's message is the one the
    campaign recorded, and the file that round is handed is the unchanged
    program the first round started from -- four remaining rounds are not
    thrown away over a fixable mistake.
    """
    paths = tiny_run(tmp_path, monkeypatch, rounds=2)
    pass_pool(tmp_path, paths)
    stub_evaluator(monkeypatch)
    seed = _write(tmp_path / "seed.py", PASS)
    written = iter(["def agent(observation, configuration=None)\n", SELLER])
    mutator = Recorder(edit=lambda _: next(written))

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log, paths, UNVENDORED)

    first, second = mutator.seen
    assert "produced nothing" not in first.message
    assert "- syntax: " in second.message
    assert second.child == first.child == PASS
    database = archive.Database(paths.archive, paths.programs)
    # The seed came from nothing; the round's program came from champion zero,
    # which is what a session starts from and is named by its pool key.
    assert [p.started_from for p in database.programs] == ["", config.POOL_CHAMPION]


def test_a_round_is_given_one_file_and_the_directory_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """The program, the program before the edit, and a way to compare them.

    The engine copy, the standing rules and the feedback file were all things
    a session read off disk; nothing reads them now, so nothing is written.
    What is written is what a round needs to measure its own edit rather than
    ship it blind, and nothing else -- no path to an opponent, no corpus, no
    channel a name could leak through. The directory goes once its program is
    in the database.
    """
    paths = tiny_run(tmp_path, monkeypatch)
    pass_pool(tmp_path, paths)
    stub_evaluator(monkeypatch)
    seed = _write(tmp_path / "seed.py", PASS)
    mutator = Recorder(edit=lambda _: SELLER)

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log, paths, UNVENDORED)

    handed = mutator.seen[0]
    assert handed.held == sorted(
        [Recorder.SKILLS_DIRS[0].parts[0], "child.py", "measure.py", "parent.py"]
    )
    assert not handed.where.exists()


@pytest.mark.slow
def test_the_first_round_is_sent_the_loops_own_verdict_and_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """The opening round is composed from a measurement the loop made itself.

    A seam test: the game the round is shown comes out of real games the loop
    played against the pool before the first round, which is what makes the
    first round no different from the fourth. Nothing in the message could have
    come from anywhere else -- the model has played nothing at this point.

    What this asserted until 2026-09-13 was a verdict line, a standings table,
    a per-opponent day-by-day section and its rows -- all of which the message
    stopped carrying. A table of every matchup says the program is losing and
    nothing about a decision it made, and the day tables are rows rather than
    message text: every game the campaign has played is in the games database,
    so the message names one and carries the query. The seam is the same, and
    the episode key is where it now shows.
    """
    paths = tiny_run(tmp_path, monkeypatch)
    pass_pool(tmp_path, paths)
    seed = _write(tmp_path / "seed.py", PASS)
    mutator = Recorder(edit=lambda _: SELLER)

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log, paths, UNVENDORED)

    message = mutator.seen[0].message
    # One game, named by the program that played it -- champion zero, which is
    # the seed enthroned at startup and goes by the pool key.
    assert "## The game" in message
    assert f"Episode `{config.POOL_CHAMPION}m" in message
    # Its result, which is what a round can act on: PASS against PASS is a dead
    # heat, so the banks it finished on are equal and the difference is zero.
    assert "finished" in message and "+0" in message
    # And where to read the rest of it, rather than the rest of it.
    assert prompt.GAMES in message
    assert "| 29 |" not in message and "| rank | agent | rating |" not in message


def test_calls_that_never_reach_a_verdict_stop_the_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """A spin that looks healthy is the failure worth stopping the machine for.

    A call that never reached the model is nobody's failure: it leaves no
    ledger line, and the worker abandons the session and starts another
    immediately. When the cause is the login, a withdrawn model or a provider
    outage, every call ends the same way and the loop runs at full rate for
    as long as nobody looks -- sessions and calls both climbing, the database
    untouched. This campaign's first launch did exactly that, sixteen
    sessions in forty-seven seconds, over a missing flag.
    """
    paths = tiny_run(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "SESSIONS", 1)
    monkeypatch.setattr(config, "NO_VERDICT_LIMIT", 3)
    pass_pool(tmp_path, paths)
    stub_evaluator(monkeypatch)
    monkeypatch.setattr(
        mutate.CodexMutator, "COMMAND", ["bash", "-c", "echo refused 1>&2; exit 1"]
    )

    with pytest.raises(SystemExit, match="3 calls in a row ran to no verdict"):
        loop.run(
            sessions=100,
            mutator=mutate.CodexMutator(model="a"),
            workers=WORKERS,
            seed_agent=_write(tmp_path / "seed.py", PASS),
            rng=random.Random(0),
            log=log,
            paths=paths,
            opponents=UNVENDORED,
        )

    # The seed and nothing else: three sessions ran and the database is as
    # empty as it was before them, which is the state the limit exists to
    # notice.
    database = archive.Database(paths.archive, paths.programs)
    assert [program.id for program in database.programs] == ["seed"]


def test_the_no_verdict_count_is_consecutive_calls_not_a_total(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """One refusal is noise. The limit is for the machine, not for a bad call.

    A campaign of thousands of calls will have some of them refused -- astra
    answered "Selected model is at capacity" twice in the first live hour --
    so a running total would eventually stop a run that is working. Four
    sessions here, refused on the first and the third, and the limit is two:
    a count that did not reset on the call in between would end this
    campaign, and it does not.
    """
    paths = tiny_run(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "SESSIONS", 1)
    monkeypatch.setattr(config, "NO_VERDICT_LIMIT", 2)
    pass_pool(tmp_path, paths)
    stub_evaluator(monkeypatch)
    # One byte per call, and the odd-numbered ones fail: a marker file, not a
    # closure, because the failure has to happen in the child process.
    calls = tmp_path / "calls"
    monkeypatch.setattr(
        mutate.CodexMutator,
        "COMMAND",
        [
            "bash",
            "-c",
            f"printf x >> {calls}; "
            f"if [ $(($(wc -c < {calls}) % 2)) -eq 1 ]; then exit 1; fi; "
            'printf "\n# edited\n" >> child.py',
        ],
    )

    state = loop.run(
        sessions=4,
        mutator=mutate.CodexMutator(model="a", fallback=""),
        workers=WORKERS,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        log=log,
        paths=paths,
        opponents=UNVENDORED,
    )

    assert state.sessions == 4
    assert calls.read_bytes() == b"xxxx"
    database = archive.Database(paths.archive, paths.programs)
    assert [program.id for program in database.programs][0] == "seed"
    assert len(database.programs) == 3


def test_a_result_that_did_not_play_every_opponent_still_reaches_the_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    log: wandb.Run,
    records: list[tuple[float, dict]],
) -> None:
    """A candidate plays a draw, so it has never played everyone.

    The gate used to demand every pool opponent. That was right while the pool
    *was* the tournament and a candidate played all eight of it, and it became
    unsatisfiable the moment the pool grew past the draw: sixteen opponents out
    of sixty-one leaves forty-five missing every time. Live, it blocked 232
    candidates and every promotion for seven hours -- silently, because the
    only line that records a gate sat below the return.

    What still has to hold is the floor, which `must_play` draws every time and
    the gate checks by name.
    """
    paths = tiny_run(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "SESSIONS", 1)
    opponents = pool.Pool(
        opponents={
            "pass": str(_write(tmp_path / "pass.py", PASS)),
            "joiner": str(_write(tmp_path / "joiner.py", PASS)),
        }
    )
    opponents.save(paths.pool)
    monkeypatch.setattr(evaluator, "VENDORED", ["pass", "joiner"])
    scored = stub_evaluator(monkeypatch)
    measure = evaluator.score

    def stale(
        agent: Path,
        program_id: str,
        opponents: pool.Pool,
        rng: random.Random,
        seeds: Sequence[int],
        workers: int,
        pool_file: Path | None = None,
        standings: dict[str, float] | None = None,
        always: Sequence[str] = (),
        duel_seeds: Sequence[int] = (),
    ) -> evaluator.Result:
        """Every measurement lands as if ``joiner`` had joined during it."""
        result = measure(agent, program_id, opponents, rng, seeds, workers, pool_file)
        result.rates.pop("joiner", None)
        return result

    monkeypatch.setattr(evaluator, "score", stale)

    loop.run(
        sessions=1,
        mutator=mutate.FakeMutator(edit=lambda source: source + "\n# edited\n"),
        workers=WORKERS,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        log=log,
        paths=paths,
        opponents=UNVENDORED,
    )

    # The verdict is what matters, not which way it went: an opponent this
    # program never drew must not stop it being judged at all. A gate record
    # exists for every program the gate ruled on, and for seven hours live
    # there were none of them -- which is the shape of the failure this
    # catches, and the reason nobody saw it.
    gates = gates_of(records)
    assert gates, "the gate returned before it judged anything"
    # And judged it, rather than turning it away for a draw it never made.
    # `gate/stale` marks the second, which is what ran 232 times live.
    assert "gate/stale" not in gates[0]
    assert "gate/score" in gates[0] and "gate/promoted" in gates[0]
    assert len(scored) > 1


def test_a_dry_run_writes_only_under_the_root_it_was_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dry run is the whole campaign with the call faked, so it must be moved.

    It evaluates, inserts, gates and promotes for real. Pointed at the
    campaign's own directory it does all of that to the live database, the
    live pool and the live floor -- a running campaign silently corrupted by
    someone checking that the plumbing works.

    It used to get moved by reassigning this module's constants, and this test
    checked those constants afterwards. There are none: `main` builds a
    `config.Run` from the root it is given and every write goes through it, so
    what there is to check is that the root it was given is the root it used.
    Through ``main``, because the entry point is what chooses.
    """
    paths = tiny_run(tmp_path, monkeypatch)
    _repository(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "ROOT", tmp_path)
    stub_evaluator(monkeypatch)
    root = tmp_path / "run"
    seed = _write(tmp_path / "seed.py", PASS)
    # A pool of its own, because a dry run that finds none falls back to the
    # whole vendored roster -- and the start plays the pool's own pairings, so
    # this test would spend 66 of them on real opponents to assert something
    # about directories. It was 571 seconds of a 894-second suite.
    dry_pool = root / "dry-run" / "pool.json"
    dry_pool.parent.mkdir(parents=True, exist_ok=True)
    pool.Pool(opponents={"pass": str(_write(tmp_path / "dry.py", PASS))}).save(dry_pool)

    loop.main(
        [
            "--sessions",
            "1",
            "--workers",
            "1",
            "--dry-run",
            "--seed-agent",
            str(seed),
            "--run-root",
            str(root),
        ]
    )

    # It ran, and everything it wrote is under the directory a dry run owns.
    dry = config.Run(root=root / "dry-run", pool=root / "dry-run" / "pool.json")
    assert dry.archive.exists()
    assert loop.state_file(dry).exists()
    assert dry.pool.exists()
    # And nowhere near the live campaign, which it cannot now reach: it was
    # never handed those paths.
    assert not dry.root.is_relative_to(config.LIVE.root)
    assert not paths.archive.exists()


def test_stagnation_says_so_once_a_champion_has_stood_too_long(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """A campaign that has never promoted has no champion's line to be stuck in.

    Sessions without a promotion are counted from the first one, so a fresh
    campaign passes the stagnation threshold before it has anything to
    stagnate from -- and it starts from the database's top ten either way,
    because there is no champion to start from instead. Telling the model it
    left the champion's line because that line is stuck would be a plain
    falsehood in the one message it reads.
    """
    paths = tiny_run(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "STAGNATION_SESSIONS", 1)
    pass_pool(tmp_path, paths)
    stub_evaluator(monkeypatch)
    seed = _write(tmp_path / "seed.py", PASS)
    mutator = Recorder(edit=lambda source: source + "\n# edited\n")

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log, paths, UNVENDORED)
    state = loop.run(
        1, mutator, WORKERS, seed, random.Random(0), log, paths, UNVENDORED
    )

    # There is always a champion now, so the note has something true to say from
    # the first session: the line it names is champion zero's. It used to be
    # withheld here, because telling a model its lineage was stuck when it had
    # never had one was a sentence about nothing.
    assert state.champion is not None
    assert state.sessions_since_promotion >= config.STAGNATION_SESSIONS
    assert len(mutator.seen) == 2
    assert any("no promotion" in seen.message for seen in mutator.seen)


def test_a_round_logs_the_win_rate_and_the_place_it_bought(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    log: wandb.Run,
    records: list[tuple[float, dict]],
) -> None:
    """Both halves of the story, because they answer different questions.

    The rating is what the gate promotes on and what the competition ranks by,
    so it is the one that says whether the campaign is winning. The win rate
    over the pool is what a rating is made of, and it stays plotted because a
    rating moves for two reasons -- the program got better, or the pool got
    harder -- and only the pair tells you which.
    """
    paths = tiny_run(tmp_path, monkeypatch)
    pass_pool(tmp_path, paths)
    stub_evaluator(monkeypatch)

    loop.run(
        sessions=1,
        mutator=mutate.FakeMutator(edit=lambda source: f"{source}\n# edited\n"),
        workers=WORKERS,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        log=log,
        paths=paths,
        opponents=UNVENDORED,
    )

    call = calls_of(records)[-1]
    assert 0.0 <= call["calls/fitness"] <= 1.0
    # One public opponent, plus champion zero.
    assert call["calls/pool"] == 2
    assert isinstance(call["calls/rating"], float)
    assert call["calls/place"] >= 1
    # The best rating so far, so the curve has a ratchet on it and not just
    # whatever the last round happened to score.
    assert "database/top_rating" in call


def test_a_program_carries_the_rating_it_was_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """Stored, not recomputed: the pool moves, and a rating is of its moment.

    Refitting an old program against today's pool would answer a different
    question from the one its evaluation asked, and `database/top_rating` is
    meant to be the best any round actually achieved.
    """
    paths = tiny_run(tmp_path, monkeypatch)
    pass_pool(tmp_path, paths)
    stub_evaluator(monkeypatch)

    loop.run(
        sessions=1,
        mutator=mutate.FakeMutator(edit=lambda source: f"{source}\n# edited\n"),
        workers=WORKERS,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        log=log,
        paths=paths,
        opponents=UNVENDORED,
    )

    database = archive.Database(paths.archive, paths.programs)
    child = next(p for p in database.programs if p.id != "seed")
    assert child.rating is not None
    assert child.place >= 1
    # The seed is scored before any pool is loaded, so it honestly has none.
    assert database.get("seed").rating is None
    assert database.get("seed").place == 0


def test_a_session_starts_from_the_best_far_more_often_than_the_tenth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """A uniform draw over ten is barely selection at all.

    With no champion there is nothing else deciding where a session begins, so
    a shuffle meant nine sessions in ten started from something worse than the
    best program the campaign had -- while one child in ten improves on its
    parent and one in five is worse. Over 259 rated programs the best rating
    peaked at the fiftieth and every cohort after was worse than it.
    """
    paths = tiny_run(tmp_path, monkeypatch)
    pass_pool(tmp_path, paths)
    stub_evaluator(monkeypatch)
    database = archive.Database(paths.archive, paths.programs)
    for rank in range(config.PARENT_POOL):
        source = database.store(f"{PASS}# rank {rank}\n", f"p{rank}")
        database.add(
            archive.Program(
                id=f"p{rank}",
                source_path=str(source),
                started_from="parent",
                instruction="tune",
                model="m",
                fitness=0.5,
                field=0.5,
                # Rank 0 is the best; the draw should reflect that ordering.
                rating=-float(rank),
                created=float(rank),
            )
        )
    campaign = loop.Campaign(
        loop.State(),
        database,
        pass_pool(tmp_path, paths),
        mutate.FakeMutator(edit=lambda source: source),
        WORKERS,
        random.Random(11),
        log,
        paths,
    )

    drawn = [campaign.start(stagnant=False)[1] for _ in range(400)]

    best = drawn.count("p0")
    worst = drawn.count(f"p{config.PARENT_POOL - 1}")
    assert best > len(drawn) * 0.35, f"the best was drawn only {best} times"
    assert best > 10 * max(1, worst), "the best must dominate the tail"
    # Still a search, not a hill climb: something other than the best is
    # taken often enough that one program cannot own every session.
    assert len(set(drawn)) >= 3


def test_a_scratch_session_is_parented_from_the_scratch_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """The niche is what makes a blank start more than a lottery ticket.

    A program that begins from nothing rates far below a champion, and
    `Database.top` ranks on rating -- so parented from the database's best, a
    scratch session would start from the champion's lineage every time after
    the first and the blank start would never compound. Parented from its own
    best it gets a ratchet of its own.
    """
    paths = tiny_run(tmp_path, monkeypatch)
    pass_pool(tmp_path, paths)
    stub_evaluator(monkeypatch)
    monkeypatch.setattr(config, "SCRATCH_CHANCE", 1.0)
    database = archive.Database(paths.archive, paths.programs)
    # The champion's lineage, rated far above anything a blank start reaches.
    for name, parent, standing in (
        ("champ", "seed", 5.0),
        ("sprout", config.SCRATCH_ID, -4.0),
        ("sapling", "sprout", -3.0),
    ):
        source = database.store(f"{PASS}# {name}\n", name)
        database.add(
            archive.Program(
                id=name,
                source_path=str(source),
                started_from=parent,
                instruction="tune",
                model="m",
                fitness=0.5,
                field=0.5,
                rating=standing,
                created=1.0,
            )
        )
    campaign = loop.Campaign(
        loop.State(),
        database,
        pass_pool(tmp_path, paths),
        mutate.FakeMutator(edit=lambda source: source),
        WORKERS,
        random.Random(3),
        log,
        paths,
    )

    _, name = campaign.start(stagnant=False)

    # The best of the scratch lineage, two generations down -- never `champ`,
    # which out-rates every one of them by nine log-odds.
    assert name == "sapling"


def test_the_first_scratch_session_begins_from_the_blank_slate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """With no scratch lineage yet, there is nothing to parent from.

    The blank is a policy that passes every turn rather than an empty file:
    nothing downstream can score an empty file, and `SEED` is the ancestry
    being escaped, so it cannot be the fallback either.
    """
    paths = tiny_run(tmp_path, monkeypatch)
    pass_pool(tmp_path, paths)
    stub_evaluator(monkeypatch)
    monkeypatch.setattr(config, "SCRATCH_CHANCE", 1.0)
    campaign = loop.Campaign(
        loop.State(),
        archive.Database(paths.archive, paths.programs),
        pass_pool(tmp_path, paths),
        mutate.FakeMutator(edit=lambda source: source),
        WORKERS,
        random.Random(5),
        log,
        paths,
    )

    source, name = campaign.start(stagnant=False)

    assert name == config.SCRATCH_ID
    assert source.read_text(encoding="utf-8") == config.SCRATCH_AGENT
    assert "PASS" in config.SCRATCH_AGENT


def _record_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> loop.Campaign:
    """A campaign built only far enough to compose a promotion record.

    `promotion_record` reads its arguments and the state counter and nothing
    else, so nothing here plays a game.
    """
    paths = tiny_run(tmp_path, monkeypatch)
    return loop.Campaign(
        loop.State(),
        archive.Database(paths.archive, paths.programs),
        pass_pool(tmp_path, paths),
        mutate.FakeMutator(edit=lambda source: source),
        WORKERS,
        random.Random(11),
        log,
        paths,
    )


def _champion(name: str, fitness: float) -> gate.Champion:
    """The floor as ``champion.json`` records it, for a record that needs one."""
    return gate.Champion(
        name=name,
        path=f"/nowhere/{name}.py",
        tarball=f"/nowhere/{name}.tar.gz",
        result=_gate_result(name, {"v54": fitness}),
    )


def test_a_champion_measured_on_other_seasons_is_not_a_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """The gate needs the champion measured on the block the candidate played.

    `champion.json` carries the result the champion was promoted on, and the
    seasons move: `GATE_SEED_ROTATION` draws a fresh block every sixty-four
    candidates and every launch draws one. Measured 2026-09-14, champion_12
    scored 0.454 over fifty shared opponents on the block it was promoted on
    and 0.205 over the same fifty on the block drawn eight hours later -- a
    drift of 0.249 against a promotion bar of 0.035. Twenty-two candidates were
    turned away that morning for failing to reach a number the champion itself
    could no longer reach; they had scored 0.234 to 0.366 and every one of them
    had beaten it.

    So a baseline from another block is refused rather than used. A session
    measures the program it starts from, so the champion is re-measured on the
    current block eight times a generation and the refusal is a short window
    after a rotation, not a wall.
    """
    campaign = _record_campaign(tmp_path, monkeypatch, log)
    standing = _champion("ours", 0.5)
    played = _gate_result("cand", {"v54": 0.9})

    # Nothing measured yet: there is a champion and no way to compare with it.
    assert campaign.paired(standing, played) is None

    # Measured, but on seasons this candidate never played.
    elsewhere = _gate_result("ours", {"v54": 0.5})
    campaign.champion_baseline = (standing.path, (99,), elsewhere)
    assert campaign.paired(standing, played) is None

    # Measured on the same seasons: the comparison is that measurement, and
    # not the one frozen at promotion time.
    here = _gate_result("ours", {"v54": 0.2})
    campaign.champion_baseline = (standing.path, tuple(played.seeds), here)
    paired = campaign.paired(standing, played)
    assert paired is not None
    assert paired.result.rates == {"v54": 0.2}, "the gate used the stale result"


def test_the_gate_asks_for_the_floor_and_not_the_whole_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """A candidate plays a draw, so it can never have played everyone.

    The gate used to demand every pool opponent, which was right while the
    pool *was* the tournament and a candidate played all eight of it. Under a
    sampled draw it is unsatisfiable by construction -- sixteen opponents out
    of sixty-one leaves forty-five missing every time -- and it blocked every
    promotion for seven hours without one line of evidence, because the only
    thing that logs a gate sits below the return.

    What still has to hold is the floor: the bar is a rating gap against that
    one named agent, and a gap against an agent this program never played is
    not a measurement.
    """
    campaign = _record_campaign(tmp_path, monkeypatch, log)
    campaign.state.champion = _champion("champion_2", 0.5)

    # Played the floor and a handful of others; nowhere near the whole pool.
    assert campaign.floor() == "champion_2"
    played = {"champion_2": 0.8, "v54": 0.9}
    assert campaign.floor() in played

    # And a result from before this floor existed has no gap to measure.
    stale = {"champion_1": 0.8, "v54": 0.9}
    assert campaign.floor() not in stale


def test_the_draw_always_holds_the_leader_and_the_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """Two questions that are usually one agent, and must not be left to dice.

    The champion has out-rated the field since the ratchet started, so the top
    of the standings and the floor are the same name -- but they are different
    questions, and when they come apart both have to be played: the leader
    because topping the field means beating it, the floor because the bar is a
    gap over that specific agent.
    """
    campaign = _record_campaign(tmp_path, monkeypatch, log)
    campaign.state.champion = _champion("champion_2", 0.5)

    same = campaign.must_play({"champion_2": 2.0, "v54": 1.0})
    apart = campaign.must_play({"v54": 2.0, "champion_2": 1.0})

    assert same == ["champion_2"]
    assert apart == ["v54", "champion_2"]


def test_only_a_promotion_writes_the_champion_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """`gate/score` carries every program judged; this carries the floor.

    A few dozen champions among many hundreds of candidates is not a series
    anyone can read off the same key, and the question the campaign is
    actually asking -- is the floor still rising -- is about the champions.
    """
    campaign = _record_campaign(tmp_path, monkeypatch, log)
    result = _gate_result("p1", {"champion_2": 0.7, "v54": 0.9})
    standings = {"p1": 1.0, "champion_2": 0.0, "v54": -1.0}
    floor = _champion("champion_2", 0.5)

    refused = campaign.promotion_record(result, False, floor, standings)
    promoted = campaign.promotion_record(result, True, floor, standings)

    assert "champion/win_rate" not in refused
    assert promoted["champion/win_rate"] == result.fitness


def test_the_champion_series_says_what_it_did_to_the_one_it_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """The win rate is against a pool that strengthens with every promotion.

    So it does not compare down the campaign: champion_30 scoring 0.62
    against seven champions is a different feat from champion_3 scoring 0.62
    against seven published kernels. The head-to-head against the floor it
    replaced always means the same thing, and a run of them near 0.5 is a
    search that has stopped finding anything.
    """
    campaign = _record_campaign(tmp_path, monkeypatch, log)
    result = _gate_result("p1", {"champion_2": 0.72, "v54": 0.9})
    standings = {"p1": 1.0, "champion_2": 0.0, "v54": -1.0}

    record = campaign.promotion_record(
        result, True, _champion("champion_2", 0.5), standings
    )

    assert record["champion/over_previous"] == 0.72


def test_the_first_champion_has_nothing_to_be_compared_against(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """No floor yet, so the head-to-head is absent rather than invented.

    A zero here would put a point on the chart saying the first champion
    never beat anything.
    """
    campaign = _record_campaign(tmp_path, monkeypatch, log)
    result = _gate_result("p1", {"v54": 0.9})

    record = campaign.promotion_record(result, True, None, {"p1": 1.0, "v54": -1.0})

    assert record["champion/win_rate"] == result.fitness
    assert "champion/over_previous" not in record


def test_no_metric_is_frozen_at_a_score_nothing_can_earn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """`database/top_field` was the maximum of a score nothing could earn.

    `field` averages the published opponents *in the pool*, and champions
    trim them out one at a time until none is left; after that every program
    scores None and drops out of the maximum. The series sat at 0.901 for 243
    programs -- a value held by the seed itself, so the chart read "nothing
    has ever beaten the starting program" when it meant "nothing since the
    hundredth has been measured at all".
    """
    campaign = _record_campaign(tmp_path, monkeypatch, log)
    result = _gate_result("p1", {"v54": 0.9})

    record = campaign.promotion_record(result, True, None, {"p1": 1.0, "v54": -1.0})

    assert not any("top_field" in key for key in record)


def test_a_cancelled_round_does_not_leave_its_workspace_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """A round the pacer cancels used to leak the directory it was working in.

    Cleanup sat below the two awaits, so it ran only when a round reached a
    verdict; a cancellation unwinds straight past it. The pacer cancels
    routinely, and 548 workspaces had collected in /tmp before anyone looked.
    """
    campaign = _record_campaign(tmp_path, monkeypatch, log)

    class Cancelled:
        """A call killed mid-flight, which is what the pacer does to a slow one."""

        SKILLS_DIRS: tuple[Path, ...] = (Path(".agents") / "skills",)
        TRANSCRIPTS: tuple[str, ...] = ("agy*.jsonl",)

        async def __call__(
            self, workspace: Path, message: str, program_id: str
        ) -> mutate.Mutation:
            """Check the directory is there to be leaked, then die."""
            assert workspace.exists(), "the round never made a workspace to leak"
            raise asyncio.CancelledError

    campaign.mutator = Cancelled()
    # Workspaces come from `tempfile.mkdtemp`, so pointing the module at
    # `tmp_path` keeps this test's leak out of the machine's own /tmp and
    # clear of any already sitting there.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            campaign.round(
                _write(tmp_path / "parent.py", PASS),
                "champion_1",
                _gate_result("p1", {"v54": 0.5}),
                "improve it",
                [],
                "margin",
            )
        )

    assert not list(tmp_path.glob("campaign-round-*"))


def test_the_campaign_harvests_while_it_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """New opponents reach the pool without anyone stopping the campaign.

    The harvest is the other half of the ratchet: promotions add champions and
    nothing else adds anything, so a pool left alone becomes the campaign
    playing itself. It ran twice by hand and then nothing ran it, which is how
    the pool reached 69 champions against 12 published agents frozen five days
    back.
    """
    campaign = _record_campaign(tmp_path, monkeypatch, log)
    monkeypatch.setattr(config, "HARVEST_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(
        loop.harvest, "vendored", lambda limit, known: {"fresh": "/vendored/main.py"}
    )

    asyncio.run(
        _a_pass(campaign.harvesting(), lambda: "fresh" in campaign.pool.opponents)
    )

    assert campaign.pool.opponents["fresh"] == "/vendored/main.py"
    # And it is on disk, because the pool file is what resolves an opponent's
    # path when the harness goes to play it.
    assert "fresh" in pool.Pool.load(campaign.paths.pool).opponents


def test_a_harvest_that_fails_does_not_end_the_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """The competition's API is somebody else's uptime.

    A run that has been evaluating for hours must not end because a listing
    timed out, so the harvester logs and waits for the next turn.
    """
    campaign = _record_campaign(tmp_path, monkeypatch, log)
    monkeypatch.setattr(config, "HARVEST_INTERVAL_SECONDS", 0)

    asked = []

    def refuses(limit: int, known: set) -> dict:
        """A listing that fails, the way a rate-limited one does."""
        asked.append(1)
        raise RuntimeError("kaggle said no")

    monkeypatch.setattr(loop.harvest, "vendored", refuses)
    before = dict(campaign.pool.opponents)

    assert asyncio.run(_a_pass(campaign.harvesting(), lambda: len(asked) > 1))

    assert campaign.pool.opponents == before


def test_the_campaign_reloads_the_games_it_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """The losses a round studies belong to the champion that is standing.

    A round diagnoses the game it is handed, and against the gate's pool this
    lineage wins; the games it loses are on the ladder. Champions 31 to 39 all
    arrived inside a day, so refreshed by hand these are the *previous*
    champion's failures within the hour -- the same drift as a pool nobody
    harvests, one level out.
    """
    campaign = _record_campaign(tmp_path, monkeypatch, log)
    monkeypatch.setattr(config, "LOSSES_INTERVAL_SECONDS", 0)
    passes = []

    def loads() -> int:
        """A refresh that finds one game the lineage had not held yet."""
        passes.append(1)
        return 1

    monkeypatch.setattr(loop.losses, "refresh", loads)
    monkeypatch.setattr(loop.losses, "deficit", dict)

    asyncio.run(_a_pass(campaign.losing(), lambda: passes))

    assert passes


def test_a_loss_refresh_that_fails_does_not_end_the_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """Somebody else's uptime again, and the same answer.

    Listing episodes and pulling 32 MB replays is more of the competition's
    API than the harvest touches, so this is the more likely of the two to
    fail -- and a run that has been evaluating for hours must not end with it.
    """
    campaign = _record_campaign(tmp_path, monkeypatch, log)
    monkeypatch.setattr(config, "LOSSES_INTERVAL_SECONDS", 0)

    asked = []

    def refuses() -> int:
        """A listing that fails, the way a rate-limited one does."""
        asked.append(1)
        raise RuntimeError("kaggle said no")

    monkeypatch.setattr(loop.losses, "refresh", refuses)
    monkeypatch.setattr(loop.losses, "deficit", dict)

    assert asyncio.run(_a_pass(campaign.losing(), lambda: len(asked) > 1))


def test_where_the_losses_are_decided_reaches_the_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    log: wandb.Run,
    records: list[tuple[float, dict]],
) -> None:
    """The figures are logged, which is the only place they are read.

    They are deliberately not gated on: the day-29 gap *is* the final margin the
    gate already scores, and a day-indexed figure before it is a correlate --
    day-10 bank trended across a selected chain of champions and was not the
    mechanism. So the value of measuring them is entirely that a person can read
    them against promotions, and a record that never reaches the run is the same
    as not measuring them at all.
    """
    campaign = _record_campaign(tmp_path, monkeypatch, log)
    monkeypatch.setattr(config, "LOSSES_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(loop.losses, "refresh", lambda: 0)
    where = {
        "losses/games": 49.0,
        "losses/gap_day10": 87.0,
        "losses/gap_days20_29": -533.0,
        "losses/gap_final": -804.0,
    }
    monkeypatch.setattr(loop.losses, "deficit", lambda: where)

    asyncio.run(
        _a_pass(
            campaign.losing(),
            lambda: any("losses/games" in record for _, record in records),
        )
    )

    logged = [record for _, record in records if "losses/games" in record]
    assert logged, "the gap has to reach wandb to be worth measuring"
    assert logged[0] == where
    # Logged on a pass that loaded nothing, because the figures describe the
    # corpus and not the delta: an hour with no new losses is still an hour a
    # promotion has to be read against.
    assert logged[0]["losses/gap_day10"] > 0


def test_a_stored_program_records_what_its_edit_did(tmp_path: Path) -> None:
    """The record says what changed, not only what it scored.

    Every promotion's change was rendered for the prompt by reading the
    champion files back and diffing them; the four hundred and fifty edits that
    were measured and refused were described nowhere. The moment a program is
    stored is the only one where both plans are already files, so that is where
    the sentence is written.
    """
    from tests.campaign.test_plan import PLAN, packed

    parent = tmp_path / "parent.py"
    parent.write_text(packed(PLAN), encoding="utf-8")
    # Through JSON, which is the trip a plan makes anyway and leaves the
    # literal's mixed value types behind.
    flipped = json.loads(json.dumps(PLAN))
    flipped["settings"]["front_run"] = True
    child = tmp_path / "child.py"
    child.write_text(packed(flipped), encoding="utf-8")

    said = loop._changed(child, parent)

    assert said and said != "the plan is unchanged"
    assert "front_run" in said


def test_a_program_whose_parent_carries_no_plan_records_nothing(
    tmp_path: Path,
) -> None:
    """Empty rather than a crash: a scored program must never be lost to a sentence.

    The seed has no parent, programs before champion_17 carry no packed plan,
    and a round can rewrite the controller into something `split` refuses. None
    of those is a failed evaluation.
    """
    from tests.campaign.test_plan import PLAN, packed

    bare = tmp_path / "bare.py"
    bare.write_text("def agent(o, c=None):\n    return {}\n", encoding="utf-8")
    child = tmp_path / "child.py"
    child.write_text(packed(PLAN), encoding="utf-8")

    assert loop._changed(child, bare) == ""
    assert loop._changed(child, None) == ""


def test_a_round_is_given_its_parent_and_a_way_to_play(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """A round can measure an edit instead of shipping it blind.

    The spec had the model run nothing -- "the loop plays; the model never
    does" -- so a round wrote a program and waited for a gate whose own
    estimator could not separate its best candidate from its fourth. Every
    improvement found by hand on 2026-09-10 came from measuring instead, and
    two of four ideas were measured as worse and dropped before costing
    anything. The round gets the same instrument: the program it started from,
    and the script that plays one against the other.
    """
    campaign = _record_campaign(tmp_path, monkeypatch, log)
    seen: dict[str, object] = {}

    class Inspect:
        """A call that only reports what it was handed."""

        SKILLS_DIRS: tuple[Path, ...] = (Path(".agents") / "skills",)
        TRANSCRIPTS: tuple[str, ...] = ("agy*.jsonl",)

        async def __call__(
            self, workspace: Path, message: str, program_id: str
        ) -> mutate.Mutation:
            """Note the directory, then leave without writing anything."""
            seen["files"] = sorted(path.name for path in workspace.iterdir())
            seen["parent"] = (workspace / "parent.py").read_text(encoding="utf-8")
            seen["skill"] = (
                workspace / self.SKILLS_DIRS[0] / "query-games" / "SKILL.md"
            ).exists()
            runner = workspace / "measure.py"
            seen["shebang"] = runner.read_text(encoding="utf-8").splitlines()[0]
            seen["runnable"] = os.access(runner, os.X_OK)
            raise asyncio.CancelledError

    campaign.mutator = Inspect()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    source = _write(tmp_path / "parent-source.py", SELLER)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            campaign.round(
                source,
                "champion_1",
                _gate_result("p1", {"v54": 0.5}, days=30),
                "improve it",
                [],
                "margin",
            )
        )

    assert seen["files"] == sorted(
        [Inspect.SKILLS_DIRS[0].parts[0], "child.py", "measure.py", "parent.py"]
    )
    # The skills go where the driving program looks for them: `.codex/skills`
    # under codex's working directory, `.agents` for agy to walk up to.
    assert seen["skill"], "the round was given no query-games skill"
    # Every game behind the verdict, at a width no message could carry. The
    # message holds the index; this is what the index points at.
    # No games file: there is one database and the round queries it.
    assert "seasons.db" not in str(seen["files"])
    # The parent is the program as it was, not the edited copy: a comparison
    # against the thing being edited measures nothing.
    assert seen["parent"] == SELLER
    # And it can be run without knowing anything about where the campaign lives.
    # A round's commands do not see an interpreter that can import the package:
    # `python` is not on their PATH and `python3` is a system one without it, so
    # every round was working that out by trial and one spent its whole call on
    # it. The shebang is the answer and it has to name this interpreter, not a
    # `/usr/bin/env` lookup that would find the wrong one.
    assert seen["shebang"] == f"#!{sys.executable}"
    assert seen["runnable"], "a round cannot run ./measure.py"


def test_candidates_share_a_block_of_seasons_and_it_rotates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """Two things at once, pulling opposite ways.

    Sharing is what makes two candidates comparable. The seeds used to be
    drawn per call, so no two programs were ever ranked on the same seasons --
    and a season is most of what the rating measures: one unchanged agent
    through the gate five times, opponents held fixed and only the seeds
    moving, gave fitted ratings from -3.466 to -2.226, a standard deviation of
    0.491 against a promotion bar that used to be 0.15. Varying the opponents
    instead moved it 0.070.

    Rotating is what stops a block becoming a set the search is selected
    against, which is what the reserved held-out seeds existed to prevent.
    """
    campaign = _record_campaign(tmp_path, monkeypatch, log)
    monkeypatch.setattr(config, "SEED_ROTATION", 3)

    blocks = [campaign.seasons() for _ in range(7)]

    # Three candidates share a block, then a fresh one is drawn.
    assert blocks[0] == blocks[1] == blocks[2]
    assert blocks[3] == blocks[4] == blocks[5]
    assert blocks[0] != blocks[3]
    assert blocks[6] != blocks[3]
    # And a block is a full gate's worth of seasons, every time.
    assert all(len(block) == config.GATE_SEEDS for block in blocks)
    assert all(len(set(block)) == len(block) for block in blocks)


def test_the_duel_block_never_shares_a_season_with_the_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """Both blocks come out of one sample, so neither can repeat the other.

    A season in both blocks is the same game played twice against the
    champion, and both copies land in that pairing's rate and its decisive
    count -- inflating the evidence behind the one condition that decides a
    promotion, which is the opposite of what deepening it is for. Two
    independent draws would collide rarely rather than never; one draw split
    in two cannot collide at all.
    """
    campaign = _record_campaign(tmp_path, monkeypatch, log)
    monkeypatch.setattr(config, "SEED_ROTATION", 1)
    # A narrow range, because over the real one -- a million seeds -- two
    # independent draws of 16 and 48 collide about once in a thousand runs,
    # and a test that only fails then is a test that never fails. From 79
    # seeds two independent draws overlap on about ten every time, while one
    # sample split in two still cannot repeat itself.
    monkeypatch.setattr(config, "GATE_SEED_RANGE", range(1, 80))

    for _ in range(5):
        block = campaign.seasons()

        assert len(block) == config.GATE_SEEDS
        assert len(campaign.duel) == config.DUEL_SEEDS - config.GATE_SEEDS
        assert not set(block) & set(campaign.duel)
        # The pairing is played over both, so together they are the depth the
        # gate's third condition is read at.
        assert len(set(block) | set(campaign.duel)) == config.DUEL_SEEDS


async def _a_pass(
    timer: Coroutine[None, None, None], until: Callable[[], object]
) -> bool:
    """Run one of the loop's timers until ``until`` holds, then stop it.

    Both timers -- the harvest and the loss refresh -- loop forever on an
    interval a test sets to zero, so this drives one and cancels it the way
    `drive` does.

    Waits for what the test is waiting *for*, rather than yielding the event
    loop a fixed number of times. It used to do the latter, 50 turns of
    `sleep(0)`, which is not a wait at all: both timers do their work in
    `to_thread`, so whether the thread had finished by the fiftieth yield was a
    question about how busy the machine was. It held here and failed on a CI
    runner, which is the worst way to learn it -- a test that passes on the
    author's box and fails on somebody else's teaches the wrong lesson twice.

    Args:
        timer: The coroutine to drive.
        until: Checked between turns; the pass is over when it is truthy. The
            deadline exists only so a broken timer fails instead of hanging.

    Returns:
        Whether it was still going when it was stopped, which is what a test
        of somebody else's outage is really asking: a timer that let the
        exception out is finished, and the campaign ended with it.
    """
    task = asyncio.ensure_future(timer)
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        await asyncio.sleep(0.01)
        if task.done() or until():
            break
    alive = not task.done()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    return alive


@pytest.mark.slow
def test_a_rounds_transcript_outlives_the_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """What a round did is kept, because the score cannot say it.

    A child whose plan is unchanged can mean the round never opened
    `plan.json`, or that it edited it, measured the edit worse and backed it
    out -- and the second is the round doing exactly what it was told. The
    workspace is temporary and used to take the only record of which with it.

    Kept under the run's own directory rather than a module constant, so a dry
    run does not write its transcripts into the live campaign's -- which is the
    mistake `config.Run` exists to prevent and which this first repeated.
    """
    paths = tiny_run(tmp_path, monkeypatch)
    pass_pool(tmp_path, paths)
    seed = _write(tmp_path / "seed.py", PASS)
    mutator = Recorder(edit=lambda _: SELLER)

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log, paths, UNVENDORED)

    kept = sorted(paths.rounds.glob("*/agy.jsonl"))
    assert kept, "the round's transcript went with the workspace"
    # Keyed by the program the round wrote, so it joins the archive.
    for transcript in kept:
        assert json.loads(transcript.read_text())["round"] == transcript.parent.name
    assert paths.rounds.is_relative_to(paths.root), "a run owns its own transcripts"


def test_every_path_argument_comes_back_absolute() -> None:
    """The island's first launch died on a relative `--seed-agent`.

    An evaluation worker is given a directory of its own and `chdir`s into it,
    so a relative path resolves at parse time and not at the moment a game
    actually opens the file. Every seat raised `FileNotFoundError` minutes
    later, which reads as a crashed agent rather than as a bad argument.

    `--pool` is read by those same workers. `--run-root` fails more slowly:
    `gate.promote` writes the champion's path into the pool as text, so a
    relative root would seed relative paths into a file every later gate reads.
    """
    parsed = loop._arguments(
        [
            "--seed-agent",
            "run/island/empty_plan_seed.py",
            "--run-root",
            "run/island",
            "--pool",
            "run/island/pool.json",
        ]
    )

    assert parsed.seed_agent.is_absolute(), parsed.seed_agent
    assert parsed.run_root.is_absolute(), parsed.run_root
    assert parsed.pool.is_absolute(), parsed.pool
    assert parsed.seed_agent.name == "empty_plan_seed.py"


def test_the_defaults_are_left_where_they_already_pointed() -> None:
    """Resolving must not move a default that was already absolute."""
    parsed = loop._arguments([])

    assert parsed.run_root == config.RUN
    assert parsed.pool == config.POOL
    assert parsed.seed_agent == config.SEED


def guarded(tmp_path: Path) -> Path:
    """A checkout-shaped tree with one source file and one prompt in it."""
    root = tmp_path / "checkout"
    (root / "src" / "kaggriculture" / "campaign").mkdir(parents=True)
    here = root / "src" / "kaggriculture" / "campaign"
    (here / "harness.py").write_text("REFERENCE_SAMPLE = 0.02\n", encoding="utf-8")
    (here / "roster.py").write_text("    raise KeyError(name)\n", encoding="utf-8")
    (here / "round_prompt.md").write_text("# how to ask\n", encoding="utf-8")
    (here / "engine.so").write_bytes(b"\x00compiled")
    return root


def a_call(program_id: str = "pdeadbeef") -> mutate.Mutation:
    """A successful call, for the guard to turn down."""
    return mutate.Mutation(
        program_id=program_id,
        child=Path("/tmp/child.py"),
        status="ok",
        reason="",
        seconds=1.0,
        input_tokens=1,
        output_tokens=1,
        model="test",
    )


def test_the_two_edits_a_round_actually_made_are_put_back(tmp_path: Path) -> None:
    """The 2026-09-20 tampering, reproduced exactly.

    A round set `REFERENCE_SAMPLE` to 0.0, turning off the reference-engine
    cross-check, and turned `roster.path`'s `raise KeyError` into a constructed
    path so any string resolves to a file. Both weaken a guard in the direction
    that makes the round's own job easier.
    """
    root = guarded(tmp_path)
    here = root / "src" / "kaggriculture" / "campaign"
    before = loop._kept(root)
    (here / "harness.py").write_text("REFERENCE_SAMPLE = 0.0\n", encoding="utf-8")
    (here / "roster.py").write_text(
        "    return Path('/data/' + name)\n", encoding="utf-8"
    )

    verdict = loop._restored(before, a_call(), "pdeadbeef")

    assert (here / "harness.py").read_text(
        encoding="utf-8"
    ) == "REFERENCE_SAMPLE = 0.02\n"
    assert (here / "roster.py").read_text(
        encoding="utf-8"
    ) == "    raise KeyError(name)\n"
    assert verdict.status == "no_output"
    assert verdict.child is None
    assert "harness.py" in verdict.reason and "roster.py" in verdict.reason


def test_the_message_a_round_is_asked_with_is_guarded_too(tmp_path: Path) -> None:
    """A round that rewrites its own instructions has rewritten the objective."""
    root = guarded(tmp_path)
    prompt_file = root / "src" / "kaggriculture" / "campaign" / "round_prompt.md"
    before = loop._kept(root)
    prompt_file.write_text("# anything goes\n", encoding="utf-8")

    verdict = loop._restored(before, a_call(), "pdeadbeef")

    assert prompt_file.read_text(encoding="utf-8") == "# how to ask\n"
    assert verdict.status == "no_output"


def test_a_round_that_changes_nothing_is_left_alone(tmp_path: Path) -> None:
    """The guard must not fail every round, which is how it would be noticed."""
    root = guarded(tmp_path)

    verdict = loop._restored(loop._kept(root), a_call(), "pdeadbeef")

    assert verdict.status == "ok"
    assert verdict.child is not None


def test_a_deleted_source_file_comes_back(tmp_path: Path) -> None:
    """Removing a guard is as effective as editing it."""
    root = guarded(tmp_path)
    gone = root / "src" / "kaggriculture" / "campaign" / "harness.py"
    before = loop._kept(root)
    gone.unlink()

    verdict = loop._restored(before, a_call(), "pdeadbeef")

    assert gone.read_text(encoding="utf-8") == "REFERENCE_SAMPLE = 0.02\n"
    assert verdict.status == "no_output"


def test_the_guard_can_only_write_back_what_it_read(tmp_path: Path) -> None:
    """The bound the git version did not have.

    That one asked git what had changed and reverted the answer, which under a
    pre-commit hook was all 250 tracked files. This holds bytes, so a file it
    never snapshotted -- anything compiled, anything outside `src` -- is one it
    cannot touch however it is called.
    """
    root = guarded(tmp_path)
    before = loop._kept(root)
    binary = root / "src" / "kaggriculture" / "campaign" / "engine.so"
    outside = tmp_path / "not_in_the_snapshot.py"
    outside.write_text("untouched\n", encoding="utf-8")
    binary.write_bytes(b"\x00changed")

    verdict = loop._restored(before, a_call(), "pdeadbeef")

    assert binary.read_bytes() == b"\x00changed"
    assert outside.read_text(encoding="utf-8") == "untouched\n"
    assert verdict.status == "ok"
    assert set(before) == {
        root / "src" / "kaggriculture" / "campaign" / name
        for name in ("harness.py", "roster.py", "round_prompt.md")
    }
