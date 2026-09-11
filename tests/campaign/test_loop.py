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
import os
import random
import signal
import sqlite3
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple

import pytest
import wandb
from git import Actor, Repo

from kaggriculture.campaign import (
    archive,
    config,
    copycheck,
    dataset,
    evaluator,
    gate,
    harness,
    loop,
    mutate,
    pool,
    rating,
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
    monkeypatch.setattr(config, "ROUNDS_PER_SESSION", rounds)
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
    monkeypatch.setattr(copycheck, "CORPUS_ROOTS", (corpus,))
    copycheck._corpus.cache_clear()
    root = tmp_path / "run"
    return config.Run(root=root, pool=root / "pool.json")


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
    same contract ``FakeMutator`` meets, and nothing but the codex process is
    stood in for.
    """

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
def test_the_pool_plays_itself_before_anything_is_judged_against_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """A tournament needs the pool's own pairings, and only a start builds them.

    `standing` fits over what the field holds and plays nothing, so against a
    pool that has never played itself every opponent is rated purely by how
    the one candidate did against it. That is not a tournament, it is a row.

    A promotion used to be the only thing that filled the field in, which
    worked for exactly as long as something else had built the file first.
    """
    paths = tiny_run(tmp_path, monkeypatch)
    opponents = pool.Pool(
        opponents={
            "one": str(_write(tmp_path / "one.py", PASS)),
            "two": str(_write(tmp_path / "two.py", PASS)),
        }
    )
    opponents.save(paths.pool)
    monkeypatch.setattr(evaluator, "VENDORED", ["one", "two"])
    stub_evaluator(monkeypatch)
    assert not paths.field.exists()

    loop.run(
        sessions=1,
        mutator=mutate.FakeMutator(edit=lambda source: source + "\n# edited\n"),
        workers=WORKERS,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        log=log,
        paths=paths,
    )

    field = rating.Field.load(paths.field)
    assert field.results(["one", "two"]), "the pool never played itself"
    assert field.games == 2 * config.GATE_SEEDS


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
    """
    paths = tiny_run(tmp_path, monkeypatch)
    pass_pool(tmp_path, paths)

    state = loop.run(
        sessions=2,
        mutator=mutate.FakeMutator(edit=lambda _: SELLER),
        workers=WORKERS,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        log=log,
        paths=paths,
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
    )

    # `config.ROOT` is the real checkout until here, because packaging a
    # champion reads the plumbing modules out of it. From here it is the
    # repository the promotion could have dirtied.
    assert state.champion is not None
    monkeypatch.setattr(config, "ROOT", tmp_path)
    assert repo.git.status("--porcelain", "--", "src") == ""
    loop._open_run(dry_run=True).finish()


def test_the_seed_is_never_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """The gate judges programs sessions wrote, not the one the campaign began on.

    The seed can be the best program in the database -- on a cold start with a
    child that ranks below it, it is -- and there is no champion for it to
    beat, so anything that put it through the gate would promote it. That
    would make `champion_1` an opponent every candidate already beats: a
    constant added to every score that separates none of them.

    It is structural rather than a rule now. The gate runs at the end of a
    round, on the program that round produced, and no round produces the seed.
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
    )

    # It is measured -- at the cold start, and again when a session begins
    # from it -- and never judged: the gate runs on what a round produced.
    assert config.SEED_ID in scored
    assert state.champion is None
    assert not (paths.floor / "main.py").exists()


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
    """
    paths = tiny_run(tmp_path, monkeypatch)
    pass_pool(tmp_path, paths)
    seed = _write(tmp_path / "seed.py", PASS)
    mutator = Recorder(edit=lambda _: SELLER)

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log, paths)

    assert [handed.child for handed in mutator.seen] == [PASS]
    assert (paths.champions / "champion_1.py").read_text() == SELLER

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log, paths)

    assert mutator.seen[1].child == SELLER
    assert "champion_1" in mutator.seen[1].message


@pytest.mark.slow
def test_a_program_is_gated_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """A program the gate has already confirmed is never sent through it again.

    Four restarts, so every session's gates have finished before the next
    session opens and the top three are the same programs each time: a
    database that forgot which programs carry a deep result would spend the
    exam block on them over and over.
    """
    paths = tiny_run(tmp_path, monkeypatch)
    strong_champion(tmp_path, paths)
    scored = stub_evaluator(monkeypatch)
    seed = _write(tmp_path / "seed.py", PASS)
    for session in range(4):
        mutator = mutate.FakeMutator(edit=lambda s, n=session: f"{s}\n# child {n}\n")
        loop.run(1, mutator, WORKERS, seed, random.Random(session), log, paths)

    assert len(scored) > 2
    assert sorted(scored) == sorted(set(scored))


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

    def watched(
        self: pool.Pool,
        name: str,
        path: str,
        standings: dict[str, float] | None = None,
    ) -> None:
        """The real pool change, with a note of the thread that made it."""
        threads.append(threading.current_thread().name)
        add_champion(self, name, path, standings)

    monkeypatch.setattr(pool.Pool, "add_champion", watched)

    state = loop.run(
        sessions=1,
        mutator=mutate.FakeMutator(edit=lambda _: SELLER),
        workers=WORKERS,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        log=log,
        paths=paths,
    )

    assert state.champion is not None
    assert threads == [threading.main_thread().name]


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
    )

    assert [artifact.name for artifact in artifacts] == ["champion_1"]
    assert list(artifacts[0].manifest.entries) == ["champion_1.tar.gz"]
    assert (paths.champions / "champion_1.tar.gz").exists()


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

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log, paths)
    loop.run(1, mutator, WORKERS, seed, random.Random(0), log, paths)

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
    """
    monkeypatch.setattr(config, "CODEX_MODEL", "gpt-5.6-astra")
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
    """The run id is the model and the revision, so a restart lands on the same run."""
    repo = _repository(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "ROOT", tmp_path)
    expected = f"{config.CODEX_MODEL}-{repo.head.commit.hexsha[:7]}"

    first = loop._open_run(dry_run=True)
    first.finish()
    second = loop._open_run(dry_run=True)
    second.finish()

    assert first.name == expected and first.id == expected
    assert second.id == first.id


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
    )

    assert state.sessions == 7 and state.sessions_since_promotion == 3
    assert state.champion == champion
    assert state.champion != stale


@pytest.mark.slow
def test_a_round_is_told_a_name_and_never_a_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """What a round starts from is interpolated raw, so it is a name, not a path.

    The champion reaches the message as its pool name, which is also how the
    verdict tells the round to beat it head to head. A path there would both
    break that sentence and point at the file it must not read. Everything a
    model is given is this one string, so this is the whole exposure.
    """
    paths = tiny_run(tmp_path, monkeypatch)
    champion = strong_champion(tmp_path, paths)
    stub_evaluator(monkeypatch)
    mutator = Recorder(edit=lambda _: SELLER)
    seed = _write(tmp_path / "seed.py", PASS)

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log, paths)

    handed = mutator.seen[0]
    assert f"`{champion.name}`" in handed.message
    assert str(tmp_path) not in handed.message
    assert "/data/kaggriculture" not in handed.message
    # The doctrine lives in `round_prompt.md` now, so this asserts on what
    # was actually delivered rather than on a constant that could drift.
    assert "Every other opponent is closed" in handed.message


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
    nothing openable reached the call.
    """
    paths = tiny_run(tmp_path, monkeypatch)
    pass_pool(tmp_path, paths)
    stub_evaluator(monkeypatch)
    mutator = Recorder(edit=lambda source: source + "\n# edited\n")
    seed = _write(tmp_path / "seed.py", PASS)

    state = loop.run(1, mutator, WORKERS, seed, random.Random(0), log, paths)

    assert state.champion is None
    handed = mutator.seen[0]
    assert str(tmp_path) not in handed.message
    assert "/data/kaggriculture" not in handed.message
    # The doctrine lives in `round_prompt.md` now, so this asserts on what
    # was actually delivered rather than on a constant that could drift.
    assert "Every other opponent is closed" in handed.message


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
    """
    paths = tiny_run(tmp_path, monkeypatch, rounds=3)
    pass_pool(tmp_path, paths)
    stub_evaluator(monkeypatch)
    seed = _write(tmp_path / "seed.py", PASS)
    mutator = Recorder(edit=lambda source: source + "# a round\n")

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log, paths)

    handed = [given.child for given in mutator.seen]
    assert handed == [PASS, PASS + "# a round\n", PASS + "# a round\n" * 2]
    database = archive.Database(paths.archive, paths.programs)
    written = [p for p in database.programs if p.id != config.SEED_ID]
    assert len(written) == 3
    assert all(program.rates for program in written)
    assert len(calls_of(records)) == 3
    assert [record["sessions/rounds"] for record in sessions_of(records)] == [3]


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

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log, paths)

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

    state = loop.run(1, mutator, WORKERS, seed, random.Random(0), log, paths)

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

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log, paths)

    first, second = mutator.seen
    assert "produced nothing" not in first.message
    assert "- syntax: " in second.message
    assert second.child == first.child == PASS
    database = archive.Database(paths.archive, paths.programs)
    assert [p.started_from for p in database.programs] == ["", config.SEED_ID]


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

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log, paths)

    handed = mutator.seen[0]
    assert handed.held == [
        ".codex",
        "child.py",
        "measure.py",
        "parent.py",
        "seasons.db",
    ]
    assert not handed.where.exists()


@pytest.mark.slow
def test_the_first_round_is_sent_the_loops_own_verdict_and_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """The opening round is composed from a measurement the loop made itself.

    A seam test: the verdict and the thirty rows below it come out of real
    games the loop played against the pool before the first round, which is
    what makes the first round no different from the fourth. Nothing in the
    message could have come from anywhere else -- the model has played
    nothing at this point.
    """
    paths = tiny_run(tmp_path, monkeypatch)
    pass_pool(tmp_path, paths)
    seed = _write(tmp_path / "seed.py", PASS)
    mutator = Recorder(edit=lambda _: SELLER)

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log, paths)

    message = mutator.seen[0].message
    assert f"The verdict on `{config.SEED_ID}`" in message
    # PASS against PASS is a dead heat, so the seed is not top of a
    # tournament it shares with the opponent it drew against.
    assert "It 2 of 2 at " in message and "below pass" in message
    # And the standings themselves, so a round can see what it has to pass.
    assert "| rank | agent | rating |" in message
    # PASS draws with PASS, and a draw is not a win, so the opponent it did
    # not beat is shown day by day for the round to learn from.
    assert "The matches it lost, day by day" in message
    assert "### `pass`, won 0.500" in message
    assert message.count("\n| 2") + message.count("\n| 1") > 0
    assert "| 29 |" in message


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


def test_stagnation_says_nothing_before_there_is_a_champion(
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

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log, paths)
    state = loop.run(1, mutator, WORKERS, seed, random.Random(0), log, paths)

    assert state.champion is None
    assert state.sessions_since_promotion >= config.STAGNATION_SESSIONS
    assert len(mutator.seen) == 2
    assert not any("no promotion" in seen.message for seen in mutator.seen)


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
    )

    call = calls_of(records)[-1]
    assert 0.0 <= call["calls/fitness"] <= 1.0
    assert call["calls/pool"] == 1
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

    async def cancelled(
        workspace: Path, message: str, program_id: str
    ) -> mutate.Mutation:
        """A call killed mid-flight, which is what the pacer does to a slow one."""
        assert workspace.exists(), "the round never made a workspace to leak"
        raise asyncio.CancelledError

    campaign.mutator = cancelled
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

    asyncio.run(_one_harvest(campaign))

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

    def refuses(limit: int, known: set) -> dict:
        """A listing that fails, the way a rate-limited one does."""
        raise RuntimeError("kaggle said no")

    monkeypatch.setattr(loop.harvest, "vendored", refuses)
    before = dict(campaign.pool.opponents)

    asyncio.run(_one_harvest(campaign))

    assert campaign.pool.opponents == before


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

    async def inspect(
        workspace: Path, message: str, program_id: str
    ) -> mutate.Mutation:
        """A call that only reports what it was handed."""
        seen["files"] = sorted(path.name for path in workspace.iterdir())
        seen["parent"] = (workspace / "parent.py").read_text(encoding="utf-8")
        seen["skill"] = (
            workspace / ".codex" / "skills" / "query-games" / "SKILL.md"
        ).exists()
        # Counted here rather than after: the workspace is removed as soon
        # as this call unwinds, and a path is not evidence once it is gone.
        seen["seasons"] = (
            sqlite3.connect(workspace / "seasons.db")
            .execute("select count(*) from days")
            .fetchone()[0]
        )
        raise asyncio.CancelledError

    campaign.mutator = inspect
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    source = _write(tmp_path / "parent-source.py", SELLER)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            campaign.round(
                source,
                "champion_1",
                _gate_result("p1", {"v54": 0.5}, days=30),
                "improve it",
                "margin",
            )
        )

    assert seen["files"] == [
        ".codex",
        "child.py",
        "measure.py",
        "parent.py",
        "seasons.db",
    ]
    # The skills go in where codex looks for them, under its working directory.
    assert seen["skill"], "the round was given no query-games skill"
    # Every game behind the verdict, at a width no message could carry. The
    # message holds the index; this is what the index points at.
    # One opponent, one season, thirty days, both sides of each.
    assert seen["seasons"] == 30 * 2, "the round was handed an empty database"
    # The parent is the program as it was, not the edited copy: a comparison
    # against the thing being edited measures nothing.
    assert seen["parent"] == SELLER


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


async def _one_harvest(campaign: loop.Campaign) -> None:
    """Run the harvester long enough for exactly one pass, then stop it."""
    task = asyncio.ensure_future(campaign.harvesting())
    for _ in range(50):
        await asyncio.sleep(0)
        if task.done():
            break
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
