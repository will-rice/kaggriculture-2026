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

import os
import random
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import pytest
import wandb
from git import Actor, Repo

from kaggriculture.campaign import (
    archive,
    config,
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
# Read once, at import, so `tiny_run` can be applied twice in one test: the
# second call would otherwise resolve paths the first call had replaced.
RUNTIME_PATHS = {
    name: getattr(config, name).relative_to(config.RUN)
    for name in ("ARCHIVE", "PROGRAMS", "FLOOR", "CHAMPIONS", "CHAMPION")
}
EXAM_SEEDS = config.EXAM_SEEDS


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


def deeps_of(records: list[tuple[float, dict]]) -> list[dict]:
    """The per-deep-evaluation records, in the order the loop logged them."""
    return [record for _, record in records if "deep/promoted" in record]


def tiny_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rounds: int = 1) -> None:
    """Point every runtime path at ``tmp_path`` and shrink the loop to one seed.

    The one-opponent test pool stands in for the vendored field and the real
    held-out set is dropped: these tests measure the pipeline, not a field,
    and playing the held-out opponents would cost games nothing asserts on.

    ``rounds`` is one by default, because a test that is not about rounds
    should cost one call and one evaluation; the tests that are about rounds
    ask for more. It is the only constant here that decides behaviour rather
    than cost, which is why it is a parameter and not a line in the body.
    ``DEEP_TOP_K`` used to be set to one, which configured away the only
    values at which the gate has more than one program to choose between.
    """
    monkeypatch.setattr(config, "ROUNDS_PER_SESSION", rounds)
    run = tmp_path / "run"
    for name, relative in RUNTIME_PATHS.items():
        monkeypatch.setattr(config, name, run / relative)
    monkeypatch.setattr(config, "POOL", run / "pool.json")
    monkeypatch.setattr(config, "EXAM_SEEDS", EXAM_SEEDS[:2])
    monkeypatch.setattr(config, "FAST_SEEDS", 1)
    monkeypatch.setattr(evaluator, "VENDORED", ["pass"])
    monkeypatch.setattr(evaluator, "HELD_OUT", [])


def pass_pool(tmp_path: Path) -> pool.Pool:
    """A saved one-opponent pool whose opponent is a PASS agent."""
    opponents = pool.Pool(opponents={"pass": str(_write(tmp_path / "pass.py", PASS))})
    opponents.save(config.POOL)
    return opponents


def strong_champion(tmp_path: Path) -> gate.Champion:
    """A champion on the floor that nothing can beat, so nothing needs docker.

    The promotion rule asks for a lower bound above the champion's score; a
    champion recorded at 1.0 has no bound above it, so every candidate in
    these tests is measured, compared and turned down without the gate ever
    reaching the image.
    """
    config.CHAMPIONS.mkdir(parents=True, exist_ok=True)
    kept = _write(config.CHAMPIONS / "champion_1.py", SELLER)
    opponents = pool.Pool(
        opponents={
            "pass": str(_write(tmp_path / "pass.py", PASS)),
            "champion_1": str(kept),
        }
    )
    opponents.save(config.POOL)
    champion = gate.Champion(
        name="champion_1",
        path=str(kept),
        tarball=str(tmp_path / "champion_1.tar.gz"),
        result=_deep_result("champion_1", {"pass": 1.0}, score=1.0),
    )
    config.CHAMPION.parent.mkdir(parents=True, exist_ok=True)
    config.CHAMPION.write_text(champion.model_dump_json(), encoding="utf-8")
    return champion


def stub_evaluator(
    monkeypatch: pytest.MonkeyPatch, deep_crashes: bool = False
) -> list[str]:
    """Answer both evaluations from the source itself, without playing a game.

    A scheduling test needs a fitness to rank on, not a game to produce one.
    Both evaluations are patched on the module rather than injected, because
    the alternative is a parameter on ``run`` that exists only for tests.
    Score is the source's length, so a child ranks above the shorter program
    it was edited from by construction rather than by luck, and the rates are
    keyed by the pool the evaluation was handed, reduced through the real
    `evaluator.opponents` so a champion here does not play itself either.

    Args:
        monkeypatch: The test's patcher.
        deep_crashes: Whether every deep evaluation raises, as one does when
            the candidate fails in its own seat on the exam block.

    Returns:
        The program ids handed to the deep evaluation, in order.
    """
    scored: list[str] = []

    def score(agent: Path) -> float:
        """A stand-in fitness: longer source, higher score, never a full 1.0."""
        return min(0.99, len(agent.read_text(encoding="utf-8")) / 1000)

    def fast(
        agent: Path,
        program_id: str,
        opponents: pool.Pool,
        rng: random.Random,
        workers: int,
    ) -> evaluator.FastResult:
        """The fast evaluation's shape, without its games.

        No day table: a scheduling test asserts on what the loop did with a
        number, never on the states behind it, and the only game that could
        produce one is the game this stands in for.
        """
        measured = evaluator.opponents(opponents, program_id, agent)
        rate = score(agent)
        names = measured.names()
        return evaluator.FastResult(
            fitness=rate,
            rates=dict.fromkeys(names, rate),
            margins=dict.fromkeys(names, harness.Margin(mean=0.0, worst=0.0, best=0.0)),
            seeds=[1],
            hardest=names[0],
            states=[],
        )

    def deep(
        agent: Path, program_id: str, opponents: pool.Pool, workers: int
    ) -> evaluator.DeepResult:
        """The deep evaluation's shape, without its games."""
        scored.append(program_id)
        if deep_crashes:
            raise RuntimeError("the candidate raised in its own seat")
        measured = evaluator.opponents(opponents, program_id, agent)
        return _deep_result(program_id, dict.fromkeys(measured.names(), score(agent)))

    monkeypatch.setattr(evaluator, "fast", fast)
    monkeypatch.setattr(evaluator, "deep", deep)
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


def test_a_better_child_is_deep_scored_and_promoted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    log: wandb.Run,
    records: list[tuple[float, dict]],
) -> None:
    """Two sessions, no schedule: the better child is confirmed and shipped.

    A seam test: real validation, real games through the real harness, the
    real gate, and only the codex session stood in for.
    """
    tiny_run(tmp_path, monkeypatch)
    pass_pool(tmp_path)

    state = loop.run(
        sessions=2,
        mutator=mutate.FakeMutator(edit=lambda _: SELLER),
        workers=WORKERS,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        log=log,
    )

    assert state.sessions == 2
    champion = state.champion
    assert champion is not None and champion.result.score > 0.5
    assert Path(champion.path).read_text() == SELLER
    assert (config.FLOOR / "main.py").read_text() == SELLER
    assert Path(champion.tarball).exists()
    assert champion.name in pool.Pool.load(config.POOL).names()
    assert gate.load_champion() == champion
    promotions = [record for record in deeps_of(records) if record["deep/promoted"]]
    assert promotions and all(record["deep/score"] > 0.5 for record in promotions)
    # The champion carries the result it was promoted on: a program the
    # database holds, measured on the exam block, and what a session is then
    # shown of the program it starts from.
    database = archive.Database(config.ARCHIVE, config.PROGRAMS)
    assert champion.result.program_id in {p.id for p in database.programs}
    assert [record["sessions"] for record in sessions_of(records)] == [1, 2]


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
    tiny_run(tmp_path, monkeypatch)
    repo = _repository(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "SERVED", tmp_path / "src" / "served" / "main.py")
    pass_pool(tmp_path)
    stub_evaluator(monkeypatch)

    state = loop.run(
        sessions=1,
        mutator=mutate.FakeMutator(edit=lambda _: SELLER),
        workers=WORKERS,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        log=log,
    )

    # `config.ROOT` is the real checkout until here, because packaging a
    # champion reads the plumbing modules out of it. From here it is the
    # repository the promotion could have dirtied.
    assert state.champion is not None
    monkeypatch.setattr(config, "ROOT", tmp_path)
    assert repo.git.status("--porcelain", "--", "src") == ""
    loop._open_run(dry_run=True).finish()


def test_the_seed_is_never_deep_scored_or_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """The gate confirms programs sessions wrote, not the one the campaign began on.

    At the production ``DEEP_TOP_K`` the seed shares the top three with the
    first children, and a child that ranks below it leaves the seed the best
    program in the database. Confirming it would promote it -- there is no
    champion to beat on a cold start -- and `champion_1` would join the pool
    at a fifth of the weight as an opponent that loses to everything, a
    constant added to every candidate's score that separates none of them.
    """
    tiny_run(tmp_path, monkeypatch)
    pass_pool(tmp_path)
    scored = stub_evaluator(monkeypatch)

    state = loop.run(
        sessions=1,
        # A child that ranks *below* the seed, so the seed is the top program.
        mutator=mutate.FakeMutator(edit=lambda _: PASS),
        workers=WORKERS,
        seed_agent=_write(tmp_path / "seed.py", SELLER),
        rng=random.Random(0),
        log=log,
    )

    assert config.DEEP_TOP_K == 3
    # One deep evaluation, and it is the child's: never the seed's, though the
    # seed was the best program in the database the whole time.
    assert loop.SEED_ID not in scored and len(scored) == 1
    # And nothing was promoted, because that child beats nobody -- which is
    # what the seed would have been promoted on if it had been confirmed.
    assert state.champion is None
    assert not (config.FLOOR / "main.py").exists()


def test_a_deep_evaluation_that_crashes_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """The exam block is spent on a program once, whatever it costs the program.

    A candidate that raises in its own seat is the lineage's failure, and it
    is recorded as one. What must not happen is the program keeping its place
    in the top three with no result to show for it: every later insert would
    send it back through the gate, and each retry is ten minutes of the exam
    block holding one of the two deep slots. Three restarts, so the ledger --
    not just the in-flight set -- is what has to remember.
    """
    tiny_run(tmp_path, monkeypatch)
    strong_champion(tmp_path)
    scored = stub_evaluator(monkeypatch, deep_crashes=True)
    seed = _write(tmp_path / "seed.py", PASS)
    for session in range(3):
        mutator = mutate.FakeMutator(edit=lambda s, n=session: f"{s}\n# child {n}\n")
        loop.run(1, mutator, WORKERS, seed, random.Random(session), log)

    assert len(scored) == 3
    assert sorted(scored) == sorted(set(scored))
    database = archive.Database(config.ARCHIVE, config.PROGRAMS)
    crashed = [
        failure
        for program in database.programs
        for failure in database.failures(program.id)
    ]
    assert len(crashed) == 3
    assert all(failure.reason.startswith("deep: ") for failure in crashed)


def test_eight_workers_run_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """Every worker's session is in flight together, not one after another.

    Each session stamps the moment its agent starts editing and the moment it
    stops; the latest start lands before the earliest stop only if all eight
    overlapped.
    """
    tiny_run(tmp_path, monkeypatch)
    strong_champion(tmp_path)
    stub_evaluator(monkeypatch)
    starts: list[float] = []
    stops: list[float] = []

    def edit(source: str) -> str:
        """Hold the session open long enough for the others to have started."""
        starts.append(time.time())
        time.sleep(0.5)
        stops.append(time.time())
        return f"{source}\n# session {len(starts)}\n"

    state = loop.run(
        sessions=config.SESSIONS,
        mutator=mutate.FakeMutator(edit=edit),
        workers=WORKERS,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        log=log,
    )

    assert state.sessions == config.SESSIONS
    assert len(starts) == len(stops) == config.SESSIONS
    assert max(starts) < min(stops)


def test_a_promotion_changes_what_the_next_session_starts_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """The champion a promotion wrote is the file the next session edits.

    A seam test: the champion here is one a real promotion produced, out of
    real games, and the second run reads it back off disk.
    """
    tiny_run(tmp_path, monkeypatch)
    pass_pool(tmp_path)
    seed = _write(tmp_path / "seed.py", PASS)
    mutator = Recorder(edit=lambda _: SELLER)

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log)

    assert [handed.child for handed in mutator.seen] == [PASS]
    assert (config.CHAMPIONS / "champion_1.py").read_text() == SELLER

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log)

    assert mutator.seen[1].child == SELLER
    assert "champion_1" in mutator.seen[1].message


def test_a_program_is_deep_scored_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """A program the gate has already confirmed is never sent through it again.

    Four restarts, so every session's gates have finished before the next
    session opens and the top three are the same programs each time: a
    database that forgot which programs carry a deep result would spend the
    exam block on them over and over.
    """
    tiny_run(tmp_path, monkeypatch)
    strong_champion(tmp_path)
    scored = stub_evaluator(monkeypatch)
    seed = _write(tmp_path / "seed.py", PASS)
    for session in range(4):
        mutator = mutate.FakeMutator(edit=lambda s, n=session: f"{s}\n# child {n}\n")
        loop.run(1, mutator, WORKERS, seed, random.Random(session), log)

    assert len(scored) > 2
    assert sorted(scored) == sorted(set(scored))


def test_a_promotion_costs_exactly_one_deep_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """Nothing is measured twice to promote once.

    The champion used to be re-scored after joining the pool, so that the
    number a candidate was compared against had been measured on the pool as
    it then stood. Nothing is compared against the champion any more -- the
    gate asks only whether a candidate beat every opponent -- so that second
    exam block bought nothing, and it cost ten minutes of the only two deep
    slots there are.
    """
    tiny_run(tmp_path, monkeypatch)
    pass_pool(tmp_path)
    scored = stub_evaluator(monkeypatch)

    state = loop.run(
        sessions=1,
        mutator=mutate.FakeMutator(edit=lambda _: SELLER),
        workers=WORKERS,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        log=log,
    )

    assert state.champion is not None and state.champion.name == "champion_1"
    assert len(scored) == 1 and scored[0] == state.champion.result.program_id


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
    tiny_run(tmp_path, monkeypatch)
    pass_pool(tmp_path)
    stub_evaluator(monkeypatch)
    threads: list[str] = []
    add_champion = pool.Pool.add_champion

    def watched(
        self: pool.Pool, name: str, path: str, rates: dict[str, float]
    ) -> str | None:
        """The real pool change, with a note of the thread that made it."""
        threads.append(threading.current_thread().name)
        return add_champion(self, name, path, rates)

    monkeypatch.setattr(pool.Pool, "add_champion", watched)

    state = loop.run(
        sessions=1,
        mutator=mutate.FakeMutator(edit=lambda _: SELLER),
        workers=WORKERS,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        log=log,
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
    tiny_run(tmp_path, monkeypatch)
    pass_pool(tmp_path)
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
    )

    assert [artifact.name for artifact in artifacts] == ["champion_1"]
    assert list(artifacts[0].manifest.entries) == ["champion_1.tar.gz"]
    assert (config.CHAMPIONS / "champion_1.tar.gz").exists()


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
    tiny_run(tmp_path, monkeypatch, rounds=3)
    pass_pool(tmp_path)
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
        mutator=mutate.CodexMutator(model="a", fallback="b", timeout=30),
        workers=WORKERS,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        log=log,
    )

    assert state.sessions == 1
    assert '"event": "failure"' not in config.ARCHIVE.read_text()
    assert [record["calls/ok"] for record in calls_of(records)] == [0]
    assert [record["calls/fallback"] for record in calls_of(records)] == [1]
    assert [record["sessions/rounds"] for record in sessions_of(records)] == [1]


def test_a_broken_pool_opponent_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """An opponent crashing in its own seat halts the loop rather than the lineage."""
    tiny_run(tmp_path, monkeypatch)
    crasher = pool.Pool(
        opponents={"crasher": str(_write(tmp_path / "crasher.py", CRASHER))}
    )
    crasher.save(config.POOL)
    # A cold start would evaluate the seed against this pool before the loop
    # ever began, so the database is seeded by hand.
    seed = _write(tmp_path / "seed.py", PASS)
    database = archive.Database(config.ARCHIVE, config.PROGRAMS)
    stored = database.store(PASS, "seed")
    database.add(
        archive.Program(
            id="seed",
            source_path=str(stored),
            started_from="",
            instruction="seed",
            model="",
            fitness=0.5,
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
        )

    assert '"event": "failure"' not in config.ARCHIVE.read_text()


def test_cancellation_kills_the_session_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """A signal stops the run and takes the session's whole process group with it.

    A session that outlived the loop would keep a core and write into a
    sandbox nothing owns any more.
    """
    tiny_run(tmp_path, monkeypatch)
    pass_pool(tmp_path)
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
        mutator=mutate.CodexMutator(timeout=60),
        workers=WORKERS,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        log=log,
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


def test_stagnation_switches_the_starting_program(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """With no promotion in sight, a session starts from the top ten and is told so."""
    tiny_run(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "STAGNATION_SESSIONS", 1)
    strong_champion(tmp_path)
    stub_evaluator(monkeypatch)
    seed = _write(tmp_path / "seed.py", PASS)
    marked = f"{SELLER}\n# a child of the champion\n"
    mutator = Recorder(edit=lambda _: marked)

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log)
    loop.run(1, mutator, WORKERS, seed, random.Random(0), log)

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
    would otherwise open the live campaign's own files.
    """
    tiny_run(tmp_path, monkeypatch)
    _repository(tmp_path, monkeypatch)
    (tmp_path / "src" / "agent.py").write_text("VERSION = 2\n", encoding="utf-8")
    monkeypatch.setattr(config, "ROOT", tmp_path)

    with pytest.raises(SystemExit, match="src/agent.py"):
        loop.main(["--sessions", "0", "--dry-run"])


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
    tiny_run(tmp_path, monkeypatch)
    champion = strong_champion(tmp_path)
    stub_evaluator(monkeypatch)
    state_file = config.ARCHIVE.with_name("state.json")
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(
        loop.State(sessions=7, sessions_since_promotion=3).model_dump_json(),
        encoding="utf-8",
    )

    state = loop.run(
        sessions=0,
        mutator=mutate.FakeMutator(edit=lambda source: source),
        workers=WORKERS,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        log=log,
    )

    assert state.sessions == 7 and state.sessions_since_promotion == 3
    assert state.champion == champion


def test_a_round_is_told_a_name_and_never_a_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """What a round starts from is interpolated raw, so it is a name, not a path.

    The champion reaches the message as its pool name, which is also how the
    verdict tells the round to beat it head to head. A path there would both
    break that sentence and point at the file it must not read. Everything a
    model is given is this one string, so this is the whole exposure.
    """
    tiny_run(tmp_path, monkeypatch)
    champion = strong_champion(tmp_path)
    stub_evaluator(monkeypatch)
    mutator = Recorder(edit=lambda _: SELLER)
    seed = _write(tmp_path / "seed.py", PASS)

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log)

    handed = mutator.seen[0]
    assert f"`{champion.name}`" in handed.message
    assert str(tmp_path) not in handed.message
    assert "/data/kaggriculture" not in handed.message
    assert prompt.DOCTRINE in handed.message


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


def _deep_result(
    program_id: str, rates: dict[str, float], score: float | None = None
) -> evaluator.DeepResult:
    """A hand-built deep result standing in for one the gate wrote."""
    point = sum(rates.values()) / len(rates) if score is None else score
    return evaluator.DeepResult(
        program_id=program_id,
        score=point,
        low=max(0.0, point - 0.05),
        high=min(1.0, point + 0.05),
        rates=rates,
        intervals={
            n: (max(0.0, r - 0.05), min(1.0, r + 0.05)) for n, r in rates.items()
        },
        field=point,
        held_out={},
        games=4,
    )


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
    tiny_run(tmp_path, monkeypatch, rounds=3)
    pass_pool(tmp_path)
    stub_evaluator(monkeypatch)
    seed = _write(tmp_path / "seed.py", PASS)
    mutator = Recorder(edit=lambda source: source + "# a round\n")

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log)

    handed = [given.child for given in mutator.seen]
    assert handed == [PASS, PASS + "# a round\n", PASS + "# a round\n" * 2]
    database = archive.Database(config.ARCHIVE, config.PROGRAMS)
    written = [p for p in database.programs if p.id != loop.SEED_ID]
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
    tiny_run(tmp_path, monkeypatch)
    pass_pool(tmp_path)
    stub_evaluator(monkeypatch)
    seed = _write(tmp_path / "seed.py", PASS)
    mutator = Recorder(edit=lambda source: source + "# a round\n")

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log)

    database = archive.Database(config.ARCHIVE, config.PROGRAMS)
    assert database.get(loop.SEED_ID).model == ""
    written = [p for p in database.programs if p.id != loop.SEED_ID]
    assert len(written) == 1 and written[0].model == "recorder"


def test_a_round_that_clears_the_bar_ends_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """There is nothing left to ask for once a program beats every opponent.

    The condition is `gate.promotion` on the round's own rates -- the same
    function, on the same reading of a win, that the message told the model
    it had to clear.
    """
    tiny_run(tmp_path, monkeypatch, rounds=3)
    pass_pool(tmp_path)
    stub_evaluator(monkeypatch)
    seed = _write(tmp_path / "seed.py", PASS)
    # Long enough that the stand-in fitness puts it above half against every
    # opponent, which is what beating them all means.
    mutator = Recorder(edit=lambda _: SELLER)

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log)

    assert len(mutator.seen) == 1


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
    tiny_run(tmp_path, monkeypatch, rounds=3)
    pass_pool(tmp_path)
    stub_evaluator(monkeypatch)
    monkeypatch.setattr(mutate.CodexMutator, "COMMAND", ["true"])

    loop.run(
        sessions=1,
        mutator=mutate.CodexMutator(model="a", fallback="", timeout=30),
        workers=WORKERS,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        log=log,
    )

    assert [record["sessions/rounds"] for record in sessions_of(records)] == [3]
    assert config.ARCHIVE.read_text().count('"no_output') == 3
    database = archive.Database(config.ARCHIVE, config.PROGRAMS)
    assert [program.id for program in database.programs] == [loop.SEED_ID]


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
    tiny_run(tmp_path, monkeypatch, rounds=2)
    pass_pool(tmp_path)
    stub_evaluator(monkeypatch)
    seed = _write(tmp_path / "seed.py", PASS)
    written = iter(["def agent(observation, configuration=None)\n", SELLER])
    mutator = Recorder(edit=lambda _: next(written))

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log)

    first, second = mutator.seen
    assert "produced nothing" not in first.message
    assert "- syntax: " in second.message
    assert second.child == first.child == PASS
    database = archive.Database(config.ARCHIVE, config.PROGRAMS)
    assert [p.started_from for p in database.programs] == ["", loop.SEED_ID]


def test_the_instruction_is_drawn_once_a_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """Rounds deepen one line; it is a new session that goes wider.

    Five rounds drawing for themselves would draw "a completely different
    algorithm" more than once in most sessions, which is several first rounds
    rather than one line taken further. The draw belongs to the session, and
    every program a session writes is stamped with it.
    """
    tiny_run(tmp_path, monkeypatch, rounds=5)
    pass_pool(tmp_path)
    stub_evaluator(monkeypatch)
    seed = _write(tmp_path / "seed.py", PASS)
    mutator = Recorder(edit=lambda source: source + "# a round\n")

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log)

    given = {handed.message.split("## Your instruction")[1] for handed in mutator.seen}
    assert len(mutator.seen) == 5 and len(given) == 1
    database = archive.Database(config.ARCHIVE, config.PROGRAMS)
    stamped = {p.instruction for p in database.programs if p.id != loop.SEED_ID}
    assert len(stamped) == 1


def test_the_session_budget_ends_it_before_the_rounds_do(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """A session is bounded by whichever of the two comes first."""
    tiny_run(tmp_path, monkeypatch, rounds=3)
    monkeypatch.setattr(config, "SESSION_LIMIT_SECONDS", 0)
    pass_pool(tmp_path)
    stub_evaluator(monkeypatch)
    seed = _write(tmp_path / "seed.py", PASS)
    mutator = Recorder(edit=lambda source: source + "# a round\n")

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log)

    assert len(mutator.seen) == 1


def test_a_round_is_given_one_file_and_the_directory_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """One file, `child.py`, and nothing else a path could leak through.

    The engine copy, the standing rules and the feedback file were all things
    a session read off disk; nothing reads them now, so nothing is written.
    The directory goes once its program is in the database.
    """
    tiny_run(tmp_path, monkeypatch)
    pass_pool(tmp_path)
    stub_evaluator(monkeypatch)
    seed = _write(tmp_path / "seed.py", PASS)
    mutator = Recorder(edit=lambda _: SELLER)

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log)

    handed = mutator.seen[0]
    assert handed.held == ["child.py"]
    assert not handed.where.exists()


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
    tiny_run(tmp_path, monkeypatch)
    pass_pool(tmp_path)
    seed = _write(tmp_path / "seed.py", PASS)
    mutator = Recorder(edit=lambda _: SELLER)

    loop.run(1, mutator, WORKERS, seed, random.Random(0), log)

    message = mutator.seen[0].message
    assert f"The verdict on `{loop.SEED_ID}`" in message
    # PASS against PASS is a dead heat, and a dead heat is not a win.
    assert "It did not beat pass at 0.500." in message
    assert "One game against `pass`, day by day" in message
    assert message.count("\n| 2") + message.count("\n| 1") > 0
    assert "| 29 |" in message
