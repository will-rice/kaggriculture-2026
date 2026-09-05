"""The whole loop end to end on real operators, with only codex stood in for.

Everything here is the campaign's own code: real validation, real games
through the real harness and its process pools, a real archive, pool and
gate, real ``state.json`` and ``champion.json`` under ``tmp_path``. The one
substitution is the codex session itself -- a ``FakeMutator``, or a
``CodexMutator`` whose ``COMMAND`` is an ordinary shell command -- because a
real session costs quota. Concurrency, the core budget and the schedule are
therefore observed the way an operator would: through the files the run
wrote, the archive it filled, and the metrics it logged.
"""

import asyncio
import random
import subprocess
import time
from pathlib import Path

import pytest
import wandb

from kaggriculture.campaign import archive as archive_module
from kaggriculture.campaign import (
    config,
    evaluator,
    gate,
    harness,
    loop,
    mutate,
    pool,
    roster,
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

# Every evaluation forks `workers` processes and `concurrency` of them can be
# in flight, so their product must fit the budget on the smallest box too.
WORKERS = max(1, min(4, config.CORE_BUDGET // 2))
CONCURRENCY = max(1, min(2, config.CORE_BUDGET // WORKERS))


@pytest.fixture
def records() -> list[tuple[float, dict]]:
    """Every metrics dict the loop hands to ``log.log``, with when it did.

    The stamp is wall clock rather than a monotonic count, so it can be
    compared with what a session's own ``date`` wrote.
    """
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


def calls_of(records: list[tuple[float, dict]]) -> list[tuple[float, dict]]:
    """The per-call records, in the order the loop logged them."""
    return [(when, record) for when, record in records if "calls/ok" in record]


def epochs_of(records: list[tuple[float, dict]]) -> list[tuple[float, dict]]:
    """The per-epoch records, in the order the loop logged them."""
    return [(when, record) for when, record in records if "deep/promoted" in record]


def tiny_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, vendored: str = "pass"
) -> None:
    """Point every runtime path at ``tmp_path`` and shrink the loop to one seed.

    The one-opponent test pool stands in for the vendored field, and the real
    held-out set is dropped: a dry run measures the pipeline, not a field, and
    playing the held-out opponents would cost games nothing here asserts on.

    Args:
        tmp_path: The test's own directory; every runtime path lands under it.
        monkeypatch: The test's patcher.
        vendored: The pool opponent that stands in for the vendored field.
    """
    run = tmp_path / "run"
    for name in (
        "ARCHIVE",
        "PROGRAMS",
        "SANDBOXES",
        "FLOOR",
        "CHAMPIONS",
        "CHAMPION",
    ):
        monkeypatch.setattr(
            config, name, run / getattr(config, name).relative_to(config.RUN)
        )
    monkeypatch.setattr(config, "POOL", run / "pool.json")
    monkeypatch.setattr(config, "EXAM_SEEDS", config.EXAM_SEEDS[:2])
    monkeypatch.setattr(config, "EPOCH_INTERVAL", 1)
    monkeypatch.setattr(config, "MIGRATION_INTERVAL", 10**6)
    monkeypatch.setattr(config, "RESET_INTERVAL", 10**6)
    monkeypatch.setattr(config, "ISLANDS", 2)
    monkeypatch.setattr(config, "ISLAND_SIZE", 3)
    monkeypatch.setattr(config, "FAST_SEEDS", 1)
    monkeypatch.setattr(gate, "SERVED", tmp_path / "served.py")
    monkeypatch.setattr(evaluator, "VENDORED", [vendored])
    monkeypatch.setattr(evaluator, "HELD_OUT", [])


def pass_pool(tmp_path: Path) -> pool.Pool:
    """A saved one-opponent pool whose opponent is a PASS agent."""
    p = pool.Pool(
        opponents={"pass": str(_write(tmp_path / "pass.py", PASS))},
        weights={"pass": 1.0},
    )
    p.save(config.POOL)
    return p


def test_two_calls_end_to_end_promote_a_better_child_over_a_pass_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    log: wandb.Run,
    records: list[tuple[float, dict]],
) -> None:
    """Two calls end to end: mutate, validate, insert, deep-evaluate, promote."""
    tiny_run(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "EPOCH_INTERVAL", 2)
    pass_pool(tmp_path)

    state = loop.run(
        calls=2,
        mutator=mutate.FakeMutator(edit=lambda _: SELLER),
        workers=WORKERS,
        concurrency=CONCURRENCY,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        log=log,
        commit=False,
    )

    assert state.calls == 2
    assert state.champion is not None and state.champion.score > 0.5
    assert (config.FLOOR / "main.py").read_text() == SELLER
    assert state.champion_path is not None
    assert Path(state.champion_path).read_text() == SELLER
    assert "champion_1" in pool.Pool.load(config.POOL).names()
    assert [record["calls"] for _, record in calls_of(records)] == [1, 2]
    assert [record["calls/ok"] for _, record in calls_of(records)] == [1, 1]
    epochs = [record for _, record in epochs_of(records)]
    assert len(epochs) == 1 and epochs[0]["deep/promoted"] == 1
    assert epochs[0]["deep/best"] == state.champion.score
    assert epochs[0]["calls"] == 2
    # The state file is the resume point, and it is current the moment the
    # last call finished, not once `run` returns.
    written = loop.State.model_validate_json(
        config.ARCHIVE.with_name("state.json").read_text(encoding="utf-8")
    )
    assert written.calls == 2 and written.calls_today == 2


@pytest.mark.skipif(
    config.CORE_BUDGET < 4, reason="four sessions in flight need four cores"
)
def test_four_sessions_run_at_once_rather_than_one_after_another(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """The barrier is gone: every session the concurrency allows is in flight.

    The old loop fired one mutation per island and waited for the slowest,
    so four one-second sessions took four seconds end to end. Each session
    here stamps the moment it starts and the moment it stops; the latest
    start lands before the earliest stop only if all four overlapped.
    """
    tiny_run(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "EPOCH_INTERVAL", 10**6)
    pass_pool(tmp_path)
    starts, stops = tmp_path / "starts", tmp_path / "stops"
    monkeypatch.setattr(
        mutate.CodexMutator,
        "COMMAND",
        [
            "bash",
            "-c",
            f"date +%s.%N >> {starts}; sleep 1; cp parent.py child.py; "
            f"date +%s.%N >> {stops}",
        ],
    )

    state = loop.run(
        calls=4,
        mutator=mutate.CodexMutator(timeout=30),
        workers=max(1, config.CORE_BUDGET // 4),
        concurrency=4,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        log=log,
        commit=False,
    )

    assert state.calls == 4
    began = [float(line) for line in starts.read_text().split()]
    ended = [float(line) for line in stops.read_text().split()]
    assert len(began) == len(ended) == 4
    assert max(began) < min(ended)


def test_an_epoch_never_stops_the_dispatcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    log: wandb.Run,
    records: list[tuple[float, dict]],
) -> None:
    """Sessions keep starting and calls keep completing while an epoch runs.

    The epoch is created inside the completion step of the call that crossed
    the interval, so that call's record is the moment it began, and its own
    record is the moment it ended. What an epoch could not do under the old
    barrier -- where it ran to completion on the one thread that dispatches
    -- is let a new codex session start while it works. Sessions started
    after that first record are exactly that, and they are why the crossings
    that follow find the epoch still running and are skipped.
    """
    tiny_run(tmp_path, monkeypatch)
    pass_pool(tmp_path)
    starts = tmp_path / "starts"
    monkeypatch.setattr(
        mutate.CodexMutator,
        "COMMAND",
        ["bash", "-c", f"date +%s.%N >> {starts}; sleep 1; cp parent.py child.py"],
    )

    state = loop.run(
        calls=6,
        mutator=mutate.CodexMutator(timeout=30),
        workers=WORKERS,
        concurrency=1,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        log=log,
        commit=False,
    )

    assert state.calls == 6
    calls = calls_of(records)
    epochs = epochs_of(records)
    assert [record["calls"] for _, record in calls] == [1, 2, 3, 4, 5, 6]
    # A session's `date` and the record's stamp are both wall clock.
    began = [float(line) for line in starts.read_text().split()]
    launched, finished = calls[0][0], epochs[0][0]
    assert len(began) == 6
    assert any(launched < start < finished for start in began)
    # Six crossings, and the epoch the first one started was still running
    # for at least one of them.
    assert 1 <= len(epochs) < 6


def test_a_spent_daily_budget_parks_dispatch_until_the_day_rolls_over(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    log: wandb.Run,
    records: list[tuple[float, dict]],
) -> None:
    """A spent cap costs the day's calls, not the schedule.

    Parking must not advance ``calls``: walking migration, the epoch and the
    island reset over a population nothing touched would burn the exam block
    on the same programs once a minute until the quota resets.
    """
    tiny_run(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "EPOCH_INTERVAL", 10**6)
    monkeypatch.setattr(config, "DAILY_CALL_BUDGET", 1)
    monkeypatch.setattr(loop, "QUOTA_SLEEP_SECONDS", 0.02)
    pass_pool(tmp_path)
    # The one input a test cannot wait for: the calendar. The dispatcher reads
    # it once to plan the first call and once per parked retry, so three days
    # of the first date leave the second call parked twice before midnight.
    days = iter(["2026-01-01"] * 3)
    monkeypatch.setattr(loop, "today", lambda: next(days, "2026-01-02"))

    state = loop.run(
        calls=2,
        mutator=mutate.FakeMutator(edit=lambda _: SELLER),
        workers=WORKERS,
        concurrency=CONCURRENCY,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        log=log,
        commit=False,
    )

    assert state.calls == 2
    assert state.day == "2026-01-02" and state.calls_today == 1
    # Two calls, no third: parking logs nothing and advances nothing.
    assert [record["calls/today"] for _, record in calls_of(records)] == [1, 1]
    assert not epochs_of(records)


def test_a_broken_pool_opponent_stops_the_run_and_takes_the_sessions_with_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """An opponent crashing in its own seat halts the loop and kills what is live.

    It is the pool that is broken, not the candidate, so the failure must not
    become a lineage's feedback. Everything in flight is cancelled with it,
    process groups included: a session that outlived the loop would keep a
    core and write into a sandbox nothing owns any more.
    """
    tiny_run(tmp_path, monkeypatch)
    p = pool.Pool(
        opponents={"crasher": str(_write(tmp_path / "crasher.py", CRASHER))},
        weights={"crasher": 1.0},
    )
    p.save(config.POOL)
    # `run` seeds a cold archive by evaluating the seed, which this pool
    # would break before the loop ever started.
    archive_module.Archive(config.ARCHIVE, config.PROGRAMS).seed(
        _write(tmp_path / "seed.py", PASS), fitness=0.5
    )
    # `mkdir` is atomic, so exactly one of the two sessions takes the first
    # branch and finishes; the other records its process group and sleeps.
    held = tmp_path / "held"
    pgid_file = tmp_path / "pgid"
    monkeypatch.setattr(
        mutate.CodexMutator,
        "COMMAND",
        [
            "bash",
            "-c",
            f"cp parent.py child.py; mkdir {held} 2>/dev/null && exit 0; "
            f"ps -o pgid= -p $$ | tr -d ' ' > {pgid_file}; sleep 30",
        ],
    )

    with pytest.raises(harness.OpponentCrash):
        loop.run(
            calls=2,
            mutator=mutate.CodexMutator(timeout=60),
            workers=WORKERS,
            concurrency=2,
            seed_agent=tmp_path / "seed.py",
            rng=random.Random(0),
            log=log,
            commit=False,
        )

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


def test_a_champion_json_outranks_a_state_json_that_never_saw_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """A kill between the promotion and the state write must not lose the champion.

    `state.json` is written after every completed call; `champion.json` is
    written before `gate.promote` returns. Restarting on the first would leave
    the gate with no baseline and promote a second time.
    """
    tiny_run(tmp_path, monkeypatch)
    champion_file = _write(tmp_path / "champion_1.py", SELLER)
    p = pool.Pool(
        opponents={
            "pass": str(_write(tmp_path / "pass.py", PASS)),
            "champion_1": str(champion_file),
        },
        weights={"pass": 0.5, "champion_1": 0.5},
    )
    p.save(config.POOL)
    record = gate.Champion(
        name="champion_1",
        path=str(champion_file),
        result=_deep_result("champion_1", {"pass": 1.0, "champion_1": 0.5}),
    )
    config.CHAMPION.parent.mkdir(parents=True, exist_ok=True)
    config.CHAMPION.write_text(record.model_dump_json(), encoding="utf-8")
    state_file = config.ARCHIVE.with_name("state.json")
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(loop.State(calls=7).model_dump_json(), encoding="utf-8")

    state = loop.run(
        calls=0,
        mutator=mutate.FakeMutator(edit=lambda source: source),
        workers=WORKERS,
        concurrency=CONCURRENCY,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        log=log,
        commit=False,
    )

    assert state.calls == 7
    assert state.champion == record.result
    assert state.champion_path == str(champion_file)


def test_a_weaker_candidate_is_not_promoted_over_a_resumed_champion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """The resumed champion is the baseline, so a PASS child stays where it is."""
    tiny_run(tmp_path, monkeypatch)
    champion_file = _write(tmp_path / "champion_1.py", SELLER)
    p = pool.Pool(
        opponents={
            "pass": str(_write(tmp_path / "pass.py", PASS)),
            "champion_1": str(champion_file),
        },
        weights={"pass": 0.5, "champion_1": 0.5},
    )
    p.save(config.POOL)
    record = gate.Champion(
        name="champion_1",
        path=str(champion_file),
        result=_deep_result("champion_1", {"pass": 1.0, "champion_1": 0.5}),
    )
    config.CHAMPION.parent.mkdir(parents=True, exist_ok=True)
    config.CHAMPION.write_text(record.model_dump_json(), encoding="utf-8")

    loop.run(
        calls=1,
        mutator=mutate.FakeMutator(edit=lambda _: PASS),
        workers=WORKERS,
        concurrency=CONCURRENCY,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        log=log,
        commit=False,
    )

    assert "champion_2" not in pool.Pool.load(config.POOL).names()
    assert gate.load_champion() == record


def test_the_epoch_scores_the_champion_on_the_pool_the_candidate_faced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """The comparison baseline is re-measured, not the one frozen at promotion.

    A champion's stored result was computed before it joined the pool, before
    the weights renormalised and before weakness pressure moved them. Comparing
    a candidate's fresh score against that is two numbers from two different
    exams, so the epoch re-runs the champion first and compares on the pool as
    it stands now. The rates it carries name the opponents it actually played,
    which is how a frozen baseline and a fresh one tell each other apart.
    """
    tiny_run(tmp_path, monkeypatch)
    p = pass_pool(tmp_path)
    archive = archive_module.Archive(config.ARCHIVE, config.PROGRAMS)
    archive.seed(_write(tmp_path / "seller.py", SELLER), fitness=0.9)
    campaign = loop.Campaign(
        loop.State(),
        archive,
        p,
        mutate.FakeMutator(edit=lambda source: source),
        WORKERS,
        CONCURRENCY,
        random.Random(0),
        log,
        commit=False,
    )

    asyncio.run(campaign.epoch())

    assert campaign.state.champion_path is not None
    # Promoted against a pool of one, so the result frozen at promotion knows
    # only that opponent.
    assert campaign.state.champion is not None
    assert set(campaign.state.champion.rates) == {"pass"}

    asyncio.run(campaign.epoch())

    # The second epoch re-measured the champion on the pool it had just
    # joined, so its baseline names itself; the frozen one never could.
    assert set(campaign.state.champion.rates) == {"pass", "champion_1"}


def test_run_refuses_a_mutation_share_that_cannot_fit_the_core_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run
) -> None:
    """Every session in flight can be evaluating at once, so the peak is the product."""
    tiny_run(tmp_path, monkeypatch)
    pass_pool(tmp_path)

    with pytest.raises(ValueError, match="CORE_BUDGET"):
        loop.run(
            calls=1,
            mutator=mutate.FakeMutator(edit=lambda source: source),
            workers=config.CORE_BUDGET,
            concurrency=2,
            seed_agent=_write(tmp_path / "seed.py", PASS),
            rng=random.Random(0),
            log=log,
            commit=False,
        )


def test_the_core_counter_serialises_takers_that_do_not_fit_together() -> None:
    """Two evaluations each wanting the whole budget never overlap."""

    async def body(budget: int, share: int) -> list[str]:
        """Run two takers of ``share`` permits against a ``budget`` counter."""
        cores = loop.Cores(budget)
        trace: list[str] = []

        async def taker(name: str) -> None:
            async with cores.take(share):
                trace.append(f"in {name}")
                await asyncio.sleep(0.05)
                trace.append(f"out {name}")

        await asyncio.gather(taker("a"), taker("b"))
        return trace

    assert asyncio.run(body(4, 4)) == ["in a", "out a", "in b", "out b"]
    assert asyncio.run(body(8, 4)) == ["in a", "in b", "out a", "out b"]


def test_the_core_counter_refuses_a_share_larger_than_the_whole_budget() -> None:
    """A wait no release could ever satisfy is an error, not a deadlock."""

    async def body() -> None:
        """Ask for more permits than exist."""
        async with loop.Cores(4).take(5):
            pass

    with pytest.raises(ValueError, match="budget is 4"):
        asyncio.run(body())


@pytest.mark.local_data
def test_four_calls_against_a_vendored_opponent_fill_the_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    log: wandb.Run,
    records: list[tuple[float, dict]],
) -> None:
    """The whole pipeline on real games against a real opponent.

    Nothing is stood in for here but the codex session: the children are
    validated, played against a vendored opponent on the engine, ranked,
    inserted and deep-evaluated exactly as a live call is.
    """
    tiny_run(tmp_path, monkeypatch, vendored="v54")
    monkeypatch.setattr(config, "EPOCH_INTERVAL", 4)
    p = pool.Pool(opponents={"v54": str(roster.TRAINING["v54"])}, weights={"v54": 1.0})
    p.save(config.POOL)

    state = loop.run(
        calls=4,
        mutator=mutate.FakeMutator(edit=lambda _: SELLER),
        workers=WORKERS,
        concurrency=2,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        log=log,
        commit=False,
    )

    assert state.calls == 4
    archive = archive_module.Archive(config.ARCHIVE, config.PROGRAMS)
    programs = [p for i in range(config.ISLANDS) for p in archive.island(i)]
    # Every call yielded a child that validated and scored: four inserts on
    # top of the seed each island started from, which carries no rates of its
    # own because nothing measured it per opponent.
    assert len(programs) == config.ISLANDS + 4
    scored = [program for program in programs if program.rates]
    assert len(scored) == 4
    assert all(program.rates.keys() == {"v54"} for program in scored)
    assert [record["calls/ok"] for _, record in calls_of(records)] == [1, 1, 1, 1]
    assert all("fast/fitness" in record for _, record in calls_of(records))
    epochs = [record for _, record in epochs_of(records)]
    assert len(epochs) == 1 and "deep/best" in epochs[0]


def _write(path: Path, source: str) -> Path:
    """Write ``source`` to ``path`` and return it."""
    path.write_text(source, encoding="utf-8")
    return path


def _deep_result(program_id: str, rates: dict[str, float]) -> evaluator.DeepResult:
    """A hand-built deep result standing in for one the gate wrote."""
    score = sum(rates.values()) / len(rates)
    return evaluator.DeepResult(
        program_id=program_id,
        score=score,
        low=score - 0.05,
        high=min(1.0, score + 0.05),
        rates=rates,
        intervals={
            n: (max(0.0, r - 0.05), min(1.0, r + 0.05)) for n, r in rates.items()
        },
        field=score,
        held_out={},
        games=4,
    )
