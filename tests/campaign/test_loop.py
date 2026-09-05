"""The whole loop on a fake mutator and a small pool, end to end."""

import datetime
import random
from collections.abc import Callable
from pathlib import Path

import pytest

from kaggriculture.campaign import archive as archive_module
from kaggriculture.campaign import config, evaluator, gate, loop, mutate, pool

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

# Every evaluation forks `workers` processes and `concurrency` of them can be
# in flight, so their product must fit the budget on the smallest box too.
WORKERS = max(1, min(4, config.CORE_BUDGET // 2))
CONCURRENCY = max(1, min(2, config.CORE_BUDGET // WORKERS))


def tiny_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point every runtime path at ``tmp_path`` and shrink the loop to one seed.

    The one-opponent test pool stands in for the vendored field, and the real
    held-out set is dropped: a dry run measures the pipeline, not a field, and
    playing the held-out opponents would cost games nothing here asserts on.
    """
    run = tmp_path / "run"
    for name in (
        "ARCHIVE",
        "PROGRAMS",
        "SANDBOXES",
        "EPOCHS",
        "FLOOR",
        "CHAMPIONS",
        "CHAMPION",
        "CALLS",
    ):
        monkeypatch.setattr(
            config, name, run / getattr(config, name).relative_to(config.RUN)
        )
    monkeypatch.setattr(config, "POOL", run / "pool.json")
    monkeypatch.setattr(config, "EXAM_SEEDS", config.EXAM_SEEDS[:2])
    monkeypatch.setattr(config, "EPOCH_INTERVAL", 1)
    monkeypatch.setattr(config, "ISLANDS", 2)
    monkeypatch.setattr(config, "ISLAND_SIZE", 3)
    monkeypatch.setattr(config, "FAST_SEEDS", 1)
    monkeypatch.setattr(gate, "SERVED", tmp_path / "served.py")
    monkeypatch.setattr(evaluator, "VENDORED", ["pass"])
    monkeypatch.setattr(evaluator, "HELD_OUT", [])


def test_dry_run_promotes_a_better_child_over_a_pass_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One iteration end to end: mutate, insert, deep-evaluate, promote."""
    tiny_run(tmp_path, monkeypatch)
    pass_agent = tmp_path / "pass.py"
    pass_agent.write_text(PASS)
    seed = tmp_path / "seed.py"
    seed.write_text(PASS)
    p = pool.Pool(opponents={"pass": str(pass_agent)}, weights={"pass": 1.0})
    p.save(config.POOL)

    state = loop.run(
        iterations=1,
        mutator=mutate.FakeMutator(edit=lambda _: SELLER),
        workers=WORKERS,
        concurrency=CONCURRENCY,
        seed_agent=seed,
        rng=random.Random(0),
        commit=False,
    )

    assert state.champion is not None and state.champion.score > 0.5
    assert (config.FLOOR / "main.py").read_text() == SELLER
    assert state.champion_path is not None
    assert Path(state.champion_path).read_text() == SELLER
    assert "champion_1" in pool.Pool.load(config.POOL).names()
    assert config.EPOCHS.read_text().count("\n") == 1


def test_the_epoch_scores_the_champion_on_the_pool_the_candidate_faced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The comparison baseline is re-measured, not the one frozen at promotion.

    A champion's stored result was computed before it joined the pool, before
    the weights renormalised and before weakness pressure moved them. Comparing
    a candidate's fresh score against that is two numbers from two different
    exams, so the epoch re-runs the champion first and compares on the pool as
    it stands now.
    """
    tiny_run(tmp_path, monkeypatch)
    pass_agent = tmp_path / "pass.py"
    pass_agent.write_text(PASS)
    p = pool.Pool(opponents={"pass": str(pass_agent)}, weights={"pass": 1.0})
    p.save(config.POOL)
    archive = archive_module.Archive(config.ARCHIVE, config.PROGRAMS)
    archive.seed(_write(tmp_path / "seller.py", SELLER), fitness=0.9)

    state = loop.epoch(loop.State(), archive, p, workers=WORKERS, commit=False)
    assert state.champion_path is not None
    promoted_rates = set(state.champion.rates) if state.champion else set()

    order: list[str] = []
    baselines: list[evaluator.DeepResult | None] = []
    real_deep, real_promotion = evaluator.deep, gate.promotion

    def spy_deep(
        agent: Path, program_id: str, opponents: pool.Pool, workers: int
    ) -> evaluator.DeepResult:
        """Record which file is deep-evaluated, then evaluate it for real."""
        order.append(f"deep:{agent}")
        return real_deep(agent, program_id, opponents, workers)

    def spy_promotion(
        candidate: evaluator.DeepResult, champion: evaluator.DeepResult | None
    ) -> tuple[bool, str]:
        """Record that the rule ran, and on which champion result."""
        order.append("promotion")
        baselines.append(champion)
        return real_promotion(candidate, champion)

    monkeypatch.setattr(evaluator, "deep", spy_deep)
    monkeypatch.setattr(gate, "promotion", spy_promotion)

    loop.epoch(state, archive, p, workers=WORKERS, commit=False)

    assert order.index(f"deep:{state.champion_path}") < order.index("promotion")
    # The pool gained champion_1 at the first promotion, so a baseline measured
    # on the current pool names it and the frozen one cannot.
    assert promoted_rates == {"pass"}
    baseline = baselines[0]
    assert baseline is not None and set(baseline.rates) == {"pass", "champion_1"}


def test_a_champion_json_outranks_a_state_json_that_never_saw_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A kill between the promotion and the state write must not lose the champion.

    `state.json` is written only once an iteration finishes; `champion.json` is
    written before `gate.promote` returns. Restarting on the first would leave
    the gate with no baseline and promote a second time.
    """
    tiny_run(tmp_path, monkeypatch)
    pass_agent = tmp_path / "pass.py"
    pass_agent.write_text(PASS)
    champion_file = _write(tmp_path / "champion_1.py", SELLER)
    p = pool.Pool(
        opponents={"pass": str(pass_agent), "champion_1": str(champion_file)},
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
    state_file.write_text(loop.State(iteration=7).model_dump_json(), encoding="utf-8")

    state = loop.run(
        iterations=0,
        mutator=mutate.FakeMutator(edit=lambda source: source),
        workers=WORKERS,
        concurrency=CONCURRENCY,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        commit=False,
    )

    assert state.iteration == 7
    assert state.champion == record.result
    assert state.champion_path == str(champion_file)


def test_a_weaker_candidate_is_not_promoted_over_a_resumed_champion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The resumed champion is the baseline, so a PASS child stays where it is."""
    tiny_run(tmp_path, monkeypatch)
    pass_agent = tmp_path / "pass.py"
    pass_agent.write_text(PASS)
    champion_file = _write(tmp_path / "champion_1.py", SELLER)
    p = pool.Pool(
        opponents={"pass": str(pass_agent), "champion_1": str(champion_file)},
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
        iterations=1,
        mutator=mutate.FakeMutator(edit=lambda _: PASS),
        workers=WORKERS,
        concurrency=CONCURRENCY,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        commit=False,
    )

    assert "champion_2" not in pool.Pool.load(config.POOL).names()
    assert gate.load_champion() == record


def test_a_spent_quota_idles_without_advancing_the_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An iteration that ran no mutation left the archive as it found it.

    Advancing the counter anyway would walk migration, the epoch and the island
    reset over a population nothing touched -- burning the exam block on the
    same programs once a minute until the quota resets.
    """
    tiny_run(tmp_path, monkeypatch)
    monkeypatch.setattr(loop, "QUOTA_SLEEP_SECONDS", 0)
    monkeypatch.setattr(loop, "epoch", _never("epoch"))
    archive = archive_module.Archive(config.ARCHIVE, config.PROGRAMS)
    p = pool.Pool(opponents={"pass": "/x/pass.py"}, weights={"pass": 1.0})
    spent = loop.State(
        iteration=24,
        calls_today=config.DAILY_CALL_BUDGET,
        day=datetime.date.today().isoformat(),
    )

    after = loop.iterate(
        spent,
        archive,
        p,
        mutator=_refusing_mutator,
        workers=WORKERS,
        concurrency=CONCURRENCY,
        rng=random.Random(0),
    )

    assert after == spent


def test_the_epoch_gets_the_whole_core_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mutation executor has joined by then, so nothing shares the box."""
    tiny_run(tmp_path, monkeypatch)
    archive = archive_module.Archive(config.ARCHIVE, config.PROGRAMS)
    archive.seed(_write(tmp_path / "seed.py", PASS), fitness=0.1)
    p = pool.Pool(opponents={"pass": "/x/pass.py"}, weights={"pass": 1.0})
    seen: list[int] = []

    def spy_epoch(
        state: loop.State,
        archive: archive_module.Archive,
        pool_: pool.Pool,
        workers: int,
        commit: bool = True,
    ) -> loop.State:
        """Record the worker count the epoch was handed."""
        seen.append(workers)
        return state

    monkeypatch.setattr(loop, "epoch", spy_epoch)
    monkeypatch.setattr(loop, "_mutate", lambda *args: None)
    # Distinct from the mutation share on every box, so a pass-through of
    # ``workers`` cannot satisfy the assertion by coincidence.
    monkeypatch.setattr(config, "CORE_BUDGET", 7)

    loop.iterate(
        loop.State(),
        archive,
        p,
        mutator=_refusing_mutator,
        workers=1,
        concurrency=1,
        rng=random.Random(0),
    )

    assert seen == [7]


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


def _refusing_mutator(sandbox: Path, program_id: str) -> mutate.Mutation:
    """A mutator that fails the test if the loop ever calls it.

    Raises:
        AssertionError: Always; the tests using it assert no mutation ran.
    """
    raise AssertionError(f"the mutator ran on {sandbox} for {program_id}")


def _never(what: str) -> Callable[..., None]:
    """Return a callable that fails the test if anything calls it."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"{what} ran when it should not have")

    return refuse
