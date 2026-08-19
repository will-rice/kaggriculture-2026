"""The held-out exam a searched route must pass before it ships."""

import ast
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from kaggriculture.search import arena
from kaggriculture.search.scripts import hillclimb, holdout
from kaggriculture.search.scripts.holdout import GATE_SEEDS, SERVED, Verdict, run

MAIN = Path("main.py")


def test_run_delegates_the_interval_to_report_wilson_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The verdict must come from `report.wilson_interval`'s own return value.

    `report.wilson_interval` already exists, already counts a tie as half a
    win, and already carries this repository's reasons for preferring Wilson
    over the normal approximation -- a second implementation of that formula
    inside this module would be the exact copy-drift class that left the price
    curve right in one place and stale in two others. Pairing a rate whose
    real interval would FAIL (0 wins) with a stubbed `wilson_interval` that
    returns a PASS-shaped interval is what makes this bite: the verdict can
    only follow the stub if `run` reads its return value rather than
    recomputing a bound of its own.
    """

    def outcomes_recorder(
        candidate: object,
        league: object,
        seeds: object,
        workers: object = None,
    ) -> list[float]:
        return [0.0] * 128

    def wilson_recorder(wins: float, games: int) -> tuple[float, float]:
        return (0.9, 0.95)

    monkeypatch.setattr(arena, "outcomes", outcomes_recorder)
    monkeypatch.setattr(holdout, "wilson_interval", wilson_recorder)

    verdict = run(candidate=[], against=SERVED)

    assert (verdict.low, verdict.high) == (0.9, 0.95)
    assert verdict.passed


def test_a_rate_above_half_with_an_interval_straddling_half_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rate over 50% is not sufficient; the whole interval must clear 0.5.

    65 of 128 is a rate of 0.508 -- above half -- but its lower bound (0.422)
    sits well below 0.5. That gap is the entire reason this gate exists rather
    than a bare `rate > 0.5` check: at this sample size a small edge is not
    distinguishable from a coin flip that happened to land favourably, which
    is exactly the case this test pins.
    """

    def recorder(
        candidate: object,
        league: object,
        seeds: object,
        workers: object = None,
    ) -> list[float]:
        return [1.0] * 65 + [0.0] * 63

    monkeypatch.setattr(arena, "outcomes", recorder)

    verdict = run(candidate=[], against=SERVED)

    assert verdict.rate > 0.5
    assert verdict.low < 0.5
    assert not verdict.passed


def test_a_rate_whose_interval_clears_half_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mirror case: an interval that clears 0.5 entirely must PASS."""

    def recorder(
        candidate: object,
        league: object,
        seeds: object,
        workers: object = None,
    ) -> list[float]:
        return [1.0] * 76 + [0.0] * 52

    monkeypatch.setattr(arena, "outcomes", recorder)

    verdict = run(candidate=[], against=SERVED)

    assert verdict.low > 0.5
    assert verdict.passed


def test_run_calls_outcomes_with_the_held_out_seeds_and_served_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exam must use `GATE_SEEDS` and the served-agent path -- structurally.

    A rate-only assertion cannot tell a gate that plays the right seeds from
    one that plays some other 128 games and happens to land on a similar
    number. Intercepting `arena.outcomes` and inspecting what it was called
    with is the only way to pin that the exam actually reads from
    `GATE_SEEDS`, which is the property the whole "held-out" claim rests on.
    """
    calls = []

    def recorder(
        candidate: object,
        league: object,
        seeds: Sequence[int],
        workers: object = None,
    ) -> list[float]:
        calls.append((candidate, league, seeds, workers))
        return [1.0] * (2 * len(seeds))

    monkeypatch.setattr(arena, "outcomes", recorder)
    candidate = [{"marker": "candidate"}]

    holdout.run(candidate, workers=4)

    assert calls == [(candidate, {"served": SERVED}, GATE_SEEDS, 4)]


def test_run_passes_a_non_default_opponent_through_to_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--against` must override which agent the league keys to `served`."""
    calls = []

    def recorder(
        candidate: object,
        league: object,
        seeds: Sequence[int],
        workers: object = None,
    ) -> list[float]:
        calls.append(league)
        return [1.0] * (2 * len(seeds))

    monkeypatch.setattr(arena, "outcomes", recorder)

    holdout.run([], against="src/kaggriculture/kaito_v23_policy.py")

    assert calls == [{"served": "src/kaggriculture/kaito_v23_policy.py"}]


def test_gate_seeds_are_disjoint_from_the_search_seed_pool() -> None:
    """A seed the hill-climb could search on must never double as the exam.

    Seeds the search ever saw cannot serve as its exam: a candidate accepted
    by the hill-climb is, by construction, whatever looked best on the boards
    it was searched against, so re-using one of those boards to grade it would
    measure the same selection effect the exam exists to catch. This documents
    the intent; `holdout.py` itself raises at import time if it is ever
    violated, so this test is a second line of defence, not the only one.
    """
    assert set(GATE_SEEDS).isdisjoint(hillclimb.SEED_POOL)


def test_gate_seeds_are_128_games() -> None:
    """64 seeds, each seat-swapped, is the 128 games the spec fixes."""
    assert len(GATE_SEEDS) == 64


def test_served_names_the_module_main_py_actually_imports_agent_from() -> None:
    """`SERVED` must track what `main.py` submits, not some other agent.

    Parsed with `ast` rather than imported: `main.py`'s own loader semantics
    are deliberately fragile (it must resolve both locally and on the
    competition runner), and importing it here would exercise that machinery
    for no reason. If `SERVED` ever points at a retired agent, this gate would
    pass a candidate that only beats a generation we no longer submit.
    """
    tree = ast.parse(MAIN.read_text())
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and any(alias.name == "agent" for alias in node.names)
    ]
    assert len(imports) == 1, "main.py must import `agent` exactly once"
    module = imports[0].module
    assert module is not None

    served_stem = Path(SERVED).stem
    imported_stem = module.rsplit(".", 1)[-1]
    assert served_stem == imported_stem, (
        f"SERVED={SERVED!r} does not match what main.py imports agent from ({module!r})"
    )


def test_main_exits_non_zero_on_a_failing_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one line that actually blocks a bad candidate must run and exit."""
    monkeypatch.setattr(holdout, "load", lambda path: [])
    monkeypatch.setattr(
        holdout,
        "run",
        lambda candidate, against, workers: Verdict(
            rate=0.0, games=128, low=0.0, high=0.1, passed=False
        ),
    )
    monkeypatch.setattr(sys, "argv", ["holdout", "candidate.json"])

    with pytest.raises(SystemExit) as excinfo:
        holdout.main()

    assert excinfo.value.code not in (0, None)


def test_main_exits_cleanly_on_a_passing_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mirror case: a PASS must complete without raising `SystemExit`."""
    monkeypatch.setattr(holdout, "load", lambda path: [])
    monkeypatch.setattr(
        holdout,
        "run",
        lambda candidate, against, workers: Verdict(
            rate=1.0, games=128, low=0.9, high=1.0, passed=True
        ),
    )
    monkeypatch.setattr(sys, "argv", ["holdout", "candidate.json"])

    holdout.main()
