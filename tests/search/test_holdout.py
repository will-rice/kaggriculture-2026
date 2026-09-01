"""The held-out exam a searched route must pass before it ships."""

import ast
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from kaggriculture.search import arena
from kaggriculture.search.scripts import hillclimb, holdout
from kaggriculture.search.scripts.holdout import GATE_SEEDS, SERVED, Verdict, run

MAIN = Path("main.py")


def _frontier_dir(tmp_path: Path, count: int = 3) -> Path:
    """Build a tmp frontier directory with ``count`` fresh route files.

    File content is irrelevant in every test that uses this helper: they all
    monkeypatch `holdout.load`, so `_frontier_league` never actually parses
    these files -- only lists them and reads their mtimes.
    """
    frontier = tmp_path / "frontier"
    frontier.mkdir()
    for index in range(count):
        (frontier / f"route-{index}.json").write_text("[]")
    return frontier


def test_run_delegates_the_interval_to_report_wilson_interval(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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
        league: Mapping[str, object],
        seeds: object,
        workers: object = None,
    ) -> list[float]:
        return [0.0] * 128

    def wilson_recorder(wins: float, games: int) -> tuple[float, float]:
        return (0.9, 0.95)

    monkeypatch.setattr(arena, "outcomes", outcomes_recorder)
    monkeypatch.setattr(holdout, "wilson_interval", wilson_recorder)
    monkeypatch.setattr(holdout, "load", lambda path: [])

    verdict = run(candidate=[], against=SERVED, frontier=_frontier_dir(tmp_path))

    assert (verdict.low, verdict.high) == (0.9, 0.95)
    assert verdict.frontier_low == 0.9
    assert verdict.passed


def test_a_rate_above_half_with_an_interval_straddling_half_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A rate over 50% is not sufficient; the whole interval must clear 0.5.

    65 of 128 is a rate of 0.508 -- above half -- but its lower bound (0.422)
    sits well below 0.5. That gap is the entire reason the served-agent leg of
    this gate exists rather than a bare `rate > 0.5` check: at this sample
    size a small edge is not distinguishable from a coin flip that happened to
    land favourably, which is exactly the case this test pins. The recorder
    ignores which league it is scoring, so the frontier leg lands on the same
    numbers and would pass its own bar on its own -- the served-agent
    condition alone is enough to fail the verdict, which is what this test
    checks.
    """

    def recorder(
        candidate: object,
        league: Mapping[str, object],
        seeds: object,
        workers: object = None,
    ) -> list[float]:
        return [1.0] * 65 + [0.0] * 63

    monkeypatch.setattr(arena, "outcomes", recorder)
    monkeypatch.setattr(holdout, "load", lambda path: [])

    verdict = run(candidate=[], against=SERVED, frontier=_frontier_dir(tmp_path))

    assert verdict.rate > 0.5
    assert verdict.low < 0.5
    assert not verdict.passed


def test_a_rate_whose_interval_clears_half_passes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The mirror case: both conditions clearing their bars must PASS.

    The recorder ignores which league it is scoring, so 76 of 128 (a Wilson
    lower bound of ~0.508) stands in for both the served-agent leg and the
    pooled frontier leg -- comfortably past 0.5 and past `FRONTIER_LOW_THRESHOLD`
    (0.45) respectively.
    """

    def recorder(
        candidate: object,
        league: Mapping[str, object],
        seeds: object,
        workers: object = None,
    ) -> list[float]:
        return [1.0] * 76 + [0.0] * 52

    monkeypatch.setattr(arena, "outcomes", recorder)
    monkeypatch.setattr(holdout, "load", lambda path: [])

    verdict = run(candidate=[], against=SERVED, frontier=_frontier_dir(tmp_path))

    assert verdict.low > 0.5
    assert verdict.frontier_low >= holdout.FRONTIER_LOW_THRESHOLD
    assert verdict.passed


def test_a_candidate_that_beats_served_but_fails_the_frontier_pool_fails_2026_08_20(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pins the 2026-08-20 failure: the old, single-condition gate is not enough.

    The searched route that shipped on 2026-08-20 scored 0.734 [0.652, 0.803]
    over 128 games against the agent we served -- comfortably clearing the
    served-agent condition alone -- and still converged ~130 rating points
    below the agent it replaced, because it lost to the current meta roughly
    4-to-1 above 1400. Its pooled frontier rate was 0.36-0.38. This test
    reproduces both numbers (94/128 served, 142/384 frontier) and asserts the
    verdict is FAIL: a gate that let this candidate through despite the
    frontier leg would be the exact defect this task exists to close.
    """
    frontier = _frontier_dir(tmp_path)

    def recorder(
        candidate: object,
        league: Mapping[str, object],
        seeds: object,
        workers: object = None,
    ) -> list[float]:
        if "served" in league:
            return [1.0] * 94 + [0.0] * 34
        return [1.0] * 142 + [0.0] * 242

    monkeypatch.setattr(arena, "outcomes", recorder)
    monkeypatch.setattr(holdout, "load", lambda path: [])

    verdict = run(candidate=[], against=SERVED, frontier=frontier)

    assert verdict.low > 0.5, (
        "the served-agent leg must clear on its own, as it did in reality"
    )
    assert verdict.frontier_rate == pytest.approx(142 / 384)
    assert verdict.frontier_low < holdout.FRONTIER_LOW_THRESHOLD
    assert not verdict.passed


def test_a_true_frontier_peer_passes_both_conditions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The mirror of the 2026-08-20 pin: also holding the frontier must PASS.

    94/128 served (the same served-leg numbers as the 2026-08-20 route) paired
    with 200/384 pooled-frontier (rate 0.521, Wilson low ~0.471) is a
    candidate that is both better than what we serve and statistically not
    worse than the frontier -- the two-condition bar this gate now enforces.
    """
    frontier = _frontier_dir(tmp_path)

    def recorder(
        candidate: object,
        league: Mapping[str, object],
        seeds: object,
        workers: object = None,
    ) -> list[float]:
        if "served" in league:
            return [1.0] * 94 + [0.0] * 34
        return [1.0] * 200 + [0.0] * 184

    monkeypatch.setattr(arena, "outcomes", recorder)
    monkeypatch.setattr(holdout, "load", lambda path: [])

    verdict = run(candidate=[], against=SERVED, frontier=frontier)

    assert verdict.low > 0.5
    assert verdict.frontier_low >= holdout.FRONTIER_LOW_THRESHOLD
    assert verdict.passed


def test_run_raises_if_the_frontier_directory_is_empty(tmp_path: Path) -> None:
    """An empty frontier cannot be examined against -- fail loudly, not quietly."""
    empty = tmp_path / "frontier"
    empty.mkdir()

    with pytest.raises(RuntimeError, match="empty"):
        run(candidate=[], against=SERVED, frontier=empty)


def test_run_raises_if_the_frontier_pool_has_fewer_than_the_minimum_tapes(
    tmp_path: Path,
) -> None:
    """Below `FRONTIER_MIN_TAPES` the 0.45 bar is unreachable by any candidate.

    Two tapes pool to 256 games; even a perfect 0.500 rate over 256 games
    caps the Wilson lower bound at ~0.439, below `FRONTIER_LOW_THRESHOLD`
    (0.45) no matter how strong the candidate is. Silently grading against a
    pool this size would produce a FAIL that reads as a weak candidate when
    the real cause is a pool too thin for the bar to mean anything, so the
    gate must raise instead -- and the message must name both the count found
    and the count required so the cause is legible.
    """
    thin = _frontier_dir(tmp_path, count=2)

    with pytest.raises(RuntimeError) as excinfo:
        run(candidate=[], against=SERVED, frontier=thin)

    message = str(excinfo.value)
    assert "2" in message
    assert str(holdout.FRONTIER_MIN_TAPES) in message


def test_run_raises_if_the_frontier_s_newest_file_is_older_than_seven_days(
    tmp_path: Path,
) -> None:
    """A stale frontier is the 2026-08-20 failure with extra steps.

    A frontier whose newest file predates the 7-day window means the operator
    has not refreshed it since `build_league` last ran -- examining a
    candidate against it would silently repeat the exact failure this task
    exists to close, just with a frontier leg that looks present but is not
    current. The gate must raise rather than quietly grading against history.
    Built at `FRONTIER_MIN_TAPES` files (all stale) so this pins the
    staleness raise specifically, not the too-few-tapes one -- a pool this
    size is otherwise large enough to pass.
    """
    stale_dir = _frontier_dir(tmp_path, count=holdout.FRONTIER_MIN_TAPES)
    eight_days_ago = time.time() - 8 * 86400
    for stale_file in stale_dir.iterdir():
        os.utime(stale_file, (eight_days_ago, eight_days_ago))

    with pytest.raises(RuntimeError, match="days old"):
        run(candidate=[], against=SERVED, frontier=stale_dir)


def test_run_does_not_raise_when_the_newest_frontier_file_is_within_seven_days(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The boundary's mirror: a frontier refreshed recently must not raise.

    Built at `FRONTIER_MIN_TAPES` files so the too-few-tapes check does not
    also fire here.
    """
    frontier = _frontier_dir(tmp_path, count=holdout.FRONTIER_MIN_TAPES)
    six_days_ago = time.time() - 6 * 86400
    for fresh_file in frontier.iterdir():
        os.utime(fresh_file, (six_days_ago, six_days_ago))

    def recorder(
        candidate: object,
        league: Mapping[str, object],
        seeds: Sequence[int],
        workers: object = None,
    ) -> list[float]:
        return [1.0] * (2 * len(seeds) * len(league))

    monkeypatch.setattr(arena, "outcomes", recorder)
    monkeypatch.setattr(holdout, "load", lambda path: [])

    run(candidate=[], against=SERVED, frontier=frontier)


def test_run_calls_outcomes_with_the_held_out_seeds_and_served_agent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The exam must use `GATE_SEEDS` and the served-agent path -- structurally.

    A rate-only assertion cannot tell a gate that plays the right seeds from
    one that plays some other 128 games and happens to land on a similar
    number. Intercepting `arena.outcomes` and inspecting what it was called
    with is the only way to pin that the exam actually reads from
    `GATE_SEEDS`, which is the property the whole "held-out" claim rests on.
    The served-agent leg must be played first, before the frontier leg.
    """
    calls = []

    def recorder(
        candidate: object,
        league: Mapping[str, object],
        seeds: Sequence[int],
        workers: object = None,
    ) -> list[float]:
        calls.append((candidate, league, seeds, workers))
        return [1.0] * (2 * len(seeds) * len(league))

    monkeypatch.setattr(arena, "outcomes", recorder)
    monkeypatch.setattr(holdout, "load", lambda path: [])
    candidate = [{"marker": "candidate"}]

    holdout.run(candidate, frontier=_frontier_dir(tmp_path), workers=4)

    assert calls[0] == (candidate, {"served": SERVED}, GATE_SEEDS, 4)


def test_run_calls_outcomes_for_the_frontier_pool_with_the_held_out_seeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The frontier leg must be scored over `GATE_SEEDS` too, keyed by file stem."""
    calls = []

    def recorder(
        candidate: object,
        league: Mapping[str, object],
        seeds: Sequence[int],
        workers: object = None,
    ) -> list[float]:
        calls.append((league, seeds))
        return [1.0] * (2 * len(seeds) * len(league))

    monkeypatch.setattr(arena, "outcomes", recorder)
    monkeypatch.setattr(holdout, "load", lambda path: ["route-marker"])
    frontier = _frontier_dir(tmp_path, count=3)

    holdout.run([], frontier=frontier)

    frontier_league, frontier_seeds = calls[1]
    assert frontier_seeds == GATE_SEEDS
    assert set(frontier_league.keys()) == {"route-0", "route-1", "route-2"}
    assert all(route == ["route-marker"] for route in frontier_league.values())


def test_run_passes_a_non_default_opponent_through_to_outcomes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--against` must override which agent the league keys to `served`."""
    calls = []

    def recorder(
        candidate: object,
        league: Mapping[str, object],
        seeds: Sequence[int],
        workers: object = None,
    ) -> list[float]:
        calls.append(league)
        return [1.0] * (2 * len(seeds) * len(league))

    monkeypatch.setattr(arena, "outcomes", recorder)
    monkeypatch.setattr(holdout, "load", lambda path: [])

    holdout.run(
        [],
        against="src/kaggriculture/kaito_v23_policy.py",
        frontier=_frontier_dir(tmp_path),
    )

    assert calls[0] == {"served": "src/kaggriculture/kaito_v23_policy.py"}


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

    Matched on the *bound* name rather than the imported one, because the served
    kernel's entry point is not called `agent` -- it defines `agent` twice and
    names its final callable something else, so `main.py` imports that under an
    alias. What has to be one is the number of imports that produce the `agent`
    this file serves, which is what an alias-aware match counts.
    """
    tree = ast.parse(MAIN.read_text())
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and any((alias.asname or alias.name) == "agent" for alias in node.names)
    ]
    assert len(imports) == 1, "main.py must bind `agent` from exactly one import"
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
        lambda candidate, against, frontier, workers: Verdict(
            rate=0.0,
            games=128,
            low=0.0,
            high=0.1,
            frontier_rate=0.0,
            frontier_low=0.0,
            passed=False,
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
        lambda candidate, against, frontier, workers: Verdict(
            rate=1.0,
            games=128,
            low=0.9,
            high=1.0,
            frontier_rate=0.9,
            frontier_low=0.85,
            passed=True,
        ),
    )
    monkeypatch.setattr(sys, "argv", ["holdout", "candidate.json"])

    holdout.main()
