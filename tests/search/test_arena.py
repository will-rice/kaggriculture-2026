"""Replaying a real episode's own actions must reproduce its own result."""

import json
import pickle
import zipfile
from pathlib import Path

import pytest

from kaggriculture.hybrid.config import HybridConfig, to_runtime
from kaggriculture.search import arena
from kaggriculture.search.arena import (
    HybridOpponent,
    evaluate,
    outcomes,
    play,
    summarize,
)
from kaggriculture.search.route import Route, from_episode
from tests.feature_golden_generator import rich_observation

ARCHIVE = Path("/data/kaggriculture/episodes/kaggriculture-episodes-2026-08-15.zip")


def test_arena_can_build_an_agent_from_a_picklable_hybrid_runtime() -> None:
    """Workers receive data, then construct the non-picklable closure locally."""
    opponent = HybridOpponent(to_runtime(HybridConfig.default()))

    restored = pickle.loads(pickle.dumps(opponent))
    side = arena._side(restored)

    assert restored == opponent
    assert not isinstance(side, str)
    assert side(rich_observation(), None)["farmer"]


def test_outcomes_records_normalized_margins_for_fitness_tie_breaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bank-scale differences are normalized before evolutionary tie-breaking."""
    candidate: Route = [{"marker": "candidate"}]

    monkeypatch.setattr(
        arena,
        "play",
        lambda seat_zero, seat_one, seeds, workers=None: (
            [(300, 100)] if seat_zero is candidate else [(50, 150)]
        ),
    )

    result = outcomes(candidate, {"other": "other-agent"}, (7,))

    assert result.margins == [200, 100]
    assert result.normalized_margins == pytest.approx([0.5, 0.5])


def test_non_done_status_is_a_failure_even_when_the_engine_supplies_rewards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ERROR/INVALID/forfeit can never enter fitness as an ordinary loss."""

    class Seat:
        def __init__(self, reward: int, status: str) -> None:
            self.reward = reward
            self.status = status

    class Environment:
        steps = [[Seat(0, "ERROR"), Seat(1, "DONE")]]

        def run(self, agents: object) -> None:
            return None

    monkeypatch.setattr(arena, "make", lambda *args, **kwargs: Environment())
    route: Route = [{"farmer": ["PASS"], "hands": [], "market": []}]

    with pytest.raises(RuntimeError, match="statuses=.*ERROR"):
        arena._run_banks(route, route, 11)
    with pytest.raises(RuntimeError, match="statuses=.*ERROR"):
        arena._one((route, route, 11))


def test_a_route_against_itself_scores_exactly_half() -> None:
    """Self-play pins ``evaluate``'s tie handling, but proves nothing about the swap.

    With the same route in both seats, ``play(candidate, opponent, seeds)`` and
    ``play(opponent, candidate, seeds)`` are textually identical calls: the
    banks come back equal, and ``win(a, b) + win(b, a) == 1`` for every pair
    regardless of which side is "ours". The result lands on 0.5 whether or not
    ``evaluate`` actually swaps seats -- an unswapped ``evaluate`` that called
    ``play(candidate, opponent, seeds)`` twice would pass this test too. What
    it does pin is that ties score as half a win rather than a whole win for
    one side, which self-play exercises on every game. The swap itself is
    checked separately, structurally, in
    ``test_evaluate_plays_both_seat_orderings`` below, because self-play has no
    behavioural signature for it to fail on.
    """
    route = [{"farmer": ["PASS"], "hands": [], "market": []}] * 720

    scores = evaluate(route, {"mirror": route}, seeds=[1, 2, 3, 4])

    assert scores["mirror"] == pytest.approx(0.5)


def test_evaluate_plays_both_seat_orderings(monkeypatch: pytest.MonkeyPatch) -> None:
    """The seat swap has to be pinned structurally, not by outcome.

    A self-play mirror scores 0.5 whether or not ``evaluate`` actually swaps
    seats, because the two calls become identical when the route is the same
    on both sides (see the test above). The only way to catch a missing swap
    is to intercept ``play`` and check what it was asked to run, so this test
    replaces ``play`` with a recorder and asserts both seat orderings were
    requested -- independent of what the recorder returns.
    """
    candidate: Route = [{"farmer": ["PASS"], "hands": [], "market": []}] * 720
    opponent = "src/kaggriculture/economic_policy.py"
    calls: list[tuple[object, object]] = []

    def recorder(
        seat_zero: object,
        seat_one: object,
        seeds: object,
        workers: object = None,
    ) -> list[tuple[int, int]]:
        calls.append((seat_zero, seat_one))
        return [(1, 0), (1, 0)]

    monkeypatch.setattr(arena, "play", recorder)

    arena.evaluate(candidate, {"econ": opponent}, seeds=[1, 2])

    assert calls == [(candidate, opponent), (opponent, candidate)]


def test_a_policy_can_stand_in_for_a_route() -> None:
    """An opponent may be an agent path, so the league can hold reacting play.

    A route is a recording and cannot respond to us. `economic_policy` can, and
    scoring against something that reacts is the only in-arena check on a
    candidate that has merely learned to beat frozen tapes.

    The assertion here is completion without raising: it establishes that a
    ``str`` opponent survives pickling to a worker process and that
    ``env.run`` accepts a policy file in the same seat position as a route.
    One seed against a policy is not enough games to be evidence of strength
    either way, so the resulting win rate is checked only for shape, not
    value.
    """
    route = [{"farmer": ["PASS"], "hands": [], "market": []}] * 720

    scores = evaluate(
        route, {"econ": "src/kaggriculture/economic_policy.py"}, seeds=[7]
    )

    assert set(scores) == {"econ"}


def test_outcomes_orders_games_by_member_then_seed_then_seat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``outcomes`` must return one entry per game in a fixed, documented order.

    A paired hill-climb comparison relies on index ``i`` in a candidate's
    outcomes and an incumbent's outcomes referring to the same board and seat
    ordering, which only holds if the order is deterministic: league member,
    then seed, then seat ordering. This pins that order structurally, by
    intercepting ``play`` rather than by an outcome value that could pass by
    coincidence.
    """
    candidate: Route = [{"marker": "candidate"}]
    league = {"alpha": "alpha-route", "beta": "beta-route"}
    seeds = [1, 2]
    calls: list[tuple[object, object, tuple[int, ...]]] = []

    def fake_play(
        seat_zero: object,
        seat_one: object,
        seeds: list[int],
        workers: int | None = None,
    ) -> list[tuple[int, int]]:
        calls.append((seat_zero, seat_one, tuple(seeds)))
        # seat_zero always banks 1, seat_one always banks 0.
        return [(1, 0) for _ in seeds]

    monkeypatch.setattr(arena, "play", fake_play)

    result = outcomes(candidate, league, seeds)

    assert calls == [
        (candidate, "alpha-route", (1, 2)),
        ("alpha-route", candidate, (1, 2)),
        (candidate, "beta-route", (1, 2)),
        ("beta-route", candidate, (1, 2)),
    ]
    # Per member: seed 1 candidate-first win (1.0), seed 1 candidate-second
    # loss (0.0), seed 2 candidate-first win (1.0), seed 2 candidate-second
    # loss (0.0) -- seed before seat ordering, member before seed.
    assert result == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]


def test_summarize_groups_outcomes_back_into_per_member_win_rates() -> None:
    """``summarize`` must invert ``outcomes``'s own chunking exactly."""
    league = {"alpha": "alpha-route", "beta": "beta-route"}
    seeds = [1, 2]
    scores = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]

    grouped = summarize(scores, league, seeds)

    assert grouped == {"alpha": 0.5, "beta": 1.0}


def test_evaluate_is_summarize_of_outcomes(monkeypatch: pytest.MonkeyPatch) -> None:
    """``evaluate`` must be one scoring rule, not a second implementation of it."""
    candidate: Route = [{"marker": "candidate"}]
    league = {"alpha": "alpha-route"}
    seeds = [1, 2, 3]

    def fake_play(
        seat_zero: object,
        seat_one: object,
        seeds: list[int],
        workers: int | None = None,
    ) -> list[tuple[int, int]]:
        return [(1, 1) for _ in seeds]

    monkeypatch.setattr(arena, "play", fake_play)

    assert evaluate(candidate, league, seeds) == summarize(
        outcomes(candidate, league, seeds), league, seeds
    )


@pytest.mark.skipif(not ARCHIVE.exists(), reason="replay corpus not on this machine")
def test_replaying_an_episode_reproduces_its_recorded_banks() -> None:
    """The engine is deterministic given seed and actions, so this is exact.

    This is the whole warrant for the arena. It covers the action grammar, seat
    assignment, market queue-position coupling and turn alignment in one
    assertion, and it is the difference between an arena that is faithful and
    one that is merely plausible. Do not weaken it to a tolerance.
    """
    with zipfile.ZipFile(ARCHIVE) as bundle:
        episode = json.loads(bundle.read("93454366.json"))
    seed = int(episode["info"]["seed"])
    expected = [(int(episode["rewards"][0]), int(episode["rewards"][1]))]

    banks = play(from_episode(episode, seat=0), from_episode(episode, seat=1), [seed])

    assert banks == expected
