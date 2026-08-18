"""Replaying a real episode's own actions must reproduce its own result."""

import json
import zipfile
from pathlib import Path

import pytest

from kaggriculture.search import arena
from kaggriculture.search.arena import evaluate, play
from kaggriculture.search.route import Route, from_episode

ARCHIVE = Path("/data/kaggriculture/episodes/kaggriculture-episodes-2026-08-15.zip")


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
