"""Tests for the identity-free state signature and its distance.

A signature that leaks the opponent's identity would tie every recorded route
to the matchup it was harvested from, so these check both that the signature
truly ignores the opponent and that it moves with our own board. Fixtures are
built the same way `tests/learn/test_encoding.py` builds them -- through the
engine's own constructors where one exists -- so a test cannot quietly agree
with code that reads a key the real game does not write.
"""

from kaggle_environments.envs.kaggriculture import kaggriculture as engine

from kaggriculture.constants import BOARD_SIZE, PRODUCTS, TURNS_PER_DAY
from kaggriculture.routes.signature import distance, signature


def _empty_farm() -> dict:
    """Return a fresh, independent farm with an empty unlocked board.

    Each call builds its own tile grid: two farms in one observation must
    never share one mutable list, or mutating one player's board silently
    mutates the other's too.
    """
    return {
        "tiles": [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)],
        "money": 3000.0,
        "farmer": [4, 4],
        "hands": [],
        "unlocked_quadrants": ["NW"],
        "hires_today": 0,
    }


def empty_observation() -> dict:
    """Return a minimal well-formed observation with an empty unlocked board.

    ``private`` comes from the engine's own ``_new_private`` rather than being
    typed out here, so the shed is dense -- every product keyed at zero -- and
    a signature that reaches for a missing key would pass here and raise on
    the first real observation.
    """
    return {
        "day": 0,
        "hour": 0,
        "farms": [_empty_farm(), _empty_farm()],
        "market": {
            "prices": dict.fromkeys(PRODUCTS, 100),
            "inventory": dict.fromkeys(PRODUCTS, 10000),
        },
        "town": {"unlocked_shops": []},
        "private": engine._new_private(),
    }


def _plant(crop: str = "WHEAT", day: int = 0) -> dict:
    """Return a freshly planted tile, built by the engine rather than by hand."""
    return engine._new_plant(crop, day, TURNS_PER_DAY)


def test_the_signature_ignores_who_the_opponent_is() -> None:
    """A prototype must transfer across matchups or the store is per-opponent.

    Nothing identifying the opponent may enter the signature -- only the public
    shape of the board. Two observations differing solely in the opponent's
    identity fields must produce the same signature.
    """
    ours = empty_observation()
    theirs = empty_observation()
    theirs["farms"][1]["money"] = 99_999.0

    assert signature(ours, 0) == signature(theirs, 0)


def test_the_signature_moves_with_our_own_board() -> None:
    """It is a description of our farm; if our farm changes it must change."""
    before = empty_observation()
    after = empty_observation()
    after["farms"][0]["tiles"][2][3] = _plant("MELON")

    assert signature(before, 0) != signature(after, 0)


def test_distance_is_phase_dominated_early_and_board_dominated_late() -> None:
    """Routes are phase-locked early and diverge in composition later.

    On day 0 an hour apart matters more than a crop apart, because every good
    route is doing the same thing at the same time. By day 25 the reverse holds.
    """
    base = empty_observation()
    an_hour = empty_observation()
    an_hour["hour"] = 3
    a_crop = empty_observation()
    a_crop["farms"][0]["tiles"][2][3] = _plant("MELON")

    early_hour = distance(signature(base, 0), signature(an_hour, 0), day=0)
    early_crop = distance(signature(base, 0), signature(a_crop, 0), day=0)
    late_hour = distance(signature(base, 0), signature(an_hour, 0), day=25)
    late_crop = distance(signature(base, 0), signature(a_crop, 0), day=25)

    assert early_hour > early_crop
    assert late_crop > late_hour


def test_distance_to_itself_is_zero() -> None:
    """A prototype must match its own recorded state exactly."""
    one = signature(empty_observation(), 0)

    assert distance(one, one, day=10) == 0.0
