"""Tests for the identity-free state signature and its distance.

A signature that leaks the opponent's identity would tie every recorded route
to the matchup it was harvested from, so these check both that the signature
truly ignores the opponent and that it moves with our own board. Fixtures are
built the same way `tests/learn/test_encoding.py` builds them -- through the
engine's own constructors where one exists -- so a test cannot quietly agree
with code that reads a key the real game does not write.
"""

import json
import zipfile

import pytest
from kaggle_environments.envs.kaggriculture import kaggriculture as engine

from kaggriculture.constants import BOARD_SIZE, PRODUCTS, TURNS_PER_DAY
from kaggriculture.learn.corpus import CORPUS
from kaggriculture.routes.signature import distance, field_contributions, signature

ARCHIVE = CORPUS / "kaggriculture-episodes-2026-08-03.zip"

_needs_corpus = pytest.mark.skipif(
    not ARCHIVE.exists(), reason="replay corpus not present on this machine"
)


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


def test_the_signature_counts_bare_structures() -> None:
    """A built coop or pasture is real board state even with nothing on it.

    Built the same shape the engine's own ``BUILD_COOP``/``BUILD_PASTURE`` ops
    write -- ``farm["tiles"][fy][fx] = {"kind": "COOP"}``, read straight out of
    the engine's ``_apply_unit_action`` -- rather than guessed, so this test
    cannot quietly agree with a signature that reads a key the engine never
    writes. Before this field existed, a farm with three bare coops looked
    identical to one with bare ground.
    """
    before = empty_observation()
    after = empty_observation()
    after["farms"][0]["tiles"][2][3] = {"kind": "COOP"}

    assert signature(before, 0) != signature(after, 0)


@pytest.mark.slow
@_needs_corpus
def test_no_single_field_dominates_a_real_distance() -> None:
    """Money must not be the only field a real comparison responds to.

    An earlier version of this module returned raw, unscaled fields and let
    `distance` weight only by phase-vs-composition group. On this exact pair
    of real episodes at day 16, that let `MONEY` alone account for 99.96% of
    the total distance -- retrieval built on that key would pick the
    prototype with the nearest bank balance and ignore everything else about
    the board, which is close to the worst possible matching signal: bank is
    an outcome of a route, not a description of the state a route should be
    selected for. This pins the fix down: after scaling every field into a
    comparable range, no single field may account for more than about half of
    a real comparison.

    The two episodes are the archive's first two entries in sorted name
    order -- a fixed, deterministic, cheap read, not hand-picked to make the
    assertion pass.
    """
    with zipfile.ZipFile(ARCHIVE) as bundle:
        names = sorted(name for name in bundle.namelist() if name.endswith(".json"))
        episode_a = json.load(bundle.open(names[0]))
        episode_b = json.load(bundle.open(names[1]))

    day = 16
    index = day * TURNS_PER_DAY
    observation_a = episode_a["steps"][index][0]["observation"]
    observation_b = episode_b["steps"][index][0]["observation"]

    contributions = field_contributions(
        signature(observation_a, 0), signature(observation_b, 0), day
    )
    total = sum(contributions)

    assert total > 0.0
    assert max(contributions) <= 0.5 * total
