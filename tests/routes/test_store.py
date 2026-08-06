"""Tests for the prototype store: harvesting, deduping and persisting routes.

Fixtures build observations the same way ``tests/routes/test_signature.py``
and ``tests/learn/test_dataset.py`` do -- through the engine's own
constructors where one exists (``engine._new_private()``) -- so a test cannot
quietly agree with code that reads a key the real game does not write.
"""

import itertools
import json
import random
import zipfile
from pathlib import Path
from typing import Any

import pytest
from kaggle_environments.envs.kaggriculture import kaggriculture as engine

from kaggriculture.constants import BOARD_SIZE, PRODUCTS
from kaggriculture.learn.corpus import CORPUS, Sample
from kaggriculture.routes.scripts import harvest as harvest_module
from kaggriculture.routes.scripts.harvest import harvest
from kaggriculture.routes.signature import signature
from kaggriculture.routes.store import Prototype, dedupe, load, save

ARCHIVE = CORPUS / "kaggriculture-episodes-2026-08-03.zip"

_needs_corpus = pytest.mark.skipif(
    not ARCHIVE.exists(), reason="replay corpus not present on this machine"
)

_ACTION: dict[str, Any] = {"farmer": ["PASS"], "hands": [], "market": []}

_episode_ids = itertools.count()


@pytest.fixture(autouse=True)
def _synthetic_corpus(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Point the harvest script's ``CORPUS`` at a scratch directory.

    ``_sample`` writes its synthetic episodes here instead of ever touching the
    real corpus, so harvest tests run on every machine, not only one with
    ``/data/kaggriculture/episodes`` mounted. Tests marked ``slow`` read the
    real corpus directly and must keep the real ``CORPUS``, so they opt out.
    """
    if request.node.get_closest_marker("slow") is None:
        monkeypatch.setattr(harvest_module, "CORPUS", tmp_path)


def _empty_farm(money: float = 3_000.0) -> dict:
    """Return a fresh, independent farm dict for a synthetic observation."""
    return {
        "tiles": [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)],
        "money": money,
        "farmer": [4, 4],
        "hands": [],
        "unlocked_quadrants": ["NW"],
        "hires_today": 0,
    }


def _observation(seat_money: float, opponent_money: float = 3_000.0) -> dict:
    """Return a minimal well-formed observation for a synthetic episode.

    ``private`` comes from the engine's own ``_new_private`` so the shed is
    dense -- every product keyed at zero -- and a signature that reaches for a
    missing key would fail here rather than only on real corpus data.
    """
    return {
        "day": 0,
        "hour": 0,
        "farms": [_empty_farm(seat_money), _empty_farm(opponent_money)],
        "market": {
            "prices": dict.fromkeys(PRODUCTS, 100),
            "inventory": dict.fromkeys(PRODUCTS, 10000),
        },
        "town": {"unlocked_shops": []},
        "private": engine._new_private(),
    }


def _sample(
    *, bank: float, opponent_bank: float = 60_000.0, rating: float = 2600.0
) -> Sample:
    """Write a synthetic two-step episode banking ``bank`` for seat 0.

    Only the final observation's money matters to ``harvest``'s floor check,
    so the episode is the shortest one that has a "before" and an "after":
    one starting step and one terminal step. Appended to a shared
    ``synthetic.zip`` in the harvest script's ``CORPUS`` -- as real archives do, one zip
    holds every episode a test needs -- under a fresh member name each call so
    two samples in one test never collide.
    """
    final = _observation(bank, opponent_bank)
    steps = [
        [{"action": dict(_ACTION), "observation": _observation(3_000.0)}, {}],
        [{"action": dict(_ACTION), "observation": final}, {}],
    ]
    name = f"{next(_episode_ids)}.json"
    with zipfile.ZipFile(harvest_module.CORPUS / "synthetic.zip", "a") as bundle:
        bundle.writestr(name, json.dumps({"steps": steps}))
    return Sample(archive="synthetic.zip", name=name, seat=0, rating=rating)


def _prototype(
    *, turns: int = 719, bank: float = 150_000.0, seed: int = 0
) -> Prototype:
    """Build a Prototype directly, bypassing harvest, for store-level tests.

    ``seed`` perturbs seat 0's farm money and so its signature: two
    prototypes built with the same seed carry byte-identical trajectories,
    and two built with different seeds do not, which is exactly what the
    dedupe test needs to tell a real duplicate from a real difference.
    """
    turn_signature = signature(_observation(3_000.0 + seed), 0)
    return Prototype(
        bank=bank,
        opponent_bank=bank - 10_000.0,
        rating=2600.0,
        actions=[dict(_ACTION) for _ in range(turns)],
        signatures=[turn_signature] * turns,
    )


def _chain_prototype(step: int, bank: float = 150_000.0) -> Prototype:
    """Return a one-turn Prototype with ``step`` bare coops on the board.

    Built with the engine's own bare-structure shape (``{"kind": "COOP"}``,
    the same shape ``test_signature.py``'s
    ``test_the_signature_counts_bare_structures`` uses) so consecutive
    ``step`` values differ by exactly one composition-field count. At day 0
    that gives a fixed, known signature distance between neighbours -- the
    shape needed to build a chain where each route is within tolerance of its
    immediate neighbours only, which is what exposes order-dependent
    clustering.
    """
    observation = _observation(3_000.0)
    for index in range(step):
        row, column = divmod(index, BOARD_SIZE)
        observation["farms"][0]["tiles"][row][column] = {"kind": "COOP"}
    return Prototype(
        bank=bank,
        opponent_bank=bank - 10_000.0,
        rating=2600.0,
        actions=[dict(_ACTION)],
        signatures=[signature(observation, 0)],
    )


def test_a_prototype_keeps_the_action_for_every_acting_turn() -> None:
    """A route is the sequence; a gap in it is a turn the agent cannot replay."""
    prototype = _prototype(turns=719)

    assert len(prototype.actions) == 719
    assert len(prototype.signatures) == 719


def test_harvest_keeps_only_routes_above_the_bank_floor() -> None:
    """The corpus median is 125,773 and our current agent banks ~118,000.

    Replaying a median route would be no better than what we already ship, so
    the floor is the point of the exercise, not a detail.
    """
    kept = harvest([_sample(bank=150_000), _sample(bank=90_000)], floor=149_120)

    assert [p.bank for p in kept] == [150_000]


def test_harvest_records_the_opponents_bank_too() -> None:
    """A route banked against a weak opponent is not the achievement it looks like.

    Task 5 needs both numbers to state which selection criterion it used, so
    ``opponent_bank`` has to actually reach the stored prototype, not just
    exist on the model.
    """
    [prototype] = harvest([_sample(bank=150_000, opponent_bank=70_000)], floor=0.0)

    assert prototype.bank == 150_000
    assert prototype.opponent_bank == 70_000


def test_dedupe_collapses_near_identical_routes_keeping_the_richest() -> None:
    """The corpus is dominated by a few public kernels.

    A thousand copies of one route is a thousand times the storage and none of
    the coverage. When two routes are within tolerance the richer one survives,
    so the store trends toward the best exemplar of each strategy.
    """
    twin_a = _prototype(bank=150_000, seed=1)
    twin_b = _prototype(bank=155_000, seed=1)
    other = _prototype(bank=151_000, seed=99)

    kept = dedupe([twin_a, twin_b, other], tolerance=1e-6)

    assert sorted(p.bank for p in kept) == [151_000, 155_000]


def test_dedupe_is_stable_under_reordering_a_tied_chain() -> None:
    """A store whose size shifts on presentation order alone is not reproducible.

    Twelve routes share one bank and form a chain: each is within tolerance
    of its immediate neighbours only, so a naive single-linkage traversal's
    survivor count depends on which order it visits them in. Sorting by
    ``bank`` alone cannot break the tie here -- every bank is identical -- so
    the canonical order also has to fall back to something that never depends
    on how the caller presented the input, which is what lets Task 3 rebuild
    the store nightly and get the same answer.
    """
    chain = [_chain_prototype(step) for step in range(12)]
    tolerance = 0.007

    canonical = {tuple(p.signatures[0]) for p in dedupe(chain, tolerance)}

    shuffled = list(chain)
    random.Random(0).shuffle(shuffled)
    reordered = {tuple(p.signatures[0]) for p in dedupe(shuffled, tolerance)}

    assert reordered == canonical


def test_a_saved_store_round_trips(tmp_path: Path) -> None:
    """The store ships inside the submission archive and is read there."""
    original = [_prototype(bank=150_000)]
    path = tmp_path / "prototypes.json.gz"

    save(original, path)
    loaded = load(path)

    assert loaded[0].actions == original[0].actions
    assert loaded[0].signatures == original[0].signatures


@pytest.mark.slow
@_needs_corpus
def test_the_corpus_s_shared_opening_would_flood_an_undeduped_store() -> None:
    """The evidence that motivates dedupe: one opening, replayed everywhere.

    Ten real episodes from one archive, compared at their day-1 signature,
    turn out to be pairwise identical -- the same public kernel playing the
    same opening. Harvesting them without dedupe would store ten near-copies
    of one route.
    """
    with zipfile.ZipFile(ARCHIVE) as bundle:
        names = sorted(n for n in bundle.namelist() if n.endswith(".json"))[:10]
        day_one = []
        for name in names:
            steps = json.load(bundle.open(name))["steps"]
            day_one.append(signature(steps[24][0]["observation"], 0))

    identical_pairs = sum(1 for a, b in itertools.combinations(day_one, 2) if a == b)

    assert identical_pairs == len(list(itertools.combinations(day_one, 2)))


@pytest.mark.slow
@_needs_corpus
def test_dedupe_still_keeps_distinct_late_game_routes_from_a_shared_opening() -> None:
    """Sharing an opening is not the same as sharing a route.

    The same ten episodes above diverge heavily after day 1 -- different
    weeds, different opponents, different market noise -- so a whole-route
    dedupe must not collapse them into one entry just because they started
    the same way. Collapsing on the opening alone would erase real late-game
    diversity the store exists to keep.
    """
    with zipfile.ZipFile(ARCHIVE) as bundle:
        names = sorted(n for n in bundle.namelist() if n.endswith(".json"))[:10]
    samples = [
        Sample(archive=ARCHIVE.name, name=name, seat=0, rating=2600.0) for name in names
    ]

    harvested = harvest(samples, floor=0.0)
    deduped = dedupe(harvested, tolerance=1e-6)

    assert len(deduped) == len(harvested) == 10
