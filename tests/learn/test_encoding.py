"""Tests for turning an observation into tensors.

A wrong plane never raises. It trains the model on a game that is not this one,
and the first symptom is an agent that plays badly for reasons nobody can find.
So these check the encoding against the engine's own rules tables and against a
board built by hand.
"""

import json
import zipfile

import pytest
import torch

from kaggriculture.constants import ANIMALS, CROPS, PRODUCTS
from kaggriculture.learn.corpus import CORPUS
from kaggriculture.learn.encoding import (
    BOARD,
    HIRE_SLOT,
    IGNORE,
    LAND_SLOT,
    MARKET_SLOTS,
    MAX_ORDERS,
    MAX_UNITS,
    QUANTITIES,
    SCALARS,
    TILE_PLANES,
    bucket_of,
    decode_market,
    encode_board,
    encode_market,
    encode_positions,
    encode_scalars,
    encode_units,
    quantity_of,
    unit_count,
)

ARCHIVE = CORPUS / "kaggriculture-episodes-2026-08-03.zip"

_needs_corpus = pytest.mark.skipif(
    not ARCHIVE.exists(), reason="replay corpus not present on this machine"
)


def _empty_farm() -> dict:
    """Return a fresh, independent farm with an empty unlocked board.

    Each call builds its own tile grid: the two farms in an observation must
    never share one mutable list, or mutating one player's board silently
    mutates the other's too.
    """
    return {
        "tiles": [[None] * BOARD for _ in range(BOARD)],
        "money": 3000.0,
        "farmer": [4, 4],
        "hands": [],
        "unlocked_quadrants": ["NW"],
        "hires_today": 0,
    }


def empty_observation(seat: int = 0) -> dict:
    """Return a minimal well-formed observation with an empty unlocked board."""
    return {
        "player": seat,
        "step": 0,
        "day": 0,
        "hour": 0,
        "farms": [_empty_farm(), _empty_farm()],
        "market": {
            "prices": dict.fromkeys(PRODUCTS, 100),
            "inventory": dict.fromkeys(PRODUCTS, 10000),
        },
        "town": {"unlocked_shops": []},
        "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
    }


def test_board_has_the_declared_shape_and_batch_dimension() -> None:
    """Every tensor in this project carries its batch dimension."""
    board = encode_board(empty_observation(), seat=0)

    assert board.shape == (1, TILE_PLANES, BOARD, BOARD)
    assert board.dtype == torch.float32


def test_a_planted_tile_lights_exactly_its_own_crop_plane() -> None:
    """Crop identity is one-hot; two crops lit at once would be a silent mixture."""
    observation = empty_observation()
    observation["farms"][0]["tiles"][2][3] = {
        "kind": "PLANT",
        "crop": "MELON",
        "planted_day": 0,
        "watered_today": False,
        "consecutive_unwatered": 0,
        "yield_units": 0,
        "fertilized_until_day": -1,
    }

    board = encode_board(observation, seat=0)
    crops = sorted(CROPS)
    lit = [board[0, index, 2, 3].item() for index in range(len(crops))]

    assert sum(lit) == 1.0
    assert lit[crops.index("MELON")] == 1.0


def test_the_opponent_s_board_occupies_its_own_planes() -> None:
    """Opponent supply decides prices here; their board must be visible and distinct."""
    observation = empty_observation()
    observation["farms"][1]["tiles"][5][5] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 0,
        "watered_today": False,
        "consecutive_unwatered": 0,
        "yield_units": 0,
        "fertilized_until_day": -1,
    }

    ours = encode_board(observation, seat=0)
    theirs = encode_board(observation, seat=1)

    assert not torch.equal(ours, theirs)


def _changed_plane(before: torch.Tensor, after: torch.Tensor) -> int:
    """Return the index of the one plane that differs between two encoded boards.

    Finding the plane rather than naming it keeps these tests off the plane
    layout: the occupancy planes' indices are derived from the engine's rules
    tables and would shift under any upstream crop or animal.
    """
    changed = (before != after).flatten(2).any(dim=2)[0].nonzero().flatten()
    assert changed.numel() == 1, f"expected one plane to move, got {changed.tolist()}"
    return int(changed.item())


def test_the_board_says_where_the_farmer_stands() -> None:
    """Without this the trunk sees a board with nobody on it.

    The whole point of the spatial head is that a unit reads the trunk at its
    own tile; if no plane marks where anyone is standing, that column carries
    the tile's crops and nothing about the unit deciding from it.
    """
    here, there = empty_observation(), empty_observation()
    there["farms"][0]["farmer"] = [7, 1]

    before, after = encode_board(here, seat=0), encode_board(there, seat=0)
    plane = _changed_plane(before, after)

    assert before[0, plane, 4, 4].item() == 1.0
    assert after[0, plane, 7, 1].item() == 1.0
    assert after[0, plane, 4, 4].item() == 0.0


def test_the_hands_plane_counts_rather_than_flags() -> None:
    """Two hands may stand on one tile -- ``[[4, 3], [4, 3]]`` occurs in the corpus.

    A flag would report a crowd the same as a lone hand, so the plane carries a
    count. It is scaled by ``MAX_UNITS`` to share the range of the other
    continuous features, which this checks by ratio rather than by value.
    """
    one, two = empty_observation(), empty_observation()
    one["farms"][0]["hands"] = [[4, 3]]
    two["farms"][0]["hands"] = [[4, 3], [4, 3]]

    single, doubled = encode_board(one, seat=0), encode_board(two, seat=0)
    plane = _changed_plane(single, doubled)

    assert single[0, plane, 4, 3].item() == pytest.approx(1.0 / MAX_UNITS)
    assert doubled[0, plane, 4, 3].item() == pytest.approx(2.0 / MAX_UNITS)


def test_the_opponent_s_units_occupy_their_own_planes() -> None:
    """Their farmer and hands are public, and where they stand decides prices."""
    empty, staffed = empty_observation(), empty_observation()
    staffed["farms"][1]["hands"] = [[8, 8]]

    plane = _changed_plane(encode_board(empty, seat=0), encode_board(staffed, seat=0))

    assert plane >= TILE_PLANES // 2


def test_positions_are_ordered_exactly_as_labels_are() -> None:
    """Slot ``k``'s position and slot ``k``'s label must describe one unit.

    The head reads slot ``k``'s trunk column at slot ``k``'s position and scores
    it against slot ``k``'s label, so a divergence in order silently issues
    every unit another unit's orders.
    """
    observation = empty_observation()
    observation["farms"][0]["farmer"] = [2, 3]
    observation["farms"][0]["hands"] = [[7, 1], [0, 9]]

    positions = encode_positions(observation, seat=0)

    assert positions.shape == (1, MAX_UNITS)
    assert positions.dtype == torch.int64
    assert positions[0, 0].item() == 2 * BOARD + 3
    assert positions[0, 1].item() == 7 * BOARD + 1
    assert positions[0, 2].item() == 0 * BOARD + 9
    assert positions[0, 3:].eq(0).all()


def test_positions_are_read_for_the_seat_asked_for() -> None:
    """``seat`` selects whose units to locate, as it does everywhere else here."""
    observation = empty_observation()
    observation["farms"][1]["farmer"] = [9, 9]

    assert encode_positions(observation, seat=0)[0, 0].item() == 4 * BOARD + 4
    assert encode_positions(observation, seat=1)[0, 0].item() == 9 * BOARD + 9


def test_a_position_count_beyond_max_units_raises() -> None:
    """Truncating here would hand a real unit another unit's tile."""
    observation = empty_observation()
    observation["farms"][0]["hands"] = [[0, 0] for _ in range(MAX_UNITS)]

    with pytest.raises(ValueError):
        encode_positions(observation, seat=0)


@pytest.mark.slow
@_needs_corpus
def test_positions_and_labels_agree_on_a_real_episode() -> None:
    """The two functions are only correct relative to one another.

    Slot ``k``'s logits are read at slot ``k``'s position and scored against
    slot ``k``'s label, so a labelled slot with no unit on it, or a unit whose
    slot carries the padded tile, is a silent training error rather than a
    crash. This is the invariant that catches them drifting apart.

    It is checked against a real episode because a hand-built observation
    agrees with whatever the encoders do. The corpus does not: 2.4% of turns
    carry an action whose hand-op list disagrees with the farm's hand list,
    which is the disagreement that made ``unit_count`` the single authority.

    The rate is archive-dependent -- 0% before 2026-08-02, up to 8.1% after --
    so ``ARCHIVE`` must stay on a day that actually disagrees or the final
    assertion here silently stops testing anything.
    """
    with zipfile.ZipFile(ARCHIVE) as bundle:
        name = next(n for n in bundle.namelist() if n.endswith(".json"))
        with bundle.open(name) as member:
            steps = json.load(member)["steps"]

    most = 0
    disagreements = 0
    for seat in (0, 1):
        for index in range(len(steps) - 1):
            action = steps[index + 1][seat].get("action")
            if not action:
                continue
            observation = steps[index][seat]["observation"]
            farm = observation["farms"][seat]
            tiles = [farm["farmer"], *farm["hands"]]
            units = unit_count(observation, seat)
            labels = encode_units(action, units)
            positions = encode_positions(observation, seat)

            assert units == len(tiles)
            assert labels[0, units:].eq(IGNORE).all()
            assert positions[0, :units].tolist() == [y * BOARD + x for y, x in tiles]
            assert positions[0, units:].eq(0).all()
            most = max(most, units)
            disagreements += 1 + len(action["hands"]) != units

    assert most > 1, "no turn had a hired hand, so this proved nothing"
    assert disagreements, "this episode never disagreed, so this proved little"


def test_scalars_have_the_declared_width() -> None:
    """The market branch is fixed-width; a ragged vector would break the model."""
    scalars = encode_scalars(empty_observation(), seat=0)

    assert scalars.shape == (1, SCALARS)
    assert torch.isfinite(scalars).all()


def test_scalars_carry_both_price_and_opponent_supply() -> None:
    """Value here is a property of a product given what the opponent is selling.

    This is the finding that a static per-animal valuation cost 10k when it
    replaced the opponent-aware one. If the model cannot see the opponent's
    supply it cannot learn the thing that decides this game.
    """
    cheap = empty_observation()
    rich = empty_observation()
    rich["market"]["prices"] = dict.fromkeys(PRODUCTS, 200)

    assert not torch.equal(encode_scalars(cheap, 0), encode_scalars(rich, 0))


def test_scalars_do_not_depend_on_the_step_key() -> None:
    """kaggle_environments only ever writes ``step`` onto agent 0's observation.

    Its own ``core.py`` sets ``new_state[0].observation.step = ...`` and never
    mirrors it to any other agent, so a real seat-1 (or later) observation has
    no ``step`` key at all. ``encode_scalars`` must derive the global step from
    ``day``/``hour``, which the game's interpreter does mirror to every agent,
    rather than reading a key that is only ever present for seat 0.
    """
    from kaggriculture.constants import EPISODE_STEPS, TURNS_PER_DAY

    absent = empty_observation()
    absent["day"], absent["hour"] = 3, 7
    del absent["step"]
    misleading = empty_observation()
    misleading["day"], misleading["hour"] = 3, 7
    misleading["step"] = 999_999

    # Asserted by equality rather than by indexing one scalar. An index into
    # this vector is a positional fact about a layout that grows whenever a
    # feature is added -- it broke the moment `hires_today` was appended -- and
    # it tests where the value sits rather than that `step` is ignored, which is
    # the property that matters.
    assert torch.equal(
        encode_scalars(absent, seat=0), encode_scalars(misleading, seat=0)
    )

    expected = (3 * TURNS_PER_DAY + 7) / EPISODE_STEPS
    assert any(
        value == pytest.approx(expected) for value in encode_scalars(absent, 0)[0]
    )


def test_the_phase_of_the_season_is_encoded() -> None:
    """Shops unlock every three days and demand steps at days 10 and 20."""
    early, late = empty_observation(), empty_observation()
    late["day"], late["hour"], late["step"] = 25, 13, 25 * 24 + 13

    assert not torch.equal(encode_scalars(early, 0), encode_scalars(late, 0))


@pytest.mark.slow
@pytest.mark.skipif(not CORPUS.is_dir(), reason="needs the replay corpus")
def test_every_order_in_the_corpus_encodes() -> None:
    """A verb or item with no slot is dropped silently, not loudly.

    Checked across every archive rather than one, because the ladder's agent
    mix changes daily -- the verb mix inverts between 07-30 and 08-04, and two
    committed claims in this repo have already come from generalising a single
    archive.
    """
    seen = set()
    for archive in sorted(CORPUS.glob("*.zip")):
        with zipfile.ZipFile(archive) as bundle:
            name = next(n for n in bundle.namelist() if n.endswith(".json"))
            with bundle.open(name) as member:
                steps = json.load(member)["steps"]
        for step in steps:
            for seat in (0, 1):
                action = step[seat].get("action") or {}
                for order in action.get("market", []) or []:
                    played = order[0] if len(order) < 3 else (order[0], order[1])
                    seen.add(played)
                    # Checked here rather than only after the sweep because
                    # `encode_market` raises on an order it cannot express, and
                    # it would abort the loop before the aggregate assertions
                    # below ever ran -- leaving a bare ValueError in place of a
                    # message naming what the corpus actually played.
                    assert played in MARKET_SLOTS or played in ("HIRE", "BUY_LAND"), (
                        f"{archive.name} plays {played}, which no slot can express"
                    )
                encode_market(action)

    unknown = {s for s in seen if isinstance(s, tuple) and s not in MARKET_SLOTS}
    assert not unknown, f"corpus plays {unknown}, which no slot can express"
    assert {s for s in seen if isinstance(s, str)} <= {"HIRE", "BUY_LAND"}


def test_the_vocabulary_covers_every_op_the_engine_implements() -> None:
    """Derived from the engine's code, not its docs.

    The published spec lists seventeen unit ops and omits DROP, which the engine
    implements and this project's own agents emit routinely. A hand-written list
    checked against another hand-written list passes while both are wrong.
    """
    import inspect
    import re

    from kaggle_environments.envs.kaggriculture import kaggriculture as engine

    from kaggriculture.learn.encoding import UNIT_OPS

    source = inspect.getsource(engine._apply_unit_action)
    implemented = set(re.findall(r'op == "([A-Z_]+)"', source))

    assert implemented
    assert implemented <= {op.split(":")[0] for op in UNIT_OPS}


def test_buy_product_s_item_gate_matches_the_engine_exactly() -> None:
    """BUY_PRODUCT's gate is a closed literal, not an open-ended op list.

    ``test_the_vocabulary_covers_every_op_the_engine_implements`` checks
    ``<=`` because the unit op list only ever grows. ``_process_market``'s
    BUY_PRODUCT gate is different in kind: it is a literal
    ``item in ("WHEAT", "FERTILIZER")``, a fixed set rather than a floor. An
    item the engine accepts that MARKET_SLOTS omits silently drops an order
    the teacher could have played; an item MARKET_SLOTS claims that the
    engine rejects is a dead slot that no-ops every turn the model fills it,
    wasting a slot out of the ten-order cap. Both are defects, so this checks
    equality, not containment.

    This regex belongs here, in a test, and not in ``_BUY_PRODUCT_ITEMS``
    itself. ``encoding`` is imported by the submitted agent
    (``kaggriculture.learn.scripts.play``), which runs inside the competition
    sandbox against a ``kaggle_environments`` build this project does not
    control, and ``len(MARKET_SLOTS)`` sets the market head's output shape.
    Parsing the engine's source at import time turns an upstream refactor
    into an import that raises, or a head shape that silently no longer
    matches a trained checkpoint -- both forfeit the episode on turn zero
    with no logs to explain why. Here, in CI, the same mismatch fails loudly
    where someone can fix it, and never runs where the agent actually plays.
    """
    import inspect
    import re

    from kaggle_environments.envs.kaggriculture import kaggriculture as engine

    source = inspect.getsource(engine._process_market)
    match = re.search(r'op == "BUY_PRODUCT" and item in \(([^)]+)\)', source)
    assert match, "engine's BUY_PRODUCT gate pattern not found in _process_market"
    gated = set(re.findall(r'"([A-Z_]+)"', match.group(1)))

    ours = {item for verb, item in MARKET_SLOTS if verb == "BUY_PRODUCT"}

    engine_only = gated - ours
    ours_only = ours - gated
    assert not engine_only and not ours_only, (
        "BUY_PRODUCT drifted from the engine's gate: "
        f"the engine now also accepts {engine_only or '{}'}, "
        f"MARKET_SLOTS claims {ours_only or '{}'} that the engine rejects"
    )


def test_unit_labels_are_padded_and_masked() -> None:
    """Hands are hired through the day, so the acting unit count varies by turn."""
    action = {"farmer": ["WATER"], "hands": [["NORTH"]], "market": []}

    labels = encode_units(action, units=2)

    assert labels.shape == (1, MAX_UNITS)
    assert labels[0, 2].item() == IGNORE
    assert labels[0, 0].item() != IGNORE


def test_ops_for_units_that_are_not_on_the_board_are_not_labelled() -> None:
    """About 5% of corpus turns order more hands than the farm has.

    The engine's ``_farmer_position`` returns ``None`` past the end of
    ``farm["hands"]``, so those ops move nothing. Labelling them would teach the
    model an op for a unit that does not exist, and the head would read that
    slot at the padded tile -- pushing a real gradient through a tile nobody is
    standing on.
    """
    action = {"farmer": ["WATER"], "hands": [["NORTH"], ["DIG"]], "market": []}

    labels = encode_units(action, units=2)

    assert labels[0, 1].item() != IGNORE
    assert labels[0, 2].item() == IGNORE


def test_a_hand_the_action_never_ordered_is_left_unlabelled() -> None:
    """The other direction: a shorter hand-op list than the farm has hands.

    Such a hand is left where it stands by the engine. It gets no label rather
    than a ``PASS`` one, for the same reason an unhired hand does not.
    """
    action = {"farmer": ["WATER"], "hands": [], "market": []}

    labels = encode_units(action, units=3)

    assert labels[0, 0].item() != IGNORE
    assert labels[0, 1:].eq(IGNORE).all()


def test_planting_a_crop_is_a_distinct_label_per_crop() -> None:
    """PLANT MELON and PLANT WHEAT are different decisions, not one op."""
    melon = encode_units({"farmer": ["PLANT", "MELON"], "hands": [], "market": []}, 1)
    wheat = encode_units({"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []}, 1)

    assert melon[0, 0].item() != wheat[0, 0].item()


def test_labels_round_trip_back_to_a_legal_action() -> None:
    """Training on labels the play path cannot invert would be silently useless."""
    from kaggriculture.learn.encoding import UNIT_OPS, decode_units

    action = {"farmer": ["PLANT", "MELON"], "hands": [["WATER"], ["DIG"]], "market": []}
    labels = encode_units(action, units=3)
    logits = torch.full((1, MAX_UNITS, len(UNIT_OPS)), -10.0)
    for unit in range(3):
        logits[0, unit, int(labels[0, unit].item())] = 10.0

    decoded = decode_units(logits, units=3)

    assert decoded["farmer"] == ["PLANT", "MELON"]
    assert decoded["hands"] == [["WATER"], ["DIG"]]


def test_a_unit_count_beyond_max_units_raises() -> None:
    """Silently truncating a real hand's action would mislabel every unit after it."""
    action = {
        "farmer": ["PASS"],
        "hands": [["PASS"] for _ in range(MAX_UNITS)],
        "market": [],
    }

    with pytest.raises(ValueError):
        encode_units(action, units=MAX_UNITS + 1)


def test_every_verb_item_pair_the_corpus_uses_has_a_slot() -> None:
    """A missing slot silently drops an order the teacher actually played.

    BUY_PRODUCT is checked against ``{"WHEAT", "FERTILIZER"}``, not
    ``set(PRODUCTS)``: the engine's own ``_process_market`` gates BUY_PRODUCT
    with a literal ``item in ("WHEAT", "FERTILIZER")``, narrower than the
    catalogue SELL, BUY_SEED and BUY_ANIMAL each cover in full. Asserting the
    wider set here would pass while MARKET_SLOTS carried seven dead slots the
    engine silently no-ops.
    """
    verbs = {verb for verb, _ in MARKET_SLOTS}

    assert verbs == {"SELL", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL"}
    assert {item for verb, item in MARKET_SLOTS if verb == "SELL"} == set(PRODUCTS)
    assert {item for verb, item in MARKET_SLOTS if verb == "BUY_SEED"} == set(CROPS)
    assert {item for verb, item in MARKET_SLOTS if verb == "BUY_ANIMAL"} == set(ANIMALS)
    assert {item for verb, item in MARKET_SLOTS if verb == "BUY_PRODUCT"} == {
        "WHEAT",
        "FERTILIZER",
    }


def test_buckets_are_exact_where_the_corpus_is_dense() -> None:
    """93.5% of orders are 12 or fewer, so those must not be lumped into ranges."""
    assert QUANTITIES[:13] == tuple(range(13))
    assert bucket_of(0) == 0
    assert bucket_of(7) == 7
    assert bucket_of(12) == 12
    assert bucket_of(13) == bucket_of(16) > 12
    assert bucket_of(85) == len(QUANTITIES) - 1


def test_a_bucket_round_trips_to_a_quantity_that_lands_in_it() -> None:
    """Decoding must not emit a count outside the bucket it came from."""
    for n in (0, 1, 5, 12, 14, 20, 30, 85):
        assert bucket_of(quantity_of(bucket_of(n))) == bucket_of(n)


def test_hire_is_a_count_and_buy_land_is_a_flag() -> None:
    """HIRE is atomic, so hiring three hands is three orders, not a quantity."""
    action = {"market": [["HIRE"], ["HIRE"], ["HIRE"], ["BUY_LAND"]]}

    labels = encode_market(action)

    assert labels[0, HIRE_SLOT].item() == bucket_of(3)
    assert labels[0, LAND_SLOT].item() == 1


def test_repeated_orders_for_one_item_sum() -> None:
    """22% of order-bearing turns repeat a pair; dropping one loses a real sale."""
    action = {"market": [["SELL", "WHEAT", 3], ["SELL", "WHEAT", 4]]}

    labels = encode_market(action)

    assert labels[0, MARKET_SLOTS.index(("SELL", "WHEAT"))].item() == bucket_of(7)


def test_decoding_never_exceeds_the_engine_s_order_cap() -> None:
    """The engine truncates past MAX_ORDERS, so anything beyond it is discarded."""
    logits = torch.zeros(1, len(MARKET_SLOTS) + 2, len(QUANTITIES))
    logits[0, :, 1] = 10.0

    orders = decode_market(logits)

    assert len(orders) <= MAX_ORDERS


def test_decoding_puts_sells_before_buys() -> None:
    """Orders fill in sequence, so a sale must fund the purchase it precedes."""
    logits = torch.zeros(1, len(MARKET_SLOTS) + 2, len(QUANTITIES))
    logits[0, MARKET_SLOTS.index(("SELL", "WHEAT")), 2] = 10.0
    logits[0, MARKET_SLOTS.index(("BUY_SEED", "MELON")), 2] = 10.0

    orders = decode_market(logits)

    assert [order[0] for order in orders] == ["SELL", "BUY_SEED"]


def test_an_empty_market_decodes_to_no_orders() -> None:
    """Half of all turns trade nothing; emitting a zero order would be rejected."""
    logits = torch.zeros(1, len(MARKET_SLOTS) + 2, len(QUANTITIES))
    logits[0, :, 0] = 10.0

    assert decode_market(logits) == []
