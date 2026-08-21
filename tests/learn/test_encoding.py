"""Tests for turning an observation into tensors.

A wrong plane never raises. It trains the model on a game that is not this one,
and the first symptom is an agent that plays badly for reasons nobody can find.
So these check the encoding against the engine's own rules tables and against a
board built by hand.
"""

import copy
import json
import zipfile

import pytest
import torch
from kaggle_environments.envs.kaggriculture import kaggriculture as engine

from kaggriculture.constants import (
    ANIMALS,
    CROPS,
    EPISODE_STEPS,
    PRODUCTS,
    SHED_CAPACITY,
    SHOPS,
    TURNS_PER_DAY,
)
from kaggriculture.learn.corpus import CORPUS
from kaggriculture.learn.encoding import (
    BOARD,
    CARE_BONUS_SCALE,
    CARRIED_SCALE,
    ENCODED_FIELDS,
    HIRE_SLOT,
    IGNORE,
    LAND_SLOT,
    MARKET_SLOTS,
    MAX_ORDERS,
    MAX_UNITS,
    NO_DEATH_SCHEDULED,
    NOT_ENCODED,
    QUANTITIES,
    SCALARS,
    SEED_SCALE,
    SHED_NAMES,
    TILE_PLANES,
    TRANSFER_QUANTITY,
    UNIT_CARRIED_SCALE,
    UNIT_OPS,
    bucket_of,
    decode_market,
    decode_units,
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
    """Return a minimal well-formed observation with an empty unlocked board.

    ``private`` comes from the engine's own ``_new_private`` rather than being
    typed out here. The shed and the seed counts are dense -- every product and
    every crop keyed at zero -- and a hand-built ``{}`` would let an encoder
    that reaches for a missing key pass here and raise on the first real
    observation.
    """
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
        "private": engine._new_private(),
    }


def _plant(crop: str = "WHEAT", day: int = 0) -> dict:
    """Return a freshly planted tile, built by the engine rather than by hand.

    Every field a plant carries comes from ``_new_plant``, so a test cannot
    quietly agree with an encoder that reads a key the real game does not
    write, or miss one it does.
    """
    return engine._new_plant(crop, day, TURNS_PER_DAY)


def _animal(animal: str = "GOOSE", day: int = 0) -> dict:
    """Return a freshly placed animal tile, built by the engine."""
    return engine._new_animal(animal, day)


def _staff(observation: dict, hands: list[list[int]], seat: int = 0) -> dict:
    """Put ``hands`` on ``seat``'s farm, with an inventory each, and return it.

    The engine's ``_do_hire`` appends an inventory as it appends a hand, so a
    farm staffed without one is a state the game cannot produce -- and
    ``encode_board`` reads one inventory per unit. Hiring through this keeps
    the fixture on the engine's invariant instead of quietly testing a board
    that could never occur.
    """
    observation["farms"][seat]["hands"] = hands
    observation["private"]["inventories"] = [{} for _ in range(1 + len(hands))]
    return observation


def _one_hot(op: int, units: int = 1) -> torch.Tensor:
    """Return unit logits that ``decode_units`` will resolve to ``op`` everywhere.

    Built at the logit shape the model emits rather than by calling ``_op``
    directly, so what is measured is the action a rollout would really send.
    """
    logits = torch.full((1, MAX_UNITS, len(UNIT_OPS)), -10.0)
    logits[0, :units, op] = 10.0
    return logits


def _permit(logits: torch.Tensor) -> torch.Tensor:
    """Return an all-True mask shaped like ``logits``.

    The decoders take a legality mask because the play path must not select an
    op the engine would discard, and that behaviour is tested where it belongs,
    against real observations, in ``test_play.py`` and ``test_mask.py``. The
    tests below ask a narrower question -- what op list a *given* vocabulary
    index decodes to, its arity, its item, the order the market slots come out
    in -- and a mask would only stop them reaching the index they mean to
    exercise. Permitting everything keeps each of these a test of the spelling
    rules alone.
    """
    return torch.ones_like(logits, dtype=torch.bool)


def test_board_has_the_declared_shape_and_batch_dimension() -> None:
    """Every tensor in this project carries its batch dimension."""
    board = encode_board(empty_observation(), seat=0)

    assert board.shape == (1, TILE_PLANES, BOARD, BOARD)
    assert board.dtype == torch.float32


def test_a_planted_tile_lights_exactly_its_own_crop_plane() -> None:
    """Crop identity is one-hot; two crops lit at once would be a silent mixture."""
    observation = empty_observation()
    observation["farms"][0]["tiles"][2][3] = _plant("MELON")

    board = encode_board(observation, seat=0)
    crops = sorted(CROPS)
    lit = [board[0, index, 2, 3].item() for index in range(len(crops))]

    assert sum(lit) == 1.0
    assert lit[crops.index("MELON")] == 1.0


def test_the_opponent_s_board_occupies_its_own_planes() -> None:
    """Opponent supply decides prices here; their board must be visible and distinct."""
    observation = empty_observation()
    observation["farms"][1]["tiles"][5][5] = _plant("WHEAT")

    ours = encode_board(observation, seat=0)
    theirs = encode_board(observation, seat=1)

    assert not torch.equal(ours, theirs)


def _changed_planes(before: torch.Tensor, after: torch.Tensor) -> list[int]:
    """Return every plane index that differs between two encoded boards.

    Finding the planes rather than naming them keeps these tests off the plane
    layout: the indices are derived from the engine's rules tables and would
    shift under any upstream crop, animal or structure kind.
    """
    return (before != after).flatten(2).any(dim=2)[0].nonzero().flatten().tolist()


def _changed_plane(before: torch.Tensor, after: torch.Tensor) -> int:
    """Return the index of the one plane that differs between two encoded boards."""
    changed = _changed_planes(before, after)
    assert len(changed) == 1, f"expected one plane to move, got {changed}"
    return changed[0]


def _moved_scalars(before: torch.Tensor, after: torch.Tensor) -> dict[int, float]:
    """Return ``{index: after-value}`` for every scalar that differs.

    Located by comparing two observations that differ in one field, never by
    indexing a fixed offset. An index into this vector is a positional fact
    about a layout that grows every time a feature is appended -- one such
    assertion broke the moment ``hires_today`` was added -- and it tests where
    a value sits rather than what it means.
    """
    changed = (before != after)[0].nonzero().flatten().tolist()
    return {index: float(after[0, index]) for index in changed}


def test_the_board_says_where_the_farmer_stands() -> None:
    """Without this the trunk sees a board with nobody on it.

    The whole point of the spatial head is that a unit reads the trunk at its
    own tile; if no plane marks where anyone is standing, that column carries
    the tile's crops and nothing about the unit deciding from it.

    ``[7, 1]`` is ``x=7, y=1`` and the planes are indexed ``[y, x]``, so the
    farmer's plane lights at ``[1, 7]``. The asymmetry is the point: the opening
    position ``[4, 4]`` reads the same under either unpacking and proves
    nothing.
    """
    here, there = empty_observation(), empty_observation()
    there["farms"][0]["farmer"] = [7, 1]

    before, after = encode_board(here, seat=0), encode_board(there, seat=0)
    plane = _changed_plane(before, after)

    assert before[0, plane, 4, 4].item() == 1.0
    assert after[0, plane, 1, 7].item() == 1.0
    assert after[0, plane, 7, 1].item() == 0.0
    assert after[0, plane, 4, 4].item() == 0.0


def test_the_hands_plane_counts_rather_than_flags() -> None:
    """Two hands may stand on one tile -- ``[[4, 3], [4, 3]]`` occurs in the corpus.

    A flag would report a crowd the same as a lone hand, so the plane carries a
    count. It is scaled by ``MAX_UNITS`` to share the range of the other
    continuous features, which this checks by ratio rather than by value.
    """
    one = _staff(empty_observation(), [[4, 3]])
    two = _staff(empty_observation(), [[4, 3], [4, 3]])

    single, doubled = encode_board(one, seat=0), encode_board(two, seat=0)
    plane = _changed_plane(single, doubled)

    assert single[0, plane, 3, 4].item() == pytest.approx(1.0 / MAX_UNITS)
    assert doubled[0, plane, 3, 4].item() == pytest.approx(2.0 / MAX_UNITS)


def test_the_opponent_s_units_occupy_their_own_planes() -> None:
    """Their farmer and hands are public, and where they stand decides prices."""
    empty, staffed = empty_observation(), empty_observation()
    _staff(staffed, [[8, 8]], seat=1)

    plane = _changed_plane(encode_board(empty, seat=0), encode_board(staffed, seat=0))

    assert plane >= TILE_PLANES // 2


def test_positions_are_ordered_exactly_as_labels_are() -> None:
    """Slot ``k``'s position and slot ``k``'s label must describe one unit.

    The head reads slot ``k``'s trunk column at slot ``k``'s position and scores
    it against slot ``k``'s label, so a divergence in order silently issues
    every unit another unit's orders.
    """
    observation = _staff(empty_observation(), [[7, 1], [0, 9]])
    observation["farms"][0]["farmer"] = [2, 3]

    positions = encode_positions(observation, seat=0)

    assert positions.shape == (1, MAX_UNITS)
    assert positions.dtype == torch.int64
    assert positions[0, 0].item() == 3 * BOARD + 2
    assert positions[0, 1].item() == 1 * BOARD + 7
    assert positions[0, 2].item() == 9 * BOARD + 0
    assert positions[0, 3:].eq(0).all()


def test_positions_are_read_for_the_seat_asked_for() -> None:
    """``seat`` selects whose units to locate, as it does everywhere else here."""
    observation = empty_observation()
    observation["farms"][1]["farmer"] = [9, 9]

    assert encode_positions(observation, seat=0)[0, 0].item() == 4 * BOARD + 4
    assert encode_positions(observation, seat=1)[0, 0].item() == 9 * BOARD + 9


def test_a_position_count_beyond_max_units_raises() -> None:
    """Truncating here would hand a real unit another unit's tile."""
    observation = _staff(empty_observation(), [[0, 0] for _ in range(MAX_UNITS)])

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
            assert positions[0, :units].tolist() == [y * BOARD + x for x, y in tiles]
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


def test_the_shed_is_encoded_because_sell_draws_from_it() -> None:
    """``SELL`` is the engine's only money-increasing op and it empties the shed.

    A clone trained without this banked 0 coins across 500 episodes: it was
    asked to choose what to sell with no way to know what it held. One product
    moving in the shed must move exactly one scalar, by the documented ratio.
    """
    empty, stocked = empty_observation(), empty_observation()
    stocked["private"]["shed"]["WHEAT"] = 20

    moved = _moved_scalars(encode_scalars(empty, 0), encode_scalars(stocked, 0))

    assert len(moved) == 1
    assert next(iter(moved.values())) == pytest.approx(20 / SHED_CAPACITY)


def test_an_animal_bought_into_the_shed_is_visible_there() -> None:
    """``BUY_ANIMAL`` puts a cow in the shed, not on the board.

    It stays there until a unit ``PICKUP``s and ``PLACE``s it, so a shed encoded
    over products alone leaves that middle step invisible: the policy can buy an
    animal and then have no way to see that it owns one. The shed the engine
    writes is keyed by every product *and* every animal.
    """
    empty, bought = empty_observation(), empty_observation()
    bought["private"]["shed"]["COW"] = 2

    moved = _moved_scalars(encode_scalars(empty, 0), encode_scalars(bought, 0))

    assert len(moved) == 1
    assert next(iter(moved.values())) == pytest.approx(2 / SHED_CAPACITY)


def test_the_shed_block_covers_every_key_the_engine_puts_there() -> None:
    """A key the shed carries and this vector omits is silently unspendable.

    Asserted against the engine's own fresh private state rather than a list
    written here, so an animal or product added upstream fails this test instead
    of quietly falling out of the encoding -- which is how the whole shed came to
    be missing in the first place.
    """
    assert set(SHED_NAMES) == set(engine._new_private()["shed"])


def test_each_product_in_the_shed_has_its_own_scalar() -> None:
    """Twenty wheat and twenty melons are different hands to play.

    One shared total would let the model learn that it has something to sell
    without ever learning what, which is the same blindness one step along.
    """
    wheat, melon = empty_observation(), empty_observation()
    wheat["private"]["shed"]["WHEAT"] = 20
    melon["private"]["shed"]["MELON"] = 20

    moved = _moved_scalars(encode_scalars(wheat, 0), encode_scalars(melon, 0))

    assert len(moved) == 2


def test_seeds_are_encoded_because_plant_requires_them() -> None:
    """The engine drops every ``PLANT`` for a crop a turn overspends seeds on.

    Seeds live outside the shed, in ``private["seeds"]``, so nothing else in
    the vector implies them.
    """
    none, sown = empty_observation(), empty_observation()
    sown["private"]["seeds"]["MELON"] = 4

    moved = _moved_scalars(encode_scalars(none, 0), encode_scalars(sown, 0))

    assert len(moved) == 1
    assert next(iter(moved.values())) == pytest.approx(4 / SEED_SCALE)


def test_carried_produce_is_totalled_across_our_units() -> None:
    """What the crew is holding is produce on its way to the shed.

    Totalled rather than per-unit here because a market decision is about the
    farm's whole holding; the board carries the per-unit split. Both
    observations staff one hand, so the hand-count scalar is identical and only
    the total can move.
    """
    idle = _staff(empty_observation(), [[4, 3]])
    laden = _staff(empty_observation(), [[4, 3]])
    laden["private"]["inventories"] = [{"WHEAT": 2}, {"WHEAT": 3}]

    moved = _moved_scalars(encode_scalars(idle, 0), encode_scalars(laden, 0))

    assert len(moved) == 1
    assert next(iter(moved.values())) == pytest.approx(5 / CARRIED_SCALE)


def test_carried_produce_is_not_the_same_scalar_as_the_shed() -> None:
    """Five wheat in hand cannot be sold; five in the shed can.

    ``_commit_unit`` gates ``SELL`` on the shed alone, and the end-of-day drop
    is what moves one to the other. Folding them into one number would tell the
    model it can sell produce that is still out in the field.
    """
    in_shed, in_hand = empty_observation(), empty_observation()
    in_shed["private"]["shed"]["WHEAT"] = 5
    in_hand["private"]["inventories"] = [{"WHEAT": 5}]

    base = encode_scalars(empty_observation(), 0)
    shed_moved = _moved_scalars(base, encode_scalars(in_shed, 0))
    hand_moved = _moved_scalars(base, encode_scalars(in_hand, 0))

    assert len(shed_moved) == len(hand_moved) == 1
    assert set(shed_moved) != set(hand_moved)


def test_each_unlocked_shop_lights_its_own_flag() -> None:
    """*Which* shops are open decides where the demand is, not how many.

    Each shop consumes a fixed list of products every few turns, so two towns
    with one shop open can be buying disjoint things. Both observations here
    open exactly one shop, which holds the pre-existing count scalar fixed and
    leaves only the per-shop flags free to move.
    """
    bakery, yarn = empty_observation(), empty_observation()
    bakery["town"]["unlocked_shops"] = ["BAKERY"]
    yarn["town"]["unlocked_shops"] = ["YARN_STORE"]

    moved = _moved_scalars(encode_scalars(bakery, 0), encode_scalars(yarn, 0))

    assert len(moved) == 2
    assert sorted(moved.values()) == [0.0, 1.0]


def test_a_bare_coop_and_a_bare_pasture_are_different_states() -> None:
    """``BUILD_COOP`` and ``BUILD_PASTURE`` are distinct ops with distinct results.

    ``PLACE`` accepts an animal only onto ``ANIMALS[item]["structure"]``, so a
    tile already built as a coop can never take a cow. Collapsed into one
    ``STRUCTURE`` state, as they were, the model could not tell which build a
    tile was already committed to.
    """
    coop, pasture = empty_observation(), empty_observation()
    coop["farms"][0]["tiles"][2][3] = {"kind": "COOP"}
    pasture["farms"][0]["tiles"][2][3] = {"kind": "PASTURE"}

    built, penned = encode_board(coop, seat=0), encode_board(pasture, seat=0)
    changed = _changed_planes(built, penned)

    assert len(changed) == 2
    assert sorted(
        (built[0, plane, 2, 3].item(), penned[0, plane, 2, 3].item())
        for plane in changed
    ) == [(0.0, 1.0), (1.0, 0.0)]


def test_an_animal_already_cared_for_today_reads_differently() -> None:
    """``CARE`` is a unit op the engine no-ops if the day's care already landed.

    Without this the model cannot tell a productive ``CARE`` from a wasted turn.
    """
    unattended, attended = empty_observation(), empty_observation()
    unattended["farms"][0]["tiles"][2][3] = _animal()
    attended["farms"][0]["tiles"][2][3] = _animal() | {"cared_today": True}

    before = encode_board(unattended, seat=0)
    after = encode_board(attended, seat=0)
    plane = _changed_plane(before, after)

    assert before[0, plane, 2, 3].item() == 0.0
    assert after[0, plane, 2, 3].item() == 1.0


def test_the_care_bonus_an_animal_is_owed_is_encoded() -> None:
    """The payoff ``CARE`` works toward, banked per cared-and-fed day.

    ``_daily_refresh_animals`` spends it on the next production day, so it is
    the difference between an animal that yields one unit and one that yields
    four.
    """
    owed, unowed = empty_observation(), empty_observation()
    unowed["farms"][0]["tiles"][2][3] = _animal()
    owed["farms"][0]["tiles"][2][3] = _animal() | {"pending_care_bonus": 3}

    after = encode_board(owed, seat=0)
    plane = _changed_plane(encode_board(unowed, seat=0), after)

    assert after[0, plane, 2, 3].item() == pytest.approx(3 / CARE_BONUS_SCALE)


def test_a_plant_near_the_end_of_its_life_reads_differently_from_a_fresh_one() -> None:
    """``_decay_plants`` starts taking a unit every other step past this step.

    Both tiles are the same crop planted on the same day, so the crop and age
    planes are identical and only the remaining lifespan can move.
    """
    fresh, doomed = empty_observation(), empty_observation()
    fresh["farms"][0]["tiles"][2][3] = _plant("WHEAT")
    doomed["farms"][0]["tiles"][2][3] = _plant("WHEAT") | {"max_lifespan_step": 10}

    young, dying = encode_board(fresh, seat=0), encode_board(doomed, seat=0)
    plane = _changed_plane(young, dying)

    expected = _plant("WHEAT")["max_lifespan_step"] / EPISODE_STEPS
    assert young[0, plane, 2, 3].item() == pytest.approx(expected)
    assert dying[0, plane, 2, 3].item() == pytest.approx(10 / EPISODE_STEPS)


def test_a_crop_with_no_death_scheduled_does_not_read_as_dying() -> None:
    """An ongoing crop carries ``max_lifespan_step`` -1 until its last yield.

    Encoded literally that is ``(-1 - step) / EPISODE_STEPS``, a small negative
    that slides to -1 across the season -- exactly where a plant that is
    already decaying sits. The sentinel gets its own value at the opposite end
    instead. 53.6% of planted tiles in the corpus carry it, so getting this
    wrong would mislabel most of the field.
    """
    ongoing, doomed = empty_observation(), empty_observation()
    forever = _plant("TOMATO")
    assert forever["max_lifespan_step"] == -1, "fixture no longer covers the sentinel"
    ongoing["farms"][0]["tiles"][2][3] = forever
    doomed["farms"][0]["tiles"][2][3] = forever | {"max_lifespan_step": 0}

    alive, expiring = encode_board(ongoing, seat=0), encode_board(doomed, seat=0)
    plane = _changed_plane(alive, expiring)

    assert alive[0, plane, 2, 3].item() == pytest.approx(NO_DEATH_SCHEDULED)
    assert expiring[0, plane, 2, 3].item() == pytest.approx(0.0)


def test_the_board_says_what_each_unit_is_carrying() -> None:
    """The unit head reads the trunk at the unit's own tile.

    Whether *this* hand has anything on it is what decides ``DROP`` and
    ``PLACE``, and a farm-wide total in the scalars cannot say which hand.
    """
    idle, laden = empty_observation(), empty_observation()
    laden["private"]["inventories"] = [{"WHEAT": 4}]

    before, after = encode_board(idle, seat=0), encode_board(laden, seat=0)
    plane = _changed_plane(before, after)

    assert plane < TILE_PLANES // 2
    assert after[0, plane, 4, 4].item() == pytest.approx(4 / UNIT_CARRIED_SCALE)


def test_the_opponent_s_carried_plane_stays_zero_because_it_is_hidden() -> None:
    """Their ``private`` is not in our observation, so their plane means unknown.

    Filling it from our own inventories -- the one mistake this plane invites --
    would teach the trunk that the opponent is carrying whatever we are.
    """
    idle, laden = empty_observation(), empty_observation()
    laden["private"]["inventories"] = [{"WHEAT": 4}]

    after = encode_board(laden, seat=0)
    plane = _changed_plane(encode_board(idle, seat=0), after)

    assert after[0, plane + TILE_PLANES // 2].eq(0.0).all()


def test_the_carried_plane_sums_units_sharing_a_tile() -> None:
    """Units share a tile routinely, most of all the four shed-access corners.

    One hand carrying nothing beside one carrying nine is not the same tile as
    two empty-handed hands, so the plane accumulates like the hands count does.
    """
    lone = _staff(empty_observation(), [[4, 3], [4, 3]])
    both = _staff(empty_observation(), [[4, 3], [4, 3]])
    lone["private"]["inventories"] = [{}, {"WHEAT": 2}, {}]
    both["private"]["inventories"] = [{}, {"WHEAT": 2}, {"WHEAT": 3}]

    after = encode_board(both, seat=0)
    plane = _changed_plane(encode_board(lone, seat=0), after)

    assert after[0, plane, 3, 4].item() == pytest.approx(5 / UNIT_CARRIED_SCALE)


# --------------------------------------------------------------------------
# The transposition guard: a unit's position is `[x, y]`, the grid is
# `tiles[y][x]`, and every assertion below is asymmetric on purpose.
# --------------------------------------------------------------------------

# Where the guard's unit stands: `x=7, y=1`, so the tile it acts on is
# `tiles[1][7]` and its mirror is `tiles[7][1]`. Both are on the board and the
# two are distinct, which the fixture's own `[4, 4]` opening position is not --
# on a symmetric position every assertion here passes under either unpacking,
# which is how the transposition survived a full test suite and a training run.
_ASYMMETRIC: tuple[int, int] = (7, 1)


def _engine_acts_at(observation: dict, seat: int = 0) -> tuple[int, int]:
    """Return the ``(y, x)`` grid cell the engine's own dispatch reads, by DIG.

    Asked of ``_apply_unit_action`` rather than asserted from the position,
    because the position-to-grid mapping is exactly the thing under test and a
    test that writes it out twice only proves it agrees with itself. ``DIG``
    is the probe because it clears ``farm["tiles"][fy][fx]`` outright, so the
    cell the engine chose is visible as the one that changed.
    """
    farm = copy.deepcopy(observation["farms"][seat])
    engine._apply_unit_action(
        farm,
        copy.deepcopy(observation["private"]),
        0,
        ["DIG"],
        BOARD,
        observation["day"],
        TURNS_PER_DAY,
        SHED_CAPACITY,
    )
    before = observation["farms"][seat]["tiles"]
    changed = [
        (y, x)
        for y in range(BOARD)
        for x in range(BOARD)
        if farm["tiles"][y][x] != before[y][x]
    ]
    assert len(changed) == 1, f"DIG did not identify one tile: {changed}"
    return changed[0]


def _gathered(board: torch.Tensor, positions: torch.Tensor, unit: int) -> torch.Tensor:
    """Return the trunk column ``Policy.forward`` gathers for one unit.

    The same ``flatten(2)`` plus flat index the model uses, so this measures the
    column the per-unit head actually reads rather than a re-derivation of it.
    """
    return board.flatten(2)[0, :, int(positions[0, unit].item())]


def test_the_gathered_tile_is_the_tile_the_engine_acts_on() -> None:
    """``encode_positions`` must point at the tile ``_apply_unit_action`` reads.

    A unit's position is ``[x, y]`` and the grid is ``tiles[y][x]``, so the flat
    index is ``y * BOARD + x``. Transposed, the head gathers the mirrored tile:
    a valid index into a valid plane, scored against this unit's real label,
    raising nothing and training the readout on somebody else's surroundings.
    The per-unit head exists precisely so a unit reads its own tile, so this is
    the invariant the whole readout rests on.

    Proven against the engine, not against a fixture. The acted-on cell is read
    out of ``_apply_unit_action`` itself, and the two candidate tiles carry
    different crops so the final assertion discriminates: swap the unpacking
    back to ``(y, x)`` and the gathered column becomes the melon's.
    """
    x, y = _ASYMMETRIC
    observation = empty_observation()
    observation["farms"][0]["farmer"] = [x, y]
    observation["farms"][0]["tiles"][y][x] = _plant("WHEAT")
    observation["farms"][0]["tiles"][x][y] = _plant("MELON")

    assert _engine_acts_at(observation) == (y, x)

    board = encode_board(observation, seat=0)
    column = _gathered(board, encode_positions(observation, seat=0), unit=0)

    assert torch.equal(column, board[0, :, y, x])
    assert not torch.equal(column, board[0, :, x, y])


def test_a_unit_s_own_planes_are_written_at_the_tile_it_gathers() -> None:
    """Occupancy and carried load must land in the column the head reads.

    ``encode_positions`` and ``_write_units``/``_write_carried`` unpack the same
    ``[x, y]`` list in three separate places, so agreeing with the engine about
    where a unit *is* does not imply writing its planes there. Transposing only
    the writers leaves the gathered column pointing at the right tile with the
    unit's own presence and load missing from it -- and the previous test still
    passes, because it compares two slices of one tensor.
    """
    x, y = _ASYMMETRIC
    idle = empty_observation()
    idle["farms"][0]["farmer"] = [x, y]
    laden = empty_observation()
    laden["farms"][0]["farmer"] = [x, y]
    laden["private"]["inventories"] = [{"WHEAT": 4}]

    occupancy = _changed_plane(
        encode_board(empty_observation(), 0), encode_board(idle, 0)
    )
    carried = _changed_plane(encode_board(idle, 0), encode_board(laden, 0))
    board = encode_board(laden, seat=0)
    column = _gathered(board, encode_positions(laden, seat=0), unit=0)

    assert column[occupancy].item() == 1.0
    assert column[carried].item() == pytest.approx(4 / UNIT_CARRIED_SCALE)


def test_every_hand_gathers_its_own_tile_and_not_another_s() -> None:
    """Slot ``k`` reads slot ``k``'s tile, on positions no transposition fixes.

    Two hands mirrored across the diagonal: under ``(y, x)`` they swap tiles
    with each other, which is the exact failure the per-unit head was rebuilt to
    avoid -- issuing one hand's orders from another hand's surroundings.
    """
    observation = _staff(empty_observation(), [[7, 1], [1, 7]])
    observation["farms"][0]["tiles"][1][7] = _plant("WHEAT")
    observation["farms"][0]["tiles"][7][1] = _plant("MELON")

    board = encode_board(observation, seat=0)
    positions = encode_positions(observation, seat=0)

    assert torch.equal(_gathered(board, positions, 1), board[0, :, 1, 7])
    assert torch.equal(_gathered(board, positions, 2), board[0, :, 7, 1])


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


# The widest any encoded scalar gets on real play, so a scalar that has lost
# its divisor is loud. Measured across the sweep below: 14.76, both times a
# money scalar (money / 10_000, and the richest farm sampled banked ~148k);
# every other scalar stays under 7. 64 leaves four times that headroom and
# still fails a raw, undivided market inventory (10,000), a raw money balance
# or the seed tail (219). It does not fail a raw shed count, which tops out at
# 96 -- the guard against that block being wrong is
# ``test_a_real_observation_s_private_state_reaches_the_scalars``, not this
# bound.
SANE_BOUND = 64.0

# One episode per archive, every seventh turn, both seats. Seven is coprime
# with the 24-turn day on purpose: a stride that divides it reads every
# inventory just after the end-of-day drop has emptied it, and would report
# carried produce as zero on every row it looked at.
_CORPUS_STRIDE = 7


def _corpus_turns(episodes: int = 1, stride: int = _CORPUS_STRIDE) -> list[tuple]:
    """Return ``(archive, observation, seat)`` from every archive on disk.

    Every archive rather than one: the ladder's agent mix changes daily, and
    two committed claims in this repo have already come from generalising a
    single archive.
    """
    turns = []
    for archive in sorted(CORPUS.glob("*.zip")):
        with zipfile.ZipFile(archive) as bundle:
            names = [n for n in bundle.namelist() if n.endswith(".json")][:episodes]
            for name in names:
                with bundle.open(name) as member:
                    steps = json.load(member)["steps"]
                for index in range(0, len(steps), stride):
                    for seat in (0, 1):
                        turns.append(
                            (archive.name, steps[index][seat]["observation"], seat)
                        )
    return turns


@pytest.mark.slow
@pytest.mark.skipif(not CORPUS.is_dir(), reason="needs the replay corpus")
def test_every_scalar_from_a_real_observation_is_finite_and_bounded() -> None:
    """A hand-built observation agrees with the encoder; a real one does not.

    This is the shape of test that would have caught the shed being absent. An
    unencoded field never appears, never raises and never widens the vector --
    the only thing that notices is code that reads a real observation end to
    end. It also catches the opposite failure, a new field encoded at the wrong
    scale, which arrives as a value nothing else in the vector is near.
    """
    turns = _corpus_turns()

    assert turns, "no archives on disk, so this proved nothing"
    for archive, observation, seat in turns:
        scalars = encode_scalars(observation, seat)
        board = encode_board(observation, seat)
        assert scalars.shape == (1, SCALARS)
        assert board.shape == (1, TILE_PLANES, BOARD, BOARD)
        assert torch.isfinite(scalars).all(), (
            f"{archive} seat {seat}: non-finite scalar"
        )
        assert torch.isfinite(board).all(), f"{archive} seat {seat}: non-finite plane"
        assert scalars.abs().max().item() <= SANE_BOUND, (
            f"{archive} seat {seat}: {scalars.abs().max().item()} exceeds {SANE_BOUND}"
        )


@pytest.mark.slow
@pytest.mark.skipif(not CORPUS.is_dir(), reason="needs the replay corpus")
def test_a_real_observation_s_private_state_reaches_the_scalars() -> None:
    """Each private group must change the vector on real play, not just in a fixture.

    Blanking one group of a real observation's private state and re-encoding is
    the direct form of the question the audit asked: does this field reach the
    model at all. It is asked on real turns because a hand-made observation can
    only ever confirm what the encoder already does -- the shed was absent for
    a whole training run while every fast test passed.

    The town's shops are checked by swapping the open set for a different set
    of the same size, which holds the pre-existing count scalar fixed so only
    the per-shop identity flags can answer.
    """
    zeroed = engine._new_private()
    reached = {"shed": 0, "seeds": 0, "inventories": 0, "shops": 0}

    for _archive, observation, seat in _corpus_turns():
        private = observation["private"]
        baseline = encode_scalars(observation, seat)
        blanked = {
            "shed": {**private, "shed": zeroed["shed"]},
            "seeds": {**private, "seeds": zeroed["seeds"]},
            "inventories": {
                **private,
                "inventories": [{} for _ in private["inventories"]],
            },
        }
        for group, replacement in blanked.items():
            stripped = {**observation, "private": replacement}
            reached[group] += not torch.equal(baseline, encode_scalars(stripped, seat))

        open_shops = observation["town"]["unlocked_shops"]
        others = sorted(SHOPS)[: len(open_shops)]
        if sorted(open_shops) != others:
            swapped = {**observation, "town": {"unlocked_shops": others}}
            reached["shops"] += not torch.equal(baseline, encode_scalars(swapped, seat))

    assert all(reached.values()), f"never reached the scalars: {reached}"


@pytest.mark.slow
@pytest.mark.skipif(not CORPUS.is_dir(), reason="needs the replay corpus")
def test_every_field_the_corpus_carries_is_either_encoded_or_argued_away() -> None:
    """A field the engine emits must be named somewhere, encoded or refused.

    This is the structural half of the fix. The shed, the seeds and the carried
    inventories went unencoded for a whole training run because nothing in the
    codebase ever had to mention them: an absent field has no width, no shape
    and no test. Now every key the corpus carries has to appear in
    ``ENCODED_FIELDS`` or in ``NOT_ENCODED`` with a reason, so an upstream
    addition fails here rather than being silently dropped into a shard.

    Checked in both directions. Forward, a key the corpus carries that neither
    set names is an omission. Backward, a key claimed as encoded that the
    corpus never carries is a stale claim -- which is how a table like this
    rots into decoration.
    """
    seen: dict[str, set[str]] = {name: set() for name in ENCODED_FIELDS}

    for _archive, observation, _seat in _corpus_turns(episodes=2, stride=4):
        seen["observation"] |= set(observation)
        seen["market"] |= set(observation["market"])
        seen["town"] |= set(observation["town"])
        seen["private"] |= set(observation["private"])
        for farm in observation["farms"]:
            seen["farm"] |= set(farm)
            for row in farm["tiles"]:
                for tile in row:
                    if isinstance(tile, dict):
                        seen["tile"] |= {str(key) for key in tile}

    assert seen["observation"], "no archives on disk, so this proved nothing"
    for name, keys in seen.items():
        unaccounted = keys - ENCODED_FIELDS[name] - NOT_ENCODED
        assert not unaccounted, (
            f"{name} carries {sorted(unaccounted)}, which is neither encoded nor "
            "listed in NOT_ENCODED with a reason"
        )
        stale = ENCODED_FIELDS[name] - keys
        assert not stale, (
            f"{name} claims {sorted(stale)}, which the corpus never carries"
        )


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


def test_pickup_and_place_carry_the_item_the_engine_requires() -> None:
    """Both return on ``len(action) < 2``, so a bare verb can never land.

    While they were bare verbs every one ``decode_units`` emitted was discarded,
    and with ``PLACE`` dead an animal bought into the shed could never reach a
    structure -- livestock was unreachable to a learned policy for the whole of
    training.

    The item lists are asserted against ``PRODUCTS`` and ``ANIMALS`` rather than
    against what the corpus plays. The corpus is a sample of other people's
    agents and never places a CARROT, an EGG or a TOMATO; the engine accepts all
    three, because PICKUP reads ``private["shed"]`` and PLACE reads a unit's
    inventory, and neither mapping is gated by an item catalogue.
    """
    catalogue = set(PRODUCTS) | set(ANIMALS)

    assert {op.split(":", 1)[1] for op in UNIT_OPS if op.startswith("PICKUP:")} == (
        catalogue
    )
    assert {op.split(":", 1)[1] for op in UNIT_OPS if op.startswith("PLACE:")} == (
        catalogue
    )
    assert "PICKUP" not in UNIT_OPS and "PLACE" not in UNIT_OPS


def test_the_shed_the_engine_writes_is_what_pickup_and_place_can_name() -> None:
    """The item lists are the shed's own keys, so an upstream addition is loud.

    This is the ``BUY_PRODUCT`` lesson applied one verb over: a hand-written
    vocabulary passes against another hand-written list while both are wrong.
    ``engine._new_private()`` is the engine's own statement of what a shed holds.
    """
    shed = set(engine._new_private()["shed"])

    assert {op.split(":", 1)[1] for op in UNIT_OPS if op.startswith("PICKUP:")} == shed


def test_a_decoded_pickup_moves_state_in_the_engine() -> None:
    """The end-to-end claim: what ``decode_units`` emits, the engine acts on.

    Run through ``_apply_unit_action`` itself rather than asserted on the shape
    of the list, because the failure being fixed here was precisely an action
    whose shape looked fine and which the engine silently dropped. The farmer
    starts on ``_default_spawn``, which is shed-adjacent, so the shed branch is
    reachable without moving anybody.
    """
    observation = empty_observation()
    observation["farms"][0]["farmer"] = list(engine._default_spawn(BOARD))
    observation["private"]["shed"]["WHEAT"] = 5
    farm = observation["farms"][0]
    private = observation["private"]

    logits = _one_hot(UNIT_OPS.index("PICKUP:WHEAT"))
    op = decode_units(logits, 1, _permit(logits))["farmer"]
    engine._apply_unit_action(
        farm, private, 0, op, BOARD, 0, TURNS_PER_DAY, SHED_CAPACITY
    )

    assert op == ["PICKUP", "WHEAT", TRANSFER_QUANTITY]
    assert private["shed"]["WHEAT"] == 5 - TRANSFER_QUANTITY
    assert private["inventories"][0] == {"WHEAT": TRANSFER_QUANTITY}


def test_a_decoded_place_puts_a_bought_animal_onto_its_structure() -> None:
    """The move that was unreachable: a cow leaves the shed and reaches a pasture.

    ``PLACE`` pairs an animal with ``ANIMALS[item]["structure"]``, so this also
    pins that the decoded op is the one the engine's animal branch accepts
    rather than the shed-drop branch it would otherwise fall through to.
    """
    observation = empty_observation()
    farm = observation["farms"][0]
    x, y = farm["farmer"]
    farm["tiles"][y][x] = {"kind": str(ANIMALS["COW"]["structure"])}
    private = observation["private"]
    private["inventories"][0] = {"COW": 1}

    logits = _one_hot(UNIT_OPS.index("PLACE:COW"))
    op = decode_units(logits, 1, _permit(logits))["farmer"]
    engine._apply_unit_action(
        farm, private, 0, op, BOARD, 0, TURNS_PER_DAY, SHED_CAPACITY
    )

    assert farm["tiles"][y][x]["animal"] == "COW"
    assert private["inventories"][0] == {}


def test_a_recorded_pickup_labels_by_item_and_drops_its_quantity() -> None:
    """``["PICKUP", "WHEAT", 2]`` and ``["PICKUP", "WHEAT", 1]`` are one label.

    The vocabulary has no slot for a count, so cloning is lossy about *how much*
    and exact about *what* -- which is the half that decides whether the op
    lands at all. Two different items must not collide.
    """
    two = encode_units({"farmer": ["PICKUP", "WHEAT", 2], "hands": [], "market": []}, 1)
    one = encode_units({"farmer": ["PICKUP", "WHEAT", 1], "hands": [], "market": []}, 1)
    other = encode_units({"farmer": ["PICKUP", "COW", 1], "hands": [], "market": []}, 1)
    bare = encode_units({"farmer": ["PLACE", "COW"], "hands": [], "market": []}, 1)

    assert two[0, 0].item() == one[0, 0].item()
    assert two[0, 0].item() != other[0, 0].item()
    assert bare[0, 0].item() == UNIT_OPS.index("PLACE:COW")


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
    action = {"farmer": ["PLANT", "MELON"], "hands": [["WATER"], ["DIG"]], "market": []}
    labels = encode_units(action, units=3)
    logits = torch.full((1, MAX_UNITS, len(UNIT_OPS)), -10.0)
    for unit in range(3):
        logits[0, unit, int(labels[0, unit].item())] = 10.0

    decoded = decode_units(logits, 3, _permit(logits))

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
    assert bucket_of(200) == len(QUANTITIES) - 1


def test_a_bucket_round_trips_to_a_quantity_that_lands_in_it() -> None:
    """Decoding must not emit a count outside the bucket it came from."""
    for n in (0, 1, 5, 12, 14, 20, 30, 85, 130, 200):
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

    orders = decode_market(logits, _permit(logits))

    assert len(orders) <= MAX_ORDERS


def test_decoding_puts_sells_before_buys() -> None:
    """Orders fill in sequence, so a sale must fund the purchase it precedes."""
    logits = torch.zeros(1, len(MARKET_SLOTS) + 2, len(QUANTITIES))
    logits[0, MARKET_SLOTS.index(("SELL", "WHEAT")), 2] = 10.0
    logits[0, MARKET_SLOTS.index(("BUY_SEED", "MELON")), 2] = 10.0

    orders = decode_market(logits, _permit(logits))

    assert [order[0] for order in orders] == ["SELL", "BUY_SEED"]


def test_an_empty_market_decodes_to_no_orders() -> None:
    """Half of all turns trade nothing; emitting a zero order would be rejected."""
    logits = torch.zeros(1, len(MARKET_SLOTS) + 2, len(QUANTITIES))
    logits[0, :, 0] = 10.0

    assert decode_market(logits, _permit(logits)) == []
