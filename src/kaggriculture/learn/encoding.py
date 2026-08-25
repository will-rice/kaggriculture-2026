"""Turning one observation into the tensors a policy network reads.

Plane order is derived from the engine's own rules tables (``CROPS``,
``ANIMALS``, ``PRODUCTS``) rather than written out by hand, so a crop or
animal added upstream shifts nothing but the width of its own block.

Both farms are public in this game, so both get real capacity. Rather than
overlay the two farms onto one shared set of tile planes -- where a crop of
ours and a crop of the opponent's at the same coordinate would silently
overwrite one another -- each farm gets its own identical block of planes,
``seat``'s own farm first and the opponent's second. The market branch keeps
a price signal and an inventory signal per product, because value in this
game is a property of a product given what the opponent is selling: an
earlier change that replaced an opponent-aware animal valuation with a
static per-animal one measured 10,000 coins worse over 20 games.

Each farm block also carries where its units stand. Without those planes the
trunk sees a board with nobody on it, so the network can learn what the day
calls for but never that *this* hand is the one standing next to the weeds.
Both farms get them, since the opponent's units are public too.

``observation["private"]`` -- the shed, the unplanted seeds and what each unit
is carrying -- reaches the scalars rather than the board. It has to reach
something: ``SELL`` is the only operation that increases a farm's money and it
draws from the shed, ``PLANT`` consumes a seed, and ``PLACE``/``DROP`` move
what a unit carries. A clone trained without any of it banked 0 coins across
500 episodes, emitting sell orders with no knowledge of what it held.

The private mapping belongs to whoever's observation it is, and the opponent's
is hidden. ``encode_x(steps[i][seat]["observation"], seat)`` is the only call
shape, so ``observation["private"]`` is already ``seat``'s own and is never
indexed by seat.

Carried produce reaches both branches. The scalars carry the per-product total
across our whole crew, which is what a market decision needs; the board carries
one plane per farm block holding how much *each* unit is carrying, written at
the tile that unit stands on, which is what a unit decision needs -- the unit
head reads the trunk at its own tile, and whether this hand has anything on it
is exactly what decides ``DROP`` and ``PLACE``. Our block is filled from
``private["inventories"]``; the opponent's stays zero because their private
state is hidden, so a zero in that block means *unknown*, not *empty*.

``ENCODED_FIELDS`` and ``NOT_ENCODED`` below record which observation keys this
module reads, and a corpus test fails if the engine ever starts emitting one
that appears in neither. The shed went unencoded for a whole training run
because an absent field simply never shows up; a field that has to be named
somewhere cannot go missing the same way.

One fact about the engine's geometry is load-bearing everywhere a unit is read.
A unit's position is ``[x, y]`` and the grid is ``tiles[y][x]``:
``_apply_unit_action`` reads ``fx, fy = pos[0], pos[1]`` and then indexes
``farm["tiles"][fy][fx]``. The two are not interchangeable and nothing raises
when they are swapped -- a farmer at ``[7, 1]`` unpacked as ``(y, x)`` writes
its plane at tile ``(1, 7)`` and reads its trunk column there too, so the
per-unit head scores the mirrored tile against the real tile's label and trains
perfectly happily on the wrong board. It did, until
``test_the_gathered_tile_is_the_tile_the_engine_acts_on`` was written. Every
unpack of a position in this module is therefore ``(x, y)``, and the tile loops
in ``encode_board`` -- which walk ``farm["tiles"]`` directly and so are already
in ``(y, x)`` -- are the one place that is not.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, cast

import torch

from kaggriculture import action_codec
from kaggriculture.constants import (
    ANIMALS,
    BOARD_SIZE,
    CROPS,
    EPISODE_STEPS,
    LAND_ORDER,
    MARKET_PARAMS,
    SEASON_DAYS,
    SHED_CAPACITY,
    SHOPS,
    TURNS_PER_DAY,
)
from kaggriculture.observation import Tile

BOARD = BOARD_SIZE

# Re-export the shared action boundary during the compatibility release.
ANIMAL_NAMES = action_codec.ANIMAL_NAMES
CROP_NAMES = action_codec.CROP_NAMES
HIRE_SLOT = action_codec.HIRE_SLOT
ITEM_VERBS = action_codec.ITEM_VERBS
LAND_SLOT = action_codec.LAND_SLOT
MARKET_SLOTS = action_codec.MARKET_SLOTS
MAX_ORDERS = action_codec.MAX_ORDERS
MAX_TRANSFER = action_codec.MAX_TRANSFER
PRODUCT_NAMES = action_codec.PRODUCT_NAMES
QUANTITIES = action_codec.QUANTITIES
SHED_NAMES = action_codec.SHED_NAMES
TRANSFER_OPS = action_codec.TRANSFER_OPS
UNIT_OPS = action_codec.UNIT_OPS
bucket_of = action_codec.bucket_of
quantity_of = action_codec.quantity_of
SHOP_NAMES = sorted(SHOPS)

# The main farmer plus enough hands for a full day's hiring. There is no
# engine-enforced cap on hand count: hires_today resets every day and each
# hire only costs mult * fib(hires_today), so a rich-enough farm can keep
# hiring. 13 (farmer + 12 hands, this project's own heuristic policy's
# hiring target) is not a bound on what other agents actually do: sampling
# 240 episodes across the six replay archives on disk turned up a farm with
# 14 hands (15 acting units) in a single day. MAX_UNITS is set with headroom
# above that observed value, and encode_units/encode_positions/decode_units
# raise rather than silently drop a real unit if it is ever exceeded, so a
# wider hand count can never become a misaligned label.
MAX_UNITS = 20

# Per farm, in order: one plane per crop, one per animal, the mutually
# exclusive tile states, the continuous per-tile features, then the unit
# planes. A crop or animal lights its own plane inside a PLANT /
# occupied-structure tile; the state planes cover everything else, including
# a bare (unoccupied) structure.
#
# A bare coop and a bare pasture are separate states, not one "structure".
# BUILD_COOP and BUILD_PASTURE are distinct ops, `_apply_unit_action`'s PLACE
# accepts an animal only onto `ANIMALS[item]["structure"]`, and DIG clears a
# bare structure but refuses an occupied one -- so collapsing the two hid
# which build a tile is already committed to. The kinds come from the rules
# table rather than being typed out, so an animal added upstream with a new
# structure kind adds its own state instead of silently colliding with one.
#
# Occupancy is a plane for the farmer and a plane for the hands rather than
# one shared plane, because the farmer and a hand are different pieces, and
# rather than a flag per unit because several hands may stand on one tile --
# [[4, 3], [4, 3]] occurs in the corpus. The hands plane is therefore a count,
# scaled by MAX_UNITS so it shares the range of the other continuous features.
# CARRIED accumulates for the same reason.
_STRUCTURE_KINDS = tuple(sorted({str(data["structure"]) for data in ANIMALS.values()}))
_TILE_STATES = ("WEED", "LOCKED", "EMPTY") + _STRUCTURE_KINDS
_TILE_FEATURES = (
    "ACTIVE_TODAY",
    "CARED_TODAY",
    "CARE_BONUS",
    "DISTRESS",
    "YIELD_FRACTION",
    "BONUS_READY",
    "AGE",
    "LIFESPAN_LEFT",
)
_UNIT_PLANES = ("FARMER", "HANDS", "CARRIED")

# How much one unit is holding, over the same 28-episode sample the scalar
# divisors were measured on: p50 1, p95 10, p99 14, max 36. Divided by 16, the
# next power of two above the p99, so almost every unit lands inside [0, 1]
# alongside the other continuous planes and a full hand stays legible rather
# than clipped.
UNIT_CARRIED_SCALE = 16.0

# CARE is a unit op with a delayed payoff, and both halves of it are here.
# CARED_TODAY says whether the day's CARE has already landed -- the engine
# no-ops a second one -- and CARE_BONUS is the yield it is working toward,
# banked one per cared-and-fed day and spent on the next production day.
# Measured over 28 episodes spanning all seven archives, both seats, every
# 7th turn: pending_care_bonus p50 2, p95 5, p99 7, max 7. Divided by 8 so the
# whole observed range lands inside [0, 1] like the other tile features.
CARE_BONUS_SCALE = 8.0

# `max_lifespan_step` is the step a one-time crop starts decaying on, and an
# ongoing crop carries -1 until its final scheduled production sets one. -1
# means "no death scheduled", which is the opposite end of this feature from
# "dies now" -- encoding it literally would read (-1 - step) / EPISODE_STEPS
# and slide from 0 to -1 across the season, colliding with the small negative
# values a genuinely decaying plant carries. It gets its own value instead,
# above everything a scheduled death can produce: over the same sample, a
# plant with a real death step reads between -0.014 and 0.431, and 53.6% of
# planted tiles carry the sentinel.
NO_DEATH_SCHEDULED = 1.0

_ANIMAL_BASE = len(CROP_NAMES)
_STATE_BASE = _ANIMAL_BASE + len(ANIMAL_NAMES)
_FEATURE_BASE = _STATE_BASE + len(_TILE_STATES)
_UNIT_BASE = _FEATURE_BASE + len(_TILE_FEATURES)
_PER_FARM_PLANES = _UNIT_BASE + len(_UNIT_PLANES)

TILE_PLANES = 2 * _PER_FARM_PLANES

# Both a price and an inventory signal per product, our money and the
# opponent's, the day/hour/step phase of the season, how much land, town shops
# and hired hands are unlocked on each side, and how many hires each side has
# already made today (the opponent's farm is public, so their counts are as
# knowable as ours).
#
# hires_today is here because it is the only thing that prices the next hire:
# the engine charges fib(hires_today) and resets the counter every day, so the
# hand count alone cannot say whether the next hire costs 1 coin or 987. A
# behaviour clone trained without it played correctly on the teacher's own
# states -- false-positive rate 0.0002 on held-out turns -- and then hired ~2
# units on every turn of its own episodes, spending its opening 3,000 down to
# zero by turn 10 and losing all 500 games. The teacher hires on 14.6% of turns
# and only in the first four hours of a day.
#
# Then our own private state, which no plane carries: the shed SELL draws from,
# the seeds PLANT consumes, and what our units are carrying between the field
# and the shed. And a flag per town shop rather than only the count of them --
# each shop consumes a specific list of products every few turns, so *which*
# shops are open is what decides where the demand actually is.
SCALARS = (
    2 * len(PRODUCT_NAMES)  # market price and inventory, per product
    + 12  # money, phase, land, shop count, hands and hires, both seats
    + len(SHED_NAMES)  # our shed, per product and per animal awaiting placement
    + len(CROP_NAMES)  # our unplanted seeds, per crop
    + len(PRODUCT_NAMES)  # carried across our units, per product
    + len(SHOP_NAMES)  # which shops the town has opened
)

# Divisors for the private counts, so each shares the roughly-unit range the
# rest of this vector sits in. Measured over 28 episodes spanning all seven
# archives, both seats, every 7th turn -- a stride coprime with the 24-turn
# day, because sampling on the hour reads every inventory immediately after
# the end-of-day drop has emptied it and reports carried produce as 0 always:
#
#   shed, per product           p50 0, p95 16, p99 42, max 96
#   seeds, per crop             p50 0, p95 11, p99 29, max 219
#   carried total, per product  p50 0, p95 16, p99 28, max 78
#
# The shed divides by the engine's own SHED_CAPACITY, which is why no named
# constant appears for it here: PLACE and the end-of-day drop both refuse to
# push the shed's total past it, and the measured maximum sits just under it.
# (BUY_PRODUCT and BUY_ANIMAL credit the shed without that check, so it is a
# near-bound rather than a hard one.) Seeds and carried produce have no engine
# cap at all, so their divisors are set at the measured p99: 99% of rows land
# inside [0, 1] and the tail stays legible rather than clipped. This is the
# `len(hands) / 8.0` mistake not repeated -- that divisor is exceeded by 59% of
# training rows and carries nothing to say whether that was ever intended.
SEED_SCALE = 32.0
CARRIED_SCALE = 32.0


@dataclass(frozen=True)
class BeliefTarget:
    """Normalized opponent-private labels kept outside policy observations."""

    shed: torch.Tensor
    seeds: torch.Tensor
    carried: torch.Tensor

    @property
    def tensor(self) -> torch.Tensor:
        """Return the fixed shed, seed, carried target schema as one vector."""
        return torch.cat((self.shed, self.seeds, self.carried))


BELIEF_TARGET_SIZE = len(SHED_NAMES) + len(CROP_NAMES) + len(PRODUCT_NAMES)


def encode_private_belief_target(observation: Mapping[str, Any]) -> BeliefTarget:
    """Encode one seat's own simulator-private state as normalized labels."""
    private = cast(Mapping[str, Any], observation["private"])
    shed = cast(Mapping[str, float], private["shed"])
    seeds = cast(Mapping[str, float], private["seeds"])
    carried: Counter[str] = Counter()
    for inventory in cast(list[Mapping[str, int]], private["inventories"]):
        carried.update(inventory)
    return BeliefTarget(
        shed=torch.tensor(
            [shed[name] / SHED_CAPACITY for name in SHED_NAMES], dtype=torch.float32
        ),
        seeds=torch.tensor(
            [seeds[name] / SEED_SCALE for name in CROP_NAMES], dtype=torch.float32
        ),
        carried=torch.tensor(
            [carried[name] / CARRIED_SCALE for name in PRODUCT_NAMES],
            dtype=torch.float32,
        ),
    )


_MAX_QUADRANTS = 1 + len(LAND_ORDER)

# Every key this module reads, by the mapping it appears in. This exists so
# that an unencoded field has to be an argued decision rather than an absence:
# the shed, the seeds and the carried inventories went unencoded for a whole
# training run precisely because a field nobody encodes is a field nobody sees.
# ``test_every_field_the_corpus_carries_is_either_encoded_or_argued_away``
# sweeps every archive and fails on any key that appears in neither this table
# nor NOT_ENCODED, so an upstream addition is loud in CI instead of silent in
# a shard.
#
# "tile" is the union over every tile kind -- PLANT, WEED and the bare and
# occupied structures each carry a subset -- because they share one block of
# planes and one dispatch in ``_write_tile``.
ENCODED_FIELDS: dict[str, frozenset[str]] = {
    "observation": frozenset(
        {"day", "hour", "player", "farms", "market", "town", "private"}
    ),
    "farm": frozenset(
        {"tiles", "farmer", "hands", "money", "unlocked_quadrants", "hires_today"}
    ),
    "market": frozenset({"prices", "inventory"}),
    "town": frozenset({"unlocked_shops"}),
    "private": frozenset({"shed", "seeds", "inventories"}),
    "tile": frozenset(
        {
            "kind",
            "crop",
            "animal",
            "planted_day",
            "placed_day",
            "watered_today",
            "fed_today",
            "cared_today",
            "pending_care_bonus",
            "consecutive_unwatered",
            "consecutive_unfed",
            "yield_units",
            "fertilized_until_day",
            "fertilizer_available",
            "max_lifespan_step",
        }
    ),
}

# The two keys a real observation carries that nothing here reads, each on
# purpose.
#
# `step` is derived from `day` and `hour` instead: kaggle_environments' core
# loop only ever writes it onto agent 0's observation, so a real seat-1
# observation has no `step` key at all and reading it would encode one seat
# and not the other. See `encode_scalars`.
#
# `remainingOverageTime` is not game state. It is how much of the compute
# budget is left, it moves with whatever machine the episode ran on, and the
# corpus was produced on other people's hardware. Its distribution at
# inference has nothing to do with its distribution in training, so a model
# that learned anything from it learned about the runner rather than the game.
NOT_ENCODED = frozenset({"step", "remainingOverageTime"})


def encode_board(observation: Mapping[str, Any], seat: int) -> torch.Tensor:
    """Return the board planes for one seat.

    ``seat``'s own farm occupies the first ``TILE_PLANES // 2`` planes and the
    opponent's farm occupies the rest. Keeping the two farms in separate
    planes, rather than overlaying them on one shared set, means a crop on the
    opponent's board can never silently overwrite one of ours at the same
    coordinate.

    Only the first block gets a carried-inventory plane. ``private`` belongs to
    whoever's observation this is, so the opponent's inventories are not
    knowable and their plane stays zero -- meaning *unknown* there, not
    *empty*.

    The step is derived from ``day`` and ``hour`` for the same reason
    ``encode_scalars`` derives it: ``observation["step"]`` exists only on seat
    0's observation.

    Args:
        observation: One turn's observation.
        seat: Which player's farm goes in the first block of planes.

    Returns:
        A ``(1, TILE_PLANES, BOARD, BOARD)`` float32 tensor.
    """
    day = observation["day"]
    step = day * TURNS_PER_DAY + observation["hour"]
    farms = observation["farms"]
    planes = torch.zeros(1, TILE_PLANES, BOARD, BOARD, dtype=torch.float32)
    for block, farm in enumerate((farms[seat], farms[1 - seat])):
        base = block * _PER_FARM_PLANES
        for y, row in enumerate(farm["tiles"]):
            for x, tile in enumerate(row):
                _write_tile(planes, base, tile, y, x, day, step)
        _write_units(planes, base, farm)
    _write_carried(planes, farms[seat], observation["private"]["inventories"])
    return planes


def _write_tile(
    planes: torch.Tensor, base: int, tile: Tile, y: int, x: int, day: int, step: int
) -> None:
    """Write one farm's tile into its block of planes, in place."""
    if tile is None:
        planes[0, base + _STATE_BASE + _TILE_STATES.index("EMPTY"), y, x] = 1.0
        return
    if tile == "LOCKED":
        planes[0, base + _STATE_BASE + _TILE_STATES.index("LOCKED"), y, x] = 1.0
        return
    kind = tile["kind"]
    if kind == "WEED":
        planes[0, base + _STATE_BASE + _TILE_STATES.index("WEED"), y, x] = 1.0
        return
    if kind == "PLANT":
        crop = tile["crop"]
        death = tile["max_lifespan_step"]
        planes[0, base + CROP_NAMES.index(crop), y, x] = 1.0
        _write_features(
            planes,
            base,
            y,
            x,
            active_today=tile["watered_today"],
            cared_today=False,
            care_bonus=0,
            distress=tile["consecutive_unwatered"],
            yield_units=tile["yield_units"],
            yield_capacity=int(CROPS[crop]["max_yield"]),
            bonus_ready=tile["fertilized_until_day"] >= day,
            age=day - tile["planted_day"],
            lifespan_left=(
                NO_DEATH_SCHEDULED if death < 0 else (death - step) / EPISODE_STEPS
            ),
        )
        return
    animal = tile.get("animal")
    if animal is None:
        planes[0, base + _STATE_BASE + _TILE_STATES.index(kind), y, x] = 1.0
        return
    planes[0, base + _ANIMAL_BASE + ANIMAL_NAMES.index(animal), y, x] = 1.0
    _write_features(
        planes,
        base,
        y,
        x,
        active_today=tile["fed_today"],
        cared_today=tile["cared_today"],
        care_bonus=tile["pending_care_bonus"],
        distress=tile["consecutive_unfed"],
        yield_units=tile["yield_units"],
        yield_capacity=int(ANIMALS[animal]["max_held"]),
        bonus_ready=tile["fertilizer_available"],
        age=day - tile["placed_day"],
        lifespan_left=0.0,
    )


def _write_features(
    planes: torch.Tensor,
    base: int,
    y: int,
    x: int,
    *,
    active_today: bool,
    cared_today: bool,
    care_bonus: int,
    distress: int,
    yield_units: int,
    yield_capacity: int,
    bonus_ready: bool,
    age: int,
    lifespan_left: float,
) -> None:
    """Write the continuous features shared by plants and animals, in place.

    A plant's ``watered_today`` / ``consecutive_unwatered`` and an animal's
    ``fed_today`` / ``consecutive_unfed`` are the same shape of feature (did
    the day's required care happen, how many days running has it been
    missed), so they share one set of planes rather than four kind-specific
    ones.

    The remaining three are one-sided, and the side that does not have them
    writes a zero rather than getting planes of its own: only an animal can be
    ``CARE``d, so a plant writes ``cared_today=False`` and ``care_bonus=0``,
    and only a plant has a scheduled death, so an animal writes
    ``lifespan_left=0.0``. The crop and animal planes already say which kind of
    tile this is, so the shared zero is never ambiguous about what it means.

    The features are keyword-only. There are eight of them, several adjacent
    and same-typed, and a positional swap between two of those -- ``distress``
    for ``care_bonus``, say -- would train the model on a different game
    without anything raising.
    """
    offset = base + _FEATURE_BASE
    planes[0, offset + _TILE_FEATURES.index("ACTIVE_TODAY"), y, x] = float(active_today)
    planes[0, offset + _TILE_FEATURES.index("CARED_TODAY"), y, x] = float(cared_today)
    planes[0, offset + _TILE_FEATURES.index("CARE_BONUS"), y, x] = (
        care_bonus / CARE_BONUS_SCALE
    )
    planes[0, offset + _TILE_FEATURES.index("DISTRESS"), y, x] = distress / 2.0
    planes[0, offset + _TILE_FEATURES.index("YIELD_FRACTION"), y, x] = (
        yield_units / yield_capacity
    )
    planes[0, offset + _TILE_FEATURES.index("BONUS_READY"), y, x] = float(bonus_ready)
    planes[0, offset + _TILE_FEATURES.index("AGE"), y, x] = age / SEASON_DAYS
    planes[0, offset + _TILE_FEATURES.index("LIFESPAN_LEFT"), y, x] = lifespan_left


def _write_units(planes: torch.Tensor, base: int, farm: Mapping[str, Any]) -> None:
    """Write one farm's unit occupancy into its block of planes, in place.

    The hands plane accumulates rather than sets: several hands may stand on
    one tile, and a flag would report a crowd of five the same as a lone hand.

    Positions are ``[x, y]`` and the planes are indexed ``[y, x]``, per the
    module docstring.
    """
    farmer_x, farmer_y = farm["farmer"]
    planes[0, base + _UNIT_BASE + _UNIT_PLANES.index("FARMER"), farmer_y, farmer_x] = (
        1.0
    )
    hands = base + _UNIT_BASE + _UNIT_PLANES.index("HANDS")
    for x, y in farm["hands"]:
        planes[0, hands, y, x] += 1.0 / MAX_UNITS


def _write_carried(
    planes: torch.Tensor, farm: Mapping[str, Any], inventories: list[Mapping[str, int]]
) -> None:
    """Write what each of our units is holding, at the tile it stands on, in place.

    Written into the first farm block only, and by this function rather than by
    ``_write_units``, because it is the one per-unit fact that is not public:
    the opponent's ``private`` is hidden, so their plane is left at zero and a
    zero there means *unknown*. Passing their farm through the same code path
    would invite someone to reach for a ``private`` that is not theirs to read.

    Accumulates for the same reason the hands plane does -- several units share
    a tile routinely, and one carrying nothing next to one carrying nine is not
    the same tile as two empty-handed units.

    ``inventories`` is indexed exactly as the units are, ``[farmer, *hands]``
    -- the same order ``encode_positions`` and ``encode_units`` use -- because
    the engine's ``_do_hire`` appends an inventory as it appends a hand. It is
    read by unit index rather than zipped so that a list shorter than the crew
    raises here instead of silently dropping the last hand's load. A longer one
    is fine and does occur: ``_farmer_inventory`` grows the list when an action
    orders a hand the farm does not have, and that entry belongs to no unit and
    stays empty, since the engine no-ops the op that created it.

    Positions are ``[x, y]`` and the planes are indexed ``[y, x]``, per the
    module docstring.
    """
    offset = _UNIT_BASE + _UNIT_PLANES.index("CARRIED")
    for index, (x, y) in enumerate([farm["farmer"], *farm["hands"]]):
        planes[0, offset, y, x] += sum(inventories[index].values()) / UNIT_CARRIED_SCALE


def encode_scalars(observation: Mapping[str, Any], seat: int) -> torch.Tensor:
    """Return the non-spatial features for one seat.

    Both a price and an inventory signal are kept per product: the price says
    what the next unit fetches right now, and the inventory says how far
    already-committed supply -- ours or the opponent's -- has pushed it from
    baseline, which is what decides where the price goes next. Land, town
    shops and hired hands are included for both seats, since the opponent's
    farm is public.

    Then our own private state, which nothing else in this module carries into
    the market branch: the shed, because ``SELL`` is the engine's only
    money-increasing operation and it draws from there; the unplanted seeds,
    because ``PLANT`` consumes one and the engine drops every ``PLANT`` for a
    crop whose seeds a turn overspends; and the produce our units are carrying,
    which is on its way to the shed and is the difference between a sale the
    farm can make this turn and one it cannot. ``observation["private"]`` is
    already this seat's own -- the opponent's is hidden and never in hand --
    so it is read directly and never indexed by seat.

    Which shops the town has opened is a flag each, not only the count that was
    here before. Each shop consumes a specific list of products every few
    turns, so the identities are what say where the demand is; two towns with
    four shops open can be buying disjoint things.

    The global step is derived from ``day`` and ``hour`` rather than read from
    ``observation["step"]``: kaggle_environments' own core loop only ever
    writes ``step`` onto agent 0's observation (``new_state[0].observation.step
    = ...`` in its ``core.py``), so any other seat's real observation has no
    ``step`` key at all. ``day`` and ``hour`` are mirrored onto every agent by
    the game's interpreter, and reconstruct ``step`` exactly, since the
    interpreter itself derives them as ``next_step // turns_per_day`` and
    ``next_step % turns_per_day``.

    Args:
        observation: One turn's observation.
        seat: Which player to encode for.

    Returns:
        A ``(1, SCALARS)`` float32 tensor.
    """
    market = observation["market"]
    prices = market["prices"]
    inventory = market["inventory"]
    farms = observation["farms"]
    ours, theirs = farms[seat], farms[1 - seat]
    private = observation["private"]
    unlocked_shops = observation["town"]["unlocked_shops"]

    # Summed across our units rather than read per unit: the market decision
    # is about the farm's whole holding, and the board already carries who is
    # standing where with what. Inventories are sparse -- the engine deletes an
    # item's key the moment its count reaches zero -- so this accumulates what
    # is there instead of reading a key per product that may not exist.
    carried: dict[str, int] = {}
    for held in private["inventories"]:
        for item, count in held.items():
            carried[item] = carried.get(item, 0) + count

    values = [
        (prices[item] - MARKET_PARAMS[item]["base"]) / MARKET_PARAMS[item]["base"]
        for item in PRODUCT_NAMES
    ]
    values += [
        (inventory[item] - MARKET_PARAMS[item]["I0"]) / MARKET_PARAMS[item]["T"]
        for item in PRODUCT_NAMES
    ]
    values += [
        ours["money"] / 10_000.0,
        theirs["money"] / 10_000.0,
        observation["day"] / SEASON_DAYS,
        observation["hour"] / TURNS_PER_DAY,
        (observation["day"] * TURNS_PER_DAY + observation["hour"]) / EPISODE_STEPS,
        len(unlocked_shops) / len(SHOP_NAMES),
        len(ours["unlocked_quadrants"]) / _MAX_QUADRANTS,
        len(theirs["unlocked_quadrants"]) / _MAX_QUADRANTS,
        len(ours["hands"]) / 8.0,
        len(theirs["hands"]) / 8.0,
        ours["hires_today"] / MAX_UNITS,
        theirs["hires_today"] / MAX_UNITS,
    ]
    values += [private["shed"][item] / SHED_CAPACITY for item in SHED_NAMES]
    values += [private["seeds"][crop] / SEED_SCALE for crop in CROP_NAMES]
    values += [carried.get(item, 0) / CARRIED_SCALE for item in PRODUCT_NAMES]
    values += [float(shop in unlocked_shops) for shop in SHOP_NAMES]
    return torch.tensor(values, dtype=torch.float32).reshape(1, SCALARS)


# torch's cross entropy ignores this index, so padded units contribute no loss.
IGNORE = -100


class TooManyUnitsError(ValueError):
    """Raised when a turn has more acting units than ``MAX_UNITS`` covers.

    Kept distinct from the plain ``ValueError`` that ``_label`` raises for an
    op outside ``UNIT_OPS`` (``tuple.index(x): x not in tuple``), so a caller
    that wants to tolerate an oversized turn -- an empirical, not proven,
    bound -- can catch exactly this and let an unknown-op bug propagate
    instead of being silently counted as the same kind of failure.
    """


def unit_count(observation: Mapping[str, Any], seat: int) -> int:
    """Return how many units act for ``seat`` from this observation.

    The single definition of how many slots are real, so positions and labels
    cannot disagree about it. The observation, not the action, is authoritative:
    the engine walks ``[farmer, *farm["hands"]]`` and its ``_farmer_position``
    returns ``None`` past the end of ``hands``, making any further op in the
    action a silent no-op.

    The counts do diverge, on 2.4% of correctly paired turns overall, and the
    rate depends entirely on which day's archive you look at: 0% across
    2026-07-30 through 08-01, then 1.7%, 8.1% and 4.4% on 08-02 through 08-04.
    Measuring one archive and generalising is how this docstring previously came
    to claim the disagreement did not exist at all. Usually the action carries
    one to five hand ops more than the farm has hands; occasionally, on 08-04,
    one fewer.

    Truncating to ``units`` is safe in both directions. The engine applies unit
    ops before ``_process_market``, where HIRE lands, so a hand hired this turn
    cannot act this turn; surplus ops address units that do not exist yet and
    the engine no-ops them identically. A short action simply leaves the trailing
    slots at ``IGNORE``, training nothing rather than training something wrong.

    That is the secondary reason for this function. The primary one is that
    ``encode_positions`` and ``encode_units`` must agree about how many slots are
    real -- if they drift apart, every unit is trained on another unit's
    surroundings and nothing raises.

    Args:
        observation: One turn's observation.
        seat: Which player's units to count.

    Returns:
        The farmer plus the hands standing on ``seat``'s farm.
    """
    return 1 + len(observation["farms"][seat]["hands"])


def encode_positions(observation: Mapping[str, Any], seat: int) -> torch.Tensor:
    """Return the tile each acting unit occupies, padded to ``MAX_UNITS``.

    Slots are ordered exactly as ``encode_units`` labels them -- slot 0 the
    farmer, slot *k* the *k*-th hand -- because the head reads slot *k*'s trunk
    column at slot *k*'s position and scores it against slot *k*'s label. If
    the two orders diverged, every unit would be trained on another unit's
    surroundings and nothing would raise.

    Padded slots take tile index 0. That is deliberate, not a fallback: the
    head gathers at these indices unconditionally, so every slot needs a valid
    one, and a padded slot's label is ``IGNORE``, so its logits never reach the
    loss and the tile it nominally read is never learned from.

    A unit's position is ``[x, y]`` and ``encode_board`` lays its planes out as
    ``[y][x]``, so the flat index is ``y * BOARD + x``. Unpacking the position
    the other way round returns the mirrored tile, which is a valid index into
    a valid plane and therefore raises nothing: the head simply gathers the
    wrong column and scores it against this unit's label.

    Args:
        observation: One turn's observation -- the state the decision was made
            from, so it must be the same observation the row's board encodes.
        seat: Which player's units to locate.

    Returns:
        A ``(1, MAX_UNITS)`` int64 tensor of flattened ``y * BOARD + x`` tile
        indices.

    Raises:
        TooManyUnitsError: If more units are on the board than ``MAX_UNITS``
            covers.
    """
    units = unit_count(observation, seat)
    if units > MAX_UNITS:
        raise TooManyUnitsError(f"{units} acting units exceeds MAX_UNITS={MAX_UNITS}")
    farm = observation["farms"][seat]
    positions = torch.zeros(1, MAX_UNITS, dtype=torch.int64)
    for index, (x, y) in enumerate([farm["farmer"], *farm["hands"]]):
        positions[0, index] = y * BOARD + x
    return positions


def encode_units(action: Mapping[str, Any], units: int) -> torch.Tensor:
    """Return one label per unit, padded to ``MAX_UNITS``.

    Units that did not act -- hands not yet hired -- are marked with the
    ignore index rather than a ``PASS`` label. Teaching the model that an
    absent hand chose to pass would train it to pass.

    ``units`` comes from ``unit_count`` on the observation the action was taken
    from, and ops past it are dropped rather than labelled. About 5% of corpus
    turns emit ops for hands the farm does not have; the engine no-ops them, so
    labelling them would attach a real op to a slot holding no unit -- and, now
    that the head reads each slot at its unit's tile, would train the readout on
    a padded tile nobody is standing on.

    Args:
        action: One recorded turn's action dict.
        units: How many units are actually on the board this turn.

    Returns:
        A ``(1, MAX_UNITS)`` int64 tensor of labels.

    Raises:
        TooManyUnitsError: If ``units`` exceeds ``MAX_UNITS``.
    """
    if units > MAX_UNITS:
        raise TooManyUnitsError(f"{units} acting units exceeds MAX_UNITS={MAX_UNITS}")
    ops = [action["farmer"], *action["hands"]][:units]
    labels = torch.full((1, MAX_UNITS), IGNORE, dtype=torch.int64)
    for index, op in enumerate(ops):
        labels[0, index] = _label(op)
    return labels


def _label(op: list[Any]) -> int:
    """Return the vocabulary index for one unit's recorded op.

    A recorded ``PICKUP`` or ``PLACE`` may carry a quantity as well as an item,
    and this label drops it: ``["PICKUP", "WHEAT", 2]`` and
    ``["PICKUP", "WHEAT", 1]`` are the same op index, because the op vocabulary
    has no slot for a count. It is not lossy about *which* item, which is what
    makes the op land at all.

    **The count is no longer lost on the way back out.** It used to be: ``_op``
    emitted a hard-coded 1 for every transfer, so a policy could pick up one
    wheat per turn and no more, and loading three wheat for three animals cost
    three turns instead of one. That constant existed because no head predicted
    a count and a fixed n>1 fits nothing -- the corpus modes are item-dependent
    (WHEAT 2, FERTILIZER 3, COW 1) and 10,660 of 11,721 recorded PICKUPs took
    strictly less than the shed held. The per-unit quantity head replaces it:
    ``decode_units`` reads a bucket per unit off the same trunk column the op
    comes from and ``_op`` emits that.

    What remains lossy is this direction alone, and only because the
    behaviour-cloning target is an op label: the quantity head is trained by
    the RL loop, which scores the bucket it sampled, and never by a corpus
    count.

    Args:
        op: One unit's op, e.g. ``["WATER"]``, ``["PLANT", "MELON"]`` or
            ``["PICKUP", "WHEAT", 2]``.

    Returns:
        The op's index into ``UNIT_OPS``.
    """
    verb = str(op[0])
    name = f"{verb}:{op[1]}" if verb in ITEM_VERBS else verb
    return UNIT_OPS.index(name)


def decode_units(
    logits: torch.Tensor,
    quantity_logits: torch.Tensor,
    units: int,
    mask: torch.Tensor,
    quantity_mask: torch.Tensor,
) -> dict[str, Any]:
    """Return the action dict implied by per-unit logits, under the legality mask.

    The mask is required rather than defaulted, and that is the whole point of
    this signature. Selection used to be a bare ``argmax`` over raw logits,
    which meant the deployed agent chose actions by a different rule than the
    one it was trained and gated under -- ``rollout`` samples from
    ``masked_fill(~mask, -inf)`` -- and could name an op the engine silently
    discards. An illegal op is not an error anywhere: ``_apply_unit_action``
    just ``return``s, and the unit has spent its turn. Over 719 turns that is a
    quiet, unmeasurable leak, so there is deliberately no overload of this
    function that can select without a mask.

    Masking before the ``argmax`` rather than after it is what makes the choice
    the *best legal* op instead of a legal fallback: filling the illegal
    positions with ``-inf`` leaves the comparison between the surviving ops
    exactly as the network scored them.

    Selection stays an ``argmax`` -- deployment is deterministic, where
    ``rollout`` samples -- so the same observation and the same weights always
    produce the same turn.

    The quantity is selected the same way, from its own head and under its own
    mask, and is spent only where the chosen op is a transfer. Every unit gets
    a bucket -- the head emits one per slot, transfer or not -- and ``_op``
    discards the ones no verb can read, so a ``WATER`` stays arity 1 however
    the quantity head scored that unit.

    Args:
        logits: A ``(1, MAX_UNITS, len(UNIT_OPS))`` tensor.
        quantity_logits: A ``(1, MAX_UNITS, len(QUANTITIES))`` tensor, the
            per-unit quantity head.
        units: How many units are actually on the board this turn.
        mask: A ``(1, MAX_UNITS, len(UNIT_OPS))`` bool tensor from
            ``learn.mask.unit_mask``, True where the engine would act on the
            op. No row may be entirely False, which ``unit_mask`` guarantees by
            keeping ``PASS`` alive on every slot -- an all-``-inf`` row has no
            meaningful ``argmax``.
        quantity_mask: A ``(1, MAX_UNITS, len(QUANTITIES))`` bool tensor from
            ``learn.mask.unit_quantity_mask``, under the same no-empty-row
            rule.

    Returns:
        An action dict with ``farmer``, ``hands`` and an empty ``market``
        (market orders are decoded elsewhere).

    Raises:
        TooManyUnitsError: If ``units`` exceeds ``MAX_UNITS``.
    """
    if units > MAX_UNITS:
        raise TooManyUnitsError(f"{units} units exceeds MAX_UNITS={MAX_UNITS}")
    legal = logits[0, :units].masked_fill(~mask[0, :units], -torch.inf)
    legal_quantity = quantity_logits[0, :units].masked_fill(
        ~quantity_mask[0, :units], -torch.inf
    )
    selected = action_codec.SelectedActions(
        tuple(int(value) for value in legal.argmax(dim=-1)),
        tuple(int(value) for value in legal_quantity.argmax(dim=-1)),
        tuple(0 for _ in range(len(MARKET_SLOTS) + 2)),
    )
    decoded = action_codec.decode_selected(selected)
    return {"farmer": decoded["farmer"], "hands": decoded["hands"], "market": []}


def transfer_slots(op_indices: torch.Tensor) -> torch.Tensor:
    """Return which sampled op slots hold a ``PICKUP`` or a ``PLACE``.

    The one place the learning path turns sampled op *indices* into the
    question "did this unit spend a quantity". It is a lookup into
    ``TRANSFER_OPS`` and not a comparison against the value the quantity head
    produced: a bucket of 1 is a real transfer of one item and an unspent
    bucket is also 1, and telling them apart by the number would put a
    never-executed decision into the importance ratio.

    ``IGNORE`` clamps to 0, which is ``PASS`` and therefore not a transfer, so
    a padded slot answers False without the caller masking it first.

    Args:
        op_indices: Any-shaped int64 op indices, possibly ``IGNORE``.

    Returns:
        A bool tensor of the same shape, True where the op reads a quantity.
    """
    table = torch.tensor(TRANSFER_OPS, dtype=torch.bool, device=op_indices.device)
    return table[op_indices.clamp(min=0)]


# Every (verb, item) pair the engine's `_process_market` will actually act on.
# SELL covers the whole PRODUCTS catalogue: `_commit_unit` gates a sale only on
# whether the shed holds the item, never on which item it is. BUY_SEED and
# BUY_ANIMAL are gated by `item in CROPS` / `item in ANIMALS`, the whole
# catalogue in each case.
#
# BUY_PRODUCT is narrower. `_process_market` gates it with a literal
# `item in ("WHEAT", "FERTILIZER")` that has nothing to do with the rest of
# PRODUCTS -- CARROT, MELON and the animal products are things a farm grows,
# not things the market will sell back to it. Building this slot from
# `sorted(PRODUCTS)` instead, as the wider catalogue tempts, would add seven
# slots the engine silently no-ops on every turn the model fills them, each
# one crowding a real order out of a ten-order turn.
#
# That gate exists only as an inline literal inside `_process_market`, not as
# a named constant, so it cannot be imported. It is typed by hand here rather
# than parsed out of the engine's source at import time: this module is
# imported by the submitted agent (`kaggriculture.learn.play`), which
# runs inside the competition sandbox against a `kaggle_environments` build
# this project does not control, and `len(MARKET_SLOTS)` sets the market
# head's output shape. A source-parse that raises or silently changes shape
# on a different engine build forfeits the episode with no logs to explain
# why; a hand-typed tuple that has drifted still plays. What keeps this tuple
# honest instead is
# `test_buy_product_s_item_gate_matches_the_engine_exactly`, which parses the
# engine's source at test time, in CI, where a mismatch fails loudly and
# someone can fix it.
# The first thirteen buckets are exact counts: 93.5% of orders in the corpus
# are 12 or fewer, and lumping that dense range into ranges would blur most of
# the distribution the model has to predict. The next four buckets summarize
# 13-64 -- 13-20, 21-32, 33-52, 53-64 -- each labelled by one representative
# quantity, so a big order is still distinguishable from no order without one
# class per order size seen only a handful of times.
#
# The last four buckets (80, 100, 130, 165) are this task's addition, sized
# from a market phase that could only ever fill 64 units of one order: 65 of
# 150 of the newest corpus archive's top-rated, engine-matched episodes (43%)
# had at least one market order the engine filled *above* 64 -- not merely
# requested above it, but actually committed, unit by unit, past the old
# axis. 80 covers the dense cluster the fills sit in (mostly WHEAT sales,
# peaking at 77 -- 26 of the 89 over-64 fills measured); 100 matches
# ``SHED_CAPACITY``, the structural ceiling for every shed-bound role (SELL,
# BUY_PRODUCT, BUY_ANIMAL can never fill past what the shed can hold); 165 is
# the ceiling, with real headroom past the measured max fill of 86 -- carried
# up from the largest *requested* quantities seen in top play (BUY_SEED is not
# shed-bound, so a request that large is not structurally impossible even
# though this scan's fills never reached it).
def encode_market(action: Mapping[str, Any]) -> torch.Tensor:
    """Return one bucketed label per market slot.

    Unlike ``encode_units``, no market slot is ever ``IGNORE``. A unit slot is
    padding when the farm has no hand standing there yet; a market slot has no
    such state; every slot -- sell this product, buy that seed, hire, buy land
    -- is a genuine decision on every turn, and "trade nothing" is itself the
    meaningful class 0 rather than something to mask out: measured across all six
    archives, the corpus trades nothing on 46.3% of turns. The market loss
    therefore masks nothing, where the unit loss masks every unhired hand's
    padding.

    Repeated orders for the same (verb, item) pair are summed before
    bucketing -- 22% of order-bearing turns repeat a pair, and keeping only
    the last one would discard a real sale or purchase the teacher made.
    ``HIRE`` carries no item and is counted rather than summed, since one
    ``HIRE`` order hires exactly one hand. ``BUY_LAND`` collapses to a flag,
    since the engine performs it at most once per turn regardless of how many
    ``BUY_LAND`` orders are queued (``_do_buy_land`` is called once per
    occurrence, but a second call the same turn is a no-op once the quadrant
    is already unlocked).

    Args:
        action: One recorded turn's action dict.

    Returns:
        A ``(1, len(MARKET_SLOTS) + 2)`` int64 tensor of bucket indices.
    """
    totals = [0] * len(MARKET_SLOTS)
    hires = 0
    land = 0
    for order in action["market"]:
        verb = order[0]
        if verb == "HIRE":
            hires += 1
        elif verb == "BUY_LAND":
            land = 1
        else:
            totals[MARKET_SLOTS.index((verb, order[1]))] += int(order[2])
    labels = [bucket_of(total) for total in totals] + [bucket_of(hires), land]
    return torch.tensor(labels, dtype=torch.int64).reshape(1, len(MARKET_SLOTS) + 2)


def decode_market(logits: torch.Tensor, mask: torch.Tensor) -> list[list[Any]]:
    """Return the market orders implied by per-slot logits, under the legality mask.

    The mask is required for the same reason it is on ``decode_units``: the
    deployed agent must select by the rule it was trained under, and an order
    the engine will not fill is silently dropped rather than refused.

    What the mask can promise here is weaker than on the unit head, and the
    difference is worth stating rather than discovering. ``unit_mask`` gates
    each unit against a resource -- a tile, an inventory -- that no other unit
    is spending, so a per-slot mask makes the whole action legal.
    ``market_mask`` gates each slot against the farm's *whole* balance, and the
    slots then spend from that one balance together, so ten individually
    affordable orders can still overdraw and ``_commit_unit`` will abort the
    later ones. Masking per slot therefore removes the orders that could never
    have filled, not every order that will not fill.

    In the other direction the mask is slightly conservative, and deliberately
    so: it prices every purchase against the balance *before* this turn's
    sales, while the orders below are emitted sells-first precisely so a sale
    can fund a purchase. A buy that only becomes affordable mid-turn is
    therefore masked off. That costs an order; permitting it would cost a
    ``_commit_unit`` abort and the orders queued behind it.

    Orders are emitted in ``MARKET_SLOTS`` order, so every ``SELL`` precedes
    every ``BUY_*`` since the ``SELL`` block is built first. The engine
    processes one turn's orders in list order, one unit of quantity at a time
    per order before moving to the next, so a sale queued ahead of a purchase
    can fund it; queued the other way, the purchase would be evaluated against
    money the sale had not yet raised. ``HIRE`` orders follow, repeated by
    count, then ``BUY_LAND``.

    The result is truncated to ``MAX_ORDERS``: the engine silently drops
    anything past ``maxMarketOrdersPerTurn``, so emitting more here would
    never reach the game.

    Args:
        logits: A ``(1, len(MARKET_SLOTS) + 2, len(QUANTITIES))`` tensor.
        mask: A ``(1, len(MARKET_SLOTS) + 2, len(QUANTITIES))`` bool tensor
            from ``learn.mask.market_mask``, True where the engine would fill
            the whole order. No row may be entirely False, which
            ``market_mask`` guarantees by keeping bucket 0 -- "trade nothing",
            which emits no order at all -- alive on every slot.

    Returns:
        A list of order lists, e.g. ``["SELL", "WHEAT", 4]`` or ``["HIRE"]``.
    """
    buckets = logits[0].masked_fill(~mask[0], -torch.inf).argmax(dim=-1)
    return action_codec.market_orders_of(tuple(int(bucket) for bucket in buckets))
