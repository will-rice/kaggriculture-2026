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
from kaggriculture import features as canonical_features
from kaggriculture.constants import (
    SHED_CAPACITY,
    SHOPS,
)
from kaggriculture.features import EncodedObservation, encode_observation

BOARD = canonical_features.BOARD

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
MAX_UNITS = canonical_features.MAX_UNITS

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
_STRUCTURE_KINDS = canonical_features.STRUCTURE_KINDS
_TILE_STATES = canonical_features.TILE_STATES
_TILE_FEATURES = canonical_features.TILE_FEATURES
_UNIT_PLANES = canonical_features.UNIT_PLANES

# How much one unit is holding, over the same 28-episode sample the scalar
# divisors were measured on: p50 1, p95 10, p99 14, max 36. Divided by 16, the
# next power of two above the p99, so almost every unit lands inside [0, 1]
# alongside the other continuous planes and a full hand stays legible rather
# than clipped.
UNIT_CARRIED_SCALE = canonical_features.UNIT_CARRIED_SCALE

# CARE is a unit op with a delayed payoff, and both halves of it are here.
# CARED_TODAY says whether the day's CARE has already landed -- the engine
# no-ops a second one -- and CARE_BONUS is the yield it is working toward,
# banked one per cared-and-fed day and spent on the next production day.
# Measured over 28 episodes spanning all seven archives, both seats, every
# 7th turn: pending_care_bonus p50 2, p95 5, p99 7, max 7. Divided by 8 so the
# whole observed range lands inside [0, 1] like the other tile features.
CARE_BONUS_SCALE = canonical_features.CARE_BONUS_SCALE

# `max_lifespan_step` is the step a one-time crop starts decaying on, and an
# ongoing crop carries -1 until its final scheduled production sets one. -1
# means "no death scheduled", which is the opposite end of this feature from
# "dies now" -- encoding it literally would read (-1 - step) / EPISODE_STEPS
# and slide from 0 to -1 across the season, colliding with the small negative
# values a genuinely decaying plant carries. It gets its own value instead,
# above everything a scheduled death can produce: over the same sample, a
# plant with a real death step reads between -0.014 and 0.431, and 53.6% of
# planted tiles carry the sentinel.
NO_DEATH_SCHEDULED = canonical_features.NO_DEATH_SCHEDULED

_ANIMAL_BASE = len(CROP_NAMES)
_STATE_BASE = _ANIMAL_BASE + len(ANIMAL_NAMES)
_FEATURE_BASE = _STATE_BASE + len(_TILE_STATES)
_UNIT_BASE = _FEATURE_BASE + len(_TILE_FEATURES)
_PER_FARM_PLANES = canonical_features.PER_FARM_PLANES

TILE_PLANES = canonical_features.TILE_PLANES

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
SCALARS = canonical_features.SCALARS

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
SEED_SCALE = canonical_features.SEED_SCALE
CARRIED_SCALE = canonical_features.CARRIED_SCALE


@dataclass(frozen=True)
class TorchObservation:
    """Tensor compatibility view of one canonical encoded observation."""

    board: torch.Tensor
    scalars: torch.Tensor
    positions: torch.Tensor
    unit_mask: torch.Tensor
    quantity_mask: torch.Tensor
    market_mask: torch.Tensor


def to_torch(encoded: EncodedObservation) -> TorchObservation:
    """Convert canonical Python storage without recalculating any feature."""
    return TorchObservation(
        board=torch.tensor(encoded.board, dtype=torch.float32).unsqueeze(0),
        scalars=torch.tensor(encoded.scalars, dtype=torch.float32).unsqueeze(0),
        positions=torch.tensor(encoded.positions, dtype=torch.int64).unsqueeze(0),
        unit_mask=torch.tensor(encoded.unit_mask, dtype=torch.bool).unsqueeze(0),
        quantity_mask=torch.tensor(encoded.quantity_mask, dtype=torch.bool).unsqueeze(
            0
        ),
        market_mask=torch.tensor(encoded.market_mask, dtype=torch.bool).unsqueeze(0),
    )


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


# Re-export the canonical schema during the compatibility release.
ENCODED_FIELDS = canonical_features.ENCODED_FIELDS
NOT_ENCODED = canonical_features.NOT_ENCODED
PLANE_INDEX = canonical_features.PLANE_INDEX
PLANE_NAMES = canonical_features.PLANE_NAMES
PER_FARM_PLANES = canonical_features.PER_FARM_PLANES
SCALAR_INDEX = canonical_features.SCALAR_INDEX
SCALAR_NAMES = canonical_features.SCALAR_NAMES


def encode_board(observation: Mapping[str, Any], seat: int) -> torch.Tensor:
    """Return the legacy batched board tensor from one canonical bundle."""
    return to_torch(encode_observation(observation, seat)).board


def encode_scalars(observation: Mapping[str, Any], seat: int) -> torch.Tensor:
    """Return the legacy batched scalar tensor from one canonical bundle."""
    return to_torch(encode_observation(observation, seat)).scalars


# Torch cross entropy ignores this index, so padded units contribute no loss.
IGNORE = -100
TooManyUnitsError = canonical_features.TooManyUnitsError


def unit_count(observation: Mapping[str, Any], seat: int) -> int:
    """Return how many units act for the requested seat."""
    return canonical_features.unit_count(observation, seat)


def encode_positions(observation: Mapping[str, Any], seat: int) -> torch.Tensor:
    """Return the legacy batched position tensor from one canonical bundle."""
    return to_torch(encode_observation(observation, seat)).positions


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


def encode_unit_quantities(action: Mapping[str, Any], units: int) -> torch.Tensor:
    """Return one quantity-bucket label per unit, padded to ``MAX_UNITS``.

    The op vocabulary has no slot for a count -- ``_label`` deliberately drops
    it -- so a clone trained on op labels alone picks up exactly one item per
    turn whatever the head scores, which is the ``_op`` constant that used to
    be hard-coded. This is the label that closes that lane: the same
    ``QUANTITIES`` buckets ``decode_units`` selects from, read off the
    teacher's own transfer counts.

    Only ``PICKUP`` and ``PLACE`` read a count. Every other op is marked
    ``IGNORE`` rather than bucket 0, for the same reason a hand that has not
    been hired is: a ``WATER`` did not *choose* to move nothing, it has no
    count to choose, and scoring it would train the head on slots whose label
    means nothing. Padded slots past ``units`` are ``IGNORE`` too.

    A transfer with no count is bucketed as 1, which is what the engine does
    with it: ``_apply_unit_action`` reads ``int(action[2]) if len(action) >= 3
    else 1`` for both verbs. Reading it as 0 would label a real one-item
    transfer as no transfer at all.

    Args:
        action: One recorded turn's action dict.
        units: How many units are actually on the board this turn.

    Returns:
        A ``(1, MAX_UNITS)`` int64 tensor of bucket indices, ``IGNORE`` where
        no count was chosen.

    Raises:
        TooManyUnitsError: If ``units`` exceeds ``MAX_UNITS``.
    """
    if units > MAX_UNITS:
        raise TooManyUnitsError(f"{units} acting units exceeds MAX_UNITS={MAX_UNITS}")
    ops = [action["farmer"], *action["hands"]][:units]
    labels = torch.full((1, MAX_UNITS), IGNORE, dtype=torch.int64)
    for index, op in enumerate(ops):
        if TRANSFER_OPS[_label(op)]:
            labels[0, index] = bucket_of(int(op[2]) if len(op) >= 3 else 1)
    return labels


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
