"""Adversarial differential campaign for the two-seat market coupling resolver.

``kaggriculture.sim.market`` resolves a market slot on a tensor quantity axis:
it quotes every candidate unit-round at once and reads the filled prefix off
cumulative costs. The reference engine instead walks unit-rounds one at a time,
and when both seats trade the same product the book they quote against moves
under both of them together. The vectorised code reproduces that by *computing*
what the lockstep would do -- a first pass while both seats are still trading,
a cutoff at whichever seat is exhausted first, and a second pass for the
survivor alone.

That two-pass handover is a proof, not a transcription. The rule tests in
``test_rules_market.py`` reach it with hand-built pairs, and the episode
campaign in ``test_episodes.py`` reaches it only when a sampled policy happens
to put both seats on one product with unequal quantities. Neither is a search.
This module is the search: it fuzzes the vectorised phase against
``tests/sim/reference_market.py`` -- the pre-vectorisation 65-round scan, which
was itself validated against the real engine -- over states built to sit on the
resolver's edges, and compares every field of the resulting state rather than a
summary.

The generators are deliberate, not merely random. Each round supplies books
already at the price floor (where a sale stops adding supply and the climb
clamps), seats too poor to fill what they asked for (where the cumulative-cost
prefix truncates), sheds at or beside capacity (where the room cap binds),
quantities drawn across the whole decoder bucket range, and -- the case this
module exists for -- both seats ordering the same product in one slot with
lopsided quantities, so one seat dies first and the handover has to fire.
``COVERAGE_FLOOR`` asserts those cases actually occurred, so a generator that
silently stops producing them fails loudly instead of passing vacuously.

The default size is 8 rounds of 512 environments -- 81,920 order slots, about
eleven seconds of work -- because the CI workflow runs every ``slow`` test on
every push and the oracle is a Python scan. Cost is almost entirely per round,
since the batch axis stays vectorised in both implementations, so widen with
``--sim-market-batch`` before lengthening with ``--sim-market-rounds``.
"""

import logging
from dataclasses import fields

import pytest
import torch

from kaggriculture.constants import MAX_MARKET_ORDERS_PER_TURN, SHED_CAPACITY
from kaggriculture.learn.encoding import QUANTITIES
from kaggriculture.sim.engine import (
    ORDER_BUY_PRODUCT,
    ORDER_SELL,
    MarketActions,
)
from kaggriculture.sim.market import apply_market_phase
from kaggriculture.sim.pricing import floor_levels
from kaggriculture.sim.state import (
    CROP_NAMES,
    PRODUCT_NAMES,
    SHED_NAMES,
    SimState,
    empty_state,
)
from kaggriculture.sim.tables import (
    BUY_PRODUCTS,
    ORDER_ITEMS,
    ORDER_PRODUCT,
    ORDER_ROLE,
    ORDER_TYPES,
    ROLE_BUY_PRODUCT,
    ROLE_SELL,
    market_orders,
)
from tests.sim.reference_market import apply_market_phase as reference_market_phase

DEVICE = torch.device("cpu")
SLOTS = MAX_MARKET_ORDERS_PER_TURN

# The oracle scans a fixed 65 unit-rounds and asserts nothing exceeds it, so the
# campaign requests at most the largest decoder bucket -- which is also the
# width of the tensor quantity axis that replaced the scan.
MAX_QUANTITY = max(QUANTITIES)

# A seat below the cheapest quote in the game cannot fill anything it is
# charged for, which is where the cumulative-cost prefix truncates to zero.
BROKE = 100

# How often each adversarial case must occur, per ten thousand order slots,
# before the campaign is allowed to pass. Rates rather than counts, so the
# check means the same thing at every campaign size and does not have to be
# retuned when CI scales the run. The generators currently produce roughly
# twice each figure; the gap is deliberate slack, and the point is that a
# generator which drifts away from a case fails loudly instead of reporting a
# vacuous green.
COVERAGE_FLOOR = {
    "coupled": 1_000,
    "handover": 1_000,
    "coupled_at_floor": 300,
    "cross_role": 300,
    "broke_seat": 150,
    "full_shed": 100,
}
COVERAGE_SCALE = 10_000


def adversarial_state(batch: int, generator: torch.Generator) -> SimState:
    """Build a batch of states sitting on the market resolver's edges.

    Args:
        batch: Number of environments to build.
        generator: Seeded generator, so a failure is reproducible.

    Returns:
        A state whose money, shed, and book fields are biased towards the
        truncating, clamping, and capacity-bound cases rather than the middle
        of their ranges.
    """
    state = empty_state(batch, DEVICE)
    floors = floor_levels(DEVICE)

    def rand(high: int, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randint(0, high, shape, generator=generator)

    seats = (batch, 2)
    state.money.copy_(rand(4_000, seats))
    # A third of the seats cannot afford much, which is where the cumulative
    # cost prefix stops short of the requested quantity.
    broke = rand(3, seats) == 0
    state.money.copy_(torch.where(broke, rand(BROKE, seats), state.money))

    state.shed.copy_(rand(9, (*seats, len(SHED_NAMES))).to(torch.int16))
    over = state.shed.sum(dim=-1) > SHED_CAPACITY - 4
    state.shed.copy_(
        torch.where(over[..., None], torch.zeros_like(state.shed), state.shed)
    )
    # A quarter of the seats sit within two units of the shed cap, so the room
    # limit binds part-way through a purchase instead of never.
    tight = rand(4, seats) == 0
    headroom = SHED_CAPACITY - state.shed.sum(dim=-1) - rand(3, seats)
    state.shed[..., 0].copy_(
        torch.where(
            tight, state.shed[..., 0] + headroom.to(torch.int16), state.shed[..., 0]
        )
    )
    state.shed.clamp_(min=0)
    state.seeds.copy_(rand(5, (*seats, len(CROP_NAMES))).to(torch.int16))

    books = (batch, len(PRODUCT_NAMES))
    # Half the books start on the price floor, where a sale no longer adds
    # supply and the lockstep climb has to clamp rather than keep rising.
    at_floor = rand(2, books) == 0
    state.inventory.copy_(
        torch.where(
            at_floor,
            floors[None, :] + rand(7, books) - 3,
            10_000 + rand(400, books) - 200,
        )
    )
    state.quadrants.copy_((1 + rand(4, seats)).to(torch.int8))
    state.hand_count.copy_(rand(3, seats).to(torch.int16))
    state.hires_today.copy_(rand(6, seats).to(torch.int16))
    state.alive[..., 0] = True
    state.unit_x[..., 0] = 4
    state.unit_y[..., 0] = 4
    state.kind.copy_(rand(2, state.kind.shape).to(torch.int8))
    state.done.copy_(rand(8, (batch,)) == 0)
    return state


def adversarial_orders(batch: int, generator: torch.Generator) -> MarketActions:
    """Build market orders that force both seats onto one product repeatedly.

    Every order type is reachable, including the malformed one, and quantities
    are drawn from the decoder's own bucket table as well as uniformly across
    the axis. Half the slots are then overwritten with a coupled pair: both
    seats trade the same product, one of them selling and -- where the product
    is buyable at all -- the other buying, with lopsided quantities so the pair
    separates part-way through the slot.

    Args:
        batch: Number of environments to build.
        generator: Seeded generator, so a failure is reproducible.

    Returns:
        Orders for both seats across all ten slots.
    """
    shape = (batch, 2, SLOTS)
    slot = (batch, 1, SLOTS)

    def rand(high: int, dims: tuple[int, ...]) -> torch.Tensor:
        return torch.randint(0, high, dims, generator=generator)

    orders = MarketActions.empty(batch, DEVICE)
    orders.order_type.copy_(rand(ORDER_TYPES, shape).to(torch.int8))
    silent = rand(3, shape) == 0
    orders.order_type.copy_(
        torch.where(silent, torch.zeros_like(orders.order_type), orders.order_type)
    )
    # Item codes span the absent marker through one past the catalogue, so
    # out-of-range codes are exercised alongside legal ones.
    orders.order_item.copy_((rand(ORDER_ITEMS + 1, shape) - 1).to(torch.int8))
    buckets = torch.tensor(QUANTITIES)[rand(len(QUANTITIES), shape)]
    orders.order_qty.copy_(
        torch.where(rand(2, shape) == 0, buckets, rand(MAX_QUANTITY + 1, shape)).to(
            torch.int32
        )
    )

    # Coupled slots. ``buyable`` is the product index each buy-product item
    # names; a seat can only buy what the reference sells back, so cross-role
    # pairs exist only on those two products.
    buyable = torch.tensor([PRODUCT_NAMES.index(name) for name in BUY_PRODUCTS])
    reverse = torch.full((len(PRODUCT_NAMES),), -1, dtype=torch.int64)
    reverse[buyable] = torch.arange(len(BUY_PRODUCTS))
    paired = rand(2, slot) == 0
    product = torch.where(
        rand(2, slot) == 0,
        buyable[rand(len(BUY_PRODUCTS), slot)],
        rand(len(PRODUCT_NAMES), slot),
    )
    buying = paired & (rand(2, shape) == 0) & (reverse[product] >= 0)
    selling = paired & ~buying
    orders.order_type.copy_(
        torch.where(
            buying,
            torch.full_like(orders.order_type, ORDER_BUY_PRODUCT),
            torch.where(
                selling,
                torch.full_like(orders.order_type, ORDER_SELL),
                orders.order_type,
            ),
        )
    )
    orders.order_item.copy_(
        torch.where(
            buying,
            reverse[product].clamp_min(0).to(torch.int8),
            torch.where(selling, product.to(torch.int8), orders.order_item),
        )
    )

    # Three quarters of the coupled slots get one short seat and one long one,
    # so the short seat is exhausted first and the resolver has to hand the
    # book over to the survivor mid-slot.
    lopsided = paired & (rand(4, slot) > 0)
    short_seat = rand(2, slot)
    seat_index = torch.arange(2)[None, :, None]
    orders.order_qty.copy_(
        torch.where(
            lopsided & (seat_index == short_seat),
            (1 + rand(6, shape)).to(torch.int32),
            torch.where(
                lopsided,
                (MAX_QUANTITY // 2 + rand(MAX_QUANTITY // 2 + 1, shape)).to(
                    torch.int32
                ),
                orders.order_qty,
            ),
        )
    )
    return orders


def coverage(state: SimState, orders: MarketActions) -> dict[str, int]:
    """Count how many order slots reach each adversarial case.

    Args:
        state: The state the slots are applied to.
        orders: The orders for all ten slots.

    Returns:
        A count per ``COVERAGE_FLOOR`` key, summed over seats and slots.
    """
    rules = market_orders(DEVICE)
    order_type = orders.order_type.to(torch.int64)
    code = (orders.order_item.to(torch.int64) + 1).clamp(0, ORDER_ITEMS - 1)
    detail = rules[order_type.clamp(0, ORDER_TYPES - 1), code]
    role = detail[..., ORDER_ROLE]
    product = detail[..., ORDER_PRODUCT]
    trading = (
        ((role == ROLE_SELL) | (role == ROLE_BUY_PRODUCT))
        & (orders.order_qty > 0)
        & ~state.done[:, None, None]
    )
    both = trading & trading.flip(1) & (product == product.flip(1))
    handover = both & (orders.order_qty != orders.order_qty.flip(1))
    floors = floor_levels(DEVICE)
    at_floor = state.inventory >= floors[None, :]
    return {
        "coupled": int(both.sum()) // 2,
        "handover": int(handover.sum()) // 2,
        "coupled_at_floor": int(
            (both & at_floor.gather(1, product.flatten(1)).view_as(both)).sum()
        )
        // 2,
        "cross_role": int((both & (role != role.flip(1))).sum()) // 2,
        "broke_seat": int((state.money < BROKE).sum()),
        "full_shed": int((state.shed.sum(dim=-1) >= SHED_CAPACITY - 2).sum()),
    }


def assert_states_identical(
    expected: SimState, actual: SimState, orders: MarketActions, label: str
) -> None:
    """Assert every state field matches, localizing the first divergent field.

    Args:
        expected: State produced by the pre-vectorisation oracle.
        actual: State produced by the vectorised phase.
        orders: The orders both were given, reported for the failing row.
        label: Identifier of the failing round, so it can be replayed.

    Raises:
        AssertionError: If any field of any environment differs.
    """
    for field in fields(SimState):
        left = getattr(expected, field.name)
        right = getattr(actual, field.name)
        if torch.equal(left, right):
            continue
        rows = (left != right).flatten(1).any(dim=1).nonzero().flatten()
        row = int(rows[0])
        raise AssertionError(
            f"{label}: {field.name} differs in {len(rows)} of {left.shape[0]} "
            f"environments; first is row {row}\n"
            f"  expected: {left[row]}\n"
            f"  actual:   {right[row]}\n"
            f"  types:    {orders.order_type[row]}\n"
            f"  items:    {orders.order_item[row]}\n"
            f"  quantity: {orders.order_qty[row]}"
        )


def run_campaign(rounds: int, batch: int, seed: int) -> dict[str, int]:
    """Fuzz the vectorised market phase against the oracle and report coverage.

    Args:
        rounds: Number of independently generated batches.
        batch: Environments per batch.
        seed: Generator seed, so a red run is reproducible from the report.

    Returns:
        The summed coverage counts over every round.

    Raises:
        AssertionError: On the first divergent field, or if a case in
            ``COVERAGE_FLOOR`` was never reached.
    """
    generator = torch.Generator().manual_seed(seed)
    totals = dict.fromkeys(COVERAGE_FLOOR, 0)
    for index in range(rounds):
        state = adversarial_state(batch, generator)
        orders = adversarial_orders(batch, generator)
        for name, count in coverage(state, orders).items():
            totals[name] += count
        assert_states_identical(
            reference_market_phase(state, orders),
            apply_market_phase(state, orders),
            orders,
            f"seed {seed} round {index}",
        )
    slots = rounds * batch * 2 * SLOTS
    logging.info("market coupling campaign: %d order slots, coverage %s", slots, totals)
    for name, floor in COVERAGE_FLOOR.items():
        rate = totals[name] * COVERAGE_SCALE // slots
        assert rate >= floor, (
            f"campaign reached {name} {totals[name]} times, {rate} per "
            f"{COVERAGE_SCALE} order slots, below the {floor} the generators "
            f"are meant to produce; the harness has stopped searching the case "
            f"it exists for"
        )
    return totals


@pytest.mark.slow
def test_market_coupling_campaign_matches_the_pre_vectorisation_scan(
    pytestconfig: pytest.Config,
) -> None:
    """Fuzz the campaign; scale it with ``--sim-market-rounds/--sim-market-batch``."""
    run_campaign(
        rounds=pytestconfig.getoption("--sim-market-rounds"),
        batch=pytestconfig.getoption("--sim-market-batch"),
        seed=20260811,
    )
