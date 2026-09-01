"""Contracts for the fail-closed merge, judged by the engine wherever possible.

The property under test is not that the merge is clever. It is that a residual
which is broken, adversarial, or simply wrong cannot degrade a frozen controller
that was already winning. Every adversarial case below therefore asserts the
same two things: the returned action is the caller's own object, and the object
was not mutated on the way through.

Legality is judged by ``kaggle_environments`` rather than restated here. The
engine's ``_parse_order``, ``_commit_unit``, ``_do_hire`` and ``_do_buy_land``
are the authority on what an order does, and a test that re-derived their rules
would agree with a merge that had copied the same misreading.
"""
# ruff: noqa: D103

import copy
import dataclasses
import importlib
import random
from typing import Any, Mapping, Sequence, cast

import pytest

from kaggriculture.action_codec import (
    MARKET_SLOTS,
    MAX_ORDERS,
    QUANTITIES,
    quantity_of,
)
from kaggriculture.constants import MARKET_PARAMS
from kaggriculture.features import PRODUCT_NAMES, EncodedObservation
from kaggriculture.market_residual.actions import (
    FallbackReason,
    ResidualDecision,
    ResidualMode,
    book_inventory,
    exact_cash,
    kaito_market_buckets,
    merge_residual_action,
    parse_market_orders,
)
from kaggriculture.market_residual.schema import (
    ALLOWED_SLOTS,
    BUCKETS,
    LEARNABLE_VERBS,
)
from tests.market_residual.conftest import Turn

ENGINE = importlib.import_module("kaggle_environments.envs.kaggriculture.kaggriculture")
BOARD_SIZE = 10
SHED_LIMIT = 100


def frozen_orders(action: Mapping[str, Any]) -> list[list[Any]]:
    """Return the orders the residual may never create, cancel or replace."""
    return [order for order in action["market"] if order[0] not in LEARNABLE_VERBS]


def maximal_buckets(
    encoded: EncodedObservation, verb: str | None = None
) -> tuple[int, ...]:
    """Return the largest bucket the canonical mask allows in each slot."""
    return tuple(
        max(bucket for bucket, ok in enumerate(encoded.market_mask[slot]) if ok)
        if verb is None or MARKET_SLOTS[slot][0] == verb
        else 0
        for slot in ALLOWED_SLOTS
    )


def replace_with(buckets: Sequence[int]) -> ResidualDecision:
    """Return a decision asking for those buckets in place of the market."""
    return ResidualDecision(ResidualMode.REPLACE, tuple(buckets))


def engine_fills(
    observation: dict[str, Any], seat: int, queue: Sequence[list[Any]]
) -> list[int]:
    """Return how many units of each queued order the engine actually commits.

    The queue is replayed against a copy of the seat's own real state using the
    engine's parser and commit rules, one seat at a time. The two-seat lockstep
    is the one part not reproduced: it only ever moves prices further, and the
    merge deliberately never counts on a price it cannot see.

    Args:
        observation: The raw engine observation the merge was built from.
        seat: Which seat's queue this is.
        queue: The market orders to replay, in queue order.

    Returns:
        The committed unit count for each order, in the same order.
    """
    farm = copy.deepcopy(observation["farms"][seat])
    private = copy.deepcopy(observation["private"])
    market = copy.deepcopy(observation["market"])
    committed: list[int] = []
    for order in list(queue)[:MAX_ORDERS]:
        parsed = ENGINE._parse_order(order)
        if parsed is None:
            committed.append(0)
            continue
        if parsed["type"] == "HIRE":
            before = farm["hires_today"]
            ENGINE._do_hire(farm, private, BOARD_SIZE)
            committed.append(farm["hires_today"] - before)
            continue
        if parsed["type"] == "BUY_LAND":
            before = len(farm["unlocked_quadrants"])
            ENGINE._do_buy_land(farm, BOARD_SIZE)
            committed.append(len(farm["unlocked_quadrants"]) - before)
            continue
        committed.append(_commit_units(parsed, farm, private, market))
    return committed


def _commit_units(
    parsed: dict[str, Any],
    farm: dict[str, Any],
    private: dict[str, Any],
    market: dict[str, Any],
) -> int:
    """Commit one counted order unit by unit exactly as the engine quotes it."""
    verb, item, done = parsed["type"], parsed["item"], 0
    while parsed["remaining"] > 0:
        if verb == "SELL":
            inventory = market["inventory"][item]
            price = ENGINE.market_price(item, inventory, market.get("params"))
        elif verb == "BUY_PRODUCT":
            inventory = market["inventory"][item]
            price = ENGINE.market_price(item, inventory - 1, market.get("params"))
        elif verb == "BUY_SEED":
            price = ENGINE.CROPS[item]["seed"]
        else:
            price = ENGINE.ANIMALS[item]["cost"]
        if not ENGINE._commit_unit(
            verb, item, price, farm, private, market, SHED_LIMIT
        ):
            break
        parsed["remaining"] -= 1
        done += 1
    return done


def ordered_units(queue: Sequence[list[Any]]) -> list[int]:
    """Return how many units each queued order asks for."""
    return [
        1 if order[0] in ("HIRE", "BUY_LAND") else order[2]
        for order in list(queue)[:MAX_ORDERS]
    ]


def test_exact_cash_is_recovered_from_the_encoding(
    kaito_turns: tuple[Turn, ...],
) -> None:
    banks = {turn.encoded.money() for turn in kaito_turns}
    for turn in kaito_turns:
        assert exact_cash(turn.encoded) == turn.observation["farms"][turn.seat]["money"]
    assert len(kaito_turns) == 1440
    assert len(banks) > 100, "a constant bank would make the recovery untested"


def test_book_inventory_is_recovered_from_the_encoding(
    kaito_turns: tuple[Turn, ...],
) -> None:
    moved = set()
    for turn in kaito_turns:
        for product in PRODUCT_NAMES:
            exact = turn.observation["market"]["inventory"][product]
            assert book_inventory(turn.encoded, product) == exact
            moved.add(exact - int(MARKET_PARAMS[product]["I0"]))
    assert moved != {0}, "an untraded market would make the recovery untested"


def test_controller_quantities_are_read_without_coercion(
    kaito_turns: tuple[Turn, ...],
) -> None:
    parsed = [kaito_market_buckets(turn.action) for turn in kaito_turns]
    assert (parsed.count(None), len(parsed)) == (316, 1440)

    refused = [
        turn
        for turn, buckets in zip(kaito_turns, parsed, strict=True)
        if buckets is None
    ]
    off_vocabulary = [
        turn
        for turn in refused
        if any(
            order[0] in LEARNABLE_VERBS and order[2] not in QUANTITIES
            for order in turn.action["market"]
        )
    ]
    repeated = [
        turn
        for turn in refused
        if len({(o[0], o[1]) for o in turn.action["market"] if o[0] in LEARNABLE_VERBS})
        != len([o for o in turn.action["market"] if o[0] in LEARNABLE_VERBS])
    ]
    assert off_vocabulary, "no turn exercised the off-vocabulary refusal"
    assert repeated, "no turn exercised the repeated-slot refusal"


def test_replaying_the_controllers_own_proposal_reproduces_its_queue(
    kaito_turns: tuple[Turn, ...],
) -> None:
    outcomes: dict[str, int] = {}
    for turn in kaito_turns:
        buckets = kaito_market_buckets(turn.action)
        if buckets is None:
            continue
        result = merge_residual_action(turn.encoded, turn.action, replace_with(buckets))
        reason = "replaced" if result.replaced else result.fallback
        outcomes[str(reason)] = outcomes.get(str(reason), 0) + 1
        if result.replaced:
            merged = sorted(map(tuple, result.action["market"]))
            assert merged == sorted(map(tuple, turn.action["market"]))
    assert outcomes == {"replaced": 860, "mask": 246, "cash": 18}


def test_every_accepted_queue_fills_completely_in_the_engine(
    kaito_turns: tuple[Turn, ...],
) -> None:
    accepted = 0
    for turn in kaito_turns:
        proposals = (kaito_market_buckets(turn.action), maximal_buckets(turn.encoded))
        for buckets in proposals:
            if buckets is None:
                continue
            result = merge_residual_action(
                turn.encoded, turn.action, replace_with(buckets)
            )
            if not result.replaced:
                continue
            accepted += 1
            queue = list(result.action["market"])
            assert engine_fills(turn.observation, turn.seat, queue) == ordered_units(
                queue
            )
    assert accepted == 1410


@pytest.mark.parametrize(
    ("verb", "merges"),
    [("HIRE", 140), ("BUY_LAND", 4), ("BUY_SEED", 50), ("BUY_ANIMAL", 4)],
)
def test_merge_never_changes_frozen_orders(
    verb: str, merges: int, kaito_turns: tuple[Turn, ...]
) -> None:
    carrying = [
        turn
        for turn in kaito_turns
        if any(order[0] == verb for order in turn.action["market"])
    ]
    assert carrying, f"no real turn ordered {verb}"
    replaced = 0
    for turn in carrying:
        proposals = (
            kaito_market_buckets(turn.action),
            (0,) * len(ALLOWED_SLOTS),
            maximal_buckets(turn.encoded),
        )
        for buckets in proposals:
            if buckets is None:
                continue
            before = copy.deepcopy(turn.action)
            result = merge_residual_action(
                turn.encoded, turn.action, replace_with(buckets)
            )
            assert turn.action == before
            assert frozen_orders(result.action) == frozen_orders(turn.action)
            assert result.action["farmer"] is turn.action["farmer"]
            assert result.action["hands"] is turn.action["hands"]
            replaced += int(result.replaced)
    assert replaced == merges


def test_use_kaito_returns_the_controllers_own_action(
    kaito_turns: tuple[Turn, ...],
) -> None:
    turn = kaito_turns[600]
    decision = ResidualDecision(
        ResidualMode.USE_KAITO, (BUCKETS - 1,) * len(ALLOWED_SLOTS)
    )

    result = merge_residual_action(turn.encoded, turn.action, decision)

    assert result.action is turn.action
    assert (result.replaced, result.fallback) == (False, None)


class ExplodingBuckets:
    """A bucket vector that fails partway through being read.

    A residual head is remote code as far as the merge is concerned. It may hold
    a lazy tensor, a memoryview onto a freed buffer, or a bug; what matters is
    that the failure lands in the middle of the merge rather than before it.
    """

    def __len__(self) -> int:
        """Return the schema's own length, so the shape check passes first."""
        return len(ALLOWED_SLOTS)

    def __getitem__(self, index: int) -> int:
        """Return zeros until the read is committed, then fail."""
        if index < len(ALLOWED_SLOTS) // 2:
            return 0
        raise RuntimeError("residual head failed while its buckets were being read")


class ExplodingDecision:
    """A decision whose buckets cannot be reached at all without raising."""

    mode = ResidualMode.REPLACE
    finite = True

    @property
    def buckets(self) -> tuple[int, ...]:
        """Raise the way a head that died between forward and merge would."""
        raise RuntimeError("residual head failed before the merge could read it")


def assert_untouched(
    result: object, action: dict[str, Any], before: dict[str, Any]
) -> None:
    """Assert the controller's action came back as the identical, unmutated object."""
    merged = cast(Any, result)
    assert merged.action is action
    assert action == before
    assert merged.replaced is False


def test_a_residual_that_raises_before_the_merge_fails_closed(
    kaito_turns: tuple[Turn, ...],
) -> None:
    turn = kaito_turns[600]
    before = copy.deepcopy(turn.action)

    result = merge_residual_action(
        turn.encoded, turn.action, cast(ResidualDecision, ExplodingDecision())
    )

    assert_untouched(result, turn.action, before)
    assert result.fallback is FallbackReason.ERROR


def test_a_residual_that_raises_mid_merge_fails_closed(
    kaito_turns: tuple[Turn, ...],
) -> None:
    turn = kaito_turns[600]
    before = copy.deepcopy(turn.action)
    decision = ResidualDecision(
        ResidualMode.REPLACE, cast(tuple[int, ...], ExplodingBuckets())
    )

    result = merge_residual_action(turn.encoded, turn.action, decision)

    assert_untouched(result, turn.action, before)
    assert result.fallback is FallbackReason.ERROR


def test_nonfinite_logits_fail_closed(kaito_turns: tuple[Turn, ...]) -> None:
    turn = kaito_turns[600]
    before = copy.deepcopy(turn.action)
    decision = ResidualDecision(
        ResidualMode.REPLACE, maximal_buckets(turn.encoded), finite=False
    )

    result = merge_residual_action(turn.encoded, turn.action, decision)

    assert_untouched(result, turn.action, before)
    assert result.fallback is FallbackReason.NONFINITE


def test_an_illegal_slot_fails_closed(kaito_turns: tuple[Turn, ...]) -> None:
    turn = next(
        candidate
        for candidate in kaito_turns
        if any(not legal for legal in maximal_buckets(candidate.encoded))
    )
    illegal = tuple(
        BUCKETS - 1 if bucket == 0 else bucket
        for bucket in maximal_buckets(turn.encoded)
    )
    before = copy.deepcopy(turn.action)

    result = merge_residual_action(turn.encoded, turn.action, replace_with(illegal))

    assert_untouched(result, turn.action, before)
    assert result.fallback is FallbackReason.MASK


@pytest.mark.parametrize(
    "buckets",
    [
        pytest.param((0,) * (len(ALLOWED_SLOTS) - 1), id="too-short"),
        pytest.param((0,) * (len(ALLOWED_SLOTS) + 1), id="too-long"),
        pytest.param((-1,) + (0,) * (len(ALLOWED_SLOTS) - 1), id="negative"),
        pytest.param((BUCKETS,) + (0,) * (len(ALLOWED_SLOTS) - 1), id="past-the-end"),
        pytest.param((1.0,) + (0,) * (len(ALLOWED_SLOTS) - 1), id="float"),
        pytest.param(("1",) + (0,) * (len(ALLOWED_SLOTS) - 1), id="string"),
        pytest.param((None,) + (0,) * (len(ALLOWED_SLOTS) - 1), id="none"),
    ],
)
def test_out_of_schema_buckets_fail_closed(
    buckets: tuple[Any, ...], kaito_turns: tuple[Turn, ...]
) -> None:
    turn = kaito_turns[600]
    before = copy.deepcopy(turn.action)

    result = merge_residual_action(turn.encoded, turn.action, replace_with(buckets))

    assert_untouched(result, turn.action, before)
    assert result.fallback is FallbackReason.SHAPE


def test_quantities_each_slot_allows_are_refused_together(
    kaito_turns: tuple[Turn, ...],
) -> None:
    outcomes: dict[str, int] = {}
    for turn in kaito_turns:
        before = copy.deepcopy(turn.action)
        result = merge_residual_action(
            turn.encoded, turn.action, replace_with(maximal_buckets(turn.encoded))
        )
        assert turn.action == before
        reason = "replaced" if result.replaced else result.fallback
        outcomes[str(reason)] = outcomes.get(str(reason), 0) + 1
    assert outcomes == {
        "replaced": 550,
        "cash": 306,
        "shed": 230,
        "orders": 38,
        "state": 316,
    }


def test_a_mask_that_lies_is_caught_by_the_independent_held_check(
    kaito_turns: tuple[Turn, ...],
) -> None:
    slot = next(
        slot for slot in ALLOWED_SLOTS if MARKET_SLOTS[slot] == ("SELL", "STRAWBERRY")
    )
    turn = next(
        candidate
        for candidate in kaito_turns
        if candidate.encoded.shed_count("STRAWBERRY") == 0
        and kaito_market_buckets(candidate.action) is not None
    )
    rows = [list(row) for row in turn.encoded.market_mask]
    rows[slot][1] = True
    lying = dataclasses.replace(
        turn.encoded, market_mask=tuple(tuple(row) for row in rows)
    )
    buckets = tuple(1 if index == slot else 0 for index in range(len(MARKET_SLOTS)))
    asked = tuple(buckets[slot] for slot in ALLOWED_SLOTS)

    result = merge_residual_action(lying, turn.action, replace_with(asked))

    assert result.action is turn.action
    assert result.fallback is FallbackReason.HELD


def test_random_bucket_vectors_never_touch_the_frozen_plan(
    kaito_turns: tuple[Turn, ...],
) -> None:
    rng = random.Random(0)
    replaced = 0
    for turn in kaito_turns[::5]:
        legal = maximal_buckets(turn.encoded)
        for draw in range(6):
            buckets = tuple(
                rng.randint(0, legal[index] if draw % 2 else BUCKETS - 1)
                for index in range(len(ALLOWED_SLOTS))
            )
            before = copy.deepcopy(turn.action)
            result = merge_residual_action(
                turn.encoded, turn.action, replace_with(buckets)
            )
            assert turn.action == before
            assert frozen_orders(result.action) == frozen_orders(turn.action)
            assert result.action["farmer"] is turn.action["farmer"]
            assert result.action["hands"] is turn.action["hands"]
            if not result.replaced:
                assert result.action is turn.action
                assert result.fallback is not None
                continue
            replaced += 1
            parsed = parse_market_orders(result.action)
            assert parsed is not None
            assert len(parsed) <= MAX_ORDERS
            wanted = {
                MARKET_SLOTS[slot]: quantity_of(bucket)
                for slot, bucket in zip(ALLOWED_SLOTS, buckets, strict=True)
                if bucket
            }
            assert {
                (entry.verb, entry.item): entry.quantity
                for entry in parsed
                if entry.verb in LEARNABLE_VERBS
            } == wanted
    assert replaced > 100, "the property never reached an accepted merge"
