"""Batched policy-action decoding parity tests."""
# ruff: noqa: ANN001, ANN202, D103

import torch

from kaggriculture.learn.encoding import MARKET_SLOTS, QUANTITIES, decode_market
from kaggriculture.sim.decode import decode_market_buckets, decode_unit_logits


def _orders(actions, batch, seat):
    names = {1: "SELL", 2: "BUY_SEED", 3: "BUY_PRODUCT", 4: "BUY_ANIMAL"}
    catalogues = {
        1: tuple(
            sorted(
                (
                    "WHEAT",
                    "CARROT",
                    "TOMATO",
                    "STRAWBERRY",
                    "MELON",
                    "EGG",
                    "MILK",
                    "WOOL",
                    "FERTILIZER",
                )
            )
        ),
        2: tuple(sorted(("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"))),
        3: ("WHEAT", "FERTILIZER"),
        4: tuple(sorted(("GOOSE", "COW", "SHEEP"))),
    }
    result = []
    for slot in range(10):
        kind = int(actions.order_type[batch, seat, slot])
        if kind == 0:
            continue
        if kind == 5:
            result.append(["HIRE"])
        elif kind == 6:
            result.append(["BUY_LAND"])
        else:
            item = catalogues[kind][int(actions.order_item[batch, seat, slot])]
            result.append(
                [names[kind], item, int(actions.order_qty[batch, seat, slot])]
            )
    return result


def test_market_bucket_decoding_matches_reference_order_and_truncation() -> None:
    generator = torch.Generator().manual_seed(93)
    buckets = torch.randint(
        0, len(QUANTITIES), (8, 2, len(MARKET_SLOTS) + 2), generator=generator
    )

    actual = decode_market_buckets(buckets)

    mask = torch.ones(1, len(MARKET_SLOTS) + 2, len(QUANTITIES), dtype=torch.bool)
    for batch in range(8):
        for seat in range(2):
            logits = torch.full_like(mask, -1.0, dtype=torch.float32)
            logits[0].scatter_(1, buckets[batch, seat, :, None], 1.0)
            assert _orders(actual, batch, seat) == decode_market(logits, mask)


def test_unit_logits_are_masked_before_decoding() -> None:
    logits = torch.tensor([[[9.0, 8.0, 7.0], [1.0, 2.0, 3.0]]])
    mask = torch.tensor([[[False, True, True], [True, False, False]]])

    assert decode_unit_logits(logits, mask).tolist() == [[1, 0]]
