"""Tests for the pure action vocabulary shared by policies and submissions."""

import json
import subprocess
import sys

import pytest
import torch

from kaggriculture.action_codec import (
    HIRE_SLOT,
    LAND_SLOT,
    MARKET_SLOTS,
    QUANTITIES,
    UNIT_OPS,
    SelectedActions,
    decode_selected,
    market_orders_of,
)
from kaggriculture.learn.encoding import MAX_UNITS, decode_market, decode_units


def import_in_fresh_process(module: str) -> set[str]:
    """Return the modules loaded while importing ``module`` in a child process."""
    output = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import importlib, json, sys; "
            f"importlib.import_module({module!r}); "
            "print(json.dumps(sorted(sys.modules)))",
        ],
        text=True,
    )
    return set(json.loads(output))


def _unit_logits(index: int) -> torch.Tensor:
    logits = torch.full((1, MAX_UNITS, len(UNIT_OPS)), -torch.inf)
    logits[0, 0, index] = 0.0
    return logits


def _quantity_logits(index: int, rows: int = MAX_UNITS) -> torch.Tensor:
    logits = torch.full((1, rows, len(QUANTITIES)), -torch.inf)
    logits[..., index] = 0.0
    return logits


def _market_logits(slot: int) -> torch.Tensor:
    logits = torch.full((1, len(MARKET_SLOTS) + 2, len(QUANTITIES)), -torch.inf)
    logits[..., 0] = 0.0
    logits[0, slot, QUANTITIES.index(2)] = 1.0
    return logits


def test_selected_decoder_matches_the_existing_tensor_decoder() -> None:
    """The pure selected representation preserves the Tensor decoder's action."""
    unit_indices = (UNIT_OPS.index("PICKUP:WHEAT"), UNIT_OPS.index("WATER"))
    quantity_indices = (QUANTITIES.index(3), QUANTITIES.index(1))
    market_indices = [0] * (len(MARKET_SLOTS) + 2)
    market_indices[HIRE_SLOT] = QUANTITIES.index(2)
    market_indices[LAND_SLOT] = QUANTITIES.index(1)
    selected = SelectedActions(unit_indices, quantity_indices, tuple(market_indices))

    actual = decode_selected(selected)

    assert actual["farmer"] == ["PICKUP", "WHEAT", 3]
    assert actual["hands"] == [["WATER"]]
    assert actual["market"][-3:] == [["HIRE"], ["HIRE"], ["BUY_LAND"]]


def test_action_codec_imports_neither_torch_nor_pydantic() -> None:
    """Submission decoding does not load learning-only dependencies."""
    imported = import_in_fresh_process("kaggriculture.action_codec")

    assert "torch" not in imported
    assert "pydantic" not in imported


@pytest.mark.parametrize("op_index", range(len(UNIT_OPS)))
def test_every_unit_label_matches_the_tensor_decoder(op_index: int) -> None:
    """Every unit vocabulary index has the same pure and Tensor action form."""
    logits = _unit_logits(op_index)
    quantities = _quantity_logits(QUANTITIES.index(3))
    selected = SelectedActions(
        (op_index,),
        (QUANTITIES.index(3),),
        (0,) * (len(MARKET_SLOTS) + 2),
    )

    actual = decode_units(
        logits,
        quantities,
        1,
        torch.ones_like(logits, dtype=torch.bool),
        torch.ones_like(quantities, dtype=torch.bool),
    )
    expected = decode_selected(selected)

    assert actual == {
        "farmer": expected["farmer"],
        "hands": expected["hands"],
        "market": [],
    }


@pytest.mark.parametrize("slot", range(len(MARKET_SLOTS) + 2))
def test_every_market_slot_matches_the_tensor_decoder(slot: int) -> None:
    """Every market row has the same pure and Tensor order construction."""
    logits = _market_logits(slot)
    expected_indices = [0] * (len(MARKET_SLOTS) + 2)
    expected_indices[slot] = QUANTITIES.index(2)

    actual = decode_market(logits, torch.ones_like(logits, dtype=torch.bool))

    assert actual == market_orders_of(tuple(expected_indices))
