"""Tests for the reference policy: shape, size and the inference budget."""

import time

import torch

from kaggriculture.learn.encoding import (
    BOARD,
    MARKET_SLOTS,
    MAX_UNITS,
    QUANTITIES,
    SCALARS,
    TILE_PLANES,
    UNIT_OPS,
)
from kaggriculture.learn.model import Policy


def _positions(batch: int) -> torch.Tensor:
    """Return one distinct tile per unit slot, batched.

    Distinct positions matter: with every slot on one tile a head that ignores
    position entirely would still produce plausible-looking output.
    """
    return torch.arange(MAX_UNITS, dtype=torch.int64)[None, :].tile(batch, 1)


def test_forward_returns_one_distribution_per_unit() -> None:
    """Each unit picks its own op, so the head is per-unit, not per-board."""
    model = Policy()
    board = torch.zeros(2, TILE_PLANES, BOARD, BOARD)
    scalars = torch.zeros(2, SCALARS)

    logits, _market, _value = model(board, scalars, _positions(2))

    assert logits.shape == (2, MAX_UNITS, len(UNIT_OPS))


def test_a_unit_reads_the_trunk_at_its_own_tile() -> None:
    """Moving one unit must move that unit's logits and nothing else.

    This is the property the pooled head could not have had: it summarised the
    whole board into one vector and emitted every slot from it, so perturbing a
    single cell was measured to move all ``MAX_UNITS`` slots, and moving a unit
    moved nothing at all because position was not an input. Both halves are
    asserted -- the moved slot changes, the others stay bit-identical -- because
    either alone passes for a head reading a global summary.
    """
    torch.manual_seed(0)
    model = Policy().eval()
    board = torch.randn(1, TILE_PLANES, BOARD, BOARD)
    scalars = torch.randn(1, SCALARS)
    here = _positions(1)
    there = here.clone()
    there[0, 0] = BOARD * BOARD - 1

    with torch.no_grad():
        before, _market, _value = model(board, scalars, here)
        after, _market, _value = model(board, scalars, there)

    assert not torch.equal(before[0, 0], after[0, 0])
    assert torch.equal(before[0, 1:], after[0, 1:])


def test_two_units_on_one_tile_receive_identical_logits() -> None:
    """One readout is shared by every slot, so a slot's index carries no meaning.

    Two hands standing on the same tile see the same game, so they must be
    scored the same way. A per-slot block of weights -- what a single wide
    ``Linear`` over a pooled vector amounts to -- would give them different
    logits for no reason available to the game.
    """
    torch.manual_seed(0)
    model = Policy().eval()
    board = torch.randn(1, TILE_PLANES, BOARD, BOARD)
    scalars = torch.randn(1, SCALARS)
    positions = _positions(1)
    positions[0, 3] = positions[0, 7] = BOARD * BOARD // 2

    with torch.no_grad():
        logits, _market, _value = model(board, scalars, positions)

    assert torch.equal(logits[0, 3], logits[0, 7])
    assert not torch.equal(logits[0, 3], logits[0, 0])


def test_the_model_fits_the_size_every_winner_used() -> None:
    """Verified winners span 1M-20M parameters; nothing above 20M is supported."""
    parameters = sum(p.numel() for p in Policy().parameters())

    assert 1e6 < parameters < 20e6


def test_a_turn_fits_the_sandbox_budget_at_two_threads() -> None:
    """The sandbox has two cores and one second a turn, not this workstation.

    Measured at two threads because timing on 64 cores flatters the sandbox by
    an order of magnitude. The Phase 0 probe measured a 10M-parameter trunk at
    ~45ms of the 1000ms budget; this asserts a wide margin, not that figure.
    """
    torch.set_num_threads(2)
    model = Policy().eval()
    board = torch.zeros(1, TILE_PLANES, BOARD, BOARD)
    scalars = torch.zeros(1, SCALARS)
    positions = _positions(1)

    with torch.no_grad():
        model(board, scalars, positions)
        start = time.perf_counter()
        for _ in range(10):
            model(board, scalars, positions)
        elapsed = (time.perf_counter() - start) / 10

    assert elapsed < 0.25, f"{elapsed * 1000:.0f}ms per turn leaves no margin"


def test_the_market_reaches_the_trunk() -> None:
    """Prices decide this game; a scalar branch that is ignored is a silent bug."""
    model = Policy().eval()
    board = torch.zeros(1, TILE_PLANES, BOARD, BOARD)
    positions = _positions(1)

    with torch.no_grad():
        cheap, _market, _value = model(board, torch.zeros(1, SCALARS), positions)
        rich, _market, _value = model(board, torch.ones(1, SCALARS), positions)

    assert not torch.allclose(cheap, rich)


def test_forward_returns_both_heads() -> None:
    """One trunk, two decisions: what the units do and what the farm trades."""
    model = Policy()
    board = torch.zeros(2, TILE_PLANES, BOARD, BOARD)

    units, market, _value = model(board, torch.zeros(2, SCALARS), _positions(2))

    assert units.shape == (2, MAX_UNITS, len(UNIT_OPS))
    assert market.shape == (2, len(MARKET_SLOTS) + 2, len(QUANTITIES))


def test_forward_returns_a_value_per_state() -> None:
    """PPO's advantage is r + gamma*V(s') - V(s); without V there is no advantage."""
    units, market, value = Policy()(
        torch.zeros(2, TILE_PLANES, BOARD, BOARD),
        torch.zeros(2, SCALARS),
        _positions(2),
    )

    assert value.shape == (2,)


def test_the_value_head_reads_the_whole_board() -> None:
    """How well we are doing is a property of the position, not of one tile."""
    torch.manual_seed(0)
    model = Policy().eval()
    board = torch.randn(1, TILE_PLANES, BOARD, BOARD)
    elsewhere = board.clone()
    elsewhere[0, :, 9, 9] += 5.0

    with torch.no_grad():
        _, _, before = model(board, torch.zeros(1, SCALARS), _positions(1))
        _, _, after = model(elsewhere, torch.zeros(1, SCALARS), _positions(1))

    assert not torch.equal(before, after)


def test_the_market_head_reads_the_whole_board() -> None:
    """What to sell depends on the whole farm, not on any one tile.

    The unit head deliberately reads only its own unit's tile. The market head
    must not: a harvest anywhere changes what there is to sell.
    """
    torch.manual_seed(0)
    model = Policy().eval()
    board = torch.randn(1, TILE_PLANES, BOARD, BOARD)
    elsewhere = board.clone()
    elsewhere[0, :, 9, 9] += 5.0

    with torch.no_grad():
        _, before, _value = model(board, torch.zeros(1, SCALARS), _positions(1))
        _, after, _value = model(elsewhere, torch.zeros(1, SCALARS), _positions(1))

    assert not torch.equal(before, after)


def test_unit_positions_do_not_move_the_market_head() -> None:
    """Where a hand stands is not a reason to trade differently."""
    torch.manual_seed(0)
    model = Policy().eval()
    board = torch.randn(1, TILE_PLANES, BOARD, BOARD)
    moved = _positions(1)
    moved[0, 0] = BOARD * BOARD - 1

    with torch.no_grad():
        _, here, _value = model(board, torch.zeros(1, SCALARS), _positions(1))
        _, there, _value = model(board, torch.zeros(1, SCALARS), moved)

    assert torch.equal(here, there)


def test_loading_a_checkpoint_without_a_value_head_still_loads_the_trunk() -> None:
    """The behaviour-cloned checkpoint predates the value head.

    ``strict=False`` alone proves nothing -- it would report the same
    "success" if the trunk's own keys had drifted and nothing but the value
    head loaded, or if nothing loaded at all. This compares a trunk parameter
    before and after loading a checkpoint with the value head's keys
    stripped, so only an actual weight transfer passes.
    """
    torch.manual_seed(0)
    trained = Policy()
    checkpoint = {
        name: tensor
        for name, tensor in trained.state_dict().items()
        if not name.startswith("value.")
    }

    torch.manual_seed(1)
    fresh = Policy()
    before = fresh.stem.weight.clone()
    result = fresh.load_state_dict(checkpoint, strict=False)

    assert not torch.equal(before, fresh.stem.weight)
    assert torch.equal(fresh.stem.weight, trained.stem.weight)
    assert set(result.missing_keys) == {"value.weight", "value.bias"}
    assert result.unexpected_keys == []


# The padding-mask guard lives in tests/learn/test_train.py, against this
# project's `unit_loss`. The version that stood here called `cross_entropy`
# directly and passed with its `ignore_index` argument deleted, because IGNORE
# is -100 and that is torch's own default -- so it asserted a library default
# rather than anything we wrote.
