"""Tests for the reference policy: shape, size and the inference budget."""

import time

import pytest
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
from kaggriculture.learn.model import Policy, load_policy_weights


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

    logits, _quantities, _market, _value = model(board, scalars, _positions(2))

    assert logits.shape == (2, MAX_UNITS, len(UNIT_OPS))


def test_forward_returns_per_unit_quantity_logits() -> None:
    """A transfer's size is the op head's own argument: one bucket vocabulary per unit.

    Shape alone would pass a head that returned the market head's own tensor by
    accident -- both are ``(batch, N, len(QUANTITIES))`` shaped -- so dtype is
    asserted too, pinning that this is float logits fresh out of a ``Linear``,
    not merely a tensor of the right size.
    """
    model = Policy()
    board = torch.zeros(2, TILE_PLANES, BOARD, BOARD)
    scalars = torch.zeros(2, SCALARS)

    _units, quantities, _market, _value = model(board, scalars, _positions(2))

    assert quantities.shape == (2, MAX_UNITS, len(QUANTITIES))
    assert quantities.dtype == torch.float32


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
        before, _quantities, _market, _value = model(board, scalars, here)
        after, _quantities, _market, _value = model(board, scalars, there)

    assert not torch.equal(before[0, 0], after[0, 0])
    assert torch.equal(before[0, 1:], after[0, 1:])


def test_two_units_on_one_tile_receive_identical_logits() -> None:
    """Without ``unit_identity`` a slot's index carries no meaning at all.

    This is the default readout, and the assertion is what it costs: the pair
    is one input to one shared ``Linear``, so no weights can give the two units
    different ops. 14.5% of the acting slots recorded from our own teacher are
    such a pair, and the clone scored 0.5035 on them against 0.9999 on every
    other slot. Passing ``unit_identity=True`` is what buys the pair apart --
    see the next test -- and this one stays to hold the default honest, not
    because being unable to answer is a property worth having.
    """
    torch.manual_seed(0)
    model = Policy().eval()
    board = torch.randn(1, TILE_PLANES, BOARD, BOARD)
    scalars = torch.randn(1, SCALARS)
    positions = _positions(1)
    positions[0, 3] = positions[0, 7] = BOARD * BOARD // 2

    with torch.no_grad():
        logits, _quantities, _market, _value = model(board, scalars, positions)

    assert torch.equal(logits[0, 3], logits[0, 7])
    assert not torch.equal(logits[0, 3], logits[0, 0])


def test_unit_identity_separates_two_units_on_one_tile() -> None:
    """A per-slot vector makes co-located units two inputs rather than one.

    The op the teacher gives a hand is a function of which hand it is -- its
    index in the farm's hand list -- so a readout that cannot tell slot 3 from
    slot 7 cannot reproduce it. The slots must still be read by the *same*
    weights, so the separation has to come from the input; the assertion that
    the two remain sensitive to the board is what says the identity vector was
    added to the gathered column rather than replacing it.
    """
    torch.manual_seed(0)
    model = Policy(unit_identity=True).eval()
    board = torch.randn(1, TILE_PLANES, BOARD, BOARD)
    scalars = torch.randn(1, SCALARS)
    positions = _positions(1)
    positions[0, 3] = positions[0, 7] = BOARD * BOARD // 2

    with torch.no_grad():
        logits, quantities, _market, _value = model(board, scalars, positions)
        moved = torch.randn(1, SCALARS)
        elsewhere, _quantities, _market, _value = model(board, moved, positions)

    assert not torch.equal(logits[0, 3], logits[0, 7])
    assert not torch.equal(quantities[0, 3], quantities[0, 7])
    assert not torch.equal(logits[0, 3], elsewhere[0, 3])


def test_unit_identity_is_the_only_key_it_adds() -> None:
    """``play.model`` reads the architecture off the checkpoint by this key.

    It builds ``Policy(unit_identity="slots.weight" in weights)`` so the file
    and the module cannot disagree about which architecture was trained. That
    is only sound while the argument adds exactly that one key and changes the
    shape of none of the others.
    """
    plain = Policy(blocks=1, channels=16).state_dict()
    identified = Policy(blocks=1, channels=16, unit_identity=True).state_dict()

    assert set(identified) - set(plain) == {"slots.weight"}
    assert not set(plain) - set(identified)
    assert all(plain[key].shape == identified[key].shape for key in plain)


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
        cheap, _quantities, _market, _value = model(
            board, torch.zeros(1, SCALARS), positions
        )
        rich, _quantities, _market, _value = model(
            board, torch.ones(1, SCALARS), positions
        )

    assert not torch.allclose(cheap, rich)


def test_forward_returns_both_heads() -> None:
    """One trunk, two decisions: what the units do and what the farm trades."""
    model = Policy()
    board = torch.zeros(2, TILE_PLANES, BOARD, BOARD)

    units, _quantities, market, _value = model(
        board, torch.zeros(2, SCALARS), _positions(2)
    )

    assert units.shape == (2, MAX_UNITS, len(UNIT_OPS))
    assert market.shape == (2, len(MARKET_SLOTS) + 2, len(QUANTITIES))


def test_forward_returns_a_value_per_state() -> None:
    """PPO's advantage is r + gamma*V(s') - V(s); without V there is no advantage."""
    _units, _quantities, _market, value = Policy()(
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
        _, _, _, before = model(board, torch.zeros(1, SCALARS), _positions(1))
        _, _, _, after = model(elsewhere, torch.zeros(1, SCALARS), _positions(1))

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
        _, _, before, _value = model(board, torch.zeros(1, SCALARS), _positions(1))
        _, _, after, _value = model(elsewhere, torch.zeros(1, SCALARS), _positions(1))

    assert not torch.equal(before, after)


def test_unit_positions_do_not_move_the_market_head() -> None:
    """Where a hand stands is not a reason to trade differently."""
    torch.manual_seed(0)
    model = Policy().eval()
    board = torch.randn(1, TILE_PLANES, BOARD, BOARD)
    moved = _positions(1)
    moved[0, 0] = BOARD * BOARD - 1

    with torch.no_grad():
        _, _, here, _value = model(board, torch.zeros(1, SCALARS), _positions(1))
        _, _, there, _value = model(board, torch.zeros(1, SCALARS), moved)

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


def test_load_policy_weights_accepts_a_checkpoint_missing_only_the_quantity_head() -> (
    None
):
    """The quantity head is the one gap ``load_policy_weights`` is built to cross.

    Built from a real ``state_dict()`` with the quantity head's own keys
    deleted, not a mock -- a mock would only prove the function accepts a
    mock, not that it moves real weights across a real gap. Comparing a trunk
    parameter before and after is what rules out the vacuous case where
    nothing loaded at all.
    """
    torch.manual_seed(0)
    trained = Policy()
    checkpoint = {
        name: tensor
        for name, tensor in trained.state_dict().items()
        if not name.startswith("quantity_head.")
    }

    torch.manual_seed(1)
    fresh = Policy()
    before = fresh.stem.weight.clone()

    missing = load_policy_weights(fresh, checkpoint)

    assert not torch.equal(before, fresh.stem.weight)
    assert torch.equal(fresh.stem.weight, trained.stem.weight)
    assert set(missing) == {"quantity_head.weight", "quantity_head.bias"}


def test_load_policy_weights_raises_on_a_missing_value_head_key() -> None:
    """A checkpoint missing anything beyond the quantity head is corrupt, not old.

    Strips one value-head key alongside the quantity head's, so the checkpoint
    looks almost like the tolerated case above and differs by exactly the key
    this function must still refuse to half-load.
    """
    trained = Policy()
    checkpoint = {
        name: tensor
        for name, tensor in trained.state_dict().items()
        if not name.startswith("quantity_head.") and name != "value.bias"
    }

    with pytest.raises(ValueError, match="value.bias"):
        load_policy_weights(Policy(), checkpoint)


# The padding-mask guard lives in tests/learn/test_train.py, against this
# project's `unit_loss`. The version that stood here called `cross_entropy`
# directly and passed with its `ignore_index` argument deleted, because IGNORE
# is -100 and that is torch's own default -- so it asserted a library default
# rather than anything we wrote.
