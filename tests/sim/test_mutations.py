"""Seeded source mutations proving the named fidelity tests can go red."""
# ruff: noqa: D103, E501

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


@dataclass(frozen=True)
class Mutation:
    """One deliberate single-line semantic fault and its expected detector."""

    number: int
    module: str
    old: str
    new: str
    test: str


MUTATIONS = (
    Mutation(
        1,
        "market.py",
        "price > 1",
        "price >= 1",
        "test_price_floor_sale_does_not_add_supply",
    ),
    Mutation(
        2,
        "market.py",
        "room = (SHED_CAPACITY - state.shed.sum(dim=-1))",
        "room = (SHED_CAPACITY + 1 - state.shed.sum(dim=-1))",
        "test_capacity_is_rechecked_for_each_bought_unit",
    ),
    Mutation(
        3,
        "pricing.py",
        "torch.round(prices)",
        "torch.floor(prices)",
        "test_market_prices_match_reference_exhaustively",
    ),
    Mutation(
        4,
        "day.py",
        "& ((step - state.max_lifespan_step) % 2 == 0)",
        "& torch.ones_like(step, dtype=torch.bool)",
        "test_plant_decay_skips_the_odd_lifespan_offset",
    ),
    Mutation(
        5,
        "day.py",
        "state.consecutive_unfed >= 2",
        "state.consecutive_unfed > 2",
        "test_tensor_day_phases_match_production_escape_weeds_and_shop_draw",
    ),
    Mutation(
        6,
        "units.py",
        "access = alive & (((x == 4)",
        "access = alive & (kind != 1) & (((x == 4)",
        "test_tensor_units_match_locked_shed_transfers_and_inventory_order",
    ),
    Mutation(
        7,
        "units.py",
        "order = torch.where(sequence[:, unit] > 0, sequence[:, unit], 2**30).argsort(dim=1)",
        "order = torch.arange(len(SHED_NAMES), device=held.device)[None].expand_as(held)",
        "test_tensor_units_match_locked_shed_transfers_and_inventory_order",
    ),
    Mutation(
        8,
        "units.py",
        '_gather(state, "fertilized_until_day", position) >= day',
        '_gather(state, "fertilized_until_day", position) > day',
        "test_fertilizer_bonus_includes_its_final_day",
    ),
    Mutation(
        9,
        "market.py",
        "torch.cat((levels, levels - 1), dim=-1)",
        "torch.cat((levels, levels), dim=-1)",
        "test_buy_product_quote_at_inventory_minus_one_nets_zero_round_trip",
    ),
    Mutation(
        10,
        "market.py",
        "    quoted = prices_for(torch.cat((levels, levels - 1), dim=-1), product[..., None])",
        "    quoted = prices_for(torch.cat((levels, levels - 1), dim=-1), product[..., None])\n"
        "    quoted = quoted + torch.arange(2, device=levels.device)[:, None]",
        "test_tensor_market_matches_sale_funding_later_purchase_and_lockstep",
    ),
    Mutation(
        11,
        "units.py",
        "position = y * 10 + x",
        "position = x * 10 + y",
        "test_tensor_units_match_movement_build_and_atomic_plant_guard",
    ),
    Mutation(
        12,
        "day.py",
        "next_day = state.day[:, None, None, None] + 1\n"
        "    since = next_day - state.planted_day - first",
        "next_day = state.day[:, None, None, None]\n"
        "    since = next_day - state.planted_day - first",
        "test_ongoing_crop_uses_the_next_day_for_production",
    ),
    Mutation(
        13,
        "day.py",
        "bonus = torch.where(was_fed, state.pending_care_bonus, 0)",
        "bonus = state.pending_care_bonus",
        "test_unfed_production_wipes_a_banked_care_bonus",
    ),
    Mutation(
        14,
        "day.py",
        "[int(name in TOWN_CENTER_PRODUCTS) for name in PRODUCT_NAMES]",
        "[1 for name in PRODUCT_NAMES]",
        "test_tensor_day_phases_match_town_consumption_and_clock",
    ),
    Mutation(
        15,
        "day.py",
        "consumed = 2 * flat_empty.sum(dim=1)",
        "consumed = 2 * empty[:, 1].flatten(1).sum(dim=1)",
        "test_tensor_day_phases_match_production_escape_weeds_and_shop_draw",
    ),
    Mutation(
        16,
        "decode.py",
        "present & (ranks < 10)",
        "present",
        "test_market_bucket_decoder_matches_reference_expansion",
    ),
    Mutation(
        17,
        "market.py",
        "state.prices.copy_(refresh_prices(state.inventory))",
        "state.prices.copy_(state.prices)",
        "test_market_phase_refreshes_stored_prices_before_the_day_phase",
    ),
    Mutation(
        18,
        "units.py",
        "selected = harvesting & (kind == 3) & (crop >= 1) & mature",
        "selected = harvesting & (kind == 3) & (crop >= 1)",
        "test_harvest_refuses_a_crop_before_first_yield_day",
    ),
    Mutation(
        19,
        "day.py",
        "state.hand_count.copy_(torch.where(end[:, None], 0, state.hand_count))",
        "state.hand_count.copy_(state.hand_count)",
        "test_tensor_day_phases_match_end_of_day_rng_and_reset",
    ),
    Mutation(
        20,
        "units.py",
        "blocked = demand > seeds",
        "blocked = torch.zeros_like(demand, dtype=torch.bool)",
        "test_tensor_units_match_movement_build_and_atomic_plant_guard",
    ),
    Mutation(
        21,
        "units.py",
        'digging = unlocked & (op == _OP["DIG"]) & (kind != 0) & (occupant == 0)',
        'digging = unlocked & (op == _OP["DIG"]) & (kind != 0)',
        "test_dig_refuses_occupied_structure_and_place_requires_matching_kind",
    ),
    Mutation(
        22,
        "units.py",
        "place_op & (kind == structure) & (occupant == 0)",
        "place_op & (kind != 1) & (occupant == 0)",
        "test_dig_refuses_occupied_structure_and_place_requires_matching_kind",
    ),
    Mutation(
        23,
        "market.py",
        "state.hires_today.to(torch.int64).clamp_max(40)",
        "(state.hires_today.to(torch.int64) + 1).clamp_max(40)",
        "test_hire_cost_uses_current_hires_today_fibonacci_index",
    ),
    Mutation(
        24,
        "day.py",
        "drawn = choices.gather(1, first[:, None]).squeeze(1)",
        "drawn = (choices.gather(1, first[:, None]).squeeze(1) + state.shop_count) % 8",
        "test_shop_draw_can_repeat_an_existing_shop",
    ),
)


@pytest.mark.slow
@pytest.mark.parametrize(
    "mutation", MUTATIONS, ids=lambda value: f"mutation-{value.number}"
)
def test_seeded_mutation_is_caught_by_named_test(
    mutation: Mutation, tmp_path: Path
) -> None:
    package = tmp_path / "src" / "kaggriculture"
    shutil.copytree(ROOT / "src" / "kaggriculture", package)
    target = package / "sim" / mutation.module
    source = target.read_text()
    assert source.count(mutation.old) == 1, f"mutation {mutation.number} is stale"
    target.write_text(source.replace(mutation.old, mutation.new))
    test_file = {
        "pricing.py": "test_pricing.py",
        "decode.py": "test_decode.py",
        "units.py": "test_rules_units.py",
        "market.py": "test_rules_market.py",
        "day.py": "test_rules_day.py",
    }[mutation.module]
    node = f"{ROOT / 'tests' / 'sim' / test_file}::{mutation.test}"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(tmp_path / "src")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", node],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode != 0, (
        f"mutation {mutation.number} survived {mutation.test}\n{result.stdout}"
    )
