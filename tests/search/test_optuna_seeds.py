"""Contract tests for fixed Optuna warm-start configurations."""

import hashlib
from pathlib import Path

from kaggriculture.hybrid.config import HybridConfig
from kaggriculture.search.evolution import SearchState
from kaggriculture.search.optuna_seeds import (
    config_sha256,
    deliberate_seed_configs,
    warm_start_configs,
)

LEGACY_STATE_PATH = Path("run/hybrid/search-state.json")


def test_deliberate_seeds_are_three_distinct_complete_configs() -> None:
    """The manual policies are complete and cover three liquidation timings."""
    seeds = deliberate_seed_configs()
    assert len(seeds) == 3
    assert len({config_sha256(config) for config in seeds}) == 3
    assert {config.liquidation_start_day for config in seeds} == {23, 25, 27}


def test_warm_start_is_default_four_elites_then_three_deliberate() -> None:
    """Warm starts preserve the legacy elite order after the baseline config."""
    configs = warm_start_configs(
        LEGACY_STATE_PATH,
        expected_sha256=hashlib.sha256(LEGACY_STATE_PATH.read_bytes()).hexdigest(),
    )
    assert len(configs) == 8
    assert configs[0] == HybridConfig.default()
    assert configs[1:5] == tuple(
        row.config for row in SearchState.load(LEGACY_STATE_PATH).elites
    )
    assert configs[5:] == deliberate_seed_configs()
    assert len({config_sha256(config) for config in configs}) == 8
