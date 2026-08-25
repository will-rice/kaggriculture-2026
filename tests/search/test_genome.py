"""One-hot optimizer genome contracts for hybrid configurations."""
# ruff: noqa: D103

import math
import random

import pytest

from kaggriculture.features import CROP_NAMES
from kaggriculture.search.genome import HAND_TARGETS, GenomeCodec, decode, encode


def test_every_integer_and_category_decodes_from_one_argmax_group() -> None:
    """Use independent argmax groups for discrete policy choices."""
    codec = GenomeCodec.default()
    vector = [0.0] * codec.width
    for group in codec.discrete_groups:
        vector[group.start + len(group.values) - 1] = 10.0

    decoded = codec.decode(vector)

    assert decoded.opening.phases[0].target_hands == HAND_TARGETS[-1]
    assert decoded.opening.phases[0].primary_crop == CROP_NAMES[-1]
    assert all(group.width == len(group.values) for group in codec.discrete_groups)
    assert all(gene.path[0] in {"jobs", "market"} for gene in codec.continuous_genes)


def test_fixed_first_phase_day_is_not_an_optimizer_parameter() -> None:
    """Keep the invariant day-zero phase out of the one-hot search surface."""
    codec = GenomeCodec.default()

    assert codec.decode([0.0] * codec.width).opening.phases[0].start_day == 0
    assert (
        "opening",
        "phases",
        0,
        "start_day",
    ) not in {group.path for group in codec.discrete_groups}


def test_codec_is_exact_for_one_hots_and_continuous_inverse_mapping() -> None:
    """Round-trip 100 deterministic valid candidates through the flat genome."""
    codec = GenomeCodec.default()
    randomizer = random.Random(732_451)

    for _ in range(100):
        vector = [0.0] * codec.width
        for group in codec.discrete_groups:
            vector[group.start + randomizer.randrange(group.width)] = 1.0
        for gene in codec.continuous_genes:
            vector[gene.index] = randomizer.random()

        config = codec.decode(vector)

        assert codec.decode(codec.encode(config)) == config


def test_codec_argmax_ties_and_invalid_vectors_are_deterministic() -> None:
    """Prefer the lower index on ties and reject malformed optimizer output."""
    codec = GenomeCodec.default()
    vector = [0.0] * codec.width

    assert codec.decode(vector).opening.phases[0].target_hands == HAND_TARGETS[0]
    with pytest.raises(ValueError, match="finite"):
        codec.decode([math.nan] * codec.width)
    with pytest.raises(ValueError, match=str(codec.width)):
        codec.decode(vector[:-1])


def test_module_helpers_use_the_default_codec() -> None:
    """Expose a stable simple API without a separate codec implementation."""
    config = GenomeCodec.default().decode([0.0] * GenomeCodec.default().width)

    assert decode(encode(config)) == config
