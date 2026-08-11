"""CPython RNG parity tests."""
# ruff: noqa: D103

import random

import torch
from kaggle_environments import make

from kaggriculture.sim.rng import random_from_words, rng_words, select_day_words
from kaggriculture.sim.state import pack


def test_rng_words_match_cpython_for_each_seed_and_day() -> None:
    seeds = torch.tensor([0, 7, 2**40 + 19], dtype=torch.int64)
    actual = rng_words(seeds, days=4, words=512)

    for batch, seed in enumerate(seeds.tolist()):
        for day in range(4):
            reference = random.Random((seed * 1_000_003) ^ day)
            expected = [reference.getrandbits(32) for _ in range(512)]
            assert actual[batch, day].tolist() == expected


def test_random_reconstruction_matches_cpython() -> None:
    seed = 918273
    reference = random.Random(seed)
    words = torch.tensor(
        [reference.getrandbits(32) for _ in range(200)], dtype=torch.int64
    )
    reference = random.Random(seed)
    expected = [reference.random() for _ in range(100)]

    assert random_from_words(words).tolist() == expected


def test_pack_materializes_the_episode_rng_stream() -> None:
    environment = make("kaggriculture", configuration={"seed": 41}, debug=True)
    environment.reset(2)

    state = pack([environment])

    assert torch.equal(state.rng_words, rng_words(state.seed))


def test_day_word_selection_preserves_unsigned_bits() -> None:
    words = torch.tensor([[[0, 2**32 - 1]], [[2**31, 17]]], dtype=torch.uint32)

    selected = select_day_words(words, torch.tensor([0, 0]))

    assert selected.dtype == torch.int64
    assert selected.tolist() == [[0, 2**32 - 1], [2**31, 17]]
