"""Batched Torch implementation of CPython's MT19937 stream."""

import torch

_N = 624
_M = 397
_MASK = 0xFFFFFFFF
_UPPER = 0x80000000
_LOWER = 0x7FFFFFFF


def _low32(value: torch.Tensor) -> torch.Tensor:
    return value & _MASK


def _seed_state(keys: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Implement ``init_genrand`` followed by CPython ``init_by_array``."""
    rows = keys.shape[0]
    device = keys.device
    state = torch.empty((rows, _N), dtype=torch.int64, device=device)
    state[:, 0] = 19_650_218
    for index in range(1, _N):
        previous = state[:, index - 1]
        state[:, index] = _low32(1_812_433_253 * (previous ^ (previous >> 30)) + index)

    state = state.clone()
    index = 1
    key_index = torch.zeros(rows, dtype=torch.int64, device=device)
    for _ in range(_N):
        previous = state[:, index - 1]
        key = keys.gather(1, key_index[:, None]).squeeze(1)
        state[:, index] = _low32(
            (state[:, index] ^ ((previous ^ (previous >> 30)) * 1_664_525))
            + key
            + key_index
        )
        index += 1
        key_index += 1
        key_index = torch.where(key_index >= lengths, 0, key_index)
        if index >= _N:
            state[:, 0] = state[:, _N - 1]
            index = 1

    for _ in range(_N - 1):
        previous = state[:, index - 1]
        state[:, index] = _low32(
            (state[:, index] ^ ((previous ^ (previous >> 30)) * 1_566_083_941)) - index
        )
        index += 1
        if index >= _N:
            state[:, 0] = state[:, _N - 1]
            index = 1
    state[:, 0] = _UPPER
    return state


def _twist(state: torch.Tensor) -> torch.Tensor:
    twisted = state.clone()
    y = (state[:, : _N - _M] & _UPPER) | (state[:, 1 : _N - _M + 1] & _LOWER)
    twisted[:, : _N - _M] = _low32(state[:, _M:] ^ (y >> 1) ^ ((y & 1) * 0x9908B0DF))
    second_start = _N - _M
    second_mid = 2 * (_N - _M)
    y = (state[:, second_start:second_mid] & _UPPER) | (
        state[:, second_start + 1 : second_mid + 1] & _LOWER
    )
    twisted[:, second_start:second_mid] = _low32(
        twisted[:, : _N - _M] ^ (y >> 1) ^ ((y & 1) * 0x9908B0DF)
    )
    y = (state[:, second_mid : _N - 1] & _UPPER) | (state[:, second_mid + 1 :] & _LOWER)
    twisted[:, second_mid : _N - 1] = _low32(
        twisted[:, second_start : _M - 1] ^ (y >> 1) ^ ((y & 1) * 0x9908B0DF)
    )
    last_y = (state[:, _N - 1] & _UPPER) | (twisted[:, 0] & _LOWER)
    twisted[:, _N - 1] = _low32(
        twisted[:, _M - 1] ^ (last_y >> 1) ^ ((last_y & 1) * 0x9908B0DF)
    )
    return twisted


def _temper(words: torch.Tensor) -> torch.Tensor:
    words = words ^ (words >> 11)
    words = words ^ ((words << 7) & 0x9D2C5680)
    words = words ^ ((words << 15) & 0xEFC60000)
    return _low32(words ^ (words >> 18))


def rng_words(seeds: torch.Tensor, *, days: int = 30, words: int = 512) -> torch.Tensor:
    """Materialise CPython-compatible words for each episode seed and day."""
    if seeds.ndim != 1:
        raise ValueError(f"seeds must be one-dimensional, got {tuple(seeds.shape)}")
    if words > _N:
        raise ValueError(f"one twist supplies at most {_N} words, got {words}")
    if bool((seeds < 0).any()):
        raise ValueError("episode seeds must be non-negative")
    day = torch.arange(days, dtype=torch.int64, device=seeds.device)
    resolved = (seeds[:, None] * 1_000_003) ^ day[None, :]
    flat = resolved.reshape(-1)
    keys = torch.stack((flat & _MASK, (flat >> 32) & _MASK), dim=-1)
    lengths = torch.where(keys[:, 1] == 0, 1, 2)
    state = _seed_state(keys, lengths)
    output = _temper(_twist(state)[:, :words]).to(torch.uint32)
    return output.reshape(len(seeds), days, words)


def random_from_words(words: torch.Tensor) -> torch.Tensor:
    """Reconstruct CPython ``random()`` values from consecutive word pairs."""
    if words.shape[-1] % 2:
        raise ValueError("random() requires an even number of MT19937 words")
    unsigned = words.to(torch.int64)
    first = unsigned[..., 0::2] >> 5
    second = unsigned[..., 1::2] >> 6
    return (first.to(torch.float64) * 67_108_864.0 + second) / 9_007_199_254_740_992.0


def select_day_words(words: torch.Tensor, day: torch.Tensor) -> torch.Tensor:
    """Select each row's day as unsigned int64, including on CUDA.

    CUDA cannot advanced-index uint32. Reinterpreting the same storage as
    int32 makes indexing supported; masking after widening restores the exact
    unsigned bits without copying the full episode tensor.
    """
    if words.ndim != 3 or day.shape != words.shape[:1]:
        raise ValueError(
            f"expected words (B,D,W) and day (B,), got {words.shape} and {day.shape}"
        )
    rows = torch.arange(len(words), device=words.device)
    signed = words.view(torch.int32)[rows, day.to(torch.int64)].to(torch.int64)
    return signed & _MASK
