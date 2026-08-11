"""Small immutable tensor constants cached per device for graph capture."""

from functools import lru_cache
from typing import Any

import torch


def _freeze(value: Any) -> Any:  # noqa: ANN401
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@lru_cache(maxsize=None)
def _constant(
    values: tuple[Any, ...],
    dtype: torch.dtype,
    device_type: str,
    device_index: int | None,
) -> torch.Tensor:
    device = (
        torch.device(device_type)
        if device_index is None
        else torch.device(device_type, device_index)
    )
    return torch.tensor(values, dtype=dtype, device=device)


def tensor_constant(
    values: object, *, dtype: torch.dtype, device: torch.device | str
) -> torch.Tensor:
    """Return one immutable tensor allocation for a value, dtype, and device."""
    resolved = torch.device(device)
    device_index = resolved.index
    if resolved.type == "cuda" and device_index is None:
        device_index = torch.cuda.current_device()
    frozen = _freeze(values)
    if not isinstance(frozen, tuple):
        frozen = (frozen,)
    return _constant(frozen, dtype, resolved.type, device_index)
