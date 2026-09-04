"""The floor: what the campaign has promoted, or the skeleton before it has."""

from collections.abc import Mapping
from typing import Any


def agent(
    observation: Mapping[str, Any], configuration: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Play a turn. The skeleton passes; the campaign replaces this file."""
    return {"farmer": ["PASS"], "hands": [], "market": []}
