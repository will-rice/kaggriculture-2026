"""The sole optional compiler and canonical policy-export boundary."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import cast

import torch
from torch import nn

from kaggriculture.learn.toad.config import ToadConfig


class CompileRequestedError(RuntimeError):
    """A requested compiler mode could not be constructed."""


def maybe_compile(module: nn.Module, config: ToadConfig) -> nn.Module:
    """Compile ``module`` exactly when the immutable runtime contract requests it.

    The caller owns where this module sits in the training graph.  Toad compiles
    its active policy subtree while retaining the Lightning module as the
    framework-owned callback and checkpoint boundary.
    """
    compile_config = config.runtime.compile
    if not compile_config.enabled:
        return module
    try:
        return cast(
            nn.Module,
            torch.compile(
                module,
                mode=compile_config.mode,
                fullgraph=compile_config.fullgraph,
                dynamic=compile_config.dynamic,
            ),
        )
    except Exception as error:
        raise CompileRequestedError(
            f"torch.compile was requested but could not be initialized: {error}"
        ) from error


def unwrap_compiled(module: nn.Module) -> nn.Module:
    """Return an eagerly-shaped module without exposing compiler internals."""
    current = module
    while True:
        original = getattr(current, "_orig_mod", None)
        if not isinstance(original, nn.Module) or original is current:
            return current
        current = original


def policy_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Export canonical policy weights from a policy or Lightning owner.

    Compiler wrappers register their wrapped module beneath a private child name,
    which would otherwise leak ``_orig_mod.`` prefixes into actors, population
    snapshots, and user-facing policy files.
    """
    owner = unwrap_compiled(module)
    policy = getattr(owner, "policy", owner)
    if not isinstance(policy, nn.Module):
        raise TypeError("compiled policy owner does not expose an nn.Module policy")
    state = unwrap_compiled(policy).state_dict()
    return dict(cast(Mapping[str, torch.Tensor], state))


def canonicalize_policy_state_keys(
    state_dict: MutableMapping[str, torch.Tensor], *, prefix: str = ""
) -> None:
    """Remove compiler-wrapper prefixes from a Lightning owner state dictionary."""
    wrapped_prefix = f"{prefix}policy._orig_mod."
    canonical_prefix = f"{prefix}policy."
    for key in tuple(state_dict):
        if key.startswith(wrapped_prefix):
            canonical_key = canonical_prefix + key.removeprefix(wrapped_prefix)
            state_dict[canonical_key] = state_dict.pop(key)


def prepare_policy_state_keys_for_load(
    module: nn.Module,
    state_dict: MutableMapping[str, torch.Tensor],
    *,
    prefix: str = "",
) -> None:
    """Adapt canonical checkpoint keys when the current policy is compiled."""
    owner = unwrap_compiled(module)
    policy = getattr(owner, "policy", None)
    if not isinstance(policy, nn.Module) or unwrap_compiled(policy) is policy:
        return
    canonical_prefix = f"{prefix}policy."
    wrapped_prefix = f"{prefix}policy._orig_mod."
    for key in tuple(state_dict):
        if key.startswith(canonical_prefix) and not key.startswith(wrapped_prefix):
            wrapped_key = wrapped_prefix + key.removeprefix(canonical_prefix)
            state_dict[wrapped_key] = state_dict.pop(key)
