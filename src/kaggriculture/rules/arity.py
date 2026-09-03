"""Classify whether a base agent's call signature accepts configuration.

Resolved once, at load time, rather than discovered by catching the
mismatched-arity failure and retrying with fewer arguments: the base
agents are stateful across turns (module-level tallies mutated on every
call), so a per-turn retry would silently invoke a stateful base twice on
its very first turn -- once raising partway through argument binding, once
succeeding. This module is
embedded whole into the generated agent (see ``kaggriculture.rules.agent``)
so the classification runs once there too, and it is a standalone module --
rather than a function defined inline in ``agent.py`` -- so it can be
imported and tested directly against several signature shapes instead of
only through a downstream call count, which cannot distinguish the fix from
the bug it replaces.
"""

import inspect
from collections.abc import Callable
from typing import Any

_POSITIONAL = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
)


def accepts_configuration(base: Callable[..., Any]) -> bool:
    """Return whether ``base`` should be called with observation and configuration.

    A parameter of kind ``VAR_POSITIONAL`` (``*args``) accepts any number of
    positional arguments, so such a base is classified as accepting
    configuration too; counting parameters by name alone would silently
    drop configuration on every turn, with no crash and no signal.

    A keyword-only parameter cannot receive a second positional argument at
    all -- passing one anyway raises "multiple values for argument" on
    every turn. ``inspect.signature`` reports exactly this for
    ``functools.partial(agent, configuration=DEFAULT)``: the bound
    ``configuration`` moves to keyword-only. Only parameters able to bind
    positionally are counted here, so such a partial classifies as
    single-argument and is called with just ``observation`` -- deliberately:
    that plays the partial's own bound default rather than crashing on it,
    and the default was the base's own choice, not a value this module
    would otherwise have to invent.

    Args:
        base: The resolved base agent callable.

    Returns:
        True when ``base`` should be called with two positional arguments.
    """
    parameters = inspect.signature(base).parameters.values()
    if any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters
    ):
        return True
    return sum(1 for parameter in parameters if parameter.kind in _POSITIONAL) >= 2
