"""Alternate entrypoint: the frozen controller, plus the trained market head.

`main.py` serves the frozen controller alone. This file serves the same
controller with `market_residual.policy.MarketResidualAgent` wrapped around it,
so the two differ in exactly one thing -- whether a trained residual is allowed
to replace the controller's market orders -- and a gate that plays this file
against `main.py`'s kernel measures that one thing and nothing else.

Three details of the loader shape this file.

**The entrypoint is resolved by position, so it is named to be unmistakable.**
`kaggle_environments` execs this file into an empty namespace and takes the
*last callable in it* (`agent.get_last_callable`). The vendored kernels are the
cautionary tale: `kaito_v56_policy` defines `agent` twice and the survivor is
not the one that plays, which is why `baseline.ENTRYPOINT` pins a name instead.
So nothing here is called `agent`, the final binding is
`market_residual_agent`, and `tests/market_residual/test_market_residual_main.py`
resolves this file the way the loader does and pins that it is what comes back.
Anything appended below that function -- an import, a class, a helper -- silently
becomes the agent.

**There is no `__file__`.** The exec namespace is `{}`, so the artifact is
located through the *package's* `__file__`, which is a real imported module.
That is also where a built archive puts it, so the same line resolves in both
places.

**Module state is per-episode, not per-process.** `Environment.run` builds a
fresh `Agent` per call and `build_agent`'s closure execs this file the first
time that agent acts, so `SEAT` below is constructed once per episode even when
one worker process plays many. That is what the wrapper needs: it carries an
event clock and a GRU hidden state that must not survive into the next season.
The test pins it, because the failure mode is silent -- a season that opened
with a warmed-up hidden state would still play a whole season, slightly wrong.

`KAGGRICULTURE_RESIDUAL_TELEMETRY` exists because the counters that say how
often the residual actually deviated live on the played agent, inside a worker
process that the arena discards. Set it to a directory and each episode writes
its own `ResidualRecord` there, rewritten every turn so the file is complete
whenever it is read and no process-exit hook has to fire for the number to
survive. Unset, nothing is written and nothing is opened.
"""

import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping

import kaggriculture.market_residual
from kaggriculture.market_residual.baseline import SERVED
from kaggriculture.market_residual.numpy_policy import NumpyResidualPolicy
from kaggriculture.market_residual.policy import build_market_residual_agent

ARTIFACT = (
    Path(kaggriculture.market_residual.__file__)
    .with_name("generated")
    .joinpath("export.json")
)

TELEMETRY_VARIABLE = "KAGGRICULTURE_RESIDUAL_TELEMETRY"

_directory = os.environ.get(TELEMETRY_VARIABLE)
TELEMETRY = (
    None
    if _directory is None
    else Path(_directory) / f"episode-{os.getpid()}-{uuid.uuid4().hex}.json"
)

SEAT = build_market_residual_agent(SERVED, NumpyResidualPolicy.load(ARTIFACT))


def market_residual_agent(
    observation: Mapping[str, Any], configuration: object | None = None
) -> Mapping[str, Any]:
    """Return this seat's action for one engine turn.

    Args:
        observation: The raw observation the engine handed this seat.
        configuration: The engine configuration, passed through untouched.

    Returns:
        The frozen controller's action, unless the residual proved a legal
        replacement for this turn's market orders.
    """
    action = SEAT(observation, configuration)
    if TELEMETRY is not None:
        TELEMETRY.write_text(
            json.dumps(
                {
                    "turns": SEAT.record.turns,
                    "events": SEAT.record.events,
                    "replacements": SEAT.record.replacements,
                    "fallbacks": {
                        str(reason): count
                        for reason, count in SEAT.record.fallbacks.items()
                    },
                }
            ),
            encoding="utf-8",
        )
    return action
