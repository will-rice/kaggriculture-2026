"""What the loader actually picks out of `market_residual_main.py`.

The competition runner does not import this entrypoint. It execs the file into
an empty namespace and takes the last callable in it, so the served agent is
decided by position, and the vendored kernels are the standing proof that
position is easy to get wrong: `kaito_v56_policy` binds `agent` twice and the
one that survives is not the one that plays. A file that resolved to the wrong
callable would still play whole seasons and simply play them differently, which
is why this is pinned by resolving the file exactly the way the loader does --
`kaggle_environments.agent.get_last_callable`, the function the runner itself
calls -- rather than by importing it.

The second property here is quieter and worse. The wrapper carries an event
clock and a GRU hidden state for one season; the loader's per-episode exec is
what keeps those from leaking into the next game of the same worker process.
Nothing raises if that stops being true -- a season would open with a warmed-up
hidden state and a stale heartbeat, and play a whole season slightly wrong --
so two real episodes are played in one process and their telemetry is required
to describe two independent seasons.

The artifact the entrypoint serves is a generated build product and is not
committed, so the fixture below writes a deterministically initialised head to
the same address when none is there and removes it again. An artifact that was
already present is left alone: it is the one the operator exported and gated,
and these tests are about the file that serves it, not about its weights.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from types import FunctionType

import pytest
import torch
from kaggle_environments import make
from kaggle_environments.agent import get_last_callable

import kaggriculture.market_residual
from kaggriculture.constants import ENVIRONMENT
from kaggriculture.learn.market_residual.export import export_numpy_policy
from kaggriculture.learn.market_residual.model import MarketResidualNet, ModelConfig
from kaggriculture.market_residual.features import MarketFeatureVector
from kaggriculture.market_residual.numpy_policy import NumpyResidualPolicy
from kaggriculture.market_residual.policy import MarketResidualAgent

MODULE = Path(__file__).parents[2] / "market_residual_main.py"

ARTIFACT = (
    Path(kaggriculture.market_residual.__file__)
    .with_name("generated")
    .joinpath("export.json")
)

# Long enough that both seats act on many turns, short enough that two whole
# episodes stay a few seconds.
PROBE_STEPS = 40

# Outside every declared bank in `schema.SEED_BANKS`, for the same reason the
# fixtures are: a test is not a measurement and may not spend a declared seed.
PROBE_SEED = 4243


@pytest.fixture(scope="module")
def artifact(event_rows: tuple[MarketFeatureVector, ...]) -> Iterator[Path]:
    """Yield the artifact address, writing a stand-in only if none is there."""
    if ARTIFACT.exists():
        yield ARTIFACT
        return
    torch.manual_seed(0)
    model = MarketResidualNet(ModelConfig.current()).eval()
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    export_numpy_policy(model, ARTIFACT, event_rows[:256])
    yield ARTIFACT
    ARTIFACT.unlink()


def loaded_namespace() -> dict[str, object]:
    """Return this file's namespace after the loader's own exec of it."""
    namespace: dict[str, object] = {}
    exec(compile(MODULE.read_text(encoding="utf-8"), str(MODULE), "exec"), namespace)  # noqa: S102
    return namespace


def test_the_loader_resolves_the_named_entrypoint(artifact: Path) -> None:
    """The last callable in the file is the one this file means to serve."""
    resolved = get_last_callable(MODULE.read_text(encoding="utf-8"), path=str(MODULE))
    assert isinstance(resolved, FunctionType)
    assert resolved.__name__ == "market_residual_agent"


def test_the_file_binds_nothing_called_agent(artifact: Path) -> None:
    """No `agent` exists here to be mistaken for the entrypoint."""
    assert "agent" not in loaded_namespace()


def test_the_entrypoint_serves_the_exported_head(artifact: Path) -> None:
    """The wrapper is built on the exported artifact, not on a deferring stub."""
    seat = loaded_namespace()["SEAT"]
    assert isinstance(seat, MarketResidualAgent)
    assert isinstance(seat.residual, NumpyResidualPolicy)


def test_each_episode_gets_its_own_seat(
    artifact: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two episodes in one process are two seasons, not one continued twice."""
    monkeypatch.setenv("KAGGRICULTURE_RESIDUAL_TELEMETRY", str(tmp_path))
    for _ in range(2):
        environment = make(
            ENVIRONMENT,
            configuration={"episodeSteps": PROBE_STEPS, "seed": PROBE_SEED},
            debug=False,
        )
        environment.run([str(MODULE), "starter"])

    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(tmp_path.iterdir())
    ]
    assert len(records) == 2
    assert {record["turns"] for record in records} == {PROBE_STEPS - 1}
