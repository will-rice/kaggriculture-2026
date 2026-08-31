"""The one property the packaged residual cannot be shipped without.

Training happens in Torch on a GPU; the submitted agent has a 4 MB archive and
no time to import Torch, so it plays a NumPy re-implementation of the same
arithmetic. Every gate this project runs measures the Torch head. If the NumPy
head computes something even slightly different, every one of those gates was
run on an agent that was never submitted, and nothing would say so -- the
exported agent would simply play a bit worse than the one that passed.

So parity is checked the way the simulator's exact-bank replay and the
wrapper's action-by-action parity are checked: against real data, at a
tolerance tight enough that only last-bit rounding fits inside it. The rows are
the ones one seat of one real season actually produced, through the real event
machine and the real schema -- not random tensors, which would never exercise
the one-hot-heavy geometry the head is actually fed.

``PARITY_ATOL`` is 1e-6, chosen from a measurement rather than a preference:
over the 508 events of one real season the two implementations were measured to
diverge by at most 8.9e-08, so the bound sits an order of magnitude above the
float32 rounding floor. ``numpy_policy`` records what that does and does not
catch, and it is worth reading before treating a pass here as proof that every
weight arrived: a single element of a GRU gate matrix moved by 1e-3 lands under
this tolerance, and it is the parameter digest, not parity, that refuses it.

Two failures parity alone would not catch have their own tests. A NumPy head
that dropped the carried hidden state would still match Torch on the first
event of every sequence, so the carry is asserted directly. And a NumPy head
that quietly imported Torch would pass everything here and then fail in the
sandbox on turn zero, so one test runs it under ``python -I`` with Torch
blocked from ever being found.
"""

import base64
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy
import pytest
import torch

from kaggriculture.learn.market_residual.export import (
    ExportParityError,
    check_export_parity,
    export_numpy_policy,
)
from kaggriculture.learn.market_residual.model import (
    MarketResidualNet,
    ModelConfig,
    greedy_decision,
    mask_quantity_logits,
)
from kaggriculture.market_residual.features import MarketFeatureVector
from kaggriculture.market_residual.numpy_policy import (
    ARTIFACT_VERSION,
    PARITY_ATOL,
    ExportIntegrityError,
    NumpyResidualPolicy,
)
from kaggriculture.market_residual.schema import (
    ALLOWED_SLOTS,
    BUCKETS,
    MarketFeatureSchema,
)

# Enough real events to make a dropped hidden state and an accumulated rounding
# difference both visible, and short enough to keep the subprocess gate quick.
WINDOW = 256


@pytest.fixture(scope="module")
def model() -> MarketResidualNet:
    """Return one deterministically initialised head in evaluation mode."""
    torch.manual_seed(0)
    return MarketResidualNet(ModelConfig.current()).eval()


@pytest.fixture(scope="module")
def rows(
    event_rows: tuple[MarketFeatureVector, ...],
) -> tuple[MarketFeatureVector, ...]:
    """Return the real event window every parity check in this file uses."""
    assert len(event_rows) >= WINDOW
    return event_rows[:WINDOW]


@pytest.fixture(scope="module")
def artifact(
    tmp_path_factory: pytest.TempPathFactory,
    model: MarketResidualNet,
    rows: tuple[MarketFeatureVector, ...],
) -> Path:
    """Return one exported artifact, written once for this module."""
    return export_numpy_policy(
        model, tmp_path_factory.mktemp("export") / "residual.json", rows
    )


def torch_sequence(
    model: MarketResidualNet, rows: tuple[MarketFeatureVector, ...]
) -> tuple[torch.Tensor, ...]:
    """Return the Torch head outputs over a whole sequence, in NumPy order.

    Args:
        model: The head to run.
        rows: The market feature rows, in the order one seat produced them.

    Returns:
        Mode logits, quantity logits, value and auxiliary outputs, each with
        the single-sequence batch axis removed.
    """
    features = torch.tensor([[row.values] for row in rows], dtype=torch.float32)
    with torch.no_grad():
        heads, _ = model(features, None)
    return (
        heads.mode_logits[:, 0],
        heads.quantity_logits[:, 0],
        heads.value[:, 0],
        heads.auxiliary[:, 0],
    )


def numpy_sequence(
    policy: NumpyResidualPolicy, rows: tuple[MarketFeatureVector, ...]
) -> tuple[numpy.ndarray, ...]:
    """Return the NumPy head outputs, stepped one event at a time with carry.

    Args:
        policy: The loaded export.
        rows: The market feature rows, in the order one seat produced them.

    Returns:
        Mode logits, quantity logits, value and auxiliary outputs, stacked on
        the event axis.
    """
    state = policy.initial_state()
    collected: list[tuple[Any, ...]] = []
    for row in rows:
        heads, state = policy.heads(row, state)
        collected.append(
            (heads.mode_logits, heads.quantity_logits, heads.value, heads.auxiliary)
        )
    return tuple(numpy.stack(part) for part in zip(*collected, strict=True))


def test_the_export_reproduces_the_torch_head_over_a_real_season(
    model: MarketResidualNet, rows: tuple[MarketFeatureVector, ...], artifact: Path
) -> None:
    """Every head must agree within one accumulated float32 rounding budget."""
    policy = NumpyResidualPolicy.load(artifact)
    expected = torch_sequence(model, rows)
    produced = numpy_sequence(policy, rows)

    for reference, candidate in zip(expected, produced, strict=True):
        deviation = numpy.abs(reference.numpy(force=True) - candidate).max()
        assert deviation <= PARITY_ATOL, deviation
    assert PARITY_ATOL == 1e-6


def test_the_export_reaches_the_same_decisions_under_the_same_masks(
    model: MarketResidualNet, rows: tuple[MarketFeatureVector, ...], artifact: Path
) -> None:
    """Logit parity is only worth what the decoded action is worth."""
    policy = NumpyResidualPolicy.load(artifact)
    generator = torch.Generator().manual_seed(4)
    mask = torch.zeros(len(rows), len(ALLOWED_SLOTS), BUCKETS, dtype=torch.bool)
    mask.bernoulli_(0.3, generator=generator)
    mask[..., 0] = True

    mode_logits, quantity_logits, _value, _auxiliary = torch_sequence(model, rows)
    masked = mask_quantity_logits(quantity_logits, mask)
    state = policy.initial_state()
    for step, row in enumerate(rows):
        heads, state = policy.heads(row, state)
        expected = greedy_decision(mode_logits[step], masked[step])
        produced = policy.decide(heads, mask[step].numpy(force=True))
        assert produced == expected


def test_the_numpy_head_carries_hidden_state_across_events(artifact: Path) -> None:
    """A head that restarts each event is a head that cannot learn timing."""
    policy = NumpyResidualPolicy.load(artifact)
    row = MarketFeatureVector(
        schema=MarketFeatureSchema.current(),
        values=tuple(float(index % 3) for index in range(policy.config.input_size)),
    )

    fresh, first = policy.heads(row, policy.initial_state())
    carried, second = policy.heads(row, first)

    assert first != policy.initial_state()
    assert second != first
    assert not numpy.array_equal(fresh.mode_logits, carried.mode_logits)
    assert not numpy.array_equal(fresh.value, carried.value)


def test_the_protocol_advances_state_without_deciding_when_told_not_to(
    artifact: Path, rows: tuple[MarketFeatureVector, ...]
) -> None:
    """Watching collection must see the state a deciding run would have seen."""
    policy = NumpyResidualPolicy.load(artifact)

    decision, watched = policy.observe(rows[0], policy.initial_state(), act=False)
    played, acted = policy.observe(rows[0], policy.initial_state(), act=True)

    assert decision is None
    assert played is not None
    assert watched == acted
    assert watched != policy.initial_state()


def test_a_perturbed_weight_is_caught_by_the_parity_check(
    model: MarketResidualNet, rows: tuple[MarketFeatureVector, ...], artifact: Path
) -> None:
    """The parity check is only load-bearing if a wrong weight fails it."""
    policy = NumpyResidualPolicy.load(artifact)
    check_export_parity(model, policy, rows)
    policy.arrays["mode.weight"][0, 0] += numpy.float32(1e-3)

    with pytest.raises(ExportParityError):
        check_export_parity(model, policy, rows)


def test_export_writes_nothing_when_parity_fails(
    model: MarketResidualNet, rows: tuple[MarketFeatureVector, ...], tmp_path: Path
) -> None:
    """An artifact that does not reproduce the head must never reach the disk."""
    destination = tmp_path / "residual.json"

    with pytest.raises(ExportParityError):
        export_numpy_policy(model, destination, rows, atol=0.0)

    assert not destination.exists()


def read_artifact(path: Path) -> dict[str, Any]:
    """Return an exported artifact as a mutable document.

    Args:
        path: The artifact written by ``export_numpy_policy``.

    Returns:
        The parsed JSON document.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def write_artifact(document: Mapping[str, Any], path: Path) -> Path:
    """Write a document back out as an artifact.

    Args:
        document: The JSON-serialisable artifact body.
        path: Where to write it.

    Returns:
        ``path``.
    """
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_a_duplicate_array_key_is_refused(artifact: Path, tmp_path: Path) -> None:
    """The last duplicate wins silently in JSON; here it must not be read."""
    text = artifact.read_text(encoding="utf-8")
    body = json.loads(text)
    record = json.dumps(body["arrays"]["mode.bias"], separators=(",", ":"))
    broken = text.replace(
        f'"mode.bias":{record}', f'"mode.bias":{record},"mode.bias":{record}', 1
    )
    assert broken != text
    path = tmp_path / "duplicate.json"
    path.write_text(broken, encoding="utf-8")

    with pytest.raises(ExportIntegrityError, match="duplicate"):
        NumpyResidualPolicy.load(path)


def test_a_nonfinite_weight_is_refused(artifact: Path, tmp_path: Path) -> None:
    """A NaN that reached the weights must fail at load, not at turn 300."""
    document = read_artifact(artifact)
    record = document["arrays"]["mode.bias"]
    values = numpy.frombuffer(base64.b64decode(record["data"]), dtype=numpy.float32)
    broken = values.copy()
    broken[0] = numpy.nan
    record["data"] = base64.b64encode(broken.tobytes()).decode("ascii")

    with pytest.raises(ExportIntegrityError, match="nonfinite"):
        NumpyResidualPolicy.load(write_artifact(document, tmp_path / "nan.json"))


def test_an_array_of_the_wrong_shape_is_refused(artifact: Path, tmp_path: Path) -> None:
    """Shapes come from the declared configuration, never from the file."""
    document = read_artifact(artifact)
    document["arrays"]["mode.bias"]["shape"] = [3]

    with pytest.raises(ExportIntegrityError, match="shape"):
        NumpyResidualPolicy.load(write_artifact(document, tmp_path / "shape.json"))


def test_a_missing_array_is_refused(artifact: Path, tmp_path: Path) -> None:
    """A head silently left out of the export would play as zeros."""
    document = read_artifact(artifact)
    del document["arrays"]["auxiliary.bias"]

    with pytest.raises(ExportIntegrityError, match="auxiliary.bias"):
        NumpyResidualPolicy.load(write_artifact(document, tmp_path / "missing.json"))


def test_an_unknown_artifact_version_is_refused(artifact: Path, tmp_path: Path) -> None:
    """A newer writer must not be read by an older runtime."""
    document = read_artifact(artifact)
    document["artifact_version"] = ARTIFACT_VERSION + 1

    with pytest.raises(ExportIntegrityError, match="artifact version"):
        NumpyResidualPolicy.load(write_artifact(document, tmp_path / "version.json"))


def test_feature_schema_drift_is_refused(artifact: Path, tmp_path: Path) -> None:
    """A head trained under one column order plays a different game under another."""
    document = read_artifact(artifact)
    document["identity"]["feature_schema_sha256"] = "0" * 64

    with pytest.raises(ExportIntegrityError, match="feature schema"):
        NumpyResidualPolicy.load(write_artifact(document, tmp_path / "schema.json"))


def test_runtime_source_drift_is_refused(artifact: Path, tmp_path: Path) -> None:
    """The arithmetic parity was proved against is part of the artifact's identity."""
    document = read_artifact(artifact)
    document["identity"]["runtime_sha256"] = "0" * 64

    with pytest.raises(ExportIntegrityError, match="runtime"):
        NumpyResidualPolicy.load(write_artifact(document, tmp_path / "runtime.json"))


def test_a_rewritten_weight_breaks_the_parameter_digest(
    artifact: Path, tmp_path: Path
) -> None:
    """Content addressing has to notice a byte, not only a shape."""
    document = read_artifact(artifact)
    record = document["arrays"]["mode.bias"]
    values = numpy.frombuffer(
        base64.b64decode(record["data"]), dtype=numpy.float32
    ).copy()
    values[0] += numpy.float32(1.0)
    record["data"] = base64.b64encode(values.tobytes()).decode("ascii")

    with pytest.raises(ExportIntegrityError, match="parameters"):
        NumpyResidualPolicy.load(write_artifact(document, tmp_path / "digest.json"))


def test_the_packaged_head_runs_with_torch_absent(
    model: MarketResidualNet,
    rows: tuple[MarketFeatureVector, ...],
    artifact: Path,
    tmp_path: Path,
) -> None:
    """The sandbox has no Torch, and this is the only test that can prove it.

    ``python -I`` drops ``PYTHONPATH`` and the user site directory, and the
    training stack is blocked outright by a meta-path finder, so an import of
    Torch from anywhere under ``numpy_policy`` raises rather than succeeding
    from this checkout's virtual environment. The subprocess then replays the
    same real sequence and prints its outputs, which are compared here against
    the Torch head at the same tolerance the in-process check uses.
    """
    features = tmp_path / "rows.json"
    features.write_text(
        json.dumps([list(row.values) for row in rows]), encoding="utf-8"
    )
    script = """
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1])))


class Blocked:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"torch", "lightning", "optuna", "wandb"}:
            raise ImportError(f"{name} is not available in the packaged runtime")
        return None


sys.meta_path.insert(0, Blocked())

from kaggriculture.market_residual.features import MarketFeatureVector
from kaggriculture.market_residual.numpy_policy import NumpyResidualPolicy
from kaggriculture.market_residual.schema import MarketFeatureSchema

policy = NumpyResidualPolicy.load(Path(sys.argv[2]))
schema = MarketFeatureSchema.current()
state = policy.initial_state()
modes = []
values = []
for row in json.loads(Path(sys.argv[3]).read_text()):
    heads, state = policy.heads(
        MarketFeatureVector(schema=schema, values=tuple(row)), state
    )
    modes.append([float(logit) for logit in heads.mode_logits])
    values.append(float(heads.value))
print(json.dumps({
    "modes": modes,
    "values": values,
    "loaded": sorted({name.split(".")[0] for name in sys.modules}),
}))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(Path("src").resolve()),
            str(artifact),
            str(features),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, result.stderr
    evidence = json.loads(result.stdout.splitlines()[-1])
    assert "torch" not in evidence["loaded"]
    mode_logits, _quantities, value, _auxiliary = torch_sequence(model, rows)
    assert (
        numpy.abs(numpy.array(evidence["modes"]) - mode_logits.numpy(force=True)).max()
        <= PARITY_ATOL
    )
    assert (
        numpy.abs(numpy.array(evidence["values"]) - value.numpy(force=True)).max()
        <= PARITY_ATOL
    )
