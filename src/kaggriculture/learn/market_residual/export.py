"""Turning trained Torch weights into an artifact that provably plays the same.

The submitted agent runs the NumPy head. Every gate this project runs measures
the Torch head. Between them sits one file, and if that file does not reproduce
the head exactly then every gate was run on an agent nobody submitted -- a
failure with no symptom, because the exported agent still plays a whole season
and simply plays it slightly worse.

So the export is not a serializer with a validation step beside it that a
caller is trusted to run. ``export_numpy_policy`` takes the feature rows it
must reproduce, loads back what it is about to write, replays the whole
sequence through both implementations, and writes the artifact only if they
agree. An export that cannot be produced without proving parity is a narrower
thing to trust than one somebody remembered to check, and the write is atomic
so a refused export leaves no half-valid file behind for a later run to load.

The rows must be real. Random tensors would exercise a geometry this head never
sees: a market feature row is mostly one-hot, with a handful of small scalars,
and the cancellation that a projection over 741 such columns produces is not
the cancellation a Gaussian produces. The seed bank ``export_determinism``
exists so an export can play its own episodes for this.
"""

import json
from pathlib import Path
from typing import Sequence

import torch

from kaggriculture.learn.market_residual.model import MarketResidualNet
from kaggriculture.market_residual.features import MarketFeatureVector
from kaggriculture.market_residual.numpy_policy import (
    ARTIFACT_VERSION,
    PARITY_ATOL,
    MarketFeatureSchema,
    NumpyResidualPolicy,
    encode_arrays,
    model_digest,
    parameters_digest,
    runtime_digest,
)
from kaggriculture.market_residual.schema import SCHEMA_VERSION


class ExportParityError(RuntimeError):
    """Raised when an exported head does not reproduce the head it came from."""


def export_numpy_policy(
    model: MarketResidualNet,
    path: Path,
    rows: Sequence[MarketFeatureVector],
    atol: float = PARITY_ATOL,
) -> Path:
    """Write the NumPy artifact for a trained head, if and only if it matches.

    Args:
        model: The trained head. It is run under ``torch.no_grad`` in whatever
            mode the caller left it in, so an export of a head still in
            training mode exports what that head computes.
        path: Where the artifact goes. Its parent must exist.
        rows: Real market feature rows, in the order one seat produced them.
            Parity is proved over this sequence, so it must be long enough for
            the recurrence to have accumulated -- one real season's events.
        atol: The absolute deviation allowed between the two implementations.

    Returns:
        ``path``.

    Raises:
        ExportParityError: If the written artifact does not reproduce the head
            within ``atol``, in which case nothing is left at ``path``.
    """
    schema = MarketFeatureSchema.current()
    records = encode_arrays(
        {
            name: tensor.numpy(force=True)
            for name, tensor in model.exported_arrays().items()
        }
    )
    document = {
        "artifact_version": ARTIFACT_VERSION,
        "identity": {
            "schema_version": SCHEMA_VERSION,
            "feature_schema_sha256": schema.sha256,
            "runtime_sha256": runtime_digest(),
            "model_sha256": model_digest(model.config),
            "parameters_sha256": parameters_digest(records),
            "config": {
                "input_size": model.config.input_size,
                "hidden_size": model.config.hidden_size,
                "projection_size": model.config.projection_size,
                "auxiliary_size": model.config.auxiliary_size,
            },
        },
        "arrays": records,
    }
    pending = path.with_name(f"{path.name}.pending")
    pending.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")
    try:
        check_export_parity(model, NumpyResidualPolicy.load(pending), rows, atol=atol)
    except BaseException:
        pending.unlink()
        raise
    pending.replace(path)
    return path


def check_export_parity(
    model: MarketResidualNet,
    policy: NumpyResidualPolicy,
    rows: Sequence[MarketFeatureVector],
    atol: float = PARITY_ATOL,
) -> float:
    """Replay one sequence through both heads and return their worst deviation.

    The NumPy head is stepped one event at a time carrying its own state, which
    is exactly how the served agent runs it, while the Torch head sees the
    whole sequence at once, which is how training runs it. Comparing those two
    call shapes rather than two single steps is what makes a dropped
    hidden-state carry fail here.

    Args:
        model: The trained head.
        policy: The loaded export.
        rows: Real market feature rows, in the order one seat produced them.
        atol: The absolute deviation allowed between the two implementations.

    Returns:
        The largest absolute difference over every head and every event.

    Raises:
        ExportParityError: If that difference exceeds ``atol``.
    """
    features = torch.tensor([[row.values] for row in rows], dtype=torch.float32)
    with torch.no_grad():
        heads, _ = model(features, None)

    state = policy.initial_state()
    worst = 0.0
    for step, row in enumerate(rows):
        exported, state = policy.heads(row, state)
        pairs = (
            ("mode_logits", heads.mode_logits[step, 0], exported.mode_logits),
            (
                "quantity_logits",
                heads.quantity_logits[step, 0],
                exported.quantity_logits,
            ),
            ("value", heads.value[step, 0], exported.value),
            ("auxiliary", heads.auxiliary[step, 0], exported.auxiliary),
        )
        for name, reference, candidate in pairs:
            deviation = float(
                (reference - torch.as_tensor(candidate, dtype=torch.float32))
                .abs()
                .max()
            )
            if deviation > atol:
                raise ExportParityError(
                    f"{name} deviates by {deviation} at event {step}, above {atol}"
                )
            worst = max(worst, deviation)
    return worst
