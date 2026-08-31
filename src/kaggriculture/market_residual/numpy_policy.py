"""The residual the submission actually plays: one GRU, in NumPy, from a file.

The head is trained in Torch on a GPU and served here without it. That is not
an optimisation. The sandbox has two cores and a 4 MB archive, importing Torch
costs about eleven seconds of a sixty-second overage pool, and the wheels alone
are fifty times the archive budget. So the trained parameters are exported as
data and the arithmetic is restated once, here, in the only dependency the
engine already brings.

Restating arithmetic is exactly the kind of second implementation this package
otherwise refuses, and it is allowed only because the drift it invites is
*checked* rather than hoped for:
``tests/market_residual/test_export.py`` runs both implementations over the
feature rows of a real season and requires they agree to ``PARITY_ATOL``, and
``learn.market_residual.export`` refuses to write an artifact that has not
passed that check. An export nobody could produce without proving parity is a
narrower thing to trust than an export somebody remembered to validate.

Three parts of the file format are load-bearing rather than decorative.

**Identity.** An artifact carries the feature-schema digest it was trained
under, the digest of this module's own source, and a digest over its
parameters. The first two are the two ways a stored artifact goes quietly
wrong: a column order that moved, and these equations changing under an
artifact that was validated against the old ones. Neither raises on its own --
the head keeps returning plausible numbers and plays a different game -- so
both are refused at load, in the process that is about to serve them.

**Finite masking.** Illegal quantities are filled with the finite float32
minimum, not ``-inf``. A slot with no legal bucket at all is an ordinary
position in this game, and ``-inf`` would make its softmax NaN and its argmax
arbitrary. A finite fill keeps every number the merge boundary inspects
finite, which is what lets ``ResidualDecision.finite`` mean "the head is
healthy" rather than "the mask was empty".

**State as a tuple of floats.** ``ResidualInference`` hands the recurrent state
back and forth as a plain tuple, so the wrapper can carry it, snapshot it, and
replay it without knowing anything about arrays. The conversion costs one copy
of ``hidden_size`` floats per market event, against roughly a thousand events
per season.

The runtime decodes greedily and without a legality mask, because the protocol
hands the head one feature row and nothing else. Nothing is trusted to that:
``actions.merge_residual_action`` re-derives legality from the observation and
returns the frozen controller's own action for anything it cannot prove, so an
illegal proposal costs a deferral, never an illegal order.
"""

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy

from kaggriculture.features import PRODUCT_NAMES
from kaggriculture.market_residual.actions import ResidualDecision, ResidualMode
from kaggriculture.market_residual.features import MarketFeatureVector
from kaggriculture.market_residual.schema import (
    ALLOWED_SLOTS,
    BUCKETS,
    SCHEMA_VERSION,
    MarketFeatureSchema,
)

ARTIFACT_VERSION = 1

# Every stored array is float32: it is what the head is trained in, and an
# export that widened it would be claiming a precision the training never had.
DTYPE = numpy.dtype(numpy.float32)

# The reset, update and new gates of one GRU cell, in PyTorch's packing order.
GATES = 3

MASK_FILL = float(numpy.finfo(DTYPE).min)

# Measured, not preferred. Over the 508 market events of one real season the two
# implementations of this head diverge by at most 8.9e-08 -- roughly one float32
# ulp at unit scale, which is what two BLAS libraries summing the same weights
# in a different order costs -- so 1e-6 sits about an order of magnitude above
# the rounding floor and well below anything an implementation error produces.
#
# What that buys is not uniform, and pretending otherwise would be the way to
# misread a passing check. Measured on the same season, a single parameter moved
# by 1e-3 shifts the exported logits by 1.0e-3 in a head bias, 1.0e-4 in a head
# weight and 3.4e-5 in the projection -- all caught -- but only 1.0e-7 in one
# element of either GRU matrix, which is under this bound. One element of a
# 192x192 gate matrix is a 192nd of one gate of one hidden unit, the update gate
# contracts it, and the readout averages 192 units: four orders of attenuation,
# and a property of the architecture rather than of the check.
#
# So the two guards divide the work. `parameters_sha256` refuses any artifact
# whose bytes are not the bytes that were exported, which is what catches a
# single changed weight. Parity catches what a digest cannot: a transposed
# matrix, a mis-sliced gate, a dropped hidden-state carry, a different
# activation -- errors in the arithmetic, which move everything at once.
PARITY_ATOL = 1e-6

MODES: tuple[ResidualMode, ...] = tuple(ResidualMode)


class ExportIntegrityError(RuntimeError):
    """Raised when an artifact is not one this runtime may serve."""


@dataclass(frozen=True)
class ModelConfig:
    """The shape of one residual head, shared by the trainer and the runtime.

    Declared here rather than beside the Torch module so that one description
    of the architecture sizes the parameters, writes the export, and validates
    it at load. A second table on the training side would agree with this one
    until the day someone widened only one of them.
    """

    input_size: int
    hidden_size: int = 192
    projection_size: int = 192
    auxiliary_size: int = 3 * len(PRODUCT_NAMES)

    @classmethod
    def current(cls) -> "ModelConfig":
        """Return the configuration sized by the schema in force."""
        return cls(input_size=MarketFeatureSchema.current().width)

    @property
    def slots(self) -> int:
        """Return how many market slots the residual may propose for."""
        return len(ALLOWED_SLOTS)

    @property
    def buckets(self) -> int:
        """Return how many canonical quantity buckets each slot chooses from."""
        return BUCKETS

    def array_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return every exported array's name and exact shape.

        Returns:
            The canonical parameter names in export order, mapped to the shape
            each one must have. Torch's own ``(out, in)`` linear layout and
            ``(3 * hidden, ...)`` gate packing are kept, so the export is a
            transcription rather than a rearrangement.
        """
        hidden = self.hidden_size
        return {
            "projection.weight": (self.projection_size, self.input_size),
            "projection.bias": (self.projection_size,),
            "gru.weight_ih": (GATES * hidden, self.projection_size),
            "gru.weight_hh": (GATES * hidden, hidden),
            "gru.bias_ih": (GATES * hidden,),
            "gru.bias_hh": (GATES * hidden,),
            "mode.weight": (len(MODES), hidden),
            "mode.bias": (len(MODES),),
            "quantities.weight": (self.slots * self.buckets, hidden),
            "quantities.bias": (self.slots * self.buckets,),
            "value.weight": (1, hidden),
            "value.bias": (1,),
            "auxiliary.weight": (self.auxiliary_size, hidden),
            "auxiliary.bias": (self.auxiliary_size,),
        }


@dataclass(frozen=True)
class ExportIdentity:
    """Everything a served artifact has to prove before it is served.

    Args:
        schema_version: The market schema generation the head was built under.
        feature_schema_sha256: The digest of the column layout it was trained
            on, which must be the layout the runtime is about to encode.
        runtime_sha256: The digest of this module's source, which is the
            arithmetic the artifact's parity was proved against.
        model_sha256: The digest of the architecture: configuration plus every
            array name and shape.
        parameters_sha256: The digest of the parameter bytes themselves.
        config: The head's declared shape.
    """

    schema_version: int
    feature_schema_sha256: str
    runtime_sha256: str
    model_sha256: str
    parameters_sha256: str
    config: ModelConfig


@dataclass(frozen=True)
class ResidualHeads:
    """One market event's outputs, before anything has been decided from them."""

    mode_logits: numpy.ndarray
    quantity_logits: numpy.ndarray
    value: float
    auxiliary: numpy.ndarray


def canonical_digest(document: object) -> str:
    """Return the SHA-256 of a document's canonical JSON form.

    Args:
        document: Any JSON-serialisable value.

    Returns:
        The hex digest over compact, key-sorted JSON, so an identity survives a
        reordering of the writer's dictionaries.
    """
    canonical = json.dumps(document, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def runtime_digest() -> str:
    """Return the SHA-256 of this module's source.

    The exported artifact records what this file computed when its parity was
    proved. Editing the equations then invalidates every artifact validated
    against the old ones, which is the intent: the alternative is a stored head
    that keeps loading and quietly plays something else.

    Returns:
        The hex digest of this file's bytes.
    """
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def model_digest(config: ModelConfig) -> str:
    """Return the digest of an architecture, independent of its weights.

    Args:
        config: The head's declared shape.

    Returns:
        A hex digest over the configuration and every array name and shape.
    """
    return canonical_digest(
        {
            "config": {
                "input_size": config.input_size,
                "hidden_size": config.hidden_size,
                "projection_size": config.projection_size,
                "auxiliary_size": config.auxiliary_size,
                "slots": config.slots,
                "buckets": config.buckets,
            },
            "arrays": {
                name: list(shape) for name, shape in config.array_shapes().items()
            },
        }
    )


def encode_arrays(arrays: Mapping[str, numpy.ndarray]) -> dict[str, dict[str, Any]]:
    """Return arrays as portable shape/dtype/base64 records.

    Args:
        arrays: Parameter arrays keyed by canonical name.

    Returns:
        One record per array, in the order given.

    Raises:
        ExportIntegrityError: If an array is not the exported dtype.
    """
    records: dict[str, dict[str, Any]] = {}
    for name, array in arrays.items():
        if array.dtype != DTYPE:
            raise ExportIntegrityError(f"{name} has dtype {array.dtype}, not {DTYPE}")
        records[name] = {
            "shape": list(array.shape),
            "dtype": DTYPE.name,
            "data": base64.b64encode(numpy.ascontiguousarray(array).tobytes()).decode(
                "ascii"
            ),
        }
    return records


def parameters_digest(records: Mapping[str, Mapping[str, Any]]) -> str:
    """Return the digest of exported parameter records.

    Args:
        records: The shape/dtype/base64 records written to the artifact.

    Returns:
        A hex digest over exactly those records, so a rewritten byte is caught
        even when the shape it sits in is still right.
    """
    return canonical_digest({name: dict(record) for name, record in records.items()})


def _no_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    """Return a JSON object, refusing one that declares a key twice.

    ``json.loads`` keeps the last of a repeated key and says nothing, so an
    artifact could carry two ``mode.weight`` entries and serve whichever the
    writer happened to put second.

    Args:
        pairs: The key/value pairs of one JSON object, in file order.

    Returns:
        The object as a dictionary.

    Raises:
        ExportIntegrityError: If a key appears more than once.
    """
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ExportIntegrityError(f"artifact declares duplicate key {key!r}")
        document[key] = value
    return document


def decode_arrays(
    records: Mapping[str, Any], config: ModelConfig
) -> dict[str, numpy.ndarray]:
    """Return the parameter arrays an artifact declares, or refuse it.

    Args:
        records: The artifact's array records.
        config: The declared head shape the arrays must fill exactly.

    Returns:
        Every array named by ``config.array_shapes``, in that order.

    Raises:
        ExportIntegrityError: If an array is missing, unknown, of the wrong
            dtype or shape, or holds a nonfinite value.
    """
    shapes = config.array_shapes()
    unknown = sorted(set(records) - set(shapes))
    if unknown:
        raise ExportIntegrityError(f"artifact declares unknown arrays: {unknown}")
    arrays: dict[str, numpy.ndarray] = {}
    for name, shape in shapes.items():
        record = records.get(name)
        if record is None:
            raise ExportIntegrityError(f"artifact is missing array {name!r}")
        if record.get("dtype") != DTYPE.name:
            raise ExportIntegrityError(
                f"{name} has dtype {record.get('dtype')!r}, not {DTYPE.name!r}"
            )
        values = numpy.frombuffer(base64.b64decode(record["data"]), dtype=DTYPE).copy()
        if tuple(record.get("shape", ())) != shape or values.size != int(
            numpy.prod(shape)
        ):
            raise ExportIntegrityError(
                f"{name} has shape {record.get('shape')} holding {values.size} "
                f"values, expected {list(shape)}"
            )
        if not numpy.isfinite(values).all():
            raise ExportIntegrityError(f"{name} holds a nonfinite value")
        arrays[name] = values.reshape(shape)
    return arrays


def sigmoid(values: numpy.ndarray) -> numpy.ndarray:
    """Return the logistic function, evaluated without overflowing.

    Args:
        values: Any real array.

    Returns:
        ``1 / (1 + exp(-values))``, computed on whichever side of zero keeps
        the exponent negative, so a saturated gate is 0.0 or 1.0 rather than a
        warning and a NaN.
    """
    exponent = numpy.exp(-numpy.abs(values))
    return numpy.where(values >= 0, 1.0, exponent) / (1.0 + exponent)


def mask_quantity_logits(logits: numpy.ndarray, mask: numpy.ndarray) -> numpy.ndarray:
    """Return quantity logits with illegal buckets pushed to a finite minimum.

    Args:
        logits: Quantity logits whose last axis is the bucket vocabulary.
        mask: A boolean array of the same shape, true where a bucket is legal.

    Returns:
        The logits with every illegal entry replaced by ``MASK_FILL``.
    """
    return numpy.where(mask, logits, numpy.asarray(MASK_FILL, dtype=logits.dtype))


def greedy_decision(
    mode_logits: numpy.ndarray, quantity_logits: numpy.ndarray
) -> ResidualDecision:
    """Return the argmax action for one market event.

    Buckets are decoded whatever the mode says, because the merge reads them
    only when the mode is ``REPLACE`` and a decision that carried them either
    way is one fewer branch to get wrong.

    Args:
        mode_logits: The two mode logits for this event.
        quantity_logits: ``(slots, buckets)`` logits, already masked.

    Returns:
        The decision to hand the merge boundary, with ``finite`` set from the
        logits themselves rather than assumed.
    """
    finite = bool(
        numpy.isfinite(mode_logits).all() and numpy.isfinite(quantity_logits).all()
    )
    return ResidualDecision(
        mode=MODES[int(numpy.argmax(mode_logits))],
        buckets=tuple(int(bucket) for bucket in numpy.argmax(quantity_logits, axis=-1)),
        finite=finite,
    )


class NumpyResidualPolicy:
    """A trained market head, loaded from an artifact and served without Torch."""

    def __init__(
        self, identity: ExportIdentity, arrays: Mapping[str, numpy.ndarray]
    ) -> None:
        """Bind one validated artifact.

        Args:
            identity: The artifact's proven identity.
            arrays: Its parameter arrays, keyed by canonical name.
        """
        self.identity = identity
        self.config = identity.config
        self.arrays = dict(arrays)

    @classmethod
    def load(cls, path: Path) -> "NumpyResidualPolicy":
        """Return the head an artifact holds, having proved it may be served.

        Args:
            path: The artifact written by ``export.export_numpy_policy``.

        Returns:
            The loaded head.

        Raises:
            ExportIntegrityError: If the artifact repeats a key, was written by
                a newer format, was built under another feature schema or
                another version of this module's arithmetic, declares an
                architecture this runtime does not agree with, or holds
                parameters that do not match their recorded digest.
        """
        document = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys
        )
        if document.get("artifact_version") != ARTIFACT_VERSION:
            raise ExportIntegrityError(
                f"unknown artifact version {document.get('artifact_version')!r}, "
                f"expected {ARTIFACT_VERSION}"
            )
        recorded = document["identity"]
        config = ModelConfig(
            input_size=int(recorded["config"]["input_size"]),
            hidden_size=int(recorded["config"]["hidden_size"]),
            projection_size=int(recorded["config"]["projection_size"]),
            auxiliary_size=int(recorded["config"]["auxiliary_size"]),
        )
        schema = MarketFeatureSchema.current()
        if recorded["schema_version"] != SCHEMA_VERSION:
            raise ExportIntegrityError(
                f"artifact schema version {recorded['schema_version']!r} is not "
                f"the served {SCHEMA_VERSION}"
            )
        if recorded["feature_schema_sha256"] != schema.sha256:
            raise ExportIntegrityError(
                "artifact was built under another feature schema: "
                f"{recorded['feature_schema_sha256']} is not {schema.sha256}"
            )
        if config.input_size != schema.width:
            raise ExportIntegrityError(
                f"artifact takes {config.input_size} columns, the feature schema "
                f"produces {schema.width}"
            )
        if recorded["runtime_sha256"] != runtime_digest():
            raise ExportIntegrityError(
                "artifact was validated against another version of the runtime "
                f"head: {recorded['runtime_sha256']} is not {runtime_digest()}"
            )
        if recorded["model_sha256"] != model_digest(config):
            raise ExportIntegrityError(
                "artifact declares an architecture this runtime does not build"
            )
        arrays = decode_arrays(document["arrays"], config)
        if recorded["parameters_sha256"] != parameters_digest(document["arrays"]):
            raise ExportIntegrityError("artifact parameters do not match their digest")
        return cls(
            ExportIdentity(
                schema_version=int(recorded["schema_version"]),
                feature_schema_sha256=recorded["feature_schema_sha256"],
                runtime_sha256=recorded["runtime_sha256"],
                model_sha256=recorded["model_sha256"],
                parameters_sha256=recorded["parameters_sha256"],
                config=config,
            ),
            arrays,
        )

    def initial_state(self) -> tuple[float, ...]:
        """Return the zero hidden state an episode starts from."""
        return (0.0,) * self.config.hidden_size

    def heads(
        self, features: MarketFeatureVector, state: tuple[float, ...]
    ) -> tuple[ResidualHeads, tuple[float, ...]]:
        """Run one market event through the projection, the GRU and the heads.

        These are PyTorch's own GRU equations, in PyTorch's gate order -- reset,
        update, new -- with the new gate's hidden term taken *after* the reset
        gate multiplies it, which is the detail that distinguishes this cell
        from the original paper's.

        Args:
            features: The canonical row for this market event.
            state: The state returned at the previous event of this episode.

        Returns:
            This event's head outputs and the state to carry forward.

        Raises:
            ExportIntegrityError: If the row was encoded under another schema,
                or the carried state is not this head's hidden size.
        """
        if features.schema.sha256 != self.identity.feature_schema_sha256:
            raise ExportIntegrityError("feature row was built under another schema")
        if len(state) != self.config.hidden_size:
            raise ExportIntegrityError(
                f"carried state has {len(state)} values, expected "
                f"{self.config.hidden_size}"
            )
        hidden = self.config.hidden_size
        row = numpy.asarray(features.values, dtype=DTYPE)
        previous = numpy.asarray(state, dtype=DTYPE)

        projected = (
            self.arrays["projection.weight"] @ row + self.arrays["projection.bias"]
        )
        projected = projected * sigmoid(projected)

        input_gates = (
            self.arrays["gru.weight_ih"] @ projected + self.arrays["gru.bias_ih"]
        )
        hidden_gates = (
            self.arrays["gru.weight_hh"] @ previous + self.arrays["gru.bias_hh"]
        )
        reset = sigmoid(input_gates[:hidden] + hidden_gates[:hidden])
        update = sigmoid(
            input_gates[hidden : 2 * hidden] + hidden_gates[hidden : 2 * hidden]
        )
        candidate = numpy.tanh(
            input_gates[2 * hidden :] + reset * hidden_gates[2 * hidden :]
        )
        current = (1.0 - update) * candidate + update * previous

        return (
            ResidualHeads(
                mode_logits=self.arrays["mode.weight"] @ current
                + self.arrays["mode.bias"],
                quantity_logits=(
                    self.arrays["quantities.weight"] @ current
                    + self.arrays["quantities.bias"]
                ).reshape(self.config.slots, self.config.buckets),
                value=float(
                    (self.arrays["value.weight"] @ current)[0]
                    + self.arrays["value.bias"][0]
                ),
                auxiliary=self.arrays["auxiliary.weight"] @ current
                + self.arrays["auxiliary.bias"],
            ),
            tuple(float(value) for value in current),
        )

    def decide(
        self, heads: ResidualHeads, mask: numpy.ndarray | None = None
    ) -> ResidualDecision:
        """Return the greedy action for one event's heads.

        Args:
            heads: The outputs of ``heads``.
            mask: ``(slots, buckets)`` legality, or ``None`` when the caller has
                none -- which is the runtime's case, since ``ResidualInference``
                hands the head a feature row and nothing else. The merge
                boundary re-derives legality from the observation either way.

        Returns:
            The decision to hand the merge boundary.
        """
        quantities = heads.quantity_logits
        if mask is not None:
            quantities = mask_quantity_logits(quantities, mask)
        return greedy_decision(heads.mode_logits, quantities)

    def observe(
        self,
        features: MarketFeatureVector,
        state: tuple[float, ...],
        *,
        act: bool,
    ) -> tuple[ResidualDecision | None, tuple[float, ...]]:
        """Advance the recurrent state, and decide only when asked to.

        Args:
            features: The canonical row for this market event.
            state: The state returned at the previous event of this episode.
            act: Whether this call's decision will be played. When ``False`` the
                state still advances and no decision is returned, so a watching
                collection run sees the state a deciding run would have seen.

        Returns:
            The decision to merge, or ``None``, and the state to carry forward.
        """
        heads, current = self.heads(features, state)
        return (self.decide(heads) if act else None), current
