"""Validated, serializable configuration for the native Toad trainer."""

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Self, Sequence, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    ValidationInfo,
    model_validator,
)

from kaggriculture.constants import BOARD_SIZE, EPISODE_STEPS
from kaggriculture.learn.toad_loss import (
    ADAM_EPS,
    CLIP_GRADS,
    DISCOUNTING,
    ENTROPY_COST,
    LEARNING_RATE,
    LMB,
    MIN_LR_MOD,
    TOTAL_STEPS,
    UNROLL_LENGTH,
    VALUE_WARMUP_BATCHES,
)
from kaggriculture.learn.toad_reward import MONEY_WEIGHT
from kaggriculture.sim.state import CROP_NAMES, PRODUCT_NAMES, SHED_NAMES

Precision = Literal["32-true", "bf16-mixed"]
CompileMode = Literal["default", "reduce-overhead", "max-autotune"]
# The largest centered odd window no wider than the board declares the amount
# of edge padding this implementation supports; the 7x7 default remains inside it.
MAX_LOCAL_PATCH_SIZE = BOARD_SIZE - 1 if BOARD_SIZE % 2 == 0 else BOARD_SIZE


class ModelConfig(BaseModel):
    """Policy architecture settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    blocks: PositiveInt = 8
    channels: PositiveInt = 128
    kernel_size: Literal[3, 5] = 3
    activation: Literal["relu", "leaky_relu"] = "relu"
    value_bound: PositiveFloat | None = 1.0
    recurrent: bool = False
    recurrent_channels: PositiveInt = 128
    recurrent_kernel_size: Literal[3, 5] = 3
    recurrent_layers: PositiveInt = 1
    transformer: bool = False
    transformer_blocks: NonNegativeInt = 0
    transformer_heads: PositiveInt = 4
    transformer_mlp_ratio: PositiveInt = 2
    local_patch: bool = False
    local_patch_size: PositiveInt = 7
    local_patch_blocks: NonNegativeInt = 0
    belief: bool = False
    belief_size: PositiveInt = len(SHED_NAMES) + len(CROP_NAMES) + len(PRODUCT_NAMES)
    belief_loss_weight: NonNegativeFloat = 0.0
    belief_feedback: bool = False
    interaction_value: bool = False

    @classmethod
    def control(cls, blocks: int = 8, channels: int = 128) -> Self:
        """Return the permanent all-optional-features-disabled topology."""
        return cls(
            blocks=blocks,
            channels=channels,
            recurrent=False,
            transformer=False,
            local_patch=False,
            belief=False,
            interaction_value=False,
        )

    @model_validator(mode="after")
    def validate_optional_model_paths(self) -> Self:
        """Reject inconsistent dimensions before torch modules are created."""
        if (self.belief_feedback or self.belief_loss_weight > 0) and not self.belief:
            raise ValueError("belief feedback/loss requires the belief head")
        expected_belief_size = len(SHED_NAMES) + len(CROP_NAMES) + len(PRODUCT_NAMES)
        if self.belief and self.belief_size != expected_belief_size:
            raise ValueError(
                "belief_size must match the fixed private-state target schema "
                f"({expected_belief_size})"
            )
        if self.local_patch and self.local_patch_size % 2 == 0:
            raise ValueError("local patch size must be odd")
        if self.local_patch and self.local_patch_size > MAX_LOCAL_PATCH_SIZE:
            raise ValueError(f"local patch size must be at most {MAX_LOCAL_PATCH_SIZE}")
        if self.transformer and self.transformer_blocks == 0:
            raise ValueError("transformer requires positive transformer_blocks")
        if (self.transformer or self.interaction_value) and self.channels < 2:
            raise ValueError("attention-enabled channels must be at least 2")
        if (
            self.transformer or self.interaction_value
        ) and self.channels % self.transformer_heads:
            raise ValueError("transformer channels must divide evenly across heads")
        return self


class TeacherSpec(BaseModel):
    """Immutable provenance, topology override, and valid heads for one teacher."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    checkpoint: Path
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    blocks: PositiveInt | None = None
    operation: bool = True
    quantity: bool = False
    market: bool = True
    value: bool = False


class PopulationConfig(BaseModel):
    """Actor population and opponent-pool settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    selfplay: float = 0.5
    scripted: float = 0.5
    frozen_opponent: float = 0.0
    teacher_distill: float = 0.0
    scripted_opponent: str = "economic"
    teacher: TeacherSpec | None = None
    actor_sync_every_rounds: PositiveInt = 4
    environments_per_rank: PositiveInt = 24
    collection_processes: PositiveInt = 24
    pool_capacity: PositiveInt = 8
    initial_snapshots: tuple[Path, ...] = ()
    snapshot_every_environment_steps: PositiveInt | None = None
    snapshot_at_start: bool = False
    pool_sampling: Literal["uniform"] = "uniform"
    pool_replacement: Literal["oldest"] = "oldest"
    population_seed: int = 0

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_teacher(cls, payload: object) -> object:
        """Atomically migrate the one-release path/depth teacher input pair."""
        if not isinstance(payload, Mapping):
            return payload
        migrated = dict(payload)
        legacy_fields = {"teacher_checkpoint", "teacher_blocks"}.intersection(migrated)
        if "teacher" in migrated and legacy_fields:
            fields = ", ".join(sorted(legacy_fields))
            raise ValueError(
                "population.teacher conflicts with legacy teacher fields: " + fields
            )
        if not legacy_fields:
            return migrated
        checkpoint = migrated.pop("teacher_checkpoint", None)
        blocks = migrated.pop("teacher_blocks", None)
        if checkpoint is None:
            if blocks is not None:
                raise ValueError("teacher_blocks requires teacher_checkpoint")
            migrated["teacher"] = None
        else:
            migrated["teacher"] = {"checkpoint": checkpoint, "blocks": blocks}
        return migrated


class EntropyControllerConfig(BaseModel):
    """Immutable target schedule and multiplicative update bounds for one head."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
    )

    initial_target: NonNegativeFloat
    target_change_per_step: float = 0.0
    target_floor: NonNegativeFloat = 0.0
    initial_multiplier: NonNegativeFloat
    multiplier_change_per_step: NonNegativeFloat
    minimum: NonNegativeFloat = 0.0
    maximum: PositiveFloat

    @model_validator(mode="after")
    def validate_multiplier_bounds(self) -> Self:
        """Keep the initial multiplier inside its declared closed interval."""
        if not self.minimum <= self.initial_multiplier <= self.maximum:
            raise ValueError("initial_multiplier must be between minimum and maximum")
        return self


def _default_entropy_controller() -> EntropyControllerConfig:
    """Return inert defaults used only after adaptive entropy is enabled."""
    return EntropyControllerConfig(
        initial_target=0.0,
        initial_multiplier=ENTROPY_COST,
        multiplier_change_per_step=0.0,
        maximum=1.0,
    )


class EntropyControllersConfig(BaseModel):
    """Typed target-entropy controller settings for each action head."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    operation: EntropyControllerConfig = Field(
        default_factory=_default_entropy_controller
    )
    quantity: EntropyControllerConfig = Field(
        default_factory=_default_entropy_controller
    )
    market: EntropyControllerConfig = Field(default_factory=_default_entropy_controller)

    def items(self) -> tuple[tuple[str, EntropyControllerConfig], ...]:
        """Return named controller configs in stable action order."""
        return (
            ("operation", self.operation),
            ("quantity", self.quantity),
            ("market", self.market),
        )


class OptimizerConfig(BaseModel):
    """Loss coefficients and optimizer schedule settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lr: PositiveFloat = LEARNING_RATE
    adam_eps: PositiveFloat = ADAM_EPS
    gamma: float = DISCOUNTING
    lmb: float = LMB
    entropy_cost: float = ENTROPY_COST
    adaptive_entropy: bool = False
    entropy: EntropyControllersConfig = Field(default_factory=EntropyControllersConfig)
    teacher_kl_cost: float = 0.0
    teacher_baseline_cost: NonNegativeFloat = 0.0
    vtrace_pg_cost: NonNegativeFloat = 1.0
    upgo_pg_cost: NonNegativeFloat = 1.0
    baseline_cost: NonNegativeFloat = 1.0
    clip_grad_norm: PositiveFloat = CLIP_GRADS
    final_lr_multiplier: float = Field(default=MIN_LR_MOD, ge=0.0, le=1.0)
    unroll_length: PositiveInt = UNROLL_LENGTH
    batch_segments: PositiveInt = 4
    value_warmup_batches: NonNegativeInt = VALUE_WARMUP_BATCHES
    value_passes: NonNegativeInt = 0


class CompileConfig(BaseModel):
    """Immutable optional ``torch.compile`` identity for one Toad run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    mode: CompileMode = "default"
    fullgraph: bool = False
    dynamic: bool = False


class RuntimeConfig(BaseModel):
    """Resolved Lightning execution settings for the native trainer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = 0
    accelerator: Literal["auto", "cpu", "gpu"] = "auto"
    devices: PositiveInt | tuple[NonNegativeInt, ...] | Literal["auto"] = 1
    num_nodes: PositiveInt = 1
    strategy: Literal["auto", "ddp"] = "auto"
    precision: Precision = "32-true"
    deterministic: bool = False
    benchmark: bool | None = None
    total_environment_steps: PositiveInt = TOTAL_STEPS
    log_every_n_steps: PositiveInt = 1
    checkpoint_every_environment_steps: PositiveInt = 1_000_000
    profiler: Literal["simple", "advanced"] | None = None
    output_dir: Path = Path("run/toad")
    resume: Path | None = None
    compile: CompileConfig = Field(default_factory=CompileConfig)
    rollout_backend: Literal["reference", "native"] = "reference"
    rollout_device: Literal["cpu", "cuda"] = "cpu"
    rollout_cuda_graph: bool = False

    @model_validator(mode="after")
    def validate_device_topology(self) -> Self:
        """Reject CPU device indexes before any Trainer can reinterpret them."""
        if self.accelerator == "cpu" and isinstance(self.devices, tuple):
            raise ValueError("CPU accelerator does not accept explicit device indexes")
        return self


class RuntimeMetadata(BaseModel):
    """Frozen resolved execution identity recorded once for a trainer run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    precision: Precision
    compile: CompileConfig
    world_size: PositiveInt
    rollout_backend: Literal["reference", "native"]


class EvaluationGate(BaseModel):
    """An evaluation threshold for curriculum progression."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: str
    minimum: float
    opponent: str
    seeds: PositiveInt


class CurriculumConfig(BaseModel):
    """Curriculum state and failure behavior."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    phase: str = "phase1"
    reward_field: Literal["shaped_money", "shaped", "sparse", "own", "margin"] = (
        "shaped_money"
    )
    money_weight: NonNegativeFloat = MONEY_WEIGHT
    warm_start_checkpoint: Path | None = None
    gate: EvaluationGate | None = None
    on_gate_failure: Literal["stop"] = "stop"


class ToadConfig(BaseModel):
    """The complete serializable native Toad experiment contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: ModelConfig = Field(default_factory=ModelConfig)
    population: PopulationConfig = Field(default_factory=PopulationConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)

    @classmethod
    def control(cls) -> Self:
        """Return the control configuration reproducing the current recipe."""
        return cls()

    @model_validator(mode="after")
    def validate_relationships(self, info: ValidationInfo) -> Self:  # noqa: C901
        """Reject combinations unsupported by the native trainer."""
        probabilities = (
            self.population.selfplay,
            self.population.scripted,
            self.population.frozen_opponent,
            self.population.teacher_distill,
        )
        if any(value < 0 for value in probabilities) or not math.isclose(
            sum(probabilities), 1.0
        ):
            raise ValueError(
                "population probabilities must be nonnegative and sum to one"
            )
        if (
            self.population.frozen_opponent > 0
            and not self.population.initial_snapshots
            and not self.population.snapshot_at_start
        ):
            raise ValueError(
                "frozen_opponent requires a first-round source: initial_snapshots "
                "or snapshot_at_start"
            )
        _validate_foundation_optimizer(self.optimizer)
        max_steps_per_round = (
            2 * self.population.environments_per_rank * (EPISODE_STEPS - 1)
        )
        for name, controller in self.optimizer.entropy.items():
            if controller.multiplier_change_per_step * max_steps_per_round >= 1:
                raise ValueError(
                    f"optimizer.entropy.{name} multiplier change across the "
                    "largest actual round must be less than 1"
                )
        if (
            self.optimizer.teacher_kl_cost
            or self.optimizer.teacher_baseline_cost
            or self.population.teacher_distill
        ) and self.population.teacher is None:
            raise ValueError(
                "teacher checkpoint is required by teacher loss or batches"
            )
        if (
            self.population.teacher_distill
            and self.population.teacher is not None
            and not all(
                (
                    self.population.teacher.operation,
                    self.population.teacher.quantity,
                    self.population.teacher.market,
                )
            )
        ):
            raise ValueError(
                "teacher_distill requires operation, quantity, and market heads"
            )
        historical = bool(info.context and info.context.get("historical"))
        if self.population.teacher is not None and not historical:
            _require_readable(
                self.population.teacher.checkpoint,
                label="teacher checkpoint",
            )
        if (
            self.optimizer.teacher_baseline_cost
            and self.population.teacher is not None
            and not self.population.teacher.value
        ):
            raise ValueError(
                "teacher_baseline_cost requires a declared teacher value head"
            )
        if (
            self.population.teacher is not None
            and self.population.teacher.value
            and any(
                (
                    self.model.recurrent,
                    self.model.transformer,
                    self.model.local_patch,
                    self.model.belief,
                    self.model.interaction_value,
                )
            )
        ):
            raise ValueError(
                "teacher value alignment requires compatible control value semantics"
            )
        if not historical:
            for snapshot in self.population.initial_snapshots:
                _require_readable(snapshot, label="initial snapshot")
        if self.curriculum.warm_start_checkpoint is not None and not historical:
            _require_readable(
                self.curriculum.warm_start_checkpoint,
                label="warm-start checkpoint",
            )
        if (
            self.curriculum.warm_start_checkpoint is not None
            and self.runtime.resume is not None
        ):
            raise ValueError("warm start and resume are mutually exclusive")
        if self.runtime.resume is not None and self.runtime.resume.suffix == ".pt":
            raise ValueError(
                "legacy .pt resume requires checkpoint migration before Lightning; "
                "conversion is scheduled for Stage 9"
            )
        return self


def _validate_foundation_optimizer(optimizer: OptimizerConfig) -> None:
    """Reject loss coefficients whose native consumers do not exist yet."""
    unsupported = {
        "vtrace_pg_cost": (optimizer.vtrace_pg_cost, 1.0),
        "upgo_pg_cost": (optimizer.upgo_pg_cost, 1.0),
        "baseline_cost": (optimizer.baseline_cost, 1.0),
    }
    for name, (value, control) in unsupported.items():
        if value != control:
            raise ValueError(f"{name} is not configurable in the foundation trainer")


def _require_readable(path: Path, *, label: str) -> None:
    """Reject a missing, non-file, or unreadable checkpoint path."""
    if not path.is_file():
        raise ValueError(f"{label} is not readable: {path}")
    try:
        with path.open("rb"):
            pass
    except OSError as error:
        raise ValueError(f"{label} is not readable: {path}") from error


def load_config(path: Path | None, overrides: Sequence[str] = ()) -> ToadConfig:
    """Load a JSON configuration and apply validated dotted overrides."""
    payload = {} if path is None else json.loads(path.read_text())
    return apply_overrides(ToadConfig.model_validate(payload), overrides)


def validate_stored_config(payload: object) -> ToadConfig:
    """Validate checkpoint schema, migrating the former eager compile literal."""
    normalized = payload
    if isinstance(payload, Mapping):
        runtime = payload.get("runtime")
        if isinstance(runtime, Mapping) and runtime.get("compile") is False:
            normalized = dict(payload)
            normalized["runtime"] = dict(runtime) | {"compile": {"enabled": False}}
    return ToadConfig.model_validate(normalized, context={"historical": True})


def apply_overrides(config: ToadConfig, overrides: Sequence[str]) -> ToadConfig:
    """Apply JSON-typed dotted overrides by reconstructing ``ToadConfig``."""
    payload = config.model_dump(mode="python")
    for override in overrides:
        path, raw = override.split("=", 1)
        cursor = payload
        parts = path.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                raise ValueError(f"unknown override path {path!r}")
            cursor = cursor[part]
        if parts[-1] not in cursor:
            raise ValueError(f"unknown override path {path!r}")
        cursor[parts[-1]] = json.loads(raw)
    return ToadConfig.model_validate(payload)


STRUCTURAL_FIELDS = (
    "model",
    "optimizer.adaptive_entropy",
    "optimizer.entropy",
    "optimizer.unroll_length",
    "optimizer.batch_segments",
    "runtime.compile",
    "runtime.rollout_backend",
    "runtime.rollout_device",
    "runtime.rollout_cuda_graph",
)


def select_paths(payload: dict[str, object], paths: Sequence[str]) -> dict[str, object]:
    """Select dotted paths while preserving their nesting in ``payload``."""
    selected: dict[str, object] = {}
    for path in paths:
        source = payload
        target = selected
        parts = path.split(".")
        for part in parts[:-1]:
            source_value = source[part]
            if not isinstance(source_value, dict):
                raise ValueError(f"path {path!r} does not contain a mapping")
            source = cast(dict[str, object], source_value)
            if part not in target:
                target[part] = {}
            target_value = target[part]
            if not isinstance(target_value, dict):
                raise ValueError(f"path {path!r} conflicts with an earlier selection")
            target = cast(dict[str, object], target_value)
        target[parts[-1]] = source[parts[-1]]
    return selected


def structural_fingerprint(config: ToadConfig) -> str:
    """Hash settings that determine checkpoint and batch compatibility."""
    payload = select_paths(config.model_dump(mode="json"), STRUCTURAL_FIELDS)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
