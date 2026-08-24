"""Validated, serializable configuration for the native Toad trainer."""

import hashlib
import json
import math
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
        if self.local_patch and self.local_patch_size % 2 == 0:
            raise ValueError("local patch size must be odd")
        if self.transformer and self.channels % self.transformer_heads:
            raise ValueError("transformer channels must divide evenly across heads")
        return self


class PopulationConfig(BaseModel):
    """Actor population and opponent-pool settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    selfplay: float = 0.5
    scripted: float = 0.5
    frozen_opponent: float = 0.0
    teacher_distill: float = 0.0
    scripted_opponent: str = "economic"
    teacher_checkpoint: Path | None = None
    teacher_blocks: PositiveInt | None = None
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


class OptimizerConfig(BaseModel):
    """Loss coefficients and optimizer schedule settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lr: PositiveFloat = LEARNING_RATE
    adam_eps: PositiveFloat = ADAM_EPS
    gamma: float = DISCOUNTING
    lmb: float = LMB
    entropy_cost: float = ENTROPY_COST
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


class RuntimeConfig(BaseModel):
    """Single-device execution settings supported by the native trainer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = 0
    accelerator: str = "auto"
    devices: Literal[1] = 1
    num_nodes: Literal[1] = 1
    strategy: Literal["auto"] = "auto"
    precision: Literal["32-true"] = "32-true"
    deterministic: bool = False
    benchmark: bool | None = None
    total_environment_steps: PositiveInt = TOTAL_STEPS
    log_every_n_steps: PositiveInt = 1
    checkpoint_every_environment_steps: PositiveInt = 1_000_000
    profiler: Literal["simple", "advanced"] | None = None
    output_dir: Path = Path("run/toad")
    resume: Path | None = None
    compile: Literal[False] = False
    rollout_backend: Literal["reference"] = "reference"


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
    def validate_relationships(self, info: ValidationInfo) -> Self:
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
        _validate_active_model(self.model)
        _validate_foundation_population(self.population)
        _validate_foundation_optimizer(self.optimizer)
        if (
            self.optimizer.teacher_kl_cost or self.population.teacher_distill
        ) and self.population.teacher_checkpoint is None:
            raise ValueError(
                "teacher checkpoint is required by teacher loss or batches"
            )
        historical = bool(info.context and info.context.get("historical"))
        if self.population.teacher_checkpoint is not None and not historical:
            _require_readable(
                self.population.teacher_checkpoint,
                label="teacher checkpoint",
            )
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


def _validate_active_model(model: ModelConfig) -> None:
    """Reject architecture paths not yet consumed by the active trainer."""
    unsupported = {
        "transformer": model.transformer,
        "local_patch": model.local_patch,
        "belief": model.belief,
        "belief_loss_weight": model.belief_loss_weight > 0,
        "belief_feedback": model.belief_feedback,
        "interaction_value": model.interaction_value,
    }
    for name, enabled in unsupported.items():
        if enabled:
            raise ValueError(f"{name} is not implemented in the active trainer")


def _validate_foundation_population(population: PopulationConfig) -> None:
    """Reject population modes implemented only by later native-port stages."""
    if population.frozen_opponent:
        raise ValueError("frozen_opponent is not implemented in the foundation trainer")
    if population.teacher_distill:
        raise ValueError("teacher_distill is not implemented in the foundation trainer")


def _validate_foundation_optimizer(optimizer: OptimizerConfig) -> None:
    """Reject loss coefficients whose native consumers do not exist yet."""
    unsupported = {
        "teacher_baseline_cost": (optimizer.teacher_baseline_cost, 0.0),
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
    """Validate checkpoint schema without rechecking historical external files."""
    return ToadConfig.model_validate(payload, context={"historical": True})


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
    "optimizer.unroll_length",
    "optimizer.batch_segments",
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
