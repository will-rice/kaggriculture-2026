"""On-policy fine-tuning of the market residual, on the seasons it actually plays.

The offline phase measured, honestly and correctly, what *one* deviation from
the frozen controller is worth when the controller plays every other event of
the season. The served agent applies hundreds of deviations per season, and the
composition of those is not the sum of the parts: the offline checkpoint loses
128 of 128 held-out games to the bare controller it wraps. That is textbook
off-policy distribution shift, and the only measurement that can escape it is
one taken on the policy's own state distribution. This module takes it.

Four decisions here are load-bearing, and none of them is a preference.

**The run starts at "always defer", not from an offline checkpoint.** Every
offline checkpoint we hold loses every game against the bare controller, so
warm-starting begins in a hole with no gradient out of it. At zero activation
the wrapped agent returns the frozen controller's own action object on every
turn, so it *is* the controller, which scores exactly 0.5 against itself. That
makes the floor unlosable and every point above it a real gain.
``initialize_deferring`` puts the head there and
``tests/market_residual/test_online.py`` proves it over a whole real season.

**The reward is a paired difference on one cell.** Candidate and control share
opponent, seed and seat, so everything about the cell that is not the residual
cancels. It has an exactness worth stating: a candidate that never deviates
plays the identical action stream, banks the identical number, and scores
*exactly* zero. Reward is therefore nonzero only in episodes that changed
something, which is what makes a terminal reward usable at all here.

**A proposal the merge boundary refused earns no policy gradient.** The
boundary is part of the environment: an illegal replacement plays exactly what
deferring plays. Scoring it as though it had been played creates a free action
-- it lands at the mean of the episodes that changed nothing, which sits *above*
the mean of all episodes because the average deviation is harmful -- and a
policy that found that would learn to propose illegal replacements everywhere.
``pg_weight`` zeroes those rows instead.

**Exploration is small, and it starts from a legal proposal.** Counterfactual
measurement over 5,636 deviation arms puts the mean win-point delta negative in
every family and in every phase of the season; only 27% of arms are positive.
Deviation is a low-probability, high-variance bet, so the mode head starts far
from replacing and the quantity heads start on the zero bucket, which is the
one proposal the merge boundary can always prove. A head that started uniform
over 21 buckets in each of 11 slots would spend every sample on a proposal the
board refuses, and would learn nothing from any of them.

The learner owns one GPU and the actors own the CPUs. Actors run whole seasons
on the reference engine in worker processes against an actor snapshot published
only at completed update boundaries, so a batch is never collected against a
half-updated head.
"""

import json
import logging
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, fields, replace
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import torch
from kaggle_environments import make
from kaggle_environments.core import Environment
from torch import nn

from kaggriculture.constants import ENVIRONMENT, EPISODE_STEPS
from kaggriculture.learn.market_residual.counterfactual import (
    canonical_digest,
    normalized_margin,
    win_points,
)
from kaggriculture.learn.market_residual.model import (
    REPLACE_INDEX,
    MarketResidualNet,
    ModelConfig,
    entropy,
    log_probability,
)
from kaggriculture.learn.market_residual.offline import (
    load_checkpoint,
    save_checkpoint,
    write_json_atomic,
)
from kaggriculture.learn.toad.core.vtrace import from_importance_weights
from kaggriculture.market_residual.actions import ResidualDecision, ResidualMode
from kaggriculture.market_residual.baseline import SERVED, load_verified_baseline
from kaggriculture.market_residual.features import MarketFeatureVector
from kaggriculture.market_residual.numpy_policy import MODES
from kaggriculture.market_residual.policy import (
    DEFAULT_EVENT_CONFIG,
    MarketResidualAgent,
)
from kaggriculture.market_residual.schema import (
    ALLOWED_SLOTS,
    BUCKETS,
    MarketFeatureSchema,
)

LOGGER = logging.getLogger(__name__)

LAST_CHECKPOINT = "last.ckpt"
BEST_CHECKPOINT = "best.ckpt"
ACTOR_CHECKPOINT = "actor.ckpt"
EXPORT_ROWS = "export-rows.pt"
REPORT_NAME = "online-report.json"

# The served controller, as an agent file the reference engine can load into
# the opposing seat. Restated as a path rather than imported from
# `search.scripts.holdout` because this package must not pull the search tree
# into collection; `tests/market_residual/test_train_cli.py` pins the two equal.
SERVED_PATH = "src/kaggriculture/kaito_v56_policy.py"

USE_KAITO_INDEX = MODES.index(ResidualMode.USE_KAITO)

# The bounded margin term can move a reward by at most this much, which is
# below the smallest nonzero paired win-point difference. A decided game
# therefore keeps its sign and the term only separates cells that tied.
MARGIN_WEIGHT = 0.1


class OnlineDriftError(RuntimeError):
    """Raised when a root's stored parameters and this run's disagree."""


class OnlineNonFiniteError(RuntimeError):
    """Raised when a head or a loss term stops being finite."""


@dataclass(frozen=True)
class OnlineConfig:
    """Everything one online run is parameterised by.

    Stored whole in the root and in every checkpoint, so a resume that changed
    any of it is refused rather than silently continued.
    """

    updates: int
    episodes_per_update: int
    seed_bank: str = "online_train"
    learning_rate: float = 3e-4
    entropy_start: float = 1e-3
    entropy_end: float = 0.0
    value_weight: float = 0.5
    auxiliary_weight: float = 0.1
    passes: int = 2
    clip_rho: float = 1.0
    gradient_clip: float = 1.0
    defer_logit: float = -6.0
    zero_bucket_logit: float = 4.5
    quantity_decay: float = 0.25
    seed: int = 0
    device: str = "cuda"
    workers: int = 48

    def __post_init__(self) -> None:
        """Refuse a run that could never produce a checkpoint.

        Raises:
            ValueError: If there is not at least one update or one episode.
        """
        if self.updates < 1:
            raise ValueError(f"updates must be at least 1, got {self.updates}")
        if self.episodes_per_update < 1:
            raise ValueError(
                f"episodes_per_update must be at least 1, "
                f"got {self.episodes_per_update}"
            )
        if self.passes < 1:
            raise ValueError(f"passes must be at least 1, got {self.passes}")

    def entropy_coefficient(self, update: int) -> float:
        """Return the entropy bonus in force at one update.

        Annealed linearly to ``entropy_end`` over the whole run, because the
        arms this policy explores are negative on average: exploration is worth
        paying for early and worth nothing at all by the time the run is being
        selected from.

        Args:
            update: The one-based index of the update about to be taken.

        Returns:
            The coefficient.
        """
        if self.updates == 1:
            return self.entropy_end
        fraction = (update - 1) / (self.updates - 1)
        return self.entropy_start + fraction * (self.entropy_end - self.entropy_start)


@dataclass(frozen=True)
class Cell:
    """One paired game: an opponent, a seed, and which seat we hold.

    The opponent's sampling weight travels on the cell so that choosing a
    batch needs no second table beside the cell list, and so that a root's
    stored parameters record the league weights it was bound to.
    """

    opponent: str
    path: str
    seed: int
    seat: int
    weight: float = 1.0


@dataclass(frozen=True)
class EventTrajectory:
    """Every market event of one episode, and nothing that was not one.

    ``executed`` is what separates a decision from a proposal: it is false
    exactly where the merge boundary refused a replacement, which the engine
    played as the controller's own action.
    """

    steps: tuple[int, ...]
    features: torch.Tensor
    modes: torch.Tensor
    buckets: torch.Tensor
    behavior_log_prob: torch.Tensor
    executed: torch.Tensor


@dataclass(frozen=True)
class PairedEpisode:
    """One candidate arm and the frozen control it is scored against.

    Both arms played the same opponent on the same seed in the same seat, so
    everything about the cell that is not the residual cancels in the
    difference.
    """

    opponent: str
    seed: int
    seat: int
    trajectory: EventTrajectory
    candidate_points: float
    control_points: float
    normalized_margin_delta: float
    candidate_bank: int
    control_bank: int
    replacements: int
    merged: int
    fallbacks: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateResult:
    """What one candidate season produced, before it has a control to pair with."""

    cell: Cell
    trajectory: EventTrajectory
    candidate_bank: int
    opponent_bank: int
    replacements: int
    merged: int
    fallbacks: dict[str, int]


@dataclass(frozen=True)
class OnlineBatch:
    """Padded episodes and every index the V-trace terms consume.

    Time is the first axis and episode the second, which is the layout the
    recurrent head and the V-trace recursion both want.
    """

    features: torch.Tensor
    valid: torch.Tensor
    modes: torch.Tensor
    buckets: torch.Tensor
    behavior_log_prob: torch.Tensor
    pg_weight: torch.Tensor
    rewards: torch.Tensor
    discounts: torch.Tensor
    auxiliary_target: torch.Tensor
    auxiliary_valid: torch.Tensor

    def to(self, device: torch.device | str) -> "OnlineBatch":
        """Return this batch with every tensor on one device.

        Args:
            device: Where the model runs.

        Returns:
            The moved batch.
        """
        return replace(
            self,
            **{item.name: getattr(self, item.name).to(device) for item in fields(self)},
        )


@dataclass(frozen=True)
class OnlineLoss:
    """Every term of one update, and what the heads looked like taking it."""

    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    auxiliary: torch.Tensor
    max_rho: float
    mean_rho: float
    replace_probability: float
    finite: bool

    def metrics(self) -> dict[str, float]:
        """Return every term and measurement as plain floats."""
        return {
            "total": float(self.total.item()),
            "policy": float(self.policy.item()),
            "value": float(self.value.item()),
            "entropy": float(self.entropy.item()),
            "auxiliary": float(self.auxiliary.item()),
            "max_rho": self.max_rho,
            "mean_rho": self.mean_rho,
            "replace_probability": self.replace_probability,
        }


def initialize_deferring(model: MarketResidualNet, config: OnlineConfig) -> None:
    """Put a fresh head at "always defer", with a legal proposal underneath it.

    Two biases, for two different reasons.

    The mode bias is the floor. At ``defer_logit = -6`` the head replaces on
    roughly one event in four hundred, so a season opens about one deviation
    and the served greedy decode opens none at all -- the argmax is
    ``USE_KAITO`` everywhere, which is the frozen controller exactly.

    The quantity bias is what makes those rare deviations worth anything. Left
    at its default the head is near-uniform over 21 buckets in each of 11
    slots, so every replacement it proposes asks for eleven random orders and
    the merge boundary refuses essentially all of them. Biased to the zero
    bucket the default proposal is the one the boundary can always prove, and
    exploration walks outward from it a slot at a time.

    Both numbers are set against a measurement rather than chosen. Over two
    real seasons the frozen controller proposes commodity orders at 48% of
    events, and 59% of the allowed slots admit no legal quantity at all at the
    moment they are looked at. So the all-zero proposal is a genuine
    cancellation half the time rather than a no-op, and a nonzero bucket drawn
    for a slot picked at random is refused more often than not --
    ``quantity_decay`` tilts those draws toward the small quantities that the
    board is most likely to admit.

    Args:
        model: The head to initialise; mutated in place.
        config: The run's parameterisation.
    """
    with torch.no_grad():
        model.mode.bias.zero_()
        model.mode.bias[REPLACE_INDEX] = config.defer_logit
        quantities = model.quantities.bias.view(
            model.config.slots, model.config.buckets
        )
        ramp = -config.quantity_decay * torch.arange(model.config.buckets).float()
        quantities.copy_(ramp.tile(model.config.slots, 1))
        quantities[:, 0] = config.zero_bucket_logit


def terminal_reward(pair: PairedEpisode) -> float:
    """Return what one paired cell was worth to the candidate.

    Args:
        pair: The candidate arm and its frozen control.

    Returns:
        The paired win-point difference, plus a bounded margin term.
    """
    return (pair.candidate_points - pair.control_points) + (
        MARGIN_WEIGHT * pair.normalized_margin_delta
    )


def auxiliary_targets(features: torch.Tensor) -> torch.Tensor:
    """Return the auxiliary head's target row for every event of one episode.

    Per product: the normalized live price, the normalized book inventory, and
    the opponent-supply bucket scaled into ``[0, 1]`` -- the same layout the
    exported artifact declares under ``auxiliary_size``. It is read off the
    feature rows through the schema's own spans rather than off the raw
    observation, so a block that moves moves this with it.

    Args:
        features: ``(events, width)`` canonical feature rows of one episode.

    Returns:
        ``(events, auxiliary_size)`` targets; row ``t`` is what event ``t``
        looked like, so the loss regresses row ``t + 1`` against event ``t``.
    """
    schema = MarketFeatureSchema.current()
    products = [
        name.split(":", 1)[1]
        for name, _width in schema.blocks
        if name.startswith("price:")
    ]
    columns = [features[:, schema.span(f"price:{name}").start] for name in products]
    columns.extend(
        features[:, schema.span(f"inventory:{name}").start] for name in products
    )
    columns.extend(
        features[:, schema.span(f"opponent_supply:{name}")].argmax(dim=-1).float()
        / (BUCKETS - 1)
        for name in products
    )
    return torch.stack(columns, dim=-1)


def collate_paired_episodes(pairs: Sequence[PairedEpisode]) -> OnlineBatch:
    """Pad a set of paired episodes into one time-major batch.

    The discount is one inside an episode and zero at its last event, so the
    V-trace recursion stops there and no padded step can leak a return into a
    shorter episode beside it.

    Args:
        pairs: The paired episodes of one update.

    Returns:
        The batch.

    Raises:
        ValueError: If there is nothing to collate.
    """
    if not pairs:
        raise ValueError("cannot collate an empty set of paired episodes")
    width = MarketFeatureSchema.current().width
    lengths = [len(pair.trajectory.steps) for pair in pairs]
    steps, batch = max(lengths), len(pairs)
    slots = len(ALLOWED_SLOTS)
    auxiliary_size = ModelConfig.current().auxiliary_size

    features = torch.zeros(steps, batch, width)
    valid = torch.zeros(steps, batch, dtype=torch.bool)
    modes = torch.zeros(steps, batch, dtype=torch.int64)
    buckets = torch.zeros(steps, batch, slots, dtype=torch.int64)
    behavior = torch.zeros(steps, batch)
    pg_weight = torch.zeros(steps, batch)
    rewards = torch.zeros(steps, batch)
    discounts = torch.zeros(steps, batch)
    auxiliary_target = torch.zeros(steps, batch, auxiliary_size)
    auxiliary_valid = torch.zeros(steps, batch, dtype=torch.bool)

    for index, pair in enumerate(pairs):
        length = lengths[index]
        trajectory = pair.trajectory
        features[:length, index] = trajectory.features
        valid[:length, index] = True
        modes[:length, index] = trajectory.modes
        buckets[:length, index] = trajectory.buckets
        behavior[:length, index] = trajectory.behavior_log_prob
        pg_weight[:length, index] = trajectory.executed.float()
        rewards[length - 1, index] = terminal_reward(pair)
        discounts[: length - 1, index] = 1.0
        targets = auxiliary_targets(trajectory.features)
        auxiliary_target[: length - 1, index] = targets[1:]
        auxiliary_valid[: length - 1, index] = True

    return OnlineBatch(
        features=features,
        valid=valid,
        modes=modes,
        buckets=buckets,
        behavior_log_prob=behavior,
        pg_weight=pg_weight,
        rewards=rewards,
        discounts=discounts,
        auxiliary_target=auxiliary_target,
        auxiliary_valid=auxiliary_valid,
    )


def replace_behavior_log_prob(batch: OnlineBatch, value: float) -> OnlineBatch:
    """Return this batch with every behavior log probability set to one value.

    Exists so a test can drive the importance ratio to overflow without
    inventing a second way to build a batch.

    Args:
        batch: The batch to rewrite.
        value: The behavior log probability to store everywhere.

    Returns:
        The rewritten batch.
    """
    return replace(
        batch, behavior_log_prob=torch.full_like(batch.behavior_log_prob, value)
    )


def vtrace_market_loss(
    batch: OnlineBatch,
    model: MarketResidualNet,
    entropy_coefficient: float = 0.0,
    value_weight: float = 0.5,
    auxiliary_weight: float = 0.1,
    clip_rho: float = 1.0,
) -> OnlineLoss:
    """Return one update's loss over a batch of whole paired episodes.

    Whole episodes are forwarded from a zero entry state rather than in
    truncated chunks, because a market episode is about five hundred events and
    fits in one forward; there is no truncation boundary here to get wrong.

    The quantity mask is all-true, matching the served runtime, which applies no
    legality mask -- an illegal argmax costs a fail-closed deferral at the merge
    rather than being masked away. A head trained under a mask the runtime
    cannot apply would be trained on a different distribution than it plays.

    Args:
        batch: One update's collated episodes.
        model: The head being trained.
        entropy_coefficient: The exploration bonus in force at this update.
        value_weight: How much the value regression counts against the policy.
        auxiliary_weight: How much the next-event regression counts.
        clip_rho: The importance-ratio ceiling.

    Returns:
        Every term, and the measurements the report reads.
    """
    heads, _state = model(batch.features, None)
    mask = torch.ones_like(heads.quantity_logits, dtype=torch.bool)
    target_log_prob = log_probability(heads, mask, batch.modes, batch.buckets)
    log_rho = torch.where(
        batch.valid, target_log_prob.detach() - batch.behavior_log_prob, 0.0
    )
    values = heads.value

    returns = from_importance_weights(
        log_rhos=log_rho,
        discounts=batch.discounts,
        rewards=batch.rewards,
        values=values.detach(),
        bootstrap_value=torch.zeros_like(values[0]),
        clip_rho_threshold=clip_rho,
        clip_pg_rho_threshold=clip_rho,
    )

    weight = batch.pg_weight
    credited = weight.sum().clamp(min=1.0)
    policy = -(weight * returns.pg_advantages * target_log_prob).sum() / credited
    value = _masked_mean(
        nn.functional.huber_loss(values, returns.vs, reduction="none"), batch.valid
    )
    exploration = _masked_mean(entropy(heads, mask), batch.valid)
    auxiliary = _masked_mean(
        nn.functional.huber_loss(
            heads.auxiliary, batch.auxiliary_target, reduction="none"
        ).mean(dim=-1),
        batch.auxiliary_valid,
    )
    total = (
        policy
        + value_weight * value
        - entropy_coefficient * exploration
        + auxiliary_weight * auxiliary
    )
    clipped = torch.where(batch.valid, log_rho.exp().clamp(max=clip_rho), 0.0)
    replace_probability = _masked_mean(
        torch.softmax(heads.mode_logits.detach(), dim=-1)[..., REPLACE_INDEX],
        batch.valid,
    )
    return OnlineLoss(
        total=total,
        policy=policy,
        value=value,
        entropy=exploration,
        auxiliary=auxiliary,
        max_rho=float(clipped.max()),
        mean_rho=float(_masked_mean(clipped, batch.valid)),
        replace_probability=float(replace_probability),
        finite=bool(
            torch.isfinite(heads.mode_logits).all()
            and torch.isfinite(heads.quantity_logits).all()
            and torch.isfinite(heads.value).all()
            and torch.isfinite(total)
        ),
    )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return the mean of ``values`` over the entries ``mask`` admits.

    Args:
        values: The quantity to average.
        mask: True where an entry counts.

    Returns:
        The masked mean, or zero when nothing is admitted.
    """
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp(min=1.0)


class SamplingResidual:
    """The training-time head, sampling one action per market event.

    The recurrent state travels through ``ResidualInference``'s own tuple
    rather than being kept on this object, because that is the boundary the
    served runtime uses and a state that lived somewhere else here would be a
    second, untested carry.

    Every row this records is a row the loss will score, so nothing is appended
    on a turn that was not an event and nothing is appended twice.
    """

    def __init__(self, model: MarketResidualNet, generator: torch.Generator) -> None:
        """Bind one episode's head and its source of randomness.

        Args:
            model: The actor snapshot, in eval mode on the CPU.
            generator: This episode's sampler, seeded by the caller.
        """
        self.model = model
        self.generator = generator
        self.features: list[tuple[float, ...]] = []
        self.modes: list[int] = []
        self.buckets: list[tuple[int, ...]] = []
        self.log_probability: list[float] = []

    def initial_state(self) -> tuple[float, ...]:
        """Return the zero recurrent state an episode starts from."""
        return (0.0,) * self.model.config.hidden_size

    def observe(
        self,
        features: MarketFeatureVector,
        state: tuple[float, ...],
        *,
        act: bool,
    ) -> tuple[ResidualDecision | None, tuple[float, ...]]:
        """Sample one decision and return the state to carry forward.

        Args:
            features: The canonical row for this market event.
            state: The state returned at the previous event of this episode.
            act: Whether this call's decision will be played.

        Returns:
            The sampled decision, or ``None`` when only the state advances, and
            the next state.
        """
        row = torch.tensor(features.values, dtype=torch.float32).reshape(1, 1, -1)
        entry = torch.tensor(state, dtype=torch.float32).reshape(1, 1, -1)
        with torch.no_grad():
            heads, current = self.model(row, entry)
        following = tuple(float(value) for value in current.flatten())
        if not act:
            return None, following

        mode_log = torch.log_softmax(heads.mode_logits[0, 0], dim=-1)
        mode = int(
            torch.multinomial(mode_log.exp(), 1, generator=self.generator).item()
        )
        total = float(mode_log[mode])
        quantity_log = torch.log_softmax(heads.quantity_logits[0, 0], dim=-1)
        if mode == REPLACE_INDEX:
            chosen = torch.multinomial(
                quantity_log.exp(), 1, generator=self.generator
            ).flatten()
            total += float(quantity_log.gather(-1, chosen[:, None]).sum())
        else:
            chosen = torch.zeros(self.model.config.slots, dtype=torch.int64)

        self.features.append(features.values)
        self.modes.append(mode)
        self.buckets.append(tuple(int(bucket) for bucket in chosen))
        self.log_probability.append(total)
        return (
            ResidualDecision(mode=MODES[mode], buckets=tuple(int(b) for b in chosen)),
            following,
        )


def play_candidate(model: MarketResidualNet, cell: Cell, seed: int) -> CandidateResult:
    """Play one whole season with the sampling head and keep every event row.

    Args:
        model: The actor snapshot; used under ``no_grad`` on the CPU.
        cell: The opponent, seed and seat to play.
        seed: This episode's sampler seed.

    Returns:
        The trajectory and both terminal banks.

    Raises:
        RuntimeError: If the season did not finish cleanly, which is never an
            ordinary result here.
    """
    generator = torch.Generator().manual_seed(seed)
    residual = SamplingResidual(model, generator)
    agent = MarketResidualAgent(
        load_verified_baseline(SERVED), residual, config=DEFAULT_EVENT_CONFIG
    )
    turns: list[int] = []
    executed: list[bool] = []

    def seat(
        observation: Mapping[str, Any], configuration: object = None
    ) -> Mapping[str, Any]:
        """Play one turn and record whether the event's decision took effect."""
        turn = agent.memory.turn
        rows = len(residual.modes)
        replacements = agent.record.replacements
        action = agent(observation, configuration)
        if len(residual.modes) > rows:
            turns.append(turn)
            executed.append(
                residual.modes[-1] == USE_KAITO_INDEX
                or agent.record.replacements > replacements
            )
        return action

    environment = make(
        ENVIRONMENT,
        configuration={"episodeSteps": EPISODE_STEPS, "seed": cell.seed},
    )
    sides: list[Any] = [seat, cell.path] if cell.seat == 0 else [cell.path, seat]
    environment.run(sides)
    ours, theirs = _terminal_banks(environment, cell.seat, cell.seed)

    return CandidateResult(
        cell=cell,
        trajectory=EventTrajectory(
            steps=tuple(turns),
            features=torch.tensor(residual.features, dtype=torch.float32),
            modes=torch.tensor(residual.modes, dtype=torch.int64),
            buckets=torch.tensor(residual.buckets, dtype=torch.int64),
            behavior_log_prob=torch.tensor(
                residual.log_probability, dtype=torch.float32
            ),
            executed=torch.tensor(executed, dtype=torch.bool),
        ),
        candidate_bank=ours,
        opponent_bank=theirs,
        replacements=agent.record.replacements,
        merged=agent.record.replacements,
        fallbacks={str(name): count for name, count in agent.record.fallbacks.items()},
    )


def play_control(cell: Cell) -> tuple[int, int]:
    """Play the same cell with the bare frozen controller in our seat.

    Args:
        cell: The opponent, seed and seat to play.

    Returns:
        Our terminal bank and the opponent's.
    """
    environment = make(
        ENVIRONMENT,
        configuration={"episodeSteps": EPISODE_STEPS, "seed": cell.seed},
    )
    control = load_verified_baseline(SERVED)
    sides: list[Any] = [control, cell.path] if cell.seat == 0 else [cell.path, control]
    environment.run(sides)
    return _terminal_banks(environment, cell.seat, cell.seed)


def _terminal_banks(environment: Environment, seat: int, seed: int) -> tuple[int, int]:
    """Return one finished episode's banks, ours first.

    Args:
        environment: The finished engine.
        seat: Which seat we held.
        seed: The episode seed, for the failure message.

    Returns:
        Our bank and the opponent's.

    Raises:
        RuntimeError: If the season did not finish with both seats done.
    """
    final = environment.steps[-1]
    statuses = (final[0].status, final[1].status)
    if statuses != ("DONE", "DONE") or any(state.reward is None for state in final):
        raise RuntimeError(f"seed {seed} did not finish cleanly (statuses={statuses})")
    banks = (int(final[0].reward), int(final[1].reward))
    return banks if seat == 0 else (banks[1], banks[0])


def pair(result: CandidateResult, control: Sequence[int]) -> PairedEpisode:
    """Score one candidate season against its frozen control on the same cell.

    Args:
        result: The candidate arm.
        control: The control arm's banks, ours first.

    Returns:
        The paired episode.
    """
    control_ours, control_theirs = control
    return PairedEpisode(
        opponent=result.cell.opponent,
        seed=result.cell.seed,
        seat=result.cell.seat,
        trajectory=result.trajectory,
        candidate_points=win_points(result.candidate_bank, result.opponent_bank),
        control_points=win_points(control_ours, control_theirs),
        normalized_margin_delta=(
            normalized_margin(result.candidate_bank, result.opponent_bank)
            - normalized_margin(control_ours, control_theirs)
        ),
        candidate_bank=result.candidate_bank,
        control_bank=control_ours,
        replacements=result.replacements,
        merged=result.merged,
        fallbacks=result.fallbacks,
    )


@dataclass(frozen=True)
class CollectionTask:
    """One candidate episode for a worker process to play."""

    cell: Cell
    actor: str
    seed: int


_ACTORS: dict[tuple[str, int], MarketResidualNet] = {}


def actor_snapshot(path: str) -> MarketResidualNet:
    """Return the published head at ``path``, loaded once per worker process.

    Keyed by the snapshot's modification time as well as its name, so a worker
    that outlives an update boundary cannot keep playing the previous head.

    Args:
        path: The actor checkpoint address.

    Returns:
        The head, in eval mode on the CPU.
    """
    key = (path, Path(path).stat().st_mtime_ns)
    cached = _ACTORS.get(key)
    if cached is None:
        cached = MarketResidualNet(ModelConfig.current())
        cached.load_state_dict(load_checkpoint(Path(path))["model"])
        cached.eval()
        _ACTORS.clear()
        _ACTORS[key] = cached
    return cached


def run_candidate_task(task: CollectionTask) -> CandidateResult:
    """Play one candidate season in a worker process.

    Args:
        task: The cell, the snapshot to play, and the sampler seed.

    Returns:
        The candidate arm's trajectory and banks.
    """
    torch.set_num_threads(1)
    return play_candidate(actor_snapshot(task.actor), task.cell, task.seed)


def run_control_task(cell: Cell) -> tuple[int, int]:
    """Play one control season in a worker process.

    Args:
        cell: The opponent, seed and seat to play.

    Returns:
        The control arm's banks, ours first.
    """
    torch.set_num_threads(1)
    return play_control(cell)


class EngineCollector:
    """Collects paired episodes on the reference engine, across a process pool.

    The control arm is the bare frozen controller on the same cell, and it is
    deterministic: the same seed, seat and opponent always bank the same pair of
    numbers. It is therefore played once per cell and cached in the root, which
    halves the engine time of every update after the first.
    """

    def __init__(self, root: Path, config: OnlineConfig) -> None:
        """Bind one run's root and worker budget.

        Args:
            root: The owned online root; holds the actor snapshot and cache.
            config: The run's parameterisation.
        """
        self.root = root
        self.config = config
        self.cache_path = root / "controls.json"
        self.rows_path = root / EXPORT_ROWS
        self.controls: dict[str, list[int]] = (
            json.loads(self.cache_path.read_text(encoding="utf-8"))
            if self.cache_path.exists()
            else {}
        )
        self.pool = ProcessPoolExecutor(
            max_workers=config.workers, mp_context=get_context("spawn")
        )

    def __enter__(self) -> "EngineCollector":
        """Return this collector, with its pool open."""
        return self

    def __exit__(self, *exception: object) -> None:
        """Shut the pool down.

        Args:
            exception: The exception triple, unread.
        """
        self.pool.shutdown()

    def __call__(self, update: int, cells: Sequence[Cell]) -> tuple[PairedEpisode, ...]:
        """Play one update's candidate arms, and any control arm still missing.

        Args:
            update: The one-based index of the update being collected for.
            cells: The cells to play.

        Returns:
            One paired episode per cell, in the order given.
        """
        actor = str(self.root / ACTOR_CHECKPOINT)
        missing = list(
            dict.fromkeys(
                cell for cell in cells if _control_key(cell) not in self.controls
            )
        )
        tasks = [
            CollectionTask(
                cell=cell,
                actor=actor,
                seed=(update * 1_000_003 + index * 7919 + self.config.seed) % 2**31,
            )
            for index, cell in enumerate(cells)
        ]
        pending = self.pool.map(run_candidate_task, tasks)
        for cell, banks in zip(
            missing, self.pool.map(run_control_task, missing), strict=True
        ):
            self.controls[_control_key(cell)] = list(banks)
        if missing:
            write_json_atomic(self.cache_path, self.controls)
        candidates = list(pending)
        self._keep_export_rows(candidates)
        return tuple(
            pair(result, self.controls[_control_key(result.cell)])
            for result in candidates
        )

    def _keep_export_rows(self, results: Sequence[CandidateResult]) -> None:
        """Keep one real season's feature rows for the export's parity proof.

        The export must be proved over rows with the geometry this head
        actually sees -- mostly one-hot, a handful of small scalars -- and the
        run is already producing them. The longest season of the first batch is
        kept, because parity is only meaningful once the recurrence has
        accumulated.

        Args:
            results: One update's candidate arms.
        """
        if self.rows_path.exists():
            return
        longest = max(results, key=lambda result: len(result.trajectory.steps))
        save_checkpoint(self.rows_path, {"features": longest.trajectory.features})


def _control_key(cell: Cell) -> str:
    """Return the cache key for one cell's control arm.

    Args:
        cell: The opponent, seed and seat.

    Returns:
        A stable string key.
    """
    return f"{cell.opponent}|{cell.seed}|{cell.seat}"


def sample_cells(
    cells: Sequence[Cell], config: OnlineConfig, update: int
) -> tuple[Cell, ...]:
    """Return the cells one update plays, drawn without replacement.

    Each opponent gets the share of the batch its weight asks for, and its
    cells are then drawn uniformly inside that share. Allocating the shares
    rather than sampling opponents independently is what keeps a batch from
    happening to hold no games at all against the opponent the gate is decided
    by.

    Drawn from a generator seeded by the run seed and the update index rather
    than from the process RNG, so which cells an update plays does not depend
    on whether the run was interrupted before it.

    Args:
        cells: Every cell the run may play.
        config: The run's parameterisation.
        update: The one-based index of the update.

    Returns:
        ``episodes_per_update`` cells, or every cell when there are fewer.
    """
    source = random.Random(f"{config.seed}:{update}")
    if len(cells) <= config.episodes_per_update:
        return tuple(cells)
    groups: dict[str, list[Cell]] = {}
    for cell in cells:
        groups.setdefault(cell.opponent, []).append(cell)
    weights = {name: group[0].weight for name, group in groups.items()}
    total = sum(weights.values())
    chosen: list[Cell] = []
    for name, group in groups.items():
        share = round(config.episodes_per_update * weights[name] / total)
        chosen.extend(source.sample(group, min(share, len(group))))
    remaining = [cell for cell in cells if cell not in set(chosen)]
    source.shuffle(remaining)
    chosen.extend(remaining[: config.episodes_per_update - len(chosen)])
    return tuple(chosen[: config.episodes_per_update])


Collector = Callable[[int, Sequence[Cell]], "tuple[PairedEpisode, ...]"]

# What a resume may change. `device` and `workers` are the machine, not the
# run: moving a root to another GPU or another worker budget does not change
# what is learned. `updates` is deliberately NOT here -- the entropy anneal is
# measured against it, so a root reopened with a different update budget is a
# different schedule, not the same run continued.
UNBOUND_PARAMETERS = frozenset({"device", "workers"})


def bound_parameters(config: OnlineConfig) -> dict[str, Any]:
    """Return the parameters a resume is checked against.

    Args:
        config: The run's parameterisation.

    Returns:
        Every field except the stopping point and the machine.
    """
    return {
        name: value
        for name, value in asdict(config).items()
        if name not in UNBOUND_PARAMETERS
    }


def train_online(
    model: MarketResidualNet,
    config: OnlineConfig,
    root: Path,
    cells: Sequence[Cell],
    collect: Collector,
    log: Callable[[int, dict[str, float]], None] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Fine-tune the head on its own seasons, resumably, and report what it did.

    Every update publishes the actor snapshot, collects a batch of paired
    episodes against it, takes one gradient step, and writes both checkpoints
    atomically -- so an interruption anywhere resumes at the update boundary it
    last completed, playing the same cells it would have played.

    No random state is checkpointed, because none is carried. Which cells an
    update plays and how each actor samples are both derived from the run seed
    and the update index, so the run's randomness is a function of where it is
    rather than of how it got there -- which is invariant to an interruption
    and to worker scheduling alike, neither of which restoring a stream would
    be.

    A batch is stepped on ``passes`` times, and that is what makes the V-trace
    correction do any work: the first pass is on-policy by construction, and
    every pass after it is scored against the snapshot the episodes were
    actually sampled under. Reusing an expensive batch matters here because a
    season costs ten seconds of reference engine and carries about one
    deviation, so the data, not the gradient, is the scarce thing.

    ``best.ckpt`` follows an exponential moving average of the paired reward
    rather than one update's mean, because a single update of a few dozen
    episodes in which deviation is rare is far too noisy to select on: the
    modal update has reward exactly zero.

    Args:
        model: The head to train; mutated, and left holding the final weights.
        config: The run's parameterisation.
        root: The owned online root.
        cells: Every cell the run may play.
        collect: Plays one update's cells and returns the paired episodes.
        log: Called after each update with its metrics row, when given.
        run_id: The W&B run identity to store in every checkpoint.

    Returns:
        The report that was written to the root.

    Raises:
        OnlineDriftError: If the root was trained under a different binding.
        OnlineNonFiniteError: If any head or loss term stops being finite.
    """
    device = torch.device(config.device)
    model.to(device)
    binding = canonical_digest(
        {
            "config": bound_parameters(config),
            "model": ModelConfig.current().__dict__,
            "cells": [asdict(cell) for cell in cells],
        }
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    root.mkdir(parents=True, exist_ok=True)
    update = 0
    best: dict[str, float] = {"update": 0, "reward": -float("inf")}
    average = 0.0
    history: list[dict[str, float]] = []
    last_path = root / LAST_CHECKPOINT
    if last_path.exists():
        stored = load_checkpoint(last_path)
        if stored["binding_sha256"] != binding:
            raise OnlineDriftError(
                f"{root} was trained under binding {stored['binding_sha256']}, "
                f"this run computes {binding}"
            )
        model.load_state_dict(stored["model"])
        optimizer.load_state_dict(stored["optimizer"])
        update = stored["update"]
        best = stored["best"]
        average = stored["average"]
        history = stored["history"]
        run_id = stored["wandb_run_id"] if run_id is None else run_id

    while update < config.updates:
        update += 1
        save_checkpoint(
            root / ACTOR_CHECKPOINT,
            {"model": {k: v.cpu() for k, v in model.state_dict().items()}},
        )
        started = perf_counter()
        episodes = collect(update, sample_cells(cells, config, update))
        collected = perf_counter() - started
        batch = collate_paired_episodes(episodes).to(device)
        model.train()
        taken: list[dict[str, float]] = []
        for _pass in range(config.passes):
            report = vtrace_market_loss(
                batch,
                model,
                entropy_coefficient=config.entropy_coefficient(update),
                value_weight=config.value_weight,
                auxiliary_weight=config.auxiliary_weight,
                clip_rho=config.clip_rho,
            )
            if not report.finite:
                raise OnlineNonFiniteError(f"update {update} produced a nonfinite loss")
            optimizer.zero_grad()
            report.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            taken.append(report.metrics())

        row = {"update": float(update)}
        row.update(
            {
                name: sum(entry[name] for entry in taken) / len(taken)
                for name in taken[0]
            }
        )
        row["max_rho"] = max(entry["max_rho"] for entry in taken)
        row["min_rho"] = min(entry["mean_rho"] for entry in taken)
        row.update(episode_metrics(episodes))
        row["collect_seconds"] = collected
        row["update_seconds"] = perf_counter() - started
        average = row["reward"] if update == 1 else 0.9 * average + 0.1 * row["reward"]
        row["reward_average"] = average
        row["entropy_coefficient"] = config.entropy_coefficient(update)
        history.append(row)
        if average > best["reward"]:
            best = {"update": float(update), "reward": average}
            save_checkpoint(
                root / BEST_CHECKPOINT,
                {
                    "model": model.state_dict(),
                    "update": update,
                    "binding_sha256": binding,
                },
            )
        save_checkpoint(
            last_path,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update": update,
                "best": best,
                "average": average,
                "history": history,
                "binding_sha256": binding,
                "wandb_run_id": run_id,
            },
        )
        LOGGER.info(
            "update %d/%d reward=%+.4f avg=%+.4f activation=%.4f "
            "merged=%.2f p_replace=%.5f rho=%.3f %.0fs (collect %.0fs)",
            update,
            config.updates,
            row["reward"],
            average,
            row["activation"],
            row["merged_share"],
            row["replace_probability"],
            row["min_rho"],
            row["update_seconds"],
            row["collect_seconds"],
        )
        if log is not None:
            log(update, row)

    document: dict[str, Any] = {
        "config": asdict(config),
        "binding_sha256": binding,
        "updates": update,
        "best": best,
        "history": history,
        "wandb_run_id": run_id,
    }
    write_json_atomic(root / REPORT_NAME, document)
    return document


def episode_metrics(episodes: Sequence[PairedEpisode]) -> dict[str, float]:
    """Return what one update's collected seasons did, as plain floats.

    Activation is measured against the events the residual was actually asked
    about, and the merged share against the replacements it proposed, so a run
    that deviates often but is refused every time is visible as such rather
    than reading as a policy that is doing something.

    Args:
        episodes: One update's paired episodes.

    Returns:
        The metrics row.
    """
    events = sum(len(episode.trajectory.steps) for episode in episodes)
    proposed = sum(
        int((episode.trajectory.modes == REPLACE_INDEX).sum()) for episode in episodes
    )
    merged = sum(episode.merged for episode in episodes)
    fallbacks = sum(sum(episode.fallbacks.values()) for episode in episodes)
    return {
        "reward": sum(terminal_reward(episode) for episode in episodes) / len(episodes),
        "candidate_points": sum(episode.candidate_points for episode in episodes)
        / len(episodes),
        "control_points": sum(episode.control_points for episode in episodes)
        / len(episodes),
        "margin_delta": sum(episode.normalized_margin_delta for episode in episodes)
        / len(episodes),
        "episodes": float(len(episodes)),
        "events": float(events),
        "proposed": float(proposed),
        "activation": merged / max(events, 1),
        "proposal_rate": proposed / max(events, 1),
        "merged_share": merged / max(proposed, 1),
        "fallback_rate": fallbacks / max(events, 1),
        "changed_episodes": float(sum(1 for episode in episodes if episode.merged > 0)),
    }
