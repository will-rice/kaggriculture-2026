"""Playing one season twice from the same instant, and scoring the difference.

A counterfactual label is a claim that *this* market order, at *this* turn, was
worth this much. The claim is only as good as the branch behind it, and a branch
has three ways to be quietly wrong: it can start from a state that is not the
state the event was opened on, it can hand the frozen controller a mind that has
not lived through the season so far, or it can score a season that never
finished. None of the three raises. All three produce a number.

So the recorded season is played by the **reference engine**, not by us. The
engine's own recorded step ``k`` is packed into tensors, and the branch runs
forward from there in the batched simulator, one batch row per alternative. That
choice makes the control arm checkable rather than assumed: row zero of every
branch replays the controller's own action, and its terminal banks must equal
the banks the reference engine actually finished that season with.
``branch_event`` asserts exactly that, on every event it scores, so the whole
dataset carries a per-row proof that the branch machinery agrees with the engine
it will be graded by. A differential that only ran in tests would tell us the
machinery was right on the seeds the tests picked.

The alignment is the one ``search.arena._replay`` paid for. Agent call ``j`` of a
season sees the observation whose ``step`` field reads ``j``, and the action it
returns is recorded at ``steps[j + 1]`` -- at the index of the state it produced,
one ahead of the state it was chosen from. A snapshot at event turn ``k`` is
therefore ``pack(steps[k])``, its transcripts cover calls ``0..k`` inclusive, and
the branch plays turns ``k..718``. Shifting that by one produces a season that
still finishes, still banks, and is a different game.

**Restoration, not copying.** The frozen controller keeps its whole per-episode
memory in module-level closures; there is no state object to copy. A branch row
therefore gets a *fresh* controller replayed through every observation the
recorded season showed it, with each replayed action checked against the one the
season recorded. That is the only construction that can be checked at all: if the
replay were to diverge, the restored controller would be a different agent, and
the check says so instead of the reward saying so three weeks later.

**The anchor never merges.** Row zero of a branch is the controller's own action
object, taken verbatim. It is not built by asking ``merge_residual_action`` to
prove the controller's own queue legal, because that question has a different
answer: the merge refuses Kaito's own buckets on 270 of 1402 real seat-turns --
it proves *replacements*, and the controller's queue is not one. Merging it would
mark 19% of the controller's own actions illegal and attach every label at those
events to an action nobody played. ``AlternativeSet`` keeps the anchor out of the
replacement sequence by type, and ``branch_rows`` binds it to a separate name
before the merge loop begins, so the loop that merges cannot reach it.

Determinism is a requirement, not a hope: the same season branched at the same
event with the same alternative must produce byte-identical artifacts across runs
and processes. Everything ordered here is ordered explicitly, every digest is
taken over canonical JSON, and the simulator's own RNG is materialized from the
episode seed rather than carried.
"""

import copy
import hashlib
import json
from dataclasses import dataclass, fields
from typing import Any, Literal, Mapping, Sequence

import torch
from kaggle_environments import make

from kaggriculture.constants import EPISODE_STEPS
from kaggriculture.features import EncodedObservation, encode_observation
from kaggriculture.learn.encoding import MAX_UNITS, UNIT_OPS
from kaggriculture.learn.market_residual.alternatives import (
    FAMILY_KAITO,
    AlternativeConfig,
    AlternativeSet,
    generate_alternatives,
)
from kaggriculture.market_residual.actions import (
    ResidualDecision,
    ResidualMode,
    merge_residual_action,
)
from kaggriculture.market_residual.baseline import (
    BaselineAgent,
    BaselineIdentity,
    load_verified_baseline,
)
from kaggriculture.market_residual.events import (
    EventConfig,
    EventMemory,
    MarketEvent,
    detect_event,
)
from kaggriculture.market_residual.schema import MarketFeatureSchema, validate_seed_bank
from kaggriculture.sim.engine import MarketActions, step
from kaggriculture.sim.fidelity import reference_identity
from kaggriculture.sim.rollout import encode_turn
from kaggriculture.sim.state import PRODUCT_NAMES, SimState, pack, unpack

ENVIRONMENT = "kaggriculture"

# The one engine identity every market-residual artifact is produced under.
ENGINE_VERSION = "1.32.7"

PASS_OPERATION = UNIT_OPS.index("PASS")

# The last agent call of a season. Call `j` sees the observation whose `step`
# field reads `j`; the terminal state at `EPISODE_STEPS - 1` is never acted on.
LAST_CALL = EPISODE_STEPS - 2


class CounterfactualIntegrityError(RuntimeError):
    """Raised when a recorded, restored, or branched season is not exact."""


def win_points(ours: int, theirs: int) -> float:
    """Return one game's score for the learner: 1.0/0.5/0.0 win/tie/loss.

    Restated from ``search.arena._win`` rather than imported, because this
    package must not pull the search tree -- and its engine, hybrid runtime and
    process pool -- into collection. ``tests/market_residual/test_counterfactual``
    pins the two equal, so moving the ladder's rule moves this with it.

    Args:
        ours: The learner's terminal bank.
        theirs: The opponent's terminal bank.

    Returns:
        ``1.0`` for a win, ``0.5`` for a tie, ``0.0`` for a loss.
    """
    return 1.0 if ours > theirs else 0.5 if ours == theirs else 0.0


def normalized_margin(ours: int, theirs: int) -> float:
    """Return a paired bank margin scaled into ``[-1, 1]`` without changing sign.

    Restated from ``search.arena._normalized_margin`` for the same reason
    ``win_points`` is, and pinned equal by the same test.

    Args:
        ours: The learner's terminal bank.
        theirs: The opponent's terminal bank.

    Returns:
        The scaled margin.
    """
    return (ours - theirs) / max(abs(ours) + abs(theirs), 1)


@dataclass(frozen=True)
class SeasonIdentity:
    """Everything a counterfactual row was produced under, as data.

    A row that outlives the code that made it is only usable if it says which
    engine, which controllers, which seat, which thresholds and which feature
    layout produced it. All of that is here, and its digest is what every row
    and every shard is stamped with.
    """

    engine_version: str
    engine_source_sha256: str
    engine_configuration_sha256: str
    learner: BaselineIdentity
    opponent: BaselineIdentity
    seed_bank: str
    seed: int
    seat: Literal[0, 1]
    feature_schema_sha256: str
    event_config: EventConfig
    alternative_config: AlternativeConfig

    @classmethod
    def current(
        cls,
        learner: BaselineIdentity,
        opponent: BaselineIdentity,
        seed_bank: str,
        seed: int,
        seat: Literal[0, 1],
        event_config: EventConfig,
        alternative_config: AlternativeConfig,
    ) -> "SeasonIdentity":
        """Return the identity of a season about to be played here and now.

        Args:
            learner: The controller in the seat the residual will wrap.
            opponent: The controller in the other seat.
            seed_bank: The declared bank ``seed`` must belong to.
            seed: The episode seed.
            seat: Which seat the learner holds.
            event_config: The event thresholds in force.
            alternative_config: The alternative budget in force.

        Returns:
            The identity, with the installed engine's digests filled in.

        Raises:
            CounterfactualIntegrityError: If the installed engine is not the one
                every market-residual artifact is declared under.
            ValueError: If the seat is not a seat, or the seed is not in bank.
        """
        engine = reference_identity()
        if engine.version != ENGINE_VERSION:
            raise CounterfactualIntegrityError(
                f"engine is {engine.version}, expected {ENGINE_VERSION}"
            )
        if seat not in (0, 1):
            raise ValueError(f"seat must be 0 or 1, got {seat}")
        validate_seed_bank(seed_bank, (seed,))
        return cls(
            engine_version=engine.version,
            engine_source_sha256=engine.source_sha256,
            engine_configuration_sha256=engine.configuration_sha256,
            learner=learner,
            opponent=opponent,
            seed_bank=seed_bank,
            seed=seed,
            seat=seat,
            feature_schema_sha256=MarketFeatureSchema.current().sha256,
            event_config=event_config,
            alternative_config=alternative_config,
        )

    def canonical(self) -> dict[str, Any]:
        """Return this identity as canonical, JSON-safe, ordered data.

        Returns:
            A mapping whose keys are sorted at serialization time and whose
            values are JSON scalars, lists, or mappings.
        """
        return {
            "engine_version": self.engine_version,
            "engine_source_sha256": self.engine_source_sha256,
            "engine_configuration_sha256": self.engine_configuration_sha256,
            "learner": _identity_fields(self.learner),
            "opponent": _identity_fields(self.opponent),
            "seed_bank": self.seed_bank,
            "seed": self.seed,
            "seat": self.seat,
            "feature_schema_sha256": self.feature_schema_sha256,
            "event_config": _config_fields(self.event_config),
            "alternative_config": _config_fields(self.alternative_config),
        }

    @property
    def sha256(self) -> str:
        """Return the digest over this identity's canonical JSON."""
        return canonical_digest(self.canonical())


def _identity_fields(identity: BaselineIdentity) -> dict[str, str]:
    """Return one controller identity as canonical data.

    Args:
        identity: The controller to describe.

    Returns:
        Its module, verified-region marker, and digest.
    """
    return {
        "module": identity.module,
        "body_start": identity.body_start,
        "sha256": identity.sha256,
    }


def _config_fields(config: EventConfig | AlternativeConfig) -> dict[str, Any]:
    """Return a frozen configuration dataclass as canonical data.

    Args:
        config: The configuration to describe.

    Returns:
        Its declared fields in declaration order.
    """
    return {field.name: getattr(config, field.name) for field in fields(config)}


def canonical_json(payload: Any) -> str:  # noqa: ANN401 - any JSON-safe value
    """Return one value's canonical JSON: sorted keys, no spaces, no NaN.

    Every digest and every published byte in this package goes through here, so
    two processes that agree on the value agree on the bytes.

    Args:
        payload: Any JSON-serializable value.

    Returns:
        Its canonical encoding.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_digest(payload: Any) -> str:  # noqa: ANN401 - any JSON-safe value
    """Return the SHA-256 of one value's canonical JSON.

    Args:
        payload: Any JSON-serializable value.

    Returns:
        The hex digest of its canonical encoding.
    """
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def canonical_action(action: Mapping[str, Any]) -> str:
    """Return one engine action as the canonical string a transcript compares.

    Args:
        action: An engine action.

    Returns:
        Its canonical JSON encoding.
    """
    return canonical_json(action)


@dataclass(frozen=True)
class RestoredAgent:
    """A fresh controller replayed back to where a recorded season left it."""

    agent: BaselineAgent
    configuration: Mapping[str, Any]
    replayed_actions: tuple[str, ...]

    def play(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return this controller's action, called the way the engine calls it.

        Args:
            observation: The runtime observation for its seat.

        Returns:
            The controller's action.
        """
        return self.agent(observation, self.configuration)


@dataclass(frozen=True)
class AgentTranscript:
    """Every observation one seat saw, and every action it answered with.

    The observations are the ones the *runtime* handed the agent, not the
    projection the engine records: seat one's recorded observation carries no
    ``step`` field, and this controller plays a whole season of ``PASS`` without
    one. Storing what the agent was actually called with is what makes a replay
    a replay.

    ``configuration`` is stored for the same reason, and it is stored on
    principle rather than on the strength of a measurement. The engine passes it
    on every call, so a replay that drops it is replaying a different question.
    That this is not academic was learned from the v54 controller, which read it
    on exactly one turn of one recorded season -- replaying with ``None``
    reproduced 689 of 719 turns and then quietly reordered a market queue. The
    v56 controller now frozen here appears to read it on none: measured over
    full seasons on three seeds, its actions are identical whether it is handed
    the real configuration, an empty one, or nothing. The field stays because
    the next kernel is not required to be as incurious, and because a failure
    here does not raise -- it produces labels for a season nobody played.
    ``tests/market_residual/test_counterfactual.py`` pins the call site rather
    than an outcome, for the same reason.
    """

    source: BaselineIdentity
    configuration: Mapping[str, Any]
    observations: tuple[Mapping[str, Any], ...]
    actions: tuple[str, ...]

    def restore(self) -> RestoredAgent:
        """Return a fresh controller advanced to the end of this transcript.

        Returns:
            The controller and the actions it produced while being replayed.

        Raises:
            CounterfactualIntegrityError: If a replayed action differs from the
                one the season recorded, which means the restored controller is
                not the controller that played it.
        """
        agent = load_verified_baseline(self.source)
        replayed: list[str] = []
        for index, observation in enumerate(self.observations):
            produced = canonical_action(agent(observation, self.configuration))
            if produced != self.actions[index]:
                raise CounterfactualIntegrityError(
                    f"restored controller diverged at call {index}: "
                    f"{produced} != {self.actions[index]}"
                )
            replayed.append(produced)
        return RestoredAgent(agent, self.configuration, tuple(replayed))

    def through(self, call: int) -> "AgentTranscript":
        """Return the prefix of this transcript up to and including one call.

        Args:
            call: The last agent call the prefix covers.

        Returns:
            A transcript of calls ``0..call``.
        """
        return AgentTranscript(
            self.source,
            self.configuration,
            self.observations[: call + 1],
            self.actions[: call + 1],
        )


@dataclass(frozen=True)
class RecordedSeason:
    """One reference-engine season, with both seats' transcripts and its banks."""

    identity: SeasonIdentity
    steps: tuple[Any, ...]
    learner: AgentTranscript
    opponent: AgentTranscript
    terminal_banks: tuple[int, int]


@dataclass(frozen=True)
class RecordedEvent:
    """One market event the real event machine opened, and where it sits.

    ``memory`` is the event machine's own state on the way out of that turn --
    clock advanced, references updated -- so a snapshot can carry the machine
    forward rather than rebuild it by replaying the season again per event.
    """

    index: int
    event: MarketEvent
    memory: EventMemory


@dataclass(frozen=True)
class CounterfactualSnapshot:
    """One exact instant of one season, and everything needed to leave it twice."""

    identity: SeasonIdentity
    index: int
    event: MarketEvent
    state: SimState
    encoded: EncodedObservation
    action: Mapping[str, Any]
    opponent_action: Mapping[str, Any]
    learner_transcript: AgentTranscript
    opponent_transcript: AgentTranscript
    memory: EventMemory
    alternatives: AlternativeSet
    control_banks: tuple[int, int]


@dataclass(frozen=True)
class BranchRow:
    """One branch arm: which alternative it is, and the action it plays."""

    family: str
    buckets: tuple[int, ...]
    action: Mapping[str, Any]


@dataclass(frozen=True)
class CounterfactualOutcome:
    """What one branch arm finished the season with."""

    banks: tuple[int, int]
    win_points: float
    margin: float
    terminal_prices: tuple[int, ...]


def record_season(identity: SeasonIdentity) -> RecordedSeason:
    """Play one reference-engine season and keep both seats' exact transcripts.

    Args:
        identity: Which controllers, seed, seat, and thresholds to play under.

    Returns:
        The recorded season, its transcripts, and its terminal banks.

    Raises:
        CounterfactualIntegrityError: If the season did not finish cleanly, or
            either seat was asked for a number of turns the alignment does not
            allow.
    """
    learner = _Recorder(identity.learner)
    opponent = _Recorder(identity.opponent)
    sides: list[Any] = [None, None]
    sides[identity.seat] = learner
    sides[1 - identity.seat] = opponent
    environment = make(
        ENVIRONMENT,
        configuration={"episodeSteps": EPISODE_STEPS, "seed": identity.seed},
        debug=False,
    )
    environment.reset(2)
    # Packing the live environment once is the configuration check: `pack`
    # refuses a board, market or shed the tensor schema does not describe.
    pack([environment])
    environment.run(sides)

    final = environment.steps[-1]
    statuses = tuple(str(final[seat]["status"]) for seat in range(2))
    if statuses != ("DONE", "DONE"):
        raise CounterfactualIntegrityError(
            f"seed {identity.seed} finished with statuses {statuses}"
        )
    for name, recorder in (("learner", learner), ("opponent", opponent)):
        if len(recorder.observations) != LAST_CALL + 1:
            raise CounterfactualIntegrityError(
                f"{name} was called {len(recorder.observations)} times, "
                f"expected {LAST_CALL + 1}"
            )
    banks = tuple(
        int(final[seat]["observation"]["farms"][seat]["money"]) for seat in range(2)
    )
    return RecordedSeason(
        identity=identity,
        steps=tuple(environment.steps),
        learner=recorder_transcript(identity.learner, learner),
        opponent=recorder_transcript(identity.opponent, opponent),
        terminal_banks=(banks[identity.seat], banks[1 - identity.seat]),
    )


class _Recorder:
    """One seat of one recorded season: a private controller that keeps its tape."""

    def __init__(self, identity: BaselineIdentity) -> None:
        """Load a private copy of one controller and start an empty tape.

        Args:
            identity: Which controller this seat plays.
        """
        self.agent = load_verified_baseline(identity)
        self.configuration: Mapping[str, Any] | None = None
        self.observations: list[Mapping[str, Any]] = []
        self.actions: list[str] = []

    def __call__(
        self,
        observation: Mapping[str, Any],
        configuration: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Play one turn and record exactly what was seen and answered.

        Args:
            observation: The raw observation the engine handed this seat.
            configuration: The engine configuration, passed through untouched.

        Returns:
            The controller's own action.
        """
        if self.configuration is None:
            self.configuration = copy.deepcopy(configuration)
        self.observations.append(copy.deepcopy(dict(observation)))
        action = self.agent(observation, configuration)
        self.actions.append(canonical_action(action))
        return action


def recorder_transcript(
    identity: BaselineIdentity, recorder: _Recorder
) -> AgentTranscript:
    """Freeze one seat's recorded tape into an immutable transcript.

    Args:
        identity: Which controller produced the tape.
        recorder: The seat that played the season.

    Returns:
        The transcript of every call that seat answered.

    Raises:
        CounterfactualIntegrityError: If the seat was never called, so there is
            no configuration to replay it under.
    """
    if recorder.configuration is None:
        raise CounterfactualIntegrityError(f"{identity.module} was never called")
    return AgentTranscript(
        identity,
        recorder.configuration,
        tuple(recorder.observations),
        tuple(recorder.actions),
    )


def season_events(season: RecordedSeason) -> tuple[RecordedEvent, ...]:
    """Return every market event the learner's own event machine opened.

    The clock is advanced exactly once per turn and the event is detected before
    the advance, which is the order ``policy.MarketResidualAgent`` uses; the
    event's ``turn`` is therefore the agent call index, and the observation it
    was opened on is ``season.learner.observations[turn]``.

    Args:
        season: A recorded reference-engine season.

    Returns:
        The events in the order they opened.
    """
    memory = EventMemory.initial()
    events: list[RecordedEvent] = []
    for call, observation in enumerate(season.learner.observations):
        encoded = encode_observation(observation, season.identity.seat)
        action = json.loads(season.learner.actions[call])
        transition = detect_event(encoded, action, memory, season.identity.event_config)
        memory = transition.memory
        memory = memory.advance(1)
        if transition.event is not None:
            events.append(RecordedEvent(len(events), transition.event, memory))
    return tuple(events)


def snapshot_event(
    season: RecordedSeason, recorded: RecordedEvent
) -> CounterfactualSnapshot:
    """Return the exact branchable instant one recorded event sits at.

    Args:
        season: The recorded reference-engine season.
        recorded: One event that season opened.

    Returns:
        The packed state, both transcripts through the event turn, the
        controller's own action, and the event's bounded alternatives.

    Raises:
        CounterfactualIntegrityError: If the packed state is not the state the
            engine recorded at that turn.
    """
    turn = recorded.event.turn
    observation = season.learner.observations[turn]
    encoded = encode_observation(observation, season.identity.seat)
    action = json.loads(season.learner.actions[turn])
    state = pack([(season.steps[turn], season.identity.seed)])
    if int(state.step[0]) != turn:
        raise CounterfactualIntegrityError(
            f"packed state reads step {int(state.step[0])}, event opened at {turn}"
        )
    return CounterfactualSnapshot(
        identity=season.identity,
        index=recorded.index,
        event=recorded.event,
        state=state,
        encoded=encoded,
        action=action,
        opponent_action=json.loads(season.opponent.actions[turn]),
        learner_transcript=season.learner.through(turn),
        opponent_transcript=season.opponent.through(turn),
        memory=recorded.memory,
        alternatives=generate_alternatives(
            recorded.event, encoded, action, season.identity.alternative_config
        ),
        control_banks=season.terminal_banks,
    )


def branch_rows(snapshot: CounterfactualSnapshot) -> tuple[BranchRow, ...]:
    """Return one playable arm per alternative, anchor first.

    The anchor is bound to its own name before the merge loop exists, so the
    loop that proves replacements legal cannot reach the controller's own
    action. It is also the caller's own object, not a rebuilt copy: the branch
    plays the very action the recorded season played.

    Args:
        snapshot: The instant to branch.

    Returns:
        Row zero playing the controller's action verbatim, then one row per
        replacement in the alternative set's own order.

    Raises:
        CounterfactualIntegrityError: If the alternatives were generated on a
            different turn, or if a replacement the generator proved legal is no
            longer legal.
    """
    if snapshot.alternatives.fingerprint != snapshot.event.fingerprint:
        raise CounterfactualIntegrityError(
            f"alternatives were generated on {snapshot.alternatives.fingerprint}, "
            f"the event opened on {snapshot.event.fingerprint}"
        )
    anchor = BranchRow(FAMILY_KAITO, snapshot.alternatives.kaito, snapshot.action)
    replacements: list[BranchRow] = []
    for alternative in snapshot.alternatives.replacements:
        merged = merge_residual_action(
            snapshot.encoded,
            snapshot.action,
            ResidualDecision(ResidualMode.REPLACE, alternative.buckets),
        )
        if not merged.replaced:
            raise CounterfactualIntegrityError(
                f"{alternative.family} row {alternative.buckets} was generated "
                f"legal and merged {merged.fallback}"
            )
        replacements.append(
            BranchRow(alternative.family, alternative.buckets, merged.action)
        )
    return (anchor, *replacements)


def branch_event(
    snapshot: CounterfactualSnapshot, rows: Sequence[BranchRow]
) -> tuple[CounterfactualOutcome, ...]:
    """Play every arm of one branch to the end of the season, exactly.

    Every arm starts from the same packed instant, plays its own action on the
    event turn against the opponent's recorded action, and is then played out by
    controllers restored from the recorded transcripts.

    Args:
        snapshot: The instant to branch.
        rows: The arms to play, the anchor first, as ``branch_rows`` returns.

    Returns:
        One outcome per arm, in the order given.

    Raises:
        CounterfactualIntegrityError: If the anchor is not first, if any arm
            fails to finish, or if the anchor's terminal banks are not the banks
            the reference engine finished this season with.
    """
    if not rows or rows[0].family != FAMILY_KAITO:
        raise CounterfactualIntegrityError("a branch must open on the anchor row")
    if any(row.family == FAMILY_KAITO for row in rows[1:]):
        raise CounterfactualIntegrityError("only row zero may be the anchor")

    seat = snapshot.identity.seat
    state = replicate_state(snapshot.state, len(rows))
    learners = [snapshot.learner_transcript.restore() for _ in rows]
    opponents = [snapshot.opponent_transcript.restore() for _ in rows]

    turn = snapshot.event.turn
    opened: list[list[Mapping[str, Any]]] = []
    for row in rows:
        pair: list[Mapping[str, Any]] = [
            snapshot.opponent_action,
            snapshot.opponent_action,
        ]
        pair[seat] = row.action
        opened.append(pair)
    state = step(state, *_encode_batch(opened))

    for _ in range(turn + 1, LAST_CALL + 1):
        played: list[list[Mapping[str, Any]]] = []
        for batch in range(len(rows)):
            pair = []
            for index in (0, 1):
                restored = learners[batch] if index == seat else opponents[batch]
                pair.append(restored.play(runtime_observation(state, batch, index)))
            played.append(pair)
        state = step(state, *_encode_batch(played))

    unfinished = int((~state.done).sum())
    if unfinished:
        raise CounterfactualIntegrityError(
            f"branch at event turn {turn} left {unfinished} arms unfinished"
        )
    outcomes = tuple(_outcome(state, batch, seat) for batch in range(len(rows)))
    if outcomes[0].banks != snapshot.control_banks:
        raise CounterfactualIntegrityError(
            f"anchor branched to {outcomes[0].banks}, the reference engine "
            f"finished this season at {snapshot.control_banks}"
        )
    return outcomes


def _outcome(state: SimState, batch: int, seat: int) -> CounterfactualOutcome:
    """Return one finished arm's banks, score, margin, and closing quotes.

    Args:
        state: The finished batched state.
        batch: Which arm to read.
        seat: Which seat the learner holds.

    Returns:
        That arm's outcome, learner first.
    """
    ours = int(state.money[batch, seat])
    theirs = int(state.money[batch, 1 - seat])
    return CounterfactualOutcome(
        banks=(ours, theirs),
        win_points=win_points(ours, theirs),
        margin=normalized_margin(ours, theirs),
        terminal_prices=tuple(
            int(state.prices[batch, index]) for index in range(len(PRODUCT_NAMES))
        ),
    )


def clone_state(state: SimState) -> SimState:
    """Return a state sharing no storage with the one it was cloned from.

    Args:
        state: The state to clone.

    Returns:
        A state whose every plane is an independent copy.
    """
    return SimState(
        **{field.name: getattr(state, field.name).clone() for field in fields(state)}
    )


def replicate_state(state: SimState, count: int) -> SimState:
    """Return ``count`` independent copies of a one-game state, batched.

    Args:
        state: A batch-of-one state.
        count: How many arms the branch has.

    Returns:
        A batch of ``count`` identical, independent games.

    Raises:
        ValueError: If the state is not a batch of one.
    """
    if state.batch_size != 1:
        raise ValueError(f"expected a batch of one, got {state.batch_size}")
    return SimState(
        **{
            field.name: getattr(state, field.name).tile(
                (count, *(1,) * (getattr(state, field.name).ndim - 1))
            )
            for field in fields(state)
        }
    )


def runtime_observation(state: SimState, batch: int, seat: int) -> dict[str, Any]:
    """Return the observation the engine's *runtime* hands one seat.

    ``sim.state.unpack`` reproduces the engine's recorded projection, which puts
    ``step`` on seat zero only. The runtime hands it to both, and this
    controller plays a whole season of ``PASS`` without it, so the field is
    restored here rather than left to the seat that happens to have it.

    Args:
        state: The batched state to read.
        batch: Which game in the batch.
        seat: Which seat to observe as.

    Returns:
        One raw observation.
    """
    return {"step": int(state.step[batch]), **unpack(state, batch, seat)}


def _encode_batch(
    actions: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[torch.Tensor, MarketActions, torch.Tensor]:
    """Return one turn's actions as the tensors ``sim.engine.step`` consumes.

    Args:
        actions: One pair of seat actions per batch row, seat zero first.

    Returns:
        Unit operations, compacted market orders, and unit quantities.
    """
    batch = len(actions)
    units = torch.full((batch, 2, MAX_UNITS), PASS_OPERATION, dtype=torch.int16)
    quantities = torch.ones((batch, 2, MAX_UNITS), dtype=torch.int16)
    market = MarketActions.empty(batch)
    for row, pair in enumerate(actions):
        for seat, action in enumerate(pair):
            encoded = encode_turn(action)
            for unit, operation in enumerate(encoded.units):
                units[row, seat, unit] = operation
            for unit, quantity in enumerate(encoded.quantities):
                quantities[row, seat, unit] = quantity
            for slot, (kind, item, quantity) in enumerate(encoded.orders):
                market.order_type[row, seat, slot] = kind
                market.order_item[row, seat, slot] = item
                market.order_qty[row, seat, slot] = quantity
    return units, market.compacted(), quantities
