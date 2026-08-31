"""The wrapper that plays the frozen controller, plus at most a market residual.

The claim this module has to earn is narrow and absolute: a residual that never
deviates plays *the frozen controller*. Not something that banks the same, not
something that wins as often -- the same action object, every turn, on every
seat and every seed. Everything else here is built to make that claim checkable
rather than hoped for.

So the turn is ordered controller-first. ``MarketResidualAgent`` calls the
frozen controller before it looks at anything, and that action is the value it
returns unless a merge has *proved* a replacement legal. The encoding, the event
machine, the feature row and the residual all run downstream of an action that
already exists, under one blanket handler: an exception anywhere in that tail
returns the controller's own action object and is counted, because a submission
that raises scores zero and a submission that quietly plays the controller
scores what the controller scores. The one exception deliberately left to
propagate is a failure inside the frozen controller itself. There is no
meaningful action to fall back to when the thing we would fall back to is what
broke, and an episode that fails there must fail loudly, in a gate, rather than
be papered over with a pass.

The clock advances exactly once per turn on every path -- fallback, no-event and
merge alike -- because ``EventMemory`` is the only thing that knows how long it
has been since the last event, and a heartbeat that skipped the turns it failed
on would fire early for the rest of the season.

``act`` is what separates "the residual is deciding" from "the residual is
watching". Counterfactual collection needs the recurrent state advanced over
turns the residual did not play, or the state it is trained on is not the state
it will see. Passing the flag down to ``observe`` rather than skipping the call
means the observation half runs identically either way.

``AlwaysUseKaito`` is the identity element of that boundary and lives here, not
in the tests, because the parity property is a claim about production: it is the
residual every gate initialises from and the one every parity check forces.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from kaggriculture.features import encode_observation
from kaggriculture.market_residual.actions import (
    FallbackReason,
    ResidualDecision,
    ResidualMode,
    merge_residual_action,
)
from kaggriculture.market_residual.baseline import (
    BaselineAgent,
    BaselineIdentity,
    load_verified_baseline,
)
from kaggriculture.market_residual.events import EventConfig, EventMemory, detect_event
from kaggriculture.market_residual.features import (
    MarketFeatureVector,
    market_feature_vector,
)

DEFAULT_EVENT_CONFIG = EventConfig()


class ResidualInference(Protocol):
    """A learned market head, as the runtime is allowed to see it.

    Narrow on purpose: the runtime hands over one encoded row and the state it
    got back last time, and receives a decision and the next state. Nothing here
    can reach the board, the raw observation, or the frozen controller.
    """

    def initial_state(self) -> tuple[float, ...]:
        """Return the recurrent state an episode starts from."""
        ...

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
                head must still advance its state and must return no decision.

        Returns:
            The decision to merge, or ``None``, and the state to carry forward.
        """
        ...


@dataclass(frozen=True)
class AlwaysUseKaito:
    """The residual that never deviates: the identity of the merge boundary."""

    def initial_state(self) -> tuple[float, ...]:
        """Return the empty state; this head remembers nothing."""
        return ()

    def observe(
        self,
        features: MarketFeatureVector,
        state: tuple[float, ...],
        *,
        act: bool,
    ) -> tuple[ResidualDecision | None, tuple[float, ...]]:
        """Return a ``USE_KAITO`` decision and the state unchanged.

        Args:
            features: The canonical row for this market event, unread.
            state: The empty state.
            act: Whether the decision will be played; either way it defers.

        Returns:
            A decision that changes nothing, and ``state``.
        """
        return ResidualDecision(mode=ResidualMode.USE_KAITO, buckets=()), state


@dataclass
class ResidualRecord:
    """What one episode's boundary actually did, counted as it happened.

    The online gate reads ``replacements`` against ``events`` for its activation
    band and ``fallbacks`` for its failure count, so these are live counters on
    the played agent rather than something reconstructed from a replay.
    """

    turns: int = 0
    events: int = 0
    replacements: int = 0
    fallbacks: Counter[FallbackReason] = field(default_factory=Counter)


class MarketResidualAgent:
    """One seat of one episode: a frozen controller, plus at most a residual."""

    def __init__(
        self,
        baseline: BaselineAgent,
        residual: ResidualInference,
        config: EventConfig = DEFAULT_EVENT_CONFIG,
        act: bool = True,
    ) -> None:
        """Bind one episode's controller, head, thresholds, and authority.

        Args:
            baseline: A private copy of the frozen controller, from
                ``load_verified_baseline``.
            residual: The learned market head.
            config: The event thresholds in force for this run.
            act: Whether the residual's decisions may reach the board.
        """
        self.baseline = baseline
        self.residual = residual
        self.config = config
        self.act = act
        self.memory = EventMemory.initial()
        self.state = residual.initial_state()
        self.previous: MarketFeatureVector | None = None
        self.record = ResidualRecord()

    def __call__(
        self,
        observation: Mapping[str, Any],
        configuration: object | None = None,
    ) -> Mapping[str, Any]:
        """Return this seat's action for one engine turn.

        Args:
            observation: The raw observation the engine handed this seat.
            configuration: The engine configuration, passed through untouched.

        Returns:
            The frozen controller's own action object, unless a replacement was
            proved legal, in which case the merged action.
        """
        action = self.baseline(observation, configuration)
        self.record.turns += 1
        try:
            played = self._decide(observation, action)
        except Exception:
            self.record.fallbacks[FallbackReason.ERROR] += 1
            played = action
        self.memory = self.memory.advance(1)
        return played

    def _decide(
        self, observation: Mapping[str, Any], action: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Run the residual boundary for one turn, without the blanket handler.

        Args:
            observation: The raw observation the engine handed this seat.
            action: The frozen controller's action for this turn.

        Returns:
            The action to play.
        """
        encoded = encode_observation(observation, int(observation["player"]))
        transition = detect_event(encoded, action, self.memory, self.config)
        self.memory = transition.memory
        if transition.event is None:
            return action

        self.record.events += 1
        features = market_feature_vector(
            encoded, transition.event.kaito_buckets, self.previous
        )
        self.previous = features
        decision, self.state = self.residual.observe(features, self.state, act=self.act)
        if decision is None:
            return action

        result = merge_residual_action(encoded, action, decision)
        if result.fallback is not None:
            self.record.fallbacks[result.fallback] += 1
        if result.replaced:
            self.record.replacements += 1
        return result.action


def build_market_residual_agent(
    identity: BaselineIdentity,
    residual: ResidualInference,
    config: EventConfig = DEFAULT_EVENT_CONFIG,
    act: bool = True,
) -> MarketResidualAgent:
    """Return a wrapped agent on a private, digest-checked controller copy.

    Args:
        identity: Which frozen controller this episode is built on.
        residual: The learned market head.
        config: The event thresholds in force for this run.
        act: Whether the residual's decisions may reach the board.

    Returns:
        One agent, safe to play one episode of one seat.
    """
    return MarketResidualAgent(
        load_verified_baseline(identity), residual, config=config, act=act
    )
