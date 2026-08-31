"""Exact parity: with the residual deferring, the wrapper *is* the controller.

This is the property the whole plan rests on. Every gate downstream compares a
learned agent against the frozen controller on paired seeds; if wrapping the
controller changed its play even slightly, every one of those comparisons would
be measuring the wrapper as well as the residual, and the sign of a small effect
would not be recoverable afterwards.

So parity is asserted three ways, and each closes a hole the others leave open.

Action-by-action over a full season is the real statement: the wrapped seat's
719 recorded actions must equal the raw controller's, in order. Final banks
alone would not do it -- two different seasons can bank the same number -- but
they are asserted too, on the exact episode the promotion gate scored, so the
wrapped agent is checked against a published measurement rather than only
against a second run of this file's own code.

The third is the one that keeps the others from being vacuous. A wrapper that
returned the controller's action and did nothing else would pass both of the
above forever, and so would a wrapper whose event machine silently raised on
every turn. ``test_the_boundary_ran_over_the_whole_season`` reads the live
counters off the played agent and requires that the residual was actually
consulted, hundreds of times, with no error fallbacks; and
``test_a_replacing_residual_moves_the_board`` requires that a residual which
does deviate visibly changes the queue. Parity means the boundary ran and chose
not to move. It does not mean the boundary was not there.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pytest
from kaggle_environments import make

from kaggriculture.boatlee_v14_policy import agent as opponent_agent
from kaggriculture.constants import ENVIRONMENT
from kaggriculture.kaito_v54_policy import agent as kaito_agent
from kaggriculture.market_residual.actions import (
    FallbackReason,
    ResidualDecision,
    ResidualMode,
)
from kaggriculture.market_residual.baseline import SERVED
from kaggriculture.market_residual.features import MarketFeatureVector
from kaggriculture.market_residual.policy import (
    AlwaysUseKaito,
    MarketResidualAgent,
    build_market_residual_agent,
)
from kaggriculture.market_residual.schema import ALLOWED_SLOTS, LEARNABLE_VERBS
from tests.test_vendored_policies import V54_REFERENCE_BANKS, V54_REFERENCE_SEED

EPISODE_STEPS = 720

# Short enough to run several of, long enough for the controller to have opened
# shops and accumulated stock, which is when it trades at all.
PROBE_STEPS = 240

# Outside every declared bank in `schema.SEED_BANKS`, and outside the consumed
# and held-out ranges: a parity check is not a measurement and may not spend a
# seed a promotion decision will need.
PARITY_SEED = 4242
BOTH_SEAT_SEEDS = tuple(range(4300, 4308))


class CountingResidual:
    """``AlwaysUseKaito`` that records what it was asked, and when."""

    def __init__(self, decide: bool = True) -> None:
        """Bind whether this head returns a decision at all.

        Args:
            decide: When ``False`` the head advances its state and returns no
                decision, which is what a watching-only collection run does.
        """
        self.decide = decide
        self.calls: list[bool] = []

    def initial_state(self) -> tuple[float, ...]:
        """Return a one-element state so advancing it is observable."""
        return (0.0,)

    def observe(
        self,
        features: MarketFeatureVector,
        state: tuple[float, ...],
        *,
        act: bool,
    ) -> tuple[ResidualDecision | None, tuple[float, ...]]:
        """Record the call, advance the counter state, and defer to Kaito.

        Args:
            features: The canonical row for this market event.
            state: The state returned at the previous event.
            act: Whether this decision will be played.

        Returns:
            A ``USE_KAITO`` decision, or ``None`` when not deciding, and the
            state with its counter incremented.
        """
        self.calls.append(act)
        decision = (
            ResidualDecision(mode=ResidualMode.USE_KAITO, buckets=())
            if self.decide
            else None
        )
        return decision, (state[0] + 1.0,)


class DropAllCommodityOrders:
    """A residual that replaces the market queue with nothing.

    The only deviation that is legal on every turn without knowing the board:
    an empty replacement keeps the controller's ``HIRE``, ``BUY_LAND``,
    ``BUY_SEED`` and ``BUY_ANIMAL`` orders and drops its ``SELL`` and
    ``BUY_PRODUCT`` ones, so it costs nothing to prove and is visible whenever
    the controller wanted to trade.
    """

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
        """Ask for an all-zero replacement.

        Args:
            features: The canonical row for this market event, unread.
            state: The empty state.
            act: Whether the decision will be played.

        Returns:
            A ``REPLACE`` decision holding zero for every allowed slot, and
            ``state``.
        """
        buckets = tuple(0 for _ in ALLOWED_SLOTS)
        return ResidualDecision(mode=ResidualMode.REPLACE, buckets=buckets), state


def play(
    agents: Sequence[Any], seed: int, steps: int = EPISODE_STEPS
) -> list[list[Any]]:
    """Play one seeded episode and return every recorded step.

    Args:
        agents: The two seats, in engine order.
        seed: The engine seed.
        steps: How many engine steps the episode runs for.

    Returns:
        The engine's recorded steps.
    """
    environment = make(
        ENVIRONMENT,
        configuration={"episodeSteps": steps, "seed": seed},
        debug=False,
    )
    environment.run(list(agents))
    return environment.steps


def seat_actions(steps: Sequence[Any], seat: int) -> list[Mapping[str, Any] | None]:
    """Return one seat's recorded action for every step.

    Args:
        steps: The engine's recorded steps.
        seat: Which seat to read.

    Returns:
        The action recorded at each step, or ``None`` where none was.
    """
    return [step[seat].get("action") for step in steps]


def commodity_orders(steps: Sequence[Any], seat: int) -> int:
    """Return how many ``SELL`` and ``BUY_PRODUCT`` orders one seat queued.

    Args:
        steps: The engine's recorded steps.
        seat: Which seat to read.

    Returns:
        The count over the whole episode.
    """
    return sum(
        order[0] in LEARNABLE_VERBS
        for action in seat_actions(steps, seat)
        if action is not None
        for order in action["market"]
    )


def banks(steps: Sequence[Any]) -> tuple[int, int]:
    """Return both seats' final banks.

    Args:
        steps: The engine's recorded steps.

    Returns:
        The two final rewards as integers.
    """
    final = steps[-1]
    return (int(final[0].reward), int(final[1].reward))


def statuses(steps: Sequence[Any]) -> tuple[str, str]:
    """Return both seats' final status.

    Args:
        steps: The engine's recorded steps.

    Returns:
        The two final statuses as strings.
    """
    final = steps[-1]
    return (str(final[0].status), str(final[1].status))


@dataclass(frozen=True)
class Played:
    """One wrapped episode, with the two objects a test reads afterwards."""

    agent: MarketResidualAgent
    residual: CountingResidual
    steps: list[list[Any]]


@pytest.fixture(scope="module")
def forced() -> Played:
    """Play a full season with a residual that defers on every event."""
    residual = CountingResidual()
    agent = build_market_residual_agent(SERVED, residual)
    return Played(agent, residual, play([agent, opponent_agent], PARITY_SEED))


@pytest.fixture(scope="module")
def reference() -> list[list[Any]]:
    """Play the same season with the served controller, unwrapped."""
    return play([kaito_agent, opponent_agent], PARITY_SEED)


def test_forced_use_kaito_matches_kaito_action_by_action(
    forced: Played, reference: list[list[Any]]
) -> None:
    """The whole point: 719 turns, same actions, same order, same banks.

    Compared step by step rather than by final bank, because two different
    seasons can end on the same number and the residual this wrapper will carry
    is meant to be judged on differences far smaller than that.
    """
    assert seat_actions(forced.steps, 0) == seat_actions(reference, 0)
    assert banks(forced.steps) == banks(reference)
    assert statuses(forced.steps) == ("DONE", "DONE")
    assert forced.agent.record.turns == len(reference) - 1


def test_the_boundary_ran_over_the_whole_season(forced: Played) -> None:
    """Parity is only worth something if the boundary was actually running.

    A wrapper that returned the controller's action and did nothing else would
    satisfy every parity assertion in this file. So this reads the counters off
    the agent that just played: the residual was consulted at hundreds of real
    market events, the merge accepted every deferral, and nothing fell back
    through the blanket handler -- which is what an encoder or event-machine
    exception on every turn would look like from the outside.
    """
    record = forced.agent.record

    assert record.events > 300, f"only {record.events} events"
    assert forced.residual.calls == [True] * record.events
    assert record.replacements == 0
    assert record.fallbacks == {}
    assert forced.agent.state == (float(record.events),)


def test_forced_use_kaito_reproduces_the_reference_banks_the_gate_measured() -> None:
    """The wrapped agent must reproduce a published number, not just itself.

    ``tests/test_vendored_policies.py`` pins this pair for the unwrapped module
    on the episode the promotion gate scored. Reproducing it here means parity
    is anchored to a measurement that exists outside this file: if both the
    wrapper and the reference run in
    ``test_forced_use_kaito_matches_kaito_action_by_action`` were broken the same
    way, they would still agree with each other, and they would not agree with
    this.

    The seed is a held-out gate seed. That is deliberate and costs nothing: the
    gate it belongs to has already been run and published, and replaying a
    decided episode is not a new measurement.
    """
    agent = build_market_residual_agent(SERVED, AlwaysUseKaito())

    steps = play([agent, opponent_agent], V54_REFERENCE_SEED)

    assert statuses(steps) == ("DONE", "DONE")
    assert banks(steps) == V54_REFERENCE_BANKS


def test_a_replacing_residual_moves_the_board() -> None:
    """A residual that does deviate must visibly change what the seat plays.

    Without this, every parity assertion above could be satisfied by a merge
    that is incapable of replacing anything. Dropping the commodity orders is
    legal on any board, so a queue that comes back unchanged means the
    replacement path never ran.

    Counted rather than required to be empty: only event turns reach the merge,
    so on the turns the event machine passes over, the controller's own
    commodity orders still stand.
    """
    agent = build_market_residual_agent(SERVED, DropAllCommodityOrders())

    steps = play([agent, opponent_agent], PARITY_SEED, steps=PROBE_STEPS)
    reference = play([kaito_agent, opponent_agent], PARITY_SEED, steps=PROBE_STEPS)

    assert agent.record.replacements > 0
    assert seat_actions(steps, 0) != seat_actions(reference, 0)
    assert commodity_orders(steps, 0) < commodity_orders(reference, 0)


def test_a_watching_residual_plays_kaito_and_still_advances_its_state() -> None:
    """A watching residual sees the state it would have had, and plays nothing.

    Counterfactual collection replays a season the residual is not playing.
    ``act=False`` has to reach the head rather than skip the call: a head whose
    recurrent state only advanced on turns it acted would be trained on states
    the served agent never occupies.
    """
    residual = CountingResidual(decide=False)
    agent = build_market_residual_agent(SERVED, residual, act=False)

    steps = play([agent, opponent_agent], PARITY_SEED, steps=PROBE_STEPS)
    reference = play([kaito_agent, opponent_agent], PARITY_SEED, steps=PROBE_STEPS)

    assert seat_actions(steps, 0) == seat_actions(reference, 0)
    assert agent.record.events > 0
    assert residual.calls == [False] * agent.record.events
    assert agent.state == (float(agent.record.events),)
    assert agent.record.replacements == 0
    assert agent.record.fallbacks == {}


def test_a_residual_that_raises_is_a_counted_fallback_not_a_dead_episode() -> None:
    """The submission must survive its own learned head failing.

    A raise out of the residual is a fallback with a reason, not a traceback:
    the controller's action is already in hand, and an episode that dies scores
    zero where one that plays the controller scores what the controller scores.
    """

    class Broken:
        """A head that fails the way a bad export would."""

        def initial_state(self) -> tuple[float, ...]:
            """Return the empty state."""
            return ()

        def observe(
            self,
            features: MarketFeatureVector,
            state: tuple[float, ...],
            *,
            act: bool,
        ) -> tuple[ResidualDecision | None, tuple[float, ...]]:
            """Raise on every event.

            Args:
                features: The canonical row, never read.
                state: The empty state.
                act: Whether the decision would be played.

            Raises:
                ValueError: Always.
            """
            raise ValueError("exported weights are the wrong shape")

    agent = build_market_residual_agent(SERVED, Broken())

    steps = play([agent, opponent_agent], PARITY_SEED, steps=PROBE_STEPS)
    reference = play([kaito_agent, opponent_agent], PARITY_SEED, steps=PROBE_STEPS)

    assert statuses(steps)[0] == "DONE"
    assert seat_actions(steps, 0) == seat_actions(reference, 0)
    assert agent.record.fallbacks[FallbackReason.ERROR] > 0


@pytest.mark.slow
@pytest.mark.parametrize("seed", BOTH_SEAT_SEEDS)
@pytest.mark.parametrize("seat", (0, 1))
def test_forced_use_kaito_is_exact_in_both_seats_on_eight_seeds(
    seed: int, seat: int
) -> None:
    """One seed and one seat could agree by luck; sixteen cells cannot.

    Both seats matter because the seat index reaches the wrapper through
    ``observation["player"]`` and nothing else: encoding seat one's board as
    seat zero's would be a wrong feature row every turn, and under a deferring
    residual it would still play perfectly, so only a seat-one parity check can
    see whether the encoder was even given the right board.
    """
    agent = build_market_residual_agent(SERVED, CountingResidual())
    seats: list[Any] = [opponent_agent, opponent_agent]
    seats[seat] = agent
    raw: list[Any] = [opponent_agent, opponent_agent]
    raw[seat] = kaito_agent

    steps = play(seats, seed)
    reference = play(raw, seed)

    assert statuses(steps) == ("DONE", "DONE")
    assert seat_actions(steps, seat) == seat_actions(reference, seat)
    assert banks(steps) == banks(reference)
    assert agent.record.events > 0
