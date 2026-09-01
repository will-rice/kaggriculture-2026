"""Real seeded engine episodes behind every market-residual feature test.

Each fixture here is a real observation taken from a real 720-step episode
played by the packaged engine agents. A hand-built market state would let a
wrong column order, a wrong scale, or a wrong recovery of an integer pass
unnoticed, because the fixture and the encoder would both be written from the
same idea of what the engine reports -- and it is the engine, not the idea,
that the residual will trade against.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pytest
from kaggle_environments import make

from kaggriculture.features import EncodedObservation, encode_observation
from kaggriculture.kaito_v54_policy import agent as kaito_agent
from kaggriculture.learn.market_residual.alternatives import AlternativeConfig
from kaggriculture.learn.market_residual.counterfactual import (
    CounterfactualSnapshot,
    RecordedEvent,
    RecordedSeason,
    SeasonIdentity,
    record_season,
    season_events,
    snapshot_event,
)
from kaggriculture.market_residual.actions import ResidualDecision, ResidualMode
from kaggriculture.market_residual.baseline import SERVED
from kaggriculture.market_residual.events import EventConfig
from kaggriculture.market_residual.features import MarketFeatureVector
from kaggriculture.market_residual.policy import build_market_residual_agent

# Outside every declared bank in `schema.SEED_BANKS`: a fixture is not a
# measurement, and it may not consume a seed a promotion decision will need.
EPISODE_SEED = 4242

# Day 25, hour 0 of that episode: eight shops open, carrots in the shed, and
# visible opponent supply, so the shop, held-quantity and opponent-supply
# blocks are exercised on nonzero values rather than on an empty board.
EVENT_STEP = 600

# Day 20, hour 20 of the same episode. Between the two, live prices, market
# inventories and our cash all move, so the delta block has something to be
# wrong about.
EARLIER_STEP = 500


@pytest.fixture(scope="session")
def episode() -> list[Any]:
    """Play one real seeded episode and return every recorded engine step."""
    env = make("kaggriculture", configuration={"seed": EPISODE_SEED}, debug=False)
    env.run(["starter", "starter"])
    return env.steps


@pytest.fixture(scope="session")
def observation(episode: list[Any]) -> dict[str, Any]:
    """Return seat zero's raw observation at the mid-season market event."""
    return episode[EVENT_STEP][0]["observation"]


@pytest.fixture(scope="session")
def earlier_observation(episode: list[Any]) -> dict[str, Any]:
    """Return seat zero's raw observation at the earlier market event."""
    return episode[EARLIER_STEP][0]["observation"]


@pytest.fixture(scope="session")
def encoded(observation: dict[str, Any]) -> EncodedObservation:
    """Return the canonical encoding of the mid-season market event."""
    return encode_observation(observation, 0)


@pytest.fixture(scope="session")
def earlier_encoded(earlier_observation: dict[str, Any]) -> EncodedObservation:
    """Return the canonical encoding of the earlier market event."""
    return encode_observation(earlier_observation, 0)


@dataclass(frozen=True)
class Turn:
    """One recorded seat-turn of a real episode, encoded exactly once."""

    turn: int
    seat: int
    observation: dict[str, Any]
    encoded: EncodedObservation
    action: dict[str, Any]


@pytest.fixture(scope="session")
def kaito_episode() -> list[Any]:
    """Play one real seeded episode of the served controller against itself.

    The action-merge tests need a frozen controller's real market queues, not a
    starter agent's. What the merge has to survive -- ten-order turns, sales of
    stock the seat does not hold, quantities outside the canonical vocabulary --
    is what this controller actually emits, and none of it is guessable.
    """
    env = make("kaggriculture", configuration={"seed": EPISODE_SEED}, debug=False)
    env.run([kaito_agent, kaito_agent])
    return env.steps


@pytest.fixture(scope="session")
def kaito_turns(kaito_episode: list[Any]) -> tuple[Turn, ...]:
    """Return every recorded seat-turn of that episode with its encoding."""
    turns: list[Turn] = []
    for turn, step in enumerate(kaito_episode):
        for seat in (0, 1):
            observation = step[seat]["observation"]
            action = step[seat].get("action")
            if "farms" not in observation or not isinstance(action, dict):
                continue
            encoded = encode_observation(observation, seat)
            turns.append(Turn(turn, seat, observation, encoded, action))
    return tuple(turns)


class RecordingResidual:
    """A deferring head that keeps every feature row it was shown, in order.

    The recurrent tests need a real event sequence: the rows one seat of one
    real episode actually produces, at the turns the real event machine opened,
    under the real schema. Deferring on every one of them keeps the recorded
    season the frozen controller's own, so the sequence is the one a residual
    initialised from Kaito would first see.
    """

    def __init__(self) -> None:
        """Start with an empty transcript."""
        self.rows: list[MarketFeatureVector] = []

    def initial_state(self) -> tuple[float, ...]:
        """Return the empty state; this head remembers nothing itself."""
        return ()

    def observe(
        self,
        features: MarketFeatureVector,
        state: tuple[float, ...],
        *,
        act: bool,
    ) -> tuple[ResidualDecision | None, tuple[float, ...]]:
        """Record the row and defer to the frozen controller.

        Args:
            features: The canonical row for this market event.
            state: The empty state.
            act: Whether the decision will be played; either way it defers.

        Returns:
            A decision that changes nothing, and ``state``.
        """
        self.rows.append(features)
        return ResidualDecision(mode=ResidualMode.USE_KAITO, buckets=()), state


@pytest.fixture(scope="session")
def event_rows() -> tuple[MarketFeatureVector, ...]:
    """Return seat zero's market feature rows from one real season, in order."""
    recorder = RecordingResidual()
    seat = build_market_residual_agent(SERVED, recorder)
    environment = make(
        "kaggriculture", configuration={"seed": EPISODE_SEED}, debug=False
    )
    environment.run([seat, kaito_agent])
    return tuple(recorder.rows)


# The counterfactual fixtures below play whole seasons, so they are shared by
# every test that needs one and are declared here rather than in either file.

# A seed from the declared counterfactual training bank. Branching is the one
# place these tests must use a real bank seed: the artifact boundary rejects
# anything else, so a fixture seed would make the boundary untestable.
BRANCH_SEED = 860_000

BRANCH_SEAT: Literal[0, 1] = 0

# A budget small enough that a branch is a handful of arms rather than twenty.
# Each arm restores its own pair of controllers, and a restore is a second of
# work, so the shipped budget would make every branch test a minute long.
BRANCH_BUDGET = AlternativeConfig(max_single=2, max_ranked_multi=1, max_alternatives=8)

# Late enough that the branch is a short tail of the season and the test stays
# in seconds, late enough to be past every shop unlock, and not the last turn,
# so the forward loop actually runs.
BRANCH_TURN = 700


@pytest.fixture(scope="session")
def branch_identity() -> SeasonIdentity:
    """Return the run identity every counterfactual fixture is produced under."""
    return SeasonIdentity.current(
        learner=SERVED,
        opponent=SERVED,
        seed_bank="counterfactual_train",
        seed=BRANCH_SEED,
        seat=BRANCH_SEAT,
        event_config=EventConfig(),
        alternative_config=BRANCH_BUDGET,
    )


@pytest.fixture(scope="session")
def recorded_season(branch_identity: SeasonIdentity) -> RecordedSeason:
    """Play one real reference-engine season and keep both seats' transcripts."""
    return record_season(branch_identity)


@pytest.fixture(scope="session")
def recorded_events(recorded_season: RecordedSeason) -> tuple[RecordedEvent, ...]:
    """Return every market event the real machine opened on that season."""
    return season_events(recorded_season)


@pytest.fixture(scope="session")
def late_event(recorded_events: tuple[RecordedEvent, ...]) -> RecordedEvent:
    """Return the first event at or after the branch turn."""
    return next(event for event in recorded_events if event.event.turn >= BRANCH_TURN)


@pytest.fixture(scope="session")
def late_snapshot(
    recorded_season: RecordedSeason, late_event: RecordedEvent
) -> CounterfactualSnapshot:
    """Return the branchable instant that event sits at."""
    return snapshot_event(recorded_season, late_event)


@pytest.fixture(scope="session")
def smoke_sequences() -> tuple[Any, ...]:
    """Return real event sequences built from the Task 7 smoke collection.

    The smoke root is the profile Task 7 measured: eight ``counterfactual_train``
    seeds, both seats, eight events per cell at the sixteen-alternative budget.
    On a machine that has it, reconciling is a second of reading; on one that
    does not, the collection CLI rebuilds it once — minutes of real branching,
    which is what the ``slow`` mark on every consumer promises.
    """
    from kaggriculture.learn.market_residual.offline import load_sequences
    from kaggriculture.scripts.market_counterfactuals import main as collect

    root = Path(__file__).parents[2] / "run/market-residual/counterfactual-smoke"
    mode = "--resume" if (root / "collection-parameters.json").exists() else "--create"
    collect(
        [
            str(root),
            mode,
            "--seed-bank",
            "counterfactual_train",
            "--seed-start",
            "860000",
            "--seed-count",
            "8",
            "--workers",
            "8",
            "--max-events",
            "8",
            "--max-alternatives",
            "16",
        ]
    )
    return load_sequences(root, workers=8)
