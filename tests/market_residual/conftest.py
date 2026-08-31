"""Real seeded engine episodes behind every market-residual feature test.

Each fixture here is a real observation taken from a real 720-step episode
played by the packaged engine agents. A hand-built market state would let a
wrong column order, a wrong scale, or a wrong recovery of an integer pass
unnoticed, because the fixture and the encoder would both be written from the
same idea of what the engine reports -- and it is the engine, not the idea,
that the residual will trade against.
"""

from typing import Any

import pytest
from kaggle_environments import make

from kaggriculture.features import EncodedObservation, encode_observation

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
