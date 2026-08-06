"""Characterization tests for the vendored economic policy.

The economic policy arrived as 65KB of third-party code and is the code we
intend to change. Refactors of something that large are easy to get subtly
wrong in ways no unit test notices: a reordered comparison or a lost early
return changes what the farm does on day nine, the season still ends with
plausible-looking coins, and nothing raises.

It is no longer the agent ``main.py`` serves -- route memory is -- but it is
still shipped and still played, on the turns route memory finds no route near
enough to replay, so what it does on day nine matters exactly as much as it did
when it was the whole submission. Both tests below name the policy directly for
that reason; the bank test used to play ``main.py``, which quietly turned it
into a test of whatever the entrypoint happened to serve.

So these pin what the agent *does* rather than how it is written, at two
resolutions. The decision test compares every turn's action and names the first
turn that differs, which is what makes a broken refactor debuggable instead of
merely detected. The bank test is the coarse cross-check: it would catch a
divergence that somehow cancelled out across the season.

Both were recorded from the vendored code exactly as pulled, before any edit of
ours, so they describe upstream behaviour rather than ours. **Never regenerate a
fixture to make a failing test pass.** A change here means the policy changed,
which is sometimes the intent and must never be an accident — when it is
deliberate, re-record and say why in the commit message.

The opponent is `pass`, which keeps the other player out of the market entirely,
so these depend only on our own decisions and on the seed's weeds and shop
unlocks. That makes them a test of this agent rather than of a matchup.
"""

import hashlib
import json
from pathlib import Path

import pytest
from kaggle_environments import make

from kaggriculture import economic_policy
from kaggriculture.constants import ENVIRONMENT, EPISODE_STEPS

DECISIONS = json.loads(
    (Path(__file__).parent / "fixtures" / "economic_policy_decisions.json").read_text()
)

# Measured 2026-08-04 from the vendored policy as pulled, before any edits.
FINGERPRINT = {11: 152803, 22: 152145, 33: 159087}


def digest(action: dict) -> str:
    """Return a short stable digest of one turn's action."""
    return hashlib.sha256(
        json.dumps(action, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]


@pytest.mark.parametrize("seed", sorted(FINGERPRINT))
def test_agent_makes_the_same_decisions_it_always_has(seed: int) -> None:
    """Every turn's action must match what the vendored policy did on this seed."""
    expected = DECISIONS[str(seed)]
    actual: list[str] = []

    def record(obs: dict, config: object = None) -> dict:
        action = economic_policy.agent(obs)
        actual.append(digest(action))
        return action

    env = make(ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS, "seed": seed})
    env.run([record, "pass"])

    assert len(actual) == len(expected)
    divergent = [
        turn
        for turn, pair in enumerate(zip(actual, expected, strict=False))
        if pair[0] != pair[1]
    ]
    assert not divergent, (
        f"seed {seed}: behaviour changed at turn {divergent[0]} "
        f"(day {divergent[0] // 24}, hour {divergent[0] % 24}); "
        f"{len(divergent)} of {len(expected)} turns differ"
    )


@pytest.mark.parametrize("seed", sorted(FINGERPRINT))
def test_agent_banks_what_it_banked_before(seed: int) -> None:
    """The coarse cross-check: a season that ends where it always ended."""
    env = make(ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS, "seed": seed})

    env.run([economic_policy.agent, "pass"])

    assert int(env.steps[-1][0].reward) == FINGERPRINT[seed]
