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

A characterization test pins an agent against an *engine*, and this one has now
been re-recorded once for a reason that was not a policy change at all. Both
recordings are kept below rather than one overwriting the other, because a
fixture that silently rebaselines records nothing:

- **kaggle-environments 1.32.3**, measured 2026-08-04. ``LEGACY_FINGERPRINT``.
- **kaggle-environments 1.32.6**, measured 2026-08-08. ``FINGERPRINT``, and the
  digests in ``economic_policy_decisions.json``. The ladder moved to 1.32.6 on
  2026-08-07; it deleted the town centre's escalating demand schedule and
  doubled ``townCenterSellInterval`` from 12 to 24, cutting late-season
  town-centre demand about eightfold. The policy source is byte-identical
  across the two recordings — what changed is the economy it prices into.
  Divergence begins at turn 73 (day 3, hour 1) on all three seeds, which is the
  turn after the first shop opens and the agent first prices the book, and
  574–584 of the 719 turns differ thereafter.

So the two tables below are not "old" and "new" values of one measurement. They
are the same agent measured in two different economies, and the gap between
them is the size of the engine change, not of any edit of ours.

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

# Terminal bank, seat 0, opponent `pass`. See the module docstring: same agent,
# two engines. The 1.32.3 column is history and no decision may be taken on it.
LEGACY_FINGERPRINT = {11: 152803, 22: 152145, 33: 159087}  # 1.32.3, 2026-08-04
FINGERPRINT = {11: 149176, 22: 147181, 33: 108290}  # 1.32.6, 2026-08-08


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

    assert int(env.steps[-1][0].reward) == FINGERPRINT[seed], (
        f"seed {seed}: this pins the agent on kaggle-environments 1.32.6; it "
        f"banked {LEGACY_FINGERPRINT[seed]} here on 1.32.3. If the installed "
        f"engine moved again, re-record both tables rather than one"
    )
