"""Characterization tests for the vendored economic policy.

The economic policy arrived as 65KB of third-party code and is the code we
intend to change. Refactors of something that large are easy to get subtly
wrong in ways no unit test notices: a reordered comparison or a lost early
return changes what the farm does on day nine, the season still ends with
plausible-looking coins, and nothing raises.

It is no longer the agent ``main.py`` serves -- ``boatlee_v14_policy`` is -- but
it is still shipped and still played, on the turns route memory finds no route
near enough to replay, and it is the scripted opponent every RL arm trains
against and every evaluation is scored on. So what it does on day nine matters
exactly as much as it did when it was the whole submission. Both tests below
name the policy directly for that reason; the bank test used to play
``main.py``, which quietly turned it into a test of whatever the entrypoint
happened to serve.

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
been re-recorded twice for reasons that were not policy changes at all. Every
recording is kept rather than overwritten, because a fixture that silently
rebaselines records nothing:

- **kaggle-environments 1.32.3**, measured 2026-08-04. ``LEGACY_FINGERPRINT``.
- **kaggle-environments 1.32.6**, measured 2026-08-08. ``FINGERPRINT_1326``. The
  ladder moved to 1.32.6 on 2026-08-07; it deleted the town centre's escalating
  demand schedule and doubled ``townCenterSellInterval`` from 12 to 24, cutting
  late-season town-centre demand about eightfold. Divergence begins at turn 73
  (day 3, hour 1) on all three seeds, which is the turn after the first shop
  opens and the agent first prices the book, and 574–584 of the 719 turns differ
  thereafter.
- **kaggle-environments 1.32.7**, measured 2026-08-16. ``FINGERPRINT``. Carrot,
  tomato and egg moved onto a new ``hinge`` curve below ``I0``, and carrot's
  ``below_target`` went 0.20 to 1.00, so those three spike in price once they
  are genuinely scarce. This one *did* require an edit: the vendored ``MARKET``
  table and ``_shape`` here are copies of the engine's, and both had to gain the
  new curve before the price model agreed with the engine again.

The policy's own decision logic is byte-identical across all three. The digests
live in ``economic_policy_decisions.json``, keyed by the engine that produced
them, and are read by the *installed* version -- so bumping the engine without
re-recording fails with a ``KeyError`` naming the version rather than silently
comparing one economy's decisions against another's.

So these tables are not "old" and "new" values of one measurement. They are the
same agent measured in three different economies, and the gaps between them are
the size of the engine changes, not of any edit of ours.

The opponent is `pass`, which keeps the other player out of the market entirely,
so these depend only on our own decisions and on the seed's weeds and shop
unlocks. That makes them a test of this agent rather than of a matchup.
"""

import hashlib
import json
from importlib import metadata
from pathlib import Path

import pytest
from kaggle_environments import make

from kaggriculture import economic_policy
from kaggriculture.constants import ENVIRONMENT, EPISODE_STEPS

ENGINE = metadata.version("kaggle-environments")

# Keyed by the engine that produced them, so a third recording adds a column
# rather than overwriting the second. Reading it by the *installed* version is
# what makes an unreviewed engine bump fail loudly here instead of comparing
# this engine's decisions against another one's.
DECISIONS = json.loads(
    (Path(__file__).parent / "fixtures" / "economic_policy_decisions.json").read_text()
)[ENGINE]

# Terminal bank, seat 0, opponent `pass`. See the module docstring: same agent,
# three engines. The older columns are history and no decision may be taken on
# them.
LEGACY_FINGERPRINT = {11: 152803, 22: 152145, 33: 159087}  # 1.32.3, 2026-08-04
FINGERPRINT_1326 = {11: 149176, 22: 147181, 33: 108290}  # 1.32.6, 2026-08-08
FINGERPRINT = {11: 135735, 22: 145267, 33: 104544}  # 1.32.7, 2026-08-16


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
        f"seed {seed}: this pins the agent on kaggle-environments {ENGINE}; it "
        f"banked {FINGERPRINT_1326[seed]} here on 1.32.6 and "
        f"{LEGACY_FINGERPRINT[seed]} on 1.32.3. If the installed engine moved "
        f"again, add a column rather than overwriting one"
    )
