"""Behaviour tests for the vendored public agents newer than v21.1.

Every file here is a decoded copy of a published kernel that shipped as a
base85 blob, so nothing about them is readable at a glance and nothing here
asserts what they say. These pin what they do: that the embedded tables decode
to the shapes the replay logic expects, that a well-formed action comes back
every turn, and that a turn fits the sandbox budget -- the same three
properties ``test_kaito_policy.py`` pins for v21.1.

v23 gets one test the others cannot have. It is the only vendored agent that
reads the engine configuration, and it uses it to choose between two whole
route tables: the ladder's move to ``townCenterSellInterval`` 24 in
kaggle-environments 1.32.6 is the entire reason the file exists. A copy whose
rebalance branch had been truncated would still import, still play, and still
pass every other test in this file, while silently playing the wrong season on
the only engine that scores us.

v54 gets two the others cannot have, because it is the one kept byte-verbatim
and the one we serve. Its header asserts the SHA-256 the notebook publishes over
its own payload, and ``test_v54_body_hashes_to_the_digest_its_header_asserts``
recomputes that digest over the file as committed -- so the claim in the header
is checked by the suite rather than by whoever wrote it. And
``test_v54_reproduces_the_reference_banks_the_gate_measured`` replays the exact
episode the promotion gate scored and asserts both final banks to the coin. A
blob that decoded but had lost a byte would pass every other test here; the
engine is deterministic given a seed and both seats' actions, so those two
numbers are the only thing that says this file is the agent that was gated.
"""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from kaggle_environments import make

from kaggriculture import (
    boatlee_v14_policy,
    kaito_v22_policy,
    kaito_v23_policy,
    kaito_v54_policy,
)

MODULES = (kaito_v22_policy, kaito_v23_policy, boatlee_v14_policy, kaito_v54_policy)
IDS = ("v22", "v23", "boatlee_v14", "v54")

# The notebook's own checksum over its own payload, quoted in the file's header.
V54_DIGEST = "9f21735aaf0354e064e1d9bab1b2e186fad12cf6446e7ecf83f5014915e04a4f"
V54_BODY_BYTES = 171_508

# Where our header stops and the author's file begins: their module docstring,
# left in place precisely so that this boundary is a line of theirs and not a
# marker of ours that an edit could quietly move.
V54_BODY_START = '"""v51: current-meta capital-flow hybrid.'

# The episode the promotion gate used as its identity reference: v54 in seat
# zero against the agent it replaced, on the first held-out gate seed, played
# through `arena._run_banks`'s configuration.
V54_REFERENCE_SEED = 700_000
V54_REFERENCE_BANKS = (107_604, 68_685)

# The bare module names v54's payload registers in ``sys.modules`` when it is
# imported. Enumerated in its header for the same reason it is enumerated here.
V54_INJECTED_MODULES = frozenset(
    {
        "_v50_frozen_v49",
        "_v51_frozen_v50",
        "scripts",
        "v19_terminal",
        "v23",
        "v24",
        "v44",
        "v48",
        "v49",
        "v50",
    }
)


def first_observation() -> dict:
    """Return turn zero of a real episode, from the engine rather than by hand."""
    environment = make("kaggriculture", debug=False)
    environment.reset(num_agents=2)
    return environment.steps[0][0]["observation"]


@pytest.mark.parametrize("module", MODULES, ids=IDS)
def test_the_agent_emits_a_well_formed_action(module) -> None:
    """A malformed action is discarded by the engine, costing a turn in silence."""
    action = module.agent(first_observation())

    assert set(action) == {"farmer", "hands", "market"}
    assert isinstance(action["farmer"], list)
    assert isinstance(action["hands"], list)
    assert isinstance(action["market"], list)


@pytest.mark.parametrize("module", MODULES, ids=IDS)
def test_a_turn_fits_the_sandbox_budget(module) -> None:
    """One second a turn on two cores, with a 60 second pool for the episode."""
    observation = first_observation()
    module.agent(observation)

    start = time.perf_counter()
    for _ in range(20):
        module.agent(observation)
    elapsed = (time.perf_counter() - start) / 20

    assert elapsed < 0.1, f"{elapsed * 1000:.0f}ms per turn leaves no margin"


def test_the_single_table_routes_decode_to_a_full_season() -> None:
    """A truncated blob would still import and would only show up as bad play."""
    assert len(kaito_v22_policy._ACTIONS) == 719
    assert len(boatlee_v14_policy._ACTIONS) == 719


def test_both_v23_routes_decode_to_a_full_season() -> None:
    """Either table being short would strand the agent on one engine only."""
    assert len(kaito_v23_policy._LEGACY_ACTIONS) == 719
    assert len(kaito_v23_policy._REBALANCE_ACTIONS) == 719


def test_both_v54_routes_decode_to_a_full_season() -> None:
    """It carries its inherited proposal prior and the override that replaces it."""
    assert len(kaito_v54_policy._V51_ROUTE) == 719
    assert len(kaito_v54_policy._V54_ROUTE) == 719


def test_v54_body_hashes_to_the_digest_its_header_asserts() -> None:
    """The header's provenance claim is only worth the bytes it is checked over.

    This is the one vendored kernel kept byte for byte, and the reason is this
    test. Everything from the author's own docstring to the end of the file is
    the notebook's payload as it decoded, and the notebook publishes a SHA-256
    of exactly those bytes. Recomputing it here means "vendored from kernel X"
    is a checked statement rather than a comment: a truncated decode, a
    resolved merge conflict, an editor that stripped a trailing space, or a
    formatter let loose on the file all change this digest, and all of them are
    things that have no business changing the agent we field.
    """
    source = Path(kaito_v54_policy.__file__).read_text(encoding="utf-8")
    body = source[source.index(V54_BODY_START) :].encode("utf-8")

    assert len(body) == V54_BODY_BYTES
    assert hashlib.sha256(body).hexdigest() == V54_DIGEST


def test_v54_import_injects_exactly_the_bare_modules_its_header_names() -> None:
    """Importing it registers whole modules in ``sys.modules`` under bare names.

    Its payload decodes and ``exec``s a bundled ancestor agent, which registers
    itself and its own ancestors under names like ``v49`` and -- the one worth
    watching -- a top-level ``scripts``, while this repository has a
    ``scripts/`` directory at its root. That one is a coincidence of naming and
    not a collision: the injected ``scripts`` is synthetic, built from the
    payload rather than found on disk, so nothing of ours was read to make it
    and nothing of ours is displaced by it.

    Both halves are asserted, because either could stop being true. The names
    are pinned so that the next vendored kernel's injection cannot quietly grow
    to cover a module our agent imports, and the origins are checked so that a
    name reaching a real file in our tree fails here rather than in the sandbox,
    where a wrong module resolves in silence and the only symptom is a zero.

    Run in a subprocess because the import is a one-time side effect: by the
    time this module is collected, this process has already had it done to it.
    """
    script = (
        "import sys, json\n"
        "before = set(sys.modules)\n"
        "import kaggriculture.kaito_v54_policy\n"
        "injected = sorted(set(sys.modules) - before)\n"
        "print(json.dumps({name: getattr(sys.modules[name], '__file__', None)\n"
        "                  for name in injected if '.' not in name}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    origins = json.loads(result.stdout.splitlines()[-1])
    package = Path(kaito_v54_policy.__file__).parent

    missing = V54_INJECTED_MODULES - set(origins)
    assert not missing, f"header names modules it no longer injects: {missing}"
    from_our_tree = {
        name: origin
        for name, origin in origins.items()
        if name in V54_INJECTED_MODULES
        and origin is not None
        and package in Path(origin).resolve().parents
    }
    assert not from_our_tree, f"injected names reached our package: {from_our_tree}"


@pytest.mark.parametrize(
    ("interval", "expected"),
    [(12, "legacy"), (23, "legacy"), (24, "rebalance"), (48, "rebalance")],
)
def test_v23_switches_route_on_the_town_centre_interval(
    interval: int, expected: str
) -> None:
    """1.32.3 reports 12 and 1.32.6 reports 24; the boundary is what it keys on."""
    assert kaito_v23_policy._regime({"townCenterSellInterval": interval}) == expected


def test_v23_falls_back_to_the_legacy_route_without_a_configuration() -> None:
    """Called with one argument it must still play, not raise."""
    assert kaito_v23_policy._regime(None) == "legacy"

    action = kaito_v23_policy.agent(first_observation())

    assert set(action) == {"farmer", "hands", "market"}


def test_v54_reproduces_the_reference_banks_the_gate_measured() -> None:
    """The served agent must be the agent that was gated, to the coin.

    ``kaggriculture.search.arena`` builds every gate episode exactly this way --
    ``episodeSteps`` 720 and a seed, nothing else -- and the engine is
    deterministic given a seed and both seats' actions. So this pair of numbers
    is a fingerprint of the whole 719-turn trajectory. Seed 700000 is the first
    of ``holdout.GATE_SEEDS`` and the opponent is the agent v54 replaced, which
    makes this one of the 128 games behind the 64/64 in ``main.py``'s serve
    comment, replayed here rather than restated.

    Asserted as an exact pair, not a margin or a win: a copy of the payload
    missing a byte either fails to decompress at import or plays a different
    season, and a range would let the second one through. Left out of ``slow``
    on purpose -- the default suite runs with ``-m 'not slow'``, and at 4.5 s
    for the full 719 turns this is cheap enough to run on every commit. It is
    the only test here that would notice the served file being swapped for a
    different agent that also plays well.
    """
    environment = make(
        "kaggriculture",
        configuration={"episodeSteps": 720, "seed": V54_REFERENCE_SEED},
        debug=False,
    )
    environment.run([kaito_v54_policy.agent, boatlee_v14_policy.agent])

    final = environment.steps[-1]

    assert (final[0].status, final[1].status) == ("DONE", "DONE")
    assert (int(final[0].reward), int(final[1].reward)) == V54_REFERENCE_BANKS


@pytest.mark.slow
@pytest.mark.parametrize("module", MODULES, ids=IDS)
def test_it_plays_a_full_episode_and_banks(module) -> None:
    """The whole point: finish 719 turns and bank far above the 3,000 opening."""
    environment = make("kaggriculture", debug=False)
    environment.run([lambda observation, *rest: module.agent(observation), "starter"])

    banked = environment.steps[-1][0]["observation"]["farms"][0]["money"]

    assert banked > 100_000, f"banked only {banked:,.0f}"
