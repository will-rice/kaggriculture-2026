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

v54 and v56 get two the others cannot have, because they are the ones kept
byte-verbatim -- v56 is what we serve, v54 is what it replaced and now a gate
opponent. Each header asserts the SHA-256 the notebook publishes over its own
payload, and ``test_v5X_body_hashes_to_the_digest_its_header_asserts``
recomputes that digest over the file as committed -- so the claim in the header
is checked by the suite rather than by whoever wrote it. And
``test_v5X_reproduces_the_reference_banks_the_gate_measured`` replays the exact
episode the promotion gate scored and asserts both final banks to the coin. A
blob that decoded but had lost a byte would pass every other test here; the
engine is deterministic given a seed and both seats' actions, so those two
numbers are the only thing that says this file is the agent that was gated.

v56 gets a third, and it is the only test here about *which function* an agent
is. That file defines ``agent`` twice and ``_kaggle_submission_entrypoint``
twice; the author's answer is a third function, ``kaggle_agent_v56``, left last
in the file because that is what ``get_last_callable`` selects. The two
survivors agree today, so importing the wrong one is currently free -- and the
shadowed ``agent``, which is the v51 backbone without v56's branch or its seed
redaction, plays a visibly different season while raising nothing.
``test_v56_is_played_through_the_final_callable_and_not_a_shadowed_agent``
refuses to let either fact go unchecked.
"""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from types import FunctionType
from typing import Any, cast

import pytest
from kaggle_environments import make
from kaggle_environments.agent import get_last_callable

from kaggriculture import (
    boatlee_v14_policy,
    kaito_v22_policy,
    kaito_v23_policy,
    kaito_v54_policy,
    kaito_v56_policy,
)

# Named one entry point at a time rather than reached for as ``module.agent``.
# v56 is the reason: ``kaito_v56_policy.agent`` exists, is callable, and is not
# what anything serves, so a rule that says "the agent is the attribute called
# agent" would pass this suite while testing a function the competition never
# runs.
AGENTS = (
    kaito_v22_policy.agent,
    kaito_v23_policy.agent,
    boatlee_v14_policy.agent,
    kaito_v54_policy.agent,
    kaito_v56_policy.kaggle_agent_v56,
)
IDS = ("v22", "v23", "boatlee_v14", "v54", "v56")

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

# v56's counterparts. The marker is deliberately not restated: both kernels
# bundle the same v51 ancestor and inherit its module docstring, so the two
# files open their verified regions with the *same* line and only the digest
# tells them apart. Writing it twice would read as though it were evidence.
V56_DIGEST = "e7f0502537ea1a79f4ec971235627930fa4ef233156547da942369679141f10c"
V56_BODY_BYTES = 199_613
V56_BODY_START = V54_BODY_START

# The author's own final callable, named here because this file contains two
# other functions called ``agent`` and one of them is not v56 at all.
V56_ENTRYPOINT = "kaggle_agent_v56"

# A season whose first revealed shop is one v56 branches on, which is what makes
# the shadowed backbone tell itself apart. Chosen by scanning small seeds, and
# small on purpose: it is outside `market_residual.schema.SEED_BANKS` and
# outside `holdout.GATE_SEEDS`, because an illustration may not spend a seed a
# promotion decision will need. 200 turns is well past the step-72 branch and
# well short of a full season.
V56_BRANCH_SEED = 7
V56_BRANCH_TURNS = 200
V56_BRANCH_STEP = 72
V56_BRANCH_SHOPS = frozenset({"YARN_STORE", "PET_CAFE"})

# The episode the promotion gate used as its identity reference: v56 in seat
# zero against the agent it replaced, on the first held-out gate seed, played
# through `arena._run_banks`'s configuration. One of the 128 games behind the
# 116/128 in `main.py`'s serve comment -- replayed, not restated.
V56_REFERENCE_SEED = 700_000
V56_REFERENCE_BANKS = (68_998, 68_451)

# Two fewer than v54 injects: v56 does not register `_v50_frozen_v49` or
# `_v51_frozen_v50`. That difference is the reason both sets are written out
# rather than shared -- "the same modules as last time" was never true.
V56_INJECTED_MODULES = frozenset(
    {
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


def body_digest(source: str, marker: str) -> str:
    """Return the SHA-256 of ``source`` from ``marker`` to its end.

    Args:
        source: A vendored policy file, header and all.
        marker: The first line of the author's own file.

    Returns:
        The hex digest of exactly the region the notebook publishes a checksum
        over -- everything from ``marker``, and nothing above it.
    """
    return hashlib.sha256(source[source.index(marker) :].encode("utf-8")).hexdigest()


def fresh_copy(source: str) -> dict[str, Any]:
    """Execute a vendored policy's source into a namespace of its own.

    These controllers keep per-episode state in module-level closures, so a
    comparison between two of them has to be between two copies rather than
    between two names in one namespace.

    Args:
        source: The file's text, header and all.

    Returns:
        The namespace it left behind.
    """
    namespace: dict[str, Any] = {"__name__": "copy", "__file__": "<copy>"}
    exec(compile(source, "<copy>", "exec"), namespace)  # noqa: S102
    return namespace


def first_observation() -> dict:
    """Return turn zero of a real episode, from the engine rather than by hand."""
    environment = make("kaggriculture", debug=False)
    environment.reset(num_agents=2)
    return environment.steps[0][0]["observation"]


@pytest.mark.parametrize("agent", AGENTS, ids=IDS)
def test_the_agent_emits_a_well_formed_action(agent) -> None:
    """A malformed action is discarded by the engine, costing a turn in silence."""
    action = agent(first_observation())

    assert set(action) == {"farmer", "hands", "market"}
    assert isinstance(action["farmer"], list)
    assert isinstance(action["hands"], list)
    assert isinstance(action["market"], list)


@pytest.mark.parametrize("agent", AGENTS, ids=IDS)
def test_a_turn_fits_the_sandbox_budget(agent) -> None:
    """One second a turn on two cores, with a 60 second pool for the episode."""
    observation = first_observation()
    agent(observation)

    start = time.perf_counter()
    for _ in range(20):
        agent(observation)
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

    assert len(source[source.index(V54_BODY_START) :].encode("utf-8")) == V54_BODY_BYTES
    assert body_digest(source, V54_BODY_START) == V54_DIGEST


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


def test_every_v56_route_decodes_to_a_full_season() -> None:
    """Five tables, not one, and the file is useless if any of them is short.

    v56 carries the v51 prior it inherited, v54's override, v55's live route and
    the two plays it chooses between on the first revealed shop. Only the last
    two are ever executed, but a truncated table anywhere means a truncated
    payload, and a truncated payload is the failure this whole file exists to
    catch. Asserted by name so that a table disappearing is as loud as a table
    shrinking.
    """
    routes = {
        "_V51_ROUTE": kaito_v56_policy._V51_ROUTE,
        "_V54_ROUTE": kaito_v56_policy._V54_ROUTE,
        "_V55_LIVE_ROUTE": kaito_v56_policy._V55_LIVE_ROUTE,
        "_V56_BACKBONE_ROUTE": kaito_v56_policy._V56_BACKBONE_ROUTE,
        "_V56_YARN_ROUTE": kaito_v56_policy._V56_YARN_ROUTE,
    }

    assert {name: len(route) for name, route in routes.items()} == dict.fromkeys(
        routes, 719
    )


def test_v56_body_hashes_to_the_digest_its_header_asserts() -> None:
    """The header's provenance claim is only worth the bytes it is checked over.

    The same contract ``test_v54_body_hashes_to_the_digest_its_header_asserts``
    states, for the kernel we now serve: everything from the author's own
    docstring to the end of the file is the notebook's payload as it decoded,
    the notebook publishes a SHA-256 of exactly those bytes, and recomputing it
    here makes "vendored from kernel X" a checked statement rather than a
    comment.
    """
    source = Path(kaito_v56_policy.__file__).read_text(encoding="utf-8")

    assert len(source[source.index(V56_BODY_START) :].encode("utf-8")) == V56_BODY_BYTES
    assert body_digest(source, V56_BODY_START) == V56_DIGEST


def test_the_hashed_region_is_the_payload_and_stops_at_our_header() -> None:
    """The guard above has never been watched fail; this is where it fails.

    A digest test that only ever sees the correct file cannot distinguish "the
    payload is intact" from "the region is computed wrongly and happens to
    match". Two ways it could be computed wrongly, both silent: hashing from
    byte zero, which would fold our own header into the provenance claim and
    make every edit to it look like payload drift; and hashing a prefix, which
    would leave the tail of the payload unchecked -- the tail being exactly
    where the entry point lives.

    So this drives the same computation over three mutations of the real file
    and demands the answer move only when the payload does: an insertion above
    the marker must not change the digest, and an edit at either end of the
    payload must.
    """
    source = Path(kaito_v56_policy.__file__).read_text(encoding="utf-8")
    start = source.index(V56_BODY_START)
    payload = source[start:]

    above_the_marker = "# a line of ours, in our own header\n" + source
    at_the_end = source + "# drift\n"
    at_the_start = source[:start] + payload.replace("719-step", "719 step", 1)

    assert body_digest(above_the_marker, V56_BODY_START) == V56_DIGEST
    assert body_digest(at_the_end, V56_BODY_START) != V56_DIGEST
    assert body_digest(at_the_start, V56_BODY_START) != V56_DIGEST


def test_v56_is_played_through_the_final_callable_and_not_a_shadowed_agent() -> None:
    """Three callables could plausibly be "the agent"; only one is served.

    ``agent`` is defined twice in this file. Rebinding a name leaves it where it
    was in the namespace, so neither definition is last, and the author's answer
    was to add ``kaggle_agent_v56`` below both and document that it must stay
    there -- which is what ``get_last_callable`` picks, and therefore what the
    competition runs and what the 128-game gate played.

    Two of the three agree: ``kaggle_agent_v56`` delegates to the surviving
    ``agent``, so serving either scores the same today. The third is the bundled
    v51 backbone, and the interesting thing about it is how nearly it also
    agrees. Its route differs from v56's in 690 of 719 entries and that changes
    *nothing* on most boards -- the route is a proposal prior the residual
    controller talks it out of -- so on the reference seed above it banks
    68,998 to the coin, exactly what v56 banks. What separates them is the shop
    branch, and only on a season where the branch fires.

    That is the whole argument for pinning a name. A behavioural check is not
    available to us in general: the wrong resolution is invisible on most seeds
    and worth 117 turns of different play on the right one. So this asserts the
    structure -- which function the loader picks, and where it sits relative to
    both definitions of ``agent`` -- and then demonstrates on a branching season
    that the shadow is not a synonym, so that the structural half is known to be
    guarding something.
    """
    path = Path(kaito_v56_policy.__file__)
    source = path.read_text(encoding="utf-8")
    resolved = cast(FunctionType, get_last_callable(source, path=str(path)))
    definitions = [
        line
        for line, text in enumerate(source.splitlines(), 1)
        if text.startswith("def agent(")
    ]
    environment = make(
        "kaggriculture",
        configuration={"episodeSteps": V56_BRANCH_TURNS, "seed": V56_BRANCH_SEED},
        debug=False,
    )
    environment.run([fresh_copy(source)[V56_ENTRYPOINT], "starter"])
    observations = [step[0]["observation"] for step in environment.steps]
    served, shadow = fresh_copy(source), fresh_copy(source)
    entry, backbone = served[V56_ENTRYPOINT], shadow["_V51_POLICY"]
    divergences = [
        turn
        for turn, observation in enumerate(observations)
        if entry(observation) != backbone(observation, None)
    ]
    revealed = observations[-1]["town"]["unlocked_shops"]

    assert len(definitions) == 2, "the shadowing this test is about is gone"
    assert resolved.__name__ == V56_ENTRYPOINT
    assert resolved.__code__.co_firstlineno > definitions[-1]
    assert kaito_v56_policy.kaggle_agent_v56.__code__.co_firstlineno == (
        resolved.__code__.co_firstlineno
    )
    assert revealed[0] in V56_BRANCH_SHOPS, "this season did not reach the branch"
    assert divergences, "the shadow played the same season; the seed stopped branching"
    assert min(divergences) >= V56_BRANCH_STEP


def test_v56_import_injects_exactly_the_bare_modules_its_header_names() -> None:
    """The names its payload registers, checked against the header that lists them.

    Same contract as the v54 test above and same subprocess, for the same two
    reasons: the injection can grow to cover a module our agent imports, and a
    bare name reaching a real file in our tree resolves in silence.

    Worth running a second time rather than trusting the v54 result, because the
    sets are not the same -- v56 drops ``_v50_frozen_v49`` and
    ``_v51_frozen_v50``. A vendoring that had copied v54's list forward would
    have passed the "does it inject what we named" half unchanged.
    """
    script = (
        "import sys, json\n"
        "before = set(sys.modules)\n"
        "import kaggriculture.kaito_v56_policy\n"
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
    package = Path(kaito_v56_policy.__file__).parent

    missing = V56_INJECTED_MODULES - set(origins)
    assert not missing, f"header names modules it no longer injects: {missing}"
    from_our_tree = {
        name: origin
        for name, origin in origins.items()
        if name in V56_INJECTED_MODULES
        and origin is not None
        and package in Path(origin).resolve().parents
    }
    assert not from_our_tree, f"injected names reached our package: {from_our_tree}"


def test_v56_reproduces_the_reference_banks_the_gate_measured() -> None:
    """The served agent must be the agent that was gated, to the coin.

    The same fingerprint ``test_v54_reproduces_the_reference_banks_the_gate_measured``
    takes, moved onto the kernel we now serve and its own opponent: v56 in seat
    zero against the v54 it replaced, on the first of ``holdout.GATE_SEEDS``,
    built the way ``kaggriculture.search.arena`` builds every gate episode. The
    engine is deterministic given a seed and both seats' actions, so this pair
    of numbers is a fingerprint of the whole 719-turn trajectory, and this is
    one of the 128 games behind the 116/128 in ``main.py``'s serve comment.

    Played through ``kaggle_agent_v56`` rather than through ``agent``, because
    that is what the gate played and what ``main.py`` imports. The margin is 547
    coins on this seed, which is worth knowing: v56 wins 116 of 128 games, not
    every game by a mile, and a test that asserted a comfortable win would be
    asserting something that is not true.
    """
    environment = make(
        "kaggriculture",
        configuration={"episodeSteps": 720, "seed": V56_REFERENCE_SEED},
        debug=False,
    )
    environment.run([kaito_v56_policy.kaggle_agent_v56, kaito_v54_policy.agent])

    final = environment.steps[-1]

    assert (final[0].status, final[1].status) == ("DONE", "DONE")
    assert (int(final[0].reward), int(final[1].reward)) == V56_REFERENCE_BANKS


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
@pytest.mark.parametrize("agent", AGENTS, ids=IDS)
def test_it_plays_a_full_episode_and_banks(agent) -> None:
    """The whole point: finish 719 turns and bank far above the 3,000 opening."""
    environment = make("kaggriculture", debug=False)
    environment.run([lambda observation, *rest: agent(observation), "starter"])

    banked = environment.steps[-1][0]["observation"]["farms"][0]["money"]

    assert banked > 100_000, f"banked only {banked:,.0f}"
