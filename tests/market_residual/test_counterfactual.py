"""Contracts for exact snapshot branching, against the engine that grades us.

Nothing here is checked against a second copy of our own reasoning. The oracle
is the installed reference engine: a branch is right when it finishes the season
at the banks that engine finishes it at, having played the same intervention.
That is the only comparison a bug in the tensor simulator, the pack, the
observation rebuild, the transcript, or the replay alignment cannot survive --
and a differential whose oracle shares our assumptions could survive all five.

Three properties carry the dataset.

*The alignment.* Agent call ``j`` sees the observation whose ``step`` reads
``j``; its action is recorded one index ahead. A snapshot at event turn ``k`` is
``pack(steps[k])`` and the branch plays turns ``k..718``. Shifting that by one
still produces a finished, banked season -- a different one -- which is why the
anchor's banks are compared to the engine's own rather than to a second run of
ours.

*The mind.* The frozen controller's per-episode state lives in closures, so a
branch arm gets a fresh controller replayed through every observation the season
showed it. ``test_a_restored_controller_finishes_the_season_it_left`` is the
check that the replay reconstitutes the agent rather than merely the tape: it
plays the *rest* of the season through the restored copy and demands the
recorded actions.

*The anchor.* Row zero is the controller's own action object, not a merge of its
buckets. On this very season the merge refuses the controller's own queue at
real events; the tests below count them, so the trap is a measurement rather
than a warning in a docstring.
"""
# ruff: noqa: D103

import copy
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pytest
from kaggle_environments import make

from kaggriculture.constants import EPISODE_STEPS
from kaggriculture.features import encode_observation
from kaggriculture.learn.market_residual.alternatives import (
    FAMILY_KAITO,
    Alternative,
)
from kaggriculture.learn.market_residual.counterfactual import (
    LAST_CALL,
    BranchRow,
    CounterfactualIntegrityError,
    CounterfactualSnapshot,
    RecordedEvent,
    RecordedSeason,
    RestoredAgent,
    branch_event,
    branch_rows,
    canonical_action,
    normalized_margin,
    runtime_observation,
    snapshot_event,
    win_points,
)
from kaggriculture.market_residual.actions import (
    ResidualDecision,
    ResidualMode,
    merge_residual_action,
)
from kaggriculture.market_residual.baseline import SERVED, load_verified_baseline
from kaggriculture.market_residual.policy import (
    AlwaysUseKaito,
    build_market_residual_agent,
)
from kaggriculture.search import arena
from tests.market_residual.conftest import BRANCH_SEAT, BRANCH_SEED

# The last stretch of a season, where a branch is cheap. On this seed the merge
# refuses the controller's own queue at turns 701, 708, 712, 713, 715 and 718,
# so the anchor test below has real cases to find here.
REFUSED_ANCHOR_TAIL = 690


@pytest.fixture(scope="session")
def late_rows(late_snapshot: CounterfactualSnapshot) -> tuple[BranchRow, ...]:
    """Return the playable arms of the late branch, anchor first."""
    return branch_rows(late_snapshot)


@pytest.fixture(scope="session")
def late_outcomes(
    late_snapshot: CounterfactualSnapshot, late_rows: tuple[BranchRow, ...]
) -> tuple[Any, ...]:
    """Play the late branch once and share its outcomes across the tests."""
    return branch_event(late_snapshot, late_rows)


def test_the_scoring_rule_is_the_ladders_own() -> None:
    banks = ((10, 5), (5, 10), (7, 7), (0, 0), (-3, 4))
    assert [win_points(ours, theirs) for ours, theirs in banks] == [
        arena._win(ours, theirs) for ours, theirs in banks
    ]
    assert [normalized_margin(ours, theirs) for ours, theirs in banks] == [
        arena._normalized_margin(ours, theirs) for ours, theirs in banks
    ]


def test_the_recorded_season_is_a_clean_reference_game(
    recorded_season: RecordedSeason,
) -> None:
    assert len(recorded_season.steps) == EPISODE_STEPS
    assert len(recorded_season.learner.observations) == LAST_CALL + 1
    assert len(recorded_season.opponent.actions) == LAST_CALL + 1
    for call, observation in enumerate(recorded_season.learner.observations):
        assert int(observation["step"]) == call


def test_the_events_are_the_wrapped_controllers_own(
    recorded_events: tuple[RecordedEvent, ...],
) -> None:
    """The collector's clock must be the one the submitted runtime keeps."""
    played = build_market_residual_agent(SERVED, AlwaysUseKaito())
    environment = make(
        "kaggriculture",
        configuration={"episodeSteps": EPISODE_STEPS, "seed": BRANCH_SEED},
        debug=False,
    )
    sides: list[Any] = [None, None]
    sides[BRANCH_SEAT] = played
    sides[1 - BRANCH_SEAT] = load_verified_baseline(SERVED)
    environment.run(sides)

    assert played.record.turns == LAST_CALL + 1
    assert played.record.events == len(recorded_events)
    assert [event.index for event in recorded_events] == list(
        range(len(recorded_events))
    )


def test_a_snapshot_rebuilds_the_observation_the_engine_handed_the_seat(
    recorded_season: RecordedSeason, late_snapshot: CounterfactualSnapshot
) -> None:
    """A packed instant must be the instant, not a state that resembles it."""
    recorded = dict(recorded_season.learner.observations[late_snapshot.event.turn])
    recorded.pop("remainingOverageTime")

    assert int(late_snapshot.state.step[0]) == late_snapshot.event.turn
    assert runtime_observation(late_snapshot.state, 0, BRANCH_SEAT) == recorded


def test_a_restored_controller_finishes_the_season_it_left(
    recorded_season: RecordedSeason, late_snapshot: CounterfactualSnapshot
) -> None:
    """Restoring must reconstitute the agent, not merely replay its tape."""
    turn = late_snapshot.event.turn
    restored = late_snapshot.learner_transcript.restore()

    assert restored.replayed_actions == recorded_season.learner.actions[: turn + 1]
    remaining = [
        canonical_action(restored.play(observation))
        for observation in recorded_season.learner.observations[turn + 1 :]
    ]
    assert remaining == list(recorded_season.learner.actions[turn + 1 :])
    assert remaining, "a restore that has nothing left to play proves nothing"


def test_a_transcript_that_did_not_produce_those_actions_is_refused(
    late_snapshot: CounterfactualSnapshot,
) -> None:
    transcript = late_snapshot.learner_transcript
    tampered = replace(
        transcript,
        actions=(
            *transcript.actions[:-1],
            canonical_action({"farmer": ["PASS"], "hands": [], "market": []}),
        ),
    )

    with pytest.raises(CounterfactualIntegrityError, match="diverged at call"):
        tampered.restore()


def test_a_transcript_replayed_without_its_configuration_is_refused(
    late_snapshot: CounterfactualSnapshot,
) -> None:
    """The engine passes a configuration every turn, and this controller reads it."""
    blind = replace(late_snapshot.learner_transcript, configuration={})

    with pytest.raises(CounterfactualIntegrityError, match="diverged at call"):
        blind.restore()


def test_a_restored_controller_plays_its_forward_turns_under_that_configuration() -> (
    None
):
    """The turns *after* a branch are called the way the recorded season was.

    ``restore`` proving the prefix is not enough. The prefix replay and the
    forward branch are two different call sites, and a forward branch that drops
    the configuration reproduces every shared branch fixture here -- they open at
    turn 700 and only nineteen turns follow. Collection opens events across the
    whole season, so the site is pinned directly rather than through an outcome
    that a short tail can hide.
    """
    seen: list[object] = []

    def spy(
        observation: Mapping[str, Any], configuration: object | None = None
    ) -> dict[str, Any]:
        seen.append(configuration)
        return {"farmer": ["PASS"], "hands": [], "market": []}

    configuration = {"episodeSteps": EPISODE_STEPS, "townCenterSellInterval": 24}
    RestoredAgent(spy, configuration, ()).play({"step": 3})

    assert seen == [configuration]


def test_row_zero_is_the_controllers_own_action_object(
    late_snapshot: CounterfactualSnapshot, late_rows: tuple[BranchRow, ...]
) -> None:
    assert late_rows[0].family == FAMILY_KAITO
    assert late_rows[0].action is late_snapshot.action
    assert late_rows[0].buckets == late_snapshot.alternatives.kaito
    assert all(row.family != FAMILY_KAITO for row in late_rows[1:])
    assert len(late_rows) == 1 + len(late_snapshot.alternatives.replacements)
    assert len(late_rows) > 1, "a branch with no replacement tests nothing"


def test_the_merge_refuses_the_controllers_own_queue_on_real_events(
    recorded_season: RecordedSeason, recorded_events: tuple[RecordedEvent, ...]
) -> None:
    """The anchor must bypass the merge, and this is why, measured here.

    If row zero were built by merging the controller's own buckets, every event
    counted below would record the controller's own action as illegal.
    """
    refused = 0
    for recorded in recorded_events:
        turn = recorded.event.turn
        observation = recorded_season.learner.observations[turn]
        encoded = encode_observation(observation, BRANCH_SEAT)
        action = json.loads(recorded_season.learner.actions[turn])
        merged = merge_residual_action(
            encoded,
            action,
            ResidualDecision(ResidualMode.REPLACE, recorded.event.kaito_buckets),
        )
        refused += not merged.replaced

    assert (len(recorded_events), refused) == (505, 82), (
        "the event or merge boundary moved; re-measure before trusting any "
        "counterfactual label produced under the new one"
    )


def test_the_anchor_still_branches_where_the_merge_refuses_the_controller(
    recorded_season: RecordedSeason, recorded_events: tuple[RecordedEvent, ...]
) -> None:
    """The one event class a merged row zero would poison must branch anyway."""
    for recorded in reversed(recorded_events):
        if recorded.event.turn < REFUSED_ANCHOR_TAIL:
            continue
        turn = recorded.event.turn
        encoded = encode_observation(
            recorded_season.learner.observations[turn], BRANCH_SEAT
        )
        action = json.loads(recorded_season.learner.actions[turn])
        if merge_residual_action(
            encoded,
            action,
            ResidualDecision(ResidualMode.REPLACE, recorded.event.kaito_buckets),
        ).replaced:
            continue
        snapshot = snapshot_event(recorded_season, recorded)
        rows = branch_rows(snapshot)
        outcomes = branch_event(snapshot, rows)

        assert rows[0].action is snapshot.action
        assert outcomes[0].banks == recorded_season.terminal_banks
        return
    raise AssertionError(
        f"no event past turn {REFUSED_ANCHOR_TAIL} refused the controller's own "
        "queue, so this test no longer exercises the case it was written for"
    )


def test_the_anchor_branch_is_the_season_the_reference_engine_played(
    recorded_season: RecordedSeason,
    late_snapshot: CounterfactualSnapshot,
    late_outcomes: tuple[Any, ...],
) -> None:
    """The replay alignment, the pack, and the rebuild, all in one number."""
    anchor = late_outcomes[0]

    assert anchor.banks == recorded_season.terminal_banks
    assert anchor.banks == late_snapshot.control_banks
    assert anchor.win_points == win_points(*anchor.banks)
    assert anchor.margin == normalized_margin(*anchor.banks)


def test_a_single_intervention_matches_the_reference_engine(
    late_snapshot: CounterfactualSnapshot,
    late_rows: tuple[BranchRow, ...],
    late_outcomes: tuple[Any, ...],
) -> None:
    """One replaced market queue, played by the engine that will grade us."""
    changed = [
        index
        for index in range(1, len(late_rows))
        if late_outcomes[index].banks != late_outcomes[0].banks
    ]
    assert changed, "no alternative moved the banks, so nothing was compared"
    index = changed[0]

    assert reference_intervention(late_snapshot, late_rows[index]) == (
        late_outcomes[index].banks
    )


def reference_intervention(
    snapshot: CounterfactualSnapshot, row: BranchRow
) -> tuple[int, int]:
    """Play the whole season in the reference engine with one action replaced.

    Args:
        snapshot: The instant the intervention happens at.
        row: The arm whose action replaces the controller's on that turn.

    Returns:
        The learner's and opponent's terminal banks.
    """
    controller = load_verified_baseline(SERVED)
    replacement = copy.deepcopy(dict(row.action))

    def intervened(
        observation: Mapping[str, Any],
        configuration: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        played = controller(observation, configuration)
        if int(observation["step"]) == snapshot.event.turn:
            return replacement
        return played

    sides: list[Any] = [None, None]
    sides[snapshot.identity.seat] = intervened
    sides[1 - snapshot.identity.seat] = load_verified_baseline(SERVED)
    environment = make(
        "kaggriculture",
        configuration={"episodeSteps": EPISODE_STEPS, "seed": snapshot.identity.seed},
        debug=False,
    )
    environment.run(sides)
    final = environment.steps[-1]
    banks = [
        int(final[seat]["observation"]["farms"][seat]["money"]) for seat in range(2)
    ]
    return (banks[snapshot.identity.seat], banks[1 - snapshot.identity.seat])


def test_a_branch_must_open_on_the_anchor(
    late_snapshot: CounterfactualSnapshot, late_rows: tuple[BranchRow, ...]
) -> None:
    with pytest.raises(CounterfactualIntegrityError, match="open on the anchor"):
        branch_event(late_snapshot, late_rows[1:])

    doubled = (*late_rows, BranchRow(FAMILY_KAITO, (), late_snapshot.action))
    with pytest.raises(CounterfactualIntegrityError, match="only row zero"):
        branch_event(late_snapshot, doubled)


def test_a_replacement_cannot_be_labelled_as_the_anchor() -> None:
    with pytest.raises(ValueError, match="is the anchor, not a replacement"):
        Alternative(FAMILY_KAITO, (0,))


def test_alternatives_from_another_turn_are_refused(
    recorded_season: RecordedSeason,
    recorded_events: tuple[RecordedEvent, ...],
    late_snapshot: CounterfactualSnapshot,
) -> None:
    other = snapshot_event(recorded_season, recorded_events[0])
    mismatched = replace(late_snapshot, alternatives=other.alternatives)

    with pytest.raises(CounterfactualIntegrityError, match="the event opened on"):
        branch_rows(mismatched)


def test_a_snapshot_reports_the_turn_it_was_taken_at(
    recorded_season: RecordedSeason, recorded_events: tuple[RecordedEvent, ...]
) -> None:
    for recorded in recorded_events[:4]:
        snapshot = snapshot_event(recorded_season, recorded)
        assert snapshot.event is recorded.event
        assert int(snapshot.state.step[0]) == recorded.event.turn
        assert len(snapshot.learner_transcript.observations) == recorded.event.turn + 1
        assert (
            snapshot.learner_transcript.actions[-1]
            == recorded_season.learner.actions[recorded.event.turn]
        )


DETERMINISM_PROBE = """
import json, sys
from kaggriculture.learn.market_residual.alternatives import AlternativeConfig
from kaggriculture.learn.market_residual.artifacts import (
    CounterfactualShard,
    ShardIdentity,
    counterfactual_rows,
    shard_bytes,
)
from kaggriculture.learn.market_residual.counterfactual import (
    SeasonIdentity,
    record_season,
    season_events,
    snapshot_event,
)
from kaggriculture.market_residual.baseline import SERVED
from kaggriculture.market_residual.events import EventConfig

seed, seat, turn = (int(value) for value in sys.argv[1:4])
identity = SeasonIdentity.current(
    learner=SERVED,
    opponent=SERVED,
    seed_bank="counterfactual_train",
    seed=seed,
    seat=seat,
    event_config=EventConfig(),
    alternative_config=AlternativeConfig(
        max_single=2, max_ranked_multi=1, max_alternatives=8
    ),
)
season = record_season(identity)
event = next(item for item in season_events(season) if item.event.turn >= turn)
shard = CounterfactualShard(
    identity=ShardIdentity.of(identity),
    rows=counterfactual_rows(snapshot_event(season, event)),
)
sys.stdout.buffer.write(shard_bytes(shard))
"""


@pytest.mark.slow
def test_a_branch_is_byte_identical_across_processes(tmp_path: Path) -> None:
    """Two interpreters, two hash seeds, one shard: the determinism standard."""
    script = tmp_path / "probe.py"
    script.write_text(DETERMINISM_PROBE)
    command = [sys.executable, str(script), str(BRANCH_SEED), str(BRANCH_SEAT), "700"]

    first = probe(command, "0")
    second = probe(command, "12345")

    assert first, "the probe published nothing to compare"
    assert first == second


def probe(command: list[str], hash_seed: str) -> bytes:
    """Return the probe's published shard bytes under one interpreter hash seed.

    Args:
        command: The interpreter, script, and cell to run.
        hash_seed: The value of ``PYTHONHASHSEED`` for that run.

    Returns:
        Exactly what the probe wrote to standard output.
    """
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        env={"PYTHONHASHSEED": hash_seed, "PATH": "/usr/bin:/bin"},
    ).stdout
