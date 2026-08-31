"""Contracts for the state machine that decides when the residual is consulted.

The measurements pinned here are the ones a later threshold change would move
silently: how often a real season opens an event at all, which triggers carry
that rate, and which triggers never fire against the served controller. A
threshold set that fires on every turn is not a bug the type checker can see --
it is a residual promoted to a second controller, which is what the merge and
this machine exist to prevent.
"""
# ruff: noqa: D103

import collections
from typing import Any

from kaggriculture.market_residual.actions import kaito_market_buckets
from kaggriculture.market_residual.events import (
    EventConfig,
    EventMemory,
    EventTransition,
    detect_event,
)
from tests.market_residual.conftest import Turn

CONFIG = EventConfig()


def walk(turns: tuple[Turn, ...], seat: int) -> list[tuple[Turn, EventTransition]]:
    """Replay one seat's episode through the machine, advancing one turn a turn."""
    memory = EventMemory.initial()
    walked: list[tuple[Turn, EventTransition]] = []
    for turn in turns:
        if turn.seat != seat:
            continue
        memory = memory.advance(1) if walked else memory
        transition = detect_event(turn.encoded, turn.action, memory, CONFIG)
        memory = transition.memory
        walked.append((turn, transition))
    return walked


def test_heartbeat_opens_once_and_deduplicates_identical_state(
    kaito_turns: tuple[Turn, ...],
) -> None:
    turn = kaito_turns[600]

    first = detect_event(turn.encoded, turn.action, EventMemory.initial(), CONFIG)
    heartbeat = detect_event(
        turn.encoded, turn.action, first.memory.advance(CONFIG.heartbeat_turns), CONFIG
    )
    duplicate = detect_event(turn.encoded, turn.action, heartbeat.memory, CONFIG)

    assert first.event is not None
    assert first.event.triggers == ("first",)
    assert heartbeat.event is not None
    assert "heartbeat" in heartbeat.event.triggers
    assert heartbeat.event.triggers == ("heartbeat",)
    assert duplicate.event is None
    assert duplicate.memory is heartbeat.memory


def test_an_unchanged_board_opens_no_second_event(
    kaito_turns: tuple[Turn, ...],
) -> None:
    turn = next(
        candidate
        for candidate in kaito_turns
        if any(kaito_market_buckets(candidate.action) or (0,))
    )
    memory = detect_event(
        turn.encoded, turn.action, EventMemory.initial(), CONFIG
    ).memory

    for _repeat in range(CONFIG.heartbeat_turns - 1):
        memory = memory.advance(1)
        transition = detect_event(turn.encoded, turn.action, memory, CONFIG)
        assert transition.event is None
        memory = transition.memory


def test_a_proposal_that_cannot_be_read_is_never_an_event(
    kaito_turns: tuple[Turn, ...],
) -> None:
    refused = [
        turn for turn in kaito_turns if kaito_market_buckets(turn.action) is None
    ]
    assert len(refused) == 38
    memory = EventMemory.initial()
    for turn in refused:
        transition = detect_event(turn.encoded, turn.action, memory, CONFIG)
        assert transition.event is None
        assert transition.memory is memory


def test_a_real_season_opens_the_measured_share_of_its_turns(
    kaito_turns: tuple[Turn, ...],
) -> None:
    opened: dict[bool, int] = collections.Counter()
    triggers: dict[str, int] = collections.Counter()
    for seat in (0, 1):
        for _turn, transition in walk(kaito_turns, seat):
            opened[transition.event is not None] += 1
            if transition.event is not None:
                triggers.update(transition.event.triggers)

    assert dict(opened) == {True: 982, False: 458}
    assert dict(triggers) == {
        "opponent_supply": 700,
        "kaito": 458,
        "price": 206,
        "slot": 82,
        "inventory": 76,
        "shop": 12,
        "first": 2,
        "liquidation": 2,
    }
    assert "heartbeat" not in triggers, (
        "the served controller moves the board often enough that the heartbeat "
        "never expires; a change that revives it is a change in event density"
    )


def test_liquidation_is_a_one_shot_switch(kaito_turns: tuple[Turn, ...]) -> None:
    for seat in (0, 1):
        walked = walk(kaito_turns, seat)
        firing = [
            turn
            for turn, transition in walked
            if transition.event is not None
            and "liquidation" in transition.event.triggers
        ]
        assert len(firing) == 1
        assert firing[0].encoded.day_count() == CONFIG.liquidation_day
        assert all(
            transition.memory.liquidated
            for turn, transition in walked
            if turn.turn >= firing[0].turn
        )


def test_thresholds_are_measured_against_the_last_event(
    kaito_turns: tuple[Turn, ...],
) -> None:
    references: list[tuple[Any, ...]] = []
    for _turn, transition in walk(kaito_turns, 0):
        if transition.event is None:
            assert transition.memory.prices == references[-1]
            continue
        references.append(transition.memory.prices)
        assert transition.memory.last_event_turn == transition.event.turn
    assert len(references) > 1


def test_advancing_the_clock_changes_nothing_else() -> None:
    memory = EventMemory.initial()

    later = memory.advance(7)

    assert later.turn == 7
    assert later == EventMemory(
        turn=7,
        last_event_turn=memory.last_event_turn,
        fingerprint=memory.fingerprint,
        prices=memory.prices,
        inventories=memory.inventories,
        supply=memory.supply,
        shops=memory.shops,
        legal=memory.legal,
        liquidated=memory.liquidated,
    )
