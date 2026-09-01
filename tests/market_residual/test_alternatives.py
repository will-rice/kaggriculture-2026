"""Contracts for the bounded counterfactual alternative set.

Three properties are load-bearing, and each is checked the only way it can
actually fail.

*Deterministic.* Collection writes an alternative index into a shard and the
learner reads it back weeks later. If the two disagree about which action row
three was, every label in the shard is attached to an action nobody played, and
nothing downstream raises. Two calls in one process cannot catch that -- a set
built in the same order twice iterates the same way twice -- so the digest is
compared across two interpreters started under different ``PYTHONHASHSEED``
values, which is exactly what reorders a set of anything string-keyed.

*Bounded.* The cap is a budget declared per family, so the families are checked
against their own caps on real events under a deliberately tight config. Under
the shipped config a real season peaks at 21 rows against a cap of 48; a test
that only asserted the cap would pass with any amount of overflow short of
doubling, which is why the tight config is the one that does the work.

*Legal.* Legality is not restated here. Every non-Kaito row is merged and then
replayed through the engine's own ``_parse_order``/``_commit_unit`` by the
crown-test machinery in ``test_actions``: an alternative the engine only partly
fills is an alternative whose recorded label describes an action nobody played.

Every state under test is a real turn of a real season played by the served
controller, and the events are the ones the real event machine opened on it.
"""
# ruff: noqa: D103

import collections
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from kaggriculture.action_codec import MARKET_SLOTS, quantity_of
from kaggriculture.learn.market_residual.alternatives import (
    ALTERNATIVE_FAMILIES,
    FAMILY_KAITO,
    Alternative,
    AlternativeConfig,
    AlternativeSet,
    generate_alternatives,
)
from kaggriculture.market_residual.actions import (
    ResidualDecision,
    ResidualMode,
    merge_residual_action,
)
from kaggriculture.market_residual.events import (
    EventConfig,
    EventMemory,
    MarketEvent,
    detect_event,
)
from kaggriculture.market_residual.schema import ALLOWED_SLOTS, BUCKETS
from tests.market_residual.conftest import Turn
from tests.market_residual.test_actions import engine_fills, ordered_units

EVENT_CONFIG = EventConfig()
CONFIG = AlternativeConfig()

# A budget small enough that a real season overruns every family, so an
# off-by-one in a selection shows up as a count rather than as slack.
TIGHT = AlternativeConfig(max_single=2, max_ranked_multi=1, max_alternatives=8)

# Every fourth opened event. The engine replay below commits each alternative
# unit by unit, and a fixed stride keeps the cells under test from moving.
ENGINE_STRIDE = 4


@dataclass(frozen=True)
class Cell:
    """One opened market event and the real turn it was opened on."""

    turn: Turn
    event: MarketEvent


def alternatives_for(cell: Cell, config: AlternativeConfig) -> AlternativeSet:
    """Return the alternative set for one real event under one budget."""
    return generate_alternatives(
        cell.event, cell.turn.encoded, cell.turn.action, config
    )


def families_of(alternatives: AlternativeSet) -> collections.Counter[str]:
    """Count the families of one recorded set, anchor included as row zero."""
    counts = collections.Counter(row.family for row in alternatives.replacements)
    counts[FAMILY_KAITO] += 1
    return counts


@pytest.fixture(scope="session")
def market_cells(kaito_turns: tuple[Turn, ...]) -> tuple[Cell, ...]:
    """Return every event the real machine opened on a real season, in order."""
    cells: list[Cell] = []
    for seat in (0, 1):
        memory = EventMemory.initial()
        started = False
        for turn in kaito_turns:
            if turn.seat != seat:
                continue
            memory = memory.advance(1) if started else memory
            started = True
            transition = detect_event(turn.encoded, turn.action, memory, EVENT_CONFIG)
            memory = transition.memory
            if transition.event is not None:
                cells.append(Cell(turn, transition.event))
    return tuple(cells)


@pytest.fixture(scope="session")
def busy_cell(market_cells: tuple[Cell, ...]) -> Cell:
    """Return the first event on which every family contributes a row."""
    return next(
        cell
        for cell in market_cells
        if set(families_of(alternatives_for(cell, CONFIG))) == set(ALTERNATIVE_FAMILIES)
    )


def test_alternatives_include_required_families(busy_cell: Cell) -> None:
    alternatives = alternatives_for(busy_cell, CONFIG)

    assert alternatives.kaito == busy_cell.event.kaito_buckets
    assert set(families_of(alternatives)) >= {
        "kaito",
        "cancel",
        "scale_down",
        "scale_up",
        "single",
        "ranked_multi",
    }
    assert 1 + len(alternatives.replacements) <= 48


def test_the_anchor_is_the_controllers_own_plan(market_cells: tuple[Cell, ...]) -> None:
    for cell in market_cells:
        alternatives = alternatives_for(cell, CONFIG)
        assert alternatives.kaito == cell.event.kaito_buckets
        assert all(row.family != "kaito" for row in alternatives.replacements)


def test_a_replacement_cannot_be_labelled_as_the_controllers_own_plan() -> None:
    with pytest.raises(ValueError, match="is the anchor, not a replacement"):
        Alternative(FAMILY_KAITO, (0,) * len(ALLOWED_SLOTS))


def test_a_real_season_reaches_every_family(market_cells: tuple[Cell, ...]) -> None:
    families: dict[str, int] = collections.Counter()
    widest = 0
    for cell in market_cells:
        alternatives = alternatives_for(cell, CONFIG)
        families.update(families_of(alternatives))
        widest = max(widest, 1 + len(alternatives.replacements))

    assert len(market_cells) == 770
    assert dict(families) == {
        "kaito": 770,
        "single": 3532,
        "ranked_multi": 1706,
        "cancel": 386,
        "scale_down": 186,
        "scale_up": 128,
    }
    assert set(families) == set(ALTERNATIVE_FAMILIES), (
        "a family that never proposes a legal row is a family the learner "
        "will never see, and no cap or legality test would notice"
    )
    assert widest == 25


def test_every_alternative_fills_completely_in_the_engine(
    market_cells: tuple[Cell, ...],
) -> None:
    replayed = 0
    for cell in market_cells[::ENGINE_STRIDE]:
        for row in alternatives_for(cell, CONFIG).replacements:
            result = merge_residual_action(
                cell.turn.encoded,
                cell.turn.action,
                ResidualDecision(ResidualMode.REPLACE, row.buckets),
            )
            assert result.replaced, f"{row.family} row was not playable at all"
            queue = list(result.action["market"])
            assert engine_fills(
                cell.turn.observation, cell.turn.seat, queue
            ) == ordered_units(queue)
            replayed += 1
    assert replayed == 1438


def test_rows_are_unique_and_lexicographically_ordered(
    market_cells: tuple[Cell, ...],
) -> None:
    for cell in market_cells:
        alternatives = alternatives_for(cell, CONFIG)
        tail = [row.buckets for row in alternatives.replacements]
        every = [alternatives.kaito, *tail]
        assert tail == sorted(tail)
        assert len(set(every)) == len(every)
        assert all(len(buckets) == len(ALLOWED_SLOTS) for buckets in every)
        assert all(0 <= bucket < BUCKETS for buckets in every for bucket in buckets)


def test_scaled_rows_are_the_plan_at_half_and_at_half_again(
    market_cells: tuple[Cell, ...],
) -> None:
    seen: dict[str, int] = collections.Counter()
    for cell in market_cells:
        plan = cell.event.kaito_buckets
        for row in alternatives_for(cell, CONFIG).replacements:
            if row.family not in ("scale_down", "scale_up"):
                continue
            seen[row.family] += 1
            numerator = 1 if row.family == "scale_down" else 3
            for asked, planned in zip(row.buckets, plan, strict=True):
                target = quantity_of(planned) * numerator // 2
                assert quantity_of(asked) <= target
                assert asked + 1 == BUCKETS or quantity_of(asked + 1) > target
    assert seen == {"scale_down": 186, "scale_up": 128}


def test_a_tight_budget_binds_every_family(market_cells: tuple[Cell, ...]) -> None:
    families: dict[str, int] = collections.Counter()
    overrun = 0
    for cell in market_cells:
        counts = families_of(alternatives_for(cell, TIGHT))
        assert sum(counts.values()) <= TIGHT.max_alternatives
        assert counts["single"] <= TIGHT.max_single
        assert counts["ranked_multi"] <= TIGHT.max_ranked_multi
        families.update(counts)
        shipped = families_of(alternatives_for(cell, CONFIG))
        overrun += int(
            shipped["single"] > TIGHT.max_single
            or shipped["ranked_multi"] > TIGHT.max_ranked_multi
        )
    assert overrun == 516, "the tight budget never actually cut anything"
    assert dict(families) == {
        "kaito": 770,
        "single": 1060,
        "ranked_multi": 490,
        "cancel": 386,
        "scale_down": 186,
        "scale_up": 128,
    }


def test_a_budget_that_cannot_fit_the_cap_is_refused() -> None:
    with pytest.raises(ValueError, match="above the cap of 48"):
        AlternativeConfig(max_single=40, max_ranked_multi=8, max_alternatives=48)


def test_the_kept_rows_are_the_best_quoted_ones(busy_cell: Cell) -> None:
    shipped = [
        row.buckets
        for row in alternatives_for(busy_cell, CONFIG).replacements
        if row.family == "single"
    ]
    tight = [
        row.buckets
        for row in alternatives_for(busy_cell, TIGHT).replacements
        if row.family == "single"
    ]

    assert len(tight) == TIGHT.max_single < len(shipped)
    assert set(tight) < set(shipped), "the budget must cut, not re-propose"


def test_state_from_another_turn_is_refused(market_cells: tuple[Cell, ...]) -> None:
    first, later = market_cells[0], market_cells[-1]

    with pytest.raises(ValueError, match="does not match the event|event recorded"):
        generate_alternatives(
            first.event, later.turn.encoded, later.turn.action, CONFIG
        )


def test_a_plan_the_codec_cannot_read_is_refused(
    kaito_turns: tuple[Turn, ...], market_cells: tuple[Cell, ...]
) -> None:
    cell = market_cells[0]
    unreadable = {**cell.turn.action, "market": [["SELL", "CARROT", 13]]}

    with pytest.raises(ValueError, match="not exactly readable|event recorded"):
        generate_alternatives(cell.event, cell.turn.encoded, unreadable, CONFIG)


DETERMINISM_PROBE = """
import json, sys
from kaggriculture.features import encode_observation
from kaggriculture.learn.market_residual.alternatives import (
    AlternativeConfig,
    generate_alternatives,
)
from kaggriculture.market_residual.events import EventConfig, EventMemory, detect_event

cell = json.load(open(sys.argv[1]))
encoded = encode_observation(cell["observation"], cell["seat"])
event = detect_event(
    encoded, cell["action"], EventMemory.initial(), EventConfig()
).event
alternatives = generate_alternatives(
    event, encoded, cell["action"], AlternativeConfig()
)
print(alternatives.sha256)
rows = [["kaito", list(alternatives.kaito)]]
rows += [[row.family, list(row.buckets)] for row in alternatives.replacements]
print(json.dumps(rows))
"""


def probe(cell_path: Path, script: Path, seed: str) -> str:
    """Return the probe's output for one cell under one interpreter hash seed."""
    return subprocess.run(
        [sys.executable, str(script), str(cell_path)],
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
    ).stdout


def test_the_alternative_set_is_byte_equal_under_a_different_hash_seed(
    busy_cell: Cell, tmp_path: Path
) -> None:
    cell_path = tmp_path / "cell.json"
    cell_path.write_text(
        json.dumps(
            {
                "observation": busy_cell.turn.observation,
                "action": busy_cell.turn.action,
                "seat": busy_cell.turn.seat,
            }
        )
    )
    script = tmp_path / "probe.py"
    script.write_text(DETERMINISM_PROBE)

    first = probe(cell_path, script, "0")
    second = probe(cell_path, script, "12345")

    assert first == second, (
        "the alternative set moved with the interpreter's hash seed, so "
        "collection and training would disagree about which row is which"
    )
    assert first.splitlines()[0] == alternatives_for(busy_cell, CONFIG).sha256


def test_the_digest_moves_with_the_rows(market_cells: tuple[Cell, ...]) -> None:
    digests = {alternatives_for(cell, CONFIG).sha256 for cell in market_cells}
    shipped = alternatives_for(market_cells[0], CONFIG)
    reordered = AlternativeSet(
        shipped.fingerprint, shipped.kaito, tuple(reversed(shipped.replacements))
    )

    assert len(digests) > 100, "a constant digest would make the identity untested"
    assert reordered.sha256 != shipped.sha256


def test_the_learnable_slots_are_the_only_ones_an_alternative_touches(
    market_cells: tuple[Cell, ...],
) -> None:
    for cell in market_cells[::ENGINE_STRIDE]:
        frozen = [
            order
            for order in cell.turn.action["market"]
            if order[0] not in ("SELL", "BUY_PRODUCT")
        ]
        for row in alternatives_for(cell, CONFIG).replacements:
            result = merge_residual_action(
                cell.turn.encoded,
                cell.turn.action,
                ResidualDecision(ResidualMode.REPLACE, row.buckets),
            )
            merged = result.action["market"]
            assert [o for o in merged if o[0] not in ("SELL", "BUY_PRODUCT")] == frozen
            assert {
                (order[0], order[1]): order[2]
                for order in merged
                if order[0] in ("SELL", "BUY_PRODUCT")
            } == {
                MARKET_SLOTS[slot]: quantity_of(bucket)
                for slot, bucket in zip(ALLOWED_SLOTS, row.buckets, strict=True)
                if bucket
            }
