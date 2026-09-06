"""What the composed message says, and what it must never say."""

import re

from kaggriculture.campaign import (
    archive,
    config,
    evaluator,
    gate,
    harness,
    prompt,
    validate,
)

IMPROVE = prompt.INSTRUCTIONS[0][1]


def day(number: int, ours: float, theirs: float) -> harness.Day:
    """One row of a day table, with something in every field."""
    return harness.Day(
        day=number,
        ours_bank=ours,
        theirs_bank=theirs,
        ours_shed={"WHEAT": 12},
        theirs_shed={"EGG": 3},
        ours_hands=2,
        theirs_hands=1,
        prices={"WHEAT": 25},
    )


def result(rates: dict[str, float], days: int = 2) -> evaluator.FastResult:
    """A fast evaluation standing in for one the loop played."""
    hardest = min(rates, key=lambda name: rates[name])
    return evaluator.FastResult(
        fitness=sum(rates.values()) / len(rates),
        field=0.5,
        rates=rates,
        margins={
            name: harness.Margin(mean=-100.0, worst=-300.0, best=50.0) for name in rates
        },
        seeds=[1, 2, 3, 4],
        hardest=hardest,
        states=[day(n, 3000.0 - n, 3000.0 + n) for n in range(days)],
    )


def test_the_message_names_the_program_and_asks_for_one_edit() -> None:
    """The model edits child.py and stops; the campaign plays it."""
    text = prompt.compose("champion_1", result({"v54": 0.3}), [], IMPROVE)

    assert "`child.py` in your working directory is `champion_1`" in text
    assert "Edit it in place and stop" in text
    assert "the campaign plays" in text
    assert f"{config.ROUND_LIMIT_SECONDS // 60}\nminutes" in text.replace(" ", "\n")


def test_the_verdict_is_the_gates_own_reading_of_a_win() -> None:
    """One implementation of "did it win", so a model cannot believe otherwise.

    The sentence naming what the program does not beat is the gate's own,
    word for word, which is what stops a model concluding it has cleared a
    bar the gate then refuses it on.
    """
    rates = {"v54": 0.3, "v56": 0.9}

    text = prompt.compose("champion_1", result(rates), [], IMPROVE)

    assert f"It {gate.promotion(rates)[1]}." in text
    assert "It did not beat v54 at 0.300." in text
    assert "| v54 | 0.300 | -100 | -300 | +50 |" in text
    assert "| v56 | 0.900 |" in text


def test_a_program_that_beats_everything_is_told_so() -> None:
    """The bar is stated the same way whether or not it has been cleared."""
    rates = {"v54": 0.9, "v56": 0.8}

    text = prompt.compose("champion_1", result(rates), [], IMPROVE)

    assert f"It {gate.promotion(rates)[1]}." in text
    assert "It beat every opponent." in text
    assert "sealed block" in text


def test_the_states_are_one_game_day_by_day() -> None:
    """One lost game against the hardest opponent, both sides, day by day."""
    text = prompt.compose(
        "champion_1", result({"v54": 0.0, "v56": 0.9}, days=30), [], IMPROVE
    )

    assert "One game against `v54`, day by day" in text
    assert text.count("| WHEAT 12 | EGG 3 |") == 30
    assert "| 29 | 2971 | 3029 |" in text
    # The shed is hidden from a player at runtime; the author is not a player.
    assert "cannot see the opponent's shed" in text


def test_the_message_names_opponents_and_never_a_path() -> None:
    """The doctrine: nothing the loop composes carries an opponent's path.

    Everything a model is given is this string, so this is the whole of the
    campaign's exposure. Opponent names travel; nothing that could be opened
    does.
    """
    text = prompt.compose(
        "champion_1", result({"v54": 0.1, "router_v1": 0.0}), [], IMPROVE
    )

    assert "/data/kaggriculture" not in text
    assert not re.search(r"/(?:home|data|Users|tmp)/\S*", text)
    assert str(config.ROOT) not in text
    assert "router_v1" in text and "v54" in text


def test_the_message_states_the_imports_the_gate_actually_allows() -> None:
    """A model told it may import our package would write a program that dies.

    One file ships, so the whitelist is the program's whole dependency
    surface. The section is rendered from `validate.ALLOWED_IMPORTS` rather
    than restated, because a model told a different set from the one that
    rejects it is worse than one told nothing.
    """
    text = prompt.compose("champion_1", result({"v54": 0.5}), [], IMPROVE)

    for name in validate.ALLOWED_IMPORTS:
        assert f"`{name}`" in text, name
    assert "`ctypes`" not in text
    assert "`kaggriculture`" not in text


def test_the_message_carries_the_rules_and_nothing_to_run() -> None:
    """The game's rules travel; the harness section does not, having nothing to run."""
    text = prompt.compose("champion_1", result({"v54": 0.5}), [], IMPROVE)

    assert "Kaggriculture policy task" in text
    assert "never read opponent source" in text
    assert "campaign play AGENT" not in text and "campaign check" not in text
    assert "uv run" not in text and "--vs" not in text
    assert "engine/kaggriculture.py" not in text
    assert "700000" not in text


def test_there_are_five_full_instructions_and_they_differ() -> None:
    """FAMOU C.2's variants, so eight workers on one champion diverge."""
    names = [name for name, _ in prompt.INSTRUCTIONS]
    texts = [text for _, text in prompt.INSTRUCTIONS]
    assert len(prompt.INSTRUCTIONS) == 5
    assert len(set(names)) == 5 and len(set(texts)) == 5
    assert all("child.py" in text for text in texts)


def failure(reason: str) -> archive.Failure:
    """One rejected round, as the ledger holds it."""
    return archive.Failure(
        started_from="champion_1",
        instruction="improve",
        reason=reason,
        created=0.0,
    )


def test_the_message_carries_the_lineages_recent_failures() -> None:
    """A rejected round's reason is what the next round can act on.

    "Your program did not parse" is feedback a model can use, so the last few
    failures on the lineage travel with the verdict. Only the last few: an
    older one is about a program this lineage has already moved past. Each
    arrives on one line, because a reason is free-form -- a codex call's own
    last message is one -- and a bullet list is no place for a traceback.
    """
    failures = [
        failure("syntax: invalid syntax (<unknown>, line 12)"),
        failure("contract: agent is shadowed by 'policy', the last callable"),
        failure("crashed: KeyError: 'EGG'"),
        failure("no_output: I rewrote the planner\nand left it in child.py"),
    ]

    text = prompt.compose("champion_1", result({"v54": 0.5}), failures, IMPROVE)

    assert "## Recent attempts on `champion_1` that produced nothing" in text
    assert "- contract: agent is shadowed by 'policy', the last callable" in text
    assert "- crashed: KeyError: 'EGG'" in text
    assert "- no_output: I rewrote the planner and left it in child.py" in text
    assert "invalid syntax" not in text
    # The instruction stays the last thing said, failures or not.
    assert text.rstrip().endswith(IMPROVE)


def test_a_lineage_with_nothing_against_it_gets_no_failure_section() -> None:
    """A heading over an empty list is noise in a message read every round."""
    text = prompt.compose("champion_1", result({"v54": 0.5}), [], IMPROVE)

    assert "produced nothing" not in text


def test_every_instruction_reaches_the_message_whole() -> None:
    """The drawn instruction is the last thing said, and only that one."""
    for index, (name, text) in enumerate(prompt.INSTRUCTIONS):
        message = prompt.compose("champion_1", result({"v54": 0.5}), [], text)
        assert message.rstrip().endswith(text), name
        others = [t for i, (_, t) in enumerate(prompt.INSTRUCTIONS) if i != index]
        assert not [other for other in others if other in message], name
