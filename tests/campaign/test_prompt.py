"""What the composed message says, and what it must never say."""

import re
from pathlib import Path

from kaggriculture.campaign import (
    archive,
    config,
    evaluator,
    gate,
    harness,
    prompt,
    validate,
)

IMPROVE = prompt.INSTRUCTION


def table(
    name: str, rates: dict[str, float], place: str = "bottom"
) -> dict[str, float]:
    """Standings with ``name`` in them, above or below the opponents.

    The real ones come from a Bradley-Terry fit over the tournament; these are
    written out, because what the message says about a place should not depend
    on a solver agreeing with the test about the numbers.
    """
    standings = {opponent: -float(index) for index, opponent in enumerate(rates)}
    standings[name] = 1.0 if place == "top" else -float(len(rates))
    return standings


def day(number: int, ours: float, theirs: float) -> harness.Day:
    """One row of a day table, with something in every field."""
    return harness.Day(
        day=number,
        ours_bank=ours,
        theirs_bank=theirs,
        ours_plants={"WHEAT": 4},
        theirs_plants={"MELON": 2},
        ours_animals={},
        theirs_animals={"COW": 1},
        ours_weeds=0,
        theirs_weeds=3,
        ours_seeds={"WHEAT": 5},
        ours_shed={"WHEAT": 12},
        theirs_shed={"EGG": 3},
        ours_hands=2,
        theirs_hands=1,
        prices={"WHEAT": 25},
    )


def result(rates: dict[str, float], days: int = 2) -> evaluator.Result:
    """A fast evaluation standing in for one the loop played."""
    hardest = min(rates, key=lambda name: rates[name])
    return evaluator.Result(
        program_id="p1",
        fitness=sum(rates.values()) / len(rates),
        field=0.5,
        rates=rates,
        margins={
            name: harness.Margin(mean=-100.0, worst=-300.0, best=50.0) for name in rates
        },
        seeds=[1, 2, 3, 4],
        hardest=hardest,
        states={
            name: [day(n, 3000.0 - n, 3000.0 + n) for n in range(days)]
            for name in rates
        },
    )


def test_the_message_names_the_program_and_asks_for_one_edit() -> None:
    """The model edits child.py and stops; the campaign plays it."""
    text = prompt.compose(
        "champion_1",
        result({"v54": 0.3}),
        [],
        [],
        IMPROVE,
        table("champion_1", {"v54": 0.5}),
    )

    assert "`child.py` in your working directory is `champion_1`" in text
    assert "Edit it in place and stop" in text
    assert "the campaign plays" in text
    # Whitespace-normalised: the paragraph is wrapped, so the sentence this
    # is about spans a line break in the source.
    assert "no time limit on this call" in " ".join(text.split())


def test_the_verdict_is_the_gates_own_reading_of_a_win() -> None:
    """One implementation of "did it win", so a model cannot believe otherwise.

    The sentence naming what the program does not beat is the gate's own,
    word for word, which is what stops a model concluding it has cleared a
    bar the gate then refuses it on.
    """
    rates = {"v54": 0.3, "v56": 0.9}
    standings = table("champion_1", rates)

    text = prompt.compose("champion_1", result(rates), [], [], IMPROVE, standings)

    assert f"It {gate.promotion(standings, 'champion_1')[1]}." in text
    # Last of three, and told which agent is above it rather than merely that
    # it lost: a place is something the next round can aim to improve.
    assert "3 of 3" in text
    # The per-opponent rates are still there, because a place says where the
    # program stands and these say against whom.
    assert "| v54 | 0.300 | -100 | -300 | +50 |" in text
    assert "| v56 | 0.900 |" in text


def test_a_program_at_the_top_of_the_tournament_is_told_so() -> None:
    """The verdict is a place, and the top of the table is stated as one."""
    rates = {"v54": 0.9, "v56": 0.8}
    standings = table("champion_1", rates, place="top")

    text = prompt.compose("champion_1", result(rates), [], [], IMPROVE, standings)

    assert f"It {gate.promotion(standings, 'champion_1')[1]}." in text
    assert "top of the tournament" in text
    # Topping it *is* the promotion now; there is no second block to clear,
    # and telling a round otherwise would describe a gate that no longer runs.
    assert "this program is the champion" in text
    assert "sealed" not in text
    # The whole table, so a round can see who it has yet to pass.
    assert "| rank | agent | rating |" in text
    assert "| 1 | **champion_1** |" in text


def test_every_opponent_it_lost_to_is_shown_day_by_day() -> None:
    """The losses, because that is where there is something to learn.

    An opponent it never beats is one it has to learn to beat; one it already
    beats has nothing left to teach, so `v56` at 0.9 gets no table.
    """
    text = prompt.compose(
        "champion_1",
        result({"v54": 0.0, "v56": 0.9}, days=30),
        [],
        [],
        IMPROVE,
        table("champion_1", {"v54": 0.0, "v56": 0.9}),
    )

    assert "The matches it lost, day by day" in text
    assert "### `v54`, won 0.000" in text
    assert "### `v56`" not in text
    assert text.count("| WHEAT 12 | EGG 3 |") == 30
    assert "| 29 | 2971 | 3029 |" in text
    # The shed is hidden from a player at runtime; the author is not a player.
    assert "cannot see the opponent's shed" in text
    # The production side of every row: what each farm was growing while the
    # banks moved, ours with the seed it had not planted yet.
    assert text.count("| WHEAT 4 / - / - | MELON 2 / COW 1 / 3 | WHEAT 5 |") == 30


def test_the_message_names_opponents_and_never_a_path() -> None:
    """The doctrine: nothing the loop composes carries an opponent's path.

    Everything a model is given is this string, so this is the whole of the
    campaign's exposure. Opponent names travel; nothing that could be opened
    does.
    """
    text = prompt.compose(
        "champion_1",
        result({"v54": 0.1, "router_v1": 0.0}),
        [],
        [],
        IMPROVE,
        table("champion_1", {"v54": 0.5}),
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
    text = prompt.compose(
        "champion_1",
        result({"v54": 0.5}),
        [],
        [],
        IMPROVE,
        table("champion_1", {"v54": 0.5}),
    )

    for name in validate.ALLOWED_IMPORTS:
        assert f"`{name}`" in text, name
    assert "`ctypes`" not in text
    assert "`kaggriculture`" not in text


def test_the_message_carries_the_rules_and_nothing_to_run() -> None:
    """The game's rules travel; the harness section does not, having nothing to run."""
    text = prompt.compose(
        "champion_1",
        result({"v54": 0.5}),
        [],
        [],
        IMPROVE,
        table("champion_1", {"v54": 0.5}),
    )

    assert "Kaggriculture policy task" in text
    assert "never read opponent source" in text
    assert "campaign play AGENT" not in text and "campaign check" not in text
    assert "uv run" not in text and "--vs" not in text
    assert "engine/kaggriculture.py" not in text
    assert "700000" not in text


def test_the_instruction_states_the_bar_and_not_a_method() -> None:
    """One instruction, and it says what the gate asks rather than how.

    There were five drawn per session. Two of them told a round to
    replace the program with something else, which cost 54% of every
    call the campaign made and returned 259 programs of which one
    scored above nought -- because the program is a rated agent now,
    and a farm bot written from scratch loses every game to this pool.
    The three that survived were within 0.06 of each other.
    """
    assert "child.py" in prompt.INSTRUCTION
    assert "top of the standings" in prompt.INSTRUCTION
    # It must not prescribe one route to the top: naming a method is
    # what the five did, and three of them named the same one.
    for route in ("tune", "restructure", "replace"):
        assert route in prompt.INSTRUCTION


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

    text = prompt.compose(
        "champion_1",
        result({"v54": 0.5}),
        failures,
        [],
        IMPROVE,
        table("champion_1", {"v54": 0.5}),
    )

    assert "## Recent attempts on `champion_1` that produced nothing" in text
    assert "- contract: agent is shadowed by 'policy', the last callable" in text
    assert "- crashed: KeyError: 'EGG'" in text
    assert "- no_output: I rewrote the planner and left it in child.py" in text
    assert "invalid syntax" not in text
    # The instruction stays the last thing said, failures or not.
    assert text.rstrip().endswith(IMPROVE)


def test_a_lineage_with_nothing_against_it_gets_no_failure_section() -> None:
    """A heading over an empty list is noise in a message read every round."""
    text = prompt.compose(
        "champion_1",
        result({"v54": 0.5}),
        [],
        [],
        IMPROVE,
        table("champion_1", {"v54": 0.5}),
    )

    assert "produced nothing" not in text


def test_the_instruction_reaches_the_message_whole() -> None:
    """It is the last thing said, and it arrives uncut."""
    message = prompt.compose(
        "champion_1",
        result({"v54": 0.5}),
        [],
        [],
        prompt.INSTRUCTION,
        table("champion_1", {"v54": 0.5}),
    )

    assert message.rstrip().endswith(prompt.INSTRUCTION)


def stored(
    db: archive.Database, name: str, parent: str, fitness: float, docstring: str
) -> archive.Program:
    """Store a program whose source opens with ``docstring``, and add it."""
    source = f'"""{docstring}"""\n\n\ndef agent(o, c=None):\n    return {{}}\n'
    program = archive.Program(
        id=name,
        source_path=str(db.store(source, name)),
        started_from=parent,
        instruction="tune",
        model="gpt-5.6-luna",
        fitness=fitness,
        field=fitness,
        rates={"v54": fitness},
        margins={"v54": harness.Margin(mean=-100.0, worst=-300.0, best=50.0)},
        created=0.0,
    )
    db.add(program)
    return program


def test_the_message_says_what_has_already_been_made_of_the_program(
    tmp_path: Path,
) -> None:
    """Sessions all start from the same program and must not repeat each other.

    A round is told what earlier rounds made of exactly the program it holds,
    and what those scored. Without it the only feedback crossing between
    attempts is a failure that produced no program at all, so a direction that
    was tried and measured as bad is indistinguishable from one never tried,
    and the campaign re-explores it for as long as it runs. AlphaEvolve and
    FAMOU both feed prior candidates' measured performance into the next
    prompt.
    """
    db = archive.Database(tmp_path / "db.jsonl", tmp_path / "programs")
    stored(db, "worse", "champion_1", 0.1, "Sold wheat on sight. Worse.")
    stored(db, "better", "champion_1", 0.4, "Held wheat for the glut to lift.")
    stored(db, "elsewhere", "champion_2", 0.9, "Another lineage entirely.")

    text = prompt.compose(
        "champion_1",
        result({"v54": 0.5}),
        [],
        db.children("champion_1"),
        IMPROVE,
        table("champion_1", {"v54": 0.5}),
    )

    assert "What has already been made of `champion_1`" in text
    # Best first, so the row that says what to beat is the one read first.
    assert text.index("better") < text.index("worse")
    assert "Held wheat for the glut to lift." in text
    assert "Sold wheat on sight. Worse." in text
    assert "0.400" in text and "0.100" in text
    # Another program's children are not this program's.
    assert "elsewhere" not in text and "Another lineage entirely" not in text


def test_only_the_best_few_siblings_are_shown_and_the_rest_are_counted(
    tmp_path: Path,
) -> None:
    """A champion accumulates children for as long as it stands.

    All of them would be most of the message and most of it noise, so the
    count is stated and the best `SIBLINGS` are shown -- the ones that say
    what the ceiling from here is.
    """
    db = archive.Database(tmp_path / "db.jsonl", tmp_path / "programs")
    for n in range(prompt.SIBLINGS + 5):
        stored(db, f"p{n}", "champion_1", n / 100, f"Attempt {n}.")

    text = prompt.compose(
        "champion_1",
        result({"v54": 0.5}),
        [],
        db.children("champion_1"),
        IMPROVE,
        table("champion_1", {"v54": 0.5}),
    )

    assert f"{prompt.SIBLINGS + 5} program(s) have been written" in text
    assert f"the best {prompt.SIBLINGS} of them" in text
    assert "Attempt 12." in text  # the best
    assert "Attempt 0." not in text  # the worst, cut


def test_a_program_nothing_has_been_made_of_gets_no_section(tmp_path: Path) -> None:
    """A heading over an empty table is noise in a message read every round."""
    db = archive.Database(tmp_path / "db.jsonl", tmp_path / "programs")

    text = prompt.compose(
        "champion_1",
        result({"v54": 0.5}),
        [],
        db.children("champion_1"),
        IMPROVE,
        table("champion_1", {"v54": 0.5}),
    )

    assert "already been made" not in text


def test_a_sibling_with_no_docstring_is_still_shown_for_its_score(
    tmp_path: Path,
) -> None:
    """The number is the point; the program's own account of itself is a bonus.

    A round is asked for a docstring saying what it changed, and mostly writes
    one. Dropping the row when it did not would hide a measured result over a
    missing comment.
    """
    db = archive.Database(tmp_path / "db.jsonl", tmp_path / "programs")
    bare = archive.Program(
        id="bare",
        source_path=str(db.store("def agent(o, c=None):\n    return {}\n", "bare")),
        started_from="champion_1",
        instruction="tune",
        model="gpt-5.6-luna",
        fitness=0.25,
        field=0.25,
        created=0.0,
    )
    db.add(bare)

    text = prompt.compose(
        "champion_1",
        result({"v54": 0.5}),
        [],
        db.children("champion_1"),
        IMPROVE,
        table("champion_1", {"v54": 0.5}),
    )

    assert "| bare | tune | 0.250 | - |" in text


def test_the_game_shown_is_against_the_agent_directly_above() -> None:
    """The next place, not the furthest one.

    Under the absolute gate the worst matchup was the binding constraint, so
    that was the game to study. Under a tournament it is usually just the
    strongest agent in the pool, and a program at the bottom loses to it
    sixteen games to nothing -- a different league, not a next step. Measured
    on the live campaign, a round was being shown `router2929` at 0.000 while
    the agent it had to overtake was `indarkarhana`, which it already took a
    quarter of its games from.
    """
    rates = {"unreachable": 0.0, "rival": 0.25, "below": 0.9}
    standings = {"unreachable": 3.0, "rival": 1.0, "champion_1": 0.0, "below": -2.0}

    text = prompt.compose("champion_1", result(rates), [], [], IMPROVE, standings)

    # Both losses are shown, worst first, so the agent it never beats leads.
    assert text.index("### `unreachable`") < text.index("### `rival`")
    # And the one it has to pass is marked, because that is the next place.
    assert "### `rival`, won 0.250 -- directly above you" in text
    # The opponent it already beats has nothing left to teach.
    assert "### `below`" not in text


def test_a_program_that_lost_nothing_is_told_so_rather_than_shown_nothing() -> None:
    """A heading over no tables would read as a section that went missing.

    It happens the moment a program beats the whole pool, which is also the
    moment it is promoted, so the message says why there is nothing here.
    """
    rates = {"second": 0.6, "third": 0.9}
    standings = {"champion_1": 2.0, "second": 1.0, "third": -1.0}

    text = prompt.compose("champion_1", result(rates), [], [], IMPROVE, standings)

    assert "beat every opponent in the pool" in text
    assert "### `second`" not in text
