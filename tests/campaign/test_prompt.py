"""What the composed message says, and what it must never say."""

import re
from pathlib import Path

import pytest

from kaggriculture.campaign import (
    archive,
    config,
    dataset,
    evaluator,
    gate,
    harness,
    prompt,
    strategy,
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


def measured(**held: float) -> dict[str, float]:
    """A full measures dict: what is named, and zero for everything else.

    Every quantity, not a chosen few -- a `Day` whose measures are partial is
    a day the selector can only half read, and the point of the dict is that
    it is total.
    """
    return dict.fromkeys(dataset.COLUMNS, 0.0) | held


def day(number: int, ours: float, theirs: float) -> harness.Day:
    """One row of a day table, with something in every field."""
    return harness.Day(
        ours=measured(bank=ours, planted=4, hands=2, shed=12, seeds=5, weeds=0),
        theirs=measured(bank=theirs, planted=2, hands=1, shed=3, weeds=3),
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


def test_a_template_names_its_own_fields(tmp_path: Path) -> None:
    """The file decides what fills it, so there is no second list to keep."""
    path = tmp_path / "t.md"
    path.write_text(
        "{alpha} and {beta}, but {{literal}} is not one\n", encoding="utf-8"
    )

    message = prompt.Message(path)

    assert message.fields == {"alpha", "beta"}
    assert message.render(alpha="a", beta="b") == "a and b, but {literal} is not one\n"


def test_a_template_and_its_caller_cannot_drift_apart(tmp_path: Path) -> None:
    """`str.format` drifts both ways and only complains about one.

    It raises on a placeholder the caller did not supply, and silently keeps
    one the template stopped using -- so half of the drift ships. Both sides
    are named here, which is what `PromptTemplate(validate_template=True)`
    would buy from langchain-core without the framework behind it.
    """
    path = tmp_path / "t.md"
    path.write_text("{alpha} and {beta}\n", encoding="utf-8")
    message = prompt.Message(path)

    with pytest.raises(ValueError, match=r"wants \['beta'\]"):
        message.render(alpha="a")
    with pytest.raises(ValueError, match=r"given \['gamma'\]"):
        message.render(alpha="a", beta="b", gamma="c")


def test_the_round_template_is_loaded_and_checked_at_import() -> None:
    """A template nobody can fill must fail before a codex call is spent."""
    assert prompt.ROUND.fields
    # Composing supplies exactly what the file asks for; `render` raises
    # otherwise, so reaching the end of this is the assertion.
    prompt.compose(
        "champion_1",
        result({"v54": 0.0}),
        [],
        [],
        IMPROVE,
        table("champion_1", {"v54": 0.0}),
    )


def test_the_message_carries_the_build_order() -> None:
    """A round is shown its own day-by-day play and nothing to read it against.

    The agents at the top of the leaderboard publish no kernels, so their
    recorded games are the only view of them there is -- and the pool, built
    from published work, cannot supply it. The table is averaged over the
    top of a rating by `build-order` and travels whole.

    Over the top of a *rating*, not the winning side of each game: about half
    of a ladder's winners are the weaker agent having a good day, and eleven
    quantities measured that way came back between 45% and 60%.
    """
    text = prompt.compose(
        "champion_1",
        result({"v54": 0.0}, days=30),
        [],
        [],
        IMPROVE,
        table("champion_1", {"v54": 0.0}),
    )

    assert "How the strongest agents build" in text
    # Matched loosely: the table is generated and then formatted, and prettier
    # pads markdown cells to align them.
    assert re.search(r"\|\s*quadrants\s*\|", text)
    assert re.search(r"\|\s*fertilised\s*\|", text)
    # It travels whole: a table cut in half is a table nobody can read down.
    assert prompt.BUILD_ORDER.read_text(encoding="utf-8").rstrip() in text
    # Aggregate only: no opponent is named and no path of theirs appears.
    assert "/data" not in text


def test_a_round_can_see_the_two_columns_the_corpus_decides_on(tmp_path: Path) -> None:
    """Quadrants and fertilizer, for both sides, on every row.

    These are the sharpest separations in 16,292 recorded games -- fertilised
    tiles on day five at 100% of 313 paired games, quadrants on day three at
    99% -- and until now a round was shown its banks, its tiles and its shed
    and could see neither. A build-order table stating a number the day table
    does not carry is a target nobody can read their own position against.
    """
    del tmp_path
    rates = {"v54": 0.3}
    days = [
        harness.Day(
            day=n,
            ours_bank=100.0,
            theirs_bank=200.0,
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
            ours_quadrants=1,
            theirs_quadrants=3,
            ours_fertilised=0,
            theirs_fertilised=17,
            prices={"WHEAT": 25},
        )
        for n in range(2)
    ]
    played = result(rates)
    scored = played.model_copy(update={"states": {"v54": days}})

    text = prompt.compose(
        "champion_1", scored, [], [], IMPROVE, table("champion_1", rates)
    )

    assert "our quads | their quads | our fert | their fert" in text
    # One quadrant against three, no fertilizer against seventeen: the gap the
    # build order is about, legible on the row.
    assert text.count("| 1 | 3 | 0 | 17 |") == 2


def test_the_templates_own_note_is_not_sent_to_the_model() -> None:
    """The file explains itself at the top, and that note is not the prompt.

    It is also a trap rather than merely noise: the note names the
    placeholders it documents, so `str.format` substitutes them and every
    section of the message is rendered twice -- silently, in a message nobody
    reads end to end. Found exactly that way.
    """
    text = prompt.compose(
        "champion_1",
        result({"v54": 0.0}, days=30),
        [],
        [],
        IMPROVE,
        table("champion_1", {"v54": 0.0}),
    )

    assert prompt.ROUND_PROMPT.read_text(encoding="utf-8").startswith("<!--")
    assert "<!--" not in text
    assert "the parts the campaign" not in text
    # One opponent, one table, thirty days: rendered once.
    assert text.count("### `v54`") == 1


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


def test_a_program_that_clears_the_bar_is_told_by_how_much() -> None:
    """The verdict is a rating gap now, not a place.

    A place had no margin in it and, over a sampled draw, was not even a
    place -- it was top of whichever sixteen opponents the candidate happened
    to draw. The gap says how far above the floor it sits, on one scale.
    """
    rates = {"v54": 0.9, "v56": 0.8}
    standings = table("champion_1", rates, place="top")

    text = prompt.compose("champion_1", result(rates), [], [], IMPROVE, standings)

    assert f"It {gate.promotion(standings, 'champion_1')[1]}." in text
    # No floor in these standings, so leading the field is the bar.
    assert "with no floor yet" in text
    # Topping it *is* the promotion now; there is no second block to clear,
    # and telling a round otherwise would describe a gate that no longer runs.
    assert "this program is the champion" in text
    assert "sealed" not in text
    # The whole table, so a round can see who it has yet to pass.
    assert "| rank | agent | rating |" in text
    assert "| 1 | **champion_1** |" in text


def test_every_opponent_that_took_a_game_is_shown_day_by_day() -> None:
    """The losses, because that is where there is something to learn.

    A loss is a game lost, not a matchup lost. The cut used to be a rate at or
    below 0.5, which was right while the lineage lost nearly everything and
    inverted the moment the campaign was seeded from a strong agent: a program
    winning 0.875 against eight opponents was told "nothing to show", so the
    better it got the less it was shown. Only an opponent beaten every single
    time has nothing left to teach.
    """
    rates = {"v54": 0.0, "v56": 0.9, "shopforge": 1.0}
    text = prompt.compose(
        "champion_1",
        result(rates, days=30),
        [],
        [],
        IMPROVE,
        table("champion_1", rates),
    )

    assert "The matches it lost, day by day" in text
    # Never beaten, so first; and 0.9 still drops a game in ten, so it is here
    # too -- that is the one this used to hide.
    assert "### `v54`, won 0.000" in text
    assert "### `v56`, won 0.900" in text
    assert text.index("`v54`") < text.index("`v56`")
    # Beaten every time: nothing left to learn from it.
    assert "### `shopforge`" not in text


def test_a_program_that_wins_everything_is_told_so_rather_than_shown_nothing() -> None:
    """The empty case has to mean what it says, because it reads as an all-clear."""
    rates = {"v54": 1.0, "v56": 1.0}
    text = prompt.compose(
        "champion_1", result(rates), [], [], IMPROVE, table("champion_1", rates)
    )

    assert "won every game against every opponent" in text


def test_the_tables_shown_are_bounded() -> None:
    """A whole pool of day tables is most of the message and most of it noise."""
    rates = {f"agent_{n}": 0.5 for n in range(12)}
    text = prompt.compose(
        "champion_1",
        result(rates, days=30),
        [],
        [],
        IMPROVE,
        table("champion_1", rates),
    )

    assert text.count("### `agent_") == prompt.MOST_TABLES
    assert text.count("| WHEAT 12 | EGG 3 |") == 30 * prompt.MOST_TABLES


def test_a_shown_game_carries_every_column_of_every_day() -> None:
    """One table is the whole game: thirty days, both farms, both sheds."""
    text = prompt.compose(
        "champion_1",
        result({"v54": 0.0}, days=30),
        [],
        [],
        IMPROVE,
        table("champion_1", {"v54": 0.0}),
    )

    assert text.count("| WHEAT 12 | EGG 3 |") == 30
    assert "| 29 | 2971 | 3029 |" in text
    # The shed is hidden from a player at runtime; the author is not a player.
    assert "cannot see the opponent's shed" in text
    # The production side of every row: what each farm was growing while the
    # banks moved, ours with the seed it had not planted yet.
    assert text.count("| WHEAT 4 / - / - | MELON 2 / COW 1 / 3 | 0 | 0 |") == 30


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
    rates = {"unreachable": 0.0, "rival": 0.25, "below": 1.0}
    standings = {"unreachable": 3.0, "rival": 1.0, "champion_1": 0.0, "below": -2.0}

    text = prompt.compose("champion_1", result(rates), [], [], IMPROVE, standings)

    # Both losses are shown, worst first, so the agent it never beats leads.
    assert text.index("### `unreachable`") < text.index("### `rival`")
    # And the one it has to pass is marked, because that is the next place.
    assert "### `rival`, won 0.250 -- directly above you" in text
    # Beaten every single time, so nothing left to teach.
    assert "### `below`" not in text


def store(tmp_path: Path, *claims: tuple[str, int, float]) -> strategy.Strategies:
    """A claim store the corpus has already spoken about.

    Built rather than monkeypatched, and passed in: `selected` takes the store
    because a test that has to reach into the module to change where it reads
    from is a test of the reaching.
    """
    opened = strategy.Strategies(tmp_path / "strategies.jsonl")
    for quantity, when, agreement in claims:
        claim = opened.propose(strategy.Form(quantity=quantity, day=when))
        opened.record(claim.id, support=4000, agreement=agreement)
    return opened


def test_a_claim_this_program_already_follows_is_not_shown(tmp_path: Path) -> None:
    """What makes a claim worth a round is not that it is true.

    Every settled claim is true of the corpus by construction, so a section
    listing them would be the same paragraph every round on every program. The
    one thing that differs between programs is which of them this program is
    not doing.
    """
    # The day table has this program ahead on planted tiles in every game, and
    # the corpus says the stronger side has more of them.
    chosen = prompt.selected(
        store(tmp_path, ("planted", 1, 0.9)), result({"v54": 0.3}).states
    )

    assert chosen == []


def test_the_claims_shown_are_the_ones_this_program_breaks_worst_first(
    tmp_path: Path,
) -> None:
    """A round that reads one claim should read the one it is furthest from."""
    # This program trails on bank and leads on planted in every recorded game.
    # The corpus says the stronger side holds more bank and fewer planted
    # tiles, so it is on the wrong side of both and of neither of the others.
    chosen = prompt.selected(
        store(
            tmp_path,
            ("bank", 1, 0.95),
            ("planted", 1, 0.20),
            ("weeds", 1, 0.10),
            ("shed", 1, 0.93),
        ),
        result({"v54": 0.3, "v16": 0.5}).states,
    )

    assert [claim.form.quantity for _, _, claim in chosen] == ["bank", "planted"]
    # Both opponents, and the row says the count rather than the share.
    assert [(wrong, seen) for wrong, seen, _ in chosen] == [(2, 2), (2, 2)]


def test_a_claim_the_corpus_has_not_settled_never_reaches_a_round(
    tmp_path: Path,
) -> None:
    """The measurement decides what is shown, and which way round it is shown.

    A claim the corpus cannot separate the sides on has no direction, so there
    is no wrong side to put a program on -- and telling a round to act on one
    anyway is worse than telling it nothing.
    """
    unsettled = store(tmp_path, ("bank", 1, 0.5))

    assert unsettled.claims[0].status == "open"
    assert prompt.selected(unsettled, result({"v54": 0.3}).states) == []


def test_the_two_quantities_the_corpus_decides_on_can_be_selected_on(
    tmp_path: Path,
) -> None:
    """Rendering them in the table is not the same as being able to select.

    `harness.Day` carries quadrants and fertilizer, and the day tables show
    them -- but the bridge the selector reads did not, so the two widest
    separations in 16,292 games were visible to a reader and invisible to the
    thing that decides what a round is told. Shipped exactly that way once.
    """
    # The day table has this program on one quadrant against three.
    days = [
        result({"v54": 0.3})
        .states["v54"][0]
        .model_copy(
            update={
                "day": 1,
                "ours_quadrants": 1,
                "theirs_quadrants": 3,
                "ours_fertilised": 0,
                "theirs_fertilised": 17,
                "ours": measured(quadrants=1, fertilised=0),
                "theirs": measured(quadrants=3, fertilised=17),
            }
        )
    ]
    played = result({"v54": 0.3}).model_copy(update={"states": {"v54": days}})

    chosen = prompt.selected(
        store(tmp_path, ("quadrants", 1, 0.99), ("fertilised", 1, 0.95)), played.states
    )

    assert {claim.form.quantity for _, _, claim in chosen} == {
        "quadrants",
        "fertilised",
    }


def test_a_claim_a_day_table_cannot_carry_is_kept_and_not_shown(
    tmp_path: Path,
) -> None:
    """Hire orders are among the store's clearest findings and cannot be shown.

    A day table is a state at a moment and does not count what was submitted
    to reach it, so there is no value to put this program on a side of. It
    stays measured in the store; it does not become a claim about a program
    whose orders nobody counted.
    """
    assert "hire_orders" in strategy.QUANTITIES

    chosen = prompt.selected(
        store(tmp_path, ("hire_orders", 1, 0.2)), result({"v54": 0.3}).states
    )

    assert chosen == []


def test_a_claim_the_corpus_reversed_selects_the_program_that_does_more(
    tmp_path: Path,
) -> None:
    """Half the findings are about doing *less* of something.

    The strongest agents sell under half what the rest do. A program that
    sells more is the one that needs telling, and that only comes out right
    because the direction is read off the measurement rather than off
    something a person wrote down first.
    """
    # The day table has this program ahead on planted tiles; the corpus says
    # the stronger side holds fewer.
    chosen = prompt.selected(
        store(tmp_path, ("planted", 1, 0.05)), result({"v54": 0.3}).states
    )

    assert [claim.form.quantity for _, _, claim in chosen] == ["planted"]


def test_the_claim_section_disappears_when_there_is_nothing_to_say() -> None:
    """A heading over an empty list is noise in a message read every round.

    The live store decides this one, so it asserts the shape of the message
    rather than a particular claim: either the section is there with its table
    under it, or it is not there at all.
    """
    text = prompt.compose(
        "champion_1",
        result({"v54": 0.3}, days=30),
        [],
        [],
        IMPROVE,
        table("champion_1", {"v54": 0.3}),
    )

    heading = "## What the strongest agents on the ladder do differently"
    if heading in text:
        assert text.count("| what the corpus says |") == 1
    else:
        assert "what the corpus says" not in text
