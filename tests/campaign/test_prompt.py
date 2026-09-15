"""What the composed message says, and what it must never say."""

import re
from pathlib import Path

import pytest

from kaggriculture.campaign import (
    archive,
    config,
    dataset,
    evaluator,
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


def played(days: list[harness.Day], seat: int = 0) -> harness.Game:
    """One game carrying those days, finishing where its last day left off."""
    return harness.Game(
        opponent="v54",
        seed=101,
        seat=seat,
        ours=days[-1].ours_bank,
        theirs=days[-1].theirs_bank,
        worst_step_seconds=0.0,
        days=days,
    )


def games(banks: list[float], days: int) -> list[harness.Game]:
    """One recorded game per bank in ``banks``, each ``days`` days long."""
    return [played([day(n, bank, 3000.0) for n in range(days)]) for bank in banks]


def result(
    rates: dict[str, float], days: int = 2, seasons: int = 1
) -> evaluator.Result:
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
            name: [
                played([day(n, 3000.0 - n, 3000.0 + n) for n in range(days)])
                for _ in range(seasons)
            ]
            for name in rates
        },
    )


def first_game(scored: evaluator.Result) -> tuple[int, int, harness.Game]:
    """The first scored game of an evaluation, as `compose` now takes it.

    A round is feedback on one game, so `compose` is given that game rather
    than the whole evaluation it came out of. Matchup and season are what the
    loop numbers them, and the database keys its episodes by.
    """
    opponent = next(iter(scored.states))
    return 1, 1, scored.states[opponent][0]


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
        first_game(result({"v54": 0.0})),
        0.316,
        [],
        [],
        IMPROVE,
    )


def test_the_message_carries_the_rules_and_its_own_play() -> None:
    """A round is shown the rules and its own games, and nothing read off others.

    The prompt used to carry a build order mined from the public replay corpus:
    what the top-rated agents hold on each day. Measured 2026-09-10, clustered
    to one opening the median candidate scored 0.275 over 68 gates and
    promotions ran about one an hour; supplied as the orders those agents send,
    the median fell to 0.026 over 156 gates and nothing promoted in ten hours.
    The ceiling hardly moved, 0.940 to 0.914, so good programs did not get
    worse -- most programs became broken. What a model does with another
    strategy's schedule is bolt it on and break the economy underneath.

    The file and the constant that named it are gone, so there is nothing left
    to assert the absence of: this checks what the message *does* carry, which
    is the rules and the program's own play. Everything numeric in it is
    measured off this program's own games.
    """
    text = prompt.compose(
        "champion_1",
        first_game(result({"v54": 0.0}, days=30)),
        0.316,
        [],
        [],
        IMPROVE,
    )

    assert prompt.TASK_PROMPT.read_text(encoding="utf-8").strip()[:80] in text
    assert prompt.GAMES in text
    assert "Episode `champion_1m1s1`" in text
    # Aggregate only, still: no opponent is named and no path of theirs appears.
    assert "/data" not in text


def test_the_message_carries_no_path_at_all() -> None:
    """The doctrine: nothing the loop composes carries an opponent's path.

    Everything a model is given is this string, so this is the whole of the
    campaign's exposure. Names no longer travel either -- see
    `test_no_opponent_is_named_anywhere_in_the_message` -- so what is left to
    check here is that nothing which could be opened reaches a round.
    """
    text = prompt.compose(
        "champion_1",
        first_game(result({"v54": 0.1, "router_v1": 0.0})),
        0.316,
        [],
        [],
        IMPROVE,
    )

    assert "/data/kaggriculture" not in text
    assert not re.search(r"/(?:home|data|Users|tmp)/\S*", text)
    assert str(config.ROOT) not in text


def test_the_message_states_the_imports_the_gate_actually_allows() -> None:
    """A model told it may import our package would write a program that dies.

    One file ships, so the whitelist is the program's whole dependency
    surface. The section is rendered from `validate.ALLOWED_IMPORTS` rather
    than restated, because a model told a different set from the one that
    rejects it is worse than one told nothing.
    """
    text = prompt.compose(
        "champion_1",
        first_game(result({"v54": 0.5})),
        0.316,
        [],
        [],
        IMPROVE,
    )

    for name in validate.ALLOWED_IMPORTS:
        assert f"`{name}`" in text, name
    assert "`ctypes`" not in text
    assert "`kaggriculture`" not in text


def test_the_message_carries_the_rules_and_nothing_to_run() -> None:
    """The game's rules travel; the harness section does not, having nothing to run."""
    text = prompt.compose(
        "champion_1",
        first_game(result({"v54": 0.5})),
        0.316,
        [],
        [],
        IMPROVE,
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
    # One instruction, not a set to draw from. That is the whole invariant,
    # and it is structural: wording is an editorial choice and a test that
    # pins it breaks on every rewrite while catching nothing.
    assert isinstance(prompt.INSTRUCTION, str)
    assert prompt.INSTRUCTION.strip()
    assert not isinstance(prompt.INSTRUCTION_NAME, (list, tuple, set, dict))
    # And it reaches the round whole, since a truncated instruction is an
    # instruction to do something else.
    text = prompt.compose(
        "champion_1",
        first_game(result({"v54": 0.0}, days=30)),
        0.316,
        [],
        [],
        prompt.INSTRUCTION,
    )
    assert prompt.INSTRUCTION in text


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
        first_game(result({"v54": 0.5})),
        0.316,
        failures,
        [],
        IMPROVE,
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
        first_game(result({"v54": 0.5})),
        0.316,
        [],
        [],
        IMPROVE,
    )

    assert "produced nothing" not in text


def test_the_instruction_reaches_the_message_whole() -> None:
    """It is the last thing said, and it arrives uncut."""
    message = prompt.compose(
        "champion_1",
        first_game(result({"v54": 0.5})),
        0.316,
        [],
        [],
        prompt.INSTRUCTION,
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


def test_a_program_nothing_has_been_made_of_gets_no_section(tmp_path: Path) -> None:
    """A heading over an empty table is noise in a message read every round.

    The string this asserted on until 2026-09-13 was "already been made",
    which no version of the section has ever rendered, so it held whatever the
    code did. It asserts on the heading `_tried_lines` actually writes now.
    """
    db = archive.Database(tmp_path / "db.jsonl", tmp_path / "programs")

    text = prompt.compose(
        "champion_1",
        first_game(result({"v54": 0.5})),
        0.316,
        [],
        db.children("champion_1"),
        IMPROVE,
    )

    assert "Edits already tried" not in text
    assert "tried_1.py" not in text


def test_the_edits_already_tried_are_named_and_scored(tmp_path: Path) -> None:
    """A round is told which way its predecessors moved, and by how much.

    A codex call remembers nothing of the ones before it, so a session's twelve
    consecutive attempts on one opponent were twelve independent guesses. On
    2026-09-13 that cost four of a run's thirteen rounds: one session wrote a
    program scoring 0.000 and spent three more rounds editing that, because
    nothing ever told a round what its last change did.

    The score is in the message and the program itself is a file, so the round
    can diff whichever one its own question is about.
    """
    db = archive.Database(tmp_path / "db.jsonl", tmp_path / "programs")
    stored(db, "p_better", "champion_1", 0.340, "nudged the opening")
    stored(db, "p_worse", "champion_1", 0.130, "rewrote the planner")

    text = prompt.compose(
        "champion_1",
        first_game(result({"v54": 0.5})),
        0.316,
        [],
        db.children("champion_1"),
        IMPROVE,
    )

    assert "## Edits already tried on `champion_1`" in text
    assert "`child.py` wins 0.316 of its games" in text
    # Best first, which is `children`'s own order, and each delta is against
    # the program in `child.py` rather than against the one above it.
    assert "- `tried_1.py` scored 0.340 (+0.024)" in text
    assert "- `tried_2.py` scored 0.130 (-0.186)" in text


def test_only_the_first_few_edits_are_sent(tmp_path: Path) -> None:
    """Eight sessions edit one champion, so the list needs a cut.

    Every program written from the champion by any session is a sibling, and
    `RECENT_ATTEMPTS` of them reach the message -- the best, since that is the
    order `children` returns and the direction that came closest is what a
    round can act on.
    """
    db = archive.Database(tmp_path / "db.jsonl", tmp_path / "programs")
    for number in range(prompt.RECENT_ATTEMPTS + 2):
        stored(db, f"p{number}", "champion_1", 0.30 - number / 100, f"try {number}")

    text = prompt.compose(
        "champion_1",
        first_game(result({"v54": 0.5})),
        0.316,
        [],
        db.children("champion_1"),
        IMPROVE,
    )

    assert f"`tried_{prompt.RECENT_ATTEMPTS}.py`" in text
    assert f"`tried_{prompt.RECENT_ATTEMPTS + 1}.py`" not in text


def test_no_opponent_is_named_anywhere_in_the_message() -> None:
    """The round is writing a program to beat any opponent, not these ones.

    "Beat this pool" is a fitting objective and "beat any opponent" is a
    generalising one, and the campaign spent weeks optimising the first while
    measuring the second. Left to it, the previous lineage evolved opponent
    fingerprinting -- recognising specific agents by their sheep and cow counts
    -- which is the correct solution to the objective it was actually given,
    and worth nothing on a ladder where the agent across the table is one it
    has never seen.

    A name is all it takes to start fitting, so no opponent's travels. The pool
    is a sample of the field, and the sections that used to rank it, place this
    program in it, and label each day table with who was across the table are
    gone.

    The program's own id does travel now, inside the episode key of the game
    the round is working on -- `<id>m<matchup>s<season>` is how the games
    database addresses it, so a round cannot read its own game without it. That
    is its own id and never an opponent's. It is a real cost only where the two
    namespaces meet: a champion is also a pool opponent, so a round starting
    from one is told that champion's pool name. What it learns there is the name
    of itself.
    """
    rates = {"v54": 0.3, "shopforge": 0.5, "router_v1": 0.9}
    text = prompt.compose(
        "champion_1",
        first_game(result(rates, days=30)),
        0.316,
        [],
        [],
        IMPROVE,
    )

    for opponent in rates:
        assert opponent not in text, f"{opponent} reached the round by name"
    # Its own id appears only inside the episode key, and nowhere else.
    assert "champion_1m1s1" in text
    assert "champion_1" not in text.replace("champion_1m1s1", "")


def test_the_templates_own_note_never_reaches_the_model() -> None:
    """The header explains the file to a reader and would render twice."""
    text = prompt.compose(
        "champion_1",
        first_game(result({"v54": 0.5}, days=30)),
        0.316,
        [],
        [],
        IMPROVE,
    )

    assert prompt.ROUND_PROMPT.read_text(encoding="utf-8").startswith("<!--")
    assert "<!--" not in text
    assert "the parts the campaign" not in text


def test_the_message_points_at_the_database_rather_than_carrying_it() -> None:
    """The index travels; the games do not, and neither does a copy of them.

    There is one store, and a round queries it. What the message owes it is
    the part it cannot work out: which matchups its last evaluation was
    against, how many seasons each holds, and how they went.
    """
    rates = {"close": 0.5, "beaten": 0.1}
    text = prompt.compose(
        "champion_1", first_game(result(rates, days=4)), 0.316, [], [], IMPROVE
    )

    assert prompt.GAMES in text, "the round is not told where to ask"
    assert "query-games" in text, "the skill that has the schema is not named"
    assert "Episode `champion_1m1s1`" in text
    # The day tables themselves stay out: they are rows, not message text.
    assert "ours_bank" not in text
    assert "ours_quadrants" not in text


def test_the_round_is_told_the_rest_of_the_database_is_there() -> None:
    """Its own games and the recorded ones are the same rows in one store.

    Stated as a fact and nothing more. The message used to add that the rest
    was "there if a question wants them and ignorable if not", which is the
    message telling a round how to work; it says what exists and leaves the
    use of it alone.
    """
    rates = {"close": 0.5}
    text = prompt.compose(
        "champion_1", first_game(result(rates, days=4)), 0.316, [], [], IMPROVE
    )

    assert "the competition has recorded" in text
    assert prompt.GAMES in text


COMMITTED = """
def agent(observation, configuration=None):
    day = observation["day"]
    if day == 3:
        return {"farmer": ["PLANT"], "hands": [], "market": []}
    if day == 3 and observation["hour"] > 4:
        return {"farmer": ["WATER"], "hands": [], "market": []}
    if day >= 29:
        return {"farmer": ["SELL"], "hands": [], "market": []}
    if observation["cash"] > 500:
        return {"farmer": ["BUY"], "hands": [], "market": []}
    return {"farmer": ["PASS"], "hands": [], "market": []}
"""


def test_the_days_a_program_has_decided_are_counted_and_the_rest_named() -> None:
    """What a round is shown about its own commitment.

    A decision taken at a fixed day is a plan whether it is written as a table
    or as `if day == 3`, and champion_15 put 32 of its 57 such conditions on
    day 29 -- an endgame, not a season. The count is what lets a round see that
    about itself; the days it names none of are the same fact from the other
    side, and they are what the message actually says out loud.

    The cash condition is here to be ignored: a comparison that names no day
    is not a commitment to one, however many numbers it holds.
    """
    counted = prompt.schedule(COMMITTED)

    assert counted == {3: 2, 29: 1}, counted
    shown = "\n".join(prompt._schedule_lines(COMMITTED))
    assert "day 3: 2" in shown and "day 29: 1" in shown
    assert "500" not in shown, "a cash threshold was counted as a day"
    # Every day it never names, so the silence is legible rather than implied.
    for day in (0, 1, 2, 4, 28):
        assert f"{day}" in shown.split("It names no day at")[1]


def test_a_program_that_will_not_parse_is_shown_no_schedule() -> None:
    """The message is composed before the gate rejects a broken edit.

    `compose` runs on whatever the last round left in `child.py`, and a round
    that wrote something unparseable must still get a message rather than take
    the session down with a `SyntaxError` from the part that describes it.
    """
    assert prompt.schedule("def agent(o, c=None):\n    return {") == {}
    assert prompt._schedule_lines("this is not python(") == []
