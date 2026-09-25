"""What the composed message says, and what it must never say."""

import re
from pathlib import Path

import pytest

from kaggriculture.campaign import (
    config,
    dataset,
    evaluator,
    harness,
    plan,
    prompt,
)


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
    )

    assert "## Rules" in text and "### Market" in text
    assert prompt.GAMES in text
    assert "Episode `champion_1m1s1`" in text
    # Aggregate only, still: no opponent is named and no path of theirs appears.
    assert str(config.ROOT) not in text


def test_the_message_carries_no_path_into_the_campaign_itself() -> None:
    """The half of the doctrine that survived 2026-09-15.

    Opponents opened that day: they are published kernels under Apache-2.0,
    the field derives from them in the open, and this round's opponent is
    copied into the box as `opponent.py`. What stays shut is this campaign's
    own tree -- the archive, the champions, the pool file, the run. A round
    edits a copy in a directory of
    its own on purpose, and a path into `config.ROOT` is a round that can edit
    the record of what every other round did.
    """
    text = prompt.compose(
        "champion_1",
        first_game(result({"v54": 0.1, "router_v1": 0.0})),
    )

    assert str(config.ROOT) not in text
    assert "/data/kaggriculture/campaign" not in text, "the pool and run are ours"
    assert not re.search(r"/(?:home|Users|tmp)/\S*", text)
    # And the one path it is now meant to carry.
    # The opponent travels as a file in the box, so no path of its own either.
    assert "/data/kaggriculture/opponents" not in text


def test_the_message_carries_the_rules_and_nothing_to_run() -> None:
    """The game's rules travel; the harness section does not, having nothing to run."""
    text = prompt.compose(
        "champion_1",
        first_game(result({"v54": 0.5})),
    )

    assert "Kaggriculture policy task" in text
    assert "campaign play AGENT" not in text and "campaign check" not in text
    assert "uv run" not in text and "--vs" not in text
    assert "engine/kaggriculture.py" not in text
    assert "700000" not in text


def test_the_round_is_told_the_whole_campaign_is_a_file_beside_it() -> None:
    """`attempts.jsonl` is named, so a round knows the record is there to grep.

    Every program the campaign has written, one JSON object per line: what it
    changed, what it scored, whether it survived a gate, and its rate against
    each opponent. A file rather than more message, because four hundred
    attempts would not fit in a round's budget and a round pays for what it
    reads -- and because worked examples in the prompt become the subject where
    a file answers the question the round brought to it.
    """
    text = prompt.compose(
        "champion_1",
        first_game(result({"v54": 0.5})),
    )

    assert "attempts.jsonl" in text
    assert "one JSON object per line" in text


def test_only_this_rounds_opponent_is_named() -> None:
    """The round is writing a program to beat any opponent, not these ones.

    "Beat this pool" is a fitting objective and "beat any opponent" is a
    generalising one, and the campaign spent weeks optimising the first while
    measuring the second. Left to it, the previous lineage evolved opponent
    fingerprinting -- recognising specific agents by their sheep and cow counts
    -- which is the correct solution to the objective it was actually given,
    and worth nothing on a ladder where the agent across the table is one it
    has never seen.

    A name is all it takes to start fitting, so the pool does not travel: the
    sections that used to rank it, place this program in it, and label each
    day table with who was across the table are gone. What does travel is the
    one opponent this round is on, because its program is in the directory and
    the round has to be able to ask the record about it.

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
    )

    # The game's own opponent, and only that one.
    assert "The opponent is `v54`" in text
    for opponent in ("shopforge", "router_v1"):
        assert opponent not in text, f"{opponent} reached the round by name"
    # Its own id appears only inside the episode key, and nowhere else.
    assert "champion_1m1s1" in text
    assert "champion_1" not in text.replace("champion_1m1s1", "")


def test_the_templates_own_note_never_reaches_the_model() -> None:
    """The header explains the file to a reader and would render twice."""
    text = prompt.compose(
        "champion_1",
        first_game(result({"v54": 0.5}, days=30)),
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
        "champion_1",
        first_game(result(rates, days=4)),
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
        "champion_1",
        first_game(result(rates, days=4)),
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


def test_there_is_one_instruction_and_it_names_the_plan() -> None:
    """One instruction, in the template, and it names the file to change.

    There were five drawn per session. Two of them told a round to replace
    the program with something else, which cost 54% of every call the
    campaign made and returned 476 programs of which one scored above nought.
    And it has to name the file: 215 rounds were filed under "plan" while the
    instruction said only to fix "this program", and every one of them read
    that as the controller.
    """
    sections = prompt.ROUND.text.split("## Your instruction")
    assert len(sections) == 2, "one instruction, not a set to draw from"
    instruction = (
        sections[1].replace("{note}", "").replace("{episode}", "champion_1m1s1")
    )
    assert plan.PLAN_FILE in instruction
    # And it reaches the round whole, at the end, since a truncated
    # instruction is an instruction to do something else.
    text = prompt.compose("champion_1", first_game(result({"v54": 0.0}, days=30)))
    assert text.rstrip().endswith(instruction.rstrip())


def test_the_stagnation_note_leads_the_instruction() -> None:
    """The one thing the instruction says differently between rounds.

    A session that did not start from the champion is told so first, and a
    session that did is told nothing extra: no blank line, no heading over an
    empty note.
    """
    plain = prompt.compose("champion_1", first_game(result({"v54": 0.0})))
    noted = prompt.compose(
        "champion_1",
        first_game(result({"v54": 0.0})),
        "This session did not start from the champion.\n\n",
    )

    assert "## Your instruction\n\nWiden" in plain
    assert (
        "## Your instruction\n\nThis session did not start from the champion.\n\nWiden"
        in noted
    )
