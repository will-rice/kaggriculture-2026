"""What the composed message says, and what it must never say."""

import csv
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


def test_the_message_carries_no_corpus_derived_target() -> None:
    """A round is shown the rules and its own play, and nothing read off others.

    The prompt used to carry two things mined from the public replay corpus:
    the build order, and the settled claims about what the ladder's winners
    hold. Both were true about the corpus, neither was ever shown to help a
    program, and the last version of the first measurably hurt.

    Measured 2026-09-10. With the build order clustered to one opening the
    median candidate scored 0.275 over 68 gates and promotions ran about one an
    hour. With the opening added as the orders those agents send, the median
    fell to 0.026 over 156 gates and nothing promoted in ten hours. The ceiling
    hardly moved -- 0.940 to 0.914 -- so good programs did not get worse, most
    programs became broken.

    What the model did with a build order is what three measured experiments
    did before it: bolt another strategy's orders onto this one and break the
    economy underneath. Handing a round a build order is an invitation to
    graft, and grafting is the thing that fails.
    """
    text = prompt.compose(
        "champion_1",
        result({"v54": 0.0}, days=30),
        [],
        [],
        IMPROVE,
        table("champion_1", {"v54": 0.0}),
    )

    # The invariant is that neither corpus artifact's *content* travels, which
    # is checked against the artifacts themselves rather than against headings
    # somebody chose. A heading can be renamed; the file either reaches the
    # round or it does not.
    build_order = prompt.BUILD_ORDER.read_text(encoding="utf-8").strip()
    assert build_order
    assert build_order not in text
    for line in build_order.splitlines():
        if line.startswith("| ") and len(line) > 40:
            assert line not in text, "a row of the build order reached the round"
            break
    # Its own play stays: the index of the seasons it was measured on, and
    # the rules.
    assert prompt.TASK_PROMPT.read_text(encoding="utf-8").strip()[:80] in text
    assert f"`{prompt.SEASONS}`" in text
    assert "| season | finished |" in text
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
        result({"v54": 0.1, "router_v1": 0.0}),
        [],
        [],
        IMPROVE,
        table("champion_1", {"v54": 0.5}),
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
        result({"v54": 0.0}, days=30),
        [],
        [],
        prompt.INSTRUCTION,
        table("champion_1", {"v54": 0.0}),
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

    assert [gap.claim.form.quantity for gap in chosen] == ["bank", "planted"]
    # Both opponents, and the row says the count rather than the share.
    assert [(gap.wrong, gap.seen) for gap in chosen] == [(2, 2), (2, 2)]
    # And each carries the figures behind it, so a round is not told it is
    # behind on something it can find no number for.
    assert chosen[0].ours == 2999.0 and chosen[0].theirs == 3001.0


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

    assert {gap.claim.form.quantity for gap in chosen} == {
        "quadrants",
        "fertilised",
    }
    # One quadrant against three, and the row says so.
    quads = next(g for g in chosen if g.claim.form.quantity == "quadrants")
    assert (quads.ours, quads.theirs) == (1.0, 3.0)


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

    assert [gap.claim.form.quantity for gap in chosen] == ["planted"]


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


def test_no_opponent_is_named_anywhere_in_the_message() -> None:
    """The round is writing a program to beat any opponent, not these ones.

    "Beat this pool" is a fitting objective and "beat any opponent" is a
    generalising one, and the campaign spent weeks optimising the first while
    measuring the second. Left to it, the previous lineage evolved opponent
    fingerprinting -- recognising specific agents by their sheep and cow counts
    -- which is the correct solution to the objective it was actually given,
    and worth nothing on a ladder where the agent across the table is one it
    has never seen.

    A name is all it takes to start fitting, so no name travels. The pool is a
    sample of the field, and the sections that used to rank it, place this
    program in it, and label each day table with who was across the table are
    gone.
    """
    rates = {"v54": 0.3, "shopforge": 0.5, "router_v1": 0.9}
    text = prompt.compose(
        "champion_1",
        result(rates, days=30),
        [],
        [],
        IMPROVE,
        table("champion_1", rates),
    )

    for opponent in rates:
        assert opponent not in text, f"{opponent} reached the round by name"
    # Nor the program's own name, which is a pool name in the same namespace.
    assert "champion_1" not in text


def test_the_templates_own_note_never_reaches_the_model() -> None:
    """The header explains the file to a reader and would render twice."""
    text = prompt.compose(
        "champion_1",
        result({"v54": 0.5}, days=30),
        [],
        [],
        IMPROVE,
        table("c", {"v54": 0.5}),
    )

    assert prompt.ROUND_PROMPT.read_text(encoding="utf-8").startswith("<!--")
    assert "<!--" not in text
    assert "the parts the campaign" not in text


def test_every_season_reaches_the_file_at_full_width() -> None:
    """The file is the evidence; the message is the index of it.

    A markdown table can be read at about fifteen columns, and a `Day` carries
    sixty-odd. So the rendered version had to choose, and what it chose cost
    the campaign real evidence: quadrants and fertilizer -- the two sharpest
    separations in 16,292 recorded games, at 99% and 100% of their paired
    samples -- were measured, stored, and shown to nobody for want of room.
    Written to a file there is no room to run out of.
    """
    rates = {"v54": 0.3, "shopforge": 0.5}
    rendered = prompt.seasons(result(rates, days=30))
    header, *rows = rendered.splitlines()
    columns = header.split(",")

    assert len(rows) == 60, "both seasons, every day of each"
    for column in ("ours_quadrants", "theirs_fertilised", "ours_sell_orders"):
        assert column in columns
    # Both sides of every quantity the campaign measures, not a chosen few.
    for measure in dataset.COLUMNS:
        assert f"ours_{measure}" in columns
        assert f"theirs_{measure}" in columns
    # The breakdowns behind the totals, and the shared market.
    assert "ours_plants_WHEAT" in columns
    assert "theirs_animals_COW" in columns
    assert "price_WHEAT" in columns


def test_season_one_is_the_game_with_the_most_left_to_learn_from() -> None:
    """The number a round is given instead of a name still has to mean something.

    Worst-beaten opponent first, because a loss is a game lost and not a
    matchup lost: an opponent beaten 0.875 took one game in eight and those
    are the games that decide whether the program finishes top. A round
    reading only season 1 is reading the one where the change is reachable, so
    the order is load-bearing rather than presentational.
    """
    rates = {"swept": 1.0, "close": 0.5, "beaten": 0.1}
    banks = {"swept": 9000.0, "close": 5000.0, "beaten": 1000.0}
    played = result(rates, days=4).model_copy(
        update={
            "states": {
                name: [day(n, bank, 3000.0) for n in range(4)]
                for name, bank in banks.items()
            }
        }
    )

    reader = list(csv.DictReader(prompt.seasons(played).splitlines()))

    assert [row["season"] for row in reader] == ["1"] * 4 + ["2"] * 4 + ["3"] * 4
    assert [row["day"] for row in reader[:4]] == ["0", "1", "2", "3"]
    # Rate ascending: the one it hardly ever beats, then the split, then the
    # one it swept -- which is in the file too, at the bottom.
    assert [reader[index]["ours_bank"] for index in (0, 4, 8)] == [
        "1000.0",
        "5000.0",
        "9000.0",
    ]


def test_the_index_names_a_season_by_how_it_finished() -> None:
    """What a round cannot work out for itself is which season is which."""
    rates = {"close": 0.5, "beaten": 0.1}
    banks = {"close": 2900.0, "beaten": 1000.0}
    played = result(rates, days=4).model_copy(
        update={
            "states": {
                name: [day(n, bank, 3000.0) for n in range(4)]
                for name, bank in banks.items()
            }
        }
    )

    text = prompt.compose("champion_1", played, [], [], IMPROVE, table("c", rates))

    assert f"`{prompt.SEASONS}`" in text
    assert "| 1 | -2,000 |" in text
    assert "| 2 | -100 |" in text


def test_the_message_does_not_grow_with_the_seasons_played() -> None:
    """Twelve opponents cost twelve lines, not twelve tables.

    This is the whole reason the games moved out of the message. Rendered, a
    game was 30 rows of fifteen columns, and six of them were 74% of a message
    that is otherwise the rules, the program and what to do -- so the size was
    capped by dropping games, which is capping the evidence. The index is one
    line each, so nothing has to be dropped to keep the message small.
    """
    one = {"v54": 0.3}
    many = {f"other_{index}": 0.3 for index in range(12)}

    small = prompt.compose(
        "champion_1", result(one, days=30), [], [], IMPROVE, table("c", one)
    )
    large = prompt.compose(
        "champion_1", result(many, days=30), [], [], IMPROVE, table("c", many)
    )

    assert len(large) - len(small) < 200
    # And the games are all still there, in the file.
    assert len(prompt.seasons(result(many, days=30)).splitlines()) == 1 + 12 * 30


def test_a_program_that_played_nothing_gets_no_seasons_section() -> None:
    """A heading over an empty index is noise in a message read every round."""
    empty = result({"v54": 0.3}).model_copy(update={"states": {}})

    text = prompt.compose(
        "champion_1", empty, [], [], IMPROVE, table("c", {"v54": 0.3})
    )

    assert "## The seasons" not in text
    assert prompt.seasons(empty) == ""
