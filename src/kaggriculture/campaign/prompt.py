"""Compose the one message a codex call is given.

The model is a mutation operator: it is handed a directory holding one file,
``child.py``, and everything else arrives on standard input as this message.
It edits that file and stops. The loop plays every game, so there is nothing
here about running a harness, no engine to read and no workspace to manage --
and no file we assemble that an opponent's path could leak through.

The message is four parts: the game's rules, the program and how to work on
it, an index of the seasons it just played, and the instruction. One function
composes it and every round is composed by it, the first included, so the
model never sees a round shaped differently from the others.

The seasons are not in the message. They are written beside `child.py` as a
CSV of every game the evaluation scored, at a width no message could carry,
and what travels here is the index: which matchup is which, how many seasons
each holds, and how they went.

Nothing mined from the public replay corpus reaches a round any more. The
build order and the settled claims about the ladder's winners were both true
about the corpus and neither earned its place -- measured 2026-09-10, adding
the opening as orders took the median candidate from 0.275 to 0.026 and
stopped promotions for ten hours. What a model did with them is what three
separate experiments did: bolt another strategy's orders onto this one and
break the economy underneath.
"""

import logging
import re
from pathlib import Path

from kaggriculture.campaign import (
    archive,
    evaluator,
    harness,
    validate,
)

LOGGER = logging.getLogger(__name__)

TASK_PROMPT = Path(__file__).with_name("task_prompt.md")
# The whole message, in order, with the measured parts left as placeholders.
# It is a file rather than a pile of string constants so that the set-up can
# be read end to end -- what a round is told, and in what order -- without
# reconstructing it from `compose`.
ROUND_PROMPT = Path(__file__).with_name("round_prompt.md")
# What the strongest agents hold on each day, written by `build-order` over
# the extracted corpus. It is here because the message already gives a round
# its own banks and tiles each day and gives it nothing to read them against
# -- and because the agents at the top of the leaderboard publish no kernels,
# so their games are the only view of them there is.
#
# It replaced `winning_pace.md`, which took medians over the winning side of
# every game. That is the wrong half of the corpus: about half of a ladder's
# winners are the weaker agent having a good day, and eleven quantities
# measured that way came back between 45% and 60%. A snapshot of a moving
# field either way -- rebuilt nightly, because the ladder turns over.
BUILD_ORDER = Path(__file__).with_name("build_order.md")

# Rendered from the gate's own whitelist, so the model is never told a
# different set from the one that rejects it. One file ships, so this list is
# the program's whole dependency surface.
IMPORTS = ", ".join(f"`{name}`" for name in sorted(validate.ALLOWED_IMPORTS))


# How many of a lineage's failures are sent, and how much of each. The last
# few are what a next attempt can act on; an older one is about a program two
# rounds back. A reason is free-form -- a codex call's own last message is one
# -- so it is collapsed onto a line and cut, because the message has a budget
# and the parser error or the exception that matters comes first.
RECENT_FAILURES = 3
REASON_CHARS = 200

# The name of the file the seasons are written to, in the same directory as
# `child.py`. They used to be rendered into the message: six games at 30 rows
# of fifteen columns, 74% of a message that is otherwise the rules, the
# program and what to do. Bounding the message therefore meant dropping games,
# which is bounding the evidence -- and the evaluation had already thrown away
# thirty-one of every thirty-two games before the message was even composed.
#
# A file bounds neither. Every game the campaign scored is in it, at full
# width: all 29 measures `dataset.measures` defines, for both sides, where the
# table had room for nine. The message carries an index instead -- one line per
# matchup -- so a round that wants day twelve of the season it nearly won can
# read day twelve, or ask which day the banks diverged across every game it
# played, and a round with a different question pays nothing for the answer to
# this one.
SEASONS = "seasons.csv"


CHANGE_CHARS = 160

# The instruction, and there is one. It says the bar the gate actually applies
# -- finish top of the standings -- rather than naming a way to go about it.
#
# There were five, FAMOU appendix C.2's rewrites, drawn one per session on the
# theory that eight workers starting from one champion would otherwise explore
# in one direction. Measured over the 476 programs of the first router-seeded
# run, that is not what they bought:
#
#     different      134 programs   mean fitness 0.000   best 0.000
#     inspired       125            mean 0.013           best 0.911
#     restructure     84            mean 0.844           best 0.940
#     improve         70            mean 0.825           best 0.969
#     tune            62            mean 0.883           best 0.964
#
# Every one of the 134 `different` programs scored exactly nought, and
# `inspired` landed once in 125. Together they are 54% of every call the
# campaign made. The cause is the seed: "replace it with a completely
# different algorithm" costs nothing against a thirty-line skeleton and means
# deleting a rated agent when the program is a published one, and a farm bot
# written from scratch loses every game to this pool. The three that survived
# are within 0.06 of each other, which is three ways of saying the same thing.
#
# What varies between sessions is the program they start from and what the
# siblings section says has already been tried from it. That was always the
# real source of spread; the draw was noise on top of it.
INSTRUCTION = (
    "Change `child.py` so that it finishes every season with a larger bank "
    "than it did. Not a better place in a table -- a bigger margin in the "
    "games themselves, and most of all in the ones it already wins narrowly. "
    f"Every game it played is in `{SEASONS}`, day by day, and every one of "
    "them was scored: find where this program left money on the field and "
    "take it. Small, local changes are welcome, and so is replacing whatever "
    "part of it is playing badly."
)
# Margin rather than rank, and the reason is a measurement rather than a
# preference.
#
# The gate has stopped separating anything. Fourteen of champion_69's
# twenty-four opponents are saturated and every one of them is ours -- a
# candidate beats the whole lineage almost always -- so "finish top of the
# standings" is a step function over a table with no gradient left in it. Every
# one of those saturated games is still a season of 719 decisions, and some of
# them are bad ones; summarising the season to a win throws that away.
#
# Margin is dense where rank is sparse. A program can always win by more, and
# the day tables it is shown are seasons rather than verdicts.
#
# The competition does not score margin -- it is relative bank, and the size of
# the win never counts -- which is exactly why this is the *instruction* and
# not the bar. Promotion still runs on a Bradley-Terry fit that is blind to
# margin by design, so a program that wins bigger and no more often gains
# nothing at the gate. The shaping steers the search; it does not decide it.
INSTRUCTION_NAME = "margin"


class Message:
    """A prompt template, and the only thing that decides what fills it.

    `str.format` drifts in two directions and only complains about one: it
    raises on a placeholder the caller did not supply, and silently keeps one
    the template stopped using. Naming the fields somewhere else would make a
    third thing to keep in step, so they are read out of the template -- which
    makes the file the single source of truth for its own shape, and makes a
    mismatch a `ValueError` naming both sides rather than a `KeyError` naming
    one.

    Loaded and checked at import, so a template nobody can fill fails before
    the run opens rather than mid-round with a codex call already spent.

    Attributes:
        fields: The placeholder names the template uses.
    """

    # A single-braced lowercase name. Doubled braces are literal braces, which
    # markdown examples may legitimately contain.
    PLACEHOLDER = re.compile(r"(?<!\{)\{([a-z_]+)\}(?!\})")

    def __init__(self, path: Path) -> None:
        """Load ``path``, dropping the note that explains it.

        That note is for whoever reads the file, not for the model -- and it
        names the placeholders it documents, so leaving it in would have
        `format` substitute them and render every section of the message
        twice.

        Args:
            path: The markdown template.
        """
        text = path.read_text(encoding="utf-8")
        if text.startswith("<!--"):
            text = text[text.index("-->") + len("-->") :].lstrip("\n")
        self.text = text
        self.fields = frozenset(self.PLACEHOLDER.findall(text))

    def render(self, **values: object) -> str:
        """Fill every placeholder, or say exactly which side is out of step.

        Raises:
            ValueError: The values and the template disagree about the fields.
        """
        given = frozenset(values)
        if given != self.fields:
            raise ValueError(
                f"{self.__class__.__name__} wants "
                f"{sorted(self.fields - given) or 'nothing more'} and was given "
                f"{sorted(given - self.fields) or 'nothing extra'}"
            )
        return self.text.format(**values)


ROUND = Message(ROUND_PROMPT)


def seasons(result: evaluator.Result) -> str:
    """Every game of one evaluation as a single CSV, day by day.

    One row per day, keyed by two numbers. `matchup` is one opponent, whose
    name is the whole of what a round is not told about who it played;
    `season` is one game inside that matchup. Every game the evaluation
    scored is here, because a round can only improve a game it is shown, and
    the campaign scores all of them.

    The two keys are what makes a comparison mean something. Within one
    matchup the opponent is fixed and what changes between seasons is the
    world -- the map, the prices, the seat -- which is the axis a general
    program has to hold up across. Across matchups the adversary changes too,
    so a difference between two of them says nothing about either.

    The width is the point of the file existing. A markdown table has room for
    about fifteen columns before it stops being readable, and `Day` carries
    sixty-odd: the 29 quantities `dataset.measures` defines for each side, the
    per-crop breakdowns behind four of those totals, and the market's prices.
    Written out, every one of them is a column a round can group by, diff
    across days, or ignore -- and the two sharpest separations in the whole
    corpus, quadrants on day three and fertilizer on day five, were quantities
    the table had no room for.

    Args:
        result: The evaluation, for its day tables and the rates that order
            them.

    Returns:
        The whole CSV, header first. Empty when nothing was recorded.
    """
    return harness.day_csv(
        [
            ({"matchup": matchup, "season": season}, days)
            for matchup, opponent in enumerate(_ordered(result), start=1)
            for season, days in enumerate(result.states[opponent], start=1)
        ]
    )


def _ordered(result: evaluator.Result) -> list[str]:
    """The matchups, worst-beaten opponent first.

    A loss is a game lost, not a matchup lost: an opponent beaten 0.875 took
    one game in eight, and those are the games that decide whether the program
    finishes top. So the order is by rate and then by margin, which puts the
    matchups with the most to learn from at the top of the index, and the ones
    it swept at the bottom rather than out of the file.
    """
    return sorted(
        (name for name in result.rates if result.states.get(name)),
        key=lambda name: (result.rates[name], result.margins[name].mean),
    )


def _states_lines(result: evaluator.Result) -> list[str]:
    """Render the index of the seasons file: one line per matchup.

    Not the games themselves. Every game the evaluation scored is in
    `SEASONS`, at a width no message could carry, and what belongs here is
    the part a round cannot work out for itself: how many seasons each matchup
    holds, how they went, and which one is worth opening first.
    """
    ordered = _ordered(result)
    if not ordered:
        return []
    lines = [
        "## The seasons it just played",
        "",
        f"Every game it played is in `{SEASONS}`, beside `child.py`: one row "
        "per day, keyed by `matchup` and `season`. A matchup is one opponent "
        "and every season it played against them; within it the opponent is "
        "fixed, so what changes from season to season is the world -- the "
        "map, the prices, the seat -- which is the variation a program has to "
        "hold up across. Between matchups the opponent changes too, so a "
        "difference there says nothing about either.",
        "",
        "Read it however suits the question: one season, one matchup, one day "
        "across every game, one column across all of them. Every game below "
        "was scored, so every one of them is a game to improve.",
        "",
        "The opponents are not named and it does not matter which they were. "
        "They are drawn from the field this program will meet, and the field "
        "turns over: the agent across the table in a scored game will be one "
        "this program has never seen. So these are samples of how a season "
        "can go against a competent opponent, not a list of agents to beat. A "
        "change that wins these seasons because it recognised who it was "
        "playing wins nothing that counts.",
        "",
        "| matchup | seasons | won | mean finish | worst | best |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for matchup, opponent in enumerate(ordered, start=1):
        finals = [
            days[-1].ours_bank - days[-1].theirs_bank
            for days in result.states[opponent]
        ]
        lines.append(
            f"| {matchup} | {len(finals)} | "
            f"{sum(1 for final in finals if final > 0)} | "
            f"{sum(finals) / len(finals):+,.0f} | "
            f"{min(finals):+,.0f} | {max(finals):+,.0f} |"
        )
    lines += [
        "",
        "Seasons are numbered narrowest first inside each matchup, so season "
        "1 is the game a small change would have turned and the last is the "
        "one furthest out of reach.",
        "",
        "Each row of the file is a day as it closed, at hour 23. Both sides "
        "are in it, the opponent's shed included: that is what the author of "
        "a program is shown afterwards, never what the program may read while "
        "it plays. `ours_*` and `theirs_*` carry every quantity the campaign "
        "measures -- banks, planted and ripe tiles, pens, weeds, bare tiles, "
        "unlocked quadrants, hands, seed and shed totals, shops, watering and "
        "feeding, fertilizer, plant age, and the running counts of every kind "
        "of market order -- with the per-crop breakdowns behind the totals as "
        "`ours_plants_WHEAT` and the like, and the shared market as `price_*`. "
        "An empty cell is a count of zero.",
    ]
    return lines


def _failure_lines(name: str, failures: list[archive.Failure]) -> list[str]:
    """Render the most recent rounds on this lineage that produced no program.

    A rejected round leaves nothing behind but its reason, and "your program
    did not parse" is exactly what a next attempt can act on. Only the last
    `RECENT_FAILURES` are sent: an older one is about a program the lineage
    has since moved past.

    Args:
        name: The program those rounds were editing, by name.
        failures: Every failure the ledger holds against it, oldest first.

    Returns:
        Lines of a markdown section, one bullet per failure.
    """
    lines = [
        f"## Recent attempts on `{name}` that produced nothing",
        "",
        "These were rejected before a single game was played, so no score came "
        "back from any of them and `child.py` is unchanged by them. They are "
        "mistakes to avoid, not results to build on.",
        "",
    ]
    for failure in failures[-RECENT_FAILURES:]:
        lines.append(f"- {' '.join(failure.reason.split())[:REASON_CHARS]}")
    return lines


def _behind(result: evaluator.Result) -> float:
    """Mean bank margin across every game, which is what the instruction moves.

    One number rather than a row per opponent. A rate against a named agent
    affords one action -- target that agent -- and that is the fitting this
    message exists not to encourage. This is the scale the instruction asks
    the program to move, and the season below is where it can be read.
    """
    if not result.margins:
        return 0.0
    return sum(margin.mean for margin in result.margins.values()) / len(result.margins)


def compose(
    name: str,
    result: evaluator.Result,
    failures: list[archive.Failure],
    siblings: list[archive.Program],
    instruction: str,
    standings: dict[str, float],
) -> str:
    """Compose the message for one round.

    Args:
        name: What the program in ``child.py`` is called -- a pool name or a
            database id. It is interpolated raw, so it must never be a path.
        result: The loop's fast evaluation of that program: the verdict, and
            the day table of one game behind it.
        failures: Every failure the ledger holds against that program, oldest
            first. The caller hands over what it has and this cuts it to the
            last few, so a caller cannot forget to.
        siblings: Programs already written from ``name`` and scored, best
            first. Cut to ``SIBLINGS`` here for the same reason.
        instruction: ``INSTRUCTION``, with any stagnation note the caller
            prepended.
        standings: Every agent's rating from the tournament this program's
            results are part of, itself included.

    Returns:
        The whole message, for codex's standard input.
    """
    # The standings still choose which game is worth showing -- the agent
    # directly above is the one whose game was closest -- but they are no
    # longer shown. A place is not something a round can act on: it cannot
    # choose its opponents or its rank, and everything it can act on is in the
    # per-opponent table. Worse, the ratings are mostly noise at this sample
    # size -- one unchanged agent's fitted rating moves with a standard
    # deviation of 0.745 across draws, where the whole table spans about five
    # -- so printing them to three decimals invited a round to reason about
    # differences a re-run would reshuffle. And the top of that table was this
    # lineage's own ancestry, which is the target the campaign spent a day
    # removing from the pool.
    # A lineage with nothing against it gets no section at all: a heading over
    # an empty list is noise in a message the model reads every round. The
    # template puts each on its own line, so an empty one leaves no gap.
    index = _states_lines(result)
    message = ROUND.render(
        task=TASK_PROMPT.read_text(encoding="utf-8").rstrip("\n"),
        imports=IMPORTS,
        seeds=len(result.seeds),
        margin=f"{_behind(result):+,.0f}",
        states="\n".join(index) + "\n" if index else "",
        failures="\n".join(_failure_lines(name, failures)) + "\n" if failures else "",
        instruction=instruction,
    )
    LOGGER.info(
        "composed a round on %s (%d prior, %d failures)",
        name,
        len(siblings),
        len(failures),
    )
    return message
