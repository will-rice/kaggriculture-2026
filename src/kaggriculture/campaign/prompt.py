"""Compose the one message a codex call is given.

The model is a mutation operator: it is handed a directory holding one file,
``child.py``, and everything else arrives on standard input as this message.
It edits that file and stops. The loop plays every game, so there is nothing
here about running a harness, no engine to read and no workspace to manage --
and no file we assemble that an opponent's path could leak through.

The message is spec section 4's six parts, in order: the game, the program,
the verdict on it, the states behind that verdict, the lineage's recent
failures, and the instruction -- and one part the spec did not have, between
the states and the lineage: what the recorded ladder's winners do that this
program does not. One function composes it and every round is composed by it,
the first included, so the model never sees a round that is shaped differently
from the others.

Two of those parts come from the public replay archive rather than from
anything the campaign played, and they are not the same thing. `winning_pace`
is what the winners held on each day, a median over the corpus; the claim
store is what the winners did *differently*, measured inside single games
where both sides had the same map and the same prices and one of them lost.
The first is a reference to read a program's own day tables against. The
second is selected: only the claims this program is on the other side of
reach it.
"""

import ast
import csv
import io
import logging
import re
from pathlib import Path
from typing import NamedTuple

from kaggriculture.campaign import (
    archive,
    evaluator,
    harness,
    strategy,
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

# How many claims a round is shown. The store is meant to grow -- every day's
# archives can propose more -- and the message is not, so what bounds it is
# not the size of the store but how many of its claims this particular program
# is on the wrong side of. Five is enough to be actionable and few enough that
# each one is read.
MOST_CLAIMS = 5
# A claim is selected when the program is on the other side of it in more than
# this share of its recorded games. Half, because one game against one
# opponent is a matchup and not a habit: a program that plants late against
# the one opponent that rushes it is not a program that plants late.
MOSTLY = 0.5

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
# program and what to do. Cutting it to one game cut the size and cut the
# evidence with it.
#
# A file does neither. Every game the evaluation played is in it, at full
# width -- all 29 measures `dataset.measures` defines, for both sides, where
# the table had room for nine -- and the message carries an index instead: one
# line per season saying how it finished. A round that wants day twelve of the
# season it nearly won can read day twelve, or load the whole file and ask
# which day the banks diverged, and a round with a different question pays
# nothing for the answer to this one.
SEASONS = "seasons.csv"

# The columns that are not a measure: the per-crop breakdowns the measures
# total up, and the shared prices. Rendered as one column per key, so a round
# can ask about WHEAT rather than about "crops".
SPREADS = ("plants", "animals", "seeds", "shed")

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
    "Change `child.py` so that it finishes each season above with a larger "
    "bank than it did. Not a better place in a table -- a bigger margin in "
    "the games themselves, and most of all in the ones it already wins "
    f"narrowly. Every season is in `{SEASONS}`, played out day by day; find "
    "where this program left money on the field and take it. Small, local "
    "changes are welcome, and so is replacing whatever part of it is playing "
    "badly."
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


def _rate_rows(result: evaluator.Result) -> str:
    """One row per pool opponent: the rate, and how far apart the banks ended.

    A win rate says how often and a margin says by how much, which is what
    separates an opponent a program nearly beats from one it is nowhere near.
    """
    return "\n".join(
        f"| {opponent} | {rate:.3f} | {result.margins[opponent].mean:+.0f} | "
        f"{result.margins[opponent].worst:+.0f} | "
        f"{result.margins[opponent].best:+.0f} |"
        for opponent, rate in result.rates.items()
    )


def seasons(result: evaluator.Result) -> str:
    """Every recorded game of one evaluation as a single CSV, day by day.

    One row per day per game, keyed by a season number that stands in for the
    opponent -- which is the whole of what a round is told about who it
    played. The games are ordered as the index in the message orders them,
    narrowest loss first, so season 1 is the one a small change would have
    turned.

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
    rows: list[dict[str, object]] = []
    for season, opponent in enumerate(_ordered(result), start=1):
        for day in result.states[opponent]:
            row: dict[str, object] = {"season": season, "day": day.day}
            for side in ("ours", "theirs"):
                for measure, value in getattr(day, side).items():
                    row[f"{side}_{measure}"] = value
                for spread in SPREADS:
                    counts = getattr(day, f"{side}_{spread}", None)
                    for item, count in (counts or {}).items():
                        row[f"{side}_{spread}_{item}"] = count
            for item, price in day.prices.items():
                row[f"price_{item}"] = price
            rows.append(row)
    if not rows:
        return ""
    # The union, because a crop nobody planted on day one has no key on day
    # one. Ordered by first appearance so the reading order is the writing
    # order rather than the alphabet, and blank where a row has no value --
    # which is a count of zero, and says so in the message.
    columns: dict[str, None] = {}
    for row in rows:
        columns.update(dict.fromkeys(row))
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(columns), restval="")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def _ordered(result: evaluator.Result) -> list[str]:
    """The recorded games, worst-beaten opponent first.

    A loss is a game lost, not a matchup lost: an opponent beaten 0.875 took
    one game in eight, and those are the games that decide whether the program
    finishes top. So the order is by rate and then by margin, which puts the
    seasons with the most to learn from at the top of the index, and the games
    it swept at the bottom rather than out of the file.
    """
    return sorted(
        (name for name in result.rates if result.states.get(name)),
        key=lambda name: (result.rates[name], result.margins[name].mean),
    )


def _states_lines(result: evaluator.Result) -> list[str]:
    """Render the index of the seasons file: one line per game, how it closed.

    Not the games themselves. They are in `SEASONS` beside `child.py`, at a
    width no message could carry, and what belongs here is the part a round
    cannot work out for itself -- which season is which, and which one is
    worth opening first.
    """
    ordered = _ordered(result)
    if not ordered:
        return []
    lines = [
        "## The seasons it just played",
        "",
        f"Every one of them is in `{SEASONS}`, beside `child.py`: one row per "
        "day per season, keyed by the `season` column below. Read it however "
        "suits the question -- the whole file, one season, one day, one "
        "column across all of them.",
        "",
        "The opponents are not named and it does not matter which they were. "
        "They are drawn from the field this program will meet, and the field "
        "turns over: the agent across the table in a scored game will be one "
        "this program has never seen. So these are samples of how a season "
        "can go against a competent opponent, not a list of agents to beat. A "
        "change that wins these seasons because it recognised who it was "
        "playing wins nothing that counts.",
        "",
        "| season | finished |",
        "| --- | --- |",
    ]
    for season, opponent in enumerate(ordered, start=1):
        final = result.states[opponent][-1]
        lines.append(f"| {season} | {final.ours_bank - final.theirs_bank:+,.0f} |")
    lines += [
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


def _both_sides(day: harness.Day) -> dict[str, tuple[float, float]]:
    """Every quantity the corpus measures, for both sides of one recorded day.

    All thirty of them, and it is one line because a `Day` now carries what
    `dataset.measures` defines rather than a hand-picked few. That matters
    more than it looks: this used to be a literal mapping of the handful the
    day row happened to name, so a settled claim about anything else was
    measured, stored, and silently dropped here -- including the two widest
    separations in the whole corpus, which had been added to the day *table*
    and not to this.

    A claim about a quantity nobody measured still cannot be selected, and
    that is now the same statement as "a quantity the dataset does not have".
    """
    return {name: (value, day.theirs[name]) for name, value in day.ours.items()}


class Gap(NamedTuple):
    """One claim a program is on the wrong side of, and by how much.

    Attributes:
        claim: What the corpus settled.
        wrong: Its own games sitting on the other side of it.
        seen: Its own games that could speak to it either way.
        ours: What it averaged on that quantity, over those games.
        theirs: What its opponents averaged, over the same games.
    """

    claim: strategy.Claim
    wrong: int
    seen: int
    ours: float
    theirs: float


def _against(claim: strategy.Claim, states: dict[str, list[harness.Day]]) -> Gap:
    """Measure one program against one claim, over every game it played.

    Every game, not only the ones whose tables the message prints: what is
    being asked is how this program plays, and six tables were chosen to bound
    a message rather than to describe it.

    The figures travel with the verdict. A claim that says only "you are on the
    wrong side of `hungry_worst` on day seven" names a quantity the day table
    does not print, so a round would be told it is behind on something it
    cannot find a number for anywhere in the message.

    Args:
        claim: The claim to check.
        states: One recorded game per opponent, day by day.

    Returns:
        A `Gap`; ``seen`` is zero for a day nobody reached.
    """
    wrong = seen = 0
    mine = yours = 0.0
    for days in states.values():
        for day in days:
            if day.day != claim.form.day:
                continue
            pair = _both_sides(day).get(claim.form.quantity)
            if pair is None:
                continue
            seen += 1
            wrong += claim.wrong_side(*pair)
            mine += pair[0]
            yours += pair[1]
    return Gap(
        claim, wrong, seen, mine / seen if seen else 0.0, yours / seen if seen else 0.0
    )


def selected(
    store: strategy.Strategies, states: dict[str, list[harness.Day]]
) -> list[Gap]:
    """The settled claims this program plays the other way round, worst first.

    Not the claims that are true -- those are a reading list. The ones worth a
    round's attention are the true ones this program is not doing, which is
    why a claim carries a form and not only a sentence: the same form that
    counts the corpus decides whether this program is the exception. That is
    also what keeps the message the size of the gap rather than the size of
    the store.

    Args:
        store: The claims and what has been measured of them.
        states: One recorded game per opponent, day by day.

    Returns:
        At most ``MOST_CLAIMS`` gaps, the ones the program is furthest from
        first, each carrying its own figures.
    """
    scored = []
    for claim in store.settled():
        gap = _against(claim, states)
        if gap.seen and gap.wrong / gap.seen > MOSTLY:
            # Ordered by how far this program is from the claim, then by how
            # far the corpus separates the sides on it -- a claim at 90% is a
            # firmer thing to be told than one at 66%.
            separation = abs(claim.agreement - 0.5)
            scored.append((gap.wrong / gap.seen, separation, gap))
    scored.sort(key=lambda row: (-row[0], -row[1]))
    return [gap for _, _, gap in scored[:MOST_CLAIMS]]


def _claim_lines(claims: list[Gap]) -> list[str]:
    """Render what the corpus confirmed and this program is not doing.

    Args:
        claims: What `selected` returned, furthest first.

    Returns:
        Lines of a markdown section, or nothing at all when the program is
        already on the right side of everything the corpus has confirmed.
    """
    if not claims:
        return []
    lines = [
        "## What the strongest agents on the ladder do differently",
        "",
        "Measured over every recorded game of the public ladder. The agents "
        "at the top of it publish no kernels, so their games are the only "
        "view of them there is, and none of them is in the pool above.",
        "",
        "Each line was checked inside single games, comparing the two players "
        "at the same day's close -- same map, same prices, same opponent -- "
        "so a difference is about what the two players did and not about the "
        "game they were given. The comparison is between the stronger and the "
        "weaker *agent*, by a rating fitted over the whole field, and not "
        "between the winner and the loser of that game: about half of all "
        "games are won by the weaker side, and measured that way none of "
        "these separates at all.",
        "",
        "Every line below is one your own games put you on the other side of. "
        "They are tendencies of strong play, not rules of the game: a "
        "tendency holding in 70% of games fails in the other 30%, and a "
        "program that wins by breaking one has beaten it rather than the "
        "other way round. Furthest first.",
        "",
        "| what the corpus says | yours | theirs | games on the other side |",
        "| --- | --- | --- | --- |",
    ]
    for gap in claims:
        lines.append(
            f"| {gap.claim.reads()} | {gap.ours:,.1f} | {gap.theirs:,.1f} "
            f"| {gap.wrong} of {gap.seen} |"
        )
    return lines


def _summary(program: archive.Program) -> str:
    """A program's own account of what it changed: its module docstring.

    Every round is asked to say in a docstring at the top of the file what it
    changed and why, so this is the author's summary rather than ours. Its
    absence is not a failure -- the program still ran and still scored, and
    the row is worth showing for the number alone.

    Args:
        program: The stored program to read.

    Returns:
        The docstring collapsed onto one line and cut, or "" if there is none.
    """
    source = Path(program.source_path).read_text(encoding="utf-8")
    try:
        docstring = ast.get_docstring(ast.parse(source))
    except SyntaxError:
        # It was validated before it was stored, so this is a file changed
        # underneath us rather than a program that never parsed.
        return ""
    return " ".join((docstring or "").split())[:CHANGE_CHARS]


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
