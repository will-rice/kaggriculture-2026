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
import logging
import re
from pathlib import Path
from typing import NamedTuple

from kaggriculture.campaign import (
    archive,
    evaluator,
    gate,
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

# The one line that differs between a program that topped the tournament and
# one that did not. Everything else in that paragraph is the same either way,
# so only this is chosen here; the rest is in the template.
PLACED_TOP = (
    "Top of it, so this program is the champion and every later candidate has "
    "to beat it. That is the bar, and it is not the job: win the games below "
    "by more."
)
PLACED_BELOW = (
    "Every place gained is progress, whoever it comes against -- but the way "
    "to gain one is to play the seasons below better, not to target an agent."
)

# How many of a lineage's failures are sent, and how much of each. The last
# few are what a next attempt can act on; an older one is about a program two
# rounds back. A reason is free-form -- a codex call's own last message is one
# -- so it is collapsed onto a line and cut, because the message has a budget
# and the parser error or the exception that matters comes first.
RECENT_FAILURES = 3
REASON_CHARS = 200

# Day tables shown, at 30 rows each. Every opponent the program did not beat
# outright earns one, worst first, and this bounds a message that is otherwise
# a whole pool's worth of games -- twelve of them is 360 rows of eleven
# columns, most of it about opponents the program is already close to.
#
# The cut used to be a rate at or below 0.5, which was right for a lineage
# losing nearly everything and inverted the moment the campaign was seeded
# from a strong agent: a program winning 0.875 against eight opponents was
# told "nothing to show", so the better a program got the less it was shown of
# how it played. The games it loses one in eight of are exactly the ones it
# has to win to top the standings.
MOST_TABLES = 6

# How many already-scored siblings are shown, and how much of each one's own
# account of itself. Best first, so the list is both the ceiling reached from
# here and the directions already measured. Without it eight workers start
# every session from the same program knowing nothing of each other, and the
# same dead end is re-explored in parallel for as long as the campaign runs;
# AlphaEvolve and FAMOU both feed prior candidates' measured performance into
# the next prompt, and this is that.
SIBLINGS = 8
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
    "Change `child.py` so that it finishes each game above with a larger bank "
    "than it did. Not a better place in the table -- a bigger margin in the "
    "games themselves, and most of all in the ones it already wins narrowly. "
    "Every table above is one season played out day by day; find where this "
    "program left money on the field and take it. Small, local changes are "
    "welcome, and so is replacing whatever part of it is playing badly."
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


def _standing_rows(name: str, standings: dict[str, float]) -> str:
    """The tournament table, best first, with this program marked.

    It goes in whole. A place says more than a yes or a no, and every place
    gained is progress the next round can aim at.
    """
    return "\n".join(
        f"| {place} | {'**' + agent + '**' if agent == name else agent} "
        f"| {value:+.3f} |"
        for place, (agent, value) in enumerate(
            sorted(standings.items(), key=lambda pair: -pair[1]), start=1
        )
    )


def _rival(
    name: str, standings: dict[str, float], states: dict[str, list[harness.Day]]
) -> str:
    """The opponent directly above ``name``, whose game is worth studying.

    The gate is a tournament, so the next place is taken from whoever is one
    rung up -- not from the agent the program does worst against, which under
    an absolute gate was the binding constraint and under this one is usually
    just the strongest agent in the pool. A program at the bottom of the table
    loses to that agent sixteen games to nothing; the one above it is a game
    it sometimes wins.

    Args:
        name: The program's own name in the standings.
        standings: Every agent's rating from the tournament it is part of.
        states: The day tables available, one per opponent played.

    Returns:
        An opponent name with a day table, above ``name`` if there is one.
    """
    ranked = [
        agent
        for agent in sorted(standings, key=lambda other: -standings[other])
        if agent == name or agent in states
    ]
    place = ranked.index(name)
    # Top of the table has nobody above it; then the nearest challenger below
    # is what it has to stay ahead of.
    above = ranked[place - 1] if place else ranked[1]
    return above


def _states_lines(
    result: evaluator.Result, rival: str, standings: dict[str, float]
) -> list[str]:
    """Render one game against every opponent that took a game off the program.

    The losses, because that is where there is something to learn -- and a
    loss is a game lost, not a matchup lost. An opponent beaten 0.875 has
    taken one game in eight, and those are precisely the games that decide
    whether the program finishes top; only an opponent it has beaten every
    single time has nothing left to teach. Worst first, capped at
    ``MOST_TABLES``, so the agents it has never taken a game from come before
    the ones it splits with.

    One game each, and the closest one played -- the game a small change would
    have flipped, rather than the widest loss, which shows the failure at its
    starkest and least reachable.

    Args:
        result: The evaluation, for its rates and its day tables.
        rival: The agent directly above in the standings, marked because
            passing it is the next place available.
        standings: Every agent's rating, to order what is shown.

    Returns:
        Lines of a markdown section: one table per opponent not beaten.
    """
    lost = sorted(
        (
            name
            for name, rate in result.rates.items()
            if rate < 1.0 and name in result.states
        ),
        key=lambda name: (result.rates[name], result.margins[name].mean),
    )[:MOST_TABLES]
    if not lost:
        return [
            "## Every match, day by day",
            "",
            "Nothing to show: this program won every game against every "
            "opponent in the pool.",
        ]
    lines = [
        "## The matches it lost, day by day",
        "",
        f"One game against each of the {len(lost)} opponents that took a game "
        "off it, worst first. A rate below 1.000 is a game lost, and those are "
        "the games that decide where it finishes: the ones at the top it has "
        "never beaten at all, and the ones lower down it beats most of the "
        "time and still drops points to. Each table is the closest game played "
        "against that opponent -- the one a small change would have flipped, "
        "rather than the widest loss, which shows the failure at its starkest "
        "and least reachable.",
        "",
        "Each row is how that day closed. You are shown both sides because you "
        "are the program's author; the program itself cannot see the "
        "opponent's shed or seed while it plays. A farm column reads "
        "`crops / animals / weeds`, counted in tiles, and `-` where there are "
        "none. Tiles are public, so the opponent's farm is here on the same "
        "terms as yours; its seed and carried inventory are private and are "
        "not. `quads` is unlocked quadrants of four and `fert` is growing "
        "tiles still under fertilizer -- the two the corpus separates the "
        "strongest agents from the rest on, and the two the build-order table "
        "above states.",
    ]
    for opponent in lost:
        mark = (
            " -- directly above you, and the next place you can take"
            if opponent == rival
            else ""
        )
        lines += [
            "",
            f"### `{opponent}`, won {result.rates[opponent]:.3f}{mark}",
            "",
            "| day | our bank | their bank | our farm | their farm | our quads | "
            "their quads | our fert | their fert | our seed | our shed | "
            "their shed | our hands | their hands | prices |",
            "| --- |" + " --- |" * 14,
        ]
        for day in result.states[opponent]:
            lines.append(
                f"| {day.day} | {day.ours_bank:.0f} | {day.theirs_bank:.0f} | "
                f"{_farm(day.ours_plants, day.ours_animals, day.ours_weeds)} | "
                f"{_farm(day.theirs_plants, day.theirs_animals, day.theirs_weeds)} | "
                f"{day.ours_quadrants} | {day.theirs_quadrants} | "
                f"{day.ours_fertilised} | {day.theirs_fertilised} | "
                f"{_items(day.ours_seeds)} | "
                f"{_items(day.ours_shed)} | {_items(day.theirs_shed)} | "
                f"{day.ours_hands} | {day.theirs_hands} | {_items(day.prices)} |"
            )
    del standings
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


def _sibling_lines(name: str, siblings: list[archive.Program]) -> list[str]:
    """Render what earlier rounds made of this same program, and what it scored.

    Args:
        name: The program they were all written from, by name.
        siblings: Its children, best first.

    Returns:
        Lines of a markdown section: one row per attempt.
    """
    shown = siblings[:SIBLINGS]
    lines = [
        f"## What has already been made of `{name}`",
        "",
        f"{len(siblings)} program(s) have been written from `{name}` and scored, "
        f"the best {len(shown)} of them below. These are results, not mistakes: "
        "each one ran and was played against the same pool on the same terms as "
        "the table above. A direction here has been measured, so repeating it "
        "spends a round to learn what this table already says; the rate to beat "
        "from where you stand is the best of them.",
        "",
        "| attempt | instruction | win rate | what it changed |",
        "| --- | --- | --- | --- |",
    ]
    for program in shown:
        lines.append(
            f"| {program.id} | {program.instruction} | {program.fitness:.3f} | "
            f"{_summary(program) or '-'} |"
        )
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


def _items(counts: dict[str, int]) -> str:
    """Render a shed or a price list as ``WHEAT 12, EGG 3``; "-" when empty."""
    return ", ".join(f"{item} {n}" for item, n in counts.items()) or "-"


def _farm(plants: dict[str, int], animals: dict[str, int], weeds: int) -> str:
    """Render one side's worked tiles as ``crops / animals / weeds``."""
    return f"{_items(plants)} / {_items(animals)} / {weeds or '-'}"


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
    rival = _rival(name, standings, result.states)
    # No floor is passed, so this asks only whether the program tops the
    # field -- which is the half of the verdict worth putting in front of a
    # model. The decisive bar guards the other half, replacing the agent that
    # stands, and with no floor named there is nothing to be indistinguishable
    # from; zero is the right value and the branch above never reads it.
    cleared, why = gate.promotion(standings, name, decisive=0)
    # Opened here rather than at import, so a store the daily measurement
    # has rewritten reaches a campaign that is already running.
    # A lineage with nothing against it gets no section at all: a heading over
    # an empty list is noise in a message the model reads every round. The
    # template puts each on its own line, so an empty one leaves no gap.
    message = ROUND.render(
        task=TASK_PROMPT.read_text(encoding="utf-8").rstrip("\n"),
        name=name,
        imports=IMPORTS,
        seeds=len(result.seeds),
        rates=_rate_rows(result),
        verdict=why,
        placing=PLACED_TOP if cleared else PLACED_BELOW,
        standings=_standing_rows(name, standings),
        states="\n".join(_states_lines(result, rival, standings)) + "\n",
        siblings="\n".join(_sibling_lines(name, siblings)) + "\n" if siblings else "",
        failures="\n".join(_failure_lines(name, failures)) + "\n" if failures else "",
        instruction=instruction,
    )
    LOGGER.info(
        "composed a round on %s (rival: %s, %d prior, %d failures)",
        name,
        rival,
        len(siblings),
        len(failures),
    )
    return message
