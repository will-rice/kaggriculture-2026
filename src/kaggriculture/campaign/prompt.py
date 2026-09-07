"""Compose the one message a codex call is given.

The model is a mutation operator: it is handed a directory holding one file,
``child.py``, and everything else arrives on standard input as this message.
It edits that file and stops. The loop plays every game, so there is nothing
here about running a harness, no engine to read and no workspace to manage --
and no file we assemble that an opponent's path could leak through.

The message is spec section 4's six parts, in order: the game, the program,
the verdict on it, the states behind that verdict, the lineage's recent
failures, and the instruction. One function composes it and every round is
composed by it, the first included, so the model never sees a round that is
shaped differently from the others.
"""

import ast
import logging
from pathlib import Path

from kaggriculture.campaign import (
    archive,
    evaluator,
    gate,
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

# Rendered from the gate's own whitelist, so the model is never told a
# different set from the one that rejects it. One file ships, so this list is
# the program's whole dependency surface.
IMPORTS = ", ".join(f"`{name}`" for name in sorted(validate.ALLOWED_IMPORTS))

# The one line that differs between a program that topped the tournament and
# one that did not. Everything else in that paragraph is the same either way,
# so only this is chosen here; the rest is in the template.
PLACED_TOP = (
    "Top of it, so this program is the champion and every later candidate has "
    "to beat it."
)
PLACED_BELOW = "Every place gained is progress, whoever it comes against."

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
    "Change `child.py` so that it finishes top of the standings above. Every "
    "agent listed there is one you have to place above, and the one directly "
    "above you is the nearest of them -- but the bar is the whole table, not "
    "that one agent. How you get there is yours to choose: tune what is there, "
    "restructure it, or replace whatever part of it is losing you games."
)
# Recorded on every program, so the database keeps saying what a round was
# asked for even though there is now only one answer.
INSTRUCTION_NAME = "beat"


def _template() -> str:
    """`round_prompt.md` without the note at the top explaining it.

    That note is for whoever reads the file, not for the model, and it names
    the placeholders it documents -- which `str.format` would substitute,
    printing every section of the message twice. Stripping it is what keeps
    the file self-explanatory without the explanation becoming part of what a
    round is sent.
    """
    text = ROUND_PROMPT.read_text(encoding="utf-8")
    if text.startswith("<!--"):
        text = text[text.index("-->") + len("-->") :].lstrip("\n")
    return text


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
        "not.",
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
            "| day | our bank | their bank | our farm | their farm | our seed | "
            "our shed | their shed | our hands | their hands | prices |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for day in result.states[opponent]:
            lines.append(
                f"| {day.day} | {day.ours_bank:.0f} | {day.theirs_bank:.0f} | "
                f"{_farm(day.ours_plants, day.ours_animals, day.ours_weeds)} | "
                f"{_farm(day.theirs_plants, day.theirs_animals, day.theirs_weeds)} | "
                f"{_items(day.ours_seeds)} | "
                f"{_items(day.ours_shed)} | {_items(day.theirs_shed)} | "
                f"{day.ours_hands} | {day.theirs_hands} | {_items(day.prices)} |"
            )
    del standings
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
    cleared, why = gate.promotion(standings, name)
    # A lineage with nothing against it gets no section at all: a heading over
    # an empty list is noise in a message the model reads every round. The
    # template puts each on its own line, so an empty one leaves no gap.
    message = _template().format(
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
