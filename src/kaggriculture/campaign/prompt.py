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

# Rendered from the gate's own whitelist, so the model is never told a
# different set from the one that rejects it. One file ships, so this list is
# the program's whole dependency surface.
IMPORTS_SECTION = """
## What your program may import

Your program is one file, and that file ships alone: nothing is packaged
beside it. It may import only these modules, and an import of anything else is
rejected before the program is scored.

{names}
""".format(names=", ".join(f"`{name}`" for name in sorted(validate.ALLOWED_IMPORTS)))

DOCTRINE = """
## Doctrine

The program in front of you began as a published agent, and building on
published work is what this competition allows. It is yours to change however
far you like -- rewrite any part of it, or all of it.

Every other opponent is closed. Never read one's source, never ask for it,
never reconstruct it: the gate rejects code that resembles any opponent your
lineage did not start from. You are given their names and what your program
scored against them, and that is the whole of what you may know about them.
"""

# What the model is asked to do with the file, stated the same way every
# round: one edit, then stop. The campaign measures; it does not.
PROGRAM_SECTION = """
## The program

`child.py` in your working directory is `{name}`, and it is the only file
there. It is the program to improve.

Edit it in place and stop. Do not run anything and do not report anything back
in your reply: whatever `child.py` holds when you finish is what the campaign
plays, against every opponent below, and the result comes back to you as
another message like this one asking you to improve it again.

`child.py` must stay one self-contained file whose last top-level callable is
`agent(observation, configuration)` -- that is what Kaggle loads. Say in a
docstring at its top what you changed and why. There is no time limit on this
call: take as long as the work needs. Keep the file complete and runnable as
you go all the same, so that what it holds is always something that could be
scored.

Nobody is reading this session. There is no human here to answer a question,
approve a design, choose between options or confirm anything, and nothing you
write in your reply is read by anyone. Any skill or process that would have you
present something and wait for approval before writing code does not apply:
plan as much as you like, but plan and then edit, and never stop to ask. The
edited file is the only thing that leaves this call, and a call that ends
without one is a round the campaign spent on nothing.

The rules above cite probes by filename. Those files are not in your
directory: take their numbers as verified and do not go looking.
"""

# How many of a lineage's failures are sent, and how much of each. The last
# few are what a next attempt can act on; an older one is about a program two
# rounds back. A reason is free-form -- a codex call's own last message is one
# -- so it is collapsed onto a line and cut, because the message has a budget
# and the parser error or the exception that matters comes first.
RECENT_FAILURES = 3
REASON_CHARS = 200

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


def _verdict_lines(
    name: str, result: evaluator.Result, standings: dict[str, float]
) -> list[str]:
    """Render what the loop measured about ``name``, and where it placed.

    The verdict comes from ``gate.promotion``, the one function that decides
    whether a program has won, so a model cannot be told it has cleared
    something the gate refuses. The standings go in whole: a place in a
    tournament says more than a yes or a no, and every place gained is
    progress the next round can aim at.

    Args:
        name: The program's name -- a pool name or a database id, never a path.
        result: The loop's own fast evaluation of it.
        standings: Every agent's rating from the tournament that evaluation is
            part of, this program included.

    Returns:
        Lines of a markdown section naming opponents only.
    """
    cleared, why = gate.promotion(standings, name)
    lines = [
        f"## The verdict on `{name}`",
        "",
        f"Played over {len(result.seeds)} seeds, both seats, against every "
        "opponent in the pool. The margin is your bank minus theirs at the "
        "final state.",
        "",
        "| opponent | win rate | mean margin | worst | best |",
        "| --- | --- | --- | --- | --- |",
    ]
    for opponent, rate in result.rates.items():
        margin = result.margins[opponent]
        lines.append(
            f"| {opponent} | {rate:.3f} | {margin.mean:+.0f} | "
            f"{margin.worst:+.0f} | {margin.best:+.0f} |"
        )
    lines += [
        "",
        f"It {why}.",
        "",
        "The bar is a Bradley-Terry tournament, which is how the competition "
        "itself ranks the field: every agent plays every other, one strength "
        "per agent is fitted from all of it at once, and the ranking is what "
        "counts. Beating a strong opponent is worth more than beating a weak "
        "one, and one bad matchup is absorbed rather than fatal -- there is "
        "no opponent you must beat, only a field you must finish above. "
        + (
            "Top of it, so this program is the champion and every later "
            "candidate has to beat it."
            if cleared
            else "Every place gained is progress, whoever it comes against."
        ),
        "",
        "| rank | agent | rating |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| {place} | {'**' + agent + '**' if agent == name else agent} "
        f"| {value:+.3f} |"
        for place, (agent, value) in enumerate(
            sorted(standings.items(), key=lambda pair: -pair[1]), start=1
        )
    ]
    return lines


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
    """Render one game against every opponent the program did not beat.

    The losses, because that is where there is something to learn: an
    opponent it never beats is one it has to learn to beat, and the opponent
    it already beats has nothing left to teach. Worst first, so the agents it
    has never taken a game from come before the ones it splits with.

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
            if rate <= 0.5 and name in result.states
        ),
        key=lambda name: (result.rates[name], result.margins[name].mean),
    )
    if not lost:
        return [
            "## Every match, day by day",
            "",
            "Nothing to show: this program beat every opponent in the pool.",
        ]
    lines = [
        "## The matches it lost, day by day",
        "",
        f"One game against each of the {len(lost)} opponents it did not beat, "
        "worst first. The ones at the top it has never taken a game from, and "
        "those are the ones it has to learn to beat. Each is the closest game "
        "played against that opponent -- the one a small change would have "
        "flipped, rather than the widest loss, which shows the failure at its "
        "starkest and least reachable.",
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
    parts = [
        TASK_PROMPT.read_text(encoding="utf-8"),
        PROGRAM_SECTION.format(name=name),
        IMPORTS_SECTION,
        DOCTRINE,
        "\n".join(_verdict_lines(name, result, standings)),
        "\n".join(_states_lines(result, rival, standings)),
    ]
    # A lineage with nothing against it gets no section at all: a heading over
    # an empty list is noise in a message the model reads every round.
    if siblings:
        parts.append("\n".join(_sibling_lines(name, siblings)))
    if failures:
        parts.append("\n".join(_failure_lines(name, failures)))
    parts.append(f"## Your instruction\n\n{instruction}\n")
    LOGGER.info(
        "composed a round on %s (rival: %s, %d prior, %d failures)",
        name,
        rival,
        len(siblings),
        len(failures),
    )
    return "\n".join(parts)
