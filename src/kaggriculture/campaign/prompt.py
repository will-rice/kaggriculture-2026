"""Compose the one message a codex call is given.

The model is a mutation operator: it is handed a directory holding one file,
``child.py``, and everything else arrives on standard input as this message.
It edits that file and stops. The loop plays every game, so there is nothing
here about running a harness, no engine to read and no workspace to manage --
and no file we assemble that an opponent's path could leak through.

The message is spec section 4's five parts, in order: the game, the program,
the verdict on it, the states behind that verdict, and the instruction. One
function composes it and every round is composed by it, the first included,
so the model never sees a round that is shaped differently from the others.
"""

import logging
from pathlib import Path

from kaggriculture.campaign import config, evaluator, gate, harness, validate

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

Never read an opponent's source, never ask for it, never reconstruct it: the
gate rejects code that resembles any opponent's. You are given their names and
what your program scored against them, and that is the whole of what you may
know about them. Your agent is ours.
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
docstring at its top what you changed and why. Your budget is {minutes}
minutes; the call is stopped then and the file is scored as it stands, so keep
it complete and runnable throughout.

The rules above cite probes by filename. Those files are not in your
directory: take their numbers as verified and do not go looking.
"""

# FAMOU appendix C.2's five rewrite instructions. One is drawn per round, so
# eight workers starting from the same champion are pushed eight different
# ways and a lineage is pushed a different way each round. The caller draws
# the pair, records the name on the program it produces, and hands ``compose``
# the text.
INSTRUCTIONS: tuple[tuple[str, str], ...] = (
    ("improve", "Improve child.py's performance against the pool."),
    (
        "different",
        "Replace child.py with a completely different algorithm for the same game.",
    ),
    (
        "inspired",
        "Create a novel approach inspired by child.py that works fundamentally "
        "differently.",
    ),
    (
        "restructure",
        "Redesign child.py's core components, keeping what the verdict says wins.",
    ),
    (
        "tune",
        "Tune child.py's constants and thresholds only; keep its structure.",
    ),
)


def _verdict_lines(name: str, result: evaluator.FastResult) -> list[str]:
    """Render what the loop measured about ``name``, and the bar it is short of.

    The bar and the sentence naming what it did not beat both come from
    ``gate.promotion``, the one function that decides whether a program has
    won, so a model cannot be told it has cleared something the gate refuses.

    Args:
        name: The program's name -- a pool name or a database id, never a path.
        result: The loop's own fast evaluation of it.

    Returns:
        Lines of a markdown section naming opponents only.
    """
    cleared, why = gate.promotion(result.rates)
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
        "The bar is beating every opponent above: more than half the games "
        "against each, over both seats. Nothing else is measured, and an "
        "average over the pool promotes nothing. "
        + (
            "Clear it again on the sealed block and it becomes the champion."
            if cleared
            else "The opponents it does not beat are what a promotion turns on."
        ),
    ]
    return lines


def _states_lines(opponent: str, states: list[harness.Day]) -> list[str]:
    """Render one game against ``opponent`` day by day.

    Args:
        opponent: The opponent that game was against, by name.
        states: The day table the loop recorded while playing it.

    Returns:
        Lines of a markdown section: one row per day, both sides.
    """
    lines = [
        f"## One game against `{opponent}`, day by day",
        "",
        "The game it lost by the most, of those it played against the opponent "
        "it does worst against. Each row is how that day closed. You are shown "
        "both sides because you are the program's author; the program itself "
        "cannot see the opponent's shed while it plays.",
        "",
        "| day | our bank | their bank | our shed | their shed | our hands | "
        "their hands | prices |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for day in states:
        lines.append(
            f"| {day.day} | {day.ours_bank:.0f} | {day.theirs_bank:.0f} | "
            f"{_items(day.ours_shed)} | {_items(day.theirs_shed)} | "
            f"{day.ours_hands} | {day.theirs_hands} | {_items(day.prices)} |"
        )
    return lines


def _items(counts: dict[str, int]) -> str:
    """Render a shed or a price list as ``WHEAT 12, EGG 3``; "-" when empty."""
    return ", ".join(f"{item} {n}" for item, n in counts.items()) or "-"


def compose(name: str, result: evaluator.FastResult, instruction: str) -> str:
    """Compose the message for one round.

    Args:
        name: What the program in ``child.py`` is called -- a pool name or a
            database id. It is interpolated raw, so it must never be a path.
        result: The loop's fast evaluation of that program: the verdict, and
            the day table of one game behind it.
        instruction: The drawn instruction's text, one of ``INSTRUCTIONS``'
            second elements, with any stagnation note the caller prepended.

    Returns:
        The whole message, for codex's standard input.
    """
    parts = [
        TASK_PROMPT.read_text(encoding="utf-8"),
        PROGRAM_SECTION.format(name=name, minutes=config.ROUND_LIMIT_SECONDS // 60),
        IMPORTS_SECTION,
        DOCTRINE,
        "\n".join(_verdict_lines(name, result)),
        "\n".join(_states_lines(result.hardest, result.states)),
        f"## Your instruction\n\n{instruction}\n",
    ]
    LOGGER.info("composed a round on %s (hardest: %s)", name, result.hardest)
    return "\n".join(parts)
