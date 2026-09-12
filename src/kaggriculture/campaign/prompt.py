"""Compose the one message a codex call is given.

The model is a mutation operator: it is handed a directory holding one file,
``child.py``, and everything else arrives on standard input as this message.
It edits that file and stops. The loop plays every game, so there is nothing
here about running a harness, no engine to read and no workspace to manage --
and no file we assemble that an opponent's path could leak through.

The message is the game's rules, the constraints on the program, one game to
work on, and the instruction. One function composes it and every round is
composed by it, the first included, so the model never sees a round shaped
differently from the others.

A round is feedback on one game. It is named rather than rendered: every game
the campaign has played and every game it recorded off the competition are rows
in one database, so the message carries the episode key and the query, and a
round reads whichever columns its own question wants. It used to carry a table
of every matchup in the evaluation instead -- twenty-four rows standing for 768
games -- which says the program is losing and nothing about a decision it made.

Nothing measured off other agents' games reaches a round; see the note on the
build order below for what happened when it did.
"""

import logging
import re
from pathlib import Path

from kaggriculture.campaign import (
    archive,
    config,
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
# Nothing measured off other agents' games reaches a round, and the reason is
# a measurement rather than a preference. A `build_order.md` used to hold what
# the strongest agents hold on each day and the message carried it whole.
# Measured 2026-09-10: clustered to one opening the median candidate scored
# 0.275 over 68 gates with promotions about one an hour; supplied as the orders
# those agents send, the median fell to 0.026 over 156 gates and nothing
# promoted in ten hours. The ceiling hardly moved, 0.940 to 0.914, so good
# programs did not get worse -- most programs became broken. What a model does
# with another strategy's schedule is bolt it on and break the economy
# underneath, which is what three separate experiments found. The file and the
# constant that named it are gone; a round has the games database and can ask
# it whatever it wants.

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

# How a round reaches the games. One database holds every game this campaign
# has played and every game recorded off the competition, and a round asks it
# questions rather than being handed a copy: the size of the evidence stops
# being the message's problem, which is the only property that scales.
GAMES = config.GAMES_URL


CHANGE_CHARS = 160

# One instruction. There were five once -- FAMOU appendix C.2's rewrites, drawn
# per session -- and two of them, "a completely different algorithm" and "a
# novel approach inspired by this one", took 54% of every call the campaign
# made and returned 476 programs of which one scored above nought.
INSTRUCTION = "Write a program that beats the opponent."
# The objective, and nothing about how to reach it.
#
# It was "finish every season with a larger bank than it did" until
# 2026-09-12, chosen as shaping because the gate of the champion_69 era was
# saturated with the lineage's own ancestors and rank had no gradient left in
# it. The pool has since inverted -- thirty-two of thirty-four opponents are
# harvested public agents and the campaign holds no champions -- so winning is
# the signal with the gradient now.
#
# It also asked for the wrong thing. A round told to improve a margin improves
# the program it was handed, and seventy-nine rounds did exactly that without
# once leaving that program's shape.
INSTRUCTION_NAME = "win"


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


def _game_lines(name: str, played: tuple[int, int, harness.Game] | None) -> list[str]:
    """Name one game, its result, and the query that reads it back.

    One game, because a round is feedback on a game. The message used to carry
    a table of every matchup in the evaluation -- twenty-four rows standing for
    768 games -- which is a number a round cannot act on: it says the program
    is losing and nothing about any decision it made.

    The game itself is not rendered here. At full width one season is 8,888
    characters across 68 columns, and it is already in the games database along
    with every game the competition has recorded, so what belongs in the
    message is its name and the query. A round reads whichever columns its own
    question wants instead of whichever fifteen would fit in a table.

    Args:
        name: The program's id, which prefixes its episode keys.
        played: The matchup, the season and the game, or None before there is
            an evaluation to draw one from.

    Returns:
        Lines of a markdown section, or nothing at all when there is no game.
    """
    if played is None:
        return []
    matchup, season, game = played
    episode = f"{name}m{matchup}s{season}"
    finish = game.ours - game.theirs
    return [
        "## The game",
        "",
        f"Episode `{episode}`. It held seat {game.seat} and finished "
        f"{game.ours:,.0f} against {game.theirs:,.0f}, {finish:+,.0f}.",
        "",
        "Every day of it, both sides, is in the games database, along with "
        "every game the competition has recorded:",
        "",
        f"    curl -s {GAMES} --data-binary "
        + f"\"select * from games.days where episode='{episode}'"
        + ' order by day, seat format Pretty"',
        "",
        "The `query-games` skill has the schema.",
    ]


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


def compose(
    name: str,
    played: tuple[int, int, harness.Game] | None,
    failures: list[archive.Failure],
    siblings: list[archive.Program],
    instruction: str,
) -> str:
    """Compose the message for one round.

    A rating used to be passed in and is not any more. It was never rendered --
    a place is not something a round can act on, it cannot choose its opponents
    or its rank -- and it cost a Bradley-Terry fit per round to compute an
    argument nothing read.

    Args:
        name: What the program in ``child.py`` is called -- a pool name or a
            database id. It is interpolated raw, so it must never be a path.
        played: The one game this round is feedback on, as its matchup, its
            season and the game itself. None before there is an evaluation.
        failures: Every failure the ledger holds against that program, oldest
            first. The caller hands over what it has and this cuts it to the
            last few, so a caller cannot forget to.
        siblings: Programs already written from ``name`` and scored. Only
            counted, for the log line that says how deep this lineage is.
        instruction: ``INSTRUCTION``, with any stagnation note the caller
            prepended.

    Returns:
        The whole message, for codex's standard input.
    """
    # A section that would be empty is rendered as nothing at all, heading
    # included: a heading over an empty list is noise in a message the model
    # reads every round. The template puts each on its own line, so an empty
    # one leaves no gap.
    section = _game_lines(name, played)
    message = ROUND.render(
        task=TASK_PROMPT.read_text(encoding="utf-8").rstrip("\n"),
        imports=IMPORTS,
        game="\n".join(section) + "\n" if section else "",
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
