"""Compose the one message a round is given.

The model is a mutation operator: it is handed a directory holding the program
to improve, and everything else arrives as this message. The message is the
game's rules, what the files in the directory are, this round's opponent and
game, and one instruction. Every round
is composed by `compose`, the first included, so no round sees a message shaped
differently from the others.

A round is feedback on one game, named rather than rendered: every game the
campaign has played and every game it recorded off the competition are rows in
one database, so the message carries the episode key and a round reads
whichever columns its own question wants.

Only this round's opponent is named. The previous lineage, shown the whole
pool by name, evolved fingerprinting -- recognising agents by their sheep and
cow counts -- which is worth nothing on a ladder where the agent across the
table is one it has never seen.
"""

import logging
import re
from pathlib import Path

from kaggriculture.campaign import config, harness

LOGGER = logging.getLogger(__name__)

# The whole message, in order, with the measured parts left as placeholders.
# It is a file rather than a pile of string constants so that the set-up can
# be read end to end -- what a round is told, and in what order -- without
# reconstructing it from `compose`.
ROUND_PROMPT = Path(__file__).with_name("round_prompt.md")

# Where a round asks about its games: the one database every game the
# campaign plays and every game it records off the competition is written to.
GAMES = config.GAMES_URL

# What a program is recorded as having been asked, since the archive keeps
# the name of the instruction beside every program. The instruction itself is
# the last section of `round_prompt.md`, and there is one: there were five
# once -- FAMOU appendix C.2's rewrites, drawn per session -- and two of them,
# "a completely different algorithm" and "a novel approach inspired by this
# one", took 54% of every call the campaign made and returned 476 programs of
# which one scored above nought.
INSTRUCTION_NAME = "plan"


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


def compose(
    name: str,
    played: tuple[int, int, harness.Game],
    note: str = "",
) -> str:
    """Compose the message for one round.

    Args:
        name: What the program in ``child.py`` is called -- a pool name or a
            database id. It is interpolated raw, so it must never be a path.
        played: The game this round is about, as its matchup, its season and
            the game itself. Every round has one: the loop records every game
            it scores before a session reads them back.
        note: What leads the instruction when a session did not start from
            the champion, or "" -- the one thing the instruction section says
            differently from one round to the next.

    Returns:
        The whole message.
    """
    matchup, season, game = played
    episode = f"{name}m{matchup}s{season}"
    message = ROUND.render(
        opponent=game.opponent,
        episode=episode,
        seat=game.seat,
        ours=f"{game.ours:,.0f}",
        theirs=f"{game.theirs:,.0f}",
        finish=f"{game.ours - game.theirs:+,.0f}",
        games=GAMES,
        note=note,
    )
    LOGGER.info("composed a round on %s", name)
    return message
