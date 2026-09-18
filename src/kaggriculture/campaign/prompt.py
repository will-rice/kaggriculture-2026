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

import ast
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

# How many of the edits already made to this program are sent. Best first,
# which is `Database.children`'s own order: what a round can act on is the
# direction that came closest and the ones that lost ground, and with eight
# sessions editing the same champion the list is long enough to need a cut.
RECENT_ATTEMPTS = 3

# How a round reaches the games. One database holds every game this campaign
# has played and every game recorded off the competition, and a round asks it
# questions rather than being handed a copy: the size of the evidence stops
# being the message's problem, which is the only property that scales.
GAMES = config.GAMES_URL


# One instruction. There were five once -- FAMOU appendix C.2's rewrites, drawn
# per session -- and two of them, "a completely different algorithm" and "a
# novel approach inspired by this one", took 54% of every call the campaign
# made and returned 476 programs of which one scored above nought.
INSTRUCTION = (
    "Find where this program loses, and change `plan.json` to fix it -- the "
    "plan is what the farm does, and the controller only steers it. "
    "`measure.py` plays as many seasons as you ask it to: run the experiment "
    "that would show your change is not an improvement, and keep it only if it "
    "survives."
)
# The objective, and the method.
#
# Rewritten 2026-09-16 into the loop a top competitor published as his own --
# "the better prompt is not: build the optimal agent, but: where does this
# agent lose, and what experiment could disprove the proposed improvement?" --
# by a team whose agent this campaign vendored at a public score of 2,863. The
# round already holds the two things that needs: one game it is losing, and a
# paired harness that resolves a 5,000-coin difference in about four games.
#
# What it replaces asked for a committed season plan, and that was measured
# inert: champions 14, 15 and 16 carried an identical 57 day-keyed conditions
# with 36 of them on the last two days, and the field rate climbed anyway. An
# instruction does not change a 2,300-line program's shape by asking.
#
#
# The second sentence was replaced on 2026-09-15. What it replaced is kept
# below because the reasoning still holds and only stopped being the binding
# constraint.
#
# Commitment is the thing this lineage has never had, and the measurement that
# says so is old. In the tape lineage the plan was worth ~136,680 mean bank and
# everything 69 promotions added on top of it was worth +440: the tape was
# byte-identical from the seed to champion_69, and 521 sessions never touched
# it. The search was not lazy, it was locked out -- a plan reaches a proposer as
# 29,820 chars of base64, which does not fit in a prompt beside a 3,220-line
# program.
#
# That was true when it was written and is not now. `base64`, `zlib`, `json`
# and `pathlib` are all on the whitelist -- the champion imports them to unpack
# its own plan -- and a round is handed the plan as `plan.json`, 4,095 lines
# with one step to a line, which `gather` packs back in. So the table can be
# edited and can be shipped, and the instruction names it.
#
# Which is the whole of what changed on 2026-09-18. The method is untouched:
# find where it loses, and run the experiment that would show the fix is not
# one. What is added is where to look, because 215 rounds filed under the name
# "plan" left the plan byte-identical -- one hash across 142 programs -- and
# for 212 of them it was a line of base85 they could not read.
#
# What makes it worth the round: the tape lineage's own first champion replays
# a fixed plan, 720 of 720 commands identical on a different season against the
# same opponent and 694 of 720 against a different one -- it does not look at
# the board at all -- and it still holds champion_14, fourteen champions of
# per-turn adaptation, to 0.5625. Adaptation is not where the coins are.
#
# The third sentence is the same change aimed at the other half of the
# problem. A round has been able to play its own games since `measure.py`
# arrived on 2026-09-10 -- paired, both seats, a difference of 5,000 coins
# resolved in about four games where the gate needs 114 -- and no message has
# ever mentioned the file. The only place it is named is the `query-games`
# skill, under the heading "Then measure the change", which is the whole
# difficulty: it reads as a way to check an edit that has already been
# decided, and a round that believes that will tweak and verify rather than
# search. No transcript is kept, so how often it is actually run is not
# something this can cite -- only that nothing ever asked for it.
#
# The gate is unchanged, so this costs nothing if it is wrong: a candidate of
# the new shape is promoted only by beating champion_14 over the same pool as
# anything else.
#
# The second sentence was added on 2026-09-14 and is a deliberate exception to
# the line below it. champion_12 had stood for five and a half hours of loop
# time with every one of the eight candidates after it scoring *below* it, the
# closest by 0.008 -- not gains too small to prove, which is what the gate was
# taught to resolve that morning, but no gains at all.
#
# What the audit of its losses found: against the nine opponents it wins least
# against it plants 50.7 tiles and sells 1,356 units, and against the nine it
# wins most it plants 49.6 and sells 1,354 -- identical to within a percent --
# while banking 95,564 against 110,197. The entire 14,633 is the price it sold
# into. Those nine opponents are three strategies counted nine times, they move
# 46 times the volume of the ones it beats, and they are 39 forks and 13% of
# every recent game on the real ladder. Its sell rule reads no price at all: a
# hardcoded table of hours for one crop on four days, and a food reserve.
#
# It names a variable and not a technique, and nothing about what to do with
# it. What it buys is a test: the information was already reachable -- the
# prices are in `games.prices` per episode per day and the skill stopped
# forbidding the comparison that finds this on 2026-09-13 -- and ten of twelve
# candidates were already editing the sell block, moving the four integers in
# that table and nothing else. If the next pass still only moves integers, the
# constraint was never what a round was told.
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


def _tried_lines(name: str, rate: float, siblings: list[archive.Program]) -> list[str]:
    """Render the edits already made to this program and what they scored.

    A round is a fresh call that remembers nothing of the ones before it, so
    without this a session's twelve consecutive attempts on one opponent are
    twelve independent guesses rather than twelve steps of a search. Early on
    guessing works, because when a program is bad most changes help; at 0.316
    against the field most changes hurt, and a search that cannot tell which
    way it moved stops climbing. Measured 2026-09-13: thirteen rounds, eight of
    them above the program they started from by +0.005 to +0.024, and nothing
    carried between them.

    `Database.children` has held this list all along -- "what the next round is
    told has already been tried from where it stands" -- and `compose` counted
    it for a log line. It was always empty, because a session advanced onto
    whatever its last round wrote, so the program being composed against was
    new and had no children yet. It fills up now that a round that lost ground
    hands the next one the program it started from.

    The edits are files rather than diffs in the message, the way the game is
    an episode key rather than a table: a round diffs whichever of them its own
    question is about, and one that does not care pays nothing for them.

    The deltas are whole-pool means differenced, not `Result.beats`'s comparison
    over the opponents both sides played. A sibling and the program it was
    edited from are measured minutes apart and almost always against the same
    pool, so the two agree; across a harvest the delta is off by about a
    forty-eighth of a rate difference. This is a direction for a round to read
    and not the comparison anything is decided on, which is why it is the cheap
    one.

    Args:
        name: The program in ``child.py``, by name.
        rate: Its mean win rate over the opponents it was measured on, so the
            deltas below have something to be deltas from.
        siblings: Programs written from it, best first, already copied into the
            round's directory as ``tried_1.py`` and so on.

    Returns:
        Lines of a markdown section, one bullet per edit.
    """
    lines = [
        f"## Edits already tried on `{name}`",
        "",
        f"`child.py` wins {rate:.3f} of its games against the pool. These "
        "programs were written from it and played, best first, and each one is "
        "in your directory:",
        "",
    ]
    for number, program in enumerate(siblings[:RECENT_ATTEMPTS], start=1):
        lines.append(
            f"- `tried_{number}.py` scored {program.fitness:.3f} "
            f"({program.fitness - rate:+.3f})"
        )
    lines += [
        "",
        "Diff them against `child.py` to see what each one changed. A better "
        "score is a direction to go further in; a worse one is a direction "
        "already measured and lost.",
    ]
    return lines


def schedule(source: str) -> dict[int, int]:
    """How many of a program's conditions name a particular day.

    A proxy for commitment, and a deliberately crude one: a decision taken at a
    fixed day is a plan whether it is written as a table or as `if day == 3`,
    and counting the days a program's conditions name is the cheapest way to
    see which parts of the season it has decided in advance.

    It undercounts. A decision committed to a day whose quantity is computed
    from inventory still reads as one condition, and a schedule expressed
    through a variable rather than a literal is invisible here. What it is for
    is the shape: measured on champion_15, 30 of 46 named day 29 and eight days
    of the season carried the other 16.

    Args:
        source: The program, as text.

    Returns:
        Day to how many conditions name it, empty if the source will not parse.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    counted: dict[int, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        parts = [node.left, *node.comparators]
        if not any(_names_the_day(part) for part in parts):
            continue
        for part in parts:
            if isinstance(part, ast.Constant) and isinstance(part.value, int):
                counted[part.value] = counted.get(part.value, 0) + 1
    return dict(sorted(counted.items()))


def _names_the_day(node: ast.expr) -> bool:
    """Whether this operand is the season's day, however it was reached."""
    if isinstance(node, ast.Name):
        return node.id == "day"
    if isinstance(node, ast.Attribute):
        return node.attr == "day"
    if isinstance(node, ast.Subscript):
        index = node.slice
        return isinstance(index, ast.Constant) and index.value == "day"
    return False


def _schedule_lines(source: str) -> list[str]:
    """Where the program has already decided, and where it decides as it goes."""
    counted = schedule(source)
    if not counted:
        return []
    named = ", ".join(f"day {day}: {count}" for day, count in counted.items())
    silent = [day for day in range(30) if day not in counted]
    lines = [
        "## Where your program has already decided",
        "",
        f"Conditions in `child.py` that name a day, counted: {named}.",
    ]
    if silent:
        lines.append(
            "It names no day at "
            + ", ".join(str(day) for day in silent)
            + " -- on those days it decides as it goes."
        )
    return lines


def compose(
    name: str,
    played: tuple[int, int, harness.Game] | None,
    rate: float,
    failures: list[archive.Failure],
    siblings: list[archive.Program],
    instruction: str,
    source: str = "",
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
        rate: The program's mean win rate over the opponents it was measured
            on, which is the number the edits below are deltas from.
        failures: Every failure the ledger holds against that program, oldest
            first. The caller hands over what it has and this cuts it to the
            last few, so a caller cannot forget to.
        siblings: Programs already written from ``name`` and scored, best
            first. The caller hands over what it has and this cuts it to the
            first few, and `round` copies the same ones into the directory.
        instruction: ``INSTRUCTION``, with any stagnation note the caller
            prepended.
        source: The program in ``child.py``, so the message can show which days
            of the season it has already decided. Empty renders no section.

    Returns:
        The whole message, for codex's standard input.
    """
    # A section that would be empty is rendered as nothing at all, heading
    # included: a heading over an empty list is noise in a message the model
    # reads every round. The template puts each on its own line, so an empty
    # one leaves no gap.
    section = _game_lines(name, played)
    tried = _tried_lines(name, rate, siblings) if siblings else []
    committed = _schedule_lines(source) if source else []
    message = ROUND.render(
        task=TASK_PROMPT.read_text(encoding="utf-8").rstrip("\n"),
        imports=IMPORTS,
        game="\n".join(section) + "\n" if section else "",
        tried="\n".join(tried) + "\n" if tried else "",
        failures="\n".join(_failure_lines(name, failures)) + "\n" if failures else "",
        schedule="\n".join(committed) + "\n" if committed else "",
        instruction=instruction,
    )
    LOGGER.info(
        "composed a round on %s (%d prior, %d failures)",
        name,
        len(siblings),
        len(failures),
    )
    return message
