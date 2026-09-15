"""Build one replayable opponent per behavioural family in the recorded corpus.

The field plays tapes. Measured 2026-09-13 over the recorded ladder: one agent's
two episodes, on different seeds and against the same opponent, issued 6,652 of
6,652 identical hand and farmer commands and 703 identical market orders -- same
day, same hour, same verb, same argument, same quantity -- while the strawberry
price in those two games was 31 and 91. A 3x market swing changes nothing about
what it does, so it is not reading the market, the map or the seat. It is a
fixed plan being executed blind.

That is what makes this worth building. A tape of a reactive agent is a
caricature, because the world it replays into is not the world it was recorded
in -- and here the world includes us, since the market is shared. A tape of a
blind plan is the agent, exactly, with nothing left out.

One per family, not one per agent. The corpus is mostly forks: of 111 agents
with five or more episodes since 2026-09-06, 52 behavioural families, 44 of them
a single agent and one of them thirty-nine -- one strategy forked thirty-nine
ways, 13% of every recent game. Sampling agents therefore weights a strategy by
how many people copied it, which is a popularity vote dressed as a field;
sampling families weights each strategy once. Uniform is the choice here because
the ladder keeps gaining public agents and today's fork counts are not
tomorrow's.

The plan is read from the replay archive, not from the games database. The
database's `moves` and `orders` are a projection of an archive's actions and the
projection is lossy in three ways that each break a replay: a no-op carries no
row, so a hand that passed cannot be told from a hand that is not there; the
`(day, hour)` a row is filed under is the step after the one the command was
issued at; and several orders share one `(day, hour)` with no column to order
them by, so they come back permuted -- and selling before buying is what makes
the buy affordable. Reconstructing from it reached day 19 with eighteen tiles
planted where the plan it came from had fifty-eight. The archive holds the
action dict verbatim, so there is nothing to reconstruct.

Nothing here reads an opponent's source. An archive is the public replay record,
the same material the corpus extraction reads.
"""

import argparse
import base64
import hashlib
import json
import logging
import math
import statistics
import zipfile
import zlib
from pathlib import Path
from typing import Any, NamedTuple

from kaggriculture.campaign import config, dataset, games

LOGGER = logging.getLogger(__name__)

# One step's commands as the archive holds them, passed through untouched.
Action = dict[str, Any]


class Profile(NamedTuple):
    """One agent's recorded behaviour, and where to read its plan from.

    Attributes:
        team: The agent, by the name the corpus records.
        episodes: How many recent games it played.
        sold: Units sold by day 29, reported so a family can be recognised.
        episode: The episode its tape is taken from.
        seat: Which side of that episode was its.
        played: The day that episode was played, which names its archive.
        rating: Its Bradley-Terry rating over the recorded corpus, which is
            what decides who represents a family. Agents that never played
            enough to be rated sit at negative infinity and are taken only when
            a family holds nothing else.
        features: The values `FEATURES` names, in that order.
    """

    team: str
    episodes: int
    sold: float
    episode: str
    seat: int
    played: str
    rating: float
    features: tuple[float, ...]


# Behavioural features a family is grouped on, and how each is read from a
# day's row. Selling volume spans six orders of magnitude across the field --
# 1,456 to 901,892 units by day 29 -- so it is compared in logs; the rest are
# tile and pen counts inside a narrow band and are compared directly.
#
# Planting is not one of them by accident: every family plants 54 to 60 tiles by
# day 19, so it separates nobody. What the field varies is what it sells and how
# long it holds it.
FEATURES = (
    ("sold", 29, "sold_units", True),
    ("pens", 19, "pens", False),
    ("held", 27, "yield_held", False),
    ("hands", 29, "hands", False),
)
# How far apart two agents may sit in normalised feature space and still be one
# family. The nearest-neighbour distribution is bimodal -- a median of 0.018
# among the forks and a long tail past 1.0 -- so anything between about 0.1 and
# 0.5 cuts it in the same place. 0.25 is the middle of that.
THRESHOLD = 0.25
# Episodes an agent needs before its behaviour is a measurement rather than a
# draw. Its plan is fixed, so this is about the record being complete, not about
# averaging away variance.
LEAST = 5
# A step the archive recorded no action for: the farmer waits and nothing else
# happens. A seat that crashed has its action substituted by the framework, and
# `tapes._is_clean` is what keeps such an episode out of the corpus entirely.
IDLE: Action = {"farmer": ["PASS"], "hands": [], "market": []}


def main() -> None:
    """Cluster the corpus into families and write one tape opponent for each."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    # The window the ratings are fitted over, not a date. This runs nightly
    # now, and a pinned day would widen by one every night until the families
    # were clustered over two fields that share no names -- which is the same
    # reason `dataset.recent` exists and the same window `evidence` measures a
    # claim over.
    parser.add_argument(
        "--since",
        default=dataset.recent(),
        help="only games played this day or later (default: the rating window)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help="family radius in normalised feature space (default: %(default)s)",
    )
    parser.add_argument(
        "--least",
        type=int,
        default=LEAST,
        help="episodes an agent needs to be clustered (default: %(default)s)",
    )
    # Beside the harvested kernels, not under them: the loop adopts what it
    # finds here by the `family_` prefix its names carry, and a directory of
    # its own would be a second place to keep opponents.
    parser.add_argument(
        "--out",
        type=Path,
        default=config.OPPONENTS,
        help="where the tapes are written (default: %(default)s)",
    )
    arguments = parser.parse_args()

    profiles = measured(arguments.since, arguments.least)
    LOGGER.info("%d agents with %d+ episodes", len(profiles), arguments.least)
    families = cluster(profiles, arguments.threshold)
    LOGGER.info("%d families", len(families))

    arguments.out.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "\n%-26s%7s%9s%10s%7s  %s",
        "family",
        "agents",
        "episodes",
        "sold",
        "steps",
        "from",
    )
    for family in families:
        pick = representative(family)
        recorded = tape(pick.played, pick.episode, pick.seat)
        name = _slug(pick.team)
        write_agent(recorded, arguments.out / name / "main.py")
        LOGGER.info(
            "%-26s%7d%9d%10.0f%7d  %s seat %d",
            name[:26],
            len(family),
            sum(p.episodes for p in family),
            pick.sold,
            len(recorded),
            pick.episode,
            pick.seat,
        )


def measured(since: str, least: int) -> list[Profile]:
    """Every recent agent's behavioural profile, from the recorded corpus.

    Medians rather than means, because an episode that ended early is a short
    row rather than a missing one and would drag a mean.

    Args:
        since: Only games played this day or later.
        least: Episodes an agent needs before it is included.

    Returns:
        One profile per agent: its name, its episode count, the features it is
        clustered on, and the episode a tape would be taken from.
    """
    # The rating the corpus fits over every recorded pairing, which is what
    # picks a family's representative. An agent with too few games to rate is
    # absent here and sorts last.
    rated = {team: value for team, _, _, value, _ in dataset.ladder(games.DATABASE)}
    picked = ", ".join(
        f"round(median(if(d.day = {day}, d.{column}, null)), 3) as {name}"
        for name, day, column, _ in FEATURES
    )
    days = ", ".join(str(day) for day in sorted({f[1] for f in FEATURES}))
    rows = games.query(
        "select d.team, count(distinct d.episode) as eps, "
        f"{picked} "
        "from games.days d join games.episodes e on e.episode = d.episode "
        f"where d.source = 'ladder' and e.played >= '{since}' "
        f"and d.day in ({days}) "
        f"group by d.team having eps >= {least} format TabSeparated"
    )
    out = []
    for line in rows.splitlines():
        parts = line.split("\t")
        if len(parts) != len(FEATURES) + 2:
            continue
        values = tuple(float(x) for x in parts[2:])
        episode, seat, played = typical(parts[0], since, values[0])
        out.append(
            Profile(
                team=parts[0],
                episodes=int(parts[1]),
                sold=values[0],
                episode=episode,
                seat=seat,
                played=played,
                rating=rated.get(parts[0], -math.inf),
                features=values,
            )
        )
    return out


def typical(team: str, since: str, sold: float) -> tuple[str, int, str]:
    """The game of ``team``'s whose selling volume is nearest its own median.

    A tape has to come from one game, and which one is not arbitrary even
    though the plan is fixed: a game that ended early is a short tape, and the
    plan's effects depend on what its purchases could afford, so the extreme
    game is the one where that went least typically.

    Args:
        team: The agent, by the name the corpus records.
        since: Only games played this day or later.
        sold: That agent's median day-29 volume, the target to sit nearest.

    Returns:
        The episode, the seat that was the agent's, and the day it was played.
    """
    rows = games.query(
        "select d.episode, d.seat, e.played "
        "from games.days d join games.episodes e on e.episode = d.episode "
        f"where d.source = 'ladder' and d.team = '{_quoted(team)}' "
        f"and e.played >= '{since}' and d.day = 29 "
        f"order by abs(d.sold_units - {sold}), d.episode limit 1 format TabSeparated"
    )
    episode, seat, played = rows.splitlines()[0].split("\t")
    return episode, int(seat), played


def _quoted(team: str) -> str:
    """A team name safe inside a single-quoted ClickHouse literal.

    The corpus records whatever display name a competitor chose, and one of
    them holding an apostrophe would otherwise end the string and take the rest
    of the query with it.
    """
    return team.replace("\\", "\\\\").replace("'", "\\'")


def cluster(profiles: list[Profile], threshold: float) -> list[list[Profile]]:
    """Group agents whose behaviour is within ``threshold`` of another's.

    Single linkage, so a chain of near-identical forks is one family however
    long it runs. That is the right joining rule for copies of one plan and the
    wrong one for genuinely graded strategies; the distance distribution says
    which this corpus is -- a median nearest neighbour of 0.018 and a tail past
    1.0, with almost nothing between.

    Args:
        profiles: What `measured` returned.
        threshold: The radius, in standard deviations of each feature.

    Returns:
        Families, largest share of the record first.
    """
    points = _normalised(profiles)
    owner = list(range(len(profiles)))

    def root(i: int) -> int:
        while owner[i] != i:
            owner[i] = owner[owner[i]]
            i = owner[i]
        return i

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if _distance(points[i], points[j]) < threshold:
                a, b = root(i), root(j)
                if a != b:
                    owner[a] = b
    grouped: dict[int, list[Profile]] = {}
    for i, profile in enumerate(profiles):
        grouped.setdefault(root(i), []).append(profile)
    return sorted(grouped.values(), key=lambda family: -sum(p.episodes for p in family))


def _normalised(profiles: list[Profile]) -> list[tuple[float, ...]]:
    """Every agent's features as standard deviations from the field's middle.

    Standardised per feature rather than compared raw, because they are not in
    the same units: selling volume runs to six figures and a pen count to two,
    so an unnormalised distance is a distance in units sold and nothing else.
    """
    columns = []
    for index, (_, _, _, logged) in enumerate(FEATURES):
        column = [
            math.log10(max(1.0, p.features[index])) if logged else p.features[index]
            for p in profiles
        ]
        middle = statistics.mean(column)
        spread = statistics.stdev(column) if len(column) > 1 else 0.0
        columns.append([(v - middle) / (spread or 1.0) for v in column])
    return [tuple(column[i] for column in columns) for i in range(len(profiles))]


def _distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Euclidean distance between two normalised profiles."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def representative(family: list[Profile]) -> Profile:
    """The agent whose record the family's tape is taken from.

    The strongest of them, by the rating the corpus fits over every recorded
    pairing. This was the one with the most episodes, on the reasoning that the
    members play the same plan and what separates them is how completely each
    was recorded. They are near-identical and not identical -- five accounts
    finishing on the same bank to the unit still differ in the last few percent
    -- and the most-copied fork is not the best-executed one. Taking the best
    makes each family's opponent the hardest version of that strategy, which is
    what a gate should be made of.

    Episodes break a tie, because a rating over more games is the surer one.
    """
    return max(family, key=lambda p: (p.rating, p.episodes))


def _slug(team: str) -> str:
    """A directory name for a team, which may hold spaces or any script.

    The corpus records display names -- "Crop Dusta", "3정훈", "沒有道歉 沒有道歉"
    -- and a pool entry is a directory. Anything outside the roster's alphabet
    becomes an underscore, and a name that reduces to nothing keeps a digest of
    itself, so that two names that both reduce to nothing do not collide.

    `blake2b` and not `hash`, which is salted per process: the same agent came
    back as `family_3f90f6864ce77c74` and `family_4bb6499b2120bae3` on two
    consecutive runs, so re-running would have enrolled a second pool entry for
    a family that already had one, and gone on doing it.
    """
    kept = "".join(c if c.isalnum() and c.isascii() else "_" for c in team).strip("_")
    if kept:
        return f"family_{kept.lower()}"[:40]
    digest = hashlib.blake2b(team.encode(), digest_size=8).hexdigest()
    return f"family_{digest}"


def tape(played: str, episode: str, seat: int) -> dict[str, Action]:
    """Every action one seat of one game returned, keyed by the step it played.

    Read from the archive verbatim. The games database cannot answer this: its
    `moves` and `orders` are a projection that drops a no-op, files a command
    one step late, and permutes the orders inside a step -- see the note at the
    top of this module for what that cost.

    Args:
        played: The day the episode was played, which names its archive.
        episode: Kaggle's own id for the episode.
        seat: Which side of it to read.

    Returns:
        ``{step: action}``. A step the archive recorded no action for is absent
        and replays as `IDLE`.

    Raises:
        FileNotFoundError: No archive for that day, or no such episode in it.
    """
    path = config.EPISODES / f"kaggriculture-episodes-{played}.zip"
    if not path.exists():
        raise FileNotFoundError(f"no archive for {played} at {path}")
    with zipfile.ZipFile(path) as archive:
        member = next(
            (n for n in archive.namelist() if n.endswith(f"{episode}.json")), None
        )
        if member is None:
            raise FileNotFoundError(f"episode {episode} is not in {path.name}")
        steps = json.loads(archive.read(member))["steps"]
    # An archived action is filed under the state it produced, not the state it
    # was taken at: `replay_corpus.replay` advances the engine with `steps[N]`'s
    # action and then compares the result against `steps[N]`'s observation, so
    # that action was returned one step earlier. The actions therefore run from
    # index 1 to 719 and replay at steps 0 to 718 -- which is exactly the 719
    # calls a seat gets, the last recorded state being the one nobody acts on.
    #
    # Keyed at the index instead, every command fires a step late, and for a
    # plan that walks blind that is not an approximation: the units stand one
    # move from where the plan expects them, so `PLANT` meets a tile that is not
    # `Tile::Empty` and the engine no-ops it in silence.
    return {
        str(index - 1): step[seat]["action"]
        for index, step in enumerate(steps)
        if index > 0 and step[seat].get("action")
    }


AGENT = '''"""One recorded plan, replayed. Built by `tape-opponents`; do not edit.

Every action the agent this was taken from returned, at the step it returned it,
read from the replay archive verbatim. It reads nothing: the plan it is replaying
did not read anything either, which is what makes replaying it faithful rather
than a caricature.
"""

import base64
import json
import zlib

_TAPE = json.loads(zlib.decompress(base64.b64decode("{blob}")).decode())
_IDLE = {idle}


def agent(observation, configuration=None):
    """The action this plan returned at this step, or nothing."""
    step = int(observation["day"]) * {day} + int(observation["hour"])
    return _TAPE.get(str(step), _IDLE)
'''
# Hours in a day, so a step is `day * DAY + hour`. That is the archive's own
# step index: its step 2 carries the observation at day 0 hour 2.
DAY = 24


def write_agent(recorded: dict[str, Action], path: Path) -> Path:
    """Write a self-contained opponent that replays ``recorded``.

    Compressed rather than written out, because a plan is seven thousand
    repetitive commands and the file is read by a loader, not by a person. The
    shape is the one our own tape lineage used, so the loader has seen it.

    Args:
        recorded: What `tape` returned.
        path: The ``main.py`` to write, whose parent is created.

    Returns:
        ``path``, unchanged.
    """
    packed = zlib.compress(json.dumps(recorded, separators=(",", ":")).encode(), 9)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        AGENT.format(
            blob=base64.b64encode(packed).decode(),
            idle=json.dumps(IDLE),
            day=DAY,
        ),
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    main()
