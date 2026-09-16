"""Build an agent from what a top-band team usually does, step by step.

Their code is closed and their play is not. The daily archives hold 1,146 of
SpaTaro's seasons, 1.27M recorded commands, and the same for every other team
above us -- the one asset the teams we cannot beat have already handed over.

A single season replayed is not them: the top of this ladder is adaptive, and a
frozen route is a plan tuned to the opponent it was recorded against, which is
why tapes of these agents measured 0.726 against champion_14 where real code
measured 0.430. What a thousand seasons give instead is the *modal* action at
each step -- what the agent does at day 7 hour 3 across every opponent it met,
with the opponent-specific noise voted out.

It is still a fixed plan and the ceiling on it is known: this replays what they
usually do, not the routing that decides when to do otherwise. What it is for
is measurement. If the consensus route of the second-rated team on the board
holds its own against our champion, their advantage is in the plan and we can
mine it; if it collapses, their advantage is in the routing and no amount of
recorded play will hand it to us.

    uv run consensus "SpaTaro" --episodes 80
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from kaggriculture.campaign import config, games
from kaggriculture.scripts.tape_opponents import Action, tape, write_agent

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Mine one team's seasons into a consensus route and write it as an agent."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "team", help="the team whose play to mine, as the corpus names it"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=80,
        help="how many of its seasons to read (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="where the agent goes (default: OPPONENTS/consensus_<team>)",
    )
    arguments = parser.parse_args()

    seasons = recorded(arguments.team, arguments.episodes)
    LOGGER.info("%d seasons of %s in the archives", len(seasons), arguments.team)
    if not seasons:
        raise SystemExit(f"no archived seasons for {arguments.team!r}")

    route, agreement = consensus(seasons)
    LOGGER.info(
        "%d steps decided; the modal action is unanimous at %d of them (%.0f%%)",
        len(route),
        sum(1 for share in agreement.values() if share == 1.0),
        100
        * sum(1 for share in agreement.values() if share == 1.0)
        / max(1, len(route)),
    )
    thin = sorted(agreement.items(), key=lambda kv: kv[1])[:5]
    LOGGER.info(
        "least agreed steps: %s",
        ", ".join(f"step {step} at {share:.2f}" for step, share in thin),
    )

    home = arguments.out or config.OPPONENTS / f"consensus_{_slug(arguments.team)}"
    written = write_agent(route, home / "main.py")
    LOGGER.info("wrote %s (%.1f KiB)", written, written.stat().st_size / 1024)


def recorded(team: str, limit: int) -> list[tuple[str, str, int]]:
    """That team's seasons, newest first, as (played, episode, seat).

    Newest because the field moves: a season from three weeks ago was played
    against agents that are no longer there, and what the route is for is the
    field as it stands.

    Args:
        team: The team, as `games.teams` names it.
        limit: How many seasons to take.

    Returns:
        One tuple per season, newest first.
    """
    quoted = team.replace("'", "''")
    rows = games.query(
        "select played, episode, if(team_0 = '" + quoted + "', 0, 1) as seat "
        "from games.episodes where source = 'ladder' and played != '' "
        f"and (team_0 = '{quoted}' or team_1 = '{quoted}') "
        f"order by played desc limit {limit} format TabSeparated"
    )
    found = []
    for line in rows.splitlines():
        played, episode, seat = line.split("\t")
        found.append((played, episode, int(seat)))
    return found


def consensus(
    seasons: list[tuple[str, str, int]],
) -> tuple[dict[str, Action], dict[str, float]]:
    """The most common action at each step, and how often the seasons agree.

    Args:
        seasons: What `recorded` returned.

    Returns:
        The route, and the share of seasons voting for the winning action at
        each step -- 1.0 where every season did the same thing.
    """
    votes: dict[str, Counter] = {}
    read = 0
    for played, episode, seat in seasons:
        try:
            actions = tape(played, episode, seat)
        except (FileNotFoundError, KeyError) as error:
            LOGGER.info("%s: %s", episode, error)
            continue
        read += 1
        for step, action in actions.items():
            votes.setdefault(step, Counter())[json.dumps(action, sort_keys=True)] += 1
    LOGGER.info("read %d of %d seasons", read, len(seasons))

    route: dict[str, Action] = {}
    agreement: dict[str, float] = {}
    for step, counted in votes.items():
        winner, count = counted.most_common(1)[0]
        route[step] = json.loads(winner)
        agreement[step] = count / sum(counted.values())
    return route, agreement


def _slug(team: str) -> str:
    """The team's name as a directory, lowercase and alphanumeric."""
    kept = "".join(c if c.isalnum() else "_" for c in team.lower())
    return "_".join(part for part in kept.split("_") if part) or "team"
