"""The games this lineage actually lost, kept current in the games database.

A round diagnoses the game it is handed, and until 2026-09-24 that game always
came from the gate's pool -- opponents chosen weeks ago that the champion beats
89% of the time. The games it loses are elsewhere: 271 distinct opponents across
293 ladder games, a median losing margin of 894 coins on banks near 98,000, and
over half of them inside a thousand.

Kaggle keeps every one. `competition_list_episodes` says which were lost and by
how much, `competition_episode_replay` returns all 720 steps of both seats, and
that is the shape `dataset._rows` already parses. The rows land under
`source = 'live'`; the tables are partitioned on `source`, so a corpus rebuild
replaces `ladder` and cannot touch them.

`refresh` is what the loop calls on a timer. Left to a person it goes stale the
moment a champion is promoted, and a search studying the previous champion's
failures is the same mistake as a pool that stopped representing the ladder.
"""

import json
import logging
from typing import TYPE_CHECKING

from kaggriculture.campaign import config, dataset, games, tapes
from kaggriculture.constants import ENVIRONMENT

if TYPE_CHECKING:
    from kaggle import KaggleApi

LOGGER = logging.getLogger(__name__)
# Where a replay lands on its way into the database, one at a time. The API
# writes to a directory rather than returning bytes, so there has to be one.
#
# It is not a cache. A replay is 31 MB, the loop refreshes hourly, and every
# submission brings thirty more -- kept, that is a gigabyte per submission for
# files nothing reads twice, because `_held` already asks the database what has
# been loaded and never fetches those again. So each is parsed and removed. If
# a pass dies between the two, the rows are not in the database either and the
# next pass simply fetches it again.
STAGING = config.EPISODES.parent / "replays"
# How many of the closest losses to hold. Closest first, because a game lost by
# 300 coins is one a small change would turn and a game lost by twenty thousand
# says only that the opponent was better.
KEEP = 30


def refresh(keep: int = KEEP, submission: int = 0) -> int:
    """Load the closest losses this submission has not already contributed.

    Args:
        keep: How many of the closest losses to hold in total.
        submission: Which submission to read, or 0 for the newest.

    Returns:
        How many games were newly loaded.
    """
    import kaggle

    api = kaggle.KaggleApi()
    api.authenticate()
    chosen = submission or _newest(api)
    lost = _losses(api, chosen)[:keep]
    if not lost:
        return 0
    held = _held()
    wanted = [row for row in lost if str(row[0]) not in held]
    if not wanted:
        return 0

    STAGING.mkdir(parents=True, exist_ok=True)
    batch = games.Batch("live")
    loaded = 0
    for episode_id, seat, margin, played in wanted:
        raw = _replay(api, episode_id)
        if raw is None:
            continue
        _hold(batch, raw, episode_id, seat, played)
        loaded += 1
        LOGGER.info("  %d lost by %s coins", episode_id, f"{margin:,}")
    if loaded:
        batch.send()
    return loaded


def _held() -> set[str]:
    """Episode keys already in the live partition."""
    rows = games.query(
        f"SELECT DISTINCT episode FROM {games.DATABASE}.episodes "
        f"WHERE source = 'live' FORMAT TabSeparated"
    )
    return {line.strip() for line in rows.splitlines() if line.strip()}


def _newest(api: "KaggleApi") -> int:
    """The most recent submission that has played anything."""
    for entry in api.competition_submissions(ENVIRONMENT) or []:
        if entry is not None:
            return int(entry.ref)
    raise LookupError("no submissions to read losses from")


def _losses(api: "KaggleApi", submission: int) -> list[tuple[int, int, int, str]]:
    """Our losses as ``(episode, our seat, margin, date)``, closest first."""
    out = []
    for episode in api.competition_list_episodes(submission):
        seats = list(episode.agents)
        if len(seats) != 2:
            continue
        first, second = seats
        # Told apart by position: a mirror match puts one id in both seats.
        ours = 0 if first.submission_id == submission else 1
        mine, other = (first, second) if ours == 0 else (second, first)
        if mine.reward is None or other.reward is None or mine.reward >= other.reward:
            continue
        out.append(
            (
                int(episode.id),
                ours,
                int(other.reward - mine.reward),
                str(episode.end_time or "")[:10],
            )
        )
    return sorted(out, key=lambda row: row[2])


def _replay(api: "KaggleApi", episode_id: int) -> dict | None:
    """One replay, parsed and then taken back off the disk.

    A file already in staging is one a previous pass downloaded and died before
    reading, so it is read rather than fetched again.
    """
    path = STAGING / f"episode-{episode_id}-replay.json"
    if not path.exists():
        api.competition_episode_replay(episode_id, path=str(STAGING))
    if not path.exists():
        LOGGER.warning("  %d: no replay returned", episode_id)
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    path.unlink()
    return raw


def _hold(
    batch: games.Batch, raw: dict, episode_id: int, seat: int, played: str
) -> None:
    """Parse one replay and hold its rows.

    The seed is `0` because Kaggle does not return one for a ladder game --
    reasonably, since it would let anyone reproduce it -- and nothing here
    needs one: a seed identifies a game we could replay locally, and a game
    against an opponent whose source we do not have is not one of those.

    The seats are `ours` and `opponent`, with no opponent named. An id would be
    an identity a lineage could learn to recognise, which does not transfer to
    a finale field; here it would buy nothing either, because almost every
    opponent is met once and never again.
    """
    episode = tapes.Episode(
        seed=0,
        engine_version=str(raw.get("module_version") or ""),
        info={
            "EpisodeId": episode_id,
            "TeamNames": ["ours", "opponent"] if seat == 0 else ["opponent", "ours"],
        },
        steps=raw["steps"],
    )
    dataset._rows(batch, episode, played)
