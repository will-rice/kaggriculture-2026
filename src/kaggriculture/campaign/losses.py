"""The games this lineage actually lost, kept current in the games database.

A round diagnoses the game it is handed, and until 2026-09-24 that game always
came from the gate's pool -- opponents chosen weeks ago that the champion beats
89% of the time. The games it loses are elsewhere: 271 distinct opponents across
293 ladder games, a median losing margin of 894 coins on banks near 98,000, and
over half of them inside a thousand.

Kaggle keeps every one. `competition_list_episodes` says which were lost and by
how much, `competition_episode_replay` returns all 720 steps of both seats, and
that is the shape `dataset.rows` already parses. The rows land under
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
# Every replay this has ever loaded, kept. A replay is 31 MB and thirty of them
# arrive per submission, so an hourly loop does accumulate -- about a gigabyte a
# submission, against 514 GB free on this disk.
#
# Worth it, because the database is a lossy derivation of this and not a
# replacement for it. `dataset.rows` keeps every order and every move, but it
# samples *state* once a day -- `steps[day * HOURS + LAST_HOUR]`, thirty of 720
# steps -- so the other twenty-three hours of each day exist here and nowhere
# else. A column added to the schema later is re-derived from these files; asked
# of the rows alone it has no answer.
#
# Nor is re-downloading a fallback. Kaggle serves a submission's episodes while
# that submission is listed, and the lineage promotes past a submission in
# hours, so these are cheap to keep and may be impossible to fetch again.
#
# `_held` means the loader itself never reads one twice: it asks the database
# which episodes it holds and does not fetch those. That makes the directory a
# record rather than a working file, which is exactly why it outlives the pass.
CACHE = config.EPISODES.parent / "replays"
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

    CACHE.mkdir(parents=True, exist_ok=True)
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


def deficit() -> dict[str, float]:
    """Where the standing program's real losses are decided, as numbers to log.

    Recorded rather than gated on. The day-29 gap *is* the final margin, which
    the gate already scores, and a day-indexed figure earlier than that is a
    correlate: day-10 bank once trended beautifully across a selected chain of
    champions and was not the mechanism. So these go to wandb, where a person
    reads them against promotions, and nowhere near `gate.promotion`.

    Returns:
        The count of games held, and the mean gap between the two banks at day
        10, over days 20 to 29, and at the close. Empty when nothing is held,
        because a refresh that has never run has nothing to say.

        Empty is decided by the count, not by whether the server answered. An
        average over no rows is `nan` and arrives as a row like any other, so
        reading "did it reply" would log four `nan`s an hour against a database
        whose live partition is simply still empty.
    """
    rows = games.query(
        "select count(distinct episode),"
        " round(avgIf(bank, team = 'ours' and day = 10)"
        " - avgIf(bank, team = 'opponent' and day = 10)),"
        " round(avgIf(bank, team = 'ours' and day >= 20)"
        " - avgIf(bank, team = 'opponent' and day >= 20)),"
        " round(avgIf(bank, team = 'ours' and day = 29)"
        " - avgIf(bank, team = 'opponent' and day = 29))"
        f" from {games.DATABASE}.days where source = 'live' FORMAT TabSeparated"
    ).strip()
    held, ten, late, close = rows.split("\t")
    if not float(held):
        return {}
    return {
        "losses/games": float(held),
        "losses/gap_day10": float(ten),
        "losses/gap_days20_29": float(late),
        "losses/gap_final": float(close),
    }


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
    """One replay, from the cache when it is already here."""
    path = CACHE / f"episode-{episode_id}-replay.json"
    if not path.exists():
        api.competition_episode_replay(episode_id, path=str(CACHE))
    if not path.exists():
        LOGGER.warning("  %d: no replay returned", episode_id)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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
    dataset.rows(batch, episode, played)
