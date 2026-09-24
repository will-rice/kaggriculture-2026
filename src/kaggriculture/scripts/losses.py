"""Load the games we actually lost into the database a round already queries.

A round is handed one game to diagnose and it comes from the gate's pool --
opponents this campaign chose weeks ago and beats 89% of the time. The games it
actually loses are somewhere else entirely: 271 distinct opponents across 293
ladder games, a median losing margin of 894 coins on banks near 98,000, and
over half of them inside a thousand. None of that had ever been in front of the
search.

Kaggle keeps every one of those games. `competition_list_episodes` says which
were lost and by how much, `competition_episode_replay` returns all 720 steps
of both seats, and the shape it returns is the shape `dataset._rows` already
parses for the daily dumps. So this is mostly wiring.

The rows land under `source = 'live'`. The tables are partitioned on `source`,
so a corpus rebuild replaces `ladder` and cannot touch these.

Closest losses first, because a replay is 32 MB and a game lost by 300 coins is
one a small change would turn, where a game lost by 20,000 says only that the
opponent was better.

    uv run losses                  the newest scored submission, 20 closest
    uv run losses --keep 50        more of them

The two seats are `ours` and `opponent`. No opponent is named: an id would be
an identity a lineage could learn to recognise, and recognition does not
transfer to a finale field. Here it would buy nothing anyway, because almost
every opponent is met once and never again.
"""

import argparse
import json
import logging
from typing import TYPE_CHECKING

from kaggriculture.campaign import config, dataset, games, tapes

if TYPE_CHECKING:
    from kaggle import KaggleApi

LOGGER = logging.getLogger(__name__)
# Replays are about 32 MB each, so they are kept rather than fetched twice.
CACHE = config.EPISODES.parent / "replays"
# How many of the closest losses to load.
KEEP = 20


def main() -> None:
    """Fetch our closest real losses and put them in the games database."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep", type=int, default=KEEP, help="how many of the closest losses"
    )
    parser.add_argument(
        "--submission", type=int, default=0, help="which submission, newest if unset"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import kaggle

    api = kaggle.KaggleApi()
    api.authenticate()
    submission = args.submission or _newest(api)
    lost = _losses(api, submission)
    LOGGER.info("submission %d: %d losses on the record", submission, len(lost))
    if not lost:
        return

    CACHE.mkdir(parents=True, exist_ok=True)
    batch = games.Batch("live")
    loaded = 0
    for episode_id, seat, margin, played in lost[: args.keep]:
        raw = _replay(api, episode_id)
        if raw is None:
            continue
        _hold(batch, raw, episode_id, seat, played)
        loaded += 1
        LOGGER.info("  %d  lost by %s coins", episode_id, f"{margin:,}")
    if loaded:
        counts = batch.send()
        LOGGER.info("loaded %d games: %s", loaded, counts)


def _newest(api: "KaggleApi") -> int:
    """The most recent submission that has played anything."""
    for entry in api.competition_submissions("kaggriculture") or []:
        if entry is not None:
            return int(entry.ref)
    raise SystemExit("no submissions")


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
        if mine.reward is None or other.reward is None:
            continue
        if mine.reward >= other.reward:
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
    needs it: the seed identifies a game we could replay locally, and a game
    against an opponent whose source we do not have is not one of those.
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


if __name__ == "__main__":
    main()
