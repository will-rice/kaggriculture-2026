"""What our submissions actually do on the ladder, from Kaggle's own records.

The gate plays 191 opponents this campaign chose and reports a win rate near
0.89. Measured 2026-09-24 against Kaggle's episode records, the same champions
win 0.483 and 0.496 of their real games. That is not noise between two
estimates of one quantity, it is two different experiments:

    the gate     191 fixed opponents, 32 games each      depth
    the ladder   271 distinct opponents, about one each  breadth

champion_32 and champion_33 met 271 different agents in 293 games. A pool the
campaign plays to exhaustion cannot stand in for that, and nothing in the loop
had ever read the games that could say so.

This is a diagnostic, not a gate. It only has something to say about a program
that has been submitted, which is too late to promote on -- but it is the one
number here that cannot be overfitted, because the opponents are whoever the
competition matched us with.

Run it after a submission has played a hundred games or so::

    uv run standing

The losing margins are the part worth reading. On banks near 100,000 our losses
run to 30 coins, to 301, to 1. An economy gain of a fraction of a percent would
turn a great many of them, which is a different instruction from "beat this
opponent".
"""

import argparse
import logging
import statistics
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kaggle import KaggleApi

LOGGER = logging.getLogger(__name__)
# Margins a loss can sit inside, in coins, against banks near 100,000.
WITHIN = (100, 500, 1_000, 2_500, 5_000)
# Submissions to read, newest first.
RECENT = 4


def main() -> None:
    """Report the real record of our recent submissions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recent", type=int, default=RECENT, help="how many submissions to read"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import kaggle

    api = kaggle.KaggleApi()
    api.authenticate()
    # The client annotates the list and its entries as optional, so both are
    # narrowed here rather than assumed: a slice of None and an attribute of
    # None are the two ways this reads the API wrong.
    submissions = [
        entry
        for entry in (api.competition_submissions("kaggriculture") or [])
        if entry is not None
    ]
    for entry in submissions[: args.recent]:
        played = _games(api, int(entry.ref))
        if not played:
            LOGGER.info("%-14s no scored games yet", str(entry.ref))
            continue
        _report(str(entry.description)[:46], played)


def _games(api: "KaggleApi", submission: int) -> list[tuple[int, int]]:
    """Every completed episode as ``(our bank, their bank)``.

    A mirror match puts one submission in both seats, so the seats are told
    apart by position and never by id.
    """
    out = []
    for episode in api.competition_list_episodes(submission):
        seats = list(episode.agents)
        if len(seats) != 2:
            continue
        first, second = seats
        mine, other = (
            (first, second) if first.submission_id == submission else (second, first)
        )
        if mine.reward is None or other.reward is None:
            continue
        out.append((int(mine.reward), int(other.reward)))
    return out


def _report(label: str, played: list[tuple[int, int]]) -> None:
    """Print one submission's record, and where its losses sat."""
    won = sum(1 for ours, theirs in played if ours > theirs)
    lost = sum(1 for ours, theirs in played if ours < theirs)
    drew = len(played) - won - lost
    rate = (won + 0.5 * drew) / len(played)
    ours = statistics.mean(o for o, _ in played)
    theirs = statistics.mean(t for _, t in played)
    LOGGER.info(
        "%-46s %3d-%-3d rate %.3f  bank %7.0f vs %7.0f  (%d games)",
        label,
        won,
        lost,
        rate,
        ours,
        theirs,
        len(played),
    )
    margins = sorted(theirs - o for o, theirs in played if o < theirs)
    if not margins:
        return
    # What a broad economy gain would be worth, which is the question a
    # per-opponent breakdown cannot answer: nearly every opponent is met once.
    close = defaultdict(int)
    for margin in margins:
        for edge in WITHIN:
            if margin <= edge:
                close[edge] += 1
    parts = " ".join(
        f"<={edge:,}: {close[edge]:d} ({close[edge] / len(margins):.0%})"
        for edge in WITHIN
    )
    LOGGER.info(
        "    losses within %s  median %s coins",
        parts,
        f"{statistics.median(margins):,.0f}",
    )


if __name__ == "__main__":
    main()
