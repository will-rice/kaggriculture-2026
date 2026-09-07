"""What winning play looks like day by day, from the recorded ladder.

A round is shown its own banks, tiles and hands each day and has nothing to
compare them to. The episode corpus is thousands of games between agents
stronger than anything in the pool -- and stronger than anything published,
since the ladder's leaders do not release kernels -- so the winner's
trajectory is a reference a round can act on: behind on bank by day twelve, or
holding four tiles where a winner holds fifty-eight.

Aggregate only. This reads public replays and writes counts and medians; no
opponent's source is read and no path of theirs appears in what it writes.

Run from the repository root::

    uv run winning-pace 400

It rewrites ``campaign/winning_pace.md``, which the round prompt includes.
Re-run it when the ladder has moved: the table is a snapshot of how the field
played, not a constant of the game.
"""

import argparse
import logging
import statistics
from pathlib import Path
from typing import Any

from tqdm import tqdm

from kaggriculture.campaign import tapes

LOGGER = logging.getLogger(__name__)

TABLE = Path(__file__).resolve().parents[1] / "campaign" / "winning_pace.md"
# A season is thirty days of twenty-four hours, and the row for a day is that
# day at its last hour: what both programs saw before their final decision in
# it. The day-end refresh clears the hands, so a table read after it says
# every day was worked alone.
DAYS = 30
LAST_HOUR = 23


def main() -> None:
    """``winning-pace``: rebuild the reference table from the corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", type=int, help="how many to sample")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = pace(args.episodes)
    TABLE.write_text(render(rows), encoding="utf-8")
    LOGGER.info("wrote %s from %d episodes", TABLE, rows["episodes"])


def pace(episodes: int) -> dict[str, Any]:
    """Per-day statistics over the winning side of ``episodes`` games.

    Args:
        episodes: How many qualifying episodes to read.

    Returns:
        ``episodes`` counted, and per-day lists of bank, planted tiles,
        animal structures and hands for whichever seat won.
    """
    banks: list[list[float]] = [[] for _ in range(DAYS)]
    planted: list[list[int]] = [[] for _ in range(DAYS)]
    animals: list[list[int]] = [[] for _ in range(DAYS)]
    hands: list[list[int]] = [[] for _ in range(DAYS)]
    used = 0
    for index in tqdm(range(episodes), desc="episodes"):
        try:
            episode = tapes.qualifying(index)
        except FileNotFoundError:
            LOGGER.info("corpus exhausted after %d episodes", used)
            break
        final = episode.steps[-1][0]["observation"]["farms"]
        money = [float(farm["money"]) for farm in final]
        if money[0] == money[1]:
            continue  # a draw has no winning side to learn from
        winner = 0 if money[0] > money[1] else 1
        used += 1
        for day in range(DAYS):
            state = episode.steps[day * 24 + LAST_HOUR][0]["observation"]
            bank, grown, built, hired = _farm(state["farms"][winner])
            banks[day].append(bank)
            planted[day].append(grown)
            animals[day].append(built)
            hands[day].append(hired)
    return {
        "episodes": used,
        "banks": banks,
        "planted": planted,
        "animals": animals,
        "hands": hands,
    }


def _farm(farm: dict[str, Any]) -> tuple[float, int, int, int]:
    """Bank, planted tiles, animal structures and hands held on one farm."""
    tiles = [tile for row in farm["tiles"] for tile in row if isinstance(tile, dict)]
    return (
        float(farm["money"]),
        sum(1 for tile in tiles if tile.get("kind") == "PLANT"),
        sum(1 for tile in tiles if tile.get("kind") in ("COOP", "PASTURE")),
        len(farm.get("hands", [])),
    )


def render(rows: dict[str, Any]) -> str:
    """The markdown the round prompt includes.

    Two bank columns because they say different things: the median is the pace
    to match and the top decile is the pace to beat. The rest are medians --
    a decile on a tile count is noise about one farm's layout.
    """
    lines = [
        "## How the ladder's winners play",
        "",
        f"Measured over {rows['episodes']} recorded games from the public replay "
        "archive, taking whichever seat won. These are the agents actually at "
        "the top of the leaderboard, most of whom publish nothing, so this is "
        "the only view of them there is -- and it is a view of what they did, "
        "never of how. Each row is that day at its last hour.",
        "",
        "The median bank is the pace to match; the top decile is the pace to "
        "beat. Do not read the early rows as poverty: a winner spends its bank "
        "down to almost nothing while it expands, and the money it is not "
        "holding is in tiles and hands.",
        "",
        "| day | median bank | top decile | planted tiles | animal pens | hands |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for day in range(DAYS):
        banks = sorted(rows["banks"][day])
        if not banks:
            continue
        decile = banks[int(0.9 * (len(banks) - 1))]
        lines.append(
            f"| {day} | {statistics.median(banks):,.0f} | {decile:,.0f} "
            f"| {statistics.median(rows['planted'][day]):.0f} "
            f"| {statistics.median(rows['animals'][day]):.0f} "
            f"| {statistics.median(rows['hands'][day]):.0f} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
