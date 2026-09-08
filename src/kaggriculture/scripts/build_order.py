"""Write the build order a round is shown: what the strongest agents hold, by day.

This replaces ``winning-pace``, which took medians over the winning side of
every recorded game. That is the wrong half of the corpus. About half of a
ladder's winners are the weaker agent having a good day, and measured that way
eleven quantities over thirteen thousand games all came back between 45% and
60% -- noise dressed as a finding, and three of them were shown to rounds.

Averaged over the top of a fitted rating instead, the same games say something
sharp and early. Between the top twenty-five and the bottom eighty: quadrants
open two to three days sooner, fertilizer is on the ground by day five where
the rest have none, planted tiles lead 31 to 22 by day six, and the bank is
*behind* until day twelve because all of it went into capacity. The weakest
separations in the corpus are all about the endgame.

Aggregate only, like the tables it replaces: this reads the extracted corpus
and writes counts. No opponent's source is read, no agent is named, and no
path of theirs appears in what it writes.

Run from the repository root, after ``uv run extract-corpus``::

    uv run build-order

It rewrites ``campaign/build_order.md``, which the round prompt includes whole.
"""

import argparse
import logging
from pathlib import Path

from kaggriculture.campaign import dataset

LOGGER = logging.getLogger(__name__)

TABLE = Path(__file__).resolve().parents[1] / "campaign" / "build_order.md"
# Rendered per column, because a bank in coins and a quadrant count out of four
# do not want the same precision and a table of "3.0" quadrants reads as noise.
PLACES = {"bank": 0, "quadrants": 1, "planted": 1, "fertilised": 1}
HEADINGS = {
    "bank": "bank",
    "quadrants": "quadrants",
    "planted": "planted tiles",
    "fertilised": "fertilised",
    "pens": "animals",
    "hands": "hands",
    "seeds": "seed in store",
    "shed": "shed",
    "weeds": "weeds",
}

PREAMBLE = """## How the strongest agents build

Averaged over the {games:,} recorded games of the public ladder, across the {best}
agents a Bradley-Terry fit over the whole field rates highest. None of them
publishes a kernel, so their games are the only view of them there is, and none
of them is in the pool you are being scored against.

Read down a column and you have what the top of the ladder holds on that day.
Every row here is also a column of your own day tables below, so you can put
your number beside theirs directly.

This is not a target to hit and not a rule of the game. It is what the
strongest quarter of a 185-agent field happens to do, and any of it can be
beaten -- but where you differ from it sharply and lose, this is the first
place to look.

"""

CLOSING = """
The shape of it, which the numbers above make hard to see all at once:

- **Capacity first, and early.** Land and quadrants by day five, animals from
  day zero, hands hired ahead of need. The strongest agents open their second
  and third quadrant two to three days before the rest of the field.
- **Poor on purpose until day twelve.** Their bank trails the field through the
  first ten days -- 447 against 1,380 on day six -- because it is in the
  ground. It crosses over around day twelve and finishes ahead, 95,647 against
  86,627.
- **Fertilizer is applied, not sold.** The weaker half of the field sells 1,787
  units of it a game; the strong sell 258 and buy more on top. On day five they
  hold fertilised tiles where the rest hold none, which is the single sharpest
  separation in the corpus: 100% of 313 paired games.
- **Seed goes in the ground.** From day six the strong carry about half the
  unplanted seed the rest do. They buy it and plant it rather than stockpiling.
- **They are net buyers.** After hiring, their busiest market activity is
  *buying* wheat. Over a season they move 1,965 units against the field's
  4,998, and finish richer -- the weak field sells 2,859 units in the last six
  days alone.
"""


def main() -> None:
    """Rewrite the build-order table from the extracted corpus."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=dataset.DATABASE,
        help="the extracted corpus (default: %(default)s)",
    )
    parser.add_argument(
        "--best",
        type=int,
        default=dataset.BEST,
        help="how many rated agents to read from (default: %(default)s)",
    )
    arguments = parser.parse_args()

    order = dataset.build_order(arguments.database, arguments.best)
    games = dataset.counts(arguments.database)["episodes"]
    TABLE.write_text(render(order, games, arguments.best), encoding="utf-8")
    LOGGER.info("wrote %s over %d games", TABLE, games)


def render(order: dict[str, list[float]], games: int, best: int) -> str:
    """The table as the round prompt includes it, days across and rows down."""
    days = [int(day) for day in order["day"]]
    lines = [PREAMBLE.format(games=games, best=best).rstrip("\n"), ""]
    lines.append("| day | " + " | ".join(f"d{day}" for day in days) + " |")
    lines.append("| --- |" + " --- |" * len(days))
    for name in dataset.TARGETS:
        cells = " | ".join(f"{value:,.{PLACES.get(name, 1)}f}" for value in order[name])
        lines.append(f"| {HEADINGS[name]} | {cells} |")
    lines.append(CLOSING.rstrip("\n"))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
