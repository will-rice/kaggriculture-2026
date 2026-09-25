"""What does the "Edits already tried" section actually say to a round?

It renders `fitness` -- the mean pool win rate -- for the program and each of its
siblings, and then tells the round that a better score is a direction to go
further in. If those numbers are all 1.000 the instruction is unfollowable: the
round is being asked to read a gradient off a constant.

`Program` carries two other measures that do not saturate: `field`, the win rate
over the vendored incumbents alone, and `margins`, the bank margin per opponent.
This asks how much signal each one still has where the campaign actually is.
"""

import logging
import statistics

from kaggriculture.campaign import archive, config

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Compare the spread of each measure over the most recent programs."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    database = archive.Database(config.LIVE.archive, config.LIVE.programs)
    programs = list(database.programs)
    recent = programs[-60:]
    LOGGER.info("the last %d programs in the archive", len(recent))

    pinned = [one for one in recent if one.fitness >= 0.999]
    LOGGER.info(
        "  fitness at the ceiling (>= 0.999): %d of %d", len(pinned), len(recent)
    )
    LOGGER.info(
        "  distinct fitness values: %d", len({round(one.fitness, 3) for one in recent})
    )

    fields = [one.field for one in recent if one.field is not None]
    if fields:
        LOGGER.info(
            "  field: %d values, %d distinct, %.3f to %.3f",
            len(fields),
            len({round(one, 3) for one in fields}),
            min(fields),
            max(fields),
        )
    else:
        LOGGER.info("  field: none carry one")

    banked = [
        statistics.fmean(m.mean for m in one.margins.values())
        for one in recent
        if getattr(one, "margins", None)
    ]
    if banked:
        LOGGER.info(
            "  mean margin: %d values, %d distinct, %+,.0f to %+,.0f",
            len(banked),
            len({round(one) for one in banked}),
            min(banked),
            max(banked),
        )
    else:
        LOGGER.info("  mean margin: none carry any")

    # What a round would literally read, for the parent with the most children.
    parents = [one for one in programs if len(database.children(one.id)) >= 3]
    if not parents:
        return
    busiest = max(parents, key=lambda one: len(database.children(one.id)))
    kids = sorted(
        database.children(busiest.id), key=lambda one: one.fitness, reverse=True
    )
    LOGGER.info(
        "\nthe section as rendered for %s (wins %.3f):", busiest.id, busiest.fitness
    )
    for number, kid in enumerate(kids[:6], start=1):
        LOGGER.info(
            "  - `tried_%d.py` scored %.3f (%+.3f)",
            number,
            kid.fitness,
            kid.fitness - busiest.fitness,
        )


if __name__ == "__main__":
    main()
