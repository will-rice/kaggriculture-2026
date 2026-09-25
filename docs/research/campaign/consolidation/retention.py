"""Does the campaign build on work it did not promote?

`Program.started_from` is the id a program was edited from, so the archive
answers this directly: if every program's parent is a champion, then a candidate
that came out better-but-not-promoted was never built on, and progress needs one
round to clear the whole gate in a single jump. If parents vary, the search does
stack small steps and the ratchet has teeth.

No log line is consulted. The last attempt at this question read a per-process
counter that resets on every restart and concluded from it.
"""

import collections
import logging

from kaggriculture.campaign import archive, config, gate

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Report where programs came from, and how deep the lineage actually goes."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    database = archive.Database(config.LIVE.archive, config.LIVE.programs)
    programs = list(database.programs)
    LOGGER.info("%d programs in the archive", len(programs))

    promoted = {
        one.id for one in programs if gate.ours(one.id) or one.id.startswith("champion")
    }
    LOGGER.info("%d of them are champions or ours_*", len(promoted))

    parents = collections.Counter(one.started_from for one in programs)
    LOGGER.info("\n%d distinct parents", len(parents))
    for parent, count in parents.most_common(8):
        mark = " (a champion)" if parent in promoted else ""
        LOGGER.info("  %-22s %3d children%s", parent or "<the seed>", count, mark)

    # The question itself: a child whose parent is not a champion is a step the
    # campaign took from unpromoted work.
    from_champions = sum(
        count for parent, count in parents.items() if parent in promoted or not parent
    )
    LOGGER.info(
        "\n%d of %d programs were edited from a champion or the seed, %d from "
        "something else",
        from_champions,
        len(programs),
        len(programs) - from_champions,
    )

    # And how long the longest chain of unpromoted work is: depth 1 everywhere
    # means every attempt started over.
    by_id = {one.id: one for one in programs}
    depths = {}

    def depth(name: str) -> int:
        """How many edits back to a champion or the seed."""
        if name in depths:
            return depths[name]
        program = by_id.get(name)
        if program is None or not program.started_from or name in promoted:
            depths[name] = 0
            return 0
        depths[name] = 1 + depth(program.started_from)
        return depths[name]

    reached = collections.Counter(depth(one.id) for one in programs)
    LOGGER.info("\nedits since the last champion:")
    for far in sorted(reached):
        LOGGER.info("  %d: %d programs", far, reached[far])


if __name__ == "__main__":
    main()
