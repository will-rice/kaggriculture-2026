"""Are the candidates worse than the champion, or better and refused?

489 programs have produced about 39 champions. Those two stories call for
opposite fixes -- a better proposer against a looser gate -- and the archive
distinguishes them: every program carries the fitness it was measured at, and
`field` is the one comparable across the whole campaign because the vendored
incumbents never change.

Compared against the champion each was edited from, not against the best ever,
because a candidate is judged on beating its own parent.
"""

import collections
import logging
import statistics

from kaggriculture.campaign import archive, config

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Report how candidates scored relative to the parent they came from."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    database = archive.Database(config.LIVE.archive, config.LIVE.programs)
    programs = {one.id: one for one in database.programs}
    LOGGER.info("%d programs", len(programs))

    scored = [one for one in database.programs if one.fitness is not None]
    LOGGER.info("%d carry a fitness", len(scored))

    better = []
    worse = []
    for one in scored:
        parent = programs.get(one.started_from)
        if parent is None or parent.fitness is None:
            continue
        (better if one.fitness > parent.fitness else worse).append(
            one.fitness - parent.fitness
        )

    compared = len(better) + len(worse)
    if not compared:
        LOGGER.info("nothing could be compared to its parent")
        return
    LOGGER.info(
        "\nof %d candidates with a scored parent: %d beat it (%.0f%%), %d did not",
        compared,
        len(better),
        100 * len(better) / compared,
        len(worse),
    )
    LOGGER.info(
        "  when better, by %+.4f median win rate; when worse, by %+.4f",
        statistics.median(better) if better else 0.0,
        statistics.median(worse) if worse else 0.0,
    )

    # How many cleared the champion of their moment at all. `fitness` is the
    # pool mean, which the gate does not promote on by itself, so this is the
    # loosest possible bar: candidates that were not even nominally ahead can
    # not have been refused by a strict gate.
    top = max(one.fitness for one in scored)
    near = [one for one in scored if one.fitness >= top - 0.01]
    LOGGER.info(
        "\nbest fitness in the archive %.4f; %d programs within 0.01 of it",
        top,
        len(near),
    )
    models = collections.Counter(one.model for one in scored if one.model)
    LOGGER.info("\nwho wrote them: %s", dict(models.most_common(5)))


if __name__ == "__main__":
    main()
