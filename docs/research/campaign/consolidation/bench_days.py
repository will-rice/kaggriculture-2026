"""What does recording a day table cost the measurement a round runs itself?

A round's own instrument is the scarce resource: it is run once or twice per
round against the matchup the round was aimed at, and a round is what the whole
campaign is made of. So the question is not whether `days=True` is cheap in the
abstract but whether it is cheap *here* -- same agent, same opponent, same
seeds, both ways, alternating so a warm cache or a busy minute cannot land on
one arm.
"""

import logging
import time

from kaggriculture.campaign import config, harness

LOGGER = logging.getLogger(__name__)
SEEDS = [11, 12, 13, 14]
ROUNDS = 3


def main() -> None:
    """Time the same games with and without the day table."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    agent = config.LIVE.floor / "main.py"
    opponent = "ours_20"
    cores = min(8, config.CORE_BUDGET)
    LOGGER.info("%d seeds, both seats, %d cores, %d rounds", len(SEEDS), cores, ROUNDS)

    timings: dict[bool, list[float]] = {False: [], True: []}
    for _ in range(ROUNDS):
        for days in (False, True):
            start = time.monotonic()
            played = harness.play(
                agent, [opponent], SEEDS, cores, days=days, pool=config.POOL
            )
            timings[days].append(time.monotonic() - start)
            rows = sum(len(game.days) for game in played)
            LOGGER.info(
                "  days=%-5s %5.1fs  %d games, %d day rows",
                days,
                timings[days][-1],
                len(played),
                rows,
            )

    plain = min(timings[False])
    recorded = min(timings[True])
    LOGGER.info(
        "best of %d: %.1fs plain, %.1fs recorded, %+.1f%%",
        ROUNDS,
        plain,
        recorded,
        (recorded / plain - 1) * 100,
    )


if __name__ == "__main__":
    main()
