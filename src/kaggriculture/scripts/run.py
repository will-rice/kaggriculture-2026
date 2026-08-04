"""Main run script for the agent harness.

Plays the submission entrypoint against the recorded league — a frozen replay
tape, an earlier heuristic agent, and the environment's built-in ``starter`` —
and reports a win rate per opponent.
"""

import argparse
import logging
from pathlib import Path

from kaggriculture.agent import EpisodeAgent
from kaggriculture.config import HarnessConfig
from kaggriculture.harness import Harness
from kaggriculture.policy import STRATEGY
from kaggriculture.report import format_standing, standings
from kaggriculture.result import Result
from kaggriculture.scripts.package import ENTRYPOINT

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for the agent harness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent", default=str(ENTRYPOINT), help="agent path or built-in name"
    )
    parser.add_argument("--games", type=int, default=8, help="episodes per opponent")
    parser.add_argument(
        "--opponents", nargs="+", default=None, help="agents to play against"
    )
    parser.add_argument("--seed", type=int, default=42, help="first episode seed")
    parser.add_argument("--workers", type=int, default=None, help="parallel episodes")
    parser.add_argument(
        "--replays", type=Path, default=None, help="directory for replay JSON"
    )
    parser.add_argument(
        "--track", action="store_true", help="log this evaluation to wandb"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = HarnessConfig(
        max_workers=args.workers,
        seed=args.seed,
        games=args.games,
        **({"opponents": tuple(args.opponents)} if args.opponents else {}),
    )
    harness = Harness(config=config)
    agent = EpisodeAgent(spec=args.agent, replay_dir=args.replays)
    results = harness.run(agent, harness.matches())
    report(results)
    if args.track:
        from kaggriculture.scripts.tracking import log_evaluation

        log_evaluation(standings(results), config, STRATEGY, args.agent)


def report(results: list[Result]) -> None:
    """Log one line per opponent, plus any episode that failed to run.

    Win rate rather than bank: the ladder scores wins, and a change that banks
    more while winning less has not improved anything. The interval is what
    decides whether a difference is real or is sixteen seeds of noise.
    """
    for standing in standings(results):
        LOGGER.info("%s", format_standing(standing))
    for result in results:
        if result.error:
            LOGGER.error("%s failed: %s", result.task_id, result.error)


if __name__ == "__main__":
    main()
