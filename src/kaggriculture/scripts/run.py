"""Main run script for the agent harness.

Plays the submission entrypoint against the environment's built-in agents and
reports a win rate per opponent.
"""

import argparse
import logging
from pathlib import Path
from statistics import mean

from kaggriculture.agent import EpisodeAgent
from kaggriculture.config import HarnessConfig
from kaggriculture.harness import Harness
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
    report(results, config)


def report(results: list[Result], config: HarnessConfig) -> None:
    """Log one line per opponent, plus any episode that failed to run."""
    for opponent in config.opponents:
        played = [result for result in results if result.opponent == opponent]
        banks = [result.scores for result in played if result.scores]
        LOGGER.info(
            "vs %-10s win_rate %.2f  (%dW %dL %dT)  bank %8.0f vs %8.0f",
            opponent,
            mean(result.score for result in played),
            sum(result.score == 1.0 for result in played),
            sum(result.score == 0.0 and not result.error for result in played),
            sum(result.score == 0.5 for result in played),
            mean(bank[0] for bank in banks),
            mean(bank[1] for bank in banks),
        )
    for result in results:
        if result.error:
            LOGGER.error("%s failed: %s", result.task_id, result.error)


if __name__ == "__main__":
    main()
