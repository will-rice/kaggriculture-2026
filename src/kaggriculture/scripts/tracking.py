"""Recording evaluation runs to Weights & Biases.

A dozen sweeps in a day left their results in terminal scrollback and in prose,
which means the only way to recheck an old constant is to run it again. One run
per evaluation, carrying the whole ``Strategy`` as config, turns the sweep into
something queryable: which configuration, against which opponent, at what win
rate, at which commit.

This module deliberately lives under ``scripts``, which ``package.py`` excludes
from the submission archive. The competition sandbox has no network and no API
key, so an import of ``wandb`` reaching it would forfeit the episode on turn
zero.
"""

import dataclasses
import logging
import subprocess
from pathlib import Path
from typing import Any

import wandb

from kaggriculture.config import HarnessConfig
from kaggriculture.policy import Strategy
from kaggriculture.report import Standing

LOGGER = logging.getLogger(__name__)

PROJECT = "kaggriculture-2026"
ENTITY = "will-rice"


def log_evaluation(
    standings: list[Standing],
    config: HarnessConfig,
    strategy: Strategy,
    agent: str,
) -> None:
    """Send one evaluation run to Weights & Biases.

    Args:
        standings: Per-opponent records from this evaluation.
        config: Harness configuration the evaluation ran under.
        strategy: Policy knobs the evaluated agent used.
        agent: Agent path or built-in name that was evaluated.
    """
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        job_type="evaluation",
        config=run_config(config, strategy, agent),
    )
    run.log(run_metrics(standings))
    run.finish()
    LOGGER.info("logged to %s", run.url)


def run_config(config: HarnessConfig, strategy: Strategy, agent: str) -> dict[str, Any]:
    """Return the configuration that produced a run, flat enough to group by.

    Every ``Strategy`` field is included because any of them may turn out to be
    the one that mattered — the herd size and the crop both did, and neither was
    predictable in advance.

    Args:
        config: Harness configuration the evaluation ran under.
        strategy: Policy knobs the evaluated agent used.
        agent: Agent path or built-in name that was evaluated.

    Returns:
        A flat mapping suitable as a wandb run config.
    """
    return {
        **dataclasses.asdict(strategy),
        "agent": agent,
        "games": config.games,
        "seed": config.seed,
        "opponents": list(config.opponents),
        "episode_steps": config.episode_steps,
        "commit": commit(),
    }


def run_metrics(standings: list[Standing]) -> dict[str, float]:
    """Return the metrics for one evaluation, keyed by opponent.

    Opponents are named by file path, so the stem is used: a slash in a metric
    key nests it into a separate chart group and the league stops being
    comparable at a glance. Two opponents in different directories can share a
    basename, which would otherwise collide onto the same key and silently
    drop a row from a record whose whole point is to be trustworthy — so that
    raises instead of picking a winner.

    ``win_rate/league`` weights each opponent equally, regardless of how many
    games it played. That normally coincides with a games-weighted mean,
    because ``Harness.matches()`` gives every opponent the same seed and game
    count by construction; the two diverge only when opponents error at
    different rates and end up scored over different game counts.

    Args:
        standings: Per-opponent records from this evaluation.

    Returns:
        A flat mapping of metric name to value, plus a league-wide win rate.

    Raises:
        ValueError: If two standings' opponents share a metric-key stem.
    """
    metrics: dict[str, float] = {}
    stems: dict[str, str] = {}
    for standing in standings:
        name = Path(standing.opponent).stem
        if name in stems and stems[name] != standing.opponent:
            raise ValueError(
                f"opponents {stems[name]!r} and {standing.opponent!r} both collide "
                f"on metric key {name!r}"
            )
        stems[name] = standing.opponent
        metrics[f"win_rate/{name}"] = standing.win_rate
        metrics[f"low/{name}"] = standing.low
        metrics[f"high/{name}"] = standing.high
        metrics[f"bank/{name}"] = standing.bank
        metrics[f"opponent_bank/{name}"] = standing.opponent_bank
    if standings:
        metrics["win_rate/league"] = sum(s.win_rate for s in standings) / len(standings)
    return metrics


def commit() -> str:
    """Return the short commit the evaluation ran at, or ``unknown``.

    Returns:
        The short commit hash, or ``"unknown"`` if it could not be determined.
    """
    finished = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return finished.stdout.strip() or "unknown"
