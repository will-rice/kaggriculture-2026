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
from importlib.machinery import ModuleSpec, SourceFileLoader
from importlib.util import module_from_spec
from pathlib import Path
from typing import Any, Optional

import wandb
from kaggriculture.config import LEAGUE, HarnessConfig
from kaggriculture.policy import Strategy
from kaggriculture.report import Standing

LOGGER = logging.getLogger(__name__)

PROJECT = "kaggriculture-2026"
ENTITY = "will-rice"


def log_evaluation(
    standings: list[Standing],
    config: HarnessConfig,
    agent: str,
) -> None:
    """Send one evaluation run to Weights & Biases.

    Args:
        standings: Per-opponent records from this evaluation.
        config: Harness configuration the evaluation ran under.
        strategy: Policy knobs the evaluated agent used.
        agent: Agent path or built-in name that was evaluated.
    """
    revision = commit()
    settings = run_config(config, agent, revision)
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        job_type=job_type(config),
        name=run_name(config, agent, revision),
        config=settings,
    )
    run.log(run_metrics(standings))
    run.finish()
    LOGGER.info("logged to %s", run.url)


def run_config(config: HarnessConfig, agent: str, revision: str) -> dict[str, Any]:
    """Return the configuration that produced a run, flat enough to group by.

    Every ``Strategy`` field is included because any of them may turn out to be
    the one that mattered — the herd size and the crop both did, and neither was
    predictable in advance.

    The strategy is read from the agent rather than passed alongside it. The two
    getting out of step is not hypothetical: this function used to take a
    ``Strategy`` argument and every caller handed it the package default, so a
    run evaluating a frozen baseline recorded the knobs of a completely
    different farm. A record that misdescribes what it measured is worse than no
    record, because it is trusted.

    The revision is passed in rather than looked up here, so that building a
    config is a pure function of its arguments. Resolving it inside would make
    every caller — including the tests — depend on the state of the working
    tree, which is exactly the coupling the dirty-tree check exists to police.

    Args:
        config: Harness configuration the evaluation ran under.
        agent: Agent path or built-in name that was evaluated.
        revision: Short commit to attribute the run to.

    Returns:
        A flat mapping suitable as a wandb run config.
    """
    strategy = strategy_of(agent)
    return {
        **(dataclasses.asdict(strategy) if strategy is not None else {}),
        "agent": agent,
        "games": config.games,
        "seed": config.seed,
        "opponents": list(config.opponents),
        "episode_steps": config.episode_steps,
        "commit": revision,
    }


def short(spec: str) -> str:
    """Return a readable stem for an agent or opponent spec.

    Built-in opponents are bare names; everything else is a path, and the
    directory is noise once the file names are distinct.
    """
    return Path(spec).stem if Path(spec).suffix else spec


def job_type(config: HarnessConfig) -> str:
    """Return the kind of evaluation this is, for grouping runs."""
    return "league" if tuple(config.opponents) == LEAGUE else "head-to-head"


def run_name(config: HarnessConfig, agent: str, commit: str) -> str:
    """Return a run name that says what was measured and at which commit.

    Weights & Biases names runs `icy-pond-3` by default, which is memorable and
    tells a reader nothing. A record whose whole purpose is to be queried months
    later should say, in the name, which agent played what at which revision —
    `main-vs-league-bc7b4ce` rather than `splendid-firefly-4`.

    Args:
        config: Harness configuration the evaluation ran under.
        agent: Agent path or built-in name that was evaluated.
        commit: Short revision the evaluation ran at.

    Returns:
        A name of the form ``<agent>-vs-<opponents>-<commit>``.
    """
    against = (
        "league"
        if tuple(config.opponents) == LEAGUE
        else "+".join(sorted(short(opponent) for opponent in config.opponents))
    )
    return f"{short(agent)}-vs-{against}-{commit}"


def strategy_of(agent: str) -> Optional[Strategy]:
    """Return the ``Strategy`` the named agent plays under, if it has one.

    Built-in opponents are named rather than pathed and carry no strategy, and
    the vendored economic policy has none either — it is not parameterised that
    way. A run of either honestly records no knobs rather than borrowing
    somebody else's.

    Args:
        agent: Agent path or built-in name.

    Returns:
        The module's ``STRATEGY`` if it exposes one, otherwise ``None``.
    """
    path = Path(agent)
    if not path.is_file():
        return None
    loader = SourceFileLoader(path.stem, str(path))
    module = module_from_spec(ModuleSpec(loader.name, loader))
    loader.exec_module(module)
    strategy = getattr(module, "STRATEGY", None)
    return strategy if isinstance(strategy, Strategy) else None


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
    """Return the short commit the evaluation ran at, refusing a dirty tree.

    Returns:
        The short commit hash, or ``"unknown"`` outside a git checkout.

    Raises:
        RuntimeError: If the working tree has uncommitted changes.
    """
    revision = (
        subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        or "unknown"
    )
    return ensure_reproducible(working_tree_status(), revision)


def working_tree_status() -> str:
    """Return git's porcelain status for the working tree, empty when clean."""
    return subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def ensure_reproducible(status: str, revision: str) -> str:
    """Return the revision, or refuse if the tree that produced it was dirty.

    A tracked run asserts that these numbers came from this revision. With
    uncommitted changes in the tree that assertion is false, and the run is a
    measurement of code that exists nowhere and can never be reproduced —
    indistinguishable in the record from one that can. Every later comparison
    against it inherits the doubt.

    Refusing is deliberate rather than warning or tagging: a record whose
    trustworthiness varies row by row is one nobody can rely on without checking
    each entry, which is the same as having no record.

    Args:
        status: Porcelain status output; empty means clean.
        revision: Short commit the run would be recorded against.

    Returns:
        ``revision`` unchanged when the tree is clean.

    Raises:
        RuntimeError: If the tree is dirty, naming what is uncommitted.
    """
    if not status:
        return revision
    raise RuntimeError(
        "refusing to track a run from a dirty working tree — the numbers would "
        f"be attributed to {revision}, which is not the code that produced "
        f"them. Commit or stash first:\n{status}"
    )
