"""The one place a candidate becomes the floor.

The promotion rule is the spec's §5.5: the candidate's lower Wilson bound
above the champion's point estimate, the vendored field not down more than
two points, no opponent regressed beyond the wider of the two intervals.

The promotion commit runs ``git commit --no-verify``. This repo's
pre-commit hook runs the full pytest suite and ruff-format on every commit;
inside the loop that would cost minutes per champion, fail on any unrelated
test failure, and let ruff-format rewrite the codex-written ``main.py``,
changing the bytes the gate just measured. The gate itself is the
verification here -- the commit is only the record -- and the file gets
linted on the next human commit that touches it.

The on-disk state -- the floor, ``served/main.py``, the pool file, the
epoch line -- is written before the commit and is the source of truth for
promotion. A failed git record (nothing to commit, a lock file, a full
disk) is logged and swallowed rather than raised: a promotion that already
happened on disk must never be undone by a commit that failed to explain
it.
"""

import json
import logging
import subprocess
import time
from pathlib import Path

from kaggriculture.campaign import config
from kaggriculture.campaign.archive import Program
from kaggriculture.campaign.evaluator import DeepResult
from kaggriculture.campaign.pool import Pool

LOGGER = logging.getLogger(__name__)

SERVED = config.ROOT / "src" / "kaggriculture" / "served" / "main.py"
FIELD_TOLERANCE = 0.02


def promotion(candidate: DeepResult, champion: DeepResult | None) -> tuple[bool, str]:
    """Whether ``candidate`` replaces ``champion``, and why not if not.

    Args:
        candidate: The deep result of the program under consideration.
        champion: The deep result of the current floor, or None if there is
            no champion yet.

    Returns:
        Whether to promote, and a reason (why not, or "promoted").
    """
    if champion is None:
        return True, "no champion yet"
    if candidate.low <= champion.score:
        return (
            False,
            f"lower bound {candidate.low:.3f} <= champion {champion.score:.3f}",
        )
    if candidate.field < champion.field - FIELD_TOLERANCE:
        return (
            False,
            f"field {candidate.field:.3f} < "
            f"champion {champion.field:.3f} - {FIELD_TOLERANCE}",
        )
    for name, rate in champion.rates.items():
        if name not in candidate.rates:
            continue
        width = max(
            champion.intervals[name][1] - champion.intervals[name][0],
            candidate.intervals[name][1] - candidate.intervals[name][0],
        )
        if candidate.rates[name] < rate - width:
            return (
                False,
                f"regressed against {name}: {candidate.rates[name]:.3f} < "
                f"{rate:.3f} - {width:.3f}",
            )
    return True, "promoted"


def promote(
    program: Program, result: DeepResult, pool: Pool, commit: bool = True
) -> str:
    """Write the floor, register the champion in the pool, record the epoch, commit.

    Args:
        program: The archive entry being promoted.
        result: Its deep evaluation.
        pool: The opponent pool, updated and saved in place.
        commit: Whether to ``git commit`` the served copy. Tests pass False
            to skip git entirely.

    Returns:
        The new champion's pool name.
    """
    number = 1 + sum(1 for n in pool.names() if n.startswith("champion_"))
    name = f"champion_{number}"
    source = Path(program.source_path).read_text(encoding="utf-8")

    config.FLOOR.mkdir(parents=True, exist_ok=True)
    floor = config.FLOOR / "main.py"
    if floor.exists():
        floor.chmod(0o644)
    floor.write_text(source, encoding="utf-8")
    floor.chmod(0o444)

    SERVED.parent.mkdir(parents=True, exist_ok=True)
    SERVED.write_text(source, encoding="utf-8")

    # add_champion and weakness pressure change the pool in memory; save
    # immediately after so nothing between here and the save can resolve the
    # champion's name through a stale pool (Task 8 inherits this order).
    pool.add_champion(name, str(floor), result.rates)
    pool.apply_weakness_pressure(pool.weakest(result.rates))
    pool.save(config.POOL)

    epoch_line(iteration=-1, results=[result], promoted=name, rho=None)

    if commit:
        message = (
            f"feat: promote {program.id} to the floor as {name}\n\n"
            f"deep {result.score:.4f} [{result.low:.4f}, {result.high:.4f}], "
            f"field {result.field:.4f}"
        )
        try:
            subprocess.run(
                ["git", "add", str(SERVED)],
                check=True,
                cwd=config.ROOT,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "commit", "--no-verify", "-q", "-m", message],
                check=True,
                cwd=config.ROOT,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as error:
            stderr = (error.stderr or "")[-300:]
            LOGGER.warning("promotion commit failed for %s: %s", name, stderr)

    LOGGER.info("promoted %s to %s", program.id, name)
    return name


def epoch_line(
    iteration: int, results: list[DeepResult], promoted: str | None, rho: float | None
) -> None:
    """Append one epoch record to ``config.EPOCHS``.

    Args:
        iteration: The loop iteration this epoch closes, or -1 for a
            promotion recorded outside the epoch cadence.
        results: The deep results measured this epoch.
        promoted: The name of the champion promoted this epoch, or None.
        rho: Correlation between fast and deep scores this epoch, or None.
    """
    config.EPOCHS.parent.mkdir(parents=True, exist_ok=True)
    with config.EPOCHS.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "ts": time.time(),
                    "iteration": iteration,
                    "promoted": promoted,
                    "rho_fast_deep": rho,
                    "results": [r.model_dump() for r in results],
                }
            )
            + "\n"
        )
