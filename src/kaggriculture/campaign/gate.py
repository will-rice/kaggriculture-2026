"""The one place a candidate becomes the floor.

The promotion rule is the spec's §5.5: the candidate's lower Wilson bound
above the champion's point estimate, the vendored field not down more than
two points, no opponent regressed beyond the wider of the two intervals.

The promotion commit runs ``git commit --no-verify``. This repo's
pre-commit hook runs the full pytest suite and ruff-format on every commit;
inside the loop that would cost minutes per champion, fail on any unrelated
test failure, and let ruff-format rewrite the codex-written ``main.py``,
changing the bytes the gate just measured. Evolved code is gated, not
linted: the deep evaluation is the whole of its verification, and
``served/`` is excluded from the format, lint and type hooks so a champion
whose annotations do not typecheck cannot block the next human commit.

The on-disk state -- ``champions/<name>.py``, the floor, ``served/main.py``,
``champion.json``, the pool file -- is written before the commit and is the
source of truth for promotion. A failed git record (nothing to commit, a
lock file, a full disk) is logged and swallowed rather than raised: a
promotion that already happened on disk must never be undone by a commit
that failed to explain it.
"""

import logging
import os
import subprocess
from pathlib import Path

from pydantic import BaseModel

from kaggriculture.campaign import config
from kaggriculture.campaign.archive import Program
from kaggriculture.campaign.evaluator import DeepResult
from kaggriculture.campaign.pool import Pool

LOGGER = logging.getLogger(__name__)

SERVED = config.SERVED
FIELD_TOLERANCE = 0.02


class Champion(BaseModel):
    """The promoted floor, as ``config.CHAMPION`` records it.

    Written before ``promote`` returns and preferred over ``state.json`` on
    restart: ``state.json`` is written after every completed call, so a kill
    between the promotion and that write would otherwise lose the champion
    and let the gate promote a second time against no baseline.

    Attributes:
        name: The champion's pool name, e.g. "champion_3".
        path: The immutable copy under ``config.CHAMPIONS`` the pool plays.
        result: The deep evaluation the promotion was decided on.
    """

    name: str
    path: str
    result: DeepResult


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


def promote(program: Program, result: DeepResult, pool: Pool) -> Champion:
    """Keep the champion's own copy, write the floor and update the pool.

    Each champion is written once to ``CHAMPIONS/<name>.py`` and it is that
    path the pool registers, so a pool holding N champions holds N different
    programs. ``FLOOR/main.py`` is the current floor and is overwritten every
    promotion; registering it instead would make every pool entry an alias
    for the newest agent and silently erase the history the pool exists to
    keep.

    Args:
        program: The archive entry being promoted.
        result: Its deep evaluation.
        pool: The opponent pool, updated and saved in place.

    Returns:
        The champion record, exactly as ``config.CHAMPION`` now holds it.
        Recording it in the repository is ``commit_floor``'s job: that runs
        subprocesses, so the loop hands it to a thread rather than blocking
        the event loop here.
    """
    number = 1 + sum(1 for n in pool.names() if n.startswith("champion_"))
    name = f"champion_{number}"
    source = Path(program.source_path).read_text(encoding="utf-8")

    config.CHAMPIONS.mkdir(parents=True, exist_ok=True)
    champion = config.CHAMPIONS / f"{name}.py"
    if champion.exists():
        raise FileExistsError(
            f"{champion} already exists: the pool and the champions directory "
            "disagree about how many champions there have been"
        )
    champion.write_text(source, encoding="utf-8")
    champion.chmod(0o444)

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
    pool.add_champion(name, str(champion), result.rates)
    pool.apply_weakness_pressure(pool.weakest(result.rates))
    pool.save(config.POOL)

    record = Champion(name=name, path=str(champion), result=result)
    _write_champion(record)

    LOGGER.info("promoted %s to %s", program.id, name)
    return record


def commit_floor(champion: Champion, program_id: str) -> None:
    """Record a promotion that already happened on disk in the repository.

    Blocking: two version-control subprocesses. The loop awaits this through
    a thread so the event loop keeps dispatching while it runs.

    Args:
        champion: The record ``promote`` just wrote.
        program_id: The archive id of the program that was promoted.
    """
    result = champion.result
    message = (
        f"feat: promote {program_id} to the floor as {champion.name}\n\n"
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
        LOGGER.warning("promotion commit failed for %s: %s", champion.name, stderr)


def _write_champion(champion: Champion) -> None:
    """Write ``config.CHAMPION`` atomically: a temporary file, then a rename.

    A reader never sees a half-written record, and the file exists in full or
    not at all -- which is what lets ``loop.run`` trust it over ``state.json``.

    Args:
        champion: The record to write.
    """
    config.CHAMPION.parent.mkdir(parents=True, exist_ok=True)
    scratch = config.CHAMPION.with_name(f"{config.CHAMPION.name}.{os.getpid()}.tmp")
    scratch.write_text(champion.model_dump_json(indent=2), encoding="utf-8")
    scratch.replace(config.CHAMPION)


def load_champion() -> Champion | None:
    """The promoted champion on disk, or None if nothing has been promoted."""
    if not config.CHAMPION.exists():
        return None
    return Champion.model_validate_json(config.CHAMPION.read_text(encoding="utf-8"))
