"""The one place a candidate becomes the floor.

The promotion rule is the spec's §5.2: the candidate's lower Wilson bound
above the champion's point estimate, the vendored field not down more than
two points, no opponent regressed beyond the wider of the two intervals.

A promotion also produces the artefact a cut uploads: the program is
packaged into `champions/<name>.tar.gz`, so a cut is one command -- upload
`champion.tarball` -- and nothing is built at cut time. What proves a
candidate runs is validation, which loads it through Kaggle's own loader and
plays a full episode; everything else in the tarball is the same bytes for
every candidate, so proving the packaging is the packager's own tests' job
and not something to re-run on each promotion.

Nothing here touches version control. Provenance is `champion.json`, the
champions directory and the archive; every write is under `run/campaign`
except `served/main.py`, which is the copy the packaging entry points read.
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path

from pydantic import BaseModel

from kaggriculture.campaign import config, harness
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
        tarball: The archive a cut uploads, written by this promotion.
        result: The deep evaluation the promotion was decided on.
    """

    name: str
    path: str
    tarball: str
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
    """Ship the tarball, then write the floor, the pool and the champion record.

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

    Raises:
        FileExistsError: The champions directory already holds this name.
    """
    number = 1 + sum(1 for n in pool.names() if n.startswith("champion_"))
    name = f"champion_{number}"
    kept = config.CHAMPIONS / f"{name}.py"
    if kept.exists():
        raise FileExistsError(
            f"{kept} already exists: the pool and the champions directory "
            "disagree about how many champions there have been"
        )
    source = Path(program.source_path)

    with tempfile.TemporaryDirectory() as scratch:
        # The tarball is built in here and moved into place whole. Packaging
        # writes a file at a time and can fail part way through; a half-built
        # archive under `champions/` would be a cut waiting to upload it.
        built = harness.package(source, Path(scratch) / f"{name}.tar.gz")
        config.CHAMPIONS.mkdir(parents=True, exist_ok=True)
        tarball = config.CHAMPIONS / f"{name}.tar.gz"
        shutil.move(str(built), str(tarball))

    code = source.read_text(encoding="utf-8")
    kept.write_text(code, encoding="utf-8")
    kept.chmod(0o444)

    config.FLOOR.mkdir(parents=True, exist_ok=True)
    floor = config.FLOOR / "main.py"
    if floor.exists():
        floor.chmod(0o644)
    floor.write_text(code, encoding="utf-8")
    floor.chmod(0o444)

    SERVED.parent.mkdir(parents=True, exist_ok=True)
    SERVED.write_text(code, encoding="utf-8")

    # add_champion and weakness pressure change the pool in memory; save
    # immediately after so nothing between here and the save can resolve the
    # champion's name through a stale pool.
    pool.add_champion(name, str(kept), result.rates)
    pool.apply_weakness_pressure(pool.weakest(result.rates))
    pool.save(config.POOL)

    record = Champion(name=name, path=str(kept), tarball=str(tarball), result=result)
    _write_champion(record)

    LOGGER.info("promoted %s to %s", program.id, name)
    return record


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
