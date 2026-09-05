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

A promotion is two halves, because the campaign runs one event loop over
eight workers. `promote` is the file work -- packaging, the champion's own
copy, the floor -- and goes in a thread. `enroll` and `record` are the
shared state -- the pool the champion joins, and the file that says it is
the floor -- and stay on the loop thread, because `Pool` is one object the
other seven workers are reading while this one promotes.

Nothing here touches version control, and every write is under
`run/campaign`. In particular nothing writes into `src/`: the committed
`served/main.py` is the seed a cold start begins from, not the floor a
campaign produces, and a promotion that dirtied a tracked file would leave
a tree the next launch refuses to start on.
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

FIELD_TOLERANCE = 0.02


class Champion(BaseModel):
    """The promoted floor, as ``config.CHAMPION`` records it.

    ``record`` writes it the moment the champion is in the pool, and it is
    preferred over ``state.json`` on restart: ``state.json`` is written after
    every completed session, so a kill in between would otherwise lose the
    champion and let the gate promote a second time against no baseline.

    Attributes:
        name: The champion's pool name, e.g. "champion_3".
        path: The immutable copy under ``config.CHAMPIONS`` the pool plays.
        tarball: The archive a cut uploads, written by this promotion.
        result: The deep evaluation this champion carries -- the one the
            promotion was decided on, and then the re-score on the pool that
            promotion changed, which is the bar the next candidate must beat.
    """

    name: str
    path: str
    tarball: str
    result: DeepResult


def promotion(candidate: DeepResult, champion: DeepResult | None) -> tuple[bool, str]:
    """Whether ``candidate`` replaces ``champion``, and why not if not.

    Four clauses, cheapest and most explanatory first: the lower bound, the
    field, the worst matchup, and the per-opponent regression.

    The worst-matchup clause is a maximin ratchet, and it is here because
    fitness is a weighted mean and this game is not transitive. A mean lets a
    candidate buy a promotion by crushing whatever carries weight while
    losing badly to something else, and the finale is a wide field where one
    hard counter is exactly the risk that matters. Requiring the candidate's
    worst per-opponent rate to be at least the champion's means the floor
    only ever rises: a good matchup can never be traded for a bad one,
    however high the average. It costs nothing early, because the first
    champion needs no comparison and its own worst rate starts near zero.

    Both that clause and the per-opponent one compare rates rather than
    interval bounds -- deliberately, so they are comparable with each other
    and with the champion's own result, which was measured the same way --
    and both compare only over the opponents the two results have in common,
    because the pool changes between measurements.

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
    both = [name for name in champion.rates if name in candidate.rates]
    if both:
        worst = min(both, key=lambda name: candidate.rates[name])
        floor = min(both, key=lambda name: champion.rates[name])
        if candidate.rates[worst] < champion.rates[floor]:
            return (
                False,
                f"worst matchup {worst} at {candidate.rates[worst]:.3f}, under "
                f"the champion's floor of {champion.rates[floor]:.3f} vs {floor}",
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


def promote(program: Program, result: DeepResult) -> Champion:
    """The file half of a promotion: the tarball, the champion's copy, the floor.

    Every file this writes is one nothing else owns, so it is safe to call
    in a thread; the pool and ``config.CHAMPION`` are ``enroll`` and
    ``record``, on the loop thread.

    Each champion is written once to ``CHAMPIONS/<name>.py`` and it is that
    path the pool registers, so a pool holding N champions holds N different
    programs. ``FLOOR/main.py`` is the current floor and is overwritten every
    promotion; registering it instead would make every pool entry an alias
    for the newest agent and silently erase the history the pool exists to
    keep.

    Args:
        program: The archive entry being promoted.
        result: Its deep evaluation.

    Returns:
        The champion record, for ``enroll`` and ``record`` to act on.

    Raises:
        FileExistsError: The champions directory already holds this name.
    """
    # Numbered off the champions directory, which only ever grows. Counting
    # the pool's `champion_` members instead would renumber after a
    # retirement and hand the sixth promotion a name the fifth already has.
    number = 1 + sum(1 for _ in config.CHAMPIONS.glob("champion_*.py"))
    name = f"champion_{number}"
    kept = config.CHAMPIONS / f"{name}.py"
    if kept.exists():
        raise FileExistsError(
            f"{kept} already exists: something other than a promotion has "
            "written to the champions directory"
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

    LOGGER.info("promoted %s to %s", program.id, name)
    return Champion(name=name, path=str(kept), tarball=str(tarball), result=result)


def enroll(champion: Champion, pool: Pool) -> None:
    """Put ``champion`` in the pool and press on its weakest opponent.

    Mutates ``pool`` in place, so it belongs on whichever thread owns it --
    for the campaign, the event loop. The save follows immediately, so
    nothing between here and it can resolve the champion's name through a
    pool file that does not have it yet.

    Args:
        champion: The record ``promote`` returned.
        pool: The opponent pool, updated and saved in place.
    """
    pool.add_champion(champion.name, champion.path, champion.result.rates)
    pool.apply_weakness_pressure(pool.weakest(champion.result.rates))
    pool.save(config.POOL)


def record(champion: Champion) -> Champion:
    """Write ``config.CHAMPION`` atomically: a temporary file, then a rename.

    A reader never sees a half-written record, and the file exists in full or
    not at all -- which is what lets ``loop.run`` trust it over ``state.json``.

    Args:
        champion: The record to write.

    Returns:
        ``champion``, so a caller can write and keep it in one expression.
    """
    config.CHAMPION.parent.mkdir(parents=True, exist_ok=True)
    scratch = config.CHAMPION.with_name(f"{config.CHAMPION.name}.{os.getpid()}.tmp")
    scratch.write_text(champion.model_dump_json(indent=2), encoding="utf-8")
    scratch.replace(config.CHAMPION)
    return champion


def load_champion() -> Champion | None:
    """The promoted champion on disk, or None if nothing has been promoted."""
    if not config.CHAMPION.exists():
        return None
    return Champion.model_validate_json(config.CHAMPION.read_text(encoding="utf-8"))
