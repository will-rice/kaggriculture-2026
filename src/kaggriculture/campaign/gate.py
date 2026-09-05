"""The one place a candidate becomes the floor.

The promotion rule is one question: does this candidate beat every opponent
in the pool? A champion is two things at once -- the program we would submit
and a gatekeeper every later candidate has to get past -- and a program that
loses to four of the six vendored kernels is no good as either. Nothing is
compared with the champion, because the champion is in the pool: beating it
is part of beating them all.

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


class Champion(BaseModel):
    """The promoted floor, as ``config.CHAMPION`` records it.

    ``record`` writes it the moment the champion is in the pool, and it is
    preferred over ``state.json`` on restart: ``state.json`` is written after
    every completed session, so a kill in between would otherwise lose the
    champion the pool file already names.

    Attributes:
        name: The champion's pool name, e.g. "champion_3".
        path: The immutable copy under ``config.CHAMPIONS`` the pool plays.
        tarball: The archive a cut uploads, written by this promotion.
        result: The deep evaluation it was promoted on, which is also what a
            session is shown of the program it starts from.
    """

    name: str
    path: str
    tarball: str
    result: DeepResult


def promotion(rates: dict[str, float]) -> tuple[bool, str]:
    """Whether a program clears the bar: it beats every pool opponent.

    One clause, absolute. "Beats" means strictly more than half the games
    against that opponent over both seats: a dead heat at 0.5 is not a win
    and does not pass. Held-out opponents are measured and logged but are
    never in this dict -- they are the generalisation number, not the bar.

    This is the campaign's only reading of "did it win", and it is applied
    twice: to a candidate's sealed-block rates, where it decides a promotion,
    and to a round's fast rates, where it is the verdict the loop sends the
    model and the condition that ends a session. Two implementations would
    let a model believe it had cleared a bar the gate then refused.

    The reason names every opponent the program failed to beat and its rate
    against each, because when nothing is promoting for a week that list is
    what says why.

    Args:
        rates: Win rate per pool opponent.

    Returns:
        Whether it clears the bar, and a reason (why not, or "beat every
        opponent").
    """
    lost = sorted((name, rate) for name, rate in rates.items() if rate <= 0.5)
    if lost:
        return False, "did not beat " + ", ".join(
            f"{name} at {rate:.3f}" for name, rate in lost
        )
    return True, "beat every opponent"


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
    """Put ``champion`` in the pool, as a gatekeeper every later candidate faces.

    Mutates ``pool`` in place, so it belongs on whichever thread owns it --
    for the campaign, the event loop. The save follows immediately, so
    nothing between here and it can resolve the champion's name through a
    pool file that does not have it yet.

    Args:
        champion: The record ``promote`` returned.
        pool: The opponent pool, updated and saved in place.
    """
    retired = pool.add_champion(champion.name, champion.path, champion.result.rates)
    pool.save(config.POOL)
    LOGGER.info(
        "%s joined the pool%s",
        champion.name,
        f", retiring {retired}" if retired else "",
    )


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
