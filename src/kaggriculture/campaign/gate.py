"""The one place a candidate becomes the floor.

The gate is a tournament. Every agent plays every other -- the candidate
against the pool and the pool against itself -- one Bradley-Terry fit ranks
them all, and a candidate is promoted when it comes out top. That is the
competition's own reading of better: the finale is a single Bradley-Terry
tournament over the episodes that keep running past the deadline, and a
leaderboard position is a skill rating.

It used to ask something else -- beat *every* opponent -- which is a minimum
where the ladder takes a strength-weighted view. Measured on the pool as it
stood on 2026-09-06, exactly one published agent cleared that bar, and the
second-strongest agent in the whole field was turned away for a single
matchup at 0.062 while winning 78.6% of everything else.

The pool is dynamic -- a champion joins and the weakest opponent makes way --
and the tournament is over the pool as it stands. What is not replayed is the
pool's games against itself: those are constants, so they are measured once
and kept, and only a new member's pairings ever run. The candidate's own row
is always played fresh, because a candidate has no history.

Champions accumulating in the pool are what makes the chain a ratchet: each
promotion came top of a field that already held every champion before it,
on seeds drawn fresh for that tournament and never seen by the lineage
being judged.

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
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from kaggriculture.campaign import config, harness, rating, roster
from kaggriculture.campaign.archive import Program
from kaggriculture.campaign.evaluator import Result
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
        result: The measurement it was promoted on, which is also what a
            session is shown of the program it starts from.
    """

    name: str
    path: str
    tarball: str
    result: Result


def refresh(
    pool: Pool,
    seeds: Sequence[int],
    workers: int,
    kept: Path | None = None,
) -> list[tuple[str, str]]:
    """Play the pool's own pairings that have never been played, and keep them.

    `standing` fits a tournament over pairings that already exist and plays
    nothing. That is what makes a verdict cheap enough to give every round --
    and it means a pairing nobody has played is silently absent from the fit,
    because `Field.results` returns only what it holds.

    Which is fine for the vendored opponents, whose pairings were measured
    once and never change, and wrong for a champion. A champion joins the pool
    the moment it is promoted and has no pairings at all, so until they are
    played it appears in every tournament with a single edge: the row of
    whichever candidate is being judged against it. Its rating is then
    inferred almost entirely from that one result -- beat it, and its rating
    falls far enough that topping the standings is easy. Each promotion would
    make the next one cheaper, which is the ratchet running backwards.

    So this is called after a promotion, and it is the only thing that writes
    the field.

    Args:
        pool: The opponents as they stand, the new champion included.
        seeds: Episode seeds; each pairing is played on all of them, both
            seats, so a pairing is ``2 * len(seeds)`` games.
        workers: Processes to fan the games over.
        kept: Where the pool's own pairings live between gates.

    Returns:
        The pairings measured, empty when the field already held them all.

    Raises:
        OpponentCrash: A pool opponent raised in its own seat.
    """
    opponents = pool.names()
    games = 2 * len(seeds)

    # Resolved here rather than defaulted in the signature: a default is
    # evaluated once, at definition, so `kept=config.FIELD` captured the
    # path as it was at import and no amount of monkeypatching moved it.
    # A dry run and every test that promoted therefore wrote their
    # champions into the live campaign's pairings.
    kept = kept or config.FIELD
    field = rating.Field.load(kept)
    # The cache carries one game count for every pairing in it, so a field
    # measured at a different count cannot be extended -- recording a new
    # pairing would relabel the old ones as having been played over games they
    # were not, and the fit weights by that number. The rates are a cache of
    # constants and re-measuring them is what the cache exists to avoid, but
    # keeping them mislabelled is worse than paying for them again.
    if field.rates and field.games != games:
        LOGGER.info(
            "field was measured over %d games and this gate plays %d: "
            "discarding %d cached pairing(s) and measuring them again",
            field.games,
            games,
            sum(len(row) for row in field.rates.values()) // 2,
        )
        field = rating.Field()
    absent = field.missing(opponents)
    if not absent:
        return []
    LOGGER.info("field: %d pairing(s) never played, measuring them", len(absent))
    for one, two in absent:
        field.record(one, two, _rate(roster.path(one), two, seeds, workers))
    field.games = games
    field.save(kept)
    return absent


def _rate(agent: Path, opponent: str, seeds: Sequence[int], workers: int) -> float:
    """``agent``'s win rate against ``opponent`` over ``seeds``, both seats."""
    played = harness.play(agent, [opponent], list(seeds), workers)
    return sum(
        1.0 if game.ours > game.theirs else 0.5 if game.ours == game.theirs else 0.0
        for game in played
    ) / len(played)


def standing(
    name: str,
    rates: dict[str, float],
    pool: Pool,
    games: int,
    kept: Path | None = None,
) -> dict[str, float]:
    """The same tournament, over games already played: no new ones.

    A round has just measured its program against every pool opponent, and the
    pool's own pairings are kept, so the standings that verdict needs are a
    fit and nothing more. That is what lets the loop tell a model where it
    ranks after every round rather than only at the gate.

    Args:
        name: The program's name in the standings.
        rates: Its win rate against each pool opponent.
        pool: The opponents those rates are against.
        games: Games behind each rate.
        kept: Where the pool's own pairings live.

    Returns:
        A rating per agent, the program included.
    """
    # Resolved here rather than defaulted in the signature: a default is
    # evaluated once, at definition, so `kept=config.FIELD` captured the
    # path as it was at import and no amount of monkeypatching moved it.
    # A dry run and every test that promoted therefore wrote their
    # champions into the live campaign's pairings.
    kept = kept or config.FIELD
    field = rating.Field.load(kept)
    opponents = [n for n in pool.names() if n in rates]
    results = field.results(opponents)
    results += [(name, two, rates[two], games) for two in opponents]
    return rating.standings(results)


def promotion(standings: dict[str, float], name: str) -> tuple[bool, str]:
    """Whether the candidate beat the pool: it came out top of the tournament.

    One clause, and it is the competition's own reading of better. The finale
    is a single Bradley-Terry tournament and a leaderboard position is a skill
    rating, so the gate asks what the ladder asks: not "did it beat every
    opponent", which is a minimum and which the ladder never asks, but "did it
    rank above them all".

    The two differ exactly where the field is not transitive. Measured on the
    pool as it stood on 2026-09-06, the second-strongest published agent wins
    78.6% of everything and loses one matchup at 0.062: top of a tournament,
    and turned away by a rule that wants no weakness.

    Args:
        standings: Every agent's rating from one tournament.
        name: The candidate's name in those standings.

    Returns:
        Whether to promote, and a reason either way.
    """
    ranked = sorted(standings, key=lambda agent: -standings[agent])
    if ranked[0] == name:
        second = ranked[1]
        return True, (
            f"top of the tournament at {standings[name]:+.3f}, "
            f"above {second} at {standings[second]:+.3f}"
        )
    best = ranked[0]
    place = ranked.index(name) + 1
    return False, (
        f"{place} of {len(ranked)} at {standings[name]:+.3f}, "
        f"below {best} at {standings[best]:+.3f}"
    )


def promote(program: Program, result: Result) -> Champion:
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
        result: Its measurement.

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
    pool.add_champion(champion.name, champion.path)
    pool.save(config.POOL)
    LOGGER.info("%s joined the pool", champion.name)


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
