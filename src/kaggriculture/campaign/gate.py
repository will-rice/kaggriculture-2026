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
    """The promoted floor, as the runs ``champion.json`` records it.

    ``record`` writes it the moment the champion is in the pool, and it is
    preferred over ``state.json`` on restart: ``state.json`` is written after
    every completed session, so a kill in between would otherwise lose the
    champion the pool file already names.

    Attributes:
        name: The champion's pool name, e.g. "champion_3".
        path: The immutable copy under the runs ``champions`` the pool plays.
        tarball: The archive a cut uploads, written by this promotion.
        result: The measurement it was promoted on, which is also what a
            session is shown of the program it starts from.
    """

    name: str
    path: str
    tarball: str
    result: Result


def refresh(
    name: str,
    against: list[str],
    seeds: Sequence[int],
    workers: int,
    paths: config.Run,
) -> list[tuple[str, str]]:
    """Play one agent's missing edges against ``against``, and keep them.

    `standing` fits a rating over pairings that already exist and plays
    nothing. That is what makes a verdict cheap enough to give every round --
    and it means a pairing nobody has played is simply absent from the fit.

    Absent is fine for an opponent that has been around; it is wrong for a
    champion. A champion joins the pool the moment it is promoted with no
    pairings at all, so until they are played it sits in every fit on a
    single edge: the row of whichever candidate is being judged against it.
    Its rating is then inferred almost entirely from that one result -- beat
    it, and its rating falls far enough to make topping the field easy. Each
    promotion would buy the next one cheaply, which is the ratchet running
    backwards.

    It plays *one agent's* edges rather than every missing pair in the pool.
    Nothing leaves the pool now, so "every missing pair" grows with its
    square: sixty opponents is one thousand seven hundred and seventy
    pairings, and a promotion cannot cost forty thousand games. A new
    champion needs edges to the agents it will be compared against, and
    `Pool.sample` already says which those are.

    Args:
        name: The agent whose edges are wanted, normally a new champion.
        against: The opponents to connect it to.
        seeds: Episode seeds; each pairing is played on all of them, both
            seats, so a pairing is ``2 * len(seeds)`` games.
        workers: Processes to fan the games over.
        paths: The run whose field the pairings are kept in.

    Returns:
        The pairings measured, empty when the field already held them all.

    Raises:
        OpponentCrash: An opponent raised in its own seat.
    """
    games = 2 * len(seeds)
    field = rating.Field.load(paths.field)
    absent = [
        other
        for other in against
        if other != name and other not in field.rates.get(name, {})
    ]
    if not absent:
        return []
    LOGGER.info("field: %d pairing(s) for %s, measuring them", len(absent), name)
    for other in absent:
        field.record(
            name,
            other,
            _rate(roster.path(name, paths.pool), other, seeds, workers, paths.pool),
            games,
        )
    field.save(paths.field)
    return [(name, other) for other in absent]


def _rate(
    agent: Path,
    opponent: str,
    seeds: Sequence[int],
    workers: int,
    pool: Path | None = None,
) -> float:
    """``agent``'s win rate against ``opponent`` over ``seeds``, both seats."""
    played = harness.play(agent, [opponent], list(seeds), workers, pool=pool)
    return sum(
        1.0 if game.ours > game.theirs else 0.5 if game.ours == game.theirs else 0.0
        for game in played
    ) / len(played)


def standing(
    name: str,
    rates: dict[str, float],
    games: int,
    paths: config.Run,
) -> dict[str, float]:
    """The same tournament, over games already played: no new ones.

    A round has just measured its program against the opponents it drew, and
    every pairing anyone has ever played is kept, so the standings that
    verdict needs are a fit and nothing more. That is what lets the loop tell
    a model where it ranks after every round rather than only at the gate.

    Fitted over the whole record rather than over the pool as it stands. That
    distinction is what makes a sampled gate work at all: a candidate draws
    sixteen opponents out of dozens, and it is the pairings among the agents
    it did *not* draw that place it against them. Restricted to the drawn few,
    every candidate would be rated in a private tournament and the numbers
    would not compare.

    Args:
        name: The program's name in the standings.
        rates: Its win rate against each opponent it played.
        games: Games behind each rate.
        paths: The run whose field the pairings are kept in.

    Returns:
        A rating per agent, the program included.
    """
    field = rating.Field.load(paths.field)
    results = field.everything()
    results += [(name, two, rate, games) for two, rate in rates.items()]
    return rating.standings(results)


def promotion(
    standings: dict[str, float], name: str, champion: str | None = None
) -> tuple[bool, str]:
    """Whether the candidate out-rates the floor by more than the margin.

    The bar used to be a rank -- top of a Bradley-Terry tournament over the
    eight pool opponents. A rank was the right shape while the pool *was* the
    tournament and every candidate played all of it. It stopped being right
    for two reasons at once.

    A rank has no margin in it. A candidate a hair above the champion topped
    the table and promoted, and at these sample sizes the hair is usually
    noise; selecting the maximum of a noisy estimator is biased upward by
    construction, which is how 78 of 471 programs once cleared a gate that
    none of them survived a deeper look at.

    And a rank over a *sample* is not a rank. A candidate now draws sixteen
    opponents out of dozens, so "top of the table" would mean top of whichever
    sixteen it happened to draw, and an easy draw would promote.

    A rating has units and one scale for everyone, so the bar can be stated
    outright: sit `config.PROMOTION_MARGIN` log-odds above the floor. That is
    a ratchet -- each champion is measurably better than the one before, on a
    scale anchored by agents that never change.

    Args:
        standings: Every agent's rating, from one fit over the whole record.
        name: The candidate's name in those standings.
        champion: The floor to beat, or None before there is one, when
            leading the field is the bar.

    Returns:
        Whether to promote, and a reason either way.
    """
    ranked = sorted(standings, key=lambda agent: -standings[agent])
    place = ranked.index(name) + 1
    mine = standings[name]
    if champion is None or champion not in standings:
        # Nothing to beat yet, so the field is the bar. This is the first
        # promotion of a run and it happens once.
        if ranked[0] == name:
            return True, f"top of {len(ranked)} at {mine:+.3f}, with no floor yet"
        best = ranked[0]
        return False, (
            f"{place} of {len(ranked)} at {mine:+.3f}, "
            f"below {best} at {standings[best]:+.3f}"
        )
    floor = standings[champion]
    gap = mine - floor
    if gap >= config.PROMOTION_MARGIN:
        return True, (
            f"{mine:+.3f}, {gap:+.3f} above {champion} at {floor:+.3f} "
            f"({place} of {len(ranked)})"
        )
    return False, (
        f"{mine:+.3f}, {gap:+.3f} against {champion} at {floor:+.3f} "
        f"and the bar is {config.PROMOTION_MARGIN:+.3f} ({place} of {len(ranked)})"
    )


def promote(program: Program, result: Result, paths: config.Run) -> Champion:
    """The file half of a promotion: the tarball, the champion's copy, the floor.

    Every file this writes is one nothing else owns, so it is safe to call
    in a thread; the pool and ``champion.json`` are ``enroll`` and
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
        paths: The run the champion is written into.

    Returns:
        The champion record, for ``enroll`` and ``record`` to act on.

    Raises:
        FileExistsError: The champions directory already holds this name.
    """
    # Numbered off the champions directory, which only ever grows. Counting
    # the pool's `champion_` members instead would renumber after a
    # retirement and hand the sixth promotion a name the fifth already has.
    number = 1 + sum(1 for _ in paths.champions.glob("champion_*.py"))
    name = f"champion_{number}"
    kept = paths.champions / f"{name}.py"
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
        paths.champions.mkdir(parents=True, exist_ok=True)
        tarball = paths.champions / f"{name}.tar.gz"
        shutil.move(str(built), str(tarball))

    code = source.read_text(encoding="utf-8")
    kept.write_text(code, encoding="utf-8")
    kept.chmod(0o444)

    paths.floor.mkdir(parents=True, exist_ok=True)
    floor = paths.floor / "main.py"
    if floor.exists():
        floor.chmod(0o644)
    floor.write_text(code, encoding="utf-8")
    floor.chmod(0o444)

    LOGGER.info("promoted %s to %s", program.id, name)
    return Champion(name=name, path=str(kept), tarball=str(tarball), result=result)


def enroll(champion: Champion, pool: Pool, paths: config.Run) -> None:
    """Put ``champion`` in the pool, as a gatekeeper every later candidate faces.

    Mutates ``pool`` in place, so it belongs on whichever thread owns it --
    for the campaign, the event loop. The save follows immediately, so
    nothing between here and it can resolve the champion's name through a
    pool file that does not have it yet.

    Args:
        champion: The record ``promote`` returned.
        pool: The opponent pool, updated and saved in place.
        paths: The run whose pool file it is saved to.
    """
    pool.add_champion(champion.name, champion.path)
    pool.save(paths.pool)
    LOGGER.info("%s joined the pool", champion.name)


def record(champion: Champion, paths: config.Run) -> Champion:
    """Write the run's ``champion.json`` atomically: a temp file, then a rename.

    A reader never sees a half-written record, and the file exists in full or
    not at all -- which is what lets ``loop.run`` trust it over ``state.json``.

    Args:
        champion: The record to write.
        paths: The run it is written into.

    Returns:
        ``champion``, so a caller can write and keep it in one expression.
    """
    paths.champion.parent.mkdir(parents=True, exist_ok=True)
    scratch = paths.champion.with_name(f"{paths.champion.name}.{os.getpid()}.tmp")
    scratch.write_text(champion.model_dump_json(indent=2), encoding="utf-8")
    scratch.replace(paths.champion)
    return champion


def load_champion(paths: config.Run) -> Champion | None:
    """The promoted champion on disk, or None if nothing has been promoted."""
    if not paths.champion.exists():
        return None
    return Champion.model_validate_json(paths.champion.read_text(encoding="utf-8"))
