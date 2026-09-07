"""Both sides of a recorded game, measured the same way at each day's close.

Paired within one episode, so a difference is not about the map, the prices or
the opponent -- those are shared. What is left is what the two players did,
which is the only thing a claim can be about.

The quantities are `strategy.QUANTITIES`, and they are here rather than in the
claim store because they are what the corpus can be asked: a claim naming
anything else is refused at the point of writing rather than stored and
quietly never checked.
"""

import collections
import functools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from tqdm import tqdm

from kaggriculture.campaign import strategy, tapes

DAYS = strategy.DAYS
LAST_HOUR = 23
# Processes a whole-corpus pass divides over. Well under the box, because the
# campaign loop is usually running while this does and the loop is what must
# not slow down: this job can take a few more minutes, a round cannot.
WORKERS = 8


def sides(episode: tapes.Episode) -> dict[int, dict[str, dict[str, float]]] | None:
    """Every quantity for the winning and losing side, day by day.

    Args:
        episode: A recorded game.

    Returns:
        ``{day: {"winner": {...}, "loser": {...}}}``, or None for a draw,
        which has no winning side to learn from.
    """
    final = episode.steps[-1][0]["observation"]["farms"]
    money = [float(farm["money"]) for farm in final]
    if money[0] == money[1]:
        return None
    winner = 0 if money[0] > money[1] else 1

    orders: dict[int, collections.Counter] = {
        0: collections.Counter(),
        1: collections.Counter(),
    }
    measured: dict[int, dict[str, dict[str, float]]] = {}
    for index in range(1, len(episode.steps)):
        for seat in (0, 1):
            action = episode.steps[index][seat].get("action") or {}
            for order in action.get("market") or ():
                name = order[0] if isinstance(order, list) else str(order)
                orders[seat][name] += 1
        # The action at step t was chosen looking at step t-1, so the state a
        # day closed on is the observation at that step, not the next one.
        day, hour = divmod(index - 1, 24)
        if hour != LAST_HOUR or day >= DAYS:
            continue
        measured[day] = _close(episode, index - 1, winner, orders)
    # The last day closes on the final recorded step, which no action follows,
    # so the walk above cannot reach it -- and the last day is exactly where a
    # claim about liquidating or still planting has to be decided. Two earlier
    # versions of this measurement silently had no day 29 at all.
    measured[DAYS - 1] = _close(episode, len(episode.steps) - 1, winner, orders)
    return measured


def _close(
    episode: tapes.Episode,
    index: int,
    winner: int,
    orders: dict[int, collections.Counter],
) -> dict[str, dict[str, float]]:
    """Both sides as they stood at one recorded step."""
    return {
        ("winner" if seat == winner else "loser"): _quantities(
            episode.steps[index][seat]["observation"], seat, orders[seat]
        )
        for seat in (0, 1)
    }


def _quantities(
    observation: dict[str, Any], seat: int, orders: collections.Counter
) -> dict[str, float]:
    """Everything measurable about one side at one moment."""
    farm = observation["farms"][seat]
    private = observation.get("private") or {}
    tiles = [tile for row in farm["tiles"] for tile in row if isinstance(tile, dict)]
    return {
        "bank": float(farm["money"]),
        "planted": sum(1 for tile in tiles if tile.get("kind") == "PLANT"),
        "ripe": sum(1 for tile in tiles if tile.get("yield_units", 0) > 0),
        "pens": sum(1 for tile in tiles if tile.get("kind") in ("COOP", "PASTURE")),
        "hands": len(farm.get("hands") or []),
        "quadrants": len(farm.get("unlocked_quadrants") or []),
        "shed": sum((private.get("shed") or {}).values()),
        "seeds": sum((private.get("seeds") or {}).values()),
        "sells": orders["SELL"],
        "hires": orders["HIRE"],
        "land": orders["BUY_LAND"],
    }


def tally(archive: Path, forms: dict[str, strategy.Form]) -> dict[str, tuple[int, int]]:
    """Count agreement for every form over one archive's games.

    One walk for all of them: the walk is what costs, and a form is a lookup
    once a game has been measured. So the store can hold hundreds of claims
    without a pass over the corpus costing any more than one.

    Args:
        archive: One daily archive of recorded games.
        forms: The measurable half of each claim, by claim id.

    Returns:
        ``{claim id: (agreed, differed)}`` -- games that agreed with the form,
        and games where the two sides differed at all. Draws and games where
        both sides matched on the quantity are in neither.
    """
    counts = {claim_id: [0, 0] for claim_id in forms}
    for episode in tapes.qualifying_episodes(archive):
        game = sides(episode)
        if game is None:
            continue
        for claim_id, form in forms.items():
            day = game.get(form.day)
            if day is None:
                continue
            verdict = form.holds(
                day["winner"][form.quantity], day["loser"][form.quantity]
            )
            if verdict is None:
                continue
            counts[claim_id][0] += int(verdict)
            counts[claim_id][1] += 1
    return {claim_id: (agreed, seen) for claim_id, (agreed, seen) in counts.items()}


def measure(
    store: strategy.Strategies, corpus: list[Path], workers: int = WORKERS
) -> dict[str, tuple[int, float]]:
    """Test every claim in ``store`` against every game in ``corpus``.

    A day's archive is the unit of work: nothing in one depends on another and
    a whole one fits a worker, so the corpus divides over processes without
    any of them agreeing about anything. Read whole rather than sampled --
    a claim confirmed at sixty games and refuted at four hundred is the case
    this exists to catch, and the only way to stop being wrong about which is
    to stop reading a corner of the corpus.

    Args:
        store: The claims to test. Every one is measured, the confirmed
            included.
        corpus: The archives to read, from `tapes.archives`. Passed rather
            than discovered so a caller decides what "every game" means.
        workers: Processes to divide the archives over. The default leaves
            the campaign loop its cores; the loop is the thing that must not
            slow down.

    Returns:
        ``{claim id: (support, agreement)}`` for every claim any game could
        speak to.
    """
    forms = {claim.id: claim.form for claim in store.claims}
    if not forms:
        return {}
    agreed: dict[str, list[int]] = {claim_id: [0, 0] for claim_id in forms}
    with ProcessPoolExecutor(max_workers=min(workers, len(corpus))) as pool:
        counted = pool.map(functools.partial(tally, forms=forms), corpus)
        for archive in tqdm(counted, total=len(corpus), desc="archives"):
            for claim_id, (yes, seen) in archive.items():
                agreed[claim_id][0] += yes
                agreed[claim_id][1] += seen
    out: dict[str, tuple[int, float]] = {}
    for claim_id, (yes, total) in agreed.items():
        if not total:
            continue
        store.record(claim_id, total, yes / total)
        out[claim_id] = (total, yes / total)
    return out
