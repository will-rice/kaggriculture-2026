"""The prototype store: routes harvested from the replay corpus, ready to replay.

A route is a season's worth of ``(signature, action)`` pairs recorded from one
seat of one corpus episode -- the state-key the seat saw on a turn, paired with
what the recorded player did next. ``routes.scripts.harvest`` builds these from
the corpus, keeping only routes whose seat banked above a floor set from the
corpus's own distribution, not the median: the corpus median seat bank is
125,773 and the agent currently shipped banks about 118,000, so replaying a
median route would teach nothing the agent does not already do -- the floor is
the point of the exercise, not a detail. ``dedupe`` then collapses
near-identical routes, because the corpus is dominated by a handful of public
kernels playing the same handful of openings -- every sampled pair of episodes
in one archive was found to share a byte-identical day-1 signature -- and
without dedupe the store would be hundreds of copies of a few routes rather
than coverage of many. (At real scale it collapsed none of 190: routes that
open identically still diverge.)

This module is on the agent path -- ``routes.play`` loads the store here -- so
it imports neither ``torch`` nor anything the submission archive leaves out.
``harvest`` used to live here and pulled in ``learn.corpus`` and ``tqdm`` with
it; ``package.py`` drops ``corpus.py`` from the archive, so that import chain
would have raised ``ModuleNotFoundError`` on turn zero rather than costing
mere seconds. It now lives in ``routes/scripts/harvest.py``, beside the only
caller that ever wanted it, and the whole ``scripts`` package is excluded from
the archive as a directory.
"""

import gzip
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from kaggriculture.constants import TURNS_PER_DAY
from kaggriculture.routes.signature import distance

LOGGER = logging.getLogger(__name__)


class Prototype(BaseModel):
    """One recorded route: a seat's signatures and the actions that produced them.

    ``signatures[i]`` is the state-key the seat saw before choosing
    ``actions[i]`` -- the same ``(observation, next action)`` pairing
    ``kaggriculture.learn.dataset`` uses, since both read the same episode
    shape and the module docstring there explains why the action is offset by
    one index from the observation it was chosen from.

    ``bank`` and ``opponent_bank`` are both seats' money at the episode's
    final observation. A route that banked 158k against a weak opponent is not
    the same achievement as one that banked 150k against a strong one, and
    later selection needs both numbers to say which criterion it used.
    """

    bank: float
    opponent_bank: float
    rating: float
    actions: list[dict[str, Any]]
    signatures: list[tuple[float, ...]]


def _trajectory_distance(a: Prototype, b: Prototype) -> float:
    """Return the mean per-turn signature distance between two routes.

    Uses the same phase-weighted ``distance`` retrieval will match routes
    with, rather than a separate ad hoc metric, so "near-identical" here means
    what it will mean at query time. Turn ``i``'s day is inferred as
    ``i // TURNS_PER_DAY``, since every route starts at the season's first
    turn.

    Args:
        a: A route.
        b: Another route.

    Returns:
        The mean weighted distance over the routes' turns. ``a`` and ``b``
        must have the same number of turns -- every harvested route runs the
        same fixed-length season, so a mismatch means one input is not a real
        route and this raises rather than comparing a truncated prefix.
    """
    turns = list(zip(a.signatures, b.signatures, strict=True))
    total = sum(
        distance(sig_a, sig_b, day=index // TURNS_PER_DAY)
        for index, (sig_a, sig_b) in enumerate(turns)
    )
    return total / len(turns)


def dedupe(prototypes: list[Prototype], tolerance: float) -> list[Prototype]:
    """Collapse near-identical routes, keeping the richest of each group.

    Two routes whose mean per-turn distance is within ``tolerance`` are
    treated as the same route recorded twice. Candidates are visited richest
    first -- sorted by ``bank`` descending, ties broken by ``signatures`` so
    the order is a pure function of the data and never of how the caller
    happened to present it -- so "keep the richest" falls out of a
    first-match traversal instead of needing a replace-if-better branch, and
    the result is stable under shuffling the input. Single-linkage-to-first-
    match clustering is an approximation, not true transitive clustering: a
    chain of routes each within tolerance of its neighbour but not of routes
    two or more steps away can still end up split across groups depending on
    which end of the chain sorts first. It is deterministic given the same
    inputs, which is what nightly reproducibility needs; it does not claim to
    find the globally optimal clustering.

    Args:
        prototypes: Routes to collapse.
        tolerance: Maximum mean per-turn distance for two routes to be
            treated as duplicates.

    Returns:
        One route per near-identical group.
    """
    ordered = sorted(
        prototypes, key=lambda prototype: (-prototype.bank, prototype.signatures)
    )
    kept: list[Prototype] = []
    for candidate in ordered:
        if not any(
            _trajectory_distance(candidate, existing) <= tolerance for existing in kept
        ):
            kept.append(candidate)
    LOGGER.info(
        "dedupe collapsed %d of %d routes at tolerance %s",
        len(prototypes) - len(kept),
        len(prototypes),
        tolerance,
    )
    return kept


def save(prototypes: list[Prototype], path: Path) -> None:
    """Write a store to disk as gzipped JSON.

    Gzip, not raw JSON: the store ships inside the submission archive, where
    every byte counts against Kaggle's package limits.

    Args:
        prototypes: Routes to persist.
        path: Destination file.
    """
    payload = [prototype.model_dump() for prototype in prototypes]
    with gzip.open(path, "wt") as handle:
        json.dump(payload, handle)


def load(path: Path) -> list[Prototype]:
    """Read a store written by ``save``.

    Args:
        path: File written by ``save``.

    Returns:
        The routes it held, in the order they were saved.
    """
    with gzip.open(path, "rt") as handle:
        payload = json.load(handle)
    return [Prototype.model_validate(item) for item in payload]
