"""The prototype store: routes harvested from the replay corpus, ready to replay.

A route is a season's worth of ``(signature, action)`` pairs recorded from one
seat of one corpus episode -- the state-key the seat saw on a turn, paired with
what the recorded player did next. ``harvest`` builds these from the corpus,
keeping only routes whose seat banked above a floor set from the corpus's own
distribution, not the median: the corpus median seat bank is 125,773 and the
agent currently shipped banks about 118,000, so replaying a median route would
teach nothing the agent does not already do -- the floor is the point of the
exercise, not a detail. ``dedupe`` then collapses near-identical routes,
because the corpus is dominated by a handful of public kernels playing the
same handful of openings -- every sampled pair of episodes in one archive was
found to share a byte-identical day-1 signature -- and without dedupe the
store would be hundreds of copies of a few routes rather than coverage of many.

This module must not import ``torch``. The agent path loads route memory
before it loads the learned model, and ``routes.signature``'s module docstring
measured importing torch alone at 10.7 seconds of the submission sandbox's
60-second overage pool -- route memory has to be usable before that cost is
ever paid.
"""

import gzip
import json
import logging
import zipfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from tqdm import tqdm

from kaggriculture.constants import TURNS_PER_DAY
from kaggriculture.learn.corpus import CORPUS, Sample
from kaggriculture.routes.signature import distance, signature

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


def harvest(samples: list[Sample], floor: float) -> list[Prototype]:
    """Turn corpus samples into routes, keeping only the ones worth replaying.

    Streams each sample's episode straight out of its archive -- never
    extracted, since an episode is ~27 MB and the corpus is ~107 GB
    uncompressed. A route pairs ``signature(observation[i], seat)`` with the
    action recorded at ``steps[i + 1][seat]["action"]``, the decision made
    *from* that state; the final step is a terminal observation with no
    following action and is read only for its bank.

    Args:
        samples: Seats to harvest, as returned by ``learn.corpus.select``.
        floor: Minimum final bank a route must clear to be kept.

    Returns:
        One ``Prototype`` per sample whose seat's final bank cleared ``floor``,
        in the order the samples were given.
    """
    by_archive: dict[str, list[Sample]] = {}
    for sample in samples:
        by_archive.setdefault(sample.archive, []).append(sample)

    prototypes: list[Prototype] = []
    for archive, group in by_archive.items():
        with zipfile.ZipFile(CORPUS / archive) as bundle:
            for sample in tqdm(group, desc=archive, unit="ep"):
                with bundle.open(sample.name) as member:
                    steps = json.load(member)["steps"]
                farms = steps[-1][sample.seat]["observation"]["farms"]
                bank = float(farms[sample.seat]["money"])
                if bank < floor:
                    continue
                opponent_bank = float(farms[1 - sample.seat]["money"])
                prototypes.append(
                    Prototype(
                        bank=bank,
                        opponent_bank=opponent_bank,
                        rating=sample.rating,
                        actions=[
                            steps[index + 1][sample.seat]["action"]
                            for index in range(len(steps) - 1)
                        ],
                        signatures=[
                            signature(
                                steps[index][sample.seat]["observation"], sample.seat
                            )
                            for index in range(len(steps) - 1)
                        ],
                    )
                )
    LOGGER.info(
        "harvested %d of %d samples above a %.0f bank floor",
        len(prototypes),
        len(samples),
        floor,
    )
    return prototypes


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
    treated as the same route recorded twice, and only the one with the
    higher ``bank`` survives -- the store should trend toward the best
    exemplar of each strategy, not the first one harvested.

    Args:
        prototypes: Routes to collapse, in harvest order.
        tolerance: Maximum mean per-turn distance for two routes to be
            treated as duplicates.

    Returns:
        One route per near-identical group.
    """
    kept: list[Prototype] = []
    for candidate in prototypes:
        group = next(
            (
                index
                for index, existing in enumerate(kept)
                if _trajectory_distance(candidate, existing) <= tolerance
            ),
            None,
        )
        if group is None:
            kept.append(candidate)
        elif candidate.bank > kept[group].bank:
            kept[group] = candidate
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
