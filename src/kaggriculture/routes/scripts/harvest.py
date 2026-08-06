"""Build the prototype store from every archive in the replay corpus.

Offline only, not part of the submission path -- ``kaggriculture.routes.store``'s
module docstring is the one that has to avoid importing torch, not this
script. Selects every seat ``learn.corpus.select`` would clone, harvests one
archive at a time, dedupes the combined result and writes it to ``STORE``.

``FLOOR`` at 149,120 came from a single archive's top decile. Measured across
the full rating-filtered selection (1,268 seats at ``select``'s own defaults),
the 90th percentile bank is 147,198 -- *below* the floor -- and a 48-sample
probe found only 6% of samples clear it, implying roughly 79 surviving
routes. That may be the right floor, or it may be too strict; this script's
job is to report the per-archive and bank evidence for that call, not to make
it. Its log line and ``log_bank_distribution`` below are the evidence.

Harvesting is done one archive at a time rather than handing every selected
sample to ``harvest`` in one call. ``harvest`` already groups its input by
archive internally to open each zip exactly once, so this costs nothing
extra, and it is the only way to report each archive's own contribution --
see the module docstring's point 3 in the task brief: a total alone hides an
archive that contributed nothing.

Run:
    uv run python -m kaggriculture.routes.scripts.harvest
"""

import json
import logging
import zipfile
from pathlib import Path

from tqdm import tqdm

from kaggriculture.learn.corpus import CORPUS, Sample, select
from kaggriculture.routes import STORE
from kaggriculture.routes.signature import signature
from kaggriculture.routes.store import Prototype, dedupe, save

LOGGER = logging.getLogger(__name__)

FLOOR = 149_120.0
TOLERANCE = 1e-3


def harvest(samples: list[Sample], floor: float) -> list[Prototype]:
    """Turn corpus samples into routes, keeping only the ones worth replaying.

    Lives in this script rather than in ``routes/store.py`` because it is the
    one part of the store that reads the corpus: it needs ``learn.corpus`` and
    ``tqdm``, and ``package.py`` ships neither, while ``routes/store.py``
    itself is imported by the agent at play time.

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
        One ``Prototype`` per sample whose seat's final bank cleared
        ``floor``. Samples are grouped by archive before reading, so this
        matches the input order exactly only when each sample's archive is
        already contiguous in it -- true of ``learn.corpus.select``'s output,
        not guaranteed for an arbitrary caller.
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


def build_store(
    archives: list[Path], floor: float, tolerance: float
) -> list[Prototype]:
    """Select, harvest and dedupe every archive into one prototype store.

    Every archive in ``archives`` gets its own log line, including ones that
    contribute zero selected seats -- a rating floor that no episode in an
    archive clears is silent otherwise, and the store would be quietly
    narrower than its headline count suggests.

    Args:
        archives: Daily corpus archives to build the store from.
        floor: Minimum final bank a route must clear to be kept, forwarded to
            ``harvest``.
        tolerance: Maximum mean per-turn signature distance for two routes to
            be treated as duplicates, forwarded to ``dedupe``.

    Returns:
        The deduped store, ready to save.
    """
    samples = select(archives)
    LOGGER.info("selected %d seats across %d archives", len(samples), len(archives))

    by_archive: dict[str, list[Sample]] = {archive.name: [] for archive in archives}
    for sample in samples:
        by_archive[sample.archive].append(sample)

    prototypes: list[Prototype] = []
    for name, group in by_archive.items():
        harvested = harvest(group, floor)
        LOGGER.info(
            "%s: %d seats selected, %d survived the %.0f floor",
            name,
            len(group),
            len(harvested),
            floor,
        )
        prototypes.extend(harvested)

    deduped = dedupe(prototypes, tolerance)
    LOGGER.info(
        "dedupe kept %d of %d harvested routes at tolerance %s",
        len(deduped),
        len(prototypes),
        tolerance,
    )
    return deduped


def log_bank_distribution(prototypes: list[Prototype]) -> None:
    """Log the min, median, p90 and max bank of what survived into the store.

    This is evidence for whether ``FLOOR`` is set correctly, not a verdict:
    every bank here already cleared the floor by construction, so a p90
    bunched just above it says the floor is binding hard, while a wide spread
    says it is not.

    Args:
        prototypes: The store's final, deduped routes.
    """
    if not prototypes:
        LOGGER.info("bank distribution: store is empty, nothing survived")
        return
    banks = sorted(prototype.bank for prototype in prototypes)
    LOGGER.info(
        "bank distribution over %d routes: min %.0f p50 %.0f p90 %.0f max %.0f",
        len(banks),
        banks[0],
        banks[len(banks) // 2],
        banks[int(0.9 * (len(banks) - 1))],
        banks[-1],
    )


def main() -> None:
    """Build and save the prototype store from every archive on disk."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    archives = sorted(CORPUS.glob("*.zip"))

    prototypes = build_store(archives, FLOOR, TOLERANCE)

    STORE.parent.mkdir(parents=True, exist_ok=True)
    save(prototypes, STORE)
    log_bank_distribution(prototypes)
    LOGGER.info(
        "saved %d routes to %s (%.1f MiB)",
        len(prototypes),
        STORE,
        STORE.stat().st_size / 2**20,
    )


if __name__ == "__main__":
    main()
