"""Measure what separates a top-rated seat from a mid-table one.

Runs the two passes described in ``learn/bands.py``. The cheap pass indexes
every episode in every archive from the manifest and the head of its JSON, which
is enough on its own to answer whether the bank correlates with the ladder. The
deep pass decodes a stratified sample and profiles both seats of each episode.

The sample is stratified by within-day rating band and by day, because the
rating scale inflates by more than a thousand points across the nine archives
and because the meta moves with it. ``DAYS`` selects which archives the deep
pass reads; the cheap pass always reads all of them, so the drift is visible
rather than assumed.

**And stratified by engine build.** The corpus spans five releases of
``kaggle-environments``, 1.32.2 to 1.32.6, and one of them changed the economy:
1.32.6 replaced the town centre's escalating demand curve with a flat rate and
doubled its interval, cutting late-season demand roughly eightfold. At an
unchanged rating the median seat banks 120,800 on 1.32.5 and 80,660 on 1.32.6.
An archive is therefore not one economy -- 268 of the 675 episodes dated
2026-08-07 are on the new build -- and any bank statistic pooled across builds
reads as a finding about players when it is a finding about the release notes.
Every ladder number this writes is cut by ``(archive, build)``.

Five training runs and an evaluator share this box. The pool is small and
nice'd on purpose: the deep pass is bounded by JSON decoding, one episode is
32 MB, and finishing ten minutes sooner is not worth slowing a training run.
"""

import argparse
import json
import logging
import os
import zipfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from kaggriculture.learn import bands as banding
from kaggriculture.learn.bands import BANDS, Indexed, Profile

LOGGER = logging.getLogger(__name__)
# Where the raw per-seat profiles land. Git-ignored: it is 480 episodes of
# per-day series, and the committed artifact is the summary beside the report.
RAW = Path("run/bands")
SUMMARY = Path("docs/research/2026-08-08-what-wins-elo.json")
# The four most recent archives. Older ones are a different game: the
# 2026-07-30 field's best seat rates 1,212 against 3,150 on 2026-08-07, and its
# bank/rating correlation is +0.69 where the recent ones read ~0. Pooling them
# would average two answers that disagree.
DAYS = ("2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07")
# Episodes per band per day. Four bands over four days is 480 episodes and 960
# seats, which puts the standard error of a band's mean bank near 1,000 coins --
# small against the 20,000-coin differences the bands are being asked about.
PER_BAND = 30
# Small and nice'd: the training runs own this box.
WORKERS = 4
NICE = 19
SEED = 0
# Our own scripted agent's per-episode bank, and the public leaderboard score it
# earned on 2026-08-04. Carried here so the corpus percentile of a number we
# control is computed rather than quoted, because it is the single cleanest
# refutation of banking as an objective.
OURS = 161_730.0
OURS_SCORE = 1_014.0
OURS_MEASURED = "2026-08-04"
# Bank buckets for the rating-against-bank table, in coins.
BUCKETS = (0, 60_000, 100_000, 120_000, 140_000, 160_000, 10**9)
# Smallest stratum worth quoting a correlation or a bucket mean from. The
# engine cut splits some archives into a large cell and a sliver, and a
# correlation over forty seats is a number that will be quoted and should not
# be.
MIN_CELL = 200
MIN_BUCKET = 30
# Scalars the band comparison ranks. Everything else in a ``Profile`` is either
# an identifier or a per-day series.
FEATURES = (
    "bank",
    "revenue",
    "sales",
    "units",
    "units_per_sale",
    "mean_sale_price",
    "mean_market_price",
    "realisation",
    "bought",
    "denied",
    "denied_share",
    "crowding",
    "coverage",
    "animals",
    "pasture",
    "plants",
    "tiles",
    "hands",
    "early_bank",
    "late_revenue_share",
    "sale_day",
    "early_units_share",
)


def main() -> None:
    """Index the corpus, profile a stratified sample, and write the summary."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-band", type=int, default=PER_BAND, help="episodes per band per day"
    )
    arguments = parser.parse_args()

    rows = banding.index(banding.archives())
    LOGGER.info("indexed %d episodes", len(rows))
    LOGGER.info("rating attribution converged after %d passes", banding.attribute(rows))
    build = engines(rows)
    LOGGER.info(
        "engine builds across the corpus: %s",
        ", ".join(
            f"{version} x{sum(1 for v in build.values() if v == version)}"
            for version in sorted(set(build.values()))
        ),
    )
    ladder = correlations(rows, build)

    assignment = banding.band_of(rows)
    recent = field(rows, assignment, build)
    chosen = sample(rows, assignment, arguments.per_band)
    LOGGER.info("profiling %d episodes over %d days", len(chosen), len(DAYS))
    measured = run(chosen, assignment)

    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "profiles.json").write_text(
        json.dumps([profile.model_dump() for profile in measured])
    )
    summary = {
        "generated": "2026-08-08",
        "archives": [path.name for path in banding.archives()],
        "deep_days": list(DAYS),
        "episodes_indexed": len(rows),
        "episodes_profiled": len({(p.day, p.episode) for p in measured}),
        "seats_profiled": len(measured),
        "ladder": ladder,
        "field": recent,
        "bands": band_table(measured),
        "separation": separation(measured),
        "paired": paired(measured),
        "margin": margins(measured),
        "denial": denial(measured),
        "season": season(measured),
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2))
    LOGGER.info("wrote %s", SUMMARY)
    report(summary)


def engines(rows: Sequence[Indexed]) -> dict[tuple[str, int], str]:
    """Return the ``kaggle-environments`` build each episode was played on.

    Read from ``module_version``, which the exporter writes into the head of
    every episode beside the team names, so this costs another truncated read
    and no decoding.

    It lives here rather than on ``Indexed`` only because ``learn/bands.py`` is
    being edited concurrently for the same 1.32.6 upgrade and this could not
    extend it without committing that work in progress. It belongs there.

    Args:
        rows: Indexed episodes.

    Returns:
        ``(day, episode) -> build``, e.g. ``"1.32.6"``.
    """
    decoder = json.JSONDecoder()
    wanted: dict[str, set[int]] = {}
    for row in rows:
        wanted.setdefault(row.archive, set()).add(row.episode)
    out: dict[tuple[str, int], str] = {}
    for archive, episodes in wanted.items():
        day = archive[-14:-4]
        with zipfile.ZipFile(banding.CORPUS / archive) as bundle:
            for episode in episodes:
                with bundle.open(f"{episode}.json") as member:
                    head = member.read(banding.HEAD_BYTES).decode("utf-8", "ignore")
                marker = '"module_version": '
                start = head.find(marker)
                if start < 0:
                    continue
                out[day, episode] = decoder.raw_decode(head, start + len(marker))[0]
    return out


def correlations(rows: Sequence[Indexed], build: Mapping[tuple[str, int], str]) -> dict:
    """Return the bank-against-rating correlation, cut by archive and by build.

    The ``(archive, build)`` cells are the only figures here that mean anything.
    Pooling across archives buys a correlation with the calendar, because
    ratings inflate daily; pooling across builds buys a correlation with the
    release notes, because 1.32.6 cut the town's demand and with it every bank
    in the episodes it played. Both pooled figures are reported anyway, so the
    size of what stratifying removes is on the record rather than asserted.

    Args:
        rows: Indexed episodes, already attributed.
        build: The output of ``engines``.

    Returns:
        Pooled, per-day and per-(day, build) correlations, plus the
        within-episode ordering check.
    """
    rating = np.array([value for row in rows for value in row.ratings])
    bank = np.array([value for row in rows for value in row.banks])
    day = np.array([row.day for row in rows for _ in row.ratings])
    engine = np.array(
        [build.get((row.day, row.episode), "") for row in rows for _ in row.ratings]
    )

    def cell(mask: np.ndarray) -> dict:
        """Return one stratum's correlation and central tendencies."""
        return {
            "seats": int(mask.sum()),
            "pearson": _pearson(rating[mask], bank[mask]),
            "spearman": _spearman(rating[mask], bank[mask]),
            "median_bank": float(np.median(bank[mask])),
            "median_rating": float(np.median(rating[mask])),
        }

    per_day = {name: cell(day == name) for name in sorted(set(day))}
    per_build = {
        f"{name} {version}": cell((day == name) & (engine == version))
        for name in sorted(set(day))
        for version in sorted(set(engine[day == name]))
        if ((day == name) & (engine == version)).sum() >= MIN_CELL
    }

    gaps, agree = [], []
    for row in rows:
        first, second = row.ratings
        one, two = row.banks
        if first == second or one == two:
            continue
        gaps.append(abs(first - second))
        agree.append((first > second) == (one > two))
    gaps, agree = np.array(gaps), np.array(agree)
    ordering = {
        "episodes": int(len(agree)),
        "higher_rated_seat_banks_more": float(agree.mean()),
        "by_gap": {
            f"{lo}-{hi}": {
                "episodes": int(((gaps >= lo) & (gaps < hi)).sum()),
                "agreement": float(agree[(gaps >= lo) & (gaps < hi)].mean()),
            }
            for lo, hi in ((0, 25), (25, 50), (50, 100), (100, 200), (200, 1000))
            if ((gaps >= lo) & (gaps < hi)).sum() > 20
        },
    }
    return {
        "seats": int(len(rating)),
        "pooled_pearson": _pearson(rating, bank),
        "pooled_spearman": _spearman(rating, bank),
        "per_day": per_day,
        "per_day_build": per_build,
        "within_episode": ordering,
    }


def field(
    rows: Sequence[Indexed], assignment: dict, build: Mapping[tuple[str, int], str]
) -> dict:
    """Return what the recent field looks like from the index alone.

    Everything here is answerable without decoding an episode, and all of it
    goes to the same question from a different side: read the ladder against the
    bank rather than the bank against the ladder, and ask whether a low bank is
    a property of the seat or of the pair it was drawn into.

    The bucket table is emitted twice, pooled and per build, and the pair is the
    point. Pooled, mean rating falls monotonically from the poorest bucket to
    the richest and looks like "the best bankers are the weakest players".
    Inside one build it is flat, because the poorest buckets are simply where
    1.32.6's episodes live. This is the single place in the study where
    stratifying overturned a conclusion rather than sharpening it, so the
    superseded version stays visible next to the one that replaced it.

    Args:
        rows: Indexed episodes, already attributed.
        assignment: The output of ``band_of``.
        build: The output of ``engines``.

    Returns:
        The bank buckets pooled and per build, where our own agent's bank falls
        in that distribution, whether extreme banks come in pairs, and each
        band's episode total against its own day's median.
    """
    recent = [row for row in rows if row.day in DAYS]
    rating = np.array([value for row in recent for value in row.ratings])
    bank = np.array([value for row in recent for value in row.banks])
    other = np.array([value for row in recent for value in row.banks[::-1]])
    engine = np.array(
        [build.get((row.day, row.episode), "") for row in recent for _ in row.ratings]
    )

    def key(row: Indexed) -> str:
        """Return one episode's build."""
        return build.get((row.day, row.episode), "")

    def table(mask: np.ndarray) -> list[dict]:
        """Return the bank buckets for one stratum of seats."""
        elite = np.percentile(rating[mask], 90.0)
        out = []
        for low, high in zip(BUCKETS[:-1], BUCKETS[1:], strict=True):
            inside = mask & (bank >= low) & (bank < high)
            if inside.sum() < MIN_BUCKET:
                continue
            out.append(
                {
                    "from": low,
                    "to": high,
                    "seats": int(inside.sum()),
                    "mean_rating": float(rating[inside].mean()),
                    "top_decile_share": float((rating[inside] >= elite).mean()),
                }
            )
        return out

    everything = np.ones(bank.size, dtype=bool)
    buckets = table(everything)
    per_build = {
        version: table(engine == version)
        for version in sorted(set(engine))
        if version and (engine == version).sum() >= MIN_CELL
    }

    # How much of a seat's bank is the episode rather than the seat. Both seats
    # trade one book, so if the book is what sets the level this correlation is
    # near 1 and a seat's own contribution is only the margin on top -- which is
    # the mechanism behind every null in this study, and the reason a
    # bank-shaped reward is mostly rewarding the draw.
    pairing = {
        version: {
            "episodes": int(
                len(pairs := [row for row in recent if key(row) == version])
            ),
            "seat_to_opponent_bank": _pearson(
                np.array([row.banks[0] for row in pairs]),
                np.array([row.banks[1] for row in pairs]),
            ),
            "median_seat_bank": float(
                np.median([value for row in pairs for value in row.banks])
            ),
        }
        for version in sorted(set(engine))
        if version and sum(1 for row in recent if key(row) == version) >= MIN_BUCKET
    }

    poor = bank < BUCKETS[1]
    rich = bank >= BUCKETS[-2]
    totals: list[float] = []
    labels: list[str] = []
    for day in DAYS:
        episodes = [row for row in recent if row.day == day]
        if not episodes:
            continue
        banked = np.array([sum(row.banks) for row in episodes])
        middle = float(np.median(banked))
        for row, total in zip(episodes, banked, strict=True):
            band = assignment.get((row.day, row.episode, 0))
            if band is not None:
                totals.append(float(total) / middle)
                labels.append(band)
    relative = np.array(totals)
    tagged = np.array(labels)

    return {
        "days": list(DAYS),
        "seats": int(bank.size),
        "builds": {
            version: int((engine == version).sum()) for version in sorted(set(engine))
        },
        "buckets": buckets,
        "buckets_by_build": per_build,
        "pairing_by_build": pairing,
        "ours": {
            "bank": OURS,
            "leaderboard": OURS_SCORE,
            "measured": OURS_MEASURED,
            "percentile": float((bank < OURS).mean() * 100.0),
        },
        "extremes": {
            "poor_seats": int(poor.sum()),
            "poor_facing_poor": float((other[poor] < BUCKETS[1]).mean()),
            "rich_seats": int(rich.sum()),
            "rich_opponent_median_bank": float(np.median(other[rich])),
        },
        "episode_total_over_day_median": {
            band: {
                "episodes": int((tagged == band).sum()),
                "median": float(np.median(relative[tagged == band])),
            }
            for band in BANDS
            if (tagged == band).any()
        },
    }


def sample(
    rows: Sequence[Indexed], assignment: dict, per_band: int
) -> list[tuple[Indexed, str]]:
    """Return the episodes to decode, stratified by day and band.

    An episode is taken for the band its *seat 0* falls in, and both seats are
    profiled. The ladder matches on rating, so the second seat almost always
    lands in the same band; where it does not, it is counted under its own.

    Args:
        rows: Indexed episodes, already attributed.
        assignment: The output of ``band_of``.
        per_band: How many episodes to take per band per day.

    Returns:
        The chosen episodes, each with the band it was drawn for.
    """
    pools: dict[tuple[str, str], list[Indexed]] = {}
    for row in rows:
        if row.day not in DAYS:
            continue
        band = assignment.get((row.day, row.episode, 0))
        if band is None:
            continue
        pools.setdefault((row.day, band), []).append(row)
    generator = np.random.default_rng(SEED)
    chosen: list[tuple[Indexed, str]] = []
    for day in DAYS:
        for band in BANDS:
            pool = pools.get((day, band), [])
            take = generator.permutation(len(pool))[:per_band]
            chosen.extend((pool[int(index)], band) for index in take)
    return chosen


def run(chosen: Sequence[tuple[Indexed, str]], assignment: dict) -> list[Profile]:
    """Decode and profile every chosen episode.

    Args:
        chosen: Episodes to decode.
        assignment: The output of ``band_of``.

    Returns:
        Every profiled seat.
    """
    out: list[Profile] = []
    with ProcessPoolExecutor(WORKERS, initializer=_polite) as pool:
        futures = [
            pool.submit(_one, row.model_dump(), assignment_of(row, assignment))
            for row, _ in chosen
        ]
        for future in tqdm(futures, desc="episodes"):
            out.extend(Profile(**record) for record in future.result())
    return out


def assignment_of(row: Indexed, assignment: dict) -> dict[int, str]:
    """Return one episode's per-seat bands, as a picklable mapping."""
    return {
        seat: assignment[row.day, row.episode, seat]
        for seat in range(len(row.teams))
        if (row.day, row.episode, seat) in assignment
    }


def _polite() -> None:
    """Drop every worker below the training runs."""
    os.nice(NICE)


def _one(row: dict, seats: dict[int, str]) -> list[dict]:
    """Decode one episode in a worker and return its seats' profiles."""
    meta = Indexed(**row)
    episode = banding.load(banding.CORPUS / meta.archive, meta.episode)
    lookup = {(meta.day, meta.episode, seat): band for seat, band in seats.items()}
    return [profile.model_dump() for profile in banding.profiles(episode, meta, lookup)]


def band_table(measured: Sequence[Profile]) -> dict:
    """Return each band's mean and median for every ranked feature."""
    table = {}
    for band in BANDS:
        rows = [profile for profile in measured if profile.band == band]
        if not rows:
            continue
        table[band] = {
            "seats": len(rows),
            "mean_rating": float(np.mean([row.rating for row in rows])),
            "win_rate": float(np.mean([row.won for row in rows])),
            "modelled_over_actual": float(
                np.sum([row.modelled for row in rows])
                / np.sum([row.revenue for row in rows])
            ),
            **{
                name: {
                    "mean": float(np.mean(values := _feature(rows, name))),
                    "median": float(np.median(values)),
                    "sem": float(np.std(values) / np.sqrt(len(values))),
                }
                for name in FEATURES
            },
        }
    return table


def separation(measured: Sequence[Profile]) -> list[dict]:
    """Rank the features by how far the top band sits from the middle one.

    Standardised mean difference, in pooled standard deviations, so features on
    different scales are comparable. Reported beside the raw means, because a
    large standardised difference on a feature nobody can act on is not a
    finding.

    Args:
        measured: Every profiled seat.

    Returns:
        One entry per feature, largest separation first.
    """
    top = [row for row in measured if row.band == "top"]
    middle = [row for row in measured if row.band == "middle"]
    out = []
    for name in FEATURES:
        a, b = _feature(top, name), _feature(middle, name)
        spread = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
        out.append(
            {
                "feature": name,
                "top": float(np.mean(a)),
                "middle": float(np.mean(b)),
                "difference": float(np.mean(a) - np.mean(b)),
                "cohens_d": float((np.mean(a) - np.mean(b)) / spread)
                if spread
                else 0.0,
                "welch_t": _welch(a, b),
            }
        )
    return sorted(out, key=lambda entry: -abs(entry["cohens_d"]))


def margins(measured: Sequence[Profile]) -> dict:
    """Return how close the decided episodes were, in coins and in proportion.

    The number that says how much of the bank matters. If episodes were decided
    by tens of thousands of coins, out-producing the opponent would be the game;
    they are decided by a couple of thousand on banks of a hundred and twenty
    thousand, which puts the decision inside the execution rather than inside
    the farm.

    Args:
        measured: Every profiled seat.

    Returns:
        The winning margin's median and mean, in coins and as a fraction of the
        two seats' mean bank.
    """
    episodes: dict[tuple[str, int], list[Profile]] = {}
    for profile in measured:
        episodes.setdefault((profile.day, profile.episode), []).append(profile)
    banks = [
        (max(seats[0].bank, seats[1].bank), min(seats[0].bank, seats[1].bank))
        for seats in episodes.values()
        if len(seats) == 2 and seats[0].bank != seats[1].bank
    ]
    gap = np.array([high - low for high, low in banks])
    scale = np.array([(high + low) / 2.0 for high, low in banks])
    return {
        "episodes": len(banks),
        "median_coins": float(np.median(gap)),
        "mean_coins": float(gap.mean()),
        "median_fraction": float(np.median(gap / scale)),
    }


def denial(measured: Sequence[Profile]) -> dict:
    """Return how much of a win the winner's price impact accounts for.

    The paired table says the winner denies more; this says how much more,
    against the thing it has to be measured against, which is the size of the
    win. It also asks the harder question: does a *bigger* denial edge buy a
    *bigger* win? If it does not, the effect is directional rather than a lever
    to pull, and saying so is the difference between a finding and a
    recommendation nobody should follow.

    Args:
        measured: Every profiled seat.

    Returns:
        The mean seat's denial, the edge as a fraction of the winning margin,
        and the correlation between the two.
    """
    episodes: dict[tuple[str, int], list[Profile]] = {}
    for profile in measured:
        episodes.setdefault((profile.day, profile.episode), []).append(profile)
    matched = [
        sorted(seats, key=lambda profile: -profile.bank)
        for seats in episodes.values()
        if len(seats) == 2 and seats[0].bank != seats[1].bank
    ]
    edge = np.array([pair[0].denied - pair[1].denied for pair in matched])
    margin = np.array([pair[0].bank - pair[1].bank for pair in matched])
    return {
        "episodes": len(matched),
        "mean_seat_denies": float(np.mean([row.denied for row in measured])),
        "mean_seat_denies_share": float(
            np.mean([row.denied_share for row in measured])
        ),
        "edge_over_margin_median": float(np.median(edge / margin)),
        "edge_margin_correlation": _pearson(edge, margin),
    }


def paired(measured: Sequence[Profile]) -> list[dict]:
    """Rank the features by how far the episode's winner sits from its loser.

    The band comparison has to average over episodes, and an episode is a large
    random object: its seed decides the weeds, the board and which shops the
    town opens, and both seats live inside all of it. Comparing the two seats of
    the *same* episode holds every one of those fixed, which is why this
    separates features the band table cannot -- and it is the comparison the
    ladder actually runs, since a rating is an accumulation of these.

    It also neutralises the shed inference's blind spot. ``coverage`` is a
    property of the episode and applies to both seats equally, so an undercount
    cannot manufacture a difference between them.

    Args:
        measured: Every profiled seat.

    Returns:
        One entry per feature, largest paired t first. Ties in the bank are
        dropped: an episode with no winner has nothing to say here.
    """
    episodes: dict[tuple[str, int], list[Profile]] = {}
    for profile in measured:
        episodes.setdefault((profile.day, profile.episode), []).append(profile)
    matched = [
        sorted(seats, key=lambda profile: -profile.bank)
        for seats in episodes.values()
        if len(seats) == 2 and seats[0].bank != seats[1].bank
    ]
    winners = [pair[0] for pair in matched]
    losers = [pair[1] for pair in matched]

    out = []
    for name in FEATURES:
        won, lost = _feature(winners, name), _feature(losers, name)
        gap = won - lost
        error = gap.std(ddof=1) / np.sqrt(len(gap))
        decided = gap[gap != 0]
        out.append(
            {
                "feature": name,
                "winner": float(won.mean()),
                "loser": float(lost.mean()),
                "difference": float(gap.mean()),
                "paired_t": float(gap.mean() / error) if error else 0.0,
                "tied": float((gap == 0).mean()),
                "winner_ahead": float((decided > 0).mean()) if decided.size else 0.0,
            }
        )
    return sorted(out, key=lambda entry: -abs(entry["paired_t"]))


def season(measured: Sequence[Profile]) -> dict:
    """Return each band's mean per-day series, for the timing question."""
    keys = (
        "day_bank",
        "day_units",
        "day_revenue",
        "day_animals",
        "day_pasture",
        "day_plants",
        "day_tiles",
        "day_hands",
    )
    out: dict[str, dict[str, list[float]]] = {}
    for band in BANDS:
        rows = [profile for profile in measured if profile.band == band]
        if not rows:
            continue
        out[band] = {}
        for key in keys:
            series = [getattr(row, key) for row in rows]
            width = min(len(entry) for entry in series)
            stack = np.array([entry[:width] for entry in series], dtype=float)
            out[band][key] = [float(value) for value in stack.mean(axis=0)]
    return out


def _feature(rows: Sequence[Profile], name: str) -> np.ndarray:
    """Return one ranked feature's value for every seat.

    Five of the ranked features are not stored on a ``Profile``: they are
    derived so that the ratio is taken per seat and then averaged, rather than
    the other way round, which is a different number.

    Args:
        rows: The seats to read.
        name: The feature.

    Returns:
        One value per seat.
    """
    if name == "units_per_sale":
        return np.array([row.units / row.sales if row.sales else 0.0 for row in rows])
    if name in ("animals", "pasture", "plants", "tiles", "hands"):
        return np.array([float(np.mean(getattr(row, f"day_{name}"))) for row in rows])
    if name == "early_bank":
        return np.array([row.day_bank[9] for row in rows])
    if name == "sale_day":
        return np.array(
            [
                float(
                    (np.array(row.day_units) * np.arange(len(row.day_units))).sum()
                    / total
                )
                if (total := sum(row.day_units))
                else 0.0
                for row in rows
            ]
        )
    if name == "early_units_share":
        return np.array(
            [
                sum(row.day_units[:15]) / total
                if (total := sum(row.day_units))
                else 0.0
                for row in rows
            ]
        )
    if name == "late_revenue_share":
        return np.array(
            [
                sum(row.day_revenue[20:]) / total
                if (total := sum(row.day_revenue))
                else 0.0
                for row in rows
            ]
        )
    return np.array([float(getattr(row, name)) for row in rows])


def _pearson(first: np.ndarray, second: np.ndarray) -> float:
    """Return Pearson's r."""
    return float(np.corrcoef(first, second)[0, 1])


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    """Return Spearman's rho, as Pearson's r on the ranks."""
    return _pearson(
        np.argsort(np.argsort(first)).astype(float),
        np.argsort(np.argsort(second)).astype(float),
    )


def _welch(first: np.ndarray, second: np.ndarray) -> float:
    """Return Welch's t statistic for two samples of unequal variance."""
    spread = np.var(first, ddof=1) / len(first) + np.var(second, ddof=1) / len(second)
    return (
        float((np.mean(first) - np.mean(second)) / np.sqrt(spread)) if spread else 0.0
    )


def report(summary: dict) -> None:
    """Log the headline numbers, so a run says what it found without the file."""
    ladder = summary["ladder"]
    LOGGER.info(
        "bank vs rating, %d seats pooled: pearson %+.3f spearman %+.3f",
        ladder["seats"],
        ladder["pooled_pearson"],
        ladder["pooled_spearman"],
    )
    for day, entry in ladder["per_day"].items():
        LOGGER.info(
            "  %s n=%4d pearson %+.3f spearman %+.3f median bank %8.0f rating %6.0f",
            day,
            entry["seats"],
            entry["pearson"],
            entry["spearman"],
            entry["median_bank"],
            entry["median_rating"],
        )
    recent = summary["field"]
    LOGGER.info(
        "our %.0f-coin agent is at the %.1fth corpus percentile and scored %.0f",
        recent["ours"]["bank"],
        recent["ours"]["percentile"],
        recent["ours"]["leaderboard"],
    )
    for bucket in recent["buckets"]:
        LOGGER.info(
            "  bank %7d-%9d n=%5d mean rating %6.0f top-decile share %.2f",
            bucket["from"],
            bucket["to"],
            bucket["seats"],
            bucket["mean_rating"],
            bucket["top_decile_share"],
        )
    for entry in summary["separation"][:5]:
        LOGGER.info(
            "  band  %-20s top %12.2f middle %12.2f d %+.2f t %+.1f",
            entry["feature"],
            entry["top"],
            entry["middle"],
            entry["cohens_d"],
            entry["welch_t"],
        )
    LOGGER.info(
        "%d decided episodes, median margin %.0f coins (%.1f%% of the bank)",
        summary["margin"]["episodes"],
        summary["margin"]["median_coins"],
        100.0 * summary["margin"]["median_fraction"],
    )
    LOGGER.info(
        "the mean seat denies %.0f coins (%.0f%% of the opponent's take); the "
        "winner's edge is %.2f of the margin and correlates %+.3f with it",
        summary["denial"]["mean_seat_denies"],
        100.0 * summary["denial"]["mean_seat_denies_share"],
        summary["denial"]["edge_over_margin_median"],
        summary["denial"]["edge_margin_correlation"],
    )
    for entry in summary["paired"][:8]:
        LOGGER.info(
            "  pair  %-20s winner %12.3f loser %12.3f t %+.1f ahead in %.2f",
            entry["feature"],
            entry["winner"],
            entry["loser"],
            entry["paired_t"],
            entry["winner_ahead"],
        )


if __name__ == "__main__":
    main()
