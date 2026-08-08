"""Comparing what strong seats do against what mid-table seats do.

The whole project has optimised the bank, and nothing had ever checked that the
bank is what the ladder rewards. The published archives make that checkable
without a single episode being replayed: each one ships a ``manifest.csv`` of
per-episode ladder ratings, and each episode's JSON carries its two team names
and both final banks in its first seven kilobytes -- before the ``steps`` key,
so a truncated read gets them for a fifth of a millisecond per episode.

Two things make the comparison harder than banding on the manifest:

- **The manifest does not say which seat is which.** It carries ``min_score``
  and ``avg_score``, so an episode's two ratings are known as an unordered pair.
  ``attribute`` resolves the pair against the team names, which are ordered, by
  the one fact that constrains it: a player carries one rating across every
  episode it plays that day. See its docstring for the convergence diagnostic.
- **The rating scale inflates daily.** The 2026-07-30 archive's best seat rates
  1,212; the 2026-08-07 archive's rates 3,150. Pooling nine days of ratings
  compares a July median against an August top decile, so every band in here is
  a *within-day* percentile.

The deep pass decodes sampled episodes and profiles both seats. It reuses
``sales.sale_metrics`` and ``sales.buy_units`` verbatim rather than restating
the completion rule, and the per-good breakdown that the market-impact
measurement needs is built on ``clears``, which is the primitive
``sale_metrics`` aggregates -- ``tests/learn/test_bands.py`` asserts on real
episodes that summing ``clears`` reproduces ``sale_metrics`` exactly, so the two
cannot drift apart.
"""

import csv
import io
import json
import logging
import math
import zipfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from kaggriculture.constants import (
    MARKET_PARAMS,
    PRODUCTS,
    SEASON_DAYS,
    SHOPS,
    TOWN_CENTER_DEMAND_SCHEDULE,
    TOWN_CENTER_PRODUCTS,
    TOWN_CENTER_SELL_INTERVAL,
    TOWN_SHOP_SELL_INTERVAL,
    TURNS_PER_DAY,
    market_price,
)
from kaggriculture.learn.corpus import CORPUS
from kaggriculture.learn.sales import buy_units, sale_metrics

LOGGER = logging.getLogger(__name__)
DECODER = json.JSONDecoder()
# `info` and `rewards` both sit before `steps` in every episode written by the
# Kaggle exporter, and `steps` is 32 MB. Eight kilobytes reaches them with room
# to spare and is the difference between indexing the corpus in nine seconds and
# in three hours.
HEAD_BYTES = 8192
# Within-day rating percentiles. The ladder matches on rating -- an episode's
# two seats are within 82 rating points of each other at the median -- so a
# band is in practice a set of episodes rather than a set of lone seats.
BANDS = {
    "top": (90.0, 100.0),
    "upper": (65.0, 90.0),
    "middle": (35.0, 65.0),
    "low": (10.0, 35.0),
}


class Indexed(BaseModel):
    """One episode, from its manifest row and the head of its JSON.

    Attributes:
        archive: The daily ``.zip`` this came from.
        day: The archive's date, which is the rating scale these ratings are on.
        episode: Kaggle's episode id, which is also the member name.
        avg: Mean of the two seats' ladder ratings.
        low: The weaker seat's rating (``min_score``).
        high: The stronger seat's rating, from ``sum_score - min_score``.
        teams: Team names in seat order.
        banks: Final banked coins in seat order.
        ratings: Ratings in seat order once ``attribute`` has run.
    """

    archive: str
    day: str
    episode: int
    avg: float
    low: float
    high: float
    teams: list[str]
    banks: list[float]
    ratings: list[float] = []


def index(archives: Sequence[Path]) -> list[Indexed]:
    """Return every episode's ratings, players and final banks, without decoding.

    Args:
        archives: Daily ``.zip`` archives.

    Returns:
        One row per episode that appears in both the manifest and the archive.
    """
    rows: list[Indexed] = []
    for archive in archives:
        day = archive.stem[-10:]
        with zipfile.ZipFile(archive) as bundle:
            manifest = {
                int(entry["episode_id"]): entry
                for entry in csv.DictReader(
                    io.StringIO(bundle.read("manifest.csv").decode())
                )
            }
            for name in bundle.namelist():
                if not name.endswith(".json"):
                    continue
                entry = manifest.get(int(name.removesuffix(".json")))
                if entry is None:
                    continue
                with bundle.open(name) as member:
                    head = member.read(HEAD_BYTES).decode("utf-8", "ignore")
                rows.append(
                    Indexed(
                        archive=archive.name,
                        day=day,
                        episode=int(entry["episode_id"]),
                        avg=float(entry["avg_score"]),
                        low=float(entry["min_score"]),
                        high=float(entry["sum_score"]) - float(entry["min_score"]),
                        teams=_head_field(head, "TeamNames"),
                        banks=_head_field(head, "rewards"),
                    )
                )
        LOGGER.info("indexed %s: %d episodes", archive.name, len(rows))
    return rows


def _head_field(head: str, key: str) -> list:
    """Decode one JSON value out of a document truncated mid-``steps``.

    A regex cannot do this: team names are arbitrary user strings and several in
    this corpus contain brackets and escaped quotes. ``raw_decode`` stops at the
    end of the value it was pointed at and never looks at the truncation.

    Args:
        head: The first bytes of an episode's JSON, decoded as text.
        key: The key whose value to return.

    Returns:
        The decoded value.
    """
    marker = f'"{key}": '
    start = head.find(marker)
    if start < 0:
        raise ValueError(f"{key} not found in the first {len(head)} characters")
    value, _ = DECODER.raw_decode(head, start + len(marker))
    return value


def attribute(rows: Sequence[Indexed], rounds: int = 60) -> int:
    """Fill in each episode's per-seat ratings, in place.

    The manifest gives an episode's two ratings as an unordered pair and the
    episode gives its two team names in seat order; neither says which rating
    belongs to which seat. What ties them together is that a player carries one
    rating through every episode it plays on a day, so the assignment that makes
    each player's ratings consistent is the right one.

    This is the obvious fixed point: estimate every player's rating as the mean
    of what it has been assigned so far, then re-assign each episode's higher
    rating to whichever of its two seats currently estimates higher. Started
    from the episode averages it converges in a handful of passes.

    It is checkable without trusting it. If the assignment were noise, the
    higher-rated seat would out-bank the lower-rated one in half of all
    episodes, at every rating gap. It does so in 68% of them overall and in 82%
    of the episodes whose two seats are 100 to 200 rating points apart -- an
    ordering signal that gets stronger exactly where the pair is easier to
    order, which noise cannot produce.

    Args:
        rows: Indexed episodes, mutated in place.
        rounds: Cap on the number of passes.

    Returns:
        How many passes it took to stop changing.
    """
    estimate: dict[tuple[str, str], float] = {}
    previous: list[list[float]] | None = None
    for row in rows:
        row.ratings = [row.avg, row.avg]
    for step in range(rounds):
        totals: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in rows:
            for name, rating in zip(row.teams, row.ratings, strict=True):
                totals[row.day, name].append(rating)
        estimate = {key: sum(v) / len(v) for key, v in totals.items()}
        for row in rows:
            first, second = row.teams
            a = estimate.get((row.day, first), row.avg)
            b = estimate.get((row.day, second), row.avg)
            row.ratings = [row.high, row.low] if a > b else [row.low, row.high]
        current = [list(row.ratings) for row in rows]
        if current == previous:
            return step
        previous = current
    return rounds


def band_of(rows: Sequence[Indexed]) -> dict[tuple[str, int, int], str]:
    """Return each seat's within-day rating band.

    The rating scale inflates by more than a thousand points across the nine
    archives, so a band has to be a rank inside one day rather than an absolute
    rating.

    Args:
        rows: Indexed episodes, already attributed.

    Returns:
        ``(day, episode, seat) -> band name``, omitting seats in no band.
    """
    per_day: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        per_day[row.day].extend(row.ratings)
    cuts = {
        day: {
            name: (_percentile(values, lo), _percentile(values, hi))
            for name, (lo, hi) in BANDS.items()
        }
        for day, values in per_day.items()
    }
    out: dict[tuple[str, int, int], str] = {}
    for row in rows:
        for seat, rating in enumerate(row.ratings):
            for name, (lo, hi) in cuts[row.day].items():
                if lo <= rating <= hi:
                    out[row.day, row.episode, seat] = name
                    break
    return out


def _percentile(values: Sequence[float], percent: float) -> float:
    """Return the linear-interpolated percentile of ``values``."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def clears(
    before: Mapping[str, Any], after: Mapping[str, Any], seat: int
) -> dict[str, int]:
    """Return what the engine cleared for one seat on one turn, per good.

    The primitive ``sale_metrics`` aggregates, exposed because market impact has
    to be priced good by good and the aggregate cannot be. The rule is that
    module's rule and not a second one: a good's shed stock falling while the
    bank rises, restricted to goods the market books, so an order the engine
    dropped moves nothing and an animal leaving the shed for a pasture is a
    placement rather than a sale.

    Args:
        before: The seat's own observation before the turn.
        after: Its own observation after it.
        seat: Which seat to score.

    Returns:
        ``{good: units}`` for the goods that cleared, empty if none did.
    """
    if float(after["farms"][seat]["money"]) <= float(before["farms"][seat]["money"]):
        return {}
    shed_before = before["private"]["shed"]
    shed_after = after["private"]["shed"]
    return {
        good: shed_before[good] - shed_after.get(good, 0)
        for good in before["market"]["prices"]
        if shed_before[good] > shed_after.get(good, 0)
    }


def purchases(
    before: Mapping[str, Any], after: Mapping[str, Any], seat: int
) -> dict[str, int]:
    """Return what one seat bought out of the book on one turn, per good.

    The mirror of ``clears`` and ``sales.buy_units``' rule, narrowed to the
    goods the book prices, because those are the only purchases that take stock
    out of the market. A harvest cannot be mistaken for one: harvested produce
    lands in the harvesting unit's own inventory and only reaches the shed when
    ``_end_of_day`` empties them, so mid-day shed growth on a turn the bank fell
    is a purchase.

    Args:
        before: The seat's own observation before the turn.
        after: Its own observation after it.
        seat: Which seat to score.

    Returns:
        ``{good: units}``, empty if the bank did not fall.
    """
    if float(after["farms"][seat]["money"]) >= float(before["farms"][seat]["money"]):
        return {}
    shed_before = before["private"]["shed"]
    shed_after = after["private"]["shed"]
    return {
        good: shed_after.get(good, 0) - shed_before.get(good, 0)
        for good in PRODUCTS
        if shed_after.get(good, 0) > shed_before.get(good, 0)
    }


def drain(step: int, shops: Sequence[str], day: int) -> dict[str, int]:
    """Return how much stock the town takes out of the book on one turn.

    The demand side of the market, transcribed from ``_town_consume``. Its only
    inputs are the turn index, the day and which shops have opened, all three of
    which the observation records, and it never looks at the inventory -- so the
    book is exactly additive in the two farms' trading, and the counterfactual
    in ``impact`` is a subtraction rather than a simulation.

    Args:
        step: The turn index, which is the ``step`` the interpreter sees.
        shops: The shops open at the start of the turn.
        day: The day that turn falls on.

    Returns:
        Units removed per good.
    """
    removed: dict[str, int] = defaultdict(int)
    if step % TOWN_SHOP_SELL_INTERVAL == 0:
        for shop in shops:
            products = SHOPS[shop]
            multiplier = 2 if len(products) == 1 else 1
            for item in products:
                removed[item] += multiplier
    if step % TOWN_CENTER_SELL_INTERVAL == 0:
        centre = next(m for at, m in TOWN_CENTER_DEMAND_SCHEDULE if day >= at)
        for item in TOWN_CENTER_PRODUCTS:
            removed[item] += centre
    return removed


def coverage(
    book: Sequence[Mapping[str, Any]],
    net: Mapping[int, Sequence[Mapping[str, int]]],
) -> float:
    """Return what fraction of real trade the shed inference accounts for.

    The book is exactly additive, so the units the two farms really put into it
    on a turn are ``inventory change + town drain`` -- no inference involved.
    The shed inference has to miss some of that, because it reads a completion
    off the sign of the bank and a turn that sells and spends more than it sold
    for shows a falling bank. This is how much it misses, and it is the number
    that says how far the trade profile and the price-impact figures can be
    pushed.

    Args:
        book: Seat 0's observations, which carry the shared market and the town.
        net: Every seat's per-turn net contribution to the book, per good.

    Compared over the season rather than turn by turn, because what the
    counterfactual actually depends on is the running total a seat has put into
    the book, not the timing of any one clear.

    Args:
        book: Seat 0's observations, which carry the shared market and the town.
        net: Every seat's per-turn net contribution to the book, per good.

    Returns:
        Inferred net units over true net units, summed over goods as absolute
        season totals.
    """
    inferred: dict[str, int] = defaultdict(int)
    truth: dict[str, int] = defaultdict(int)
    for turn, state in enumerate(book[:-1]):
        removed = drain(turn, state["town"]["unlocked_shops"], state["day"])
        levels = state["market"]["inventory"]
        after = book[turn + 1]["market"]["inventory"]
        for good in PRODUCTS:
            truth[good] += after[good] - levels[good] + removed[good]
            inferred[good] += sum(net[seat][turn].get(good, 0) for seat in net)
    total = sum(abs(value) for value in truth.values())
    return sum(abs(value) for value in inferred.values()) / total if total else 0.0


class Profile(BaseModel):
    """One seat of one episode, measured.

    Attributes:
        archive: The daily ``.zip`` this episode came from.
        day: The archive's date.
        episode: Kaggle's episode id.
        seat: Which seat this is.
        team: The player's name.
        rating: The seat's attributed ladder rating.
        band: Its within-day rating band.
        bank: Final banked coins.
        won: Whether this seat out-banked the other one.
        sales: Completed clears, one per good that cleared on a turn.
        units: Units those clears moved.
        mean_sale_price: Coins per unit realised.
        mean_market_price: Volume-weighted book price at the moment of sale.
        realisation: Realised over book, where 1.0 is par.
        revenue: Gross coins the clears banked.
        bought: Units purchases credited to the shed.
        modelled: Revenue recomputed from the book by walking the price curve
            unit by unit. A diagnostic: it should track ``revenue``, and where
            it does not the market-impact counterfactual is not to be trusted.
        denied: Coins the opponent did not receive because this seat's earlier
            sales had already pushed the book down. See ``impact``.
        denied_share: ``denied`` over what the opponent would have made without
            this seat in the book.
        crowding: Cosine similarity between the two seats' units-sold-per-good
            vectors. 1.0 is selling exactly the same mix, which is what a seat
            steering away from its opponent's goods would not do.
        coverage: What fraction of the episode's real book movement the shed
            inference accounted for. Below 1.0 the trade profile understates
            trading, and ``denied`` with it.
        day_bank: Banked coins at the end of each day.
        day_units: Units sold on each day.
        day_revenue: Gross sale proceeds on each day.
        day_animals: Animals on the board at the end of each day.
        day_pasture: Pasture tiles at the end of each day.
        day_plants: Planted tiles at the end of each day.
        day_tiles: Unlocked tiles at the end of each day.
        day_hands: Peak hands hired during each day; the engine dismisses them
            every night, so this is a daily purchase and not held capital.
    """

    archive: str
    day: str
    episode: int
    seat: int
    team: str
    rating: float
    band: str
    bank: float
    won: bool
    sales: float
    units: float
    mean_sale_price: float
    mean_market_price: float
    realisation: float
    revenue: float
    bought: float
    modelled: float
    denied: float
    denied_share: float
    crowding: float
    coverage: float
    day_bank: list[float]
    day_units: list[float]
    day_revenue: list[float]
    day_animals: list[int]
    day_pasture: list[int]
    day_plants: list[int]
    day_tiles: list[int]
    day_hands: list[int]


def profiles(
    episode: Mapping[str, Any], meta: Indexed, bands: Mapping
) -> list[Profile]:
    """Return both seats of one decoded episode, measured.

    Args:
        episode: The decoded replay.
        meta: The episode's index row, already attributed.
        bands: The output of ``band_of``.

    Returns:
        One profile per seat, for the seats that fall in a band.
    """
    steps = episode["steps"]
    seats = range(len(meta.teams))
    views = {seat: [step[seat]["observation"] for step in steps] for seat in seats}
    sold = {
        seat: [
            clears(before, after, seat)
            for before, after in zip(view[:-1], view[1:], strict=True)
        ]
        for seat, view in views.items()
    }
    net = {
        seat: [
            {
                good: cleared.get(good, 0) - got.get(good, 0)
                for good in set(cleared) | set(got)
            }
            for cleared, got in zip(
                sold[seat],
                [
                    purchases(before, after, seat)
                    for before, after in zip(view[:-1], view[1:], strict=True)
                ],
                strict=True,
            )
        ]
        for seat, view in views.items()
    }
    impacts = {seat: impact(views[0], sold, net, seat) for seat in seats}
    accounted = coverage(views[0], net)
    sold_mix = {seat: _mix(sold[seat]) for seat in seats}

    out: list[Profile] = []
    for seat in seats:
        band = bands.get((meta.day, meta.episode, seat))
        if band is None:
            continue
        metrics = sale_metrics(views[seat], seat)
        other = 1 - seat
        out.append(
            Profile(
                archive=meta.archive,
                day=meta.day,
                episode=meta.episode,
                seat=seat,
                team=meta.teams[seat],
                rating=meta.ratings[seat],
                band=band,
                bank=meta.banks[seat],
                won=meta.banks[seat] > meta.banks[other],
                sales=metrics["sales"],
                units=metrics["units"],
                mean_sale_price=metrics["mean_sale_price"],
                mean_market_price=metrics["mean_market_price"],
                realisation=metrics["price_realisation"],
                revenue=impacts[seat]["revenue"],
                bought=buy_units(views[seat], seat),
                modelled=impacts[seat]["modelled"],
                denied=impacts[seat]["denied"],
                denied_share=impacts[seat]["denied_share"],
                crowding=_cosine(sold_mix[seat], sold_mix[other]),
                coverage=accounted,
                **_capital(views[0], seat, sold[seat]),
            )
        )
    return out


def impact(
    book: Sequence[Mapping[str, Any]],
    sold: Mapping[int, Sequence[Mapping[str, int]]],
    net: Mapping[int, Sequence[Mapping[str, int]]],
    seat: int,
) -> dict[str, float]:
    """Price what one seat's sales cost the other seat.

    The two farms sell into one book. ``market_price`` is a pure function of
    that book's inventory, and a sale raises the inventory of the good it sold
    by one unit per unit sold, so the coins a seat took out of its opponent's
    prices are computable rather than inferable: reprice every unit the opponent
    cleared against the inventory the book would have held had this seat never
    sold into it, and difference the two.

    Two things bound what this is worth. It is a partial-equilibrium
    counterfactual -- it removes the sales from the book without asking what
    else the seat would have done with the goods, so it is an attribution of
    price impact and not a prediction of the game without this player. And
    prices within a turn are walked from the inventory standing at the start of
    it, because the engine interleaves the two seats unit by unit and the
    observation does not record the order; ``modelled`` against ``revenue`` is
    the check on how much that costs.

    Args:
        book: Seat 0's observations, which carry the shared market.
        sold: Every seat's per-turn clears, from ``clears``.
        net: Every seat's per-turn net contribution to the book -- clears less
            purchases -- which is what the counterfactual removes.
        seat: The seat whose impact on the other one is being priced.

    Returns:
        ``revenue`` and ``modelled`` for this seat, and ``denied`` /
        ``denied_share`` for what it took from the other one.
    """
    other = 1 - seat
    held: dict[str, int] = dict.fromkeys(PRODUCTS, 0)
    modelled = 0.0
    theirs = 0.0
    counterfactual = 0.0
    for turn, state in enumerate(book[:-1]):
        levels = state["market"]["inventory"]
        for good, count in sold[other][turn].items():
            theirs += _walk(good, levels[good], count)
            counterfactual += _walk(good, levels[good] - held[good], count)
        for good, count in sold[seat][turn].items():
            modelled += _walk(good, levels[good], count)
        for good, count in net[seat][turn].items():
            held[good] += count
    denied = counterfactual - theirs
    return {
        "revenue": _proceeds(book, sold, seat),
        "modelled": modelled,
        "denied": denied,
        "denied_share": denied / counterfactual if counterfactual else 0.0,
    }


def _proceeds(
    book: Sequence[Mapping[str, Any]],
    sold: Mapping[int, Sequence[Mapping[str, int]]],
    seat: int,
) -> float:
    """Return the coins one seat's clears actually banked, from the bank itself."""
    total = 0.0
    for turn, cleared in enumerate(sold[seat]):
        if not cleared:
            continue
        total += float(book[turn + 1]["farms"][seat]["money"]) - float(
            book[turn]["farms"][seat]["money"]
        )
    return total


def _walk(good: str, inventory: int, units: int) -> float:
    """Return what ``units`` of ``good`` fetch, walking the book down unit by unit."""
    return float(
        sum(
            market_price(good, inventory + step, MARKET_PARAMS) for step in range(units)
        )
    )


def _mix(sold: Sequence[Mapping[str, int]]) -> dict[str, float]:
    """Return units sold per good over a whole episode."""
    mix: dict[str, float] = defaultdict(float)
    for cleared in sold:
        for good, count in cleared.items():
            mix[good] += count
    return dict(mix)


def _cosine(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    """Return the cosine similarity of two good-keyed vectors."""
    goods = set(first) | set(second)
    dot = sum(first.get(g, 0.0) * second.get(g, 0.0) for g in goods)
    a = math.sqrt(sum(v * v for v in first.values()))
    b = math.sqrt(sum(v * v for v in second.values()))
    return dot / (a * b) if a and b else 0.0


def _capital(
    book: Sequence[Mapping[str, Any]], seat: int, sold: Sequence[Mapping[str, int]]
) -> dict[str, list]:
    """Return one seat's per-day capital, bank and trading.

    Every farm's board is visible in seat 0's observation, so this reads one
    view for both seats. Hands are sampled as the day's peak rather than its
    close because ``_end_of_day`` dismisses every hand each night: a hand is a
    daily purchase on a fibonacci price ladder, not held capital, and reading it
    at the end of the day would report zero for everyone.

    Args:
        book: Seat 0's observations.
        seat: Which farm to read.
        sold: That seat's per-turn clears.

    Returns:
        The ``day_*`` fields of a ``Profile``.
    """
    series: dict[str, list] = {
        key: []
        for key in (
            "day_bank",
            "day_units",
            "day_revenue",
            "day_animals",
            "day_pasture",
            "day_plants",
            "day_tiles",
            "day_hands",
        )
    }
    for day in range(SEASON_DAYS):
        last = min((day + 1) * TURNS_PER_DAY, len(book)) - 1
        if last < day * TURNS_PER_DAY:
            break
        farm = book[last]["farms"][seat]
        tiles = [tile for row in farm["tiles"] for tile in row]
        series["day_bank"].append(float(farm["money"]))
        series["day_animals"].append(
            sum(1 for t in tiles if isinstance(t, dict) and t.get("animal"))
        )
        series["day_pasture"].append(
            sum(1 for t in tiles if isinstance(t, dict) and t.get("kind") == "PASTURE")
        )
        series["day_plants"].append(
            sum(1 for t in tiles if isinstance(t, dict) and t.get("kind") == "PLANT")
        )
        series["day_tiles"].append(sum(1 for t in tiles if t != "LOCKED"))
        series["day_hands"].append(
            max(
                len(book[turn]["farms"][seat]["hands"])
                for turn in range(day * TURNS_PER_DAY, last + 1)
            )
        )
        window = range(day * TURNS_PER_DAY, min(last + 1, len(sold)))
        series["day_units"].append(float(sum(sum(sold[t].values()) for t in window)))
        series["day_revenue"].append(
            float(
                sum(
                    float(book[t + 1]["farms"][seat]["money"])
                    - float(book[t]["farms"][seat]["money"])
                    for t in window
                    if sold[t]
                )
            )
        )
    return series


def load(archive: Path, episode: int) -> dict[str, Any]:
    """Return one decoded episode, streamed out of its archive.

    Args:
        archive: The daily ``.zip``.
        episode: Kaggle's episode id, which is the member's stem.

    Returns:
        The decoded replay.
    """
    with zipfile.ZipFile(archive) as bundle, bundle.open(f"{episode}.json") as member:
        return json.load(member)


def archives() -> list[Path]:
    """Return every daily archive on disk, oldest first."""
    return sorted(CORPUS.glob("*.zip"))
