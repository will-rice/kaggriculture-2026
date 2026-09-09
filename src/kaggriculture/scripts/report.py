"""Render the ladder report: what the strongest agents do, from the corpus.

Every number in the page is queried at render time, so the report is a view of
the dataset rather than a document that was true once. The prose around them is
fixed where it reads a direction rather than a value. That was argued to be
safe -- "the bank trails until day twelve" survives the bank changing, and a
reversal would be visible in the numbers beside it -- and on 2026-09-09 the
direction reversed and the sentence did not. The top were ahead on cash from
day zero to day eight, behind only at day ten and day fourteen, and the page
went on claiming they ran poor for the first ten days and crossed over at
twelve. A reader takes the prose at its word and reads the table *through* it.
So the claims that name days are now generated from the days.

Grouped by fitted rating throughout, never by who won a given game: about half
of a ladder's winners are the weaker agent having a good day, and eleven
quantities measured that way over thirteen thousand games all came back between
45% and 60%.

Aggregate only. No opponent's source is read and no agent is named -- the
ratings are computed from team names the archive carries, and none of them
reaches the page.

Run from the repository root, after ``uv run extract-corpus``::

    uv run report

It writes ``docs/reports/ladder.html``, which is published as an artifact.
"""

import argparse
import datetime
import logging
import sqlite3
import string
from pathlib import Path

from kaggriculture.campaign import dataset, strategy

LOGGER = logging.getLogger(__name__)

PAGE = Path(__file__).resolve().parents[3] / "docs" / "reports" / "ladder.html"
# The two ends of the ladder the page compares, as shares of the rated field
# rather than counts of agents. Counts were 25 and 80, sized when a fit over
# the whole corpus rated 185 teams; once the rating took a ten-day window it
# rated 82, and the two ends silently overlapped by 23 agents -- the page went
# on comparing the strong against the weak with a third of each group being
# the same agents. Shares cannot drift that way, and `_ends` refuses outright
# rather than returning an overlap.
TOP_SHARE = 0.15
TAIL_SHARE = 0.40
DAYS = (0, 3, 5, 6, 8, 10, 14, 20, 25, 29)
# Rows of the build-order table, as (column, label, decimal places).
SHOWN = (
    ("bank", "Bank", 0),
    ("quadrants", "Quadrants", 1),
    ("planted", "Planted tiles", 1),
    ("fertilised", "Fertilised", 1),
    ("pens", "Animals", 1),
    ("hands", "Hands hired", 1),
    ("seeds", "Seed in store", 1),
)
BANDS = 5
BAND = 6
# Our own team name in the replay archive. Our ladder games land in the public
# corpus like everyone else's, so the campaign's agents can be rated in the
# same Bradley-Terry fit, over the same window, against the same opponents --
# which is the one measurement of our own progress that does not move when the
# field does. A ladder score cannot do that: the same bytes scored 2386.8 and
# 1555.8 five days apart, and identical agents have landed 455 and 512 apart on
# the same day.
#
# Kept here rather than in `config` because only this page reads it, and
# `config` is imported by every worker of a running loop.
TEAM = "kaggricodex"


def main() -> None:
    """Query the corpus and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=dataset.DATABASE)
    parser.add_argument("--out", type=Path, default=PAGE)
    arguments = parser.parse_args()

    facts = measure(arguments.database)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(TEMPLATE.substitute(facts), encoding="utf-8")
    LOGGER.info("wrote %s over %s games", arguments.out, facts["games"])


def measure(database: Path) -> dict[str, str]:
    """Every number the page states, queried once."""
    rows = dataset.counts(database)
    ladder = dataset.ladder(database)
    top, tail = _ends(ladder)
    connection = sqlite3.connect(database)
    try:
        first, last = connection.execute(
            "SELECT min(played), max(played) FROM episodes"
        ).fetchone()
        won_a, won_b, drawn = connection.execute(
            "SELECT sum(winner=0), sum(winner=1), sum(winner IS NULL) FROM episodes"
        ).fetchone()
        margin, share = connection.execute(
            "SELECT avg(abs(bank_0-bank_1)), "
            "avg(abs(bank_0-bank_1)*1.0/((bank_0+bank_1)/2+1)) FROM episodes"
        ).fetchone()
        bank = connection.execute(
            "SELECT avg((bank_0+bank_1)/2) FROM episodes"
        ).fetchone()[0]
        low, high = connection.execute(
            "SELECT min(rating), max(rating) FROM teams"
        ).fetchone()
        # Every per-team figure below reads the same window the ladder above
        # it was fitted over. Read over the whole corpus instead, a team that
        # carried across the turnover brings its games against a vanished
        # field into a row headed by its current rank.
        first_rated = dataset.recent(connection)
        rated = connection.execute(
            "SELECT count(*) FROM episodes WHERE played >= ?", (first_rated,)
        ).fetchone()[0]
        build = {
            name: (
                _by_day(connection, name, top, first_rated),
                _by_day(connection, name, tail, first_rated),
            )
            for name, _, _ in SHOWN
        }
        sold = (
            _sold(connection, top, first_rated),
            _sold(connection, tail, first_rated),
        )
        quads = (
            _by_day(connection, "quadrants", top, first_rated),
            _by_day(connection, "quadrants", tail, first_rated),
        )
        finals = (
            _final(connection, top, first_rated),
            _final(connection, tail, first_rated),
        )
        trailing, crossover = _crossing(connection, top, tail, first_rated)
        ours = _ours(connection, first_rated, ladder)
    finally:
        connection.close()

    store = strategy.Strategies(strategy.STORE)
    settled = store.settled()
    return {
        "games": f"{rows['episodes']:,}",
        "rated_games": f"{rated:,}",
        "window": f"{dataset.WINDOW}",
        **ours,
        "since": first_rated,
        "bank_story": _bank_story(trailing, crossover),
        "thin_until": "no point" if crossover is None else f"day {crossover}",
        "teams": f"{len(ladder):,}",
        "days_rows": f"{rows['days']:,}",
        "orders": f"{rows['orders'] / 1e6:.1f}M",
        "moves": f"{rows['moves'] / 1e6:.1f}M",
        "first": first,
        "last": last,
        "margin_share": f"{100 * share:.1f}%",
        "margin": f"{margin:,.0f}",
        "bank": f"{bank:,.0f}",
        "won_a": f"{won_a:,}",
        "won_b": f"{won_b:,}",
        "drawn": f"{drawn:,}",
        "top_final": f"{finals[0]:,.0f}",
        "tail_final": f"{finals[1]:,.0f}",
        "premium": f"{100 * (finals[0] / finals[1] - 1):.1f}%",
        "spread": f"{high - low:.1f}",
        "top": str(len(top)),
        "tail": str(len(tail)),
        "day_heads": "".join(f"<th>d{day}</th>" for day in DAYS),
        "build_rows": _build_rows(build),
        "quad_top": _points(quads[0]),
        "quad_tail": _points(quads[1]),
        "sell_bars": _bars(sold),
        "sell_peak": f"{max(sold[1]):,.0f}",
        "top_sold": f"{sum(sold[0]):,.0f}",
        "tail_sold": f"{sum(sold[1]):,.0f}",
        "claim_rows": _claim_rows(settled),
        "settled": str(len(settled)),
        "asked": str(len(store.claims)),
        "generated": datetime.datetime.now(datetime.UTC).strftime("%d %B %Y"),
    }


def _ends(ladder: list[tuple]) -> tuple[list[str], list[str]]:
    """The two ends of the rated field, as disjoint lists of team names.

    Raises:
        ValueError: If the shares would overlap. A page that compares a group
            against itself reads as a page that found no difference, which is
            the one failure that looks like a result.
    """
    names = [team for team, *_ in ladder]
    strong, weak = round(len(names) * TOP_SHARE), round(len(names) * TAIL_SHARE)
    if strong + weak > len(names):
        raise ValueError(
            f"{strong} strongest and {weak} weakest overlap in a field of "
            f"{len(names)}: the ends of the ladder are not disjoint"
        )
    return names[:strong], names[-weak:]


def _ours(
    connection: sqlite3.Connection, first: str, ladder: list[tuple]
) -> dict[str, str]:
    """Our own team's standing in the window, rated or not yet.

    Shown below the threshold as well as above it, because the interesting
    part is watching it fill: `dataset.LEAST` games is two or three days of
    ladder play, and until then the campaign has no era-controlled measurement
    of itself at all.

    Args:
        connection: The dataset.
        first: The first day of the rating window.
        ladder: The rated field, best first, to place ourselves against.

    Returns:
        The facts the page states about us, already formatted.
    """
    played, won = connection.execute(
        "SELECT count(*), sum(CASE WHEN (team_0 = ? AND winner = 0) "
        "OR (team_1 = ? AND winner = 1) THEN 1 ELSE 0 END) "
        "FROM episodes WHERE (team_0 = ? OR team_1 = ?) AND played >= ?",
        (TEAM, TEAM, TEAM, TEAM, first),
    ).fetchone()
    played, won = int(played or 0), int(won or 0)
    rated = {team: value for team, value, *_ in ladder}
    days = connection.execute(
        "SELECT played, count(*), sum(CASE WHEN (team_0 = ? AND winner = 0) "
        "OR (team_1 = ? AND winner = 1) THEN 1 ELSE 0 END) "
        "FROM episodes WHERE (team_0 = ? OR team_1 = ?) AND played >= ? "
        "GROUP BY played ORDER BY played",
        (TEAM, TEAM, TEAM, TEAM, first),
    ).fetchall()
    if TEAM in rated:
        place = 1 + sorted(rated.values(), reverse=True).index(rated[TEAM])
        standing = (
            f"<strong>{rated[TEAM]:+.3f}</strong> log-odds, "
            f"{173.7 * rated[TEAM]:+,.0f} Elo — {place} of {len(rated)}"
        )
    else:
        short = max(0, dataset.LEAST - played)
        standing = (
            f"not yet rated: {played} games of the {dataset.LEAST} a rating "
            f"needs, {short} short"
        )
    return {
        "ours_team": TEAM,
        "ours_standing": standing,
        "ours_games": f"{played:,}",
        "ours_rate": "—" if not played else f"{won / played:.1%}",
        "ours_days": "".join(
            f"<tr><td>{day}</td><td>{count:,}</td><td>{wins:,}</td>"
            f"<td>{wins / count:.1%}</td></tr>"
            for day, count, wins in days
        )
        or '<tr><td colspan="4">no games in the window yet</td></tr>',
    }


def _crossing(
    connection: sqlite3.Connection, top: list[str], tail: list[str], first: str
) -> tuple[list[int], int | None]:
    """The days the top hold less cash than the field, and the last crossover.

    Read over every day of the season rather than the ten the table shows.
    The crossover is a day, and the table's marks are six days apart where it
    falls -- "between day fourteen and day twenty" is the best a reading off
    the marks could manage, and it is not what the sentence claims.

    Returns:
        The days the top trail on mean bank, and the last of them, after which
        they lead for the rest of the season. ``None`` when they never trail.
    """
    series = {}
    for side, names in (("top", top), ("tail", tail)):
        marks = ",".join("?" * len(names))
        series[side] = dict(
            connection.execute(
                f"SELECT d.day, avg(d.bank) FROM days d "  # noqa: S608 - column names are this module's own constants
                f"JOIN episodes e ON e.episode = d.episode "
                f"WHERE d.team IN ({marks}) AND e.played >= ? GROUP BY d.day",
                (*names, first),
            ).fetchall()
        )
    trailing = [
        day
        for day in sorted(series["top"])
        if series["top"][day] < series["tail"].get(day, 0.0)
    ]
    return trailing, trailing[-1] if trailing else None


def _bank_story(trailing: list[int], crossover: int | None) -> str:
    """The lede for finding one, in whatever shape the days actually make."""
    if crossover is None:
        return (
            "The strong agents are never behind on cash. They buy capacity "
            "early and stay ahead while they do it."
        )
    if len(trailing) == 1:
        when = f"On day {trailing[0]} alone"
    elif trailing == list(range(trailing[0], trailing[-1] + 1)):
        when = f"From day {trailing[0]} to day {trailing[-1]}"
    else:
        days = ", ".join(str(day) for day in trailing[:-1])
        when = f"On days {days} and {trailing[-1]}"
    return (
        f"{when} the strong agents hold <em>less</em> cash than the field. "
        "They are spending it — on land, on quadrants, on animals, on "
        f"hands. The bank crosses over after day {crossover} and never comes "
        "back."
    )


def _by_day(
    connection: sqlite3.Connection, column: str, teams: list[str], first: str
) -> list[float]:
    """One column's mean on each shown day, over those teams."""
    marks = ",".join("?" * len(teams))
    return [
        float(
            connection.execute(
                f"SELECT avg(d.{column}) FROM days d "  # noqa: S608 - column names are this module's own constants
                f"JOIN episodes e ON e.episode=d.episode "
                f"WHERE d.day=? AND d.team IN ({marks}) AND e.played>=?",
                (day, *teams, first),
            ).fetchone()[0]
            or 0.0
        )
        for day in DAYS
    ]


def _final(connection: sqlite3.Connection, teams: list[str], first: str) -> float:
    """Mean final bank over those teams' games."""
    marks = ",".join("?" * len(teams))
    return float(
        connection.execute(
            f"SELECT avg(d.bank) FROM days d "  # noqa: S608
            f"JOIN episodes e ON e.episode=d.episode "
            f"WHERE d.day=29 AND d.team IN ({marks}) AND e.played>=?",
            (*teams, first),
        ).fetchone()[0]
        or 0.0
    )


def _sold(connection: sqlite3.Connection, teams: list[str], first: str) -> list[float]:
    """Units sold per side, in six-day bands."""
    marks = ",".join("?" * len(teams))
    found = dict(
        connection.execute(
            f"""SELECT o.day/{BAND},
                       sum(o.quantity)*1.0/count(DISTINCT o.episode||o.seat)
                FROM orders o JOIN days d
                  ON d.episode=o.episode AND d.seat=o.seat AND d.day=0
                JOIN episodes e ON e.episode=o.episode
                WHERE o.verb='SELL' AND d.team IN ({marks})
                  AND e.played>=? GROUP BY 1""",  # noqa: S608
            (*teams, first),
        ).fetchall()
    )
    return [float(found.get(band, 0.0)) for band in range(BANDS)]


def _build_rows(build: dict[str, tuple[list[float], list[float]]]) -> str:
    """The build-order table body: each metric as a top row and a tail row."""
    out = []
    for name, label, places in SHOWN:
        top, tail = build[name]
        lead = [t > r * 1.2 for t, r in zip(top, tail, strict=True)]
        cells = "".join(
            f'<td class="{"lead" if hot else ""}">{v:,.{places}f}</td>'
            for v, hot in zip(top, lead, strict=True)
        )
        out.append(
            f'<tr class="gap"><td rowspan="2">{label}</td>'
            f'<td class="who t">top</td>{cells}</tr>'
        )
        out.append(
            '<tr><td class="who r">rest</td>'
            + "".join(f"<td>{v:,.{places}f}</td>" for v in tail)
            + "</tr>"
        )
    return "\n          ".join(out)


def _points(values: list[float]) -> str:
    """A polyline over the shown days, scaled to one to four quadrants."""
    span = max(DAYS)
    return " ".join(
        f"{34 + (day / span) * 276:.0f},{112 - ((value - 0.9) / 2.4) * 102:.0f}"
        for day, value in zip(DAYS, values, strict=True)
    )


def _bars(sold: tuple[list[float], list[float]]) -> str:
    """Grouped bars for units sold, both series on one scale."""
    ceiling = max(max(sold[0]), max(sold[1])) or 1.0
    out = []
    for series, colour, offset in ((sold[0], "strong", 0), (sold[1], "rest", 20)):
        for band, value in enumerate(series):
            height = max(1.0, (value / ceiling) * 102)
            x = 36 + band * 58 + offset
            out.append(
                f'<rect x="{x}" y="{110 - height:.0f}" width="18" '
                f'height="{height:.0f}" fill="var(--{colour})"></rect>'
            )
    return "\n            ".join(out)


def _claim_rows(settled: list[strategy.Claim]) -> str:
    """The strongest settled claims, widest separation first."""
    out = []
    for claim in settled[:9]:
        share = claim.agreement if claim.status == "leads" else 1 - claim.agreement
        out.append(
            f"<tr><td>{claim.form.quantity}</td><td>{claim.form.day}</td>"
            f"<td>{'more' if claim.status == 'leads' else 'less'}</td>"
            f"<td>{claim.support:,}</td>"
            f'<td class="lead">{100 * share:.0f}%</td></tr>'
        )
    return "\n          ".join(out)


# `$name` rather than `{name}`: the page is mostly CSS, and every brace in it
# would have to be doubled for `str.format` -- a rule nobody remembers while
# editing a stylesheet, and one whose failure is a stack trace at render time.
# `Template.substitute` also raises on a placeholder nobody supplied, which is
# what stops a number quietly rendering as literal text.
TEMPLATE = string.Template(
    (Path(__file__).with_name("report_template.html")).read_text(encoding="utf-8")
)

if __name__ == "__main__":
    main()
