"""Render the ladder report: what the strongest agents do, from the corpus.

Every number in the page is queried at render time, so the report is a view of
the dataset rather than a document that was true once. The prose around them is
fixed, and is a reading of a direction rather than of a value -- "the bank
trails until day twelve" survives the bank changing, and if that direction ever
reverses the numbers beside it will say so plainly.

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
# The two ends of the ladder the page compares. Twenty-five is the same top the
# build order is read from; eighty is a tail wide enough that no one agent's
# habits carry a column.
TOP = 25
TAIL = 80
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
    top = [team for team, *_ in ladder[:TOP]]
    tail = [team for team, *_ in ladder[-TAIL:]]
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
        build = {
            name: (_by_day(connection, name, top), _by_day(connection, name, tail))
            for name, _, _ in SHOWN
        }
        sold = (_sold(connection, top), _sold(connection, tail))
        quads = (
            _by_day(connection, "quadrants", top),
            _by_day(connection, "quadrants", tail),
        )
        finals = (
            _final(connection, top),
            _final(connection, tail),
        )
    finally:
        connection.close()

    store = strategy.Strategies(strategy.STORE)
    settled = store.settled()
    return {
        "games": f"{rows['episodes']:,}",
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
        "top": str(TOP),
        "tail": str(TAIL),
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


def _by_day(
    connection: sqlite3.Connection, column: str, teams: list[str]
) -> list[float]:
    """One column's mean on each shown day, over those teams."""
    marks = ",".join("?" * len(teams))
    return [
        float(
            connection.execute(
                f"SELECT avg({column}) FROM days WHERE day=? AND team IN ({marks})",  # noqa: S608 - column names are this module's own constants
                (day, *teams),
            ).fetchone()[0]
            or 0.0
        )
        for day in DAYS
    ]


def _final(connection: sqlite3.Connection, teams: list[str]) -> float:
    """Mean final bank over those teams' games."""
    marks = ",".join("?" * len(teams))
    return float(
        connection.execute(
            f"SELECT avg(bank) FROM days WHERE day=29 AND team IN ({marks})",  # noqa: S608
            teams,
        ).fetchone()[0]
        or 0.0
    )


def _sold(connection: sqlite3.Connection, teams: list[str]) -> list[float]:
    """Units sold per side, in six-day bands."""
    marks = ",".join("?" * len(teams))
    found = dict(
        connection.execute(
            f"""SELECT o.day/{BAND},
                       sum(o.quantity)*1.0/count(DISTINCT o.episode||o.seat)
                FROM orders o JOIN days d
                  ON d.episode=o.episode AND d.seat=o.seat AND d.day=0
                WHERE o.verb='SELL' AND d.team IN ({marks}) GROUP BY 1""",  # noqa: S608
            teams,
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
