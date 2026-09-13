"""The skills a round is given, and whether what they promise works."""

import re
import urllib.error

import pytest

from kaggriculture.campaign import config, dataset, games

from .fixtures import day, game

QUERY_GAMES = config.SKILLS / "query-games" / "SKILL.md"


def running() -> bool:
    """Whether the server is up, so the suite can say why it skipped."""
    try:
        return games.query("SELECT 1") == "1"
    except (urllib.error.URLError, OSError, RuntimeError):
        return False


live = pytest.mark.skipif(not running(), reason="no ClickHouse on GAMES_URL")


@pytest.fixture
def stocked(request: pytest.FixtureRequest) -> str:
    """A database of this test's own, holding both sources of games."""
    name = f"skill_{abs(hash(request.node.name)):x}"[:40]
    games.query(f"DROP DATABASE IF EXISTS {name}")
    request.addfinalizer(lambda: games.query(f"DROP DATABASE IF EXISTS {name}"))
    games.create(name)
    games.record(
        "probe",
        [
            (1, season, "probe", game([day(n, 100.0 - n, 100.0) for n in range(6)]))
            for season in (1, 2)
        ],
        name,
    )
    # And a little of the recorded ladder, which half the skill is about.
    ladder = games.Batch("ladder")
    for episode in range(1, 4):
        ladder.add(
            "episodes",
            [
                f"L{episode}",
                episode,
                episode,
                "1.32.7",
                "2026-09-01",
                "Ada",
                "Grace",
                900.0,
                300.0,
                0,
            ],
        )
        for number in range(6):
            for seat in (0, 1):
                ladder.add(
                    "days",
                    [f"L{episode}", seat, number, "Ada" if seat == 0 else "Grace"]
                    + [1.0] * len(dataset.COLUMNS),
                )
    ladder.send(name)
    games.query(
        f"INSERT INTO {name}.teams (team, games, wins, rating, place) "
        "VALUES ('Ada', 3, 2, 0.5, 1), ('Grace', 3, 1, -0.5, 2)"
    )
    return name


def queries() -> list[str]:
    """Every SQL statement the skill prints, as the skill prints it."""
    text = QUERY_GAMES.read_text(encoding="utf-8")
    blocks = [
        "\n".join(line[4:] for line in block.splitlines()).strip()
        for block in re.findall(r"(?:^ {4}.*\n)+", text, re.M)
    ]
    return [block for block in blocks if block.lower().startswith(("select", "with"))]


def test_the_skill_is_shaped_as_a_skill() -> None:
    """Codex reads the frontmatter to decide whether to open it at all.

    A skill with no description is one the model never loads, and the round's
    context already carries fifty-odd of them -- codex warns that it shortens
    descriptions to fit -- so this one has to say what it is for in the line
    that gets read.
    """
    text = QUERY_GAMES.read_text(encoding="utf-8")
    front = text.split("---")[1]

    assert "name: query-games" in front
    assert "description:" in front
    assert "database" in front, "the description should say what it opens"


@live
def test_every_query_the_skill_prints_actually_runs(stocked: str) -> None:
    """A documented query that does not run costs a round the call it spends.

    The skill is written beside `games`, which owns the schema, and a column
    renamed in one and not the other is invisible until a round pays for it.
    So the queries are executed against a real database rather than read.
    """
    printed = queries()

    assert len(printed) >= 4, "the skill stopped carrying its worked queries"
    for sql in printed:
        games.query(sql.replace("games.", f"{stocked}.") + " FORMAT TabSeparated")


@live
def test_the_skill_finds_this_program_among_everything_else(stocked: str) -> None:
    """Both sources in one store, and a query that separates them.

    The whole point of one database is that a round's own games and the
    recorded ones are the same rows. The whole point of `source` is that it can
    still tell them apart when it wants to.
    """
    counted = games.query(
        f"SELECT source, count() FROM {stocked}.days GROUP BY source "
        "ORDER BY source FORMAT TabSeparated"
    )

    assert counted.splitlines() == ["campaign\t24", "ladder\t36"]
    # And its own games are reachable through `candidate`, which is what the
    # skill tells it to join on.
    mine = games.query(
        f"SELECT count() FROM {stocked}.days d "
        f"JOIN {stocked}.candidate c ON c.episode = d.episode"
    )
    assert int(mine) == 24


def test_the_skill_names_only_columns_the_schema_has() -> None:
    """Prose drifts from a schema more quietly than SQL does.

    The measure names are listed in the skill so a round knows what it can ask
    for without reading the schema first, and that list is a copy of
    `dataset.COLUMNS` the moment it is written. A name here that is not a
    column is a round asking a question with no answer.
    """
    text = QUERY_GAMES.read_text(encoding="utf-8")
    listed = set(re.findall(r"`([a-z_]+)`", text))
    # Everything in backticks that looks like a measure has to be one. The
    # tables, the keys and the two source labels are known and excluded.
    known = {
        *games.TABLES,
        "teams",
        "source",
        "campaign",
        "ladder",
        "opponent",
        "matchup",
        "season",
        "seat",
        "team",
        "day",
        "count",
        "item",
        "kind",
        "episode",
    }
    for name in listed - known - set(dataset.COLUMNS):
        assert "." in name or name.endswith(".py"), (
            f"`{name}` is in the skill and is not a column, a table or a file"
        )
