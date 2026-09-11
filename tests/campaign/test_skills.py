"""The skills a round is given, and whether what they promise works."""

import re
import sqlite3
from pathlib import Path

import pytest

from kaggriculture.campaign import browse, config, dataset, harness

QUERY_GAMES = config.SKILLS / "query-games" / "SKILL.md"


def day(number: int, ours: float, theirs: float) -> harness.Day:
    """One day, with something in every field."""
    return harness.Day(
        day=number,
        ours_bank=ours,
        theirs_bank=theirs,
        ours=dict.fromkeys(dataset.COLUMNS, 0.0) | {"bank": ours, "planted": 4.0},
        theirs=dict.fromkeys(dataset.COLUMNS, 0.0) | {"bank": theirs, "planted": 2.0},
        ours_plants={"WHEAT": 4},
        theirs_plants={"MELON": 2},
        ours_animals={},
        theirs_animals={"COW": 1},
        ours_weeds=0,
        theirs_weeds=3,
        ours_seeds={"WHEAT": 5},
        ours_shed={"WHEAT": 12},
        theirs_shed={"EGG": 3},
        ours_hands=2,
        theirs_hands=1,
        prices={"WHEAT": 25},
    )


def database(path: Path) -> Path:
    """Two matchups of two seasons, enough for every documented query."""
    games = []
    for matchup in (1, 2):
        for season in (1, 2):
            days = [day(n, 100.0 - n * matchup, 100.0) for n in range(5)]
            games.append(
                (
                    matchup,
                    season,
                    "champion_1",
                    harness.Game(
                        opponent="v54",
                        seed=100 + season,
                        seat=season - 1,
                        ours=days[-1].ours_bank,
                        theirs=days[-1].theirs_bank,
                        worst_step_seconds=0.0,
                        days=days,
                    ),
                )
            )
    return browse.write(path, games)


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
    assert "seasons.db" in front, "the description should name what it opens"


def test_every_query_the_skill_prints_actually_runs(tmp_path: Path) -> None:
    """A documented query that does not run costs a round the call it spends.

    The skill is written beside `browse`, which owns the schema, and a column
    renamed in one and not the other is invisible until a round pays for it.
    So the queries are executed against a real database rather than read.
    """
    db = sqlite3.connect(database(tmp_path / "seasons.db"))
    printed = queries()

    assert len(printed) >= 5, "the skill stopped carrying its worked queries"
    for sql in printed:
        db.execute(sql).fetchall()


def test_the_skill_names_only_columns_the_schema_has(tmp_path: Path) -> None:
    """Prose drifts from a schema more quietly than SQL does.

    The measure names are listed in the skill so a round knows what it can ask
    for without reading `.schema` first, and that list is a copy of
    `dataset.COLUMNS` the moment it is written. A name here that is not a
    column is a round asking a question with no answer.
    """
    del tmp_path
    text = QUERY_GAMES.read_text(encoding="utf-8")
    listed = set(re.findall(r"`([a-z_]+)`", text))
    columns = set(dataset.COLUMNS)
    # Everything in backticks that looks like a measure has to be one. Names
    # that are tables, views, files or keys are known and excluded.
    known = {
        "days",
        "holdings",
        "prices",
        "episodes",
        "candidate",
        "orders",
        "moves",
        "teams",
        "gaps",
        "swings",
        "moved",
        "matchup",
        "season",
        "seat",
        "team",
        "day",
        "count",
        "item",
        "kind",
    }
    for name in listed - known - columns:
        assert not name.islower() or "." in name or name.endswith(".py"), (
            f"`{name}` is in the skill and is not a column, table or file"
        )


@pytest.mark.parametrize("view", ["gaps", "swings"])
def test_the_views_the_skill_leans_on_exist(view: str, tmp_path: Path) -> None:
    """The skill teaches the views, so the views have to be there."""
    db = sqlite3.connect(database(tmp_path / f"{view}.db"))

    kind = db.execute(
        "select type from sqlite_master where name = ?", (view,)
    ).fetchone()

    assert kind == ("view",)
