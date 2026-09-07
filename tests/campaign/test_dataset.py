"""What a tape becomes once it is a table.

The extraction is the one place a question about the corpus can go wrong
silently: a column filled from the wrong key still holds numbers, and every
query over it comes back with an answer.
"""

import json
import sqlite3
import zipfile
from pathlib import Path

from kaggriculture.campaign import dataset, tapes

HOURS = dataset.HOURS
STEPS = HOURS * dataset.DAYS


def crop(day: int, watered: bool, dry: int, ripe: int) -> dict:
    """A growing tile, with the husbandry a count of tiles cannot show."""
    return {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": day,
        "yield_units": ripe,
        "watered_today": watered,
        "consecutive_unwatered": dry,
        "fertilized_until_day": -1,
        "max_lifespan_step": 400,
    }


def pen(fed: bool, hungry: int) -> dict:
    """An animal tile."""
    return {
        "kind": "COOP",
        "animal": "CHICKEN",
        "placed_day": 0,
        "yield_units": 1,
        "fed_today": fed,
        "consecutive_unfed": hungry,
        "cared_today": True,
        "pending_care_bonus": 0,
        "fertilizer_available": 0,
    }


def farm(money: float, tiles: list[dict], hands: int = 0) -> dict:
    """One side's public farm."""
    return {
        "money": money,
        "tiles": [tiles],
        "hands": [{} for _ in range(hands)],
        "unlocked_quadrants": ["NW"],
        "hires_today": 0,
        "farmer": [0, 0],
    }


def game(
    episode_id: int, banks: tuple[float, float], teams: list[str]
) -> tapes.Episode:
    """A whole recorded season: one side tends its crops, the other does not.

    Each bank rises by one a step, so the state is different at every hour of
    the season. A fixture that held still would pass whichever hour the
    extraction read a day off, and did: reading day ten at hour zero instead
    of hour twenty-three went unnoticed until the banks started moving.
    """
    tended = [crop(0, watered=True, dry=0, ripe=2), pen(fed=True, hungry=0)]
    neglected = [crop(0, watered=False, dry=5, ripe=0), {"kind": "WEED"}]
    steps = []
    for index in range(STEPS):
        farms = [
            farm(banks[0] + index, tended, hands=2),
            farm(banks[1] + index, neglected),
        ]
        observation = {
            "farms": farms,
            "private": {"seeds": {"WHEAT": 4, "MELON": 0}, "shed": {"WHEAT": 7}},
            "market": {"prices": {"WHEAT": 40}, "inventory": {"WHEAT": 9000}},
            "town": {"unlocked_shops": ["MARKET"]},
        }
        steps.append(
            [
                {
                    "observation": observation,
                    "action": {
                        "farmer": ["PLANT", "WHEAT"] if index == 5 else ["PASS"],
                        "hands": [["WATER"], ["PASS"]] if index == 5 else [],
                        "market": [["SELL", "WHEAT", 48]] if index == 5 else [],
                    },
                    "status": "DONE",
                }
                for seat in (0, 1)
            ]
        )
    return tapes.Episode(
        seed=7,
        engine_version=tapes.ENGINE,
        info={"seed": 7, "EpisodeId": episode_id, "TeamNames": teams},
        steps=steps,
    )


def archive(path: Path, episodes: list[tapes.Episode]) -> Path:
    """Write episodes as a daily archive, shaped as the ladder's are."""
    with zipfile.ZipFile(path, "w") as opened:
        for number, episode in enumerate(episodes):
            opened.writestr(
                f"{number}.json",
                json.dumps(
                    {
                        "info": episode.info,
                        "module_version": episode.engine_version,
                        "steps": episode.steps,
                    }
                ),
            )
    return path


def built(tmp_path: Path, episodes: list[tapes.Episode]) -> sqlite3.Connection:
    """Extract those episodes and hand back an open connection."""
    written = archive(tmp_path / "kaggriculture-episodes-2026-09-01.zip", episodes)
    database = tmp_path / "corpus.sqlite"
    dataset.build([written], database, workers=1)
    return sqlite3.connect(database)


def test_an_episode_records_who_played_it(tmp_path: Path) -> None:
    """The whole reason for the dataset.

    "What the winner did" averages over a ladder where half of every game's
    winners are the weaker agent, which is why eleven claims measured that way
    all sat at fifty percent. Naming both sides turns it into a question about
    particular agents, and the tapes have carried the names all along.
    """
    connection = built(tmp_path, [game(1001, (900.0, 300.0), ["Ada", "Grace"])])

    row = connection.execute(
        "SELECT kaggle_id, team_0, team_1, bank_0, bank_1, winner FROM episodes"
    ).fetchone()

    # The banks are the final state, which is what decides the game.
    assert row == (1001, "Ada", "Grace", 900.0 + STEPS - 1, 300.0 + STEPS - 1, 0)


def test_a_draw_has_no_winner_rather_than_a_default_one(tmp_path: Path) -> None:
    """A draw is filed as neither side.

    A query that groups on the winner cannot then silently hand every drawn
    game to seat zero.
    """
    connection = built(tmp_path, [game(1002, (500.0, 500.0), ["Ada", "Grace"])])

    assert connection.execute("SELECT winner FROM episodes").fetchone() == (None,)


def test_a_day_carries_the_husbandry_a_tile_count_cannot_show(tmp_path: Path) -> None:
    """Tending is half the game and a tile count cannot see it.

    Forty growing tiles of which six were watered is not forty of which forty
    were, and both used to read as "planted 40".
    """
    connection = built(tmp_path, [game(1003, (900.0, 300.0), ["Ada", "Grace"])])

    tended, neglected = connection.execute(
        "SELECT seat, planted, watered, dry_worst, ripe, fed, weeds, bank "
        "FROM days WHERE day = 10 ORDER BY seat"
    ).fetchall()

    # A day's row is that day at its last hour: what both players saw before
    # their final decision in it, and not the state it opened on.
    close = 10 * HOURS + dataset.LAST_HOUR
    assert tended == (0, 1, 1, 0, 2, 1, 0, 900.0 + close)
    assert neglected == (1, 1, 0, 5, 0, 0, 1, 300.0 + close)


def test_every_day_of_the_season_is_present(tmp_path: Path) -> None:
    """Including the last, which closes on the final step no action follows.

    `paired.sides` lost that day twice, and both times every claim about the
    close came back with zero support rather than with an error.
    """
    connection = built(tmp_path, [game(1004, (900.0, 300.0), ["Ada", "Grace"])])

    days = connection.execute(
        "SELECT seat, count(*), min(day), max(day) FROM days GROUP BY seat"
    ).fetchall()

    assert days == [(0, dataset.DAYS, 0, 29), (1, dataset.DAYS, 0, 29)]


def test_an_order_keeps_its_item_and_quantity(tmp_path: Path) -> None:
    """An order is a verb, an item and a quantity, and all three are kept.

    Counting the verb alone made "sells more" a claim about how many orders
    were sent rather than how much produce moved, and those are different
    games.
    """
    connection = built(tmp_path, [game(1005, (900.0, 300.0), ["Ada", "Grace"])])

    row = connection.execute(
        "SELECT day, hour, verb, item, quantity FROM orders WHERE seat = 0"
    ).fetchone()

    # Submitted at step 5, which was chosen looking at step 4: day 0, hour 4.
    assert row == (0, 4, "SELL", "WHEAT", 48)


def test_the_farmer_and_the_hands_are_recorded_and_passes_are_not(
    tmp_path: Path,
) -> None:
    """The commands are the policy, and a PASS is the absence of one.

    There are millions of passes; a query counting moves per day gets the same
    answer whether or not they are stored.
    """
    connection = built(tmp_path, [game(1006, (900.0, 300.0), ["Ada", "Grace"])])

    moves = connection.execute(
        "SELECT actor, verb, argument FROM moves WHERE seat = 0 ORDER BY actor"
    ).fetchall()

    assert moves == [(-1, "PLANT", "WHEAT"), (0, "WATER", None)]


def test_only_what_is_held_is_stored(tmp_path: Path) -> None:
    """Non-zero rows only.

    The shed alone names a dozen commodities, and a season of empty ones would
    be most of the table.
    """
    connection = built(tmp_path, [game(1007, (900.0, 300.0), ["Ada", "Grace"])])

    held = connection.execute(
        "SELECT kind, item, count FROM holdings WHERE seat = 0 AND day = 3 "
        "ORDER BY kind, item"
    ).fetchall()

    assert held == [
        ("animal", "CHICKEN", 1),
        ("plant", "WHEAT", 1),
        ("seed", "WHEAT", 4),
        ("shed", "WHEAT", 7),
    ]


def test_the_ladder_is_read_off_the_corpus(tmp_path: Path) -> None:
    """Who actually wins, which is what the pool cannot tell us.

    The agents at the top of the leaderboard publish no kernels, so the pool
    is built from work that is not theirs. Their games are here.
    """
    connection = built(
        tmp_path,
        [
            game(1, (900.0, 300.0), ["Ada", "Grace"]),
            game(2, (900.0, 300.0), ["Ada", "Grace"]),
            game(3, (100.0, 800.0), ["Ada", "Grace"]),
        ],
    )
    connection.close()

    standing = dataset.leaderboard(tmp_path / "corpus.sqlite", least=3)

    assert standing == [("Ada", 3, 2 / 3), ("Grace", 3, 1 / 3)]
