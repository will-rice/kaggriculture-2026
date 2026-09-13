"""What a tape becomes once it is a table.

The extraction is the one place a question about the corpus can go wrong
silently: a column filled from the wrong key still holds numbers, and every
query over it comes back with an answer.
"""

import json
import zipfile
from pathlib import Path

import pytest

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


def farm(money: float, tiles: list[dict], hands: int = 0, quadrants: int = 1) -> dict:
    """One side's public farm."""
    return {
        "money": money,
        "tiles": [tiles],
        "hands": [{} for _ in range(hands)],
        "unlocked_quadrants": ["NW", "NE", "SW", "SE"][:quadrants],
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


def scratch(name: str) -> str:
    """A database of this test's own, emptied first."""
    from kaggriculture.campaign import games

    database = f"test_{abs(hash(name)):x}"[:40]
    games.query(f"DROP DATABASE IF EXISTS {database}")
    games.create(database)
    return database


def rows(database: str, sql: str) -> list[tuple]:
    """Every row of a query, typed as ClickHouse typed them."""
    from kaggriculture.campaign import games

    # 64-bit integers come back quoted by default, so that JavaScript
    # does not silently round them. A test comparing against an int wants
    # the int.
    answer = games.query(
        f"{sql.format(database=database)} FORMAT JSONCompact "
        "SETTINGS output_format_json_quote_64bit_integers = 0"
    )
    return [tuple(row) for row in json.loads(answer)["data"]]


def one(database: str, sql: str) -> tuple:
    """The first row, or an empty tuple."""
    found = rows(database, sql)
    return found[0] if found else ()


def built(tmp_path: Path, episodes: list[tapes.Episode], name: str = "") -> str:
    """Extract those episodes into a database of this test's own."""
    written = archive(tmp_path / "kaggriculture-episodes-2026-09-01.zip", episodes)
    database = scratch(name or str(tmp_path))
    dataset.build([written], database, workers=1)
    return database


def test_a_failed_rebuild_leaves_the_ladder_it_was_replacing(tmp_path: Path) -> None:
    """A rebuild that raises must not take the working ladder with it.

    On 2026-09-09 one unreadable market order, in one archive of twenty-five,
    ended a fifty-five minute extraction with no corpus at all -- because the
    target was emptied before the first archive was read. Everything
    downstream reads it, so a parser surprise in a single episode became an
    outage: no ratings, no report.

    The store changed and the guarantee did not. The rebuild fills a staging
    database and swaps it in a partition at a time, and `REPLACE PARTITION` is
    atomic, so a raise leaves every row exactly where it was.
    """
    database = scratch("failed rebuild")
    good = archive(
        tmp_path / "kaggriculture-episodes-2026-09-01.zip",
        [game(1, (100.0, 50.0), ["a", "b"])],
    )
    dataset.build([good], database, workers=1)
    before = rows(database, "SELECT episode, team_0 FROM {database}.episodes")
    assert before

    # Not a zip at all, so reading it raises rather than yielding no episodes.
    broken = tmp_path / "kaggriculture-episodes-2026-09-02.zip"
    broken.write_text("not a zip", encoding="utf-8")

    with pytest.raises(Exception):  # noqa: B017, PT011 - whatever the reader raises
        dataset.build([good, broken], database, workers=1)

    assert rows(database, "SELECT episode, team_0 FROM {database}.episodes") == before
    # And the staging database is not left behind to be mistaken for the real one.
    from kaggriculture.campaign import games

    assert (
        games.query(
            f"SELECT count() FROM system.databases WHERE name = '{database}_building'"
        )
        == "0"
    )


def test_an_episode_records_who_played_it(tmp_path: Path) -> None:
    """The whole reason for the dataset.

    "What the winner did" averages over a ladder where half of every game's
    winners are the weaker agent, which is why eleven claims measured that way
    all sat at fifty percent. Naming both sides turns it into a question about
    particular agents, and the tapes have carried the names all along.
    """
    database = built(tmp_path, [game(1001, (900.0, 300.0), ["Ada", "Grace"])])

    row = one(
        database,
        "SELECT kaggle_id, team_0, team_1, bank_0, bank_1, winner FROM "
        "{database}.episodes",
    )

    # The banks are the final state, which is what decides the game.
    assert row == (1001, "Ada", "Grace", 900.0 + STEPS - 1, 300.0 + STEPS - 1, 0)


def test_a_draw_has_no_winner_rather_than_a_default_one(tmp_path: Path) -> None:
    """A draw is filed as neither side.

    A query that groups on the winner cannot then silently hand every drawn
    game to seat zero.
    """
    database = built(tmp_path, [game(1002, (500.0, 500.0), ["Ada", "Grace"])])

    assert one(database, "SELECT winner FROM {database}.episodes") == (-1,)


def test_a_day_carries_the_husbandry_a_tile_count_cannot_show(tmp_path: Path) -> None:
    """Tending is half the game and a tile count cannot see it.

    Forty growing tiles of which six were watered is not forty of which forty
    were, and both used to read as "planted 40".
    """
    database = built(tmp_path, [game(1003, (900.0, 300.0), ["Ada", "Grace"])])

    tended, neglected = rows(
        database,
        "SELECT seat, planted, watered, dry_worst, ripe, fed, weeds, bank "
        "FROM {database}.days WHERE day = 10 ORDER BY seat",
    )

    # A day's row is that day at its last hour: what both players saw before
    # their final decision in it, and not the state it opened on.
    close = 10 * HOURS + dataset.LAST_HOUR
    assert tended == (0, 1, 1, 0, 2, 1, 0, 900.0 + close)
    assert neglected == (1, 1, 0, 5, 0, 0, 1, 300.0 + close)


def test_every_day_of_the_season_is_present(tmp_path: Path) -> None:
    """Including the last, which closes on the final step no action follows.

    The tape walk this replaced lost that day twice, and both times every
    claim about the close came back with zero support rather than an error.
    """
    database = built(tmp_path, [game(1004, (900.0, 300.0), ["Ada", "Grace"])])

    days = rows(
        database,
        "SELECT seat, count(*), min(day), max(day) FROM {database}.days GROUP BY seat",
    )

    assert days == [(0, dataset.DAYS, 0, 29), (1, dataset.DAYS, 0, 29)]


def test_an_order_keeps_its_item_and_quantity(tmp_path: Path) -> None:
    """An order is a verb, an item and a quantity, and all three are kept.

    Counting the verb alone made "sells more" a claim about how many orders
    were sent rather than how much produce moved, and those are different
    games.
    """
    database = built(tmp_path, [game(1005, (900.0, 300.0), ["Ada", "Grace"])])

    row = one(
        database,
        "SELECT day, hour, verb, item, quantity FROM {database}.orders WHERE seat = 0",
    )

    # Submitted at step 5, which was chosen looking at step 4: day 0, hour 4.
    assert row == (0, 4, "SELL", "WHEAT", 48)


def test_an_empty_market_slot_is_no_order_rather_than_a_broken_one(
    tmp_path: Path,
) -> None:
    """A side may submit several market slots and leave some of them unused.

    The action below is copied from the archive: episode 411 of
    2026-09-08, step 122, seat 1. Two slots, one used. The engine's own
    MARKET_OPS calls an empty one NONE, and `_moves` already drops PASS for
    exactly this reason -- counting them would inflate every order tally with
    actions nobody took.

    Unhandled, this one record ended a fifty-five minute extraction and, since
    the target was unlinked first, left no corpus at all.
    """
    episode = game(1006, (900.0, 300.0), ["Ada", "Grace"])
    for seat in (0, 1):
        episode.steps[9][seat]["action"] = {
            "farmer": ["EAST"],
            "hands": [["PICKUP", "WHEAT", 2], ["PASS"], ["NORTH"]],
            "market": [["HIRE"], []],
        }

    database = built(tmp_path, [episode])

    orders = rows(
        database,
        "SELECT verb, item, quantity FROM {database}.orders "
        "WHERE seat = 0 AND day = 0 AND hour = 8",
    )

    # The HIRE is kept whole; the empty slot contributes no row at all.
    assert orders == [("HIRE", "", 0)]


def test_the_farmer_and_the_hands_are_recorded_and_passes_are_not(
    tmp_path: Path,
) -> None:
    """The commands are the policy, and a PASS is the absence of one.

    There are millions of passes; a query counting moves per day gets the same
    answer whether or not they are stored.
    """
    database = built(tmp_path, [game(1006, (900.0, 300.0), ["Ada", "Grace"])])

    moves = rows(
        database,
        "SELECT actor, verb, argument FROM {database}.moves WHERE seat = 0 "
        "ORDER BY actor",
    )

    assert moves == [(-1, "PLANT", "WHEAT"), (0, "WATER", "")]


def test_only_what_is_held_is_stored(tmp_path: Path) -> None:
    """Non-zero rows only.

    The shed alone names a dozen commodities, and a season of empty ones would
    be most of the table.
    """
    database = built(tmp_path, [game(1007, (900.0, 300.0), ["Ada", "Grace"])])

    held = rows(
        database,
        "SELECT kind, item, count FROM {database}.holdings WHERE seat = 0 AND day = 3 "
        "ORDER BY kind, item",
    )

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
    database = built(
        tmp_path,
        [
            game(1, (900.0, 300.0), ["Ada", "Grace"]),
            game(2, (900.0, 300.0), ["Ada", "Grace"]),
            game(3, (100.0, 800.0), ["Ada", "Grace"]),
        ],
    )

    standing = dataset.leaderboard(database, least=3)

    assert standing == [("Ada", 3, 2 / 3), ("Grace", 3, 1 / 3)]


def test_a_rating_is_not_a_win_rate(tmp_path: Path) -> None:
    """Who they played is most of what a win rate says.

    Here Ada beats Cyd every time and Cyd beats Bea every time, so Ada is
    strongest -- but Bea's record is the best in the table, because Bea only
    ever played Dot, who is worse than everyone. A win rate ranks Bea first
    and a Bradley-Terry fit does not, which is the whole reason for the
    column: the agent this campaign was seeded from wins 74% of 668 games and
    rates 46th.
    """
    games: list[tapes.Episode] = []

    def series(one: str, two: str, won: int, lost: int) -> None:
        """``one`` beats ``two`` that many times, and loses the rest."""
        for banks in [(900.0, 300.0)] * won + [(300.0, 900.0)] * lost:
            games.append(game(len(games) + 1, banks, [one, two]))

    # Ada's 60% is against Cyd, who is far above Dot. Bea's 90% is against Dot
    # alone -- the same 90% Cyd manages against Dot, so Bea has shown nothing
    # Cyd has not, and nothing at all against anyone above Dot.
    series("Ada", "Cyd", 6, 4)
    series("Cyd", "Dot", 9, 1)
    series("Bea", "Dot", 9, 1)
    database = built(tmp_path, games)

    dataset.rate(database, least=10)

    rated = [
        team
        for (team,) in rows(
            database, "SELECT team FROM {database}.teams ORDER BY place"
        )
    ]
    by_rate = [team for team, _, _ in dataset.leaderboard(database, least=10)]

    # Bea has the best record in the table and is not the best agent in it.
    assert by_rate[0] == "Bea"
    assert rated[0] == "Ada"
    assert rated.index("Ada") < rated.index("Bea")
    assert rated[-1] == "Dot"


def test_a_rating_needs_both_enough_games_and_a_path_to_the_field(
    tmp_path: Path,
) -> None:
    """Two ways to have no comparison, and both leave a team unrated.

    A team that played twice has a record and not evidence. A pair that only
    ever played each other cannot be placed against anyone else however many
    games they played. Fitting a number for either produces the prior wearing
    a rating, which reads exactly like a measurement.
    """
    games = [game(n, (900.0, 300.0), ["Ada", "Cyd"]) for n in range(1, 7)]
    games += [game(n, (900.0, 300.0), ["Far", "Off"]) for n in range(7, 13)]
    # Played once, won it. Not the best agent on this ladder.
    games.append(game(13, (900.0, 300.0), ["Cameo", "Ada"]))
    database = built(tmp_path, games)

    dataset.rate(database, least=6)

    rated = {team for (team,) in rows(database, "SELECT team FROM {database}.teams")}

    assert "Cameo" not in rated
    # One island or the other, never both: they share no game.
    assert rated in ({"Ada", "Cyd"}, {"Far", "Off"})


def dated(tmp_path: Path, days: dict[str, list[tapes.Episode]]) -> str:
    """Extract several days of archives at once, keyed by the date each bears."""
    database = scratch(f"windowed {tmp_path}")
    dataset.build(
        [
            archive(tmp_path / f"kaggriculture-episodes-{day}.zip", episodes)
            for day, episodes in days.items()
        ],
        database,
        workers=1,
    )
    return database


def test_a_rating_reads_a_window_and_not_the_whole_history(tmp_path: Path) -> None:
    """Teams that stopped playing before the window are not rated at all.

    The ladder's field turns over inside a fortnight, and Bradley-Terry has no
    notion of time: fitted over everything it reads "played against the field
    of three weeks ago" as a strength, and the order it produces stops matching
    the one the ladder is currently producing. So a rating names a window, and
    a team outside it is absent rather than stale.
    """
    # One team spans both eras. Without it the two days would be disconnected
    # islands and the fit would drop one of them for that reason instead of
    # for its date, which is a different rule being tested by accident.
    database = dated(
        tmp_path,
        {
            "2026-08-01": [game(1, (100.0, 50.0), ["a", "b"])],
            "2026-08-02": [game(2, (100.0, 50.0), ["c", "d"])],
            "2026-08-03": [game(3, (100.0, 50.0), ["e", "f"])],
        },
    )

    assert dataset.recent(database, window=1) == "2026-08-03"
    assert dataset.recent(database, window=2) == "2026-08-02"
    # More window than corpus is the whole corpus, not an error: the window
    # bounds how far back to look, it does not demand days.
    assert dataset.recent(database, window=99) == "2026-08-01"
