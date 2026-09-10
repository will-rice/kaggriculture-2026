"""What a tape becomes once it is a table.

The extraction is the one place a question about the corpus can go wrong
silently: a column filled from the wrong key still holds numbers, and every
query over it comes back with an answer.
"""

import json
import sqlite3
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


def built(tmp_path: Path, episodes: list[tapes.Episode]) -> sqlite3.Connection:
    """Extract those episodes and hand back an open connection."""
    written = archive(tmp_path / "kaggriculture-episodes-2026-09-01.zip", episodes)
    database = tmp_path / "corpus.sqlite"
    dataset.build([written], database, workers=1)
    return sqlite3.connect(database)


def test_a_failed_rebuild_leaves_the_corpus_it_was_replacing(tmp_path: Path) -> None:
    """A rebuild that raises must not take the working corpus with it.

    On 2026-09-09 one unreadable market order, in one archive of twenty-five,
    ended a fifty-five minute extraction with no corpus at all -- because the
    target was unlinked before the first archive was read. Everything
    downstream reads that file, so a parser surprise in a single episode became
    an outage: no ratings, no build order, no report.
    """
    database = tmp_path / "corpus.sqlite"
    good = archive(
        tmp_path / "kaggriculture-episodes-2026-09-01.zip",
        [game(1, (100.0, 50.0), ["a", "b"])],
    )
    dataset.build([good], database, workers=1)
    before = database.read_bytes()

    # Not a zip at all, so reading it raises rather than yielding no episodes.
    broken = tmp_path / "kaggriculture-episodes-2026-09-02.zip"
    broken.write_text("this is not an archive", encoding="utf-8")

    with pytest.raises(Exception):  # noqa: B017 - the pool re-raises the worker's own
        dataset.build([good, broken], database, workers=1)

    assert database.exists(), "the rebuild destroyed the corpus it was replacing"
    assert database.read_bytes() == before


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

    The tape walk this replaced lost that day twice, and both times every
    claim about the close came back with zero support rather than an error.
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

    connection = built(tmp_path, [episode])

    orders = connection.execute(
        "SELECT verb, item, quantity FROM orders "
        "WHERE seat = 0 AND day = 0 AND hour = 8"
    ).fetchall()

    # The HIRE is kept whole; the empty slot contributes no row at all.
    assert orders == [("HIRE", None, None)]


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
    connection = built(tmp_path, games)
    connection.close()
    database = tmp_path / "corpus.sqlite"

    dataset.rate(database, least=10)

    connection = sqlite3.connect(database)
    rated = [
        team for (team,) in connection.execute("SELECT team FROM teams ORDER BY place")
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
    connection = built(tmp_path, games)
    connection.close()
    database = tmp_path / "corpus.sqlite"

    dataset.rate(database, least=6)

    connection = sqlite3.connect(database)
    rated = {team for (team,) in connection.execute("SELECT team FROM teams")}

    assert "Cameo" not in rated
    # One island or the other, never both: they share no game.
    assert rated in ({"Ada", "Cyd"}, {"Far", "Off"})


def dated(tmp_path: Path, days: dict[str, list[tapes.Episode]]) -> Path:
    """Extract several days of archives at once, keyed by the date each bears."""
    database = tmp_path / "windowed.sqlite"
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
            "2026-08-01": [game(1, (100.0, 50.0), ["retired", "carried_over"])],
            "2026-08-30": [game(2, (100.0, 50.0), ["carried_over", "arrived"])],
        },
    )

    assert dataset.rate(database, least=1, window=1) == 2
    assert {row[0] for row in dataset.ladder(database)} == {"carried_over", "arrived"}

    assert dataset.rate(database, least=1, window=99) == 3
    assert {row[0] for row in dataset.ladder(database)} == {
        "retired",
        "carried_over",
        "arrived",
    }


def test_the_window_counts_back_from_the_newest_archive_not_from_today(
    tmp_path: Path,
) -> None:
    """The first day of the window is read off the data, not off the clock.

    The fetch runs behind the ladder -- four days behind, the morning this was
    written -- so a window measured from today's date would begin after the
    last archive ends and select nothing. Counted back from the newest day
    present, a late fetch narrows the corpus rather than emptying it.
    """
    database = dated(
        tmp_path,
        {
            "2026-08-01": [game(1, (100.0, 50.0), ["a", "b"])],
            "2026-08-02": [game(2, (100.0, 50.0), ["c", "d"])],
            "2026-08-03": [game(3, (100.0, 50.0), ["e", "f"])],
        },
    )
    connection = sqlite3.connect(database)
    try:
        assert dataset.recent(connection, window=1) == "2026-08-03"
        assert dataset.recent(connection, window=2) == "2026-08-02"
        # More window than corpus is the whole corpus, not an error: the
        # window bounds how far back to look, it does not demand days.
        assert dataset.recent(connection, window=99) == "2026-08-01"
    finally:
        connection.close()


def opener(
    episode_id: int,
    teams: list[str],
    takes: tuple[int, int],
    theirs: tuple[int, int],
    ours_wins: bool = True,
) -> tapes.Episode:
    """A season where each seat takes its quadrants on its own schedule.

    Both seats, because a team plays both across a corpus and its signature is
    a median over all of them. Give the opening to seat 0 alone and every
    seat-1 game reads as never reaching a second quadrant, which drags the
    median to fifty-two -- as it did, the first time this was written.

    Everything else is held still. What is being tested is the grouping, and a
    fixture that also varied the farms would not say which of the two it keyed
    on.
    """
    tiles = [crop(0, watered=True, dry=0, ripe=2)]
    steps = []
    for index in range(STEPS):
        day = index // HOURS
        mine = 1 + (day >= takes[0]) + (day >= takes[1])
        yours = 1 + (day >= theirs[0]) + (day >= theirs[1])
        # Who wins is given rather than derived, so a test can build any
        # rating structure it needs -- including one where the single best
        # agent sits in a group whose mean is low.
        rich, poor = (900.0, 300.0) if ours_wins else (300.0, 900.0)
        farms = [
            farm(rich + index, tiles, quadrants=mine),
            farm(poor + index, tiles, quadrants=yours),
        ]
        observation = {
            "farms": farms,
            "private": {"seeds": {"WHEAT": 4}, "shed": {"WHEAT": 7}},
            "market": {"prices": {"WHEAT": 40}, "inventory": {"WHEAT": 9000}},
            "town": {"unlocked_shops": ["MARKET"]},
        }
        steps.append(
            [
                {"observation": observation, "action": {}, "status": "DONE"}
                for _ in (0, 1)
            ]
        )
    return tapes.Episode(
        seed=7,
        engine_version=tapes.ENGINE,
        info={"seed": 7, "EpisodeId": episode_id, "TeamNames": teams},
        steps=steps,
    )


def test_agents_that_open_alike_are_grouped_and_the_strongest_group_leads(
    tmp_path: Path,
) -> None:
    """A mean across the top of a ladder is a mean across different strategies.

    Measured 2026-09-09: the top twelve hold 1.16 quadrants on day three, which
    is 84% of them holding one and 16% holding two. No agent holds 1.16. The
    three that take land on day three are the three best on the ladder and sit
    0.9 log-odds clear of fourth, and averaging deletes exactly that.
    """
    games = []
    # Two rushers and two plodders, each playing enough to be rated. The
    # rushers win their games, so the fit puts them above.
    for number in range(24):
        # Every rusher meets every plodder, so the fit sees one connected
        # field. Paired `number % 2` against `number % 2` they are two disjoint
        # islands and the rating keeps only one of them.
        rusher = f"rush_{number % 2}"
        plodder = f"plod_{(number // 2) % 2}"
        # The plodders' game is written first, so insertion order puts the
        # weaker opening first and the assertion below tests the sort rather
        # than the order the groups happened to be built in.
        games.append(
            opener(
                100 + number,
                [plodder, rusher],
                takes=(6, 11),
                theirs=(3, 8),
                ours_wins=False,
            )
        )
        games.append(opener(number, [rusher, plodder], takes=(3, 8), theirs=(6, 11)))
    database = dated(tmp_path, {"2026-09-01": games})
    dataset.rate(database, least=1, window=99)

    groups = dataset.openings(database, best=4, window=99)

    assert len(groups) == 2, "two openings, so two groups"
    assert groups[0].signature == (3, 8), "the rushers rate above and come first"
    assert sorted(groups[0].teams) == ["rush_0", "rush_1"]
    assert groups[1].signature == (6, 11)
    assert groups[0].rating > groups[1].rating


def test_the_build_order_can_be_read_from_one_group_alone(tmp_path: Path) -> None:
    """Restricted to agents that open alike, the table is followable.

    Blended, the quadrant row asked for 1.2 on day three and 2.3 on day eight --
    fractions of a thing that comes in whole numbers, and a target no agent can
    hit. Within one opening it is 1, 1, 1, 2, and a policy can execute it.
    """
    games = []
    for number in range(24):
        games.append(
            opener(number, [f"rush_{number % 2}", "plod"], takes=(3, 8), theirs=(6, 11))
        )
        games.append(
            opener(
                100 + number,
                ["plod", f"rush_{number % 2}"],
                takes=(6, 11),
                theirs=(3, 8),
            )
        )
    database = dated(tmp_path, {"2026-09-01": games})
    dataset.rate(database, least=1, window=99)

    order = dataset.build_order(database, window=99, teams=["rush_0", "rush_1"])
    quadrants = dict(zip(dataset.MARKS, order["quadrants"], strict=False))

    # Whole numbers throughout: one group, one behaviour.
    assert quadrants[0] == 1.0
    assert quadrants[3] == 2.0, "the rushers hold two on day three"
    assert quadrants[2] == 1.0, "and one the day before"


def test_a_group_is_ranked_by_its_own_strength_not_by_its_best_member(
    tmp_path: Path,
) -> None:
    """The ladder is read strongest-first, so the groups arrive in that order.

    Which makes the ordering easy to get wrong and easy to test wrongly: while
    the best agent also sits in the best group, a sort by group mean and no
    sort at all give the same answer. Here the best agent shares its opening
    with the worst, so the two disagree.

    Ranking by mean is the useful reading. An opening is worth copying if the
    agents using it are strong on the whole -- one outlier inside it says more
    about that agent than about the opening.
    """
    rush, plod = (3, 8), (6, 11)
    games = []
    for number in range(16):
        # `star` beats everyone and `stray` loses to everyone, and they share
        # an opening: that group's mean is middling however good its best is.
        games.append(opener(number, ["star", "stray"], rush, rush))
        games.append(opener(100 + number, ["star", "mid_0"], rush, plod))
        games.append(opener(200 + number, ["star", "mid_1"], rush, plod))
        # The middling pair beat the stray, so their group's mean sits above.
        games.append(opener(300 + number, ["mid_0", "stray"], plod, rush))
        games.append(opener(400 + number, ["mid_1", "stray"], plod, rush))
        games.append(
            opener(
                500 + number, ["mid_0", "mid_1"], plod, plod, ours_wins=number % 2 == 0
            )
        )
    database = dated(tmp_path, {"2026-09-01": games})
    dataset.rate(database, least=1, window=99)

    groups = dataset.openings(database, best=4, window=99)

    strongest = {team for team, _, _, rating, _ in dataset.ladder(database)}
    assert "star" in strongest
    # The best agent is a rusher, so a list left in ladder order leads with the
    # rushers. Ranked by group mean, the middling pair lead instead.
    assert groups[0].signature == plod, "ranked by the group, not by its best member"
    assert set(groups[0].teams) == {"mid_0", "mid_1"}
    assert groups[0].rating > groups[1].rating
    assert "star" in groups[1].teams
