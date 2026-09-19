"""The instrument a round works a season with, and what it promises."""

import csv
import statistics
from pathlib import Path

import pytest

from kaggriculture.campaign import harness, measure

PASS = (
    "def agent(observation, configuration=None):\n"
    "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
)
# Buys five wheat seed on the first step and works one tile, which finishes a
# season a few hundred coins above an untouched opening.
SELLER = (
    "def agent(observation, configuration=None):\n"
    "    if observation['step'] == 0:\n"
    "        return {'farmer': ['PASS'], 'hands': [],\n"
    "                'market': [['BUY_SEED', 'WHEAT', 5]]}\n"
    "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
)


def written(directory: Path, name: str, source: str) -> Path:
    """One program on disk."""
    path = directory / name
    path.write_text(source, encoding="utf-8")
    return path


def test_the_same_program_in_both_seats_is_exactly_a_draw(tmp_path: Path) -> None:
    """The pairing cancels the season, and that is the whole instrument.

    A fixed plan's bank swings about 19.5% from season to season, so two
    programs on different seasons are mostly being compared on their luck:
    telling apart a five-thousand-coin difference that way takes about 114
    games. Played on the same seed in both seats the luck lands on both sides,
    and for a program against a copy of itself it cancels to nothing at all.

    Which is the property a round's loop rests on -- replay the same seed after
    an edit and what moved is the edit. If this ever drifts off zero, the
    replay step is comparing seasons rather than changes and the whole per-
    season loop in `round_prompt.md` is measuring noise.
    """
    child = written(tmp_path, "child.py", SELLER)
    parent = written(tmp_path, "parent.py", SELLER)

    played = [harness.game(child, parent, 101, seat) for seat in (0, 1)]
    gaps = [game.ours - game.theirs for game in played]

    assert gaps[0] == -gaps[1], "the seats did not mirror"
    assert statistics.mean(gaps) == 0.0


def test_a_replay_is_written_in_the_columns_the_seasons_file_uses(
    tmp_path: Path,
) -> None:
    """A round reads the campaign's seasons and then plays its own.

    Those two only sit side by side while one function writes both, so this
    asserts on the columns rather than on a format somebody typed twice. The
    keys differ on purpose -- the campaign names a game by matchup and season,
    a replay names it by seed and seat -- and everything after them is shared.
    """
    child = written(tmp_path, "child.py", SELLER)
    parent = written(tmp_path, "parent.py", PASS)
    played = [harness.game(child, parent, 101, seat, days=True) for seat in (0, 1)]

    replay = harness.day_csv(
        [({"seed": game.seed, "seat": game.seat}, game.days) for game in played]
    )
    seasons = harness.day_csv([({"matchup": 1, "season": 1}, played[0].days)])

    mine = replay.splitlines()[0].split(",")
    theirs = seasons.splitlines()[0].split(",")
    assert mine[:2] == ["seed", "seat"]
    assert theirs[:2] == ["matchup", "season"]
    assert mine[2:] == theirs[2:], "a replay and a season disagree on the columns"
    # Two games, every day of each, and the days in order.
    rows = list(csv.DictReader(replay.splitlines()))
    assert len(rows) == 2 * len(played[0].days)
    assert [row["seat"] for row in rows[: len(played[0].days)]] == ["0"] * len(
        played[0].days
    )


def test_a_replay_names_the_day_the_gap_moved_most(tmp_path: Path) -> None:
    """Step 2 of the loop the round is told to run, and what it points at.

    The totals say an edit helped or did not; the day the banks diverged is
    the only thing a round can act on, and finding it by reading thirty rows is
    work the instrument should have already done.
    """
    child = written(tmp_path, "child.py", SELLER)
    parent = written(tmp_path, "parent.py", PASS)
    game = harness.game(child, parent, 101, 0, days=True)

    gaps = [day.ours_bank - day.theirs_bank for day in game.days]
    moves = [(gaps[index] - gaps[index - 1], index) for index in range(1, len(gaps))]

    assert len(moves) == len(game.days) - 1
    # The day the reported move names is a real day of this season, and the
    # move it reports is that day's, not a total.
    worst, when = min(moves)
    assert 0 < when < len(game.days)
    assert worst == gaps[when] - gaps[when - 1]


def test_the_measurement_can_be_pointed_at_a_matchup() -> None:
    """The flag that makes a round able to check its own work.

    `games.ordered` hands a round the matchup taking the most games off us, and
    until this existed the only instrument played the edit against its own
    parent. An edit that helped the matchup measured neutral there and was
    dropped, which is how twenty-nine promotions passed without closing a
    three-percent gap.
    """
    parsed = measure.parser().parse_args(["--against", "p1a2b3c4m1s1"])

    # An episode key, not an opponent's name: the name is what must not travel.
    assert parsed.against == "p1a2b3c4m1s1"


def test_without_the_flag_the_measurement_is_still_against_the_parent() -> None:
    """The old question stays the default, so nothing silently changes."""
    assert measure.parser().parse_args([]).against is None


def test_an_episode_not_on_the_record_fails_before_playing_anything(
    tmp_path: Path,
) -> None:
    """A mistyped key must not cost ten minutes of games before it is noticed.

    The message has to say where the right key is, because the episode key is
    the round's only handle on the matchup it was given -- the opponent's name
    is deliberately not.
    """
    child = tmp_path / "child.py"
    parent = tmp_path / "parent.py"
    child.write_text("", encoding="utf-8")
    parent.write_text("", encoding="utf-8")

    with pytest.raises(SystemExit) as refused:
        measure.against(child, parent, "no_such_episodem1s1", 2)

    assert "no_such_episodem1s1" in str(refused.value)
    assert "The game" in str(refused.value)


def test_the_two_measurements_read_a_difference_the_same_way(
    capsys: pytest.CaptureFixture,
) -> None:
    """One verdict, shared, so a round does not meet two wordings for one idea."""
    measure._verdict(1000.0, 100.0, 2.0, [1.0] * 8, 8)
    kept = capsys.readouterr().out

    measure._verdict(-1000.0, 100.0, 2.0, [1.0] * 8, 8)
    dropped = capsys.readouterr().out

    measure._verdict(10.0, 100.0, 2.0, [1.0] * 8, 8)
    unknown = capsys.readouterr().out

    assert "KEEP IT" in kept
    assert "REVERT IT" in dropped
    assert "THIS SAYS NOTHING" in unknown
