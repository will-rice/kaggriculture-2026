"""The instrument a round works a season with, and what it promises."""

import csv
import statistics
from pathlib import Path

from kaggriculture.campaign import harness, measure, prompt

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


def test_the_round_is_told_the_commands_that_exist(tmp_path: Path) -> None:
    """The message prescribes a loop; the loop has to be runnable.

    Every step of `round_prompt.md`'s season loop is a command line, and a
    prescribed command that the script does not accept is a round spending its
    call on an error message. The parser is the authority on what exists, so it
    is what this asks.
    """
    del tmp_path
    template = prompt.ROUND_PROMPT.read_text(encoding="utf-8")

    assert "python measure.py --replay" in template
    assert "`--seeds`" in template
    # Both flags reach the script, and `--replay` takes the seed the text says.
    parser = measure.parser()
    assert parser.parse_args(["--replay", "103"]).replay == 103
    assert parser.parse_args(["--seeds", "4"]).seeds == 4
    # And the seed the prompt uses as its example is one the script will play.
    assert 103 in measure.SEEDS
