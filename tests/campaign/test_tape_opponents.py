"""A tape has to reproduce the game it was taken from, to the coin.

This is the only check that can catch the thing that went wrong. The actions in
a tape were byte-identical to the archive's on all 720 steps of both seats, the
agent was delivered every step from 0 to 718 in order, the seed and the
configuration matched the recording field for field, and the engine replays the
corpus exactly -- and the replay still banked 52,143 where the record said
71,793, because an archived action is filed under the state it *produced* rather
than the state it was taken at. Nothing short of playing a tape and comparing
the result was going to say so.
"""

import os
import subprocess
import sys

import pytest

from kaggriculture.campaign import games, harness
from kaggriculture.scripts.tape_opponents import _slug, tape, write_agent


def test_a_family_is_named_the_same_way_in_every_process() -> None:
    """A pool entry's name has to survive a restart, or it enrolls twice.

    The corpus records whatever display name a competitor chose, and one that
    holds nothing in the roster's alphabet -- "沒有道歉 沒有道歉" -- falls back to a
    digest of itself. That was `hash`, which is salted per process: the same
    agent came back as `family_3f90f6864ce77c74` and `family_4bb6499b2120bae3`
    on two consecutive runs, so every rebuild would have added another pool
    entry for a family that already had one, played it, rated it, and gone on
    adding them.

    Two child processes with different `PYTHONHASHSEED`, not two calls here: the
    salt is fixed for the life of an interpreter, so comparing inside one proves
    nothing -- and a suite that happens to run under a pinned seed would let the
    salted version pass.
    """
    name = "沒有道歉 沒有道歉"
    assert not any(c.isalnum() and c.isascii() for c in name), (
        "the example stopped exercising the digest path"
    )
    said = [
        subprocess.run(  # noqa: S603 - this interpreter, a fixed script
            [
                sys.executable,
                "-c",
                "from kaggriculture.scripts.tape_opponents import _slug;"
                f"print(_slug({name!r}))",
            ],
            capture_output=True,
            text=True,
            check=True,
            env=os.environ | {"PYTHONHASHSEED": seed},
        ).stdout.strip()
        for seed in ("1", "2")
    ]

    assert said[0] == said[1], f"the name moved between processes: {said}"


@pytest.mark.local_data
@pytest.mark.parametrize("index", range(3), ids=lambda i: f"episode{i}")
def test_both_seats_of_a_recorded_game_replay_to_the_recorded_banks(
    index: int, tmp_path
) -> None:
    """Tape both sides of one recorded episode, play them, and compare.

    Exactly, not approximately. These agents read nothing -- one agent's two
    episodes, on different seeds against the same opponent, issued 6,652 of
    6,652 identical commands while the strawberry price in them was 31 and 91 --
    so a faithful tape of one is the agent, and the only question a replay can
    answer is whether the tape is faithful. A margin of error here would hide
    the off-by-one this exists to catch: that replay tracked the record's shape
    and was wrong by a third of the bank.

    Parametrised over an index rather than over episodes, because pytest
    evaluates parametrize arguments at collection time and querying the corpus
    there would make every `pytest` invocation in the repo pay for it.
    """
    chosen = games.query(
        "select episode, seed, played, round(bank_0), round(bank_1) "
        "from games.episodes where source = 'ladder' "
        "order by episode limit 3 format TabSeparated"
    ).splitlines()
    if len(chosen) <= index:
        pytest.skip("the corpus holds no recorded ladder episodes")
    episode, seed, played, bank_0, bank_1 = chosen[index].split("\t")

    seats = [
        write_agent(tape(played, episode, seat), tmp_path / str(seat) / "main.py")
        for seat in (0, 1)
    ]
    game = harness.game(seats[0], seats[1], seed=int(seed), seat=0)

    assert (round(game.ours), round(game.theirs)) == (
        round(float(bank_0)),
        round(float(bank_1)),
    ), f"episode {episode} did not replay to its recorded banks"
