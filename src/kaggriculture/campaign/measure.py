"""Play `child.py` against the program this round started from, one season at a time.

Copied into every round's directory beside `child.py` and `parent.py`, so a
round can test an edit instead of shipping it and hoping. Two ways to run it:

    ./measure.py              every season, both seats, and the verdict
    ./measure.py --replay 103 one season, day by day, written to a file

Run it as `./measure.py`, not `python measure.py`: the loop copies this file
with a shebang naming the interpreter that can import what it needs, and the
interpreters on a round's PATH cannot.

The campaign used to hand a round a program, a table of results and nothing to
run, and the loop's own docstring said so: the model "measures nothing, owns
nothing and runs nothing". What that bought was a proposer editing blind. Every
useful change found by hand on 2026-09-10 came from a measurement -- counting
bare tiles, counting plantings, comparing seeds held against plots wanted --
and two of the four ideas tried were measured as worse and thrown away before
they cost anything. A round could do none of that.

The comparison is paired on purpose, and that is the whole value of it. A fixed
plan's bank swings about 19.5% across episodes, so scoring two programs on
different seeds means mostly measuring which seeds they drew: telling apart a
5,000-coin difference takes about 114 games that way. Played on the *same*
seeds, in *both* seats, the episode's noise appears on both sides and cancels
-- 82% of it -- and the same difference resolves in about 4 games. The gate the
campaign promotes on does not do this; it draws fresh seeds for every
candidate. So a round measuring its own edit this way has a sharper instrument
than the thing that will judge it.

`--replay` is the other half. The aggregate says whether an edit helped; it
never says where. A replay writes the season out day by day, in the columns
`harness.day_csv` renders, and names the day the gap moved most against the
child.

It used to say those were "the same columns as `seasons.csv`", a file the round
prompt promised and nothing has ever written: the campaign's own games went to
the games database, and the reference outlived the file. They are in the
database, day by day and both sides, which is where to read them against this.

A game costs about a tenth of a second spread over the cores the machine has
spare -- the engine steps in microseconds and the programs themselves are what
take the time -- so the default comparison is a hundred and twenty-eight games
in roughly twelve seconds. Play more seasons if a result is close, and expect
to need four times as many to halve the error: it falls with the square root of
the count, so going from four seasons to eight is not worth the wait and going
from four to sixty-four is.

What this says is not the verdict. The campaign plays every scored game itself,
against opponents this never sees, and that is what promotes a program. This is
for deciding whether an edit helped before spending a gate on it.
"""

import argparse
import os
import pathlib
import statistics

from kaggriculture.campaign import config, harness, pools

HERE = pathlib.Path(__file__).resolve().parent
# Fixed, so two runs of this compare the same seasons and a difference between
# them is the edit rather than the draw.
#
# Sixty-four of them, because sixteen was a ceiling rather than a default: a
# round that measured +63 with a spread of +/-185 had already played every
# season it was allowed, while being told to play more if the result was close.
# The edits this lineage has kept were worth +108 and +346 a game, and +/-185
# cannot see the first. The gate still draws its own seeds, which a round never
# sees, so widening this improves a round's own decision without touching the
# thing that promotes.
SEEDS = tuple(range(101, 165))


def main() -> None:
    """Compare the two programs, or replay one season of the comparison."""
    args = parser().parse_args()

    # Absolute, because a worker is given a directory of its own and a relative
    # path stops meaning anything the moment it moves there.
    child, parent = args.child.resolve(), args.parent.resolve()
    if args.replay is not None:
        replay(child, parent, args.replay)
        return
    compare(child, parent, args.seeds)


def parser() -> argparse.ArgumentParser:
    """The command line, as its own function so a test can ask it what exists.

    `round_prompt.md` prescribes a loop and every step of it is a command line.
    A prescribed flag the script does not accept is a round spending its call
    on a usage error, and the parser is the only authority on which flags are
    real -- so the message is checked against this rather than against a second
    list of flags somebody kept in step by hand.
    """
    argue = argparse.ArgumentParser(
        description="Is `child.py` better than `parent.py`?"
    )
    argue.add_argument(
        "--seeds",
        type=int,
        default=len(SEEDS),
        help=f"episodes, each played twice with the seats swapped (max {len(SEEDS)})",
    )
    argue.add_argument(
        "--replay",
        type=int,
        default=None,
        metavar="SEED",
        help=f"write one season day by day, from {SEEDS[0]} to {SEEDS[-1]}",
    )
    argue.add_argument("--child", type=pathlib.Path, default=HERE / "child.py")
    argue.add_argument("--parent", type=pathlib.Path, default=HERE / "parent.py")
    return argue


def compare(child: pathlib.Path, parent: pathlib.Path, seeds: int) -> None:
    """Play every seed in both seats and say whether the edit is worth keeping.

    The per-season lines are there because the mean hides the shape. An edit
    that gains twenty thousand on one season and loses six on three others has
    the same mean as one that gains five hundred everywhere, and only the
    second is an improvement to the program rather than to its luck on one map.
    """
    played = _games(child, parent, SEEDS[:seeds], days=False)
    gaps = [game.ours - game.theirs for game in played]
    wins = sum(gap > 0 for gap in gaps)
    draws = sum(gap == 0 for gap in gaps)
    mean = statistics.mean(gaps)
    error = statistics.stdev(gaps) / len(gaps) ** 0.5 if len(gaps) > 1 else 0.0

    print(f"child.py against parent.py, {len(gaps)} games over both seats")
    print(f"  won {wins}   drew {draws}   lost {len(gaps) - wins - draws}")
    print(f"  mean relative bank {mean:+,.0f} +/- {error:,.0f}\n")
    print("  season   seat 0     seat 1     paired")
    for seed in SEEDS[:seeds]:
        sides = [game.ours - game.theirs for game in played if game.seed == seed]
        pair = statistics.mean(sides)
        print(f"  {seed:<8} {sides[0]:>+10,.0f} {sides[1]:>+10,.0f} {pair:>+10,.0f}")

    # What the numbers mean, said in coins. A round read "inside the error" five
    # times and shipped anyway; the gate then spent a full cycle establishing
    # what these runs had already told it.
    if mean > 2 * error and error:
        print(
            f"\nKEEP IT. The edit is worth {mean:+,.0f} a game and the noise in "
            f"this measurement is +/-{error:,.0f}, so the gain is real: it is "
            f"{mean / error:.1f} times the noise."
        )
    elif mean < -2 * error and error:
        print(
            f"\nREVERT IT. The edit costs {mean:+,.0f} a game against noise of "
            f"+/-{error:,.0f}, so the loss is real: it is {abs(mean) / error:.1f} "
            f"times the noise."
        )
    else:
        # How many seasons would make this difference readable. The error falls
        # with the square root of the games, so the answer is rarely "a few
        # more".
        need = (
            min(
                len(SEEDS),
                max(seeds * 4, int(seeds * (2 * error / abs(mean)) ** 2) + 1),
            )
            if mean
            else len(SEEDS)
        )
        print(
            f"\nTHIS SAYS NOTHING. The edit measured {mean:+,.0f} a game but the "
            f"noise in {len(gaps)} games is +/-{error:,.0f}, which is larger, so "
            f"you cannot tell whether it helped or hurt."
            f"\n\nTwo ways forward, and picking neither means shipping a coin "
            f"flip:"
            f"\n  - play more seasons: `--seeds {need}` would roughly settle a "
            f"difference this size, because the noise falls with the square root "
            f"of the games"
            f"\n  - or make a bigger change. Edits that have been kept in this "
            f"lineage were worth +108 and +346 a game; an edit worth tens will "
            f"not be visible here and will not matter on the ladder either."
        )
    worst = min(SEEDS[:seeds], key=lambda s: _paired(played, s))
    print(f"\nthe season this edit does worst on is {worst}: `--replay {worst}`")


def replay(child: pathlib.Path, parent: pathlib.Path, seed: int) -> None:
    """Write one season day by day, and name the day the gap moved most.

    Rendered by `harness.day_csv`, the same function the campaign's own day
    tables go through, so a replay and a game out of the games database line up
    column for column rather than having to be translated between.
    """
    played = _games(child, parent, (seed,), days=True)
    out = HERE / f"replay-{seed}.csv"
    out.write_text(
        harness.day_csv(
            [({"seed": game.seed, "seat": game.seat}, game.days) for game in played]
        ),
        encoding="utf-8",
    )
    print(f"season {seed}, both seats, day by day: {out}")
    for game in played:
        gaps = [day.ours_bank - day.theirs_bank for day in game.days]
        moves = [(gaps[i] - gaps[i - 1], game.days[i].day) for i in range(1, len(gaps))]
        lost, when = min(moves) if moves else (0.0, 0)
        print(
            f"  seat {game.seat}: finished {gaps[-1]:+,.0f}; "
            f"the gap moved most on day {when}, by {lost:+,.0f}"
        )


def _paired(played: list, seed: int) -> float:
    """The seat-averaged gap on one season, which is the noise-cancelled one."""
    return statistics.mean(
        game.ours - game.theirs for game in played if game.seed == seed
    )


def _games(child: pathlib.Path, parent: pathlib.Path, seeds: tuple, days: bool) -> list:
    """Play both seats of every seed, on as many cores as are spare.

    Through `harness.game`, which is the same function the campaign plays its
    own scored games with. It used to be a second engine loop written out here,
    which is the shape where a measurement and the thing it measures share a
    bug and agree with each other about it.
    """
    work = [(seed, seat) for seed in seeds for seat in (0, 1)]
    with pools.workers(min(len(work), _spare_cores())) as pool:
        futures = [
            pool.submit(harness.game, child, parent, seed, seat, days)
            for seed, seat in work
        ]
        return [future.result() for future in futures]


def _spare_cores() -> int:
    """Cores to play on: what the campaign leaves for a round, and at least one.

    Eight sessions share the machine and each is usually waiting on its own
    codex call rather than computing, so there is nearly always room. Bounded
    anyway, because eight rounds measuring at once must not take the machine
    away from the gates that decide anything.
    """
    return max(1, ((os.cpu_count() or 2) - 24) // config.SESSIONS)


if __name__ == "__main__":
    main()
