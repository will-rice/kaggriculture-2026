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

import scipy.stats

from kaggriculture.campaign import config, games, harness, pools

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
    """Say whether the edit helped the matchup, and whether it beats its parent.

    Both, not either. They preview the two things the gate asks -- the matchup
    is the job a round is given, the head-to-head against the program it started
    from is the bar the gate promotes on -- and a round offered the choice took
    the one that could not see its job for twenty-nine promotions.
    """
    args = parser().parse_args()

    # Absolute, because a worker is given a directory of its own and a relative
    # path stops meaning anything the moment it moves there.
    child, parent = args.child.resolve(), args.parent.resolve()
    if args.replay is not None:
        replay(child, parent, args.replay)
        return
    # Both questions, always, because a round that has to choose between
    # them chooses the one that cannot see what it was sent to do. The matchup
    # first: it is the job, and the head-to-head is the bar the job has to
    # clear on the way.
    if args.against is not None:
        against(child, parent, args.against, args.seeds)
        print()
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
    argue.add_argument(
        "--against",
        default=None,
        metavar="EPISODE",
        help=(
            "also measure both programs against the matchup of this episode, "
            "which is how to tell whether an edit helped the matchup this round "
            "was given. Takes the episode key from the message, e.g. "
            "`--against p1a2b3c4m1s1`. Without it only the head-to-head against "
            "the program you started from is reported, which cannot see the "
            "matchup at all"
        ),
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
    # One number per season, because the two seats of a season correlate at
    # 1.000: swapping them cancels seat advantage, it does not draw a second
    # independent sample. Counting games made the error 1.42 times too small.
    seasons = [
        statistics.fmean(
            game.ours - game.theirs for game in played if game.seed == seed
        )
        for seed in SEEDS[:seeds]
    ]
    gaps = [game.ours - game.theirs for game in played]
    wins = sum(gap > 0 for gap in gaps)
    draws = sum(gap == 0 for gap in gaps)
    mean = statistics.mean(seasons)
    error = statistics.stdev(seasons) / len(seasons) ** 0.5 if len(seasons) > 1 else 0.0
    # And the multiplier that makes the interval mean what it says. Two is
    # right when the spread is known; it is estimated here from the same few
    # seasons, so a small sample needs a wider one. Without this, 26% of
    # four-season draws printed an interval that excluded the truth.
    bar = float(scipy.stats.t.ppf(0.975, len(seasons) - 1)) if len(seasons) > 1 else 0.0

    print(f"child.py against parent.py, {len(gaps)} games over both seats")
    print(f"  won {wins}   drew {draws}   lost {len(gaps) - wins - draws}")
    print(
        f"  mean relative bank {mean:+,.0f} +/- {bar * error:,.0f} "
        f"over {len(seasons)} seasons\n"
    )
    print("  season   seat 0     seat 1     paired")
    for seed in SEEDS[:seeds]:
        sides = [game.ours - game.theirs for game in played if game.seed == seed]
        pair = statistics.mean(sides)
        print(f"  {seed:<8} {sides[0]:>+10,.0f} {sides[1]:>+10,.0f} {pair:>+10,.0f}")

    _verdict(mean, error, bar, seasons, seeds)
    worst = min(SEEDS[:seeds], key=lambda s: _paired(played, s))
    print(f"\nthe season this edit does worst on is {worst}: `--replay {worst}`")


def _verdict(mean: float, error: float, bar: float, seasons: list, seeds: int) -> None:
    """Say in coins whether a measured difference is real.

    Shared by both measurements. They ask different questions -- does this beat
    its parent, does this help the matchup it was given -- but the reading is
    the same arithmetic against the same noise, and a round that learned to
    read one should not have to learn a second wording for the other.

    Args:
        mean: The paired difference, in coins a game.
        error: Its standard error over seasons.
        bar: The t multiplier for the season count.
        seasons: The per-season differences, for reporting the count.
        seeds: Seasons played, so a wider block can be suggested.
    """
    # What the numbers mean, said in coins. A round read "inside the error" five
    # times and shipped anyway; the gate then spent a full cycle establishing
    # what these runs had already told it.
    if mean > bar * error and error:
        print(
            f"\nKEEP IT. The edit is worth {mean:+,.0f} a game and the noise in "
            f"this measurement is +/-{bar * error:,.0f}, so the gain is real: it is "
            f"{mean / (bar * error):.1f} times that."
        )
    elif mean < -bar * error and error:
        print(
            f"\nREVERT IT. The edit costs {mean:+,.0f} a game against noise of "
            f"+/-{bar * error:,.0f}, so the loss is real: it is "
            f"{abs(mean) / (bar * error):.1f} times that."
        )
    else:
        # How many seasons would make this difference readable. The error falls
        # with the square root of the games, so the answer is rarely "a few
        # more".
        need = (
            min(
                len(SEEDS),
                max(seeds * 4, int(seeds * (bar * error / abs(mean)) ** 2) + 1),
            )
            if mean
            else len(SEEDS)
        )
        print(
            f"\nTHIS SAYS NOTHING. The edit measured {mean:+,.0f} a game but the "
            f"noise in {len(seasons)} seasons is +/-{bar * error:,.0f}, which "
            f"is larger, so you cannot tell whether it helped or hurt."
            f"\n\nTwo ways forward, and picking neither means shipping a coin "
            f"flip:"
            f"\n  - play more seasons: `--seeds {need}` would roughly settle a "
            f"difference this size, because the noise falls with the square root "
            f"of the games"
            f"\n  - or make a bigger change. Edits that have been kept in this "
            f"lineage were worth +108 and +346 a game; an edit worth tens will "
            f"not be visible here and will not matter on the ladder either."
        )


def against(
    child: pathlib.Path, parent: pathlib.Path, episode: str, seeds: int
) -> None:
    """Did the edit help against the matchup the round was aimed at?

    `compare` answers a different question -- whether the edit beats its own
    parent -- and that is the question a round was being graded on while being
    told to work on a matchup. An edit that takes games off matchup 1 has no
    reason to beat a sibling, so the instrument said neutral and the round
    dropped it. Twenty-nine champions never closed a three-percent gap that way.

    Both programs play the same opponent on the same seeds in both seats, and
    what is reported is the difference between them per season. Paired that
    way, the map and the seat cancel and what is left is the edit.

    Addressed by episode key rather than by opponent name, and reported as
    "the matchup" throughout, because the name is the one thing that must not
    reach a round. Given a pool it could identify, the previous lineage evolved
    opponent fingerprinting -- recognising specific agents by their sheep and
    cow counts -- which solves "beat this pool" and is worth nothing against an
    agent it has never seen.

    Args:
        child: The edited program.
        parent: What it was edited from.
        episode: The episode key from the message, `<id>m<matchup>s<season>`.
        seeds: How many seasons, from the fixed block.

    Raises:
        SystemExit: No such episode is on the record, which is a mistyped key
            rather than a measurement worth ten minutes.
    """
    opponent = _matchup(episode)
    block = list(SEEDS[:seeds])
    mine = _versus(child, opponent, block)
    theirs = _versus(parent, opponent, block)

    # Keyed by season and seat, so the subtraction is between the same game
    # played by two programs rather than between two draws.
    seasons = []
    for seed in block:
        pairs = [
            mine[(seed, seat)] - theirs[(seed, seat)]
            for seat in (0, 1)
            if (seed, seat) in mine and (seed, seat) in theirs
        ]
        if pairs:
            seasons.append(statistics.fmean(pairs))

    won = sum(one > 0 for one in mine.values())
    was = sum(one > 0 for one in theirs.values())
    ours = statistics.fmean(mine.values())
    base = statistics.fmean(theirs.values())
    print(f"child.py and parent.py against the matchup of {episode}")
    print(f"  child  won {won} of {len(mine)}, mean {ours:+,.0f}")
    print(f"  parent won {was} of {len(theirs)}, mean {base:+,.0f}")

    mean = statistics.fmean(seasons)
    error = statistics.stdev(seasons) / len(seasons) ** 0.5 if len(seasons) > 1 else 0.0
    bar = float(scipy.stats.t.ppf(0.975, len(seasons) - 1)) if len(seasons) > 1 else 0.0
    print(
        f"\n  the edit is worth {mean:+,.0f} a game against this opponent, "
        f"+/-{bar * error:,.0f} over {len(seasons)} seasons\n"
    )
    print("  season      child     parent     paired")
    for seed, paired in zip(block, seasons, strict=False):
        child_season = statistics.fmean(mine[(seed, seat)] for seat in (0, 1))
        parent_season = statistics.fmean(theirs[(seed, seat)] for seat in (0, 1))
        print(
            f"  {seed:<8} {child_season:>+10,.0f} "
            f"{parent_season:>+10,.0f} {paired:>+10,.0f}"
        )

    _verdict(mean, error, bar, seasons, seeds)


def _versus(
    agent: pathlib.Path, opponent: str, seeds: list
) -> dict[tuple[int, int], float]:
    """One program's bank margin against one opponent, by season and seat."""
    games = harness.play(agent, [opponent], seeds, _spare_cores(), pool=config.POOL)
    return {(game.seed, game.seat): game.ours - game.theirs for game in games}


def _matchup(episode: str) -> str:
    """Which opponent an episode was played against.

    Read from the games database rather than passed in, so the name never has to
    appear in a message or in this script's output. Every game the campaign
    plays is recorded with both sides by roster name, and the episode key is
    `<id>m<matchup>s<season>`, so the id is the part before the first `m` and
    the opponent is whichever side is not it.

    Args:
        episode: The episode key from the message.

    Returns:
        The opponent's pool name, for `harness.play` to resolve.

    Raises:
        SystemExit: No such episode, which is a mistyped key.
    """
    quoted = episode.replace("'", "")
    rows = games.query(
        "select team_0, team_1 from games.episodes "
        f"where episode = '{quoted}' limit 1 format TabSeparated"
    ).strip()
    if not rows:
        raise SystemExit(
            f"no episode {episode!r} on the record. Use the key from `The game` "
            "section of the message, which looks like `p1a2b3c4m1s1`."
        )
    first, second = rows.split("\t")[:2]
    mine = episode.split("m")[0]
    return second if first == mine else first


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
