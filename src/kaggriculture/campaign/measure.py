"""Play `child.py` against the program this round started from.

Copied into every round's directory beside `child.py` and `parent.py`, so a
round can test an edit instead of shipping it and hoping. Run it as
``python measure.py``.

The campaign used to hand a round a program, a table of results and nothing to
run, and the loop's own docstring said so: the model "measures nothing, owns
nothing and runs nothing". What that bought was a proposer editing blind. Every
useful change found by hand on 2026-09-10 came from a measurement -- counting
bare tiles, counting plantings, comparing seeds held against plots wanted --
and two of the four ideas tried were measured as worse and thrown away before
they cost anything. A round could do none of that.

The comparison here is paired on purpose, and that is the whole value of it.
A fixed plan's bank swings about 19.5% across episodes, so scoring two
programs on different seeds means mostly measuring which seeds they drew:
telling apart a 5,000-coin difference takes about 114 games that way. Played
on the *same* seeds, in *both* seats, the episode's noise appears on both
sides and cancels -- 82% of it -- and the same difference resolves in about 4
games. The gate the campaign promotes on does not do this; it draws fresh
seeds for every candidate. So a round measuring its own edit this way has a
sharper instrument than the thing that will judge it.

A game costs about two and a half seconds -- the engine steps in
microseconds, and the programs themselves are what take the time -- so the
default comparison is thirty-two games spread over as many cores as the
machine has spare. Play more seeds if a result is close: the error falls with
the square root of the count.

What this says is not the verdict. The campaign plays every scored game
itself, against opponents this never sees, and that is what promotes a
program. This is for deciding whether an edit helped before spending a gate on
it.
"""

import argparse
import os
import pathlib
import statistics

from kaggle_environments.utils import structify

from kaggriculture.campaign import config, pools
from kaggriculture.campaign.engine.wrapper import Engine
from kaggriculture.campaign.harness import (
    OVERAGE_SECONDS,
    argument_count,
    configuration,
    load_agent,
)

HERE = pathlib.Path(__file__).resolve().parent
# Fixed, so two runs of this compare the same seasons and a difference between
# them is the edit rather than the draw.
SEEDS = tuple(range(101, 117))


def main() -> None:
    """Play both programs over the seeds, both seats, and report the gap."""
    parser = argparse.ArgumentParser(
        description="Is `child.py` better than `parent.py`?"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=len(SEEDS),
        help=f"episodes, each played twice with the seats swapped (max {len(SEEDS)})",
    )
    parser.add_argument("--child", type=pathlib.Path, default=HERE / "child.py")
    parser.add_argument("--parent", type=pathlib.Path, default=HERE / "parent.py")
    args = parser.parse_args()

    work = [
        (str(args.child), str(args.parent), seed, seat)
        for seed in SEEDS[: args.seeds]
        for seat in (0, 1)
    ]
    with pools.workers(min(len(work), _spare_cores())) as pool:
        gaps = list(pool.map(_gap, work))
    wins = sum(gap > 0 for gap in gaps)
    draws = sum(gap == 0 for gap in gaps)
    mean = statistics.mean(gaps)
    error = statistics.stdev(gaps) / len(gaps) ** 0.5 if len(gaps) > 1 else 0.0
    print(f"child.py against parent.py, {len(gaps)} games over both seats")
    print(f"  won {wins}   drew {draws}   lost {len(gaps) - wins - draws}")
    print(f"  mean relative bank {mean:+,.0f} +/- {error:,.0f}")
    if mean > 2 * error and error:
        print("\nthe edit is ahead by more than twice its error: keep it")
    elif mean < -2 * error and error:
        print("\nthe edit is behind by more than twice its error: revert it")
    else:
        print(
            "\ninside the error, so this says nothing yet -- play more seeds,"
            "\nor make a change big enough to see"
        )


def _spare_cores() -> int:
    """Cores to play on: what the campaign leaves for a round, and at least one.

    Eight sessions share the machine and each is usually waiting on its own
    codex call rather than computing, so there is nearly always room. Bounded
    anyway, because eight rounds measuring at once must not take the machine
    away from the gates that decide anything.
    """
    return max(1, ((os.cpu_count() or 2) - 24) // config.SESSIONS)


def _gap(work: tuple) -> float:
    """One episode's bank difference, from `child.py`'s point of view.

    Loads both programs inside the worker: a policy holds state between turns
    and is not always safe to send across processes, and loading is
    milliseconds against a game's seconds.
    """
    child_path, parent_path, seed, seat = work
    child = load_agent(pathlib.Path(child_path))
    parent = load_agent(pathlib.Path(parent_path))
    players = [child, parent] if seat == 0 else [parent, child]
    arities = [argument_count(player) for player in players]
    engine = Engine(seed=seed)
    conf = configuration()
    while not engine.done:
        actions = []
        for index, player in enumerate(players):
            observation = structify(
                {
                    **engine.observation(index),
                    "step": engine.step_index,
                    "remainingOverageTime": OVERAGE_SECONDS,
                }
            )
            actions.append(player(*(observation, conf)[: arities[index]]))
        engine.step(actions[0], actions[1])
    return engine.bank(seat) - engine.bank(1 - seat)


if __name__ == "__main__":
    main()
