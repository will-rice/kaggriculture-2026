"""Capture a lost game with the trajectory, not just seven isolated states.

The first probe gave the model one observation per moment and it reasoned about
what one observation can show: labour is cheap, do not sell what you could
apply. Every claim it made landed on a quantity the corpus measures over
thousands of games and cannot separate the sides on. The corpus's own levers --
land, quadrants, animals, pens, all inside the first week -- went unmentioned
seven times out of seven.

Those levers are not visible in a single frame. "The stronger side has more
land by day three" is a statement about two farms across days, and the seat's
observation shows one farm at one hour. So this keeps what the harness already
measures: all twenty-nine quantities for both sides on every day of the season,
beside the moments themselves.

Moments are chosen from the trajectory rather than fixed in advance, at the
days the two sides came apart, because where a game turned is a property of
that game.
"""

import json
import pathlib
import sys

from kaggriculture.campaign import dataset, harness, roster
from kaggriculture.campaign.engine.wrapper import Engine, render_private

OURS = pathlib.Path("run/campaign/champions/champion_65.py")
THEIRS = "router2929"
OUT = pathlib.Path(
    "/tmp/claude-1000/-home-will-projects-kaggriculture-2026/"
    "521a4f31-d72e-47e3-adbb-c5944812e9ba/scratchpad/trajectory.json"
)
SEEDS = list(range(9_300_001, 9_300_033))
ASKED = 6


def main() -> None:
    """Find a narrow loss, then keep its trajectory and its turning points."""
    opponent = str(pathlib.Path(roster.path(THEIRS)).resolve())
    games = harness.play(OURS, [THEIRS], SEEDS, workers=8, days=True)
    lost = [game for game in games if game.ours < game.theirs]
    if not lost:
        print("no losses to study")
        sys.exit(1)
    worst = min(lost, key=lambda game: game.theirs - game.ours)
    print(
        f"seed {worst.seed} seat {worst.seat}: {worst.ours:,.0f} against "
        f"{worst.theirs:,.0f}, {worst.theirs - worst.ours:,.0f} behind"
    )
    days = [
        {"day": row.day, "ours": row.ours, "theirs": row.theirs} for row in worst.days
    ]
    turns = _turning_days(days)
    print(f"days the sides came apart most: {turns}")
    OUT.write_text(
        json.dumps(
            {
                "seed": worst.seed,
                "seat": worst.seat,
                "opponent": THEIRS,
                "final_ours": worst.ours,
                "final_theirs": worst.theirs,
                "trajectory": days,
                "moments": _replay(worst.seed, worst.seat, opponent, turns),
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"wrote {OUT}")


def _turning_days(days: list[dict]) -> list[int]:
    """The days our lead over the field moved against us hardest.

    A turning point is a change rather than a level: being behind on day
    twenty is usually the consequence of a day much earlier, and the day the
    gap opened is the one a decision could have altered.
    """
    banks = [(row["day"], row["ours"]["bank"] - row["theirs"]["bank"]) for row in days]
    swings = [
        (banks[index][0], banks[index][1] - banks[index - 1][1])
        for index in range(1, len(banks))
    ]
    worst = sorted(swings, key=lambda pair: pair[1])[:ASKED]
    return sorted(day for day, _ in worst)


def _replay(seed: int, seat: int, opponent: str, days: list[int]) -> list[dict]:
    """Replay, keeping our observation and move at the first hour of each day."""
    sources = [str(OURS.resolve()), opponent]
    if seat == 1:
        sources.reverse()
    agents = [harness.load_agent(pathlib.Path(path)) for path in sources]
    arities = [harness.argument_count(agent) for agent in agents]
    engine = Engine(seed=seed)
    conf = harness.configuration()
    wanted = {day * dataset.HOURS for day in days}
    kept = []
    while not engine.done:
        step = engine.step_index
        actions = []
        for player, agent in enumerate(agents):
            view = engine.observation(player)
            action = agent(view, conf) if arities[player] > 1 else agent(view)
            actions.append(action)
            if player == seat and step in wanted:
                kept.append(
                    {
                        "day": step // dataset.HOURS,
                        "hour": step % dataset.HOURS,
                        "observation": view,
                        "private": render_private(engine.state.farms[player]),
                        "played": action,
                    }
                )
        engine.step(actions[0], actions[1])
    return kept


if __name__ == "__main__":
    main()
