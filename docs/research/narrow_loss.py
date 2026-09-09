"""Find a game champion_65 lost narrowly, and keep the states it lost it in.

The oracle probe needs a game whose answer we already know: one our champion
lost, so "was this a mistake" has a right answer somewhere in it. Narrow rather
than wide, because a narrow loss turns on a decision a reader could plausibly
find, where a rout turns on everything at once.
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
    "521a4f31-d72e-47e3-adbb-c5944812e9ba/scratchpad/loss.json"
)
SEEDS = list(range(9_100_001, 9_100_025))
# The hours a decision is asked about. A season is 719 calls and a probe cannot
# read them all, so these span the opening the corpus says decides games (the
# settled claims are all days 0 to 8) plus two later checkpoints.
ASKED = (0, 24, 72, 120, 192, 336, 480)


def main() -> None:
    """Play until a narrow loss turns up, then dump its states."""
    opponent = str(pathlib.Path(roster.path(THEIRS)).resolve())
    games = harness.play(OURS, [THEIRS], SEEDS, workers=8, days=True)
    lost = [game for game in games if game.ours < game.theirs]
    if not lost:
        print(f"champion_65 lost none of {len(games)} games against {THEIRS}")
        sys.exit(1)
    worst = min(lost, key=lambda game: game.theirs - game.ours)
    print(
        f"{len(lost)} losses of {len(games)}; narrowest is seed {worst.seed} "
        f"seat {worst.seat}, {worst.ours:,.0f} against {worst.theirs:,.0f} "
        f"({worst.theirs - worst.ours:,.0f} behind)"
    )
    OUT.write_text(
        json.dumps(_replay(worst.seed, worst.seat, opponent), indent=1),
        encoding="utf-8",
    )
    print(f"wrote {OUT}")


def _replay(seed: int, seat: int, opponent: str) -> dict:
    """Replay that game, keeping our full observation and move at each mark."""
    sources = [str(OURS.resolve()), opponent]
    if seat == 1:
        sources.reverse()
    agents = [harness.load_agent(pathlib.Path(path)) for path in sources]
    arities = [harness.argument_count(agent) for agent in agents]
    engine = Engine(seed=seed)
    conf = harness.configuration()
    kept = []
    while not engine.done:
        step = engine.step_index
        actions = []
        for player, agent in enumerate(agents):
            view = engine.observation(player)
            action = agent(view, conf) if arities[player] > 1 else agent(view)
            actions.append(action)
            if player == seat and step in ASKED:
                kept.append(
                    {
                        "step": step,
                        "day": step // dataset.HOURS,
                        "hour": step % dataset.HOURS,
                        "observation": view,
                        "private": render_private(engine.state.farms[player]),
                        "played": action,
                    }
                )
        engine.step(actions[0], actions[1])
    banks = [float(farm["money"]) for farm in engine.observation(0)["farms"]]
    return {
        "seed": seed,
        "seat": seat,
        "opponent": THEIRS,
        "final_ours": banks[seat],
        "final_theirs": banks[1 - seat],
        "moments": kept,
    }


if __name__ == "__main__":
    main()
