"""Does following the corpus's build order beat the champion that ignores it?

One variable. The pacer is champion_67 with the top twelve's capacity
purchases bolted on -- the same policy, the same code, plus the land and
animals the corpus says the ladder's best hold on each day. So the result
attributes: if the pacer wins, the build order is prescriptive, and the forty
claims champion_67 misses are forty targets rather than forty coincidences.

If it loses, the build order describes what winners look like and not how to
become one, and the corpus stops being a place to aim.

Also worth knowing either way: the gate has fourteen saturated opponents out of
twenty-four and every one is ours. A pacer that is merely *competitive* is
still the first unsaturated sparring partner the pool has had in dozens of
generations.
"""

import math
import pathlib

from kaggriculture.campaign import harness

PACER = pathlib.Path(
    "/tmp/claude-1000/-home-will-projects-kaggriculture-2026/"
    "521a4f31-d72e-47e3-adbb-c5944812e9ba/scratchpad/pacer.py"
)
CHAMPION = pathlib.Path("run/campaign/champions/champion_67.py")
SEEDS = list(range(9_600_001, 9_600_065))


def main() -> None:
    """Play the pacer against the champion it was built from."""
    registry = PACER.with_name("pace_pool.json")
    registry.write_text(
        '{"opponents": {"champion_67": "%s"}, "history": []}'
        % str(CHAMPION.resolve()),
        encoding="utf-8",
    )
    games = harness.play(
        PACER, ["champion_67"], SEEDS, workers=8, pool=registry, days=True
    )
    won = sum(
        (game.ours > game.theirs) + 0.5 * (game.ours == game.theirs) for game in games
    )
    rate = won / len(games)
    error = math.sqrt(max(rate * (1 - rate), 1e-9) / len(games))
    margins = sorted(game.ours - game.theirs for game in games)
    print(f"pacer against champion_67 over {len(games)} games\n")
    print(f"  win rate      {rate:.4f}  +/- {error:.4f}")
    print(f"  95% interval  [{rate - 1.96 * error:.4f}, {rate + 1.96 * error:.4f}]")
    print(f"  bank margin   median {margins[len(margins) // 2]:+,.0f}")
    print(f"  worst {margins[0]:+,.0f}   best {margins[-1]:+,.0f}")
    decisive = sum(1 for game in games if game.ours != game.theirs)
    print(f"  decisive      {decisive} of {len(games)}")

    # Did it actually buy what it was told to? A wrapper that never fires is a
    # null result about the wrapper, not about the build order.
    quads = [row.ours["quadrants"] for game in games for row in game.days if row.day == 8]
    pens = [row.ours["pens"] for game in games for row in game.days if row.day == 8]
    print(
        f"\n  day 8: pacer holds {sum(quads) / len(quads):.1f} quadrants, "
        f"{sum(pens) / len(pens):.1f} pens"
    )
    print("  champion_67 held 2.0 quadrants; the corpus top twelve hold 2.3")


if __name__ == "__main__":
    main()
