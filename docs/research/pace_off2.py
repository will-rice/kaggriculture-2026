"""Does the opening mined from the public games beat the champion that ignores it?

One variable. The pacer is champion_69 topped up to the orders the ladder's
strongest opening sends -- seven hires on day zero, six pens, land on day
three -- and nothing else about it differs. So the result attributes.

The first attempt at this chased the averaged build order and never fired,
because a 1.2-quadrant target is not something a policy can buy. These orders
are whole and were really sent, so a null result this time is about the opening
rather than about the wrapper. The check at the end says which it was.
"""

import math
import pathlib

from kaggriculture.campaign import harness

PACER = pathlib.Path(
    "/tmp/claude-1000/-home-will-projects-kaggriculture-2026/"
    "521a4f31-d72e-47e3-adbb-c5944812e9ba/scratchpad/pacer2.py"
)
CHAMPION = pathlib.Path("run/campaign/champions/champion_69.py")
SEEDS = list(range(9_700_001, 9_700_065))


def main() -> None:
    """Play the pacer against the champion it was built from."""
    registry = PACER.with_name("pace2_pool.json")
    registry.write_text(
        '{"opponents": {"champion_69": "%s"}, "history": []}' % CHAMPION.resolve(),
        encoding="utf-8",
    )
    games = harness.play(
        PACER, ["champion_69"], SEEDS, workers=8, pool=registry, days=True
    )
    won = sum(
        (game.ours > game.theirs) + 0.5 * (game.ours == game.theirs) for game in games
    )
    rate = won / len(games)
    error = math.sqrt(max(rate * (1 - rate), 1e-9) / len(games))
    margins = sorted(game.ours - game.theirs for game in games)
    decisive = sum(1 for game in games if game.ours != game.theirs)
    print(f"pacer against champion_69 over {len(games)} games\n")
    print(f"  win rate      {rate:.4f}  +/- {error:.4f}")
    print(f"  95% interval  [{rate - 1.96 * error:.4f}, {rate + 1.96 * error:.4f}]")
    print(f"  decisive      {decisive} of {len(games)}")
    print(f"  bank margin   median {margins[len(margins) // 2]:+,.0f}")
    print(f"  worst {margins[0]:+,.0f}   best {margins[-1]:+,.0f}")

    # Did the wrapper actually change the play? A pacer that never fires is a
    # null result about the wrapper, not about the opening.
    print("\n  what the pacer actually held, against what it was aiming at:")
    for day, want in ((0, "7 hires, 6 pens"), (3, "2 quadrants"), (4, "-")):
        rows = [row for game in games for row in game.days if row.day == day]
        if not rows:
            continue
        print(
            f"    day {day}: hands {sum(r.ours['hands'] for r in rows) / len(rows):.1f}"
            f"  pens {sum(r.ours['pens'] for r in rows) / len(rows):.1f}"
            f"  quadrants {sum(r.ours['quadrants'] for r in rows) / len(rows):.2f}"
            f"  bank {sum(r.ours['bank'] for r in rows) / len(rows):>8,.0f}"
            f"   (target: {want})"
        )
        print(
            f"      champion  hands {sum(r.theirs['hands'] for r in rows) / len(rows):.1f}"
            f"  pens {sum(r.theirs['pens'] for r in rows) / len(rows):.1f}"
            f"  quadrants {sum(r.theirs['quadrants'] for r in rows) / len(rows):.2f}"
            f"  bank {sum(r.theirs['bank'] for r in rows) / len(rows):>8,.0f}"
        )


if __name__ == "__main__":
    main()
