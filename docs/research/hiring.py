"""Does champion_65 actually under-hire, as four independent verdicts claim?

The oracle's repeated finding is the part worth checking, because a diagnosis
that recurs across separate calls is either a real pattern or a habit of the
model. The corpus already says what the top of the ladder holds by day, so the
claim is directly testable against agents that are not ours.
"""

import pathlib
import statistics

from kaggriculture.campaign import dataset, harness, roster

OURS = pathlib.Path("run/campaign/champions/champion_65.py")
THEIRS = "router2929"
SEEDS = list(range(9_200_001, 9_200_017))
MARKS = (0, 3, 5, 6, 8, 10, 14, 20, 25, 29)


def main() -> None:
    """Play some games, then set our hands beside the ladder's best."""
    games = harness.play(OURS, [THEIRS], SEEDS, workers=8, days=True)
    ours: dict[int, list[float]] = {day: [] for day in MARKS}
    for game in games:
        for row in game.days:
            if row.day in ours:
                ours[row.day].append(float(row.ours["hands"]))
    order = dataset.build_order(best=dataset.BEST)
    top = dict(zip(dataset.MARKS, order["hands"], strict=False))
    print(f"champion_65 over {len(games)} games against {THEIRS}\n")
    print(f"{'day':>4}{'ours':>9}{'top 12':>9}{'gap':>9}")
    for day in MARKS:
        if not ours[day] or day not in top:
            continue
        mine = statistics.mean(ours[day])
        print(f"{day:>4}{mine:>9.1f}{top[day]:>9.1f}{mine - top[day]:>+9.1f}")


if __name__ == "__main__":
    main()
