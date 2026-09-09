"""Does the champion already do what the settled claims say strong play does?

The question behind "why not make it a rule" has a cheaper prior question. A
rule is only worth adding where the agent is not already doing the thing, so
before asking whether a claim can be encoded, ask whether it is already met.

Each settled claim says the stronger side holds more (or less) of a quantity on
a day. So play the champion, take its own value for that quantity on that day,
and set it beside what the corpus's top twelve hold. Where it already matches
or exceeds them, there is no rule to add.
"""

import pathlib
import statistics

from kaggriculture.campaign import dataset, harness, roster, strategy

OURS = pathlib.Path("run/campaign/champions/champion_67.py")
THEIRS = "router2929"
SEEDS = list(range(9_500_001, 9_500_013))
SHOWN = 14


def main() -> None:
    """Compare the champion against every settled claim the build order covers."""
    games = harness.play(OURS, [THEIRS], SEEDS, workers=8, days=True)
    mine: dict[tuple[str, int], list[float]] = {}
    for game in games:
        for row in game.days:
            for quantity, value in row.ours.items():
                mine.setdefault((quantity, row.day), []).append(float(value))

    order = dataset.build_order(best=dataset.BEST)
    top = {
        (quantity, day): value
        for quantity, series in order.items()
        if quantity != "day"
        for day, value in zip(dataset.MARKS, series, strict=False)
    }

    settled = sorted(
        (c for c in strategy.Strategies(strategy.STORE).claims if c.settled),
        key=lambda c: -max(c.agreement, 1 - c.agreement),
    )
    print(f"{len(settled)} settled claims; the corpus's top {dataset.BEST} as the bar\n")
    print(f"{'quantity':<16}{'day':>4}{'claim':>7}{'ours':>9}{'top12':>9}  meets it?")
    checked = met = 0
    for claim in settled:
        key = (claim.form.quantity, claim.form.day)
        if key not in mine or key not in top:
            continue
        checked += 1
        ours = statistics.mean(mine[key])
        want = "more" if claim.ahead() else "less"
        good = ours >= top[key] if want == "more" else ours <= top[key]
        met += good
        if checked <= SHOWN:
            print(
                f"{claim.form.quantity:<16}{claim.form.day:>4}{want:>7}"
                f"{ours:>9.1f}{top[key]:>9.1f}  {'yes' if good else 'NO'}"
            )
    print(f"\nmeets {met} of {checked} settled claims it can be compared on")


if __name__ == "__main__":
    main()
