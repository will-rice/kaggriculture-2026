"""Is our field's residual real non-transitivity, or is it sampling noise?

The first pass fitted least squares on logits and left 35% unexplained. Two
things make that number untrustworthy on its own. It is not the fit we deploy
-- `rating.standings` is a maximum-likelihood MM fit, weighted by games -- and
every rate here is measured over 32 games, so a residual is expected even if
the model is perfect.

So: take the deployed ratings, and compare the residual against what binomial
sampling alone would produce. Under the model, a measured logit differs from
the fitted one with variance about `1 / (n * p * (1 - p))` by the delta method.
If the observed residual variance sits at that floor, one number per agent
explains the field and the model is right. Whatever exceeds it is structure
Bradley-Terry cannot represent, however many games we play.

Saturated pairings -- every game to one side -- are reported apart. Their logit
is infinite and the delta method does not apply, so including them would let
the clipping constant decide the answer.
"""

import json
import math
import pathlib

from kaggriculture.campaign import rating

FIELD = pathlib.Path("run/campaign/field.json")
# Where the delta method stops being usable. Inside this band a rate is far
# enough from the boundary that its logit and variance both mean something.
LIVE = 0.03


def main() -> None:
    """Fit as the campaign fits, then split the residual into noise and cycles."""
    field = rating.Field.load(FIELD)
    fitted = rating.standings(field.everything())

    inside, saturated, wrong = [], 0, []
    for one in sorted(field.rates):
        for two, rate in field.rates[one].items():
            if one >= two or one not in fitted or two not in fitted:
                continue
            games = field.depth(one, two)
            if not LIVE < rate < 1 - LIVE:
                saturated += 1
                continue
            seen = math.log(rate / (1 - rate))
            says = fitted[one] - fitted[two]
            # Variance of the measured logit under the fitted model, by the
            # delta method: the floor a perfect model would still sit at.
            floor = 1.0 / (games * rate * (1 - rate))
            inside.append((seen - says, floor))
            wrong.append((abs(seen - says), one, two, rate, says))

    residual = sum(gap**2 for gap, _ in inside) / len(inside)
    noise = sum(floor for _, floor in inside) / len(inside)
    print(f"{len(fitted)} rated agents; {len(inside)} pairings inside the band, "
          f"{saturated} saturated and set aside")
    print(f"residual variance   {residual:.3f} logits^2")
    print(f"noise floor         {noise:.3f} logits^2  (what 32 games alone gives)")
    excess = max(0.0, residual - noise)
    print(f"excess              {excess:.3f} logits^2 -> {math.sqrt(excess):.3f} logits"
          f" = {173.7 * math.sqrt(excess):.0f} Elo of structure the rating cannot hold")
    print(f"share of residual that is genuine: {100 * excess / residual:.0f}%")

    print("\nthe pairings the deployed rating gets most wrong:")
    for gap, one, two, rate, says in sorted(wrong, reverse=True)[:12]:
        expected = 1 / (1 + math.exp(-says))
        print(
            f"  {one[:24]:<26}vs {two[:24]:<26}"
            f"measured {rate:.3f}  rating says {expected:.3f}  ({rate - expected:+.3f})"
        )

    # A cycle is the thing a single number provably cannot hold. Count the
    # triples that beat each other round.
    names = [n for n in fitted if n in field.rates]
    cycles = total = 0
    for a_pos, a in enumerate(names):
        for b in names[a_pos + 1 :]:
            if b not in field.rates[a]:
                continue
            for c in names[a_pos + 1 :]:
                if c <= b or c not in field.rates[a] or c not in field.rates.get(b, {}):
                    continue
                total += 1
                ab, bc, ca = field.rates[a][b], field.rates[b][c], field.rates[c][a]
                if (ab > 0.5) == (bc > 0.5) == (ca > 0.5):
                    cycles += 1
    print(f"\ncyclic triples: {cycles} of {total} measured triples "
          f"({100 * cycles / max(total, 1):.1f}%)")


if __name__ == "__main__":
    main()
