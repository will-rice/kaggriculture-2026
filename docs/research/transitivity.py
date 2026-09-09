"""How much of our field does one number per agent actually explain?

Bradley-Terry -- and Elo, which is the same model -- asserts that one latent
strength per agent explains every matchup. Balduzzi et al. show the assertion
fails on cyclic games and that a rating can be inflated by cloning. Both are
claims about the *residual*: fit the transitive part, and whatever is left is
structure the model cannot represent.

The decomposition is theirs. Every pairwise result becomes a logit, giving a
skew-symmetric matrix L. The transitive part is the best fit of the form
`theta_i - theta_j`; the cyclic part is what remains. If the remainder is
small, one number per agent is the right model and there is nothing to change.
If it is large, the rating is averaging over structure that decides games.

Run against the live field, which is every pairing the campaign has measured.
"""

import json
import math
import pathlib

import numpy

FIELD = pathlib.Path("run/campaign/field.json")
# A rate of exactly 0 or 1 has an infinite logit. Clipping at half a game
# treats it as the boundary of what the sample could resolve, which is what
# the sample actually says.
FLOOR = 0.5


def main() -> None:
    """Fit the transitive part, measure what is left, and name the cycles."""
    field = json.loads(FIELD.read_text())
    rates, played = field["rates"], field["played"]
    names = sorted(rates)
    index = {name: number for number, name in enumerate(names)}

    rows, logits, weights, pairs = [], [], [], []
    for one in names:
        for two, rate in rates[one].items():
            if index[one] >= index[two]:
                continue
            games = played.get(one, {}).get(two, field.get("games", 32)) or 32
            bounded = min(max(rate, FLOOR / games), 1 - FLOOR / games)
            row = numpy.zeros(len(names))
            row[index[one]], row[index[two]] = 1.0, -1.0
            rows.append(row)
            logits.append(math.log(bounded / (1 - bounded)))
            weights.append(games)
            pairs.append((one, two))

    design = numpy.stack(rows)
    target = numpy.array(logits)
    # Ratings are identified only up to a constant, so pin the mean at zero
    # rather than leaving the system singular.
    design = numpy.concatenate([design, numpy.ones((1, len(names)))])
    target = numpy.concatenate([target, numpy.zeros(1)])

    theta = numpy.linalg.lstsq(design, target, rcond=None)[0]
    predicted = design[:-1] @ theta
    observed = target[:-1]
    residual = observed - predicted

    total = float((observed**2).sum())
    left = float((residual**2).sum())
    rms = float(numpy.sqrt((residual**2).mean()))
    print(f"{len(names)} agents, {len(pairs)} measured pairings")
    print(f"transitive part explains {100 * (1 - left / total):.1f}% of the logits")
    print(f"cyclic residual          {100 * left / total:.1f}%")
    print(f"residual rms             {rms:.3f} logits")
    print(f"  which is               {173.7 * rms:.0f} Elo")

    print("\nthe ten pairings the rating gets most wrong:")
    order = numpy.argsort(-numpy.abs(residual))
    for slot in order[:10]:
        one, two = pairs[slot]
        seen = 1 / (1 + math.exp(-observed[slot]))
        fit = 1 / (1 + math.exp(-predicted[slot]))
        print(
            f"  {one[:26]:<28}vs {two[:26]:<28}"
            f"measured {seen:.3f}  rating says {fit:.3f}  ({seen - fit:+.3f})"
        )

    # Clones: agents whose results against the opponents they share are the
    # same results. This is the population Balduzzi's property P1 excludes.
    print("\npairs of agents that play the same game (shared opponents, |diff|<0.02):")
    twins = 0
    for a_pos, one in enumerate(names):
        for two in names[a_pos + 1 :]:
            shared = set(rates[one]) & set(rates[two]) - {one, two}
            if len(shared) < 5:
                continue
            gap = sum(abs(rates[one][s] - rates[two][s]) for s in shared) / len(shared)
            if gap < 0.02:
                twins += 1
                if twins <= 12:
                    print(
                        f"  {one[:26]:<28}{two[:26]:<28}"
                        f"{len(shared):>3} shared, mean |diff| {gap:.4f}"
                    )
    print(f"  ... {twins} such pairs in total")


if __name__ == "__main__":
    main()
