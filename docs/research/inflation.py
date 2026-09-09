"""Is the residual a cycle, or is the lineage inflating its own scale?

Only 3.6% of measured triples are cyclic, so rock-paper-scissors is not what
the 175 Elo of excess residual is made of. The alternative is Balduzzi's other
failure: a rating inflated by the composition of the population. Our field is
fifty-odd champions descended from one another and a handful of public agents,
so if the lineage's internal margins set the scale, the fit will extrapolate
that scale onto the outsiders and claim more than it can win.

The test splits every pairing by whose it is and signs the error toward the
side the fit favours, so over- and under-confidence cannot cancel.
"""

import math
import pathlib

from kaggriculture.campaign import rating

FIELD = pathlib.Path("run/campaign/field.json")
LIVE = 0.03


def main() -> None:
    """Group the rating's errors by whether the pairing crosses the lineage."""
    field = rating.Field.load(FIELD)
    fitted = rating.standings(field.everything())
    groups: dict[str, list[float]] = {
        "champion vs champion": [],
        "champion vs public": [],
        "public vs public": [],
    }
    for one in sorted(field.rates):
        for two, rate in field.rates[one].items():
            if one >= two or one not in fitted or two not in fitted:
                continue
            if not LIVE < rate < 1 - LIVE:
                continue
            says = 1 / (1 + math.exp(-(fitted[one] - fitted[two])))
            mine, theirs = _ours(one), _ours(two)
            key = (
                "champion vs champion"
                if mine and theirs
                else "public vs public"
                if not mine and not theirs
                else "champion vs public"
            )
            strong, claimed = (rate, says) if says >= 0.5 else (1 - rate, 1 - says)
            groups[key].append(claimed - strong)

    print(f"{'group':<24}{'n':>5}{'mean over-claim':>18}{'rms':>9}")
    for key, gaps in groups.items():
        if not gaps:
            continue
        mean = sum(gaps) / len(gaps)
        rms = math.sqrt(sum(gap * gap for gap in gaps) / len(gaps))
        print(f"{key:<24}{len(gaps):>5}{mean:>+18.4f}{rms:>9.4f}")
    print("\npositive = the fit predicted a bigger win than the games gave")


def _ours(name: str) -> bool:
    """Whether the agent is one of ours rather than a harvested opponent."""
    return name.startswith("champion_") or name == "seed"


if __name__ == "__main__":
    main()
