"""A Bradley-Terry fit over a set of pairings: one strength per agent.

The gate does not read this. It did until 2026-09-12, when every candidate
began playing the whole pool on the same seeds in both seats, which makes the
win rate directly comparable and leaves a rating estimating something already
measured. What remains is the diagnostic: `dataset.leaderboard` rates the
public corpus with it, where agents play different opponents on different
seeds and a mean win rate says nothing, and `scripts/champions.py` rates the
champion lineage on fixed files.

One latent strength per agent, fitted from every pairing at once, such that
``P(i beats j) = p_i / (p_i + p_j)``. Beating a strong opponent counts for
more than beating a weak one, and one bad matchup is absorbed rather than
fatal.

Ratings are identified only up to a constant, so a number means nothing on its
own: only differences within one fit do.

The prior matters. Without it an agent that has won nothing has strength zero
and a rating of negative infinity. Two phantom games at even odds keep every
rating finite, and the winless then tie at the bottom rather than being
incomparable.

It is blind to margin, because the competition is: the win condition is
relative bank and the margin never counts.
"""

import math

# Phantom games at even odds against an average opponent, so a winless agent
# still has a finite rating. Two is weak enough to move a well-measured agent
# very little and strong enough to keep the fit off negative infinity.
PRIOR_GAMES = 2.0
# The MM iteration converges geometrically; this is far more than a field of
# this size needs, and it stops as soon as it settles.
ITERATIONS = 10_000
TOLERANCE = 1e-12


def ratings(
    rates: dict[str, dict[str, float]], games: dict[str, dict[str, float]]
) -> dict[str, float]:
    """Fit one strength per agent from every pairing between them.

    The standard MM (Zermelo) iteration, which for Bradley-Terry is monotone
    and needs no gradient and no dependency:

        p_i  <-  W_i / sum_j  n_ij / (p_i + p_j)

    where ``W_i`` is i's wins over the field and ``n_ij`` the games between i
    and j. The prior enters as ``PRIOR_GAMES`` games at even odds against an
    agent of average strength.

    Args:
        rates: ``rates[a][b]`` is a's win rate against b, ties as half.
        games: ``games[a][b]`` is how many games that rate was measured over.

    Returns:
        A rating per agent, centred so the field's mean strength is one. Only
        differences between them mean anything.

    Raises:
        ValueError: A pairing appears from one side only, so the results are
            not a tournament.
    """
    names = sorted(rates)
    for one in names:
        for two in rates[one]:
            if two not in rates or one not in rates[two]:
                raise ValueError(f"pairing {one} vs {two} has no mirror")
    if not names:
        # Nobody to rate.
        return {}
    strength = dict.fromkeys(names, 1.0)
    for _ in range(ITERATIONS):
        updated = {}
        for one in names:
            won = PRIOR_GAMES / 2 + sum(
                rates[one][two] * games[one][two] for two in rates[one]
            )
            spread = PRIOR_GAMES / (strength[one] + 1.0) + sum(
                games[one][two] / (strength[one] + strength[two]) for two in rates[one]
            )
            updated[one] = won / spread
        scale = sum(updated.values()) / len(updated)
        moved = 0.0
        for one in names:
            updated[one] /= scale
            moved = max(moved, abs(updated[one] - strength[one]))
        strength = updated
        if moved < TOLERANCE:
            break
    return {name: math.log(value) for name, value in strength.items()}


def standings(results: list[tuple[str, str, float, int]]) -> dict[str, float]:
    """Rate every agent named in ``results``, which must be a full tournament.

    Args:
        results: One ``(one, two, rate, games)`` per pairing played, where
            ``rate`` is one's win rate against two.

    Returns:
        A rating per agent, best highest.
    """
    rates: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, float]] = {}
    for one, two, rate, games in results:
        rates.setdefault(one, {})[two] = rate
        rates.setdefault(two, {})[one] = 1.0 - rate
        counts.setdefault(one, {})[two] = float(games)
        counts.setdefault(two, {})[one] = float(games)
    return ratings(rates, counts)
