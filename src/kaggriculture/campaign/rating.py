"""The tournament the gate is: Bradley-Terry over a candidate and the pool.

The finale is a single Bradley-Terry tournament over the episodes that keep
running after the deadline, and a leaderboard position is a skill rating. The
gate runs that same tournament: every agent plays every other, one fit ranks
them all, and beating the pool means coming out on top of it.

The candidate's own games are played every time. The pool's games against each
other are played once and kept, because they are constants: a game is decided
by the two agents' files and the seed, the opponents are fixed files, and none
of them draws on randomness -- verified by replaying pairings to identical
rates and by there being no `random` call anywhere in the roster. Replaying
them each gate would re-derive numbers already known.

That keeps the pool dynamic without paying for it twice. A champion joining
brings pairings nobody has played, so those are measured and kept too; an
opponent leaving takes its row out and costs nothing. Only a new member's
pairings ever run.

The one thing that would break it is a harvested opponent that draws on
unseeded randomness, which would make a kept rate a stale sample rather than a
fact. `Field.record` is where a rate enters, and `tests/campaign/test_rating`
is where the roster is checked for it.

One latent strength per agent, fitted from every pairing at once, such that
``P(i beats j) = p_i / (p_i + p_j)``. Beating a strong opponent counts for
more than beating a weak one, and one bad matchup is absorbed rather than
fatal -- which is where it differs from asking whether a program beat *every*
opponent. Measured on 2026-09-06, the second-strongest agent in the published
field loses one matchup at 0.062, and an absolute gate turns that agent away.

Ratings are identified only up to a constant, so a number means nothing on its
own: only differences within one fit do. The gate only ever takes differences
within one fit, so that is enough.

The prior matters. Without it an agent that has won nothing has strength zero
and a rating of negative infinity, and almost every program this campaign has
produced has won nothing -- 264 of the first 266. Two phantom games at even
odds keep every rating finite, and the winless then tie at the bottom rather
than being incomparable.

It is blind to margin, because the competition is: the win condition is
relative bank and the margin never counts. Two programs that lose every game
rate the same however close one came, which is why `Database.top` breaks its
ties on the bank margin instead.
"""

import math
import os
from pathlib import Path

from pydantic import BaseModel

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


class Field(BaseModel):
    """The pool's pairings against each other, kept between gates.

    Every rate here is a constant: the agents are fixed files and the games
    are seeded, so a pairing measured once is measured for good. What is
    *not* here is any candidate's row -- a candidate is a new program every
    time and its games are always played fresh.

    Attributes:
        rates: ``rates[a][b]``, a's win rate against b, ties as half.
        games: Games behind each rate. One number, because every pairing is
            played on the same seeds in both seats.
    """

    rates: dict[str, dict[str, float]] = {}
    games: int = 0

    @classmethod
    def load(cls, path: Path) -> "Field":
        """Read the kept pairings, or an empty field if there are none yet."""
        if not path.exists():
            return cls()
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        """Write the pairings, atomically.

        A temporary file and a rename: gates run concurrently, and a reader
        landing inside a truncate would get half a document and a parse error
        nothing catches.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        scratch = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        scratch.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        scratch.replace(path)

    def missing(self, names: list[str]) -> list[tuple[str, str]]:
        """Pairings among ``names`` not measured yet, in a stable order.

        A pool that gained a champion has that champion's pairings missing
        and nothing else, which is the whole point: only a new member's
        pairings ever run.
        """
        return [
            (one, two)
            for index, one in enumerate(names)
            for two in names[index + 1 :]
            if two not in self.rates.get(one, {})
        ]

    def record(self, one: str, two: str, rate: float) -> None:
        """Keep a measured pairing, both ways round.

        Args:
            one: An agent.
            two: The agent it played.
            rate: One's win rate against two, ties as half.
        """
        self.rates.setdefault(one, {})[two] = rate
        self.rates.setdefault(two, {})[one] = 1.0 - rate

    def results(self, names: list[str]) -> list[tuple[str, str, float, int]]:
        """The kept pairings among ``names``, as `standings` takes them.

        Restricted rather than returned whole, because an opponent that has
        left the pool is not in this tournament even though its games are
        still on the record: dropping it costs nothing, and if it ever comes
        back its pairings are still here.
        """
        return [
            (one, two, self.rates[one][two], self.games)
            for index, one in enumerate(names)
            for two in names[index + 1 :]
            if two in self.rates.get(one, {})
        ]
