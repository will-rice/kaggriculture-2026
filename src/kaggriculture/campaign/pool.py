"""The opponent pool: who a candidate is measured against, and how much each counts.

A port of the released FAMOU ``opponent_pool.py``: champions join at a fixed
weight, weights renormalise, a full pool retires the lowest-weight opponent
the new champion already beats, and weakness pressure doubles the weakest
opponent's weight up to a cap. Paths are stored here and shown nowhere.
"""

import time
from pathlib import Path

from pydantic import BaseModel

from kaggriculture.campaign import config, roster


class Pool(BaseModel):
    """The set of opponents a candidate is measured against, with weights."""

    opponents: dict[str, str]
    weights: dict[str, float]
    history: list[dict] = []

    @classmethod
    def initial(cls) -> "Pool":
        """Build the pool from the training roster, with equal weights."""
        names = list(roster.TRAINING)
        return cls(
            opponents={n: str(roster.TRAINING[n]) for n in names},
            weights={n: 1.0 / len(names) for n in names},
        )

    @classmethod
    def load(cls, path: Path) -> "Pool":
        """Load a pool from JSON at ``path``."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        """Save this pool to JSON at ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    def names(self) -> list[str]:
        """Every opponent currently in the pool."""
        return list(self.opponents)

    def weighted(self, rates: dict[str, float]) -> float:
        """Pool-weighted win rate over the opponents present in ``rates``."""
        return sum(self.weights[n] * rates[n] for n in self.opponents if n in rates)

    def weakest(self, rates: dict[str, float]) -> str:
        """The pool opponent with the lowest rate in ``rates``."""
        return min((n for n in self.opponents if n in rates), key=lambda n: rates[n])

    def add_champion(self, name: str, path: str, rates: dict[str, float]) -> str | None:
        """Add at ``CHAMPION_WEIGHT``; retire one crushed opponent if the pool is full.

        Args:
            name: The champion's opponent name.
            path: The champion's floor path.
            rates: Win rates of the outgoing champion against current pool
                opponents, keyed by name.

        Returns:
            The name of the retired opponent, or None if none was retired.
        """
        retired = None
        if len(self.opponents) >= config.POOL_CAP:
            crushed = [
                (self.weights[n], n)
                for n, r in rates.items()
                if n in self.opponents and r >= config.RETIRE_THRESHOLD
            ]
            if crushed:
                retired = min(crushed)[1]
                del self.opponents[retired]
                del self.weights[retired]
        self.opponents[name] = path
        self.weights[name] = config.CHAMPION_WEIGHT
        self._rebalance()
        self.history.append(
            {
                "action": "add_champion",
                "name": name,
                "retired": retired,
                "ts": time.time(),
            }
        )
        return retired

    def apply_weakness_pressure(self, name: str) -> None:
        """Double ``name``'s weight up to ``WEAKNESS_CAP``, scaling the rest down."""
        old = self.weights[name]
        new = min(old * 2, config.WEAKNESS_CAP)
        scale = (1.0 - new) / (1.0 - old)
        for n in self.weights:
            self.weights[n] = new if n == name else self.weights[n] * scale
        self.history.append(
            {
                "action": "weakness_pressure",
                "name": name,
                "old": old,
                "new": new,
                "ts": time.time(),
            }
        )

    def _rebalance(self) -> None:
        """Renormalise weights to sum to 1."""
        total = sum(self.weights.values())
        for n in self.weights:
            self.weights[n] /= total
