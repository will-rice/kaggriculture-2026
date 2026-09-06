"""The opponent pool: who a candidate is measured against.

Every opponent counts the same, and the gate is a place rather than a
standard: a candidate is promoted when it comes out top of a Bradley-Terry
tournament over this pool, which is how the competition itself ranks a field.
There is no weight for a candidate to buy a promotion with and no single
opponent it must beat -- only a field it has to finish above.

Champions join as gatekeepers, so a later candidate has to beat every
program the campaign has already confirmed as well as the vendored kernels.
The pool is the top of the tournament. A champion joins it and `trim` keeps
the highest-rated ``POOL_SIZE``, so the field a candidate has to finish above
is the strongest one there is and the weakest opponent makes way. Rating is
what decides, and it is the same rating the gate promotes on, so the pool and
the bar cannot drift apart. Paths are stored here and shown nowhere.
"""

import math
import os
import time
from pathlib import Path

from pydantic import BaseModel

from kaggriculture.campaign import roster


class Pool(BaseModel):
    """The set of opponents a candidate is measured against."""

    opponents: dict[str, str]
    history: list[dict] = []

    @classmethod
    def initial(cls) -> "Pool":
        """Build the pool from the training roster."""
        return cls(opponents={n: str(p) for n, p in roster.TRAINING.items()})

    @classmethod
    def load(cls, path: Path) -> "Pool":
        """Load a pool from JSON at ``path``."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        """Save this pool to JSON at ``path``, atomically.

        A temporary file and a rename, not a truncate-and-write: every
        evaluation re-reads this file to resolve a champion's name, in a
        thread, while a promotion rewrites it on the loop. A read landing
        inside a truncate returns half a document, and the parse error that
        follows is a `ValueError` that no handler catches, so it would take
        the whole campaign down.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        scratch = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        scratch.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        scratch.replace(path)

    def names(self) -> list[str]:
        """Every opponent currently in the pool."""
        return list(self.opponents)

    def trim(self, standings: dict[str, float], size: int) -> list[str]:
        """Keep the ``size`` highest-rated opponents; drop and name the rest.

        The pool exists to tell candidates apart, and an opponent every
        candidate beats does that no better than a coin: it costs a game a
        round and buys nothing. Rating is what says which those are, and it
        is the same rating the gate promotes on, so the pool and the bar
        cannot drift apart.

        Args:
            standings: A rating per opponent, from one fit over the pool.
            size: How many to keep.

        Returns:
            The names dropped, weakest first.
        """
        ranked = sorted(self.opponents, key=lambda n: -standings.get(n, -math.inf))
        dropped = ranked[size:]
        for name in dropped:
            del self.opponents[name]
        if dropped:
            self.history.append(
                {"action": "trim", "dropped": dropped, "ts": time.time()}
            )
        return list(reversed(dropped))

    def add_champion(self, name: str, path: str) -> None:
        """Put ``name`` in the pool as an opponent every later candidate faces.

        Nothing leaves here. Who leaves is `trim`'s decision and it takes it
        on rating, from the same tournament that promoted this champion, so
        the pool and the bar cannot drift apart. Adding and trimming were one
        method once, with a retirement rule of its own -- an opponent beaten
        at 0.95 or better made way -- and that rule could keep an opponent no
        candidate had crushed but every candidate beat.

        Args:
            name: The champion's opponent name.
            path: The champion's own immutable copy.
        """
        self.opponents[name] = path
        self.history.append({"action": "add_champion", "name": name, "ts": time.time()})
