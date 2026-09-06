"""The opponent pool: who a candidate is measured against.

Every opponent counts the same. The gate is an absolute standard -- a
candidate is promoted when it beats *every* opponent here -- so there is no
weight for a candidate to buy a promotion with, and nothing to bend toward
the hardest opponent: beating that one outright is the requirement.

Champions join as gatekeepers, so a later candidate has to beat every
program the campaign has already confirmed as well as the vendored kernels.
At ``POOL_CAP`` a joining champion retires whichever opponent it beats most
decisively, provided it beats it by at least ``RETIRE_THRESHOLD``: the pool
keeps the opponents that still separate programs. Paths are stored here and
shown nowhere.
"""

import os
import time
from pathlib import Path

from pydantic import BaseModel

from kaggriculture.campaign import config, roster


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

    def add_champion(self, name: str, path: str, rates: dict[str, float]) -> str | None:
        """Add ``name`` to the pool, retiring one opponent if it is full.

        Args:
            name: The champion's opponent name.
            path: The champion's own immutable copy.
            rates: The champion's win rate per current pool opponent.

        Returns:
            The name of the retired opponent, or None if none was retired.
        """
        retired = None
        if len(self.opponents) >= config.POOL_CAP:
            crushed = [
                (rate, n)
                for n, rate in rates.items()
                if n in self.opponents and rate >= config.RETIRE_THRESHOLD
            ]
            if crushed:
                retired = max(crushed)[1]
                del self.opponents[retired]
        self.opponents[name] = path
        self.history.append(
            {
                "action": "add_champion",
                "name": name,
                "retired": retired,
                "ts": time.time(),
            }
        )
        return retired
