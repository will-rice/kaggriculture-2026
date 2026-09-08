"""The opponent pool: who a candidate is measured against.

Every opponent counts the same, and the gate is a place rather than a
standard: a candidate is promoted when it comes out top of a Bradley-Terry
tournament over this pool, which is how the competition itself ranks a field.
There is no weight for a candidate to buy a promotion with and no single
opponent it must beat -- only a field it has to finish above.

Champions join as gatekeepers, so a later candidate has to beat every program
the campaign has already confirmed as well as the vendored kernels. Nothing
leaves. The pool used to keep the highest-rated eight and drop the rest, on
the reasoning that an opponent every candidate beats separates two candidates
no better than a coin.

That reasoning holds for a win rate and fails for a rating, and it cost us a
real thing: champion_1 was trimmed out early, and champion_37 -- promoted
thirty times later and rated five log-odds above it -- beats it only 0.729 of
the time, where a rating fitted through the surviving chain says 0.994. The
pool had discarded the one agent that counters our champion, and no gate could
have noticed, because the gate only ever saw what the pool still held.

So the pool keeps everything and `sample` draws the opponents for one gate
from it: anchors that span the range and are played every time, the
highest-rated contenders, and a random remainder. A rating is fitted over
every pairing anyone has ever played, so a candidate only has to add its own
edges to that graph rather than meet the whole field. Paths are stored here
and shown nowhere.
"""

import math
import os
import random
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

    def sample(
        self,
        standings: dict[str, float],
        rng: random.Random,
        exclude: str = "",
    ) -> list[str]:
        """The opponents for one gate, drawn three ways from the whole pool.

        A candidate cannot play a pool of hundreds -- that is thousands of
        games for one verdict -- and with a rating it does not have to. The
        fit is over every pairing the campaign has ever played, so a
        candidate contributes its own edges and is placed against agents it
        never met through the ones it did.

        What it plays has to be chosen rather than drawn flat, because the
        three things a gate needs are different things:

        - **Anchors**, every time. `config.GATE_ANCHORS` spans the strength
          range and never changes, so every candidate has direct edges to
          fixed points at every level. Without them a champion is rated
          through a chain of overlapping pool eras, and that chain is
          measurably wrong: it put champion_37 at 0.994 against champion_1,
          which beats it 0.729 in the games themselves.
        - **Contenders**, the highest rated. Topping the field still means
          beating the best of it, and a candidate that never played the
          leaders cannot be said to have.
        - **The rest, at random.** Coverage, so the graph does not go stale
          everywhere but the top -- and the only way a counter is found
          rather than quietly never played again.

        Args:
            standings: A rating per opponent; anything unrated sorts last.
            rng: The generator the random remainder is drawn from.
            exclude: A name never to draw, so a champion in the pool is not
                measured against itself.

        Returns:
            Opponent names, at most `config.GATE_OPPONENTS` of them.
        """
        available = [name for name in self.opponents if name != exclude]
        drawn = [name for name in config.GATE_ANCHORS if name in available]
        rated = sorted(
            (name for name in available if name not in drawn),
            key=lambda name: -standings.get(name, -math.inf),
        )
        drawn += rated[: config.GATE_CONTENDERS]
        remainder = list(rated[config.GATE_CONTENDERS :])
        room = config.GATE_OPPONENTS - len(drawn)
        if room > 0:
            drawn += rng.sample(remainder, min(room, len(remainder)))
        return drawn[: config.GATE_OPPONENTS]

    def add_champion(self, name: str, path: str) -> None:
        """Put ``name`` in the pool as an opponent every later candidate faces.

        Nothing ever leaves. There was a trim that kept the highest-rated
        eight, and it discarded champion_1 -- which counters our current
        champion at 0.729 where the rating says 0.994. A field this
        non-transitive cannot afford to drop an agent for rating low, because
        rating low against the field and beating *us* are different facts.

        Args:
            name: The champion's opponent name.
            path: The champion's own immutable copy.
        """
        self.opponents[name] = path
        self.history.append({"action": "add_champion", "name": name, "ts": time.time()})
