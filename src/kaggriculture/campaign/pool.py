"""The opponent pool: who a candidate is measured against.

Every opponent counts the same and every candidate plays all of them: a
promotion is measured against the champion over the opponents both played,
and there is no weight for a candidate to buy a promotion with.

The pool holds two kinds of thing on two different terms.

Harvested agents never leave. They are the only evidence about the field we
are actually scored against, the campaign cannot produce another one, and the
competition publishes them faster than they go stale -- so the harvest adds
and nothing removes.

Our own champions are kept to the best `config.POOL_CHAMPIONS`. They join as
gatekeepers, so a candidate must beat what the campaign has already confirmed;
but the tenth-best rung of a ladder we built ourselves teaches a candidate
nothing the best rung does not, and keeping every one of them is what produced
the monoculture: sixty-nine champions holding ten of a gate's twenty-four
slots, so nearly half of every gate replayed our own ancestry.

Nothing used to leave at all, on the reasoning that champion_1 counters
champion_37 at 0.729 where a rating fitted through the surviving chain says
0.994 -- a counter the pool had discarded and no gate could notice. Both
halves of that read differently now. The discrepancy was measured inside a
lineage where every champion was 86.5% one recording and self-play drew thirty
games in thirty-two, which is exactly where a rate and a fit come apart for
want of decided games. And the field is not the non-transitive thing that
argument assumed: a round-robin of fourteen public implementations over 96
fresh seeds in both seats found no intransitive triple, the newer beating the
older in 86 of 91 chronological pairs. On a ladder, an agent that rates low is
the one worth dropping.

Paths are stored here and shown nowhere.
"""

import os
import re
import time
from pathlib import Path

from pydantic import BaseModel

from kaggriculture.campaign import config, roster

# What `gate.promote` names a champion, and so how one is told apart from a
# harvested agent. The two are kept on different terms -- ours are trimmed to
# the best few, a published agent never is -- and this is the only thing that
# distinguishes them.
CHAMPION_PREFIX = "champion_"


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

    def adopt(self, root: Path) -> list[str]:
        """Take every vendored agent under ``root`` the pool does not hold.

        Agents reach ``root`` from two writers and only one of them ever
        wrote the pool. `harvest` adds what it vendors. `tape_opponents`
        writes the families it rebuilds from recorded episodes and stops
        there, on a comment saying "the loop adopts what it finds here by
        the `family_` prefix its names carry" -- and nothing adopted them.
        Measured 2026-09-19: sixty-one playable agents sat unplayed in the
        opponents directory, thirty-six of them families rebuilt from the
        ladder we are scored against.

        So membership is read from the directory rather than from whichever
        script remembered to call `save`. A vendored agent is in the pool
        because it is on disk, which is a thing a new writer cannot get
        wrong the way it could forget a call.

        Matched on the resolved path, never on the directory name. The
        roster holds a dozen of these under short aliases -- `v56` is
        `kaito_v56` -- so adopting by name would enrol a second entry
        against the same file, and a candidate would play that agent twice
        with both copies counting toward its rate.

        Args:
            root: The opponents directory.

        Returns:
            The names adopted, sorted, for the caller to log.
        """
        held = {Path(one).resolve() for one in self.opponents.values()}
        found = sorted(
            one.name
            for one in root.iterdir()
            if (one / "main.py").exists() and (one / "main.py").resolve() not in held
        )
        self.opponents.update({one: str(root / one / "main.py") for one in found})
        return found

    def add_champion(self, path: str) -> None:
        """Put our champion in the pool, and keep the one it replaces.

        `config.POOL_CHAMPION` is the current champion and is overwritten
        every promotion, because there is one champion. The one it displaces is
        not replaced but kept, under `config.POOL_ANCESTOR`, and nothing
        removes it: every rung the campaign has climbed stays in the pool.

        Ancestry is kept because the gate's field comparison excludes the
        champion itself, so a pool of agents the champion already beats gives a
        candidate nothing to fail: champion_19 beat all 67 harvested agents on
        2026-09-16 and the rate pinned at 1.000, where no program that could
        ever exist scores higher. Fifty candidates were refused in six hours,
        every one of them at 0.994 against a champion at 1.000.

        The keys are `ours_N`, not `champion_N`. That prefix is overloaded --
        the pool also holds `champion_1` and `champion_65` from the abandoned
        tape lineage, which are opponents, not rungs of ours -- and trimming by
        it retired an agent the pool was keeping on purpose, then on 2026-09-12
        a promotion numbering itself from an empty champions directory picked
        `champion_1` and overwrote the tape agent of that name outright.
        Harvested names are an author and a kernel slug, so nothing else can
        produce `ours_`.

        Args:
            path: The champion's own immutable copy, which the pool plays.
        """
        standing = self.opponents.get(config.POOL_CHAMPION)
        if standing is not None and standing != path:
            # Named from the file it is, not from a counter passed in: the
            # champion's copy is `champion_N.py` and N is what it was.
            was = re.search(r"champion_(\d+)", Path(standing).name)
            if was:
                key = config.POOL_ANCESTOR.format(number=was.group(1))
                self.opponents[key] = standing
        self.opponents[config.POOL_CHAMPION] = path
        self.history.append({"action": "add_champion", "path": path, "ts": time.time()})
