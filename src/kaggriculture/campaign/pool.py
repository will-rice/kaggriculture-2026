"""The opponent pool: who a candidate is measured against.

Every opponent counts the same, and the gate is a place rather than a
standard: a candidate is promoted when it comes out top of a Bradley-Terry
tournament over this pool, which is how the competition itself ranks a field.
There is no weight for a candidate to buy a promotion with and no single
opponent it must beat -- only a field it has to finish above.

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

`sample` draws one gate's opponents from what remains: anchors that span the
range and are played every time, every harvested agent, the highest-rated
contenders, and a random remainder. A rating is fitted over every pairing
anyone has ever played -- retired champions included, since their games stay
on the record -- so a candidate only has to add its own edges to that graph
rather than meet the whole field. Paths are stored here and shown nowhere.
"""

import math
import os
import random
import re
import time
from collections.abc import Sequence
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

    def sample(
        self,
        standings: dict[str, float],
        rng: random.Random,
        exclude: str = "",
        always: Sequence[str] = (),
    ) -> list[str]:
        """The opponents for one gate, drawn three ways from the whole pool.

        A candidate cannot play a pool of hundreds -- that is thousands of
        games for one verdict -- and with a rating it does not have to. The
        fit is over every pairing the campaign has ever played, so a
        candidate contributes its own edges and is placed against agents it
        never met through the ones it did.

        What it plays has to be chosen rather than drawn flat, because the
        three things a gate needs are different things:

        - **Every vendored opponent**, always. They are the only agents in
          the pool this campaign did not write, so they are the only evidence
          about the field we are actually scored against -- and drawing them
          by chance starved them. Measured 2026-09-09: the two that happen to
          be anchors held 33 and 31 pairings, while four harvested on 09-06
          held one between them and `tetsutani_shape0905` had never been
          played at all. There are about a dozen and the set does not grow
          with promotions, so playing all of them is affordable in a way that
          playing the whole pool is not.
        - **Anchors**, every time. `config.GATE_ANCHORS` spans the strength
          range and never changes, so every candidate has direct edges to
          fixed points at every level. Without them a champion is rated
          through a chain of overlapping pool eras, and that chain is
          measurably wrong: it put champion_37 at 0.994 against champion_1,
          which beats it 0.729 in the games themselves.
        - **Contenders**, the highest rated, and ``always`` on top of them.
          Topping the field means beating the best of it, so the top-ranked
          agent is drawn every time rather than left to the dice -- a
          candidate rejected because it happened not to draw the leader would
          be rejected for the sampler's luck. The floor is drawn for the same
          reason: a promotion is a rating gap over it, and a gap against an
          agent you never played is not measurable.
        - **The rest, at random.** Coverage, so the graph does not go stale
          everywhere but the top -- and the only way a counter is found
          rather than quietly never played again.

        Args:
            standings: A rating per opponent; anything unrated sorts last.
            rng: The generator the random remainder is drawn from.
            exclude: A name never to draw, so a champion in the pool is not
                measured against itself.
            always: Names to include whatever the draw says -- the top-ranked
                agent and the floor. Ignored where the pool does not hold
                them, which is the cold start.

        Returns:
            Opponent names, at most `config.GATE_OPPONENTS` of them.
        """
        available = [name for name in self.opponents if name != exclude]
        # `always` leads, and the order is load-bearing rather than tidy: the
        # list is truncated to `config.GATE_OPPONENTS` at the end, and the
        # floor is the one opponent a promotion cannot be measured without.
        # Anything dropped by that truncation has to be a contender or a
        # vendored opponent, never the floor or the leader.
        wanted = [*always, *config.GATE_ANCHORS, *roster.TRAINING]
        drawn: list[str] = []
        for name in wanted:
            if name in available and name not in drawn:
                drawn.append(name)
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
