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

    def add_champion(
        self, name: str, path: str, standings: dict[str, float] | None = None
    ) -> None:
        """Put ``name`` in the pool, and keep only our best `config.POOL_CHAMPIONS`.

        Our own champions are trimmed; harvested agents never are. They are
        two different kinds of thing. A published agent is evidence about the
        field we are actually scored against, and there is no substitute for
        it -- the campaign cannot generate one. A champion of ours is a rung
        on a ladder we built, and the tenth-best rung teaches a candidate
        nothing the best rung does not.

        Nothing used to leave, on the reasoning that champion_1 "counters our
        current champion at 0.729 where the rating says 0.994" and that a
        field this non-transitive cannot drop an agent for rating low. Neither
        half of that survives the measurements since.

        The counterexample was taken inside a lineage where every champion was
        86.5% one recording: those agents draw thirty games in thirty-two
        against each other, which is exactly where a rate and a fitted rating
        come apart for want of decided games. It was noise in an inbred pool,
        not a counter.

        And the field is not non-transitive. A round-robin of fourteen public
        implementations over 96 fresh seeds in both seats, published
        2026-09-03, found no intransitive triple at all, with the newer
        implementation beating the older in 86 of 91 chronological pairs. It
        is a ladder. On a ladder, the agent that rates low really is the one
        worth dropping.

        What the old rule actually bought was the monoculture: sixty-nine
        champions holding ten of a gate's twenty-four slots, so a candidate
        spent nearly half its gate replaying its own ancestry.

        Args:
            name: The champion's opponent name.
            path: The champion's own immutable copy.
            standings: A rating per agent, deciding which champions stay. The
                new one is always kept -- it just won the gate, and a rating
                fitted before it joined has little to say about it. Without
                standings nothing is trimmed, because dropping champions in an
                order nobody measured is worse than keeping them all.
        """
        self.opponents[name] = path
        self.history.append({"action": "add_champion", "name": name, "ts": time.time()})
        if standings is None:
            return
        ours = [
            other
            for other in self.opponents
            if other.startswith(CHAMPION_PREFIX) and other != name
        ]
        ours.sort(key=lambda other: standings.get(other, float("-inf")), reverse=True)
        for retired in ours[max(0, config.POOL_CHAMPIONS - 1) :]:
            del self.opponents[retired]
            self.history.append(
                {"action": "retire_champion", "name": retired, "ts": time.time()}
            )
