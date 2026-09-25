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

# Four, and every one of them a different agent. It was six, and four of those
# were champion_1, _10, _20 and _30 -- which read as four points spanning the
# strength range and were four ages of one recording. The 720-step table
# underneath that lineage is byte-identical from its seed through champion_69,
# sha ef59f6f4a545d342, and 86.5% of every action any of them emits comes
# straight out of it; what differs between them is the repair layer over the
# other 13%.
#
# An anchor's whole job is to be a fixed point a rating is calibrated against,
# so anchoring the scale to one agent at four ages is the failure that
# calibration exists to prevent. champion_65 replaces the three: it is the
# strongest of that lineage and the bar a submission has to clear, so it earns
# a slot on its own account rather than as a reference point.
#
# The rest of that lineage left the pool with them. Sixty-nine champions held
# ten of a gate's twenty-four slots, which bought ten readings of one
# recording; its run is kept whole under `run/campaign-tape-lineage/` and its
# pairings stay in `field.json`, so the fit still places it.
# Highest-rated agents drawn beyond the anchors and the vendored set. Four
# rather than six because the leader is already drawn through `always` and the
# vendored opponents now take a dozen slots: the contenders were competing for
# room with the only cross-population evidence the gate gets.
GATE_CONTENDERS = 4

# Opponents drawn for one gate. The pool itself is now everything the campaign
# has ever produced or harvested and nothing leaves it, so this is a sample
# and not the pool: a rating is fitted over every pairing anyone has ever
# played, and each candidate only has to add its own edges to that graph.
#
# The pool used to keep the top eight by rating and drop the rest, on the
# reasoning that an opponent every candidate beats separates two candidates no
# better than a coin. That is true of a *win rate* and false of a rating, and
# it cost us: champion_1 was trimmed out long ago, and champion_37 -- thirty
# promotions later, rated five log-odds above it -- beats it only 0.729 of the
# time. A field this non-transitive keeps its counters or walks past them.
#
# Twenty-four rather than sixteen, because the draw now includes every vendored
# opponent: about a dozen of those, plus the leader and floor, plus the four
# anchors that are not themselves vendored, plus `GATE_CONTENDERS`. Sixteen
# would have truncated exactly the agents the change exists to include.
#
# The cost is real and was weighed against playing the pool entire. That is
# roughly 75 opponents today, 2,400 games a candidate against 512, and it grows
# with every promotion -- while buying no extra *share* of cross-population
# evidence, since the pool is itself 84% champions. Drawing all the vendored
# agents and sampling the rest lifts that share from about a sixth to about a
# half for half the added cost, and does not grow.
GATE_OPPONENTS = 24

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
            Opponent names, at most `GATE_OPPONENTS` of them.
        """
        available = [name for name in self.opponents if name != exclude]
        # `always` leads, and the order is load-bearing rather than tidy: the
        # list is truncated to `GATE_OPPONENTS` at the end, and the
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
        drawn += rated[:GATE_CONTENDERS]
        remainder = list(rated[GATE_CONTENDERS:])
        room = GATE_OPPONENTS - len(drawn)
        if room > 0:
            drawn += rng.sample(remainder, min(room, len(remainder)))
        return drawn[:GATE_OPPONENTS]

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
