# Spec: a readable state view, and the divergence report it exists for

**Goal.** Project one turn of Kaggriculture state into a pydantic model that
renders as something an LLM can reason about, and use it to answer the question
we currently answer slowly and badly: _where does our agent decide differently
from a strong one, and why._

**Status.** Spec only. Written 2026-08-05, while Task 7 of the imitation plan is
still measuring whether behaviour cloning clears `starter`. That result should
land before this is planned — if cloning already plays well, the priority here
drops; if it plateaus, this is how we find out what it is failing to learn.

---

## What this is not

It is not an in-episode LLM agent, and no amount of engineering makes it one.

- The competition sandbox has **no network and no API key**. This is why
  `scripts/tracking.py` is quarantined and excluded from the submission archive:
  an import that reaches out forfeits the episode on turn zero.
- The budget is **1 second per turn** with a **60-second overage pool** shared
  across all 720 turns. One API round-trip would consume the whole pool inside
  the first handful of decisions.

Everything below runs offline, on replays we already have.

## Why it is worth building

We have spent this project diagnosing agent behaviour by reading aggregate
numbers. A self-paced loop ran four iterations and rejected three hypotheses —
the liquidation window, planting deadlines, animal valuation — each costing a
full evaluation sweep to disprove. Two other changes were adopted, measured, and
reverted: split-carrier selling banked 41.2k against the trickle's 49.1k, and
throughput-aware animal valuation cost 10k because a static per-animal EV
overrode a crowding term that already reasoned about opponent supply.

In every one of those cases the evidence that would have settled it fastest was
a specific turn, a specific board, and two agents choosing differently on it.
That is what this produces.

## The spine: counterfactual replay

Two agents playing two episodes diverge on the first turn and never face the
same state again, so comparing episodes side by side compares nothing.

Instead, **force the state**. Take a reference episode from the corpus, and at
each turn hand the _recorded observation_ to our agent and record what it would
have done. Same state, two decisions, directly comparable. We already stream
replays this way to build the imitation dataset, so the machinery exists and the
cost is one forward pass per turn.

This yields, per turn, a triple: the state, what the reference did, what we do.
Aggregated it is an agreement rate against a strong player, broken down by op —
a far more diagnostic number than a win rate, because it says _which_ decisions
we get wrong rather than that we lost. Filtered, it is a shortlist of positions
worth a human or an LLM actually looking at.

## Data model

New package `src/kaggriculture/analysis/`. It must never be imported from the
agent path — add it to `package.py`'s exclusions alongside `scripts/`, and pin
that with a test, because the failure mode is silent until a submission dies on
turn zero.

```python
class TileView(BaseModel):
    at: tuple[int, int]
    kind: Literal["PLANT", "WEED", "STRUCTURE", "LOCKED", "EMPTY"]
    crop: str | None
    animal: str | None
    ready: bool
    tended_today: bool          # watered for crops, fed for animals
    neglected_days: int
    yield_units: int
    age_days: int

class FarmView(BaseModel):
    money: int
    farmer: tuple[int, int]
    hands: list[tuple[int, int]]
    hires_today: int
    unlocked_quadrants: list[str]
    tiles: list[TileView]       # occupied tiles only; empties are implied

class ProductView(BaseModel):
    name: str
    price: int
    market_inventory: int
    held: int                   # what we have in the shed

class TurnView(BaseModel):
    day: int
    hour: int
    step: int
    seat: int
    us: FarmView
    them: FarmView
    market: list[ProductView]
    shops: list[str]
```

`TurnView.of(observation, seat)` builds it. The existing
`kaggriculture.observation.Observation` is the natural input — it already wraps
the raw mapping without copying the tile grid.

Listing only occupied tiles matters: a 10×10 board serialized in full is 100
objects of mostly nothing, and the signal drowns.

## Rendering

`render(view) -> str`, markdown, with the two farms as **ASCII grids** rather
than object lists. Spatial structure is the whole point — an LLM reads

```
  0 1 2 3 4 5 6 7 8 9
0 . . . w W W . . . .
1 . . . w W W . . . .
```

and sees a weed block beside a wheat block. Given the same information as a list
of coordinates it does not. A legend maps glyphs to crops, animals, weeds,
structures and locked land; units overlay as digits (0 = farmer, 1..n = hands).

Below the grids: a market table, the unit roster with positions, and money.

**Budget the output.** Assert the render stays under a fixed character ceiling,
tested on the densest late-game turn in a sampled episode — day 29 with full
land unlocked and a large crew. A view that quietly grows past a context window
fails at exactly the moment it becomes interesting.

## Entry points

- `analysis/scripts/inspect.py EPISODE TURN SEAT` — print one rendered turn.
  For piping into whatever you like.
- `analysis/scripts/diverge.py EPISODE AGENT` — run the counterfactual across an
  episode; report the agreement rate overall and per op, and dump the top _N_
  disagreements as rendered turns.

Required argparse arguments take no `--` prefix, per project convention.

## Ranking disagreements

Not all disagreements are interesting. Rank by:

1. **Reference acted, we passed** — the clearest evidence of something unlearned.
2. **Different verb**, not merely a different target — `HARVEST` vs `WATER` is a
   strategy gap; watering a different tile usually is not.
3. **Rarity of the reference's op** in the corpus — a common op we miss is a
   systematic hole; a rare one may be noise.

Report the counts behind the ranking, not just the shortlist. A silent top-_N_
reads as "these are the problems" when it is "these are the first ten".

## Testing

Against real episodes at `/data/kaggriculture/episodes`, marked `slow` and
skipping cleanly without the corpus — the fast suite must not read it.

- `TurnView.of` round-trips a real observation: every occupied tile present,
  positions matching `farm["farmer"]` and `farm["hands"]`, money exact.
- The render stays under the character ceiling on the densest sampled turn.
- The counterfactual finds a genuine disagreement on an episode where our agent
  and the reference are known to differ, and finds none when the agent under
  test _is_ the reference — **this second case is the one that matters**, since a
  comparator that reports disagreements against a perfect copy of itself is
  broken in a way the first test cannot see.

Four tests in the imitation plan passed while agreeing with the bug they were
meant to catch. Every guard here must be observed to fail when the behaviour is
broken, not merely asserted.

## Risks

- **The reference may not be strong.** The corpus is dominated by one public
  kernel. Filter by ladder rating, as `corpus.select` already does, and say which
  reference an agreement rate was measured against — an agreement rate against a
  mediocre agent is a target worth missing.
- **Agreement is not skill.** A policy can match a strong agent on 90% of turns
  and lose on the 10% that decide the game. Treat this as a diagnostic that
  generates hypotheses, never as a substitute for the league gate.
- **Forced state is off-policy.** Our agent sees states its own play would never
  reach, so a disagreement may reflect an unfamiliar position rather than a bad
  decision. This is the standard behaviour-cloning distribution-shift problem and
  it bounds interpretation; it does not invalidate the shortlist.

## Out of scope

Calling an LLM from inside this repo. The render is a string on stdout; feeding
it to a model is a shell pipe away, and building a client here would commit us to
a provider and a prompt before we know whether the view is any good.

Also out of scope: using LLM judgements as training labels. That is a plausible
second teacher alongside the replay corpus, but it should wait until the view has
demonstrably produced a hypothesis that survived measurement.
