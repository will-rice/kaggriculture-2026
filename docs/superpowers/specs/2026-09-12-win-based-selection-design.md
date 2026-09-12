# Selection by games won

Date: 2026-09-12. Replaces the rating half of
`2026-09-05-campaign-script-design.md` sections 5.1 and 5.2.

The campaign has produced 79 programs and zero champions. This design is what
the measurements taken that day say to do about it.

## What was wrong

Every mechanism that would let a search climb was broken at once, so none of
them was ever tested.

- **Promotion could not fire.** `gate.promotion` requires `place == 1` of a
  Bradley-Terry tournament in both of its regimes -- the "no floor yet" branch
  sits behind the same check. The best program the campaign has produced placed
  25 of 28 at a rating of -1.983, against +1.634 for rank 1: a gap of 3.617 in
  log-odds, 628 Elo. Four of the five agents above it are champions of the
  abandoned tape lineage, whose strength was 86.5% verbatim replay out of a
  720-step recording -- the mechanism `ALLOWED_IMPORTS` bans for our own
  candidates. Three of them (`champion_10`, `champion_20`, `champion_30`) are
  not in the pool at all and cannot be drawn, so they hold ranks no candidate
  can ever play for.
- **Nothing was retained.** With no promotion there is no champion, so
  `POOL_CHAMPIONS` never engaged, no floor was ever written, and the search was
  free to re-converge. It did: of 14 programs in the run of 2026-09-12, five
  descended from a parent at 0.206 and every child landed at 0.180-0.203.
- **Selection tied the top.** `Database.top` ranked on the fitted rating.
  `p0ffda21082e1` (win rate 0.206) and `p7439c25a364e` (0.331) had the same
  rating of -1.983, so `PARENT_DECAY ** rank` drew the worse one about half the
  time and the better one a quarter. The second and third best by games won sat
  at rating ranks 14 and 10, outside `PARENT_POOL` entirely.
- **The refusals were silent.** `consider` logged the gate's reason only when
  it promoted. Seventy-nine precise refusals were discarded, so the absence of
  a champion had no explanation in the log or in wandb.
- **The prompt asked for the wrong thing, locally.** The instruction was
  "finish every season with a larger bank"; the work loop said a program is
  improved by finding one bad day in one season and fixing it; the feedback was
  24 rows standing for 768 games. All 79 programs hold between 5,777 and 6,236
  on day 10 -- a band 460 wide -- against opponents averaging 16,219.

None of that is evidence about what a model can write. It is evidence that the
scaffolding held the search in one basin. Handed the games database and the
`query-games` skill, codex produced an unprompted and correct diagnosis of the
day-10 problem before any of this was measured by hand.

## The design

### The pool

Every harvested public agent, plus the most recent champion. One champion,
replaced on promotion rather than accumulated: `POOL_CHAMPIONS` becomes 1.

The previous campaign accumulated them and became self-play -- 69 champions
holding 10 of 24 slots, the field frozen on the day someone last ran the
harvest by hand. One champion keeps the ratchet (a candidate must beat what we
have) without the pool drifting toward our own lineage.

### The score

Win rate over that pool. Every candidate plays all of it, on the session's
shared seed block, in both seats.

This is the change that removes Bradley-Terry from the campaign. A rating is
the right instrument for a sparse, unbalanced graph -- A played B, B played C, A
never played C -- and the campaign had that shape for exactly one reason:
`GATE_OPPONENTS = 24` drawn from a pool of 34, re-drawn per candidate, leaving
two candidates' win rates incomparable. Playing everyone makes the design
complete and balanced, and the win rate is then the whole of what the rating was
estimating.

Measured 2026-09-12: 24 of 34 opponents is 768 games and 6.4 minutes at five
workers; all 34 is 1,088 games and 9.1 minutes, against a codex call that takes
ten to twenty-five. The rating cost an additive constant that is undefined
(hence `_anchored`), a dependence on other agents' stored pairings (hence
`field.json` and the three unplayable champions), and 0.745 of fit noise on an
unchanged agent -- to save 320 games.

This makes the pool's size a cost multiplier where it used to be free: 60
opponents is 16 minutes an evaluation and 100 is 27. The cap on the pool is
load-bearing from here.

Bradley-Terry stays in `dataset.py`, fitted over the ladder corpus, where the
48,337 recorded games were paired by Kaggle rather than by us and the graph
really is sparse. That is what tells `strategies` which public agent is
stronger.

The frozen `field` metric -- a win rate over `roster.TRAINING`, 12 vendored
agents -- is retired as a selection signal. It was comparable across time
because it ignored the field getting stronger: 22 of the pool's 34 opponents are
invisible to it, including `thomastschinkel_kaggriculture_93_8_win_r`, the
strongest agent in the pool. Comparability now comes from the head-to-head
below, which is measured directly on the same seeds every time.

### Promotion

Two conditions, both read off the candidate's own evaluation:

1. a higher pool win rate than the champion's, as recorded when the champion was
   promoted, and
2. beats the champion head-to-head: the Wilson lower bound of
   `result.rates[champion]` above 0.5, over at least `DECISIVE_GAMES` decisive
   games.

Condition 1 compares across blocks, and that is a known weakness rather than an
oversight: the champion's recorded rate was measured on its own seeds against
the pool as it then stood, and seed blocks rotate every `SEED_ROTATION`
candidates while harvest adds opponents. It is a cheap screen, not a proof.
Condition 2 is the rigorous half -- same block, same seeds, both seats, paired,
with the interval widening when the evidence is thin -- and it is the one that
makes a promotion mean something. Re-scoring the champion alongside every
candidate would put condition 1 on the same footing too, and costs a second full
evaluation per round; it is not worth that, given condition 2 is required
anyway.

The champion is in the pool, so condition 2 is always directly measured, paired
on the same seeds in both seats. `DECISIVE_GAMES = 8` stays because it caught a
real case: champion_55 was promoted over champion_54 on two wins and thirty
exact draws in 32 games -- two programs playing the same game, which a rating
cannot distinguish from seventeen wins and fifteen losses.

Both conditions are required. Condition 1 alone promotes an agent that beats the
field on average and loses to the specific thing it replaces; condition 2 alone
promotes an agent that counters the champion and nothing else. The field is
non-transitive -- shopforge scores 0.979 against the field and 0.6875 against
v56 -- so neither implies the other.

**Cold start.** With no champion, condition 2 has nothing to measure and the bar
is condition 1 against the best program in the database. Reachable from the
first round: the seed's win rate is what has to be beaten, not rank 1 of 28.

### What a round is given

One game. Its episode key and the query that reads it out of the games
database, day by day and both sides. Not the game itself: at full width one
season is 8,888 characters across 68 columns, and the database already holds it
along with every recorded competition game, so a round reads whichever columns
its own question wants.

A different game each round, so the `ROUNDS_PER_SESSION = 12` rounds of a
session see twelve maps.

Variety rather than depth, and the division of labour is the reason. The
campaign's job is to stop the search entrenching on one map: a change that helps
only the map that motivated it gets no second round to build on. Verification is
the round's own, with `measure.py` in its directory.

This makes the chain load-bearing rather than a nicety. Twelve rounds on twelve
maps are twelve independent attempts unless something ties them together, and the
only thing that does is what each round changed and the win rate it got. So: for
each round before it in this session, what it changed and the win rate that came
back.

The chain is the session's own record, held in the session, not a query over a
lineage graph. It needs a short description of what each round changed, taken
from the docstring at the top of the file that round wrote -- which means the
round prompt asks for that docstring again. It asked once and the line was cut
earlier on 2026-09-12 while trimming instructions about _how to work_; a record
of what was done is a different thing.

### No lineage

Nothing records or consults what came from what. `Program.started_from`,
`Database.children`, `Database.descendants`, the scratch lineage
(`SCRATCH_CHANCE`, `SCRATCH_ID`, `SCRATCH_AGENT`) and the stagnation note are all
deleted. A program's ancestry has no bearing on whether it wins, which is the
only thing being selected on.

A round still edits a file, and that file is whatever the round before it in the
session produced -- but that is the session's local state, not a recorded
relation, and nothing downstream can ask about it.

What goes with the scratch lineage is the campaign's only mechanism for starting
outside the seed's basin, and that basin is where all 79 programs sat in a
460-coin band. The replacement is the instruction: "Write a program that beats the
opponent" permits a rewrite where "finish every season with a larger bank" asked
for an edit. The escape belongs in what a round is asked for, not in a special
category of program.

### What `measure.py` measures

The champion, on `GATE_SEEDS` seeds in both seats: 32 games, about twenty
seconds at five workers. Before there is a champion, the best program in the
database.

It compared the round's program against `parent.py`, a copy of whatever the round
was handed. That is gone with the rest of the lineage, and what replaces it is
better: the head-to-head rate against the champion, with its Wilson interval, is
_exactly_ promotion condition 2. The round's local tool and the gate's bar become
the same statistic, computable in twenty seconds, so a round can check the real
thing instead of a proxy for it.

`--seeds` trades time for tightness. Playing the whole pool is 1,088 games and
nine minutes, which is too slow to iterate against and is the campaign's job
anyway.

### Selection

A session starts from a program drawn from `Database.top(DRAW_POOL)`, ranked on
the pool win rate, weighted `DRAW_DECAY ** rank`. The rank-0 tie is what this
fixes.

`PARENT_POOL` and `PARENT_DECAY` are renamed: with no lineage there are no
parents, only the best programs there are and how sharply the draw favours
them.

## What is deleted

| deleted                                                              | why                                                   |
| -------------------------------------------------------------------- | ----------------------------------------------------- |
| `rating` from the campaign path                                      | the design is balanced; the win rate is sufficient    |
| `field.json`, `rating.Field`                                         | stored pairings existed only to connect the fit       |
| `gate.refresh`, the startup anchor pairings                          | nothing needs a connected graph now                   |
| `GATE_ANCHORS` as a rating origin, `_anchored`                       | no additive constant left to pin                      |
| `GATE_OPPONENTS`, `GATE_CONTENDERS`, `must_play`, `Pool.sample`      | everyone is played                                    |
| `Program.rating`, `Program.place`                                    | nothing selects or promotes on them                   |
| `Program.started_from`, `Database.children`, `Database.descendants`  | ancestry does not bear on winning                     |
| `SCRATCH_CHANCE`, `SCRATCH_ID`, `SCRATCH_AGENT`, the stagnation note | lineage machinery; the instruction carries the escape |
| `parent.py` in the round directory                                   | `measure.py` measures the champion now                |
| the `place == 1` gate and its paired-margin test                     | replaced by the two conditions above                  |
| the frozen `field` win rate as a selection signal                    | ignores 22 of 34 opponents                            |

`rating.py` itself stays, for `dataset.py`.

`Program.field` stays as a recorded number and nothing reads it to decide
anything. It is the only quantity in the existing archive that means the same
thing across the whole campaign, so the 79 programs already written stay
comparable to each other; new programs keep getting it for the same reason. Its
retirement is from selection and promotion, not from the record.

## Already implemented

Green and uncommitted at the time of writing, from the same day:

- `evaluator.score` plays the whole pool rather than a draw of 24.
- `Database.top` ranks on the win rate, with the margin as tie-break.
- `Kept` carries the evaluation, the standings and the gate's verdict out of
  `keep`, so a round makes one fit where it made three and `session` no longer
  re-asks the gate a question it has answered.
- `consider` logs its reason whether it promotes or refuses.

## Testing

Each of these is a test that fails before the change and passes after, verified
by mutation rather than assumed:

- A candidate with a better pool win rate and a decisive head-to-head win over
  the champion is promoted; one with either condition alone is not.
- With no champion, the first candidate beating the database's best is
  promoted -- the condition that was unreachable before.
- A candidate that draws nearly every game against the champion is refused
  however its win rate reads.
- `Database.top` prefers the program that won more games over one with a higher
  stored rating.
- The round message names one game and carries the chain of earlier rounds with
  what each changed and scored.
- Twelve rounds of a session are given twelve different games, so a seed held
  across rounds fails the test.
- `measure.py` reports the head-to-head rate against the champion, and the
  number it prints matches what the gate computes for condition 2 on the same
  games.
- Nothing in the campaign reads `started_from`, and a program can be scored,
  ranked and promoted without it.
- No module outside `dataset.py` imports `rating`.

## Seed robustness, and what carries it

Worth stating in one place, because it is spread across four mechanisms and none
of them is obviously the one doing the work.

- **The score is not one map.** 16 seeds, both seats, the whole pool: 1,088
  games. A fixed plan's bank swings about 19.5% season to season, so a single
  season decides nothing here.
- **Condition 2 is paired.** Candidate and champion on identical seeds in both
  seats, so the episode's own swing lands on both sides and cancels. Measured:
  telling apart a five-thousand-coin difference takes about 114 games unpaired
  and about 4 paired.
- **The chain reports the global rate, not the map's.** A round that improved
  only the map it was shown sees the number fail to move, which is the feedback
  that punishes a map-specific hack.
- **Feedback rounds are different maps.** Twelve rounds, twelve seeds, so
  nothing gets twelve consecutive attempts at entrenching on one.

What is deliberately _not_ here: a confirmation pass on a fresh block before
promoting. It is the textbook winner's-curse guard and a two-stage gate was
deleted once already, because its cheap stage was 8 seeds and 78 of 471 programs
topped it with none surviving the deep look. The four mechanisms above are the
bet that one measurement at 16 paired seeds is deep enough to promote on. If
promotions start arriving and then failing to hold up, a fresh-block confirmation
is the first thing to add, and it is cheap because it runs only on a promotion.
