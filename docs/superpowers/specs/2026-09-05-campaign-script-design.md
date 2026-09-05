# The campaign

Date: 2026-09-05. The simplest system that can work, written from scratch.

Eight coding agents run at once. Each is handed the best program we have
and told to beat it. Anything that beats it, measured properly, becomes
the new best, and the next eight agents start from that.

The architecture is AlphaEvolve's (2506.13131) asynchronous controller
loop; the selection is FAMOU's (2606.10389) co-evolving evaluator with a
fast/deep cascade. Everything either paper does that this does not need is
listed in §9 and left out on purpose.

## 1. The script

`loop.py`, `main()` at the top, four steps:

1. **Initialize wandb from `{model}-{hash}`.**
2. **Start the event loop.**
3. **Spin off `SESSIONS = 8` workers**, each running §3 forever.
4. **Gate the improvements** (§5), fired by what the workers produce.

## 2. Definitions

**Program.** One Python file whose last top-level callable is
`agent(observation, configuration)`. That is what Kaggle loads, so it is
what we evolve and what we ship. Nothing else: no package, no markers.

**Game.** 720 turns of Kaggriculture on our C++ engine port, one program
in each seat, won by relative bank. The port is bit-identical to Kaggle's
`kaggle_environments` 1.32.7 (§7) and fast enough that games are free.

**Pool.** The opponents a program is scored against, with weights summing
to one — FAMOU's evaluator `E = (O, w, F)`. It starts as six vendored
public kernels at equal weight and grows as champions join it (§5.3).

**Fitness.** `F(c) = Sum_i w_i * winrate(c, o_i)` over the pool, both
seats per seed, ties counting half. This one number is what selection
uses.

**Champion.** The best program the campaign has confirmed on the sealed
exam block. There is exactly one. It is the program we would submit, the
program every session starts from, and a member of the pool.

**Database.** Every program ever kept, with its scores. Shared by all
eight workers: they all start from the same champion and all write here.

## 3. A worker

```
while True:
    box  = build_sandbox(champion, instruction, feedback)     # section 4
    child = await session(box)                                # section 6, subprocess
    if no child:            record why; continue
    if not validate(child): record why; continue
    result = fast(child, pool)                                # section 5.1
    program = database.add(child, result)
    if program is in the top DEEP_TOP_K and has no deep result:
        create_task(gate(program))                            # section 5
```

That is AlphaEvolve's controller loop — `database.sample` then
`prompt.build` then `llm.generate` then `evaluator.execute` then
`database.add` — with the diff step removed, because the session edits the
file itself.

Eight workers run this concurrently against one database, one pool and one
champion. A promotion by any worker changes what the other seven start
from next. There are no islands, no subpopulations, no worker identity in
any record: the concurrency is the only structure.

## 4. What a session is given

A directory, built fresh:

| file          | content                                                                                                                                                                  |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `AGENTS.md`   | the game: objective, rules, verified economics, the interface, how to run the harness, and the doctrine — measure opponents through the harness, never read their source |
| `child.py`    | the champion, copied in; the file the session edits                                                                                                                      |
| `feedback.md` | the bar (4.1), the champion's win rate against each pool opponent, the pool weights, the weakest opponent by name, and the last three failures on this lineage           |
| `PROMPT.md`   | the instruction (4.2) and the budget                                                                                                                                     |
| `engine/`     | the C++ port and Kaggle's engine source, so the session can read the rules as code and run its own games at speed                                                        |

### 4.1 The bar

The session is told to **keep editing `child.py` until it beats the
champion**, and to test that itself with the harness as it goes:

1. **Higher weighted pool fitness than the champion's**, over every pool
   opponent — the vendored kernels and every past champion still in the
   pool.
2. **A winning head-to-head record against the champion**, which is a pool
   opponent by name.

Both, because the game is non-transitive: a session told only to beat the
champion would breed champion-counters. It stops as soon as it clears
both; a child that clears only the first is still worth writing.

### 4.2 The instruction

One of five, drawn uniformly (FAMOU App. C.2): improve it; design a
completely different algorithm; a novel approach inspired by the context
program; restructure the core components; tune constants and thresholds
only. Each ends with: keep what the feedback says is winning, change what
is losing, never read or reconstruct an opponent's source. The draw is
recorded.

Eight sessions therefore start from the same code and are pushed eight
different ways, each a separate stochastic run of an agent that spends
twenty minutes searching on its own.

## 5. The gate

FAMOU Algorithm 1, lines 7-10: when a new best appears, confirm it deeply,
add it to the opponent pool, and apply weakness pressure. The trigger is
an event, not a schedule.

### 5.1 Two evaluations

AlphaEvolve's cascade: cheap first, and only what survives goes on.

- **Fast** — `FAST_SEEDS = 4` fresh seeds, never the exam block, every
  pool opponent, both seats. About a minute. This is what enters the
  database and ranks it. FAMOU's equivalent is 3 games per opponent, "for
  coarse population ranking".
- **Deep** — the sealed `EXAM_SEEDS = 64`, every pool opponent and every
  held-out opponent, both seats. About ten minutes. Returns the fitness
  with 95% Wilson bounds per opponent, the equal-weight rate over the
  vendored opponents (`field`), and the held-out rates. FAMOU: 20 games
  per opponent, "for true score confirmation"; "final decisions use deep
  evaluation".

A program that enters the top `DEEP_TOP_K = 3` by fitness is deep-scored
once, ever, bounded by `DEEP_CONCURRENCY = 2` at a time. FAMOU measured
Spearman 0.11 between their fast and deep scores; we log the same
statistic over every program that has both.

### 5.2 Promotion

The deep result is compared with the champion's. Promote if all three
hold: the lower bound beats the champion's score; `field` has not dropped
by more than `FIELD_TOLERANCE = 0.02`; no single opponent's rate has
fallen by more than the wider of the two intervals. The first champion
needs no comparison.

Then, in order: package the program into `champions/<name>.tar.gz` — the
file `submit` uploads — and play it in the Kaggle docker image; a tarball
that will not run there is not promoted. Write the champion file, the
floor, `champion.json`, and log the tarball as a wandb artifact.

**A cut is one command**: upload the current champion's tarball. Nothing
is built at cut time. Nothing is ever committed to git; every write is
under `run/campaign`.

### 5.3 The pool co-evolves

FAMOU 4.4: the new champion "becomes a high-weight gatekeeper, so later
candidates must outperform both original opponents and all previous
champions". It joins the pool at `CHAMPION_WEIGHT = 0.20`; at
`POOL_CAP = 10` the lowest-weight opponent it beats at 0.95 or better
retires.

FAMOU 4.5, weakness pressure: the opponent the champion beats least has
its weight doubled, capped at `WEAKNESS_CAP = 0.5`. Its name goes into
`feedback.md`, so the pressure reaches the prompt as well as the score.

Then the new champion is re-scored on the pool it just changed, and that
becomes the number the next candidate is compared against.

**Held-out** opponents are scored and never trained against:
`salemali7_2900`, `lynnsakurai_v5`, and each new ladder kernel found,
which enters here first. This is the honest generalisation number.

## 6. A session

One `codex exec` process in the sandbox directory, `workspace-write`,
`approval_policy=never`, `--json`, transcript captured. It is an agent,
not a completion: it reads the files, runs the harness, runs its own
games, and edits `child.py` until it clears the bar.

**Short-lived.** It exits when it clears the bar or spends
`SESSION_LIMIT_SECONDS = 1500`; then its process group is killed and
whatever `child.py` holds is evaluated. Nothing carries over inside the
agent — the database is what carries over.

**Two models.** `gpt-6-astra`, falling back once to `gpt-5.6-sol` when the
provider refuses the turn (astra reports "at capacity" some of the time)
or codex dies. A failure on both is logged and the worker starts its next
session; it is never charged to the program's lineage.

## 7. Machine

Games run on a C++ port of the Kaggle environment through ctypes. Its only
justification is being identical, so that is tested, not asserted: 200
archived episodes from the real competition replay bit-for-bit, and 2% of
live games are audited against Kaggle's own engine. The same library ships
in every champion tarball, so an evolved program may use it for lookahead.

Evaluations fan their games over `CORE_BUDGET // SESSIONS` processes, one
game per process. `CORE_BUDGET` is the machine's cores less eight, kept
for the sessions' own harness runs. Every game runs in its own scratch
directory.

**No latency requirement.** These are rule-based policies: microseconds a
turn. A step time measured here would not predict Kaggle's anyway, and a
program slow enough to matter is a rewrite in a compiled language, not a
rejection. Nothing measures or thresholds per-step time.

**Two time limits, both liveness, neither about speed.** A session
(`SESSION_LIMIT_SECONDS = 1500`) and a validation game
(`GAME_LIMIT_SECONDS = 120`, because a 720-turn game takes under a second,
so one still running after two minutes is stuck). Both are crude on
purpose: generous enough never to fire on working code, small enough that
a wedged worker is back within the hour.

**Validation**, before any game is played for score: it parses; `agent` is
the last top-level callable, by reading the file and by loading it through
Kaggle's loader; it imports only from an allowed list and calls no
`__import__`, `eval`, `exec`, `open`; it shares no 8-token shingle set
with any opponent above 0.03 Jaccard; it plays a full game without
raising. Every program kept is therefore shippable.

**State.** The database is an append-only JSON-lines log replayed on
start, holding one record per program (source, what it started from, the
instruction, fitness, per-opponent rates), one per deep result, and one
per failure. `state.json` after every session; `champion.json` written by
the gate. A restart resumes all of it and loses only the sessions in
flight. Signals cancel the workers and kill their process groups.

**Stagnation.** If `STAGNATION_SESSIONS = 40` sessions pass with no
promotion, sessions start from a program drawn from the database's top ten
instead of the champion, and say so in the prompt. FAMOU's trajectories
are "punctuated equilibrium, long stagnation interrupted by sudden
breakthroughs", and the breakthrough often comes from a line that was
behind. One branch, one constant.

**Run identity.** The wandb run id and name are both
`{CODEX_MODEL}-{revision}`, `resume="allow"`. The script refuses to start
with uncommitted changes under `src/`: a run named by a hash must be that
hash.

## 8. Constants

| constant                | value       | source                  |
| ----------------------- | ----------- | ----------------------- |
| `SESSIONS`              | 8           | fits the machine        |
| `SESSION_LIMIT_SECONDS` | 1500        | measured; FAMOU 300     |
| `GAME_LIMIT_SECONDS`    | 120         | liveness                |
| `FAST_SEEDS`            | 4           | FAMOU 3 games/opponent  |
| `EXAM_SEEDS`            | 64, sealed  | FAMOU 20 games/opponent |
| `DEEP_TOP_K`            | 3           | FAMOU                   |
| `DEEP_CONCURRENCY`      | 2           |                         |
| `CHAMPION_WEIGHT`       | 0.20        | FAMOU gatekeeper        |
| `POOL_CAP`              | 10          |                         |
| `RETIRE_THRESHOLD`      | 0.95        |                         |
| `WEAKNESS_CAP`          | 0.5         | FAMOU x2, renormalise   |
| `FIELD_TOLERANCE`       | 0.02        |                         |
| `STAGNATION_SESSIONS`   | 40          |                         |
| `CODEX_MODEL`           | gpt-6-astra |                         |
| `CODEX_FALLBACK_MODEL`  | gpt-5.6-sol |                         |
| `CORE_BUDGET`           | cores - 8   |                         |

## 9. Deliberately not built

- **Islands, migration, island resets, MAP-Elites, UCB parent selection.**
  Every session starts from the champion, so a subpopulation has nothing
  to add that the champion does not carry. Diversity is the five
  instructions, the model's sampling, and the stagnation branch.
- **A daily or weekly call cap.** The provider's limits are the limits.
- **Meta-prompt evolution** (AlphaEvolve 2.2) and **LLM-graded feedback**
  (2.4). The evaluator is games.
- **SEARCH/REPLACE diffs** (AlphaEvolve 2.3). The agent edits in place.
- **Multiple scores per program** (AlphaEvolve 2.4). One fitness, plus the
  deep record.
- **Sessions that outlive a restart.** A restart kills them.

## 10. Telemetry

Per session, on the `calls` axis: whether it produced a child, whether it
timed out, whether it fell back, tokens in and out, seconds, the child's
fitness, the database's best fitness and size. Per deep evaluation: score,
bounds, `field`, per-opponent and held-out rates, the champion's score,
the fast/deep rank correlation, and whether it promoted. Each promotion
logs its tarball as an artifact.

## 11. Tests

Real code, not mocks. Only the codex process is substituted — a fake that
edits the file, or a shell command in codex's place. Real validation, real
games through the real harness and process pools, real database, pool and
gate, real files, wandb disabled.

1. End to end: two workers, two sessions; the better child is deep-scored
   on insert and promoted with no schedule; champion file, floor, pool,
   tarball and `champion.json` as specified; one deep record that
   promoted.
2. Eight workers' sessions overlap in time.
3. A promotion changes what the next session starts from.
4. A program is deep-scored once; a restart reads the stored result.
5. After a promotion the champion is re-scored on the new pool and the
   next comparison uses that score.
6. A candidate whose tarball will not run in the Kaggle image is not
   promoted and the floor is unchanged.
7. A provider failure falls back, and a failure on both models is not
   charged to the lineage.
8. Cancellation kills the session's process group; an opponent crashing in
   its own seat stops the run.
9. Stagnation switches the starting program and the prompt says so.
10. Dirty `src/` refuses to start; a clean tree names the run and a second
    launch resumes it.
11. Resume from `state.json` and `champion.json`.

Each test is shown red by a named mutation of the code it covers.

## 12. Size

`loop.py` under 300 lines including docstrings.
