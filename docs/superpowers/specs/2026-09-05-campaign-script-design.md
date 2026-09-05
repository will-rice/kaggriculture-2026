# The campaign

Date: 2026-09-05, revised through the day. The simplest system that can
work.

The loop owns everything except writing code. It takes the best program it
has, asks a model to improve it, plays the result against every opponent,
and keeps what beats them all. The model is a mutation operator: it does
not measure anything, own anything or run anything.

This is AlphaEvolve's controller loop (2506.13131 section 2.6) with FAMOU's
evaluator (2606.10389 sections 4.4-4.6). Where we differ from them is
listed in section 9, and each difference has a reason.

## 1. The script

`loop.py`, `main()` at the top, four steps:

1. **Initialize wandb from `{model}-{hash}`.**
2. **Start the event loop.**
3. **Spin off `SESSIONS = 8` workers**, each running section 3 forever.
4. **Gate the improvements** (section 5), fired by what the workers produce.

## 2. Definitions

**Program.** One Python file whose last top-level callable is
`agent(observation, configuration)`. That is what Kaggle loads, so it is
what we evolve and what we ship. Nothing else: no package, no engine, no
markers. The first campaign's champion reached that shape on its own — its
only import is `math`.

**Game.** 720 turns on our C++ engine port, one program in each seat, won
by relative bank. The port is bit-identical to `kaggle_environments` 1.32.7
(section 7) and fast enough that games are free.

**Pool.** The opponents a program is scored against: six vendored public
kernels, plus every champion, all counting equally. There are no weights.

**Fitness.** The mean win rate over the pool, both seats per seed, ties
half. It ranks the database and nothing else.

**Champion.** The best program the campaign has confirmed: the one program
that beats every pool opponent on the sealed block. It is what we would
submit, what every session starts from, and a member of the pool.

**Database.** Every program ever kept, with its scores, shared by all eight
workers.

## 3. A worker

```
while True:
    program = champion                      # or a top-ten draw when stagnant
    for round in range(ROUNDS_PER_SESSION):
        prompt  = compose(rules, program, verdict, day states, instruction)
        child   = await codex(prompt)       # writes one file, then stops
        if not validate(child): record why; break
        result  = score(child, pool)        # the loop plays; the model never does
        program = database.add(child, result)
        if program is in the top DEEP_TOP_K and has no deep result:
            create_task(gate(program))      # section 5
        if result beats every opponent: break
```

That is AlphaEvolve's loop — `database.sample`, `prompt.build`,
`llm.generate`, `evaluator.execute`, `database.add` — and FAMOU's "feeds
the resulting performance summary back into the next mutation prompt".

Eight workers run it concurrently against one database, one pool and one
champion. A promotion by any worker changes what the other seven start from
next. There are no islands and no worker identity in any record: the
concurrency is the only structure.

**Rounds go deeper, sessions go wider.** A round continues from its own
previous program; a new session starts again from the champion.

## 4. What the model is given

A prompt on standard input, and a temporary directory holding one file,
`child.py`, which it edits in place. Nothing else — no engine, no harness,
no workspace to manage.

The prompt is:

| part            | content                                                                                                                                                                           |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| the game        | objective, rules, verified economics, the interface, and the doctrine: never read, request or reconstruct an opponent's program                                                   |
| the program     | `child.py` is the program to improve; edit it and stop                                                                                                                            |
| the verdict     | its win rate against each opponent, the bank margins, and the opponents it does not beat                                                                                          |
| the states      | one lost game against the opponent it does worst against, day by day: both banks, both sheds, both hand counts, market prices                                                     |
| the instruction | one of five, drawn uniformly (FAMOU appendix C.2): improve it; a completely different algorithm; a novel approach inspired by it; restructure its components; tune constants only |

**The bar** is beating every opponent, stated in the prompt, and it is the
same condition the gate applies.

**Sizes, measured on a 719-step game.** Every turn with both observations
and both actions is 4.2 MB; every day with both full observations is
171 KB; every day with decision fields only is 17.6 KB. Only the last is
sendable, which is why the states are a day table and not a transcript.

## 5. The gate

FAMOU Algorithm 1 lines 7-10: when a new best appears, confirm it deeply
and add it to the opponent pool. The trigger is an event, not a schedule.

### 5.1 Two evaluations

AlphaEvolve's cascade: cheap first, and only what survives goes on.

- **Fast** — `FAST_SEEDS = 4` fresh seeds, never the exam block, every pool
  opponent, both seats. About a minute. This is what a round is scored on
  and what enters the database. FAMOU's equivalent is 3 games per opponent
  "for coarse population ranking".
- **Deep** — the sealed `EXAM_SEEDS = 64`, every pool opponent and every
  held-out opponent, both seats. Returns per-opponent rates with 95% Wilson
  intervals, the equal-weight rate over the vendored opponents (`field`),
  and the held-out rates. FAMOU: "final decisions use deep evaluation".

A program entering the top `DEEP_TOP_K = 3` by fitness is deep-scored once,
ever, bounded by `DEEP_CONCURRENCY = 2` at a time.

**A program never plays itself.** The evaluator drops the pool entry
matching the program being scored, so every metric is over real opponents.

### 5.2 Promotion

**A candidate is promoted when it wins the majority of the sealed games
against every pool opponent.** That is the rule, entire: one clause, no
weights, no comparison with the champion, no tolerance. Ties at exactly
half do not pass. Held-out opponents are measured and logged but are not
part of the gate.

Why absolute rather than relative: the champion joins the pool as a
gatekeeper and is the program we would submit. Promoting something that
loses to four of six kernels adds a weak gatekeeper and ships a weak agent.
The only question the gate must answer is whether a candidate is strong
enough to be both.

The failure message names every opponent the candidate failed to beat and
its rate against each. That message is the campaign's main diagnostic for
why nothing is promoting.

On a pass: package the program into `champions/<name>.tar.gz`, the file a
cut uploads; write the champion file, the floor and `champion.json`; add it
to the pool; log the tarball as a wandb artifact. **A cut is one command**:
upload that tarball. Nothing is built at cut time, and nothing is ever
committed to git — every write is under `run/campaign`.

### 5.3 The pool co-evolves

FAMOU section 4.4: the new champion "becomes a high-weight gatekeeper, so
later candidates must outperform both original opponents and all previous
champions". Here it simply joins, since all opponents count equally. At
`POOL_CAP = 10` the opponent the champion beats most decisively retires,
and only if it is beaten at `RETIRE_THRESHOLD = 0.95` or better.

**Held-out** opponents are scored and never trained against:
`salemali7_2900`, `lynnsakurai_v5`, and each new ladder kernel found, which
enters here first. This is the honest generalisation number.

## 6. A call

One `codex exec` process: `workspace-write`, `approval_policy=never`,
`--json`, the prompt on standard input, the transcript captured. Fresh
every round — no resumed conversation, no session ids, and a bad round
cannot poison the next. A fresh call also cannot anchor on its own earlier
hypothesis: the program embodies its prior work and the verdict says what
that work achieved.

Bounded by `ROUNDS_PER_SESSION = 5` and `SESSION_LIMIT_SECONDS = 1500`,
whichever comes first.

**Two models.** `CODEX_MODEL`, falling back once to `CODEX_FALLBACK_MODEL`
when the provider refuses the turn or codex dies. A failure on both is
logged and the worker starts its next session; it is never charged to the
program's lineage. One model at a time, recorded on every program, so a
block of quota can be judged after the fact.

We call codex rather than a completion API, unlike either paper, because
the quota available is a codex subscription. That is why one file exists on
disk at all.

## 7. Machine

Games run on a C++ port of the Kaggle environment through ctypes. Its only
justification is being identical, so that is tested: 200 archived episodes
replay bit-for-bit, and 2% of live games are audited against Kaggle's own
engine.

Evaluations fan their games over `CORE_BUDGET // SESSIONS` processes, one
game per process. Nothing else on the machine plays games: the loop is the
only player, so the core budget means what it says.

**Python 3.12**, the version Kaggle's image runs, so a candidate cannot
pass validation here and fail there.

**No latency requirement.** These are rule-based policies: microseconds a
turn. A step time measured here would not predict Kaggle's anyway, and a
program slow enough to matter is a rewrite in a compiled language, not a
rejection.

**Two time limits, both liveness, neither about speed.** A session
(`SESSION_LIMIT_SECONDS`) and a validation game (`GAME_LIMIT_SECONDS`,
because a 720-turn game takes under a second, so one still running after
two minutes is stuck).

**Validation**, before any game is played for score: it parses; `agent` is
the last top-level callable, by reading the file and by loading it through
Kaggle's loader; it imports only from an allowed list — the standard
library, nothing of ours, since nothing of ours ships; it shares no 8-token
shingle set with any opponent above 0.03 Jaccard; it plays a full game
without raising.

**State.** The database is an append-only JSON-lines log replayed on start:
one record per program (source, what it was edited from, the instruction,
the model, fitness, per-opponent rates and margins), one per deep result,
one per failure. `state.json` after every session; `champion.json` by the
gate. A restart resumes all of it and loses only what was in flight.

**Stagnation.** If `STAGNATION_SESSIONS = 40` sessions pass with no
promotion, sessions start from a program drawn from the database's top ten
instead of the champion, and the prompt says so. FAMOU's trajectories are
"punctuated equilibrium, long stagnation interrupted by sudden
breakthroughs", and the breakthrough often comes from a line that was
behind.

**Run identity.** The wandb run id and name are both
`{CODEX_MODEL}-{revision}`, `resume="allow"`. The script refuses to start
with uncommitted changes under `src/`.

## 8. Constants

| constant                     | source                               |
| ---------------------------- | ------------------------------------ |
| `SESSIONS` 8                 | fits the machine                     |
| `ROUNDS_PER_SESSION` 5       | with scoring, fits the session limit |
| `SESSION_LIMIT_SECONDS` 1500 | measured; FAMOU 300                  |
| `GAME_LIMIT_SECONDS` 120     | liveness                             |
| `FAST_SEEDS` 4               | FAMOU 3 games/opponent               |
| `EXAM_SEEDS` 64, sealed      | FAMOU 20 games/opponent              |
| `DEEP_TOP_K` 3               | FAMOU                                |
| `DEEP_CONCURRENCY` 2         |                                      |
| `POOL_CAP` 10                |                                      |
| `RETIRE_THRESHOLD` 0.95      |                                      |
| `STAGNATION_SESSIONS` 40     |                                      |
| `CORE_BUDGET` cores - 8      |                                      |

## 9. Deliberately not built

- **Islands, migration, resets, MAP-Elites, UCB parents.** Every session
  starts from the champion, so a subpopulation adds nothing the champion
  does not carry. Diversity is the five instructions, the model's sampling,
  and the stagnation branch.
- **Weights and weakness pressure** (FAMOU 4.4, 4.5). An absolute gate
  demands beating the hardest opponent outright, which is what bending
  weights toward it approximated.
- **A daily call cap.** The provider's limits are the limits.
- **A workspace for the model.** It gets a prompt and one file. The loop
  plays every game, so the measurement is ours and the cores are accounted
  for.
- **Meta-prompt evolution** (AlphaEvolve 2.2), **LLM-graded feedback**
  (2.4), **SEARCH/REPLACE diffs** (2.3), **multiple scores** (2.4).
- **A Kaggle image check at promotion.** It existed to prove a compiled
  library loaded under Kaggle's glibc; with one stdlib-only file there is
  nothing to catch, and the interpreter gap is closed by running 3.12.
- **Commands for things a person asks occasionally.** The database is
  JSONL; a query is two lines of Python at the moment someone wants it.

## 10. Telemetry

Per round, on the `calls` axis: whether it produced a child, whether it
timed out, whether it fell back, the model, tokens in and out, seconds, the
child's fitness, the database's best fitness and size. Per deep evaluation:
per-opponent rates and margins, `field`, held-out rates, the fast/deep rank
correlation, and whether it promoted. Each promotion logs its tarball.

## 11. Size

`loop.py` under 520 lines, of which under 330 are code — blank, comment and
docstring lines excluded. Two numbers, because a single budget is met by
deleting docstrings, which is the wrong trade.
