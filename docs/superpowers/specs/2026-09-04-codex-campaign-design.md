# Codex campaign: FAMOU-style co-evolution over our own agent's code — design

2026-09-04. Supersedes the offline rule synthesis design of 2026-09-03 (one
codex batch, hook-only mutation surface, four inert and one ruinous candidate:
not a test of the idea, and closed). Follows the strategy in
[docs/research/2026-09-04-strategy-remaining-time.md](../../research/2026-09-04-strategy-remaining-time.md)
and the method of Li et al., _Beyond Static Evaluation: Co-Evolutionary
Mechanisms for LLM-Driven Strategy Evolution in Adversarial Games_
([2606.10389](https://www.alphaxiv.org/abs/2606.10389); repo
[1xiangliu1/FAMOU-CoEvo](https://github.com/1xiangliu1/FAMOU-CoEvo)). The
paper's evolution loop is not released (it is Baidu's FM-Agent); its
evaluator, opponent-pool manager, task prompt and hyperparameters are, and
this design reproduces the loop from the paper's Algorithm 1 and mirrors the
released pieces.

## 1. Decisions already taken

Settled in discussion; not reopened here.

1. **The agent is ours.** Nothing from another competitor's agent goes into
   it. Vendored kernels are used as _opponents_ only.
2. **The engine port is adopted, the policy is not.** `sim.hpp` and
   `pyrandom.hpp` from the yhay81 kernel (Apache-2.0, ~900 lines, pinned to
   the `kaggle-environments` 1.32.7 source SHA) are copied with their license
   headers and a NOTICE; `policy.cpp`, `tape.inc` and
   `six_day_budget_guard.hpp` are not. A Rust port is optional later.
3. **The method is FAMOU**, the established one: an archive of whole agent
   files; LLM mutation (`full` and `cross`); UCB parent selection; island
   model; a co-evolving opponent pool with weakness pressure; hierarchical
   evaluation (fast for ranking, deep for champions); a held-out opponent set
   for generalisation. No persistent agents, no shared board: collaboration
   is crossover and migration, memory is the archive and the feedback in the
   prompt.
4. **Every mutation is a `codex exec` call.** Codex is the mutation operator.
   It gets the task prompt, the parent (and an inspiration for `cross`), the
   parent's per-opponent feedback, and a sandbox where it may smoke-test its
   output; it returns a complete file.
5. **The rules are in every call.** The task prompt — rules, objective,
   interface, verified insights — is `AGENTS.md` in the mutation sandbox,
   which codex reads at the start of every session. Phase 1 writes it.
6. **The exam gate is the deep evaluation.** Seeds 700000–700063 remain
   sealed from every fast evaluation and every mutation sandbox.
7. **Everything unrelated is removed first** (§9).

## 2. The loop (Algorithm 1, adapted)

```
archive: I islands × P programs, each with (source, fitness mean, n_evals, parent ids, lineage)
pool.json: training opponents {name -> path (hidden), weight}; held-out opponents never in the pool

repeat for iteration t:
    for each island (in parallel, bounded by the codex concurrency budget):
        parent      <- UCB over the island's programs
        kind        <- `full` (p=0.7) or `cross` (p=0.3, inspiration = a second UCB draw)
        child       <- codex exec(task prompt, parent, [inspiration], parent feedback)
        if child fails validation: record, continue                # syntax, contract, imports, copy check, load, latency
        fitness     <- fast eval(child)                            # pool-weighted win rate, fresh non-exam seeds
        insert child into the island (replace the worst if full); record everything
    if t % migration_interval == 0: ring-migrate the top M of each island
    if t % epoch_interval == 0:
        deep eval the archive's top K on the EXAM block                     # the gate
        if best deep score's lower bound > champion's point estimate:
            champion <- best; pool += champion (weight 0.20, renormalise; retire if full)
            weakness pressure: double the weight of the champion's weakest opponent (cap 0.5)
            report held-out score; write the floor; commit
    if t % reset_interval == 0: reseed the worst island from the champion
```

Fitness is **win rate only**, pool-weighted. FAMOU's 30% margin term is
dropped deliberately: this game's objective is win/loss and the project
record shows margin and win rate opposed (a route that banked +9,593 lost
0.24 of field win rate).

## 3. Hyperparameters

| parameter         | FAMOU                                                      | ours                                                                                             | why                                                                             |
| ----------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| islands × size    | 4 × 50                                                     | 4 × 12                                                                                           | codex sessions cost minutes, not seconds; the archive must be fillable in a day |
| migration         | every 15 iterations, 2 migrants, ring                      | every 10, 2, ring                                                                                | same shape, shorter horizon                                                     |
| reset interval    | 60                                                         | 40                                                                                               |                                                                                 |
| parent selection  | UCB, c tuned                                               | UCB, c = 0.5 on win-rate scale                                                                   |                                                                                 |
| mutation types    | `full`, `cross`                                            | `full` 0.7, `cross` 0.3                                                                          |                                                                                 |
| LLM               | Gemini-2.5-Flash / DeepSeek-V4-Flash, T = 0.8              | `codex exec`, gpt-5.6-sol, reasoning high                                                        | the paper found the stronger coder wins; codex is what we have                  |
| fast eval         | 3 games/opponent                                           | 8 games/opponent (4 fresh non-exam seeds × both seats)                                           | the C++ engine makes it free; still ranking-only                                |
| deep eval         | 20 games/opponent, top-3, every 50 iterations              | 128 games/opponent (64 exam seeds × both seats), top-3, every 25 iterations                      | the existing gate; ρ(fast, deep) is measured and reported each epoch            |
| pool              | 5 training, cap 10, retire at ≥ 0.95, champion weight 0.20 | same                                                                                             | mirrors `opponent_pool.py`                                                      |
| weakness pressure | ×2, cap 0.5                                                | same                                                                                             |                                                                                 |
| held-out          | 5 unseen opponents                                         | `salemali7_2900`, `lynnsakurai_v5`, plus each new kernel-watch find until promoted into the pool | the honest generalisation number                                                |
| budget            | 400 iterations, ~30k LLM calls                             | ~200 codex calls/day → ~5,000 over the run                                                       | quota-bound; logged per call                                                    |
| seed              | a 217-line heuristic already beating "hard"                | our skeleton, ~0 against the field                                                               | the paper's own finding: evaluator design mattered more than seed quality       |

One iteration = one mutation per island (4 codex calls). Concurrency: up to
8 codex sessions at once across islands; the fast evaluator and the deep
gate share the core budget in `config.py`.

## 4. Layout

```
src/kaggriculture/
  constants.py observation.py actions.py report.py replay.py   kept plumbing
  campaign/
    config.py       paths; EXAM_SEEDS 700000-700063; hyperparameters above; core budget
    engine/         sim.hpp pyrandom.hpp NOTICE bridge.cpp build.py wrapper.py
    arena.py        reference-engine games, both seats (from search/arena.py, stripped)
    harness.py      `campaign play | check | package` — opponents by name, exam seeds refused
    prompt.py       builds AGENTS.md + PROMPT.md for one mutation
    mutate.py       one codex exec call in a sandbox -> child source, or a recorded failure
    validate.py     syntax, contract, imports, copy check, Kaggle-shaped load, latency
    archive.py      islands, UCB, insert/replace, migration, reset; persisted as JSONL
    evaluator.py    fast eval (fresh seeds, pool weights) and deep eval (exam block)
    pool.py         port of FAMOU's OpponentPool: add_champion, retire, weakness pressure
    gate.py         deep eval + promotion rule + floor write + commit
    kernel_watch.py ladder scan -> held-out set, then pool, by name
    loop.py         Algorithm 1
  served/           the floor: main.py, NOTICE (the .so is rebuilt from engine/)
run/campaign/                     runtime state, gitignored
  archive.jsonl                   every program ever evaluated: source path, fitness, n, parents, kind
  pool.json                       as FAMOU's, paths hidden from sandboxes
  epochs.jsonl                    one line per epoch: deep scores, held-out scores, rho(fast, deep), pool change
  programs/<id>.py                sources
  sandboxes/<id>/                 one per mutation: AGENTS.md, parent.py, [inspiration.py], feedback.md, engine/, child.py, codex JSONL
  floor/agent/                    champion, 444
/data/kaggriculture/opponents/    vendored lineages; never inside a sandbox
/data/kaggriculture/episodes/     36 daily replay archives; differential-test corpus
```

## 5. Components

### 5.1 Engine and bridge

`engine/bridge.cpp` exposes `extern "C"`: `kag_new(seed)`, `kag_step(handle,
actions_json_0, actions_json_1)`, `kag_observe(handle, seat)`, `kag_bank`,
`kag_free`. Loaded by `ctypes.CDLL` by absolute path under the unique name
`kaggriculture_engine.so`. **Acceptance:** 200 recorded tapes from the 1.32.7
archives plus 100 random-seed games replayed through the reference engine
and the port with zero disagreement in per-step bank and observation. **Kaggle
load test, once per `.so` change:** the packaged floor unpacked into
`gcr.io/kaggle-gpu-images/python:latest`, loaded as the runner does, one
validation episode. `kag_from_observation` is built only if an evolved agent
needs runtime lookahead; whether hidden state and RNG make that worthwhile is
a phase-1 finding.

### 5.2 The mutation call

`mutate.py` prepares `sandboxes/<id>/`:

- `AGENTS.md` — the task prompt (§5.3). Static across the run except when
  phase 1 revises it.
- `parent.py`; for `cross`, `inspiration.py`.
- `feedback.md` — the parent's fast-eval rates per opponent (names only),
  the pool's current weights, the weakest opponent, and the last three
  validation failures in this lineage (why they failed).
- `engine/kaggriculture.py` — reference source, read-only.
- The harness, callable as `campaign check child.py` and `campaign play
child.py --vs NAME --seeds` (non-exam only, at most 16 games), so codex can
  smoke-test; it is told the evaluator does the real measuring.

Then runs

```
codex exec -C <sandbox> -s workspace-write -c approval_policy=never -m gpt-5.6-sol --json - < PROMPT.md
```

with a 10-minute cap. `PROMPT.md` is FAMOU's mutation instruction: produce a
complete `child.py` implementing the interface; keep what works; change what
the feedback says is losing; for `cross`, combine the two programs' ideas.
Output is `child.py` or a recorded failure (`no_output`, `timeout`,
`exec_error`). Tokens are parsed from the JSONL and logged per call.

### 5.3 The task prompt (phase 1's product)

Mirrors `config/task_prompt.txt`: objective (win = more coins banked at turn
720; the ladder scores win/loss only, never margin), the rules as the engine
enforces them, the interface (`agent(observation, configuration)`, last
callable, 1 s per step, tarball layout), the observation layout including the
opponent's board in `farms[1]`, the verified economy (crop and animal
returns, cycle times, price curve with the 1.32.7 hinge, shop draws, market
impact), and the doctrine: measure opponents through the harness; never read
their source; the gate rejects copied code. Every number in it carries the
script that produced it, under `docs/research/campaign/experiments/`.

### 5.4 Validation (before any game)

Syntax; exactly the interface; import allowlist; copy check — token-shingle
Jaccard (k = 8) against every file under `/data/kaggriculture/opponents/` and
the router's `policy.cpp`, threshold calibrated on the skeleton and the
former hybrid agent; Kaggle-shaped load (clean unpack, last callable, empty
globals); one validation episode vs itself on the reference engine, worst
step ≤ 0.5 s. Each rejection is a recorded reason in the archive and in the
lineage's feedback.

### 5.5 Evaluators

- **Fast:** 4 fresh non-exam seeds × both seats × every pool opponent,
  weighted by pool weights → fitness; stored as a running mean with count
  (UCB needs both). Seeds rotate per evaluation so a program cannot fit a
  block.
- **Deep (the gate):** 64 exam seeds × both seats × every pool opponent;
  per-opponent rates with Wilson intervals; pool-weighted score; plus
  `field` (equal-weight over the vendored lineages, for continuity with the
  project record) and the **held-out** score, reported and never used for
  selection. Promotion when the deep score's lower Wilson bound exceeds the
  current champion's point estimate on the same pool, and no pool opponent's
  rate fell below the champion's by more than the wider of the two
  intervals.

### 5.6 Pool

A port of the released `opponent_pool.py`: `add_champion` (weight 0.20,
renormalise, retire the lowest-weight opponent the champion beats ≥ 0.95
when the pool is full at 10), `apply_weakness_pressure` (×2, cap 0.5),
history. Initial pool: `router_v1`, `router2929`, `v54`, `v56`, `shopforge`,
`indarkarhana`, equal weights. Held-out: `salemali7_2900`, `lynnsakurai_v5`.
`kernel_watch` adds a new ladder kernel to the held-out set first; it enters
the pool only by a decision recorded in `epochs.jsonl`.

### 5.7 Shipping

Manual. Two scored slots, two lineages (ours and the vendored hedge, per the
finale memory). A ship replaces the weaker slot when the champion's deep
score exceeds what that slot holds by more than its interval; never on a
live-rating move.

## 6. Phases

| phase                                   | what                                                                                                                                                                                                                                    | done when                                                   | kill                                                                                                                                                                                                            |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **0. Cut and scaffold** (us, 2–3 days)  | §9 removal; engine bridge + differential + Kaggle load tests; harness; validate; archive; evaluators; pool; gate; loop; a dry run with a fake mutator (copies the parent, changes one constant) through the whole loop on a 4-seed pool | dry run promotes a champion end to end; all tests pass      | —                                                                                                                                                                                                               |
| **1. Task prompt** (codex, in parallel) | 2–3 long `codex exec` sessions against the reference engine: "explain how the game works; verify every number with a script; write the task prompt"                                                                                     | `AGENTS.md` drafted; five spot-checked numbers hold         | —                                                                                                                                                                                                               |
| **2. Evolve**                           | the loop, 8 concurrent codex calls, epoch every 25 iterations                                                                                                                                                                           | continuous; ρ(fast, deep) and held-out reported every epoch | day 10 with champion `field` < 0.3: stop, ship the vendored hedge pair. Day 20 with `field` < 0.9115: finale pair is champion + hedge only if champion beats the hedge head-to-head, else two vendored lineages |
| **3. Freeze** (09-27 → 09-29)           | last wave's kernel into the pool with weakness pressure on it; 24 h of evolution; ship the final pair 09-29; nothing 09-30                                                                                                              | —                                                           | —                                                                                                                                                                                                               |

## 7. Testing

- **Engine:** the differential test of §5.1 — first written, first run.
- **Archive:** UCB picks the under-sampled program when means tie; insert
  replaces the worst; migration moves exactly M; reset reseeds — each on a
  hand-built archive, each mutated once to confirm it goes red.
- **Pool:** add/retire/weakness pressure reproduce the released
  `opponent_pool.py` numerically on its own example.
- **Evaluators and gate:** Wilson, weighted score, promotion rule on
  hand-built results.
- **Validate:** verbatim opponent file → `copy` at ~1.0; skeleton ~0; exam
  seeds and opponent paths refused by the harness; worker cap.
- **Loop:** dry run with the fake mutator on a 4-seed pool, end to end.
- **Prompt:** golden test that a sandbox contains every required file and no
  opponent path.

## 8. Error handling

- A mutation that fails validation or the codex call is an archive line
  with a reason; the loop continues. A lineage with three consecutive
  failures gets them in its feedback.
- A quota error pauses mutation with the provider's reset time logged;
  evaluation and epochs continue on the archive as it stands.
- A deep-eval failure not caused by the candidate halts the gate loudly;
  the gate never records a verdict it did not compute.
- The floor is written only by the gate on promotion.

## 9. Removal (phase 0, first task)

On a new branch `campaign` from the current HEAD; explicit `git rm` paths,
never `add -A`; `kaito_v54`/`v56` copied to `/data/kaggriculture/opponents/`
before their `src/` copies go.

**Keep:** `constants.py`, `observation.py`, `actions.py`, `report.py`,
`replay.py`; `search/arena.py` → `campaign/arena.py`;
`search/scripts/holdout.py` → the constant in `campaign/config.py`;
`scripts/field_gate.py` → folded into `campaign/evaluator.py` and `gate.py`;
`scripts/kernel_watch.py` → `campaign/kernel_watch.py`; `scripts/package.py`
and `scripts/submit.py` without the `routes.STORE` requirement; `docs/`;
tests for the kept modules.

**Remove:** `learn/`, `market_residual/`, `hybrid/`, `routes/`, `sim/`,
`search/` (all but the two above), `hooks/`, the staged `rules/` deletion;
`scripts/autonomous|budget|market_*|meta|tracking|rule_search|run`;
`features.py`, `action_codec.py`, `harness.py`, `task.py`, `result.py`,
`config.py`, `agent.py`, `policy.py`; every vendored policy in `src/`; torch,
lightning, wandb, optuna and dependants from `pyproject`; their tests.
`README.md` rewritten to describe what remains.

## 10. Open questions, settled by measurement

1. ρ(fast, deep) on our pool — FAMOU measured 0.11 at 3 games; at 8 games on
   a deterministic engine ours may be usable for more than coarse ranking.
   Reported every epoch; if it stays low, fast eval grows to 16 games.
2. What of the state is reconstructible from an observation, and is the RNG
   recoverable? Phase 1.
3. The copy-check threshold (§5.4 calibration).
4. The host's answer on team score = max of two
   ([739410](https://www.kaggle.com/competitions/kaggriculture/discussion/739410)).
