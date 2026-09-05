# Strategy for the remaining 26 days

**2026-09-04.** Sources: the Kaggle API (leaderboard CSV of 7,629 teams pulled
today, our submission log, the competition page), the discussion and kernel
caches refreshed today via the nvidia-kaggle skill, six papers read through
alphaXiv, and this repository's own record. Every number is dated; the
leaderboard ones expire in days.

The commissioning question was "can this be salvaged, and how", with the
standing objection that **a rule-based approach will not medal**. This
document takes that objection seriously, says where the evidence agrees and
where it does not, and ranks what to do.

---

## 1. Where we stand today

| fact                           | value                                                                                                                         | source                                       |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| Live rank                      | **576 / 7,629**, score 2,151.1                                                                                                | leaderboard CSV 2026-09-04 17:16             |
| Medal cutoffs on today's board | gold ≥ **2,786** (top 25), silver ≥ **2,314** (top 381), bronze ≥ **1,991** (top 763)                                         | same CSV, Kaggle's 10+0.2% / 5% / 10% rule   |
| Scored pair (latest 2)         | `counter_43_38` 2,151 and `squeeze_v58` 2,079 — **both v58-lineage**                                                          | `kaggle competitions submissions`            |
| What we displaced to get there | `yhay81` shop router rebuilt from its notebook, **2,386.8** — would rank **300**, inside silver                               | submission 55980036, 2026-09-03 11:14        |
| Top of board                   | 3,029 (#1), 2,915 (#5), 2,636 (#100)                                                                                          | same CSV                                     |
| Deadline / finale              | 2026-09-30 23:59; games continue ~2 weeks; **Bradley-Terry over post-deadline episodes** decides                              | competition page                             |
| Whether team = max of its two  | **unanswered** — asked of the host 2026-09-04 ([739410](https://www.kaggle.com/competitions/kaggriculture/discussion/739410)) | discussion cache                             |
| Gate cost                      | ~17 s per reference-engine game; a 768-game field gate is **4–11 min** at 60–20 workers (64 cores)                            | timed today, `arena.outcomes` on seed 700000 |
| LLM proposers on the box       | `codex` 0.147 (gpt-5.6-sol, reasoning high), `claude` CLI                                                                     | `~/.codex/config.toml`                       |

Two things in that table are self-inflicted. We are in bronze, not silver,
because on 2026-09-03 we submitted two v58 variants after the router and
pushed the router out of the scored pair. And the pair we hold is the one
configuration the record says is worst: two agents of one lineage, so the
finale's "max of two" (if that is the rule) hedges nothing.

## 2. The objection: "a rule-based approach will not medal"

Split it into the two claims it contains, because the evidence answers them
differently.

### 2a. A _static_ rule set will not medal — agreed, and now measured three ways

- **A frozen plan loses 0.42 of win rate to an opponent that re-plans**, with
  seed and strength held fixed by construction
  ([generative-prior report §5](2026-09-03-generative-prior-over-plans.md)).
  That is the single largest effect measured on this project.
- **The field is kernel waves that rotate in 2–3 days.** A kernel sweeps to
  50–80% of top-band seats within 48 h and collapses within days
  ([what-the-top-build §1](2026-09-02-what-the-top-build.md)); sobameshi's
  independent fingerprinting over 19 daily dumps sees the same and adds that
  among 14 public implementations the ladder is _transitive_ — 86 of 91
  chronological pairs, newer beats older, "a stronger economy is simply a
  stronger economy"
  ([739273](https://www.kaggle.com/competitions/kaggriculture/discussion/739273)).
  A rule set frozen on 09-30 is optimised against a field that will have
  turned over by 10-03.
- **The #1 on 09-01 was adaptive**: destbreso's replay x-ray found "every turn
  open to change, first divergence at t=0, all three channels moving", 40-0
  with median margin +13,065, still climbing — then withdrawn. The router that
  replaced it at #1 grinds 23-17 inside its own family
  ([739179](https://www.kaggle.com/competitions/kaggriculture/discussion/739179)).
  Michael Timbs, who fingerprints every top-100 episode: "nearly everyone is
  still playing static policies … those that do react on opponent do so in
  very primitive ways"
  ([737955](https://www.kaggle.com/competitions/kaggriculture/discussion/737955)).

So the medal-shaped agent is one that **conditions on the opponent's board and
the shop draw at runtime**. The record and the field agree.

### 2b. Therefore train a model — the evidence does not support this route in 26 days

Eight directions are measured closed in this repository — offline market
residual 128/128 losses, on-policy V-trace negative in every block over 19,200
paired seasons, route harvesting 0.000 with a working control, behavioural
cloning 0 of 48 with one DAgger round, a 15M-parameter generative prior that
scores held-out perplexity 2.19 and banks 269 against a corpus median of
92,912 ([literature review, closing section](2026-09-02-literature-competitive-sim-agents.md)).
The public record says the same: PPO with the full action space at ~10k
steps/s plateaued at 80k terminal cash with saturated logits
([738619](https://www.kaggle.com/competitions/kaggriculture/discussion/738619));
a BC model at ~100% offline accuracy chose PASS for a whole game once teacher
forcing was removed
([738079](https://www.kaggle.com/competitions/kaggriculture/discussion/738079)).
The diagnosis is not "we did it wrong": the clone failure is non-realisability
(train accuracy = holdout accuracy), which more DAgger cannot fix, and the
generative prior's failure is an absorbing-state property of the game
(movement is legal almost everywhere, productive verbs almost nowhere).

The literature that _does_ show learned policies winning simulation
competitions (GRF 2020, Lux S1, Lux S3) had a fast simulator **and** months.
We have the simulator — exact to the coin, batched, 148 ms per step at batch
256 — and 26 days.

### 2c. The 2026 literature dissolves the dichotomy

The question is not rules-versus-model. It is **what produces the adaptive
policy**: gradient descent, a human, or search. Five papers, all read in full
today, on exactly this:

| paper                                                                                               | what it did                                                                                                                                                                                                                                        | the number that matters here                                                                                                                                                |
| --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Co-evolutionary LLM strategy evolution (FAMOU), Jun 2026](https://www.alphaxiv.org/abs/2606.10389) | "The reinforcement learning policies we trained failed to outperform handwritten heuristic strategies", so they evolved the **whole 500–1700-line strategy file** with LLM mutations; evaluator co-evolution, weakness pressure, hierarchical eval | Won the AAMAS 2026 MCTF hardware round-robin. 3-game screens had **Spearman ρ = 0.11** with 20-game deep eval — the same screen-does-not-transfer failure we hit four times |
| [Code-Space Response Oracles (DeepMind), Mar 2026](https://www.alphaxiv.org/abs/2603.10098)         | Replaces PSRO's RL best-response oracle with an LLM writing code; opponent code/summaries in the prompt; AlphaEvolve as the inner refinement                                                                                                       | Aggregate score **+122 vs −532** for PSRO-IMPALA in repeated RPS; "opponent conditioning is the most important component of the prompt"                                     |
| [Digital Red Queen (Sakana), Jan 2026](https://www.alphaxiv.org/abs/2601.03335)                     | Static optimisation vs one fixed opponent produces specialists; each round evolving against _all previous champions_ produces generalists                                                                                                          | Specialists collectively beat 89% of human warriors but **each beats only 28%**. That is what `counter_43_38` (1.000 vs five lineages) is                                   |
| [Janus, Aug 2026](https://www.alphaxiv.org/abs/2608.08189)                                          | When real evaluation is expensive, LLM-written proxy evaluators co-evolve with the programs; proxies only prioritise, real validation gates                                                                                                        | 59% fewer real evaluations for 99% of the gain. Our gate is cheap enough (4–11 min) that this is a later optimisation, not a prerequisite                                   |
| [MEMENTO, Jul 2026](https://www.alphaxiv.org/abs/2607.22832)                                        | Single-elite memetic search: sequential hill-climb with a **memory of rejected proposals**, macro-mutation, crossover; evaluator evolved first                                                                                                     | Zero-shot generation fails outright; Eureka-style independent sampling never solves the task; MEMENTO 0.97                                                                  |

The common finding: an LLM proposing **structurally different code** against
an **exact, adversarially refreshed evaluator** produced the winning agent in
the one 2026 adversarial competition where it was compared head-to-head with
RL. That is a searched policy in code space. It is not hand-written rules,
and it is not gradient descent.

**What the last session ran was not a test of this.** One `codex` batch, five
candidates, mutation surface = a `hook(observation, action)` over a frozen
base — four inert, one ruinous (FIELD 0.486 vs base 0.664)
(`run/rule-search/results.jsonl`). FAMOU needed 400 iterations, whole-file
mutations, four structurally different seeds and a co-evolving opponent pool
before it beat its own seeds. The hook design has the failure DRQ and MEMENTO
both name: a tiny mutation surface around one elite, no memory, no diversity.

## 3. What actually decides the finale

The Bradley-Terry fit runs over games played **after** the deadline, among
agents active at the deadline. So:

1. **The finale field is the last kernel wave plus the few adaptive runtime
   agents.** Whatever kernel publishes around 09-25 will be on half the seats.
   It cannot be countered before it exists. The decisive capability is
   therefore a **counter loop that turns a new kernel into a gated counter in
   24–48 h**, not any one agent built now. Most of that loop exists:
   `kernel_watch` (386 kernels scanned), `field_gate` (six lineages, exam
   seeds), `arena` (reference engine, both seats), `hooks.synthesize`
   (`codex exec` with validation).
2. **A specialist counter is obsolete on arrival** (DRQ). The evolved agent
   must beat every previous champion _and_ the vendored field, or it will lose
   to the wave after next.
3. **The two slots must be two lineages.** Whether the team score is the max
   or something else, two copies of one lineage tie each other and hedge
   nothing ([hedge memory, measured 09-02](../../.claude/) — v58 beats
   shopforge 0.69 head-to-head while shopforge beats the rest harder; BT cannot
   express both).
4. **Denial is under-explored and may be the strongest lever.** Today's timing
   run: `tuned_v58` vs `router_v1`, seed 700000, both seats — **v58 banks 0**.
   `field_gate.py` records the same for `router_v2` vs `router_v1`. In a
   relative-bank game against a 59%-static field, an agent that starves a
   fixed script of what its route assumes (shop stock, book depth) wins at any
   absolute bank. `counter_43_38`'s day-0 wheat pre-buy is a first instance of
   this. Nobody has yet established the mechanism; it is also a gate poison
   until understood.

## 4. The plan, ranked, with kill gates

### Track 0 — today, costs nothing: repair the scored pair

Resubmit the router (`/data/kaggriculture/agents/yhay81_router_v1` or the
2929 revision, whichever gates higher against the other) into one slot. It
displaces `squeeze_v58` (the weaker of our two) and leaves
{`counter_43_38`, router}: two lineages, and by today's board a silver-band
agent back in the pair. Then **no further submissions** until something gates
above both — every slot spent is a slot that pushes one of these out.

### Track 1 — days 1–14: build the evolution loop the papers describe

Concretely, on top of what exists:

- **Mutation surface = a whole agent file.** Seeds are the four vendored
  lineages we already hold as opponents (v58-tuned, router, shopforge,
  indarkarhana) — public code, so vendoring is allowed
  ([738837](https://www.kaggle.com/competitions/kaggriculture/discussion/738837)).
  The proposer gets the seed's source, the opponent sources or summaries
  (CSRO's finding), the per-lineage gate breakdown, and the rejected-proposal
  memory (MEMENTO). It writes Python against the runtime object
  ([memory: LLM writes code, not DSL](../../.claude/)).
- **Evaluator co-evolution.** Every champion joins `FIELD` as a gatekeeper.
  Weakness pressure: double the weight of the worst lineage (for v58 that is
  `router_v1` at 0.000). Refresh the pool from `kernel_watch` daily — the
  rotation timescale is days.
- **Hierarchical evaluation, honestly.** Keep the 3-seed screen only to reject
  crashes and inert hooks; it ranks nothing (ρ = 0.11 in FAMOU, four transfer
  failures here). Deep eval is the 64-exam-seed, both-seat, all-lineage gate.
  At 4–11 min per gate that is **100–300 deep evaluations a day**, more than
  FAMOU's whole 400-iteration budget in two days.
- **Diversity by build fingerprint.** We already compute the 120-number
  capital fingerprint per seat; use it as the MAP-Elites descriptor (DRQ) so
  the archive holds structurally different farms, which is the top decile's
  defining property (corr(variety, rating) = +0.667).
- **The adaptive dimension is mandatory in the prompt.** `farms[1]` carries
  the opponent's full board at call time. Proposals that ignore it are
  proposals for a static script, which §2a already rules out.

**Kill gates.** Day 4: a champion must exceed the shipped 0.9115 FIELD _and_
score > 0.5 vs `router_v1`. Day 10: a champion must beat every prior champion
and every vendored lineage on the exam block. Miss either and the loop is
closed; Track 0's pair stands.

### Track 2 — days 1–2, in parallel: the denial mechanism

One day of replay reading: why does `router_v1` zero `v58`, and `router_v2`
zero `router_v1`? Find the turn the loser's bank stops moving and what the
winner did to the shop or book just before. If it is a general mechanism
(starve a fixed route of an input it assumes), it goes into the proposer's
prompt as a named lever and into the gate as a first-class opponent. If it is
an engine defect, report it to the host and exclude the pairing.

### Track 3 — optional, only if the GPU is idle: the one learned formulation with a path

A **learned plan selector** over the vendored portfolio: at a fixed early
turn, a small classifier over public state (shop draw, opponent board) picks
which of K plans to run. Labels come from the exact simulator — play every
plan against every lineage on every seed and record which won. This sidesteps
every measured failure: labels are plan ids (realisable by construction), one
decision instead of 719 (no horizon), no action history (no copycat).

The prerequisite is a portfolio whose members win on _different_ boards. Our
own four agents share one backbone and a perfect oracle gained zero; the
router/shopforge/v58 triple is non-transitive, so the oracle gain is nonzero
in principle. **Day-1 test:** compute the perfect-oracle FIELD over
{v58, router, shopforge}. Under +0.03 over the best single member, kill it.

### Final week — freeze protocol

- 09-23 onwards: `kernel_watch` daily, fingerprint census on each archive.
- 09-27: vendor whatever kernel leads the newest wave; run Track 1's loop
  against it for 24 h with it as the top-weighted gatekeeper.
- 09-29: submit the final pair — one evolved champion, one best vendored or
  counter — from **two lineages**. Confirm both validation episodes pass.
- 09-30: nothing. The BT finale discards the live rating; a last-day re-roll
  buys nothing and risks a validation error in a scored slot.

## 5. Honest odds

- **Bronze** (≥ ~1,991 today): Track 0 alone should hold it; the router at
  2,387 sat at rank 300.
- **Silver** (≥ ~2,314): Track 0 gets a silver-band agent into the pair
  today; keeping it there through the finale needs the counter loop to work
  once, in the last week, against the final wave.
- **Gold** (≥ ~2,786, top 25): requires beating the adaptive runtime agents
  at the top, not the kernel clones. Only Track 1 has any route there, and it
  is the route the 2026 competition literature took to a first place. It is
  not a rule set. It is search, and the searched object happens to be code.

## 6. Housekeeping before any of this

- The `rule-synthesis` branch has the `rules/` → `hooks/` rename staged but
  uncommitted; `tests/hooks` passes (32 tests). Commit it as the loop's
  starting point or discard it; do not leave it half-staged.
- Add `router_v1` and the champion pool to the gate's default `FIELD` with
  the weakness weighting described above; the current equal-weight mean is
  the quantity the gate's own docstring says has saturated.
- Watch [739410](https://www.kaggle.com/competitions/kaggriculture/discussion/739410)
  for the host's answer on team score = max of two; it decides whether the
  second slot is a free hedge.

## Sources gathered this run

Discussions (cache `data/discussions.db`, ingested 2026-09-04): 739534,
739410, 739273, 739179, 738619, 738079, 737955, 736219, 734000. Kernels
(cache `data/kernels.db`): `yhay81/three-day-shop-router` (118 votes, last
run 09-03), `tetsutani/shape-the-shop-work-the-pasture` (105), `salemali7/
kaggriculture-2900` (76), `destbreso/x-ray-your-agent` (19),
`georgymamarin/kaggriculture-daily-replays-the-live-meta-report` (37),
`motemen/kaggriculture-meta-census-lineage-occupancy`. Votes measure
attention, not rank; only the router's score was verified by our own
submission. Papers: 2606.10389, 2603.10098, 2601.03335, 2608.08189,
2607.22832, 2604.04328 (Soft Tournament Equilibrium — read, not used: it
argues BT misranks cyclic fields, which is true and not actionable since the
host chose BT).
