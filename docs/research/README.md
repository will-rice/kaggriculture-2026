# Research index

Primary-source research for the Kaggriculture RL work. Each document states its
own scope; this index says which question each one answers and what it is not
good for, so a later reader can pick the right one without opening all of them.

**Standing rules for anything filed here.** They exist because each was learned
the expensive way:

- **Primary sources only** — a paper or the authors' own repository. A prior
  research pass on this project cited a secondary roundup that had transposed a
  model's parameter count with its training-step count.
- **Every number carries its artifact**, or an explicit `UNVERIFIED` label.
- **Separate research-scale from single-box.** We run one workstation with a
  ~10M-parameter network. "AlphaStar did X" and "a solo competitor did X" are
  different claims and only the second one is ours.
- **Where the literature contradicts our own measurements, the measurements
  win** — and the document should say which of the paper's assumptions fails to
  hold here. That is more useful than a recommendation.
- **Date the numbers.** Corpus statistics and public leaderboard scores both
  drift as the field moves; an undated one is meaningless. Quote the archive set
  and the date it was measured on.

## Documents

| document                                                                   | answers                                                                                                                                                                                                                                       | do not use it for                                                                                    |
| -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| [2026-08-07-self-play-rl-findings.md](2026-08-07-self-play-rl-findings.md) | Why our self-play loop learned nothing: advantage cancellation with a difference reward, policy-invariance of shaping, what single-box Lux winners actually did, whether a league is needed, whether throughput is the constraint.            | Its Finding 5 (drop the KL to a weak clone) was **refuted by measurement** — see below.              |
| [2026-08-07-fable-rl-grounding.md](2026-08-07-fable-rl-grounding.md)       | Which published solution is closest to our problem _and_ reproducible. Ranks Lux S1/S2/S3, Halite, Kore candidates against game shape, single-box compute, and whether a public artifact exists. Picks Toad Brigade and specifies its recipe. | Its recipe values are incomplete — four load-bearing values live outside the phase YAMLs. See below. |
| [2026-08-08-market-execution-rl.md](2026-08-08-market-execution-rl.md)     | Selling into a market you move yourself: optimal execution and market impact RL, multi-agent settings where actions form prices, and reward designs for selling _well_ rather than merely selling.                                            | (in progress)                                                                                        |

## Corrections these documents do not contain

Recorded here because the documents are snapshots and the measurements came
later. The full experimental ledger is at
`.superpowers/sdd/2026-08-07-self-play-rl/progress.md` (git-ignored scratch).

- **The teacher KL is load-bearing, not a pin.** The self-play findings doc
  argues the KL toward a zero-banking clone is a cost. We ran the no-KL arm:
  removing it gradually _and_ abruptly both collapse the policy. The doc's own
  closing line — "set empirically, per project" — is what held up.
- **Fresh initialisation cannot discover this game's economy.** Four arms at or
  near the full 2e7-step budget (Toad's reward verbatim; money component clamped
  at 0.001; clamped at 0.01, killed on a reward-farming pump; signed at 0.01) all
  banked ~0. Initialisation is load-bearing. DORA-style "no BC needed" does not
  transfer at our budget.
- **The phase YAMLs are not a complete recipe.** Reproducing Toad required four
  values absent from them, two of which live in their Python and one only in the
  run configs saved beside their checkpoints: the `/500.` reward normaliser,
  phase-1 `teacher_kl_cost: 0.`, `clip_grads: 10.0`, and a _bounded_
  `BaselineLayer` (sigmoid rescaled to the reward range, then expanded by
  `MAX_DAYS` when `only_once` is false — so the effective bound is ±1, and
  "tightening toward theirs" at ±1/720 is a trap).
- **Self-play bank is not evidence.** Checkpoints self-playing at ~50 bank ~8
  against a real scripted opponent, 0.3rd percentile of the corpus. Mirror
  equilibria are private. Only real-opponent evaluation counts.
- **Corpus median seat banks 114,404**, directly measured from 396 seats across
  8 daily archives on 2026-08-08. An earlier figure of 125,773 did not
  reproduce; sampling bias was ruled out (head-of-archive vs randomised differ
  by 1%). Quote the archive set and date with any future figure.
