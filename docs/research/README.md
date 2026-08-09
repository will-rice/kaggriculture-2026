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

| document                                                                           | answers                                                                                                                                                                                                                                                             | do not use it for                                                                                                        |
| ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| [2026-08-07-self-play-rl-findings.md](2026-08-07-self-play-rl-findings.md)         | Why our self-play loop learned nothing: advantage cancellation with a difference reward, policy-invariance of shaping, what single-box Lux winners actually did, whether a league is needed, whether throughput is the constraint.                                  | Its Finding 5 (drop the KL to a weak clone) was **refuted by measurement** — see below.                                  |
| [2026-08-07-fable-rl-grounding.md](2026-08-07-fable-rl-grounding.md)               | Which published solution is closest to our problem _and_ reproducible. Ranks Lux S1/S2/S3, Halite, Kore candidates against game shape, single-box compute, and whether a public artifact exists. Picks Toad Brigade and specifies its recipe.                       | Its recipe values are incomplete — four load-bearing values live outside the phase YAMLs. See below.                     |
| [2026-08-08-market-execution-rl.md](2026-08-08-market-execution-rl.md)             | Selling into a market you move yourself: optimal execution and market impact RL, multi-agent settings where actions form prices, and reward designs for selling _well_ rather than merely selling.                                                                  | (in progress)                                                                                                            |
| [2026-08-08-what-wins-elo.md](2026-08-08-what-wins-elo.md)                         | Whether banked coins move the ladder (they no longer do), what separates the winner of an episode from its loser, and how much of the margin is price impact on the opponent. Measured off the nine daily archives, not off the literature.                         | Anything about how to train it. It says what to optimise, not how. Its bank figures are pre-1.32.6.                      |
| [2026-08-08-engine-1326-rebaseline.md](2026-08-08-engine-1326-rebaseline.md)       | What kaggle-environments 1.32.6 changed, the head-to-head reference table re-measured on it (five scripted agents, 512 games per pair, plus the shaped-PPO checkpoint), and an explicit list of every project number the upgrade voided.                            | Deciding what to submit. It recommends and stops; submission is a human decision.                                        |
| [2026-08-09-critic-explained-variance.md](2026-08-09-critic-explained-variance.md) | Whether the critic explains the return (it does not: EV ≤ +0.339 on every checkpoint we hold, negative on 14 of 16), what the falling value loss actually measures, whether advantages shrink over training, and how much of the return's variance a seat controls. | Diagnosing what _does_ cause the ten failures. It removes one explanation and names candidates; it settles none of them. |

## Corrections these documents do not contain

Recorded here because the documents are snapshots and the measurements came
later. The full experimental ledger is at
`.superpowers/sdd/2026-08-07-self-play-rl/progress.md` (git-ignored scratch).

- **The engine changed under all of them.** The ladder moved to
  `kaggle-environments` 1.32.6 on 2026-08-07 and this project re-pinned on
  2026-08-08. Late-season town-centre demand fell about eightfold and every
  bank figure filed here roughly halved. The agent _ranking_ did not change.
  [2026-08-08-engine-1326-rebaseline.md](2026-08-08-engine-1326-rebaseline.md)
  section 4 lists which numbers in the documents above are now void; read it
  before quoting a bank from any of them.
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
  by 1%). Quote the archive set and date with any future figure. The full index
  of all nine archives puts the same statistic at 112,866 over 14,112 seats,
  and shows why any single number for it is misleading: the median seat banked
  129,152 on 2026-08-04 and 109,032 on 2026-08-07.
- **The bank is no longer the objective.** Rating against final bank reads
  −0.059 over the 536 seats of the 2026-08-07 archive that were played on
  kaggle-environments 1.32.6, and every archive-and-build cell since 2026-08-01
  falls between −0.064 and +0.175, though it was +0.686 on 2026-07-30. The
  reason is that a seat's bank correlates **+0.98 with its opponent's**: the
  shared book sets the level and the seat only contributes the 2.2% margin. See
  [2026-08-08-what-wins-elo.md](2026-08-08-what-wins-elo.md); every document
  written before it assumes banking is the goal.
- **The critic never explained the return, and the value loss was never
  evidence that it did.** Measured 2026-08-09 across sixteen checkpoint readings:
  explained variance against the discounted return-to-go is **+0.339 at best and
  negative on fourteen of sixteen**, while the `baseline` term's own TD(λ) target
  reads 0.89–0.999 on the same episodes. The falling value loss measures
  self-consistency. Advantages do not shrink, and three quarters to nine tenths
  of the return's variance is moved by the seat's own actions rather than the
  market draw — so the "+0.98 seat-to-opponent bank correlation implies the
  return is exogenous" inference does not hold, even though the correlation
  itself does. This overturns the mechanism recorded in commit `088f9fd` and in
  the ledger's margin-arm entry.
  [2026-08-09-critic-explained-variance.md](2026-08-09-critic-explained-variance.md)
- **Cut corpus statistics by engine build, not only by date.** The nine archives
  span kaggle-environments 1.32.2 to 1.32.6 and the builds are mixed _within_
  an archive — 268 of the 675 episodes dated 2026-08-07 are 1.32.6, which banks
  80,660 at the median against 1.32.5's 120,800 on the same day at the same
  rating. Pooling across builds produced a confident, monotone and entirely
  false "the best bankers are the weakest players" in the first draft of that
  document. `module_version` sits in the first 8 KB of every episode.
