# Corrections to RESEARCH.md — primary-source verification pass

Run 2026-08-04, deep mode, three parallel retrieval agents. Every claim below was checked against the
competitor's or author's own artifact. Where no primary supports a figure it is marked UNVERIFIED rather
than cited to a secondary.

## Executive summary

**The quality gate was met on this run**: 28 evidence spans, every registered numeric claim traced to a
primary artifact or explicitly marked UNVERIFIED. No claim in this document rests on the roundup [8].

The prior run's failure was not random. Source [8] commits the **same units error at least twice** —
reporting environment-step counts in the parameter-count column — and RESEARCH.md inherited both. The
prior run detected one of them (23 billion "parameters") and excluded it as _implausible_, which was the
right instinct and the wrong diagnosis: 23B is a real, correct figure sitting in the wrong column. Having
mislabelled a units error as an outlier, the run did not generalise, and the same source's "300M
parameters" for Lux S3 second place passed through unchallenged into our architecture planning.

Three further claims came from [8] alone and collapse on inspection. One citation is misattributed to the
wrong author. One compute claim is flatly contradicted by its own source, and it is the evidential basis
for a phase of the plan.

**Recommendation: strike [8] from the bibliography entirely.** Every number taken from it that has now
been checked has been wrong. It retains value only as an index of who placed where, which is available
from the leaderboard directly.

---

## A. Numbers that were wrong

| #   | RESEARCH.md says                                                                         | Primary source says                                                                                                                                                                                          | Source |
| --- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------ |
| A1  | Lux S3 2nd: "dual 300-million-parameter PPO models for ten million steps"                | **10,000,000 parameters**, ~**300,000,000 game steps** (600M per-player observations). Transposed.                                                                                                           | [16]   |
| A2  | Lux S3 10th: 23-billion-parameter model, single RTX 4090 (excluded as implausible)       | **23B is environment steps** for submission #2. Model is **~1.8M params**, ~3.2M with critic. Hardware: local RTX 4090 **plus 2–4 rented cloud RTX 4090s** over ~10 days.                                    | [17]   |
| A3  | Lux S3 1st: "a 200-million-parameter IMPALA model on eight H100s for three to four days" | Writeup states **no parameter count and names no GPU** — "We had substantial computing resources". Verified: **~1.5B env steps**, ~3–4 days, 20B+ steps across the competition.                              | [6]    |
| A4  | microRTS: "70 GPU-days, later reduced to 23 by bootstrapping with behaviour cloning"     | **CONTRADICTED.** RAISocketAI = 70 GPU-days. RAI-BC (23 GPU-days) is **behaviour-cloning only and materially weaker** (44% vs Mayari). BC + PPO fine-tuning = 23 + 49 = **72 GPU-days, more than plain RL**. | [9]    |
| A5  | `[9] S. Huang et al.` for arXiv:2402.08112                                               | The paper is by **Scott Goodfriend**. Huang et al. wrote arXiv:2105.13807 (Gym-µRTS), the environment/GridNet/masking paper it builds on.                                                                    | [9]    |
| A6  | Kore 2022: "rule-based agents took first, second and fourth"                             | **All of the top five** were rule-based. Third place abandoned RL explicitly.                                                                                                                                | [4]    |
| A7  | "a Lux S2 competitor abandoned a **trained** model"                                      | The model was **still training** — FLG cancelled DoubleCone(6,8,6) ~100M steps in. The 404-logs and un-installable-runtime details are verbatim accurate.                                                    | [3]    |
| A8  | Generals.io simulator: "roughly a 10,000× speedup"                                       | Paper claims "more than four orders of magnitude"; its own figures give 50.7M vs 3,500 steps/s ≈ **14,500×**.                                                                                                | [10]   |
| A9  | `[3] FLG ... 4th place`                                                                  | The writeup **never states a placement**. The prior run's own `sources.jsonl` flagged this `metadata_status: unverified`; we printed it as fact.                                                             | [3]    |

## B. Claims that survived verification

- Toad Brigade: 24 residual blocks, 128-channel 5×5 convolutions, no normalisation, **~20M parameters**,
  IMPALA + UPGO + TD(λ) + frozen-teacher KL, trained on "my personal PC — an 8-core/16-thread dual-GPU
  system". Verbatim. [1]
- Lux S2 winner: forward simulation, "about 2.9 seconds every invocation", "5 to 50+ steps worth of
  planning". Verbatim, except the source says "units **and factories**". [2]
- FLG: agents predicting fewer than 4–6 steps ahead were "crippling". Teacher-KL weight ~5e-3, teacher
  kept ~30M steps behind — which is exactly what STRATEGY.md Phase 3 specifies. [3]
- Nebula's Rust rewrite: 12.9 ms/step → 0.08 ms/step = 161×. The "160×" rounding is fair. [5]
- Halite IV went to a non-learned agent — though more precisely a heuristic-scoring plus greedy-planning
  system, constants tuned by Bayesian optimisation, after an abandoned deep-RL attempt. [18]
- microRTS: 100 ms per-turn deadline, GridNet with invalid-action masking, downscaled backbone chosen for
  inference speed, shaped→sparse reward schedule, iterative fine-tuning against prior winners. [9]

## C. New facts the prior run missed, and their consequences

**C1. Every verified winning model is small.** microRTS 5.0M; Huang et al.'s best <1M; Lux S3 10th 1.8M
(3.2M with critic); Lux S3 2nd 10M; Lux S1 1st ~20M. **No primary source anywhere supports a winning
model above 20M parameters.** The two figures that suggested otherwise — 300M and 23B — were both step
counts. Our Phase 2 sizing should start at ~10M and treat 20M as the ceiling to justify, not the floor.

**C2. Toad Brigade independently validates our sandbox probe.** They report "2–2.5 seconds for inference
on the Kaggle servers with a batch size of 2" [1]. Reconstructing their architecture gives 19.7M
parameters (they say ~20M) and 40.3 GMACs per forward. Our probe measured 20–26 GMAC/s, predicting
1.5–2.0 s. Two independent measurements five years apart agree, so the GMAC budget is well founded.

**C3. The submission cap is 100MB, and it is what bounded their model.** "I was still 62MB shy of the
100MB submission file size limit" [16] — a 10M-parameter model ships at ~38MB. Kaggriculture's own cap
remains UNVERIFIED (its rules pages do not render for unauthenticated fetches) but the Lux S3 figure is a
strong prior.

**C4. Every one of these competitions ran CPU-only inference under a hard per-turn deadline.** microRTS
100 ms on a 6-thread 2018 Mac Mini; Lux S2 "CPU-only VMs... mostly single-core"; Kore 3 s plus a 60 s
bank; Kaggriculture 1 s plus 60 s. The constraint that shapes these solutions is universal, and it is the
one our Phase 0 probe measured directly.

**C5. A fast simulator did not win Kore.** Nebula finished **19th**. The Rust rewrite is real, but their
search agent "did not really perform well against a rule based baseline" and the submission reverted to
heuristics. This strengthens STRATEGY.md's existing decision to defer the rewrite.

**C6. Phase 2's premise needs revisiting.** "Pick the architecture with cheap supervised signal before
spending the expensive RL budget" was justified by the microRTS 70→23 GPU-day saving, which does not
exist (A4). Imitation there produced a _weaker_ policy, and BC-plus-fine-tuning cost _more_ than direct
RL. Imitation may still be worth doing here to escape a cold start, but the compute-saving argument for
it is withdrawn.

---

## Bibliography additions

- [16] I. Pressman et al. (Frog Parade). _kaggle-lux-2024_ — code and `write-up.md`, Lux AI Season 3 2nd
  place. https://github.com/IsaiahPressman/kaggle-lux-2024
- [17] Boey. _End-to-End JAX RL_, Lux AI Season 3 10th place.
  https://www.kaggle.com/competitions/lux-ai-season-3/discussion/570196
- [18] T. Van de Wiele. _1st Place – Winning Solution_, Halite IV.
  https://www.kaggle.com/competitions/halite/discussion/183543

`[9]` re-attributed: S. Goodfriend, _A Competition Winning Deep Reinforcement Learning Agent in
microRTS_, arXiv:2402.08112.
