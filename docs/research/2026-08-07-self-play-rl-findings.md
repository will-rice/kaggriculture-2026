# Self-play RL: why our loop learns nothing, and what the literature says to do

Research run 2026-08-07, four parallel streams: cached Lux/Kore competitor
writeups on disk, degenerate self-play and league literature, weak-teacher BC
initialisation and long-horizon credit assignment, and Kaggle simulation
winners' own repositories. Primary sources only; anything unverifiable is
labelled.

## The observation this explains

```
iteration 23 | 76 trajectories | bank 0 | margin 0 | entropy 1.645 | kl 0.1470 | value 0.0002
iteration 24 | 76 trajectories | bank 0 | margin 0 | entropy 1.646 | kl 0.1492 | value 0.0002
advantage -0.0008 +- 0.0128
```

Twenty-four iterations, no movement. Reward is the per-turn change in
`(our bank − their bank)`. Initialised from a behaviour clone that reaches
90.5% held-out action accuracy and banks 0 in play.

## Finding 1 — the advantage cancels by construction, and both seats made it exact

**SPIRAL** [1] documents the mechanism directly: when one policy plays both
seats and the reward is a difference, `R₁ = −R₀`, so a **shared baseline makes
the advantage cancel**. Their observed collapse is stark — output length fell
from ~3,500 characters to near zero within ~200 steps, gradient norm spiked then
flattened. Their fix is **role-conditioned advantage estimation**: a separate
EMA-tracked baseline per seat, not one shared across both.

This lands directly on a change we made this morning. Task 5 began keeping
**both seats** from each episode — a free doubling of data. With a differential
reward and a single shared value baseline, that doubling is precisely the
configuration in which the two seats' advantages cancel exactly.

SPIRAL frames the problem as _role asymmetry_ (first-move advantage). Our case
has no asymmetry to exploit, which makes it **harder**, not different.

## Finding 2 — our reward is not policy-invariant, and that is a theorem

Ng, Harada & Russell [11] prove that `F(s,a,s′) = γΦ(s′) − Φ(s)` is **necessary
and sufficient** for a shaping term to leave the optimal policy unchanged. Note
the `γ`.

Our reward is an **undiscounted** per-turn delta, optimised under a discounted
objective. That is not the required form, so it is **not policy-invariant**: it
pays more for an early lead than for the same lead late, which is a different
objective from maximising terminal margin.

This is a direct application of the theorem, not an empirical replication — no
source runs this exact ablation on an economic game. **Theoretically grounded,
empirically unverified for this game.**

## Finding 3 — every single-box winner shaped first and switched later

Three independent primary sources, all one machine:

| solution                         | what they did                                                                                                                                  |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **FLG**, Lux S2 4th [3]          | heavily shaped self-interested reward for **~65M steps** — move-to-resource, clear rubble, mine — _then_ switched to the zero-sum differential |
| **Frog Parade**, Lux S3 gold [4] | sparse ±1 _"as soon as training was running stably"_                                                                                           |
| **Toad Brigade**, Lux S1 1st [2] | shaped for the **first 20M steps**, then distilled onto the sparse reward with the smaller net as teacher                                      |

**Flat Neurons** (Lux S3 1st) names the exact failure: _"Partial scoring for
relic points helped prevent 'do nothing' stagnation, when agent thinks it will
guaranteed win or lose."_ A purely relative reward goes flat whenever the
outcome stops being in doubt.

FLG's differential formulation — the closest published analogue to ours — is
**not purely differential**. Alongside the relative terms sits **+0.3 per
factory**, absolute, paying regardless of the opponent. Their reward keeps
gradient under symmetry. Ours does not.

**Bansal et al.** [5] give the general form: a dense-to-sparse curriculum exists
specifically to solve exploration in symmetric self-play.

Our spec forbade shaping, on the grounds that it "would encode our own beliefs
about good play, and the whole point is to exceed them". That confuses the final
objective with the training curriculum, and the evidence is against it.

## Finding 4 — entropy alone will not rescue it

**Tang et al.** [6] report a negative result worth having: standard self-play
policy gradient converges to a degenerate equilibrium in symmetric
multi-equilibrium games _"even using state-of-the-art exploration techniques"_,
entropy included. Their working fix is reward randomisation.

Our entropy sits at 1.645 and is not falling. That is not the problem, and
raising it will not help.

## Finding 5 — the KL to our clone is probably doing harm

**Jump-Start RL** [3] documents the failure directly: naively continuing RL from
a pretrained policy causes _"actor performance decays, as the untrained critic
provides a poor learning signal, causing the good initial policy to be
forgotten."_ **Posterior Behavioral Cloning** [4] shows theoretically that
standard BC _"can fail to ensure coverage over the demonstrator's actions, a
minimal condition necessary for effective [RL] finetuning"_ — a clone can be
actively bad as an initialisation, not merely neutral.

**Kickstarting** [5] treats the KL coefficient as something to actively decay,
reporting 42% final-performance improvement over from-scratch _because_ it
decays. AlphaStar [6] uses its KL to the supervised policy as a regulariser for
exploration, not as a target.

**DORA** [8] trains no-press Diplomacy entirely from self-play with **no BC
initialisation and no human data**, reaching strong play in a different
equilibrium — discarding the clone is a published, viable option.

And our clone is a textbook instance of what **Codevilla et al.** [2] describe:
open-loop accuracy is a poor predictor of closed-loop success. 90.5% held-out
accuracy with a bank of zero is exactly that, and **Ross & Bagnell**'s [1]
`εH²` compounding bound is the mechanism.

No source gives a rule for when to decay based on teacher weakness. Set
empirically, per project.

## Finding 6 — a league is overkill; the cheap parts are the evidenced parts

**AlphaStar**'s league [2] is 12 agents, each 32 TPUv3 for 44 days — order
17,000 TPU-days total (arithmetic from sourced figures, not stated verbatim).
**PSRO** [9] costs O(n²) pairwise simulations per iteration. Neither is
available to us and neither is validated at our scale.

What is cheap and evidenced:

- **N frozen past checkpoints + M scripted opponents.** OpenAI Five [3] sampled
  80% latest-self / 20% past checkpoints. Mechanically this breaks the exact
  symmetry that zeroes our margin.
- **PFSP-style win-rate-weighted sampling** over that pool — a reweighted
  sampling table, no extra simulation cost.
- **Per-role baselines** (SPIRAL's RAE) — trivial to add, directly targets
  Finding 1.

## Finding 7 — throughput is not our constraint, and further work on it is waste

Frog Parade [4] reached gold on one workstation — Ryzen 9950X, RTX 3090 + RTX
2070 Super — at **~300M game steps over ~8 days**, about **434 effective steps
per second** end to end. Our 2.5M steps/hour is ~694/second, already faster.

They needed a custom simulator running at 110,000 steps/second to get there. We
do not need one. Engine speed was their practical constraint; it is not ours.

## Actions, in order

1. **Per-seat value baselines.** Directly addresses the exact cancellation
   Finding 1 names, and we made it exact this morning by keeping both seats.
   Measure: advantage standard deviation should stop being symmetric noise.
2. **Phase the reward.** Dense own-bank progress first, until bank rises off
   zero reliably; then the differential. Measure: bank per iteration.
3. **Put banking opponents in the pool.** `economic_policy` and the fixed route
   both bank six figures, so the margin is signed from iteration one. Keep
   vendored kaito out — it is the gate.
4. **Drop or fast-decay the teacher KL**, and run the no-KL arm against the
   current schedule. Evidence says a KL toward a 0-banking clone is a cost, not
   a stabiliser.
5. **Fix γ for a 719-step horizon**, rather than inheriting a value tuned for a
   20,000-step game, and clip policy and value gradients **separately** [12][13].
6. **Do not spend further effort on throughput.**

## What would falsify the diagnosis

If per-seat baselines and a banking opponent both land and the margin still
sits at exactly zero, the problem is not reward degeneracy — it is more likely
that `bank` is being read from a field that is always zero, which looks
identical in the logs. Play one episode with live weights and print the final
bank directly before assuming otherwise.

## Finding 8 — chain shaping does not move a warmed policy, and the terminal-zero condition is not what is holding it back

Measured 2026-08-07 21:03–22:21 from two runs resumed from the same checkpoint
(`resumed-5a0add8-1786129121/snapshot-00080.pt`), same seed, same reward label
`own+1.00*progress`, differing only in the last line of `progress_reward`:

| arm | commit | terminal potential | iters | bank iter 0 -> last |
|---|---|---|---|---|
| control | `3f96a02` | zeroed (Grzes condition enforced) | 23 | 22,384 -> 18,785 |
| ablation | `3cc6068` | carried | 8+ | 22,384 -> 19,256 |

**The two arms are indistinguishable.** Iteration-by-iteration bank, `stored`,
entropy and `kl` all agree within run-to-run noise (iter 3: 19,765 vs 20,440;
iter 7: 19,256 vs 19,287; `stored` 977 vs 1,053). `progress.py` calls the
un-zeroed form "the third instance of a failure this project has already
shipped twice" and predicts a filling shed with a flat bank. The shed does not
fill in either arm — `stored` oscillates in 920–1,160 in both — so the predicted
failure mode is not observable at this scale. **The terminal zeroing is correct
theory and should stay; it is not load-bearing, and it is not the reason the
bank does not move.**

The reason it is not observable is the larger result: **neither arm learns
anything.** The control ran 23 iterations and never once banked above its
iteration-0 value of 22,384. That sits inside the known warmed-plateau band
(~21,000) and far below the corpus median seat (125,773).

Two structural facts in the same logs say where the gap is:

- **`livestock` is 0 in every one of the 31 logged iterations across both
  arms**, and `land` is non-zero in exactly 2 (247 and 6 coins, both reverting
  the next iteration). `progress.py` prices seeds, land and livestock at cost
  specifically so that capital purchases are potential-neutral rather than a
  loss. They are neutral, and they still never happen.
- `growing` (2,570–3,480) and `stored` (920–1,160) are flat across 31
  iterations. The pipeline is neither filling nor draining.

A shaped term that makes an action free does not make it *sampled*. The chain
reward was justified by a fresh policy reaching `SELL`-legal on 0 of 719 turns —
an exploration measurement — but it was then applied to a checkpoint that
already has the chain, where it can only re-pay behaviour already present. On
that reading it is doing exactly what it should and the barrier is untouched.

**What would falsify this.** Log `BUY_LAND` / `BUY_ANIMAL` sample counts per
iteration. If those ops are being sampled at a non-trivial rate and simply not
retained, this is a credit-assignment failure and the shaping is the wrong
shape. If they are sampled ~never, it is exploration and no reward reweighting
will fix it. That counter is cheap and settles which of the two it is; it
should be added before the next reward arm is run.

## Sources

Competitor primary artifacts: Toad Brigade
<https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021>; FLG
<https://www.kaggle.com/competitions/lux-ai-season-2/writeups/flg-flg-s-approach-deep-reinforcement-learning-wit>;
Frog Parade
<https://raw.githubusercontent.com/IsaiahPressman/kaggle-lux-2024/main/write-up.md>;
Flat Neurons and others cached at `/data/kaggriculture/research/`.

Literature: SPIRAL <https://arxiv.org/html/2506.24119v2>; Bansal et al.
<https://arxiv.org/pdf/1710.03748>; Tang et al.
<https://arxiv.org/abs/2103.04564>; Jump-Start RL
<https://arxiv.org/abs/2204.02372>; Posterior BC
<https://arxiv.org/abs/2512.16911>; Kickstarting
<https://arxiv.org/abs/1803.03835>; DORA <https://arxiv.org/abs/2110.02924>;
Ross & Bagnell (AISTATS 2011); Codevilla et al. (ICCV 2019); Ng, Harada &
Russell (ICML 1999); PopArt <https://arxiv.org/abs/1602.07714>; Engstrom et al.
<https://arxiv.org/abs/2005.12729>; OpenAI Five
<https://cdn.openai.com/dota-2.pdf>; AlphaStar
<https://www.nature.com/articles/s41586-019-1724-z>; PSRO (Lanctot et al.).

A prior research run on this project cited a secondary roundup that transposed
a model's parameter count with its training-step count. Every number here comes
from a competitor's own repository or a paper, or is labelled unverified.
