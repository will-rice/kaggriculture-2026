# Explained variance of the critic, and where the return's variance comes from

**Measured 2026-08-09** on `kaggle-environments` 1.32.6, over 16 checkpoint
readings of 16 fresh episodes each (32 for the two mirror readings) plus a
96-episode variance decomposition. Seeds 900,000–900,015 and 910,000–910,005,
chosen disjoint from every seed any arm trained on — the arms walk
`range(update * 24, (update + 1) * 24)` and the longest ran to update 707, so
training consumed seeds below ~17,000. Reproduce with
`uv run python -m kaggriculture.learn.scripts.critic_ev`; the machine-readable
output is [`2026-08-09-critic-explained-variance.json`](2026-08-09-critic-explained-variance.json)
and the code is `src/kaggriculture/learn/scripts/critic_ev.py` on branch
`worktree-agent-a787fb13772848793`, which is where the whole Toad line lives —
none of `toad_loss`, `toad_phase1` or the vendored returns exist on this branch,
so the script cannot be run from here.

**The question.** Ten arms have failed the same way — the policy-gradient term
wanders toward zero while the value loss falls. The standing explanation, in the
ledger and in commit `088f9fd`, is that a seat's return is dominated by exogenous
shared-market variance, that the critic therefore explains nearly all of it, and
that a perfectly predicted return leaves no advantage for the policy gradient.
That predicts an explained variance near 1.0 and advantages decaying to a noise
floor.

---

## 1. The headline: the explained variance is nowhere near 1.0

`EV = 1 − Var(G − V) / Var(G)`, where `G` is the **full-episode discounted
return-to-go** at Toad's `DISCOUNTING = 0.999` over the same reward series the
arm's learner read, and `V` is the value the checkpoint itself emitted at that
state during the rollout. `G` is computed by the learner's own vendored
`td_lambda.td_lambda` at `lmb = 1.0`, where the `(1 − lmb)` factor deletes the
only term values enter through — so it is the pure Monte Carlo return and cannot
disagree with the learner's target about the discount, the ordering or the
episode boundary. Verified against a hand-rolled backward scan to 1.3e−6.

Every reading is against `economic_policy` in seat 1 unless marked _mirror_.

| checkpoint                                    | reward    | update |  **EV** | EV within turn | EV of a clock-only critic | EV vs the learner's own TD(λ) target |
| --------------------------------------------- | --------- | -----: | ------: | -------------: | ------------------------: | -----------------------------------: |
| `toad-phase1m-margin_000225`                  | `margin`  |    225 |  −5.325 |         −0.219 |                     0.761 |                               0.9921 |
| `toad-phase1m-margin_000300`                  | `margin`  |    300 | −27.282 |         −0.401 |                     0.780 |                               0.9978 |
| `toad-phase1m-margin_000400`                  | `margin`  |    400 | −29.979 |         −0.398 |                     0.750 |                               0.9992 |
| `toad-phase1m-margin_000500`                  | `margin`  |    500 | −14.993 |         −0.054 |                     0.789 |                               0.9981 |
| `toad-phase1m-margin_000600`                  | `margin`  |    600 | −14.775 |         +0.042 |                     0.835 |                               0.9984 |
| `toad-phase1m-margin_000700`                  | `margin`  |    700 | −12.451 |         +0.061 |                     0.724 |                               0.9983 |
| `toad-phase1m-margin_000700` _mirror_         | `margin`  |    700 | −237.66 |         −5.885 |                     0.000 |                               0.9983 |
| `phase1p_000225` (live arm)                   | `shaped`  |    225 | −28.297 |         −2.702 |                     0.249 |                               0.9907 |
| `phase1p_000300` (live arm)                   | `shaped`  |    300 | −192.47 |        −32.918 |                     0.418 |                               0.9892 |
| `toad-phase1c3-clone-teacher_000025`          | `shaped`  |     25 |  −3.720 |         −2.420 |                     0.271 |                               0.8923 |
| `toad-phase1c3-clone-teacher_000100`          | `shaped`  |    100 |  −0.677 |         −0.718 |                     0.112 |                               0.9806 |
| `toad-phase1c3-clone-teacher_000200`          | `shaped`  |    200 |  −0.222 |         −0.605 |                     0.340 |                               0.8961 |
| `toad-phase1c3-clone-teacher_000300`          | `shaped`  |    300 |  +0.016 |         −0.093 |                     0.255 |                               0.9018 |
| `toad-phase1c3-clone-teacher_000300` _mirror_ | `shaped`  |    300 |  −0.569 |         −0.508 |                     0.056 |                               0.9794 |
| **control** shaped-PPO, `own`                 | `own`     |      — |  +0.339 |         −2.835 |                     0.981 |                               0.9908 |
| **control** shaped-PPO, `rewards`             | `rewards` |      — |  −0.300 |         +0.010 |                     0.690 |                               0.9721 |

**The best explained variance in the table is +0.339, and fourteen of the
sixteen readings are negative** — on those, the critic is a worse predictor of
the return than the return's own mean. Only two readings clear zero at all:
`c3_000300` at +0.016 and the control at +0.339. The hypothesis needed ~1.0. It
is refuted.

Three columns matter beyond the headline.

**"EV within turn"** centres both `G` and `V` at each turn index across episodes
before scoring, so the deterministic profile of the return-to-go in `t` is
removed from numerator and denominator alike. It is the number the hypothesis is
really about: does the critic tell two episodes apart at the same point in the
season? The best reading is **+0.061**. The critic discriminates the draw
essentially not at all.

**"EV of a clock-only critic"** is the same estimator with `V` replaced by the
cross-episode mean of `G` at each turn index — a critic that has learned the
calendar and nothing else. It scores **0.72–0.84 on the margin arm** and **0.981
on the control**. So most of the pooled return variance is the turn index, and
the learned critics do not reach even that.

**"EV vs the learner's own TD(λ) target"** is what the `baseline` loss term
measures: the same 16-turn segments `toad_phase1._step` builds, `lmb = 0.8`,
bootstrapped from the value of the segment's own last state. It reads
**0.89–0.999 everywhere.** Reconstructed end to end, the smooth-L1 sum comes out
at 0.00169 per batch against the 0.00129 the margin arm logged at update 701 —
same quantity, the gap being population (this is scripted-opponent episodes only)
and actor lag.

> **This is the correction the project needs.** The falling `baseline` term
> measures the critic's **self-consistency**, not its accuracy. A value function
> can be a near-exact fixed point of its own bootstrapped target (0.998) while
> being a twelvefold-worse predictor of the actual return than a constant
> (−12.5). Both numbers are from the same 16 episodes of the same checkpoint.
> "The critic learned to predict the return very well" does not follow from the
> value loss, and it is false here.

## 2. What the critic learned: the local reward rate, extrapolated past the horizon

Averaged over the 16 episodes of `toad-phase1m-margin_000700`, the critic runs
the wrong way down the season:

| turn |   V(s) | true G(s) | V/G measured | V/G predicted by the fixed point |
| ---: | -----: | --------: | -----------: | -------------------------------: |
|    0 | −0.377 |    −0.177 |         2.13 |                             1.95 |
|  100 | −0.487 |    −0.196 |         2.49 |                             2.17 |
|  200 | −0.709 |    −0.216 |         3.28 |                             2.47 |
|  300 | −0.782 |    −0.217 |         3.60 |                             2.92 |
|  400 | −0.855 |    −0.208 |         4.11 |                             3.66 |
|  500 | −0.932 |    −0.162 |         5.74 |                             5.08 |
|  600 | −0.963 |    −0.116 |         8.33 |                             8.91 |
|  718 | −0.976 |    −0.020 |            — |                                — |

The true return-to-go **rises** toward zero as the season runs out. The critic
**falls** toward its −1 rail. That anti-correlation is the whole of the negative
EV.

The mechanism is arithmetic, and the last column is the check. A constant critic
`c` is a fixed point of a 16-step self-bootstrapped TD(λ) target when
`c = Σ_{k<16} γ^k r + γ^16 c`, i.e. `c = 1000 r` at `γ = 0.999`. The true
finite-horizon return-to-go with `n` turns left is `r (1 − γ^n) / (1 − γ)`, which
is `512 r` at turn 0 and `113 r` at turn 600. The predicted ratio is therefore
1.95 rising to 8.91, and the measured ratio is 2.13 rising to 8.33, over a
sevenfold change, from a one-parameter model with nothing fitted. **The critic
has learned the local reward rate extrapolated to an infinite horizon.** Nothing
inside a 16-turn window says the season ends in 300 turns, and the terminal fact
reaches only the final segment of each episode.

**This is not a transcription bug and it is not, on its own, evidence of
failure.** The segmentation is Toad's own — `unroll_length: 16`,
`discounting: 0.999` — and the ledger's faithfulness audit found `toad_loss`
faithful to `monobeast.py:355-425`. Their game is 360 turns against our 719
(deviation D2 in `toad_phase1`'s own list), where the same arithmetic gives a
ratio of **3.3 at turn 0, worse than our 1.95**, and they won with it. A smoothly biased critic still yields a usable TD error. What the
measurement establishes is narrower and firmer: the value loss is not a
measurement of the critic's accuracy, so no conclusion about the return's
predictability can be drawn from it.

## 3. The return's variance is mostly ours, not the market's

Law of total variance, `Var(G) = E_seed[Var_action(G | seed)] + Var_seed(E_action[G | seed])`,
estimated by playing each of 6 seeds under 8 independent action streams — one
lockstep group per seed holding that seed repeated, so every environment in the
group is the same market draw and only the sampling differs. The engine makes
this exact: the seed fixes the weed spawns and the shop unlock order, and
`economic_policy` is a deterministic function of state, so what is left is what
our own sampling did (including the opponent's reaction to it, which belongs on
our side of the ledger).

`Var_seed` of a mean over 8 draws is inflated by `Var_action / 8`; both the raw
and the bias-corrected share are given.

| policy             | quantity       | within-seed (ours) | between-seed (the draw) | controllable share | corrected |
| ------------------ | -------------- | -----------------: | ----------------------: | -----------------: | --------: |
| `margin_000700`    | episode return |          4.396e−04 |               1.501e−04 |              74.5% | **82.2%** |
| `margin_000700`    | final bank     |             20,452 |                   1,841 |              91.7% |     ~100% |
| `margin_000700`    | final margin   |          3.316e+08 |               1.152e+08 |              74.2% | **81.8%** |
| control shaped-PPO | episode return |          3.435e−02 |               6.823e−03 |              83.4% | **93.1%** |
| control shaped-PPO | final bank     |            612,508 |                 134,787 |              82.0% | **91.3%** |
| control shaped-PPO | final margin   |          2.185e+08 |               1.357e+08 |              61.7% | **66.8%** |

**Three quarters to nine tenths of the return's variance is moved by our own
action sampling.** The exogenous draw contributes the minority share. On the
margin arm's own final bank, the between-seed component is smaller than its own
sampling error — indistinguishable from zero.

**Confidence: moderate-to-high on the direction, low on the third digit.** Six
seeds is five degrees of freedom for the between-seed term, so the shares carry
several points of error; the _ordering_ is not in doubt at that resolution. The
one confound that cannot be measured away is stated plainly: the controllable
share is the variance **this policy's entropy actually explores**, not the
variance an optimal policy could induce. It is a lower bound on controllability,
and an upper bound on nothing. The corrected column is the honest one; the raw
column is what a naive estimator would print.

**Where the +0.98 argument goes wrong.** The corpus figure — a seat's final bank
correlates +0.977/+0.976 with its opponent's on 1.32.5/1.32.6 — is a statement
about **levels across a population of seats spanning a wide bank range**. It does
not license the inference that the variance a **fixed policy** faces across
episodes is exogenous, and measured directly it is not. Within our own 16
episodes the same correlation reads +0.161 for the control (bank 14,426 ± 712)
and −0.010 for the margin arm — restricted-range statistics that neither confirm
nor refute the corpus number, and that is the point: they are different
quantities. The corpus number survives; the inference drawn from it does not.

## 4. Advantages do not shrink toward a noise floor

RMS of the V-trace `pg_advantages`, over the exact segments and batches the
learner builds. The two reference columns are the same estimator with the critic
swapped for one that knows only the mean return, and one that knows only the turn
index — so the critic is the only thing that differs.

| checkpoint      | learned critic | flat critic | clock critic | mean per-turn \|reward\| |
| --------------- | -------------: | ----------: | -----------: | -----------------------: |
| `margin_000225` |        0.01762 |     0.00575 |      0.00337 |                 0.000582 |
| `margin_000300` |        0.02257 |     0.00575 |      0.00331 |                 0.000567 |
| `margin_000400` |        0.01311 |     0.00549 |      0.00328 |                 0.000569 |
| `margin_000500` |        0.01586 |     0.00576 |      0.00330 |                 0.000580 |
| `margin_000600` |        0.01249 |     0.00551 |      0.00320 |                 0.000580 |
| `margin_000700` |        0.01258 |     0.00579 |      0.00335 |                 0.000583 |
| `c3_000025`     |        0.02300 |     0.00706 |      0.00591 |                        — |
| `c3_000100`     |        0.00714 |     0.00566 |      0.00450 |                        — |
| `c3_000200`     |        0.00506 |     0.00450 |      0.00304 |                        — |
| `c3_000300`     |        0.00478 |     0.00493 |      0.00350 |                        — |

On the margin arm the advantage magnitude is flat across 475 updates — 0.0176 at
update 225 and 0.0126 at update 700, non-monotone in between — and is **two to
four times larger** than what the same policy gradient would see with no critic
worth the name, and ~22x the mean per-turn reward. Nothing is decaying to a floor.

The same conclusion from the arm's own log. `vtrace_pg` is a **signed** sum over
64 elements, and the ledger's "0.058 → −0.0026 over 500 updates" is two single
samples of a quantity whose per-window standard deviation is ~0.07. In 50-update
windows:

| updates | mean `vtrace_pg` | mean \|`vtrace_pg`\| |   s.d. | `baseline` | `entropy` term |
| ------- | ---------------: | -------------------: | -----: | ---------: | -------------: |
| 201–250 |          +0.0581 |               0.0666 | 0.0805 |    0.00497 |        −0.1639 |
| 251–300 |          +0.0096 |               0.0228 | 0.0295 |    0.00305 |        −0.1440 |
| 301–350 |          +0.0153 |               0.0271 | 0.0323 |    0.00221 |        −0.1420 |
| 351–400 |          +0.0124 |               0.0227 | 0.0275 |    0.00159 |        −0.1751 |
| 401–450 |          +0.0163 |               0.0374 | 0.0727 |    0.00134 |        −0.1938 |
| 451–500 |          +0.0412 |               0.0505 | 0.0479 |    0.00217 |        −0.2461 |
| 501–550 |          −0.0185 |               0.0416 | 0.0519 |    0.00176 |        −0.2753 |
| 551–600 |          −0.0018 |               0.0599 | 0.0713 |    0.00153 |        −0.3176 |
| 601–650 |          −0.0274 |               0.0631 | 0.0757 |    0.00145 |        −0.3181 |
| 651–700 |          −0.0067 |               0.0549 | 0.0668 |    0.00146 |        −0.3273 |

The **mean** oscillates through zero; the **magnitude** dips mid-run and returns
to where it started. The entropy term doubles, so the policy broadened rather
than collapsed. `baseline` is the only term that falls monotonically, and section
1 says what it is measuring.

**The one arm where advantages do shrink is the c3 lineage**, 0.0230 → 0.0048
over 300 updates, converging on its own flat-critic reference of 0.0049 — and
that is the arm whose EV climbs from −3.72 to +0.016 over the same span. That
pairing is what the hypothesis predicted; it happens on the shaped reward, and
the arm still banks 5.8 against `economic_policy`'s ~137,000.

## 5. The control

`/data/kaggriculture/selfplay/resumed-3cc6068-1786139297/policy.pt`, the 8-block
/ 256-channel pre-Toad shaped-PPO checkpoint, loaded with an **unbounded** value
head — it was trained without Toad's sigmoid rescaling, and constructing it
bounded would score a value function that never existed. It banks **14,426 ± 712**
here, reproducing the independent gate reading of **14,339** on this engine
([engine-1326-rebaseline](2026-08-08-engine-1326-rebaseline.md)) and confirming
the harness.

Its EV is **+0.339** against its own-bank return and **−0.300** against the bank
differential. Better than every Toad arm, and still nowhere near 1.0 — and a
clock-only critic beats it, 0.981 to 0.339. So the falsification the brief
specified does not fire: high EV is **not** a property shared by working and
non-working policies here, because nothing we hold has high EV.

**A declared limitation on this reading.** The control was trained on
`kaggle-environments` 1.32.3, which banked roughly twice what 1.32.6 does. Its
`V(s_0)` is +5.03 against a measured own-return of +2.64 — consistent with a
critic well calibrated to a season worth ~21,000 coins being scored on one worth
~14,400. Its EV here is therefore a **lower bound**, depressed by an
out-of-distribution level shift we did not correct. Its progress-shaping term is
also not reconstructible on this branch and is omitted from its return; a
potential-based term telescopes to its terminal value alone, so the error is
small, but it is an approximation and not an identity.

## 6. What could be an artefact

- **`EV within turn` at n = 16.** Sixteen episodes is a thin estimate of a
  cross-episode variance. The readings are far from 1.0 rather than marginally
  below it, so the conclusion survives the noise, but no single one of them
  should be quoted to three digits.
- **`EV at turn 0` is 0.000 to float error on all sixteen readings, and that is
  structural, not a finding.** The encoded opening state is byte-identical across
  seeds (verified: `encode_board` and `encode_scalars` agree exactly on seeds
  900,000–900,002), so `V(s_0)` is a constant and its residual variance equals
  the return's by construction. It says the draw is invisible at turn 0, which is
  a fact about the encoder and the engine, not about the critic.
- **Zero actor lag.** A checkpoint is scored as the policy that played it, so the
  V-trace importance ratios are exactly 1. In training the actor lags by up to
  `SYNC_EVERY = 4` updates, and clipping at `ρ = 1` can only shrink advantages —
  so the measured magnitudes are an upper bound on the training ones. They would
  have to fall by more than an order of magnitude to reach the flat-critic
  reference, and the logged `vtrace_pg` does not show that.
- **The mirror readings are a different population.** `margin_000700` in mirror
  self-play has a return that is very nearly deterministic (episode-return s.d.
  0.0061 against 0.0302 versus the scripted opponent) and a clock-critic EV of
  exactly 0.000. The extreme −237 EV there is a small denominator, not a worse
  critic. Half of every training round was mirror, so this population is real,
  but its EV should not be compared numerically with the scripted one.
- **What would overturn section 3.** A policy with materially more entropy, or a
  competent one, could face a different variance split. The control — a policy
  that banks 14,426 rather than 66 — reads 93.1% controllable, which is the
  opposite of the direction that would worry us, but two policies is two policies.

## 7. What follows

**Advantage-based RL is not shown to be structurally mismatched to this game, and
the evidence that was thought to show it does not.** The critic never explained
the return; the value loss was measuring its agreement with itself; the
advantages never collapsed; and the seat's own actions move three to nine times
more of the return's variance than the market draw does. The ten failures need a
different explanation, and this measurement removes the one that had been
adopted.

Two things this does **not** say. It does not say more compute will work — the
arms failed on the objective at full budget and that is untouched here. And it
does not identify the cause; the horizon-blind critic of section 2 is a real
defect but the winning recipe carried a worse one, so it is a candidate and not a
verdict.

The cheapest next reads, in order of what they would settle:

1. **Log a real EV during training**, not just `baseline`. One number per update
   against a Monte Carlo return over completed episodes would have caught this in
   an hour rather than after ten arms.
2. **Give the critic the horizon.** The turn index is already in `SCALARS`, but
   the 16-turn self-bootstrapped target gives it no reason to use it. Either
   bootstrap the last segment of an episode from zero rather than from its own
   value, or lengthen the unroll toward the season. One variable, and section 2
   predicts the size of the effect in advance.
3. **Re-test the `c3` pairing.** It is the only arm where EV rises and the
   advantage falls together — the behaviour the hypothesis described — and it
   still banks nothing. Whatever is wrong is visible there without the critic
   confound.
