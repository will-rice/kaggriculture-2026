# Is it the hyperparameters? A sweep to find out

**Design, 2026-08-22.** Engine `kaggle-environments` 1.32.7.

## 1. The confound this exists to remove

Eleven RL arms have been recorded as evidence that from-scratch RL fails at
this budget. Checking their configs in wandb, every one of them ran at:

|                 | every arm |
| --------------- | --------- |
| learning rate   | 1e-4      |
| blocks          | 8         |
| channels        | 256       |
| teacher KL cost | 0.005     |

What varied was reward shaping (`shaped`, `margin`, `shaped_money`),
initialisation, opponent mix and `value_passes`. The optimiser and the
architecture were never varied. One deliberate lr change exists -- a 1e-3 run
that collapsed for a diagnosable reason (a self-referential target with a
saturated bounded head) and was withdrawn.

So "from-scratch RL is dead" is confounded with "this architecture at this
learning rate with this KL cost does not work". Eleven replications of one
point in hyperparameter space is one experiment repeated, not eleven
experiments. Phase 2 was about to be justified by that conclusion.

## 2. What the failure actually looks like

It is not that nothing is learned. Arm T's mean terminal bank against the
scripted opponent rose from 19 to about 750 across 1228 updates, and its shaped
reward rose 2.5x. The optimiser works and the policy improves.

The problem is the rate. Because the scripted opponent's bank is near-constant,
`margin ~= our_bank - 130,000`. Arm T closed about 730 coins of a 130,000-coin
gap -- roughly 200x too slowly. **A system that learns steadily but two orders
of magnitude too slowly is the signature a too-low learning rate produces**,
which is exactly the hypothesis the sweep tests.

## 3. Arms

Six arms, 400 updates each, three concurrent, about 14 wall-clock hours. Run on
the local workstation, the same host arm T ran on, so **arm T is the control by
construction** and no separate control arm is needed. One variable moves per
arm; everything else stays at arm T's settings (`--teacher --value-warmup
--econ-fraction 0.5 --channels 256 --value-passes 4`, `shaped` reward).

| arm | change                    | rationale                                                            |
| --- | ------------------------- | -------------------------------------------------------------------- |
| H1  | `lr` 3e-4                 | learning happens but ~200x too slowly; this is the direct test       |
| H2  | `lr` 1e-3                 | one decade up; the previous 1e-3 failure had a separate, fixed cause |
| H3  | `lr` 3e-3                 | far enough up to fail loudly if the direction is wrong               |
| H4  | entropy cost x3           | if the policy is stuck in a local optimum rather than merely slow    |
| H5  | teacher KL 0.005 -> 0.001 | the teacher may be pinning the policy near a weak clone              |
| H6  | channels 256 -> 384       | capacity; the one non-optimiser axis worth a slot                    |

An arm whose bank collapses to zero and stays there for 50 updates is stopped
early and recorded. That is a result, not a failure of the sweep -- it is what
the 1e-3 TD run did, and knowing where the ceiling is has value.

## 4. The bar, pre-registered before any arm runs

**Metric:** mean `diag/bank_vs_econ` over updates 300-420.

Bank is a poor capability proxy -- corpus mining put bank against ladder rating
at Pearson -0.043 -- but it is the right _learning-speed_ proxy here, because
margin is bank minus a near-constant, and bank is what visibly moves at this
timescale. `objective/margin_vs_econ` is reported beside it and never replaced
by it.

**Control, read from arm T's log before the sweep starts:**

| statistic                      | value                         |
| ------------------------------ | ----------------------------- |
| mean bank, u300-420            | 77.6                          |
| sd                             | 66.8                          |
| standard error (n=121)         | 6.1                           |
| smallest detectable difference | about 17, i.e. 22% of control |

**Bar: an arm must reach a mean of 233 (3x control) to count.** That sits 18
standard errors above the control. Consecutive updates are autocorrelated, so
the true standard error is larger than 6.1; the bar survives even a fivefold
inflation of it, at about 3.6 SE. It is deliberately far above the noise
because the claim it would license -- that eleven prior arms were confounded --
should require an unmistakable effect, not a marginal one.

## 5. What each outcome means

- **An arm clears the bar.** The hyperparameters were the problem, or part of
  it. The RL path is not dead, Phase 2's justification weakens, and the next
  step is a longer run at the winning setting rather than behaviour cloning.
- **No arm clears the bar.** "Not the hyperparameters" becomes a measured
  finding rather than an assumption, across three decades of learning rate plus
  entropy, KL and capacity. Phase 2 proceeds on firmer ground than it has now.
- **An arm collapses.** Records the upper edge of the usable range, which no
  prior arm established.

Either way the sweep costs about 14 hours and settles a question that a
multi-week phase is currently resting on.

## 6. What this requires from the code

`lr` and the entropy cost are module constants in
`learn/scripts/toad_phase1.py`, not flags, so the sweep cannot vary them
without exposing them. That is the only production change this design needs:
add `--lr` and `--entropy-cost`, defaulting to the current constants so every
existing invocation behaves identically.

## 7. Non-goals

- No reward-function work: `shaped` throughout, since reward shaping is the one
  axis the eleven arms _did_ explore.
- No architecture search beyond the single width arm.
- No changes to the simulator, the gate, or the search stack.
- The sweep does not attempt to produce a submittable agent. It answers one
  question.
