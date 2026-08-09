# Task report — the production arm (phase1p)

Shape production only. No margin term, no money term. This is C6's configuration
— clone lineage, teacher KL 0.005, value warmup, half-scripted opponent mix —
re-run on kaggle-environments 1.32.6, which the ledger names as the first honest
baseline available on the live engine.

- Commit `78a909b`, branch `worktree-agent-a787fb13772848793`.
- Learner pid `2858665`, pid file `/data/kaggriculture/toad/phase1p_run.pid`.
- wandb <https://wandb.ai/will-rice/kaggriculture-2026/runs/6tqs3flj>
- jsonl `/data/kaggriculture/toad/phase1p_1786276268.jsonl`, log
  `/data/kaggriculture/toad/phase1p_run.log`, checkpoints `phase1p_%06d.pt`
  every 25 updates.

## What the arm is

```
.venv/bin/python -m kaggriculture.learn.scripts.toad_phase1 \
  --name phase1p \
  --resume /data/kaggriculture/toad/toad-phase1c3-clone-teacher_000200.pt \
  --channels 256 --teacher --teacher-kl-cost 0.005 \
  --value-warmup --econ-fraction 0.5
```

The reward field is the default `shaped`, so the learner reads Toad's five
components plus the capital term and nothing else. `reward_mean` equals
`shaped_mean` on the first record, which is the check that no other series
leaked in — on the margin arm those two differ.

Both things the ledger proved necessary for continuing from a competent policy
are on: value warmup (4,000 batches ≈ 7.6 updates at 528 batches/update) and
teacher KL at 0.005 — 0.001 cliffed within five updates and 0 collapsed within
four. Opponents are 12 mirror and 12 scripted `economic_policy` per round of 24,
training on our seat alone in the scripted envs. `boatlee_v14_policy` and
`kaito_v23_policy` are never constructed anywhere in the training path; `OPPONENT`
is `economic_policy` and nothing else, and kaito appears only in `gate.py`.

## The reward, term by term, and what each pays for that is free or reversible

The question is asked of every term before it gets a weight, because all four of
this project's pump defects share one shape: a term paying for something free or
reversible.

| term          | weight | measured over a reference season | free? undoable and redoable?                                                                                                                                                                                                      |
| ------------- | ------ | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `city`        | 1.0    | 0.100–0.110                      | No. Unlocking costs coins, planting costs a 10–100 coin seed. DIG removes a plant for −1 and replanting costs another seed, so the cycle is net zero and opens on the negative leg. Bounded by the board at ≤ 200/500 = 0.4.      |
| `unit`        | 0.5    | 0.008                            | No, and only because it is UNCLAMPED. `_end_of_day` wipes all hands every 24 turns, so the daily hiring telescopes to ≈0. Clamped it would pay 0.5 × 10 hands × 30 days / 500 = 0.3 — the largest pump latent in this reward set. |
| `research`    | 0.1    | 0.0016                           | No. Town-driven, monotone, capped at the shop count.                                                                                                                                                                              |
| `fuel`        | 0.005  | 0.0092–0.0098                    | **Yes — this is the one reversible term.** See below.                                                                                                                                                                             |
| `step`        | 0.005  | 0.00719 exactly                  | Action-independent constant; pays for nothing because it pays for everything.                                                                                                                                                     |
| `game_result` | 10.0   | ±0.020                           | Terminal rank on the last turn. Not a state.                                                                                                                                                                                      |
| `capital`     | 0.05   | 0.0015                           | No. Three engine rules and two inequalities, below.                                                                                                                                                                               |

Reference figures measured on 1.32.6, `economic_policy` mirror, seeds 7 and 11
(banks 92,565 and 13,686; both seats end holding 15 animals). Whole shaped
reward excluding capital: 0.146–0.157.

### `capital` at 0.05 — the term this arm adds, and why not Toad's 1.0

**Costs nothing?** No. `_commit_unit`'s `BUY_ANIMAL` charges
`ANIMALS[item]["cost"]` — a fixed 300 / 400 / 500, not a market quote that can be
walked down. Contrast the count the term deliberately is not: `BUILD_COOP` and
`BUILD_PASTURE` write a tile for zero coins on any empty square, so a structure
count would pay for ~100 free tiles a season. `_hosts_animal` tests for the
`"animal"` key, which the engine writes only via `_new_animal`.

**Undone and redone?** No, and three separate engine rules close it.

- `_process_market` quotes `SELL` only for `item in PRODUCTS`, and `ANIMALS` is
  disjoint from `PRODUCTS`. An animal can never become coins again.
- `DIG` returns early on a tile holding an animal ("Does NOT remove a placed
  animal").
- The one cycle the engine permits — PICKUP an animal out of the shed, PLACE it
  on its structure — moves it between `herd` and `placed`, and `capital` sums
  both, so the round trip is exactly zero. It is also unclamped, so the negative
  leg lands first and is not forgiven. Measured over a reference season, gross
  positive capital deltas are 29 against a net 15: that is the cycle showing up
  and cancelling.

Capital falls only when an animal escapes after two consecutive unfed days,
which destroys the asset with no refund.

**So the season total is bounded by coins irreversibly spent:**

```
capital <= 3000/300 + earned/300                    (startingMoney / cheapest animal)
reward  <= 0.05 * (10 + earned/300) / 500 = 0.0010 + earned * 3.3e-7
```

An agent that sells nothing all season caps at 0.0010, one twentieth of the
terminal `game_result` at 0.020. Every animal past the tenth has to be paid for
by running the whole production chain first.

**And acquiring must not out-pay operating**, which is what fixes the magnitude:

```
capital at 0.05 * 15 / 500 = 0.0015   15% of the produce it enabled, 1.0% of the shaped total
capital at 1.00 * 15 / 500 = 0.0300   3x every unit of produce harvested all season, 1.5x game_result
```

Requiring the term to stay under the produce it enables caps the weight at
`0.0098 * 500 / 15 = 0.33`. **Toad's published 1.0 for this slot fails that by
3x and is not reused**, which is a deviation from the brief's framing and is
named as one: the brief pointed at the weight-1.0 compounding slot, and the
measurement says that slot's number cannot carry an animal count in a reward with
no coin term to charge the purchase price.

The margin arm's 0.05 is kept, but **its justification does not transfer** and
was re-derived. There it was safe because the margin term charged the animal's
full 300–500 coins, six to nine times the credit. This arm has no margin and no
money term, so coins carry no reward at all and that structural guard is simply
absent.

PRE-REGISTERED: if this arm underperforms, the capital weight is the first knob
and 0.33 is its ceiling. Not something to tune mid-run.

### `fuel` is reversible, is vendored, and is left alone

`BUY_PRODUCT` (WHEAT and FERTILIZER only) raises shed stock; `SELL` lowers it and
the non-negative clamp forgives the fall. So buy → sell → buy is a genuine pump.
Measured on the engine, the wheat round-trip spread is 0–1 coin per unit
(`market_price(inv-1)` against `market_price(inv)` at realistic inventories), so
the coin cost is near zero — and coins carry no reward on this arm. At the
`maxMarketOrdersPerTurn` cap of 10 the ceiling is ~3,595 units over a season =
**0.036**, against the reference's honest 0.0092–0.0098 and a shaped total of
~0.15.

It is not changed. It is Toad's component at Toad's published weight, the brief
prescribes it verbatim, and unclamping it would punish selling — worse than the
disease. Instead it is made **detectable**: `bought_vs_econ` and `bought_mirror`
are logged from update 1, and the monitor's churn condition fires on volume
rising while realisation falls and bank stays flat. This is the honest residual
risk in the arm and it is stated rather than absorbed.

## Code change

`CAPITAL_WEIGHT` 1.0 → 0.05 in `toad_reward.py`, `capital_weight` added to the
wandb config so a run's reward is identifiable from the dashboard, and the fuel
clamp's comment moved onto the fuel line — it sat above the capital line and read
as licence to clamp it.

Four guards added to `tests/learn/test_toad_reward.py`, each observed red on its
named break and green restored:

- weight back to Toad's 1.0 → the sizing bound and the opening-float bound both fail
- capital delta clamped like fuel → the shed-to-tile cycle stops telescoping
- `_hosts_animal` counting structures → the free-build guard fails

`uv run pre-commit run -a` passes, full suite included.

## Launch discipline

`setsid` execs in place, so pid 2858665 is a session leader (`sid == pgid == pid`)
and the pid file holds the learner itself — verified by its argv being the module,
not a wrapper. The margin arm lost a run to a `pgrep` that matched its own shell;
the pid here was taken by enumerating `PPID == 1 && argv ~ .venv/bin/python &&
argv ~ toad_phase1`.

Resume proven before launch, not after: `_restore` on the c3 u200 checkpoint
returns update 200 / step 6,902,400 and a learning rate of 6.54576857e-05, which
is `1e-4 × decay(200)` exactly — a schedule reset would read 1e-4. Adam moments
present.

## First two records

|                               | u201       | u202       |
| ----------------------------- | ---------- | ---------- |
| `steps`                       | 6,928,284  | 6,954,168  |
| **`win_rate_vs_econ`**        | **0.0000** | **0.0000** |
| `win_rate_mirror`             | 0.375      | 0.417      |
| `bank_vs_econ`                | 4.67       | 2.25       |
| `bank_mirror`                 | 48.21      | 44.33      |
| `margin_mean_vs_econ`         | −137,731   | −147,790   |
| `final_capital_vs_econ`       | 1.50       | 1.50       |
| `units_sold_vs_econ`          | 24.3       | 24.7       |
| `mean_sale_price_vs_econ`     | 43.26      | 36.93      |
| `realisation_vs_econ`         | 0.903      | 0.892      |
| `bought_vs_econ`              | 22.5       | 22.0       |
| `shaped_mean` = `reward_mean` | 0.05814    | 0.05497    |
| `baseline` = `total`          | 0.00026    | 0.00117    |
| `vtrace_pg`                   | −0.0385    | −0.0754    |
| `illegal`                     | 0          | 0          |
| `warming` / `warmup_left`     | True/3604  | True/3208  |
| `lr`                          | 6.5e-05    | 6.5e-05    |

`reward_mean == shaped_mean` to five decimals: the learner is reading `shaped`
and no other series leaked in. `total == baseline` while `warming` is True: the
policy gradient is correctly excluded during value warmup. `illegal` is 0.

**These two records are bit-identical to the margin arm's u201 and u202** —
bank 4.67 / 2.25, margin −137,731 / −147,790, the same sale prices and volumes.
Same seeds, same c3 u200 checkpoint, same frozen actor, and the value head warms
alone for the first ~7.6 updates, so the two arms cannot diverge until the policy
gradient switches on around u208. That the numbers coincide exactly is the
cleanest available check that the reward field is the only difference between
this arm and the one it replaces.

`final_capital_vs_econ` of 1.5 puts the capital term at 1.5 × 0.05 / 500 =
1.5e-4 against a shaped reward of 0.058, i.e. 0.26%. At the reference's 15
animals it would be the designed 1.0%.

Cadence 72.7 s/update, so the remaining 507 updates to `TOTAL_STEPS` = 2e7 are
about 10.2 hours.

## Reads, pre-registered

- The number that matters is `bank_vs_econ` climbing by orders of magnitude. The
  reference banks ~73,000 on this engine; we open at ~5.
- Churn: `units_sold` up while `mean_sale_price` or `realisation` falls and bank
  does not move.
- Inertia: `baseline` toward ~0.001 with `vtrace_pg` near zero. **Caveat worth
  recording**: c3 ran the shaped reward at `baseline` 0.0002–0.003 throughout
  with `vtrace_pg` between −0.015 and +0.031, so a small baseline is normal for a
  reward whose episode total is ~0.06 and is not on its own diagnostic. Both
  conditions must hold together, over a window.
- No progress claim off fewer than ~100 updates. A previous arm's 26-update
  optimism (bank 4.8 → 114) was relayed upward and the full run read 69.6 → 68.9.
