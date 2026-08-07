# Self-Play RL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the two-headed policy by self-play PPO until it beats the vendored kaito agent.

**Architecture:** An engine-derived legality mask, a value head on the existing trunk, rollout workers that play episodes against a sampled opponent pool and return trajectories, and a PPO update with GAE, entropy and a decaying KL penalty toward the behaviour-cloned checkpoint.

**Tech Stack:** PyTorch, `kaggle_environments`, `ProcessPoolExecutor`, Weights & Biases. Lightning only for `seed_everything`.

## Global Constraints

- Run everything with `uv run`. Add dependencies with `uv add`. Never `pip`, never bare `python`.
- `uv run pre-commit run -a` must pass before every commit. Fix type errors; never add ignore comments.
- Google-style docstrings on every public function, class and module. Absolute imports. Never `from __future__ import annotations`.
- `logging.info` / `logging.warning`, never `print()`. `tqdm` for long iterables. Required argparse arguments take no `--` prefix. Prefer CONSTANTS over CLI arguments. `seed_everything(SEED, workers=True)`.
- Always include the batch dimension. Prefer `nn.functional` unaliased, `.tile` over `.unsqueeze().expand()`, `.flatten` for collapsing dims, `.numpy(force=True)`. Never `einops.rearrange`.
- No defensive code, no unused code paths. Raise a concise error for unsupported cases.
- **Never `git add -A` or `git add .`** — stage the explicit paths your commit message names.
- **If a command has returned, act on its output.** Never end a turn waiting for a separate completion signal. Run verification in the foreground and read the result.
- **Every test guarding load-bearing behaviour must be observed to fail when that behaviour is broken.** Break it, watch it go red, restore, report what you broke.
- **Anything derived from the engine must be asserted against the engine**, by executing it — not against the rules documentation, which has been wrong three times here.
- Training code lives under `learn/`; the submitted agent must never import `wandb` or anything under `scripts/`.
- Do not mention Claude or AI assistance in commit messages.

## Context

Spec: `docs/superpowers/specs/2026-08-07-self-play-rl.md`.

Everything this project has tried is bounded above by its data — cloning copies a teacher, replay is a teacher, retrieval picks among teachers. Measured yesterday: retrieval over 190 routes scores _worse_ than replaying one fixed route, and our route memory scored 1452.3 against a blind single replay's 1457.1.

Throughput makes this viable: 1.29 s per 719-turn episode with trivial agents, 128M game steps an hour across 64 cores.

### Interfaces this plan consumes

`kaggriculture.learn.encoding`: `encode_board`, `encode_scalars`, `encode_positions`, `encode_units(action, units)`, `decode_units(logits, units)`, `encode_market`, `decode_market`, `unit_count`, `BOARD`, `TILE_PLANES` (48), `SCALARS` (64), `MAX_UNITS` (20), `UNIT_OPS` (22), `MARKET_SLOTS` (19), `QUANTITIES` (17), `IGNORE`.

`kaggriculture.learn.model`: `Policy.forward(board, scalars, positions) -> (unit_logits, market_logits)`, 10.2M parameters.

`kaggriculture.learn.play`: `CHECKPOINT`, the behaviour-cloned weights.

Opponents: `kaggriculture.kaito_policy.agent`, `kaggriculture.economic_policy.agent`, `kaggriculture.routes.play`.

## File Structure

| File                                                         | Responsibility                                                                |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------- |
| `src/kaggriculture/learn/mask.py`                            | **Create.** Legal-action masks, derived from and asserted against the engine. |
| `src/kaggriculture/learn/model.py`                           | **Modify.** Add a value head.                                                 |
| `src/kaggriculture/learn/rollout.py`                         | **Create.** Play one episode, return a trajectory.                            |
| `src/kaggriculture/learn/ppo.py`                             | **Create.** GAE, clipped losses, entropy, teacher KL.                         |
| `src/kaggriculture/learn/scripts/selfplay.py`                | **Create.** The loop: workers, opponent pool, updates, tracking.              |
| `tests/learn/test_mask.py`, `test_rollout.py`, `test_ppo.py` | **Create.**                                                                   |

---

### Task 1: The legality mask

**Files:**

- Create: `src/kaggriculture/learn/mask.py`
- Test: `tests/learn/test_mask.py`

**Interfaces:**

- Produces: `unit_mask(observation, seat) -> torch.Tensor` of shape `(1, MAX_UNITS, len(UNIT_OPS))`, bool; `market_mask(observation, seat) -> torch.Tensor` of shape `(1, len(MARKET_SLOTS) + 2, len(QUANTITIES))`, bool.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_unit_may_always_pass() -> None:
    """PASS is the one op the engine never refuses; a mask that forbids it deadlocks."""
    mask = unit_mask(empty_observation(), seat=0)

    assert mask[0, 0, UNIT_OPS.index("PASS")]


def test_planting_needs_a_seed() -> None:
    """PLANT consumes stock the engine checks before it does anything."""
    without = empty_observation()
    with_seed = empty_observation()
    with_seed["private"]["seeds"]["MELON"] = 3

    assert not unit_mask(without, 0)[0, 0, UNIT_OPS.index("PLANT:MELON")]
    assert unit_mask(with_seed, 0)[0, 0, UNIT_OPS.index("PLANT:MELON")]


def test_selling_needs_stock_in_the_shed() -> None:
    """SELL draws from the shed; the engine refuses it empty."""
    empty = empty_observation()
    stocked = empty_observation()
    stocked["private"]["shed"]["WHEAT"] = 5
    slot = MARKET_SLOTS.index(("SELL", "WHEAT"))

    assert not market_mask(empty, 0)[0, slot, 1]
    assert market_mask(stocked, 0)[0, slot, 1]


def test_a_quantity_beyond_the_shed_is_masked() -> None:
    """Bucket 12 against five sacks is an order the engine part-fills and wastes."""
    observation = empty_observation()
    observation["private"]["shed"]["WHEAT"] = 5
    slot = MARKET_SLOTS.index(("SELL", "WHEAT"))

    mask = market_mask(observation, 0)

    assert mask[0, slot, QUANTITIES.index(4)]
    assert not mask[0, slot, QUANTITIES.index(12)]
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/learn/test_mask.py -q`
Expected: FAIL on `ImportError`.

- [ ] **Step 3: Implement, reading the engine**

Read `_apply_unit_action` and `_process_market` and mirror their guards. Do not work from the rules documentation: it omitted `DROP`, it overstated `BUY_PRODUCT`'s item set, and it says nothing about `step` being written for one seat only. Every one of those cost this project a debugging cycle.

Bucket 0 of every market slot is always legal — "trade nothing" is always available.

- [ ] **Step 4: Run**

Run: `uv run pytest tests/learn/test_mask.py -q`
Expected: PASS.

- [ ] **Step 5: Assert the mask against the engine itself — this is the task**

Write a `slow` test that, on real observations sampled from the corpus, enumerates every `(unit, op)` pair, applies it through a fresh copy of the engine, and records whether the engine accepted it. Then assert:

- the mask never permits an op the engine refuses (too permissive wastes budget), and
- the mask never forbids one the engine accepts (**too strict removes a winning move and nothing ever raises**).

Report both counts. If they are not both zero, report the disagreements rather than widening the mask to match.

- [ ] **Step 6: Commit**

```bash
git add src/kaggriculture/learn/mask.py tests/learn/test_mask.py
git commit -m "feat: legal-action masks derived from the engine's own guards"
```

---

### Task 2: A value head

**Files:**

- Modify: `src/kaggriculture/learn/model.py`
- Test: `tests/learn/test_model.py`

**Interfaces:**

- Produces: `Policy.forward(board, scalars, positions) -> (unit_logits, market_logits, value)` where `value` is `(batch,)`.

PPO needs `V(s)`, and the network has no value head. This is the smallest task in the plan and the loop cannot run without it.

- [ ] **Step 1: Write the failing tests**

```python
def test_forward_returns_a_value_per_state() -> None:
    """PPO's advantage is r + gamma*V(s') - V(s); without V there is no advantage."""
    units, market, value = Policy()(
        torch.zeros(2, TILE_PLANES, BOARD, BOARD), torch.zeros(2, SCALARS), _positions(2)
    )

    assert value.shape == (2,)


def test_the_value_head_reads_the_whole_board() -> None:
    """How well we are doing is a property of the position, not of one tile."""
    torch.manual_seed(0)
    model = Policy().eval()
    board = torch.randn(1, TILE_PLANES, BOARD, BOARD)
    elsewhere = board.clone()
    elsewhere[0, :, 9, 9] += 5.0

    with torch.no_grad():
        _, _, before = model(board, torch.zeros(1, SCALARS), _positions(1))
        _, _, after = model(elsewhere, torch.zeros(1, SCALARS), _positions(1))

    assert not torch.equal(before, after)
```

- [ ] **Step 2: Run and watch them fail**

Expected: FAIL — `forward` returns two values.

- [ ] **Step 3: Implement**

Pool the trunk and project to a scalar, as the market head does. Update every call site: `train.py`, `play.py`, `test_model.py`, `test_train.py`.

- [ ] **Step 4: Run the full suite, then commit**

The behaviour-cloned checkpoint predates the value head, so loading it must not fail. Decide how — `strict=False`, or initialise the head separately — and say which in the docstring, because a silently-unloaded trunk would train from noise while looking fine.

```bash
git add src/kaggriculture/learn/model.py tests/learn/test_model.py
git commit -m "feat: a value head, so advantages have something to subtract"
```

---

### Task 3: Rollouts

**Files:**

- Create: `src/kaggriculture/learn/rollout.py`
- Test: `tests/learn/test_rollout.py`

**Interfaces:**

- Produces: `Trajectory` (board, scalars, positions, unit actions, market actions, masks, log-probs, values, rewards, dones), `rollout(policy, opponent, seed) -> Trajectory`.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_trajectory_covers_every_acting_turn() -> None:
    """719 decisions; a short trajectory silently truncates the season."""
    trajectory = rollout(_untrained(), "starter", seed=0)

    assert len(trajectory.rewards) == 719


def test_reward_is_the_change_in_bank_differential() -> None:
    """Dense, sums to the terminal margin, and is the win condition itself.

    Terminal bank alone is one scalar after 719 decisions. This is the same
    quantity, delivered per turn.
    """
    trajectory = rollout(_untrained(), "starter", seed=0)

    assert trajectory.rewards.sum() == pytest.approx(trajectory.final_margin, abs=1.0)


def test_sampled_actions_are_always_legal() -> None:
    """The mask is applied before sampling, not checked after."""
    trajectory = rollout(_untrained(), "starter", seed=0)

    assert trajectory.illegal == 0
```

- [ ] **Step 2: Run, watch fail, implement**

Sample from the masked logits. Set masked logits to `-inf` before the softmax rather than zeroing probabilities after, so the log-probs PPO stores are the log-probs of the distribution actually sampled from.

Record the mask alongside the action: the update must re-apply the same mask, or the ratio is computed against a different distribution than the one that acted.

- [ ] **Step 3: Measure throughput with the network in the loop**

The spec's 128M steps/hour is with trivial agents. Report episodes/hour with the policy actually running, single-process and across the pool. **The plan's run length depends on this number, so measure it before Task 5 rather than assuming it.**

- [ ] **Step 4: Commit**

```bash
git add src/kaggriculture/learn/rollout.py tests/learn/test_rollout.py
git commit -m "feat: self-play rollouts with masked sampling"
```

---

### Task 4: The PPO update

**Files:**

- Create: `src/kaggriculture/learn/ppo.py`
- Test: `tests/learn/test_ppo.py`

**Interfaces:**

- Produces: `advantages(rewards, values, gamma, lam)`, `policy_loss(ratio, advantage, clip)`, `update(policy, teacher, batch, config) -> dict[str, float]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_advantages_sum_toward_the_return() -> None:
    """GAE with lambda=1 and gamma=1 is the Monte Carlo advantage."""
    rewards = torch.ones(5)
    values = torch.zeros(6)

    computed = advantages(rewards, values, gamma=1.0, lam=1.0)

    assert computed[0] == pytest.approx(5.0)


def test_the_clip_bounds_the_step() -> None:
    """PPO's whole safety property: one batch cannot move the policy far."""
    ratio = torch.tensor([10.0])
    advantage = torch.tensor([1.0])

    loss = policy_loss(ratio, advantage, clip=0.2)

    assert loss == pytest.approx(-1.2)


def test_the_teacher_penalty_falls_to_zero() -> None:
    """It exists to survive the first updates, not to pin us to a weak clone."""
    assert kl_weight(step=0) > kl_weight(step=1000) > kl_weight(step=10_000)
    assert kl_weight(step=10_000) == pytest.approx(0.0, abs=1e-3)
```

- [ ] **Step 2: Run, watch fail, implement**

Both heads contribute policy loss. The unit head's loss must ignore padded slots exactly as `unit_loss` does; the market head masks nothing, because every slot is a real decision.

- [ ] **Step 3: Prove the guards discriminate**

Remove the clip and watch the clip test fail. Fix `kl_weight` to a constant and watch its test fail.

- [ ] **Step 4: Commit**

```bash
git add src/kaggriculture/learn/ppo.py tests/learn/test_ppo.py
git commit -m "feat: clipped PPO updates with a decaying teacher penalty"
```

---

### Task 5: The loop, and the gate

**Files:**

- Create: `src/kaggriculture/learn/scripts/selfplay.py`

- [ ] **Step 1: Write the loop**

Workers play episodes against an opponent sampled from the pool — the vendored kaito agent excluded, held out as the gate — and return trajectories. The learner updates and periodically adds its own checkpoint to the pool.

Log per iteration: mean bank, mean margin, win rate against each fixed opponent, illegal-action rate, entropy, KL to the teacher, and value loss. **Illegal-action rate is the fastest signal that masking is wrong; if it is not zero, stop and fix Task 1 rather than training through it.**

- [ ] **Step 2: Run the short comparison the spec asks for**

The BC checkpoint banked 0 in play, so initialising from it may be worse than starting fresh. Run both for a small equal budget and report both curves. Do not assume.

- [ ] **Step 3: Train**

Length set by Task 3's measured throughput. Commit before training; tracking refuses a dirty tree.

- [ ] **Step 4: Gate**

Report bank and win rate with Wilson intervals against, in order: the BC checkpoint, `economic_policy`, the fixed best route, and vendored kaito. Report the rung reached and stop — do not tune until a number looks better.

- [ ] **Step 5: Repoint `main.py` only if it beats kaito**

Otherwise leave it and say so.

---

## Self-Review

**Spec coverage.** Masking derived from and asserted against the engine — Task 1, whose Step 5 is the assertion. Dense differential reward — Task 3. Opponent pool with kaito held out — Task 5. BC initialisation with decaying teacher KL — Tasks 2 and 4. The four-rung gate and the illegal-action signal — Task 5. Throughput remeasured with the network in the loop — Task 3, Step 3, before any run length is chosen.

**Type consistency.** `Policy.forward` returns a three-tuple from Task 2 onward, and Tasks 3, 4 and 5 unpack it; every existing call site is updated in Task 2. Masks are bool tensors of the same shape as their logits, produced in Task 1 and consumed in Tasks 3 and 4.

**The value head is the addition the spec missed.** PPO cannot compute an advantage without it, and it is not in the network today.

**Risks carried from the spec, unsolved here.** Self-play collapse, a differential reward that rewards neutralising as much as outproducing, silent masking bugs in the too-strict direction, and a BC checkpoint that may be a worse starting point than noise. Each is measured rather than assumed: the held-out gate, absolute bank logged alongside margin, the engine-asserted mask, and Step 2's paired short run.
