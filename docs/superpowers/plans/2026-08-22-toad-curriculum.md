# Toad Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the five-phase Toad Brigade curriculum faithfully, instead of the misconfigured phase 1 every prior arm ran.

**Architecture:** Almost everything the recipe needs already exists — UPGO, V-trace, TD(lambda), the shaped reward with its 10x terminal term, and `VALUE_WARMUP_BATCHES = 4000` are all implemented and correct. What is missing is the ability to _vary by phase_: the teacher is a boolean welded to a behaviour clone, the block count and lambda are constants, and there is no sparse reward and no runner. Tasks 1-4 unweld those knobs; Task 5 adds the runner; Task 6 runs phase 1 against its gate.

**Tech Stack:** Python 3.11, PyTorch, `kaggle-environments` 1.32.7, pytest, `uv`, wandb.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-22-toad-curriculum-design.md`. Recipe numbers come from `docs/research/2026-08-07-fable-rl-grounding.md` §A.3 and are **not** to be re-derived or "improved".
- **Faithfulness over taste.** Where the recipe specifies a value, use it even if another looks better. Deviations are permitted but must be named in the run log and in `docs/experiments/2026-08-07-rl-ledger.md`, never absorbed silently.
- Engine is `kaggle-environments>=1.32.7`; do not upgrade it.
- Full default suite stays green; the reference engine is the authority and its manifest is never re-recorded to fit.
- Python 3.11; Google-style docstrings with `Args:`/`Returns:`; absolute imports; no `from __future__ import annotations`; prefer `nn.functional`; `.tile` over `.unsqueeze(1).expand`; `.numpy(force=True)`; `.flatten` for collapsing dims; `logging.info` not `print`.
- Required argparse args take no `--` prefix; optional ones do. Keep module CONSTANTS as flag defaults — the constant stays the single source of truth.
- Never `git add -A`. **Never `pre-commit run -a`** — pre-existing import-sort debt pulls unrelated files into the commit.
- Commit with `git commit -F <file>`, naming paths explicitly on the commit; message explains **why, not what**; **no Claude reference of any kind and no trailers.**
- The pre-commit hook runs the full suite (~9 min): **1800s foreground timeout**, foreground, never edit while it runs, never pipe a needed exit code through `tail`/`head`.
- Never end a turn waiting for a background signal. Grep for callers rather than trusting a task's file list — every task in the last plan found sites its brief had not named.

---

### Task 1: The teacher is a checkpoint, not a boolean

The recipe's teachers are "always the pipeline's own earlier, smaller checkpoints". Ours is `--teacher` (store_true) hardwired to the behaviour clone at `kaggriculture.learn.CHECKPOINT`, which is both the wrong artifact and unloadable since the action space widened.

**Files:**

- Modify: `src/kaggriculture/learn/scripts/toad_phase1.py` — the `--teacher` declaration (~:376), `_teacher()` (~:940), `_warm_start()` (~:1062), and the logged config (~:884)
- Test: `tests/learn/test_toad_runner.py`

**Interfaces:**

- Produces: `--teacher PATH` (optional, default `None` = no teacher, which is phase 1's setting). `_teacher(arguments, device) -> Teacher | None` unchanged in shape, but loads the named path. `Teacher.quantity` keeps its current meaning — computed from the loaded checkpoint's missing keys.
- The `--clone-init` flag keeps its own separate meaning; do not merge them. Warm-starting weights and anchoring a KL are different decisions and the recipe uses them in different phases.

- [ ] **Step 1: Write the failing tests**

```python
def test_no_teacher_flag_means_no_teacher() -> None:
    """Phase 1 runs teacher-free; the recipe sets teacher_kl_cost to 0.

    Every prior arm passed --teacher in phase 1 at phase 2's cost, anchoring a
    from-random policy to a behaviour clone the recipe uses nowhere. The
    default must be no teacher at all, not a clone.
    """
    # Parse an argv with no --teacher; assert _teacher(...) returns None.

def test_teacher_loads_the_named_checkpoint() -> None:
    """A phase names its teacher; phases 2+ each anchor to a different one."""
    # Write a real Policy state_dict to a tmp path, pass --teacher <path>,
    # assert the returned Teacher's parameters equal that checkpoint's,
    # not CHECKPOINT's.
```

Write both fully against the real `Policy` and a real `state_dict()` — no mocks. Read the file's existing fixtures first and follow them.

- [ ] **Step 2: Run to verify they fail** — the flag is a boolean today.
- [ ] **Step 3: Implement.** `--teacher` takes `type=Path, default=None`. `_teacher` returns `None` when it is `None`, else loads that path via `load_policy_weights`. Delete the help text about "the clone"; say what it now does. The logged config records the teacher's path, or `None`.
- [ ] **Step 4: Full suite green.** Grep for every reader of `arguments.teacher` — it is a truthiness test today and a `Path | None` now.
- [ ] **Step 5: Mutation check** — make `_teacher` ignore its path and load `CHECKPOINT`; confirm `test_teacher_loads_the_named_checkpoint` fails; restore. Record verbatim.
- [ ] **Step 6: Commit.**

---

### Task 2: A sparse +/-1 terminal reward

Phases 2 onward use `GameResultReward`: **+1 terminal for a win, -1 for a loss, zero-sum, nothing else.** The existing `margin` field is _not_ this — its own docstring says it is the win condition "decomposed onto the turns that produced it, plus a small signed own-bank term and Toad's terminal rank". Decomposing the terminal signal onto turns is the opposite of sparse.

**Files:**

- Modify: `src/kaggriculture/learn/toad_reward.py` (add the series), `src/kaggriculture/learn/rollout.py` (record it on `Trajectory`, with its docstring entry), `src/kaggriculture/learn/scripts/toad_phase1.py` (`_field`, and a `--sparse` flag)
- Test: `tests/learn/test_toad_reward.py`

**Interfaces:**

- Produces: `Trajectory.sparse` of shape `(turns,)`, zero on every turn except the last, which carries `+1.0` for a win, `-1.0` for a loss, and `0.0` for a draw. `_field` returns `"sparse"` when `--sparse` is passed.

- [ ] **Step 1: Write the failing test**

```python
def test_sparse_is_zero_everywhere_but_the_final_turn() -> None:
    """Phases 2+ train on the game result alone.

    Sparse means sparse: a non-zero mid-episode entry would be a shaped
    reward wearing the name, and the phase boundary would stop meaning what
    the recipe says it means.
    """
    # Play a short real episode; assert sparse[:-1] are all exactly 0.0,
    # sparse[-1] is +1.0 when our terminal bank exceeds theirs, -1.0 when it
    # does not, 0.0 when equal. Assert it sums to exactly that same value.
```

- [ ] **Step 2: Run to verify it fails** (`Trajectory` has no `sparse`).
- [ ] **Step 3: Implement**, mirroring how `margin` and `shaped` are built and recorded so the arm remains "a choice of field rather than a re-run".
- [ ] **Step 4: Full suite green**, including whatever asserts `ACTED_FIELDS` or the recorded-field set.
- [ ] **Step 5: Mutation check** — emit `+1/-1` on every turn instead of the last; confirm the test fails; restore. Record verbatim.
- [ ] **Step 6: Commit.**

---

### Task 3: The remaining phase knobs

`BLOCKS = 8` is a constant but phases 3-5 need 16 and 24. `lmb` is already a parameter of `toad_loss.losses` defaulting to `LMB` but is not reachable from the command line, and phase 5 needs 0.9. `--value-warmup` is a boolean while the recipe specifies a batch count (`VALUE_WARMUP_BATCHES = 4000`, already correct).

**Files:**

- Modify: `src/kaggriculture/learn/scripts/toad_phase1.py` (`BLOCKS` ~:112, the `--value-warmup` declaration ~:412, `warmup_left` ~:505, the `losses(...)` call site, the logged config)
- Test: `tests/learn/test_toad_runner.py`

**Interfaces:**

- Produces: `--blocks` (default `BLOCKS`), `--lmb` (default `LMB`, named to match `toad_loss`'s own parameter rather than shadowing the Python keyword), `--value-warmup-batches` (default `VALUE_WARMUP_BATCHES`, `0` meaning none). `--value-warmup` is removed; anything passing it must be updated.

- [ ] **Step 1: Write the failing tests** — one per flag, each asserting the value reaches its _consumer_, not the namespace: `--blocks 16` builds a 16-block `Policy` (count the residual blocks); `--lmb 0.9` reaches `toad_loss.losses`; `--value-warmup-batches 3` runs exactly three `baseline_only` updates and then stops. Parsing a flag and never applying it is the bug these exist to catch.
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement.** Keep the constants as defaults.
- [ ] **Step 4: Full suite green.**
- [ ] **Step 5: Mutation check** — hardcode `BLOCKS` at the `Policy` construction; confirm the blocks test fails; restore. Record verbatim.
- [ ] **Step 6: Commit.**

---

### Task 4: The phase runner

A phase boundary should be a config, not a remembered command line. This is also what stops phase 2 from silently inheriting phase 1's teacher setting, which is the exact error this whole plan exists to correct.

**Files:**

- Create: `src/kaggriculture/learn/scripts/curriculum.py`
- Test: `tests/learn/test_curriculum.py`

**Interfaces:**

- Produces: `PHASES: tuple[Phase, ...]` where `Phase` is a frozen dataclass carrying `name, blocks, steps, reward, teacher_kl_cost, lr, entropy_cost, lmb, teacher_from`. `teacher_from` names the phase whose checkpoint teaches this one, or `None` for phase 1. CLI: `curriculum <phase-name>` runs one phase; the runner resolves `teacher_from` to that phase's checkpoint path on disk and refuses to start if it is absent.
- The five phases, verbatim from the recipe:

| name     | blocks | steps | reward | teacher_kl | lr   | entropy | lmb | teacher_from |
| -------- | ------ | ----- | ------ | ---------- | ---- | ------- | --- | ------------ |
| `phase1` | 8      | 2e7   | shaped | 0.0        | 1e-4 | 1e-3    | 0.8 | None         |
| `phase2` | 8      | 1e7   | sparse | 0.005      | 1e-4 | 1e-3    | 0.8 | `phase1`     |
| `phase3` | 16     | 2e7   | sparse | 0.01       | 5e-5 | 2e-4    | 0.8 | `phase1`     |
| `phase4` | 16     | 2e7   | sparse | 0.001      | 5e-5 | 2e-4    | 0.8 | `phase1`     |
| `phase5` | 24     | 2e7   | sparse | 0.005      | 5e-5 | 2e-4    | 0.9 | `phase3`     |

Note phases 3 and 4 are taught by the frozen **8-block** net and phase 5 by the **16-block** one — the recipe's teachers are smaller nets, not the immediately preceding checkpoint. Phase 4 continues phase 3's weights while keeping phase 3's teacher; only the KL cost changes.

- [ ] **Step 1: Write the failing test**

```python
def test_the_phase_table_matches_the_recipe() -> None:
    """The five phases are transcribed from Toad's config YAMLs.

    This test is the transcription's proof. Every prior arm ran phase 1 with
    phase 2's teacher cost, and nothing caught it, because the numbers lived
    in a command line rather than in a table anything could check.
    """
    # Assert each field of each phase against the literal table above.

def test_phase_one_has_no_teacher() -> None:
    """The single most consequential number in this plan."""
    # PHASES[0].teacher_kl_cost == 0.0 and PHASES[0].teacher_from is None.

def test_a_phase_refuses_to_start_without_its_teacher() -> None:
    """A missing teacher checkpoint must fail loudly, not train unanchored."""
    # Ask the runner for phase2 with no phase1 checkpoint on disk; expect a
    # raise naming the phase and the path it wanted.
```

- [ ] **Step 2: Run to verify they fail.** **Step 3: Implement** the dataclass, the table, and a `main()` that translates a `Phase` into the flags Tasks 1-3 exposed and invokes the existing training entry point. Do not duplicate the training loop.
- [ ] **Step 4: Full suite green. Step 5:** mutation check — change one table value (e.g. phase 1's `teacher_kl_cost` to 0.005, the historical error); confirm `test_the_phase_table_matches_the_recipe` and `test_phase_one_has_no_teacher` both fail; restore. Record verbatim.
- [ ] **Step 6: Commit.**

---

### Task 5: Rename, and record what changed

`toad_phase1.py` runs a curriculum now, and its name is part of why five phases read as one.

**Files:**

- Rename: `src/kaggriculture/learn/scripts/toad_phase1.py` -> `src/kaggriculture/learn/scripts/toad.py` (`git mv`, so history follows)
- Modify: every importer (grep — `curriculum.py`, tests, `docs/`), and `docs/experiments/2026-08-07-rl-ledger.md`
- Test: existing suite

- [ ] **Step 1:** `git mv`, then grep for `toad_phase1` across `src`, `tests`, `scripts`, `docs` and update every hit. **Step 2:** full suite green.
- [ ] **Step 3:** append a ledger entry stating plainly what was found — that arms A-T ran phase 1 of five with phase 2's teacher cost, against a behaviour clone the recipe forbids, at 256 channels rather than 128 — and that the curriculum now exists. Name the deviations from §5 of the spec, including the unhedged one: **no phase of this curriculum teaches trading.**
- [ ] **Step 4: Commit.**

---

### Task 6: Run phase 1, faithfully, and gate it

**Not a code task.** This runs the thing and reports.

- [ ] **Step 1:** launch `curriculum phase1` — 8 blocks, 128 channels, shaped reward, **no teacher**, lr 1e-4, entropy 1e-3, lmb 0.8, 2e7 steps (about six hours on this box).
- [ ] **Step 2:** while it runs, confirm from the run's own logged config that `teacher_kl_cost` is `0.0` and no teacher path is set. If either is wrong, stop: the plan has failed at the step it exists for.
- [ ] **Step 3:** at completion, report mean terminal bank against `economic_policy`, `objective/win_rate_vs_econ`, and the shaped-return trajectory.
- [ ] **Step 4: The gate.** Mean terminal bank must exceed **36,189** — the best figure any arm reached under the misconfigured phase 1, recorded in the 2026-08-07 ledger gate. Above it, proceed to phase 2. Below it, **stop and diagnose**: a faithful phase 1 that cannot beat the unfaithful one means the diagnosis in the spec's §1 is wrong, and phases 2-5 would be built on it.
- [ ] **Step 5:** record the outcome in the ledger either way, with the numbers.

---

## After the plan

Phases 2-5 run under the gates in the spec's §4, the sharpest being phase 2 -> 3: **win rate against `economic_policy` must exceed 0.000**, a rung eleven arms never cleared. Shipping still goes through the frontier-hardened holdout gate, unchanged.
