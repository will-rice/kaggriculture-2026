# Task 4 report: privileged opponent-belief supervision

## Result

Implemented privileged opponent-private belief targets, stateful belief prediction
and feedback, masked supervised loss, production collector/Lightning wiring, and
stable family metrics without changing the disabled-feature control path.

## RED evidence

- `pytest tests/learn/test_toad_belief.py -v`: collection error because
  `encode_private_belief_target` did not exist.
- Target/data cycle: 1 failed, 2 passed because `Trajectory` had no
  `belief_targets` field.
- Model cycle: 4 failed because belief-only inputs entered the bare 4-D policy
  and recurrent output had no belief logits.
- Loss cycle: 3 failed because `belief` terms were absent, missing labels were
  accepted, and zero-valid behavior was not implemented.
- Mask-safety cycle: 1 failed because NaN payloads on invalid rows survived
  multiplication by a zero mask and made total loss NaN.
- Belief-only replay cycle: 1 failed because learner state replay was still
  conditioned on recurrence and passed `None` instead of the nonzero segment
  entry prior prediction.
- Broad regression cycle: 4 failed, 171 passed, 1 xfailed because optional
  belief names changed the frozen control segment/loss mappings. Separating
  optional `BELIEF_FIELDS` and enabled-only belief terms restored control
  parity.

## GREEN evidence

- Final focused gate:
  `pytest tests/learn/test_toad_belief.py tests/learn/test_encoding.py tests/learn/test_rollout.py tests/learn/test_toad_data.py tests/learn/test_toad_lightning.py tests/learn/test_toad_model.py tests/learn/test_toad_checkpoint.py tests/learn/test_toad_control_fixture.py tests/learn/test_toad_final_repairs.py -q`
  -> 176 passed, 6 deselected, 1 expected xfail.
- Scoped `ruff check` -> all checks passed.
- Scoped `ruff format --check` -> 10 files already formatted.
- Full-source `ty check` -> all checks passed.
- `git diff --check` -> passed.

## Files

- `src/kaggriculture/learn/encoding.py`: fixed-schema `BeliefTarget`, normalized
  family encoder, and target width.
- `src/kaggriculture/learn/rollout.py`: opposing-seat target capture, explicit
  validity, trajectory storage, and single policy-owned actor reset clock.
- `src/kaggriculture/learn/toad/config.py`: strict target-width validation and
  belief stage-gate activation.
- `src/kaggriculture/learn/toad/data.py`: optional label segmentation and unused
  legacy worker alias removal.
- `src/kaggriculture/learn/toad/model.py`: belief-only/recurrent belief head,
  differentiable prior-prediction feedback, terminal resets, and next state.
- `src/kaggriculture/learn/toad/lightning.py`: state/label replay, bootstrap
  exclusion, safe masked Smooth-L1, weighted total, family metrics, and strict
  stateful policy loading.
- `src/kaggriculture/learn/scripts/toad.py`: stateful belief worker selection.
- `tests/learn/test_toad_belief.py`: target, anti-leakage, capture, masking,
  feedback, reset, gradient, parity, and production-boundary coverage.
- `tests/learn/test_toad_model.py`: belief stage/width validation.
- `tests/learn/test_toad_final_repairs.py`: typed worker-field assertion after
  removal of the obsolete tuple alias.

## Concerns

- Repository-wide `ruff check .` still reports 535 pre-existing violations in
  unrelated legacy/vendored sources. Every Task 4-touched file passes Ruff.
- The focused gate retains the existing sale-metrics xfail and emits existing
  environment/Lightning warnings; neither is introduced by Task 4.
- Simulator collection always has complete private state, so reference-rollout
  validity is true. The explicit false mask remains supported for alternate or
  incomplete backends and is NaN-safe in learner loss.

## Fix round 1/5

### Review findings addressed

- Opponent-private encoding and storage now occur only when the learner is a
  `StatefulPolicy` with an active belief head. Bare control and recurrent-only
  rollouts retain `None` trajectory fields and emit no belief segment keys.
- Belief-only `PolicyState` now uses batch-aligned `(batch, 0, 0, 0)` hidden and
  cell tensors on the input device/dtype. Actor recording, unbatching,
  trajectory stacking, segmentation, learner batching, reset, and transfer
  preserve zero elements rather than allocating unused ConvLSTM maps.
- Stateful entry validation rejects nonempty belief-only spatial state while
  preserving Task 3's recurrent-only compatibility: the prior-belief width is
  strict only when an active belief head consumes it.
- The anti-leakage test now crosses `_decide`, the collector's actual policy
  input/target construction seam, and proves that changing only the opposing
  private observation changes the stored target without changing board,
  scalars, or positions.

### RED evidence

- Review regression selection:
  `pytest tests/learn/test_toad_belief.py -k 'opponent_private_label or nonbelief_rollout or zero_sized' -v`
  -> 4 failed, 15 deselected. `_decide` did not accept opposing target
  observations, both disabled topologies stored belief tensors, and the
  belief-only trajectory stored 38,400 dummy recurrent elements in the reduced
  two-turn fixture.
- Strict-state cycle:
  `pytest tests/learn/test_toad_belief.py -k rejects_nonempty -v`
  -> 1 failed, 19 deselected because belief-only forward silently accepted a
  nonempty hidden/cell map.
- First broad regression run -> 2 failed, 178 passed, 6 deselected, 1 expected
  xfail. Both failures showed the new exact prior-belief-width check reaching
  recurrent-only Task 3 fixtures, where the unused compatibility slot had
  width 9 rather than the configured inactive-head width. The check was
  narrowed to active belief heads; recurrent-only state remains batch- and
  metadata-validated.

### GREEN evidence

- Review regression selection -> 5 passed, 15 deselected.
- Complete belief suite -> 20 passed.
- The two recurrent-only compatibility reproductions plus the belief suite ->
  22 passed.
- Final focused belief/encoding/rollout/data/Lightning/model/checkpoint/control
  gate -> 180 passed, 6 deselected, 1 expected xfail in 72.82 seconds.
- Scoped Ruff and full-source `ty check src` -> all checks passed.
- `git diff --check` -> passed.

### Remaining concerns

- The pre-existing sale-metrics xfail and existing environment/Lightning
  warnings remain unchanged. No new runtime or static concern was found in
  this fix round.
