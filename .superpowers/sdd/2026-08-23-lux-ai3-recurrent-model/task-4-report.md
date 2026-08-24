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
