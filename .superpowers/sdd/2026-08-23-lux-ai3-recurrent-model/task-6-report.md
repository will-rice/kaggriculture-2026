# Task 6 report: padded local residual unit readouts

## Outcome

Implemented exact integer-indexed local patch extraction and the optional local
unit readout. `extract_unit_patches(features, positions, size)` pads the 10x10
feature map by the window radius, appends a separate out-of-bounds plane, uses
`torch.nn.functional.unfold`, and gathers centered windows by the existing flat
unit indices. Its output is `(B, units, C + 1, size, size)`.

`LocalUnitHead` shares one patch preprocessing convolution and the configured
SE residual blocks, then uses separate operation and quantity projections sized
from `UNIT_OPS` and `QUANTITIES`. `StatefulPolicy` applies this readout only when
`local_patch` is enabled and only after the shared trunk, recurrence, belief
prediction, and optional spatial transformer. Market and value readouts retain
their existing pooled/interaction paths. Disabled control still constructs the
bare `Policy` and remains covered by the exact fixture.

The declared maximum patch size is 9, the largest odd window no wider than the
10x10 board. This gives the supported padding a simple board-derived bound,
keeps the specified 7x7 default valid, and rejects 11 and larger at both config
and direct helper/module boundaries.

Adding `local_patch` to `uses_stateful_policy` makes both Lightning and the
typed reference worker construct `StatefulPolicy` and strictly load the exact
enabled state dictionary. With that production consumer complete, the final
local-patch stage gate was removed.

## RED evidence

- Initial patch selector could not collect because
  `extract_unit_patches` did not exist (`ImportError`), the intended missing
  feature.
- Config ceiling cycle: 10 cases passed and the new
  `local_patch_size=11` case failed because validation accepted it.
- Local routing/composition cycle: 5 cases passed and 4 failed. Local-only
  input fell through to bare `Policy` with a 5-D tensor, the combined model had
  no local parameters/gradients, and `local_head` did not exist at the
  post-transformer boundary.
- Runtime size-type cycle: 3 cases passed and `size=7.0` failed inside
  `functional.pad` with a `TypeError` instead of the required clear boundary
  error.

The first requested `uv run pytest` attempt could not create a temporary file
under the read-only shared uv cache. Retrying with a `/tmp` cache then could not
download `nvidia-curand` because network/DNS access is unavailable. All actual
RED/GREEN and final verification therefore used the repository's established
Python 3.11 virtual environment with this worktree's `src` on `PYTHONPATH`.

## GREEN evidence

- Exact patch extraction, four corners, indicator semantics, batches/units,
  and malformed input slice: `15 passed` before the later size-type case;
  final size-validation slice: `4 passed`.
- Local head projection/gradient slice: `3 passed`.
- Config dimension slice: `11 passed`.
- Local-only/all-feature routing and post-transformer slice: all selected
  cases passed after implementation.
- Padded placeholder position plus injected operation/quantity logit
  invariance: passed; total loss and every parameter gradient match, while the
  padded logit gradients are exactly zero.
- Belief isolation under padded local-position perturbation: passed; belief
  loss and belief-head gradients are exactly equal and the local head receives
  no belief-only gradient.
- Production integration selection across model, Lightning, worker, belief,
  and exact control: `50 passed, 65 deselected`.
- Post-refactor local/model integration selection: `62 passed`.
- Config regression suite: `14 passed`.
- Required full focused model/Lightning/belief/rollout/data/checkpoint/control
  result, touched-file Ruff, full-source ty, and diff evidence are recorded in
  the final verification section below.

## Files changed

- `src/kaggriculture/learn/toad/model.py`: validated unfold/gather extraction,
  local residual head, stateful selector, and post-feature unit routing.
- `src/kaggriculture/learn/toad/config.py`: board-derived maximum patch size,
  maximum validation, and final stage-gate removal.
- `tests/learn/test_toad_model.py`: centering, all corners, independent
  indicator, batch/unit, invalid input, independent projection, gradient,
  composition, ordering, and unchanged market/value coverage.
- `tests/learn/test_toad_lightning.py`: strict local-only production restore and
  padded position/logit total-loss plus full-gradient invariance.
- `tests/learn/test_toad_data.py`: strict local-only reference-worker rebuild.
- `tests/learn/test_toad_belief.py`: belief loss/gradient independence from a
  padded local position.
- `.superpowers/sdd/2026-08-23-lux-ai3-recurrent-model/task-6-report.md`.

## Final verification

- Required focused gate:
  `pytest tests/learn/test_toad_model.py tests/learn/test_toad_lightning.py tests/learn/test_toad_belief.py tests/learn/test_rollout.py tests/learn/test_toad_data.py tests/learn/test_toad_checkpoint.py tests/learn/test_toad_control_fixture.py -q`
  -> `151 passed, 1 xfailed, 18 warnings in 73.79s`.
- Touched-file `ruff check`: all checks passed.
- Touched-file `ruff format --check`: 6 files already formatted.
- Full-source `ty check src`: all checks passed.
- `git diff --check`: passed.

## Concerns

- The worktree-local uv environment cannot be freshly resolved without a
  writable populated cache or network access. Direct verification uses the
  repository's established Python 3.11 environment.
- The focused rollout gate retains its pre-existing expected xfail and existing
  NVML, Lightning pytree/data-loader, and scheduler-order warnings. No new
  runtime, numerical-control, masking, belief, recurrent-reset, or checkpoint
  concern was found.
