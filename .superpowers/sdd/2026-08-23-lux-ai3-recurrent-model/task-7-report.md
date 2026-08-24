# Task 7 report: recurrent-model integration gate

## Outcome

Added a deterministic 32-turn, two-segment synthetic training fixture with all
implemented optional architecture paths enabled at small CPU widths:
ConvLSTM, belief prediction/feedback/loss, spatial transformer, interaction
value, and local residual patches. Turn 15 is terminal and turn 16 begins the
next episode. The fixture records state on the actor clock, uses every unit as
an active transfer decision, enables every operation/quantity/market mask, and
provides valid, nonconstant belief targets.

The integration gate exercises production `compute_loss` and two Lightning
automatic-optimization batches. It proves finite losses; finite, nonzero
gradients; and parameter updates independently for the trunk stem/scalar path/
residual blocks, market head, recurrent gates and merge, belief feedback and
head, transformer position/attention/MLP, interaction query/global/attention/
MLP/value projections, local preprocessing/residual blocks, and both local
operation and quantity projections.

Actor-clock reset evidence is explicit: hidden, cell, and prior belief are all
nonzero both before and after the observation that produces terminal action 15;
the input state seen by observation 16 is exactly zero. The segmenter's entry
state at 16 equals that actor input state, and a production `compute_loss`
forward consumes the same tensors.

Added a real all-feature boundary resume through `ReferenceRoundSource`,
`ToadDataModule`, Lightning automatic optimization, `ActorSyncCallback`, and
`BoundaryCheckpoint`. The authoritative checkpoint is written after the first
completed two-batch collection boundary. It restores the learner, Adam,
scheduler, logical counters, a nonzero remaining warmup count, DataModule game
stream and RNG, published actor snapshot, full serialized/fingerprinted config,
and present teacher metadata. The resumed next game ID, zero episode-start
hidden/cell/prior tensors, and exactly one next optimizer update match the
uninterrupted same-world-size run. No mid-episode queue or recurrent actor state
was added to the checkpoint contract.

Full-enabled Lightning policy extraction now has explicit coverage for exact
strict loading, a missing enabled weight, and control-only weights. Every new
model configuration field is asserted present in JSON serialization, survives
round-trip validation, and individually changes the structural fingerprint.
The full all-feature Lightning consumer proves that no accepted optional model
gate is open without a production consumer. The permanent disabled control
fixture remains exact.

The parked `LocalUnitHead` diagnostic was naturally covered. The head now
stores its configured channel width and raises a clear boundary `ValueError`
before patch extraction/convolution when a standalone caller supplies a
different feature width.

## RED evidence

- Command:
  `pytest tests/learn/test_toad_model.py::test_local_unit_head_rejects_a_feature_channel_mismatch tests/learn/test_toad_model_integration.py::test_two_segment_training_resets_and_updates_every_component -vv`
  -> `1 failed, 1 passed`. The new all-feature integration guarantee passed on
  base `e911947`; the channel-mismatch regression failed exactly because
  `LocalUnitHead` exposed PyTorch's low-level convolution `RuntimeError` rather
  than the required boundary `ValueError`.
- The strict full-enabled extraction, serialization/fingerprint matrix, and
  production boundary-resume tests all passed when first run against base
  production. They therefore exposed no production checkpoint defect and no
  checkpoint implementation was changed.
- Full-source `ty check` initially found eight new helper annotation issues and
  one existing intentional runtime-invalid size call in the now-touched local
  patch test. These were test typing issues, not runtime failures.

The initial requested `uv run --python 3.11` probe attempted to rebuild the
worktree-local environment and could not download `nvidia-cublas` because
network/DNS access is unavailable. All test and quality evidence below uses the
repository's provisioned Python 3.11.12 environment with this worktree's `src`
on `PYTHONPATH`. The all-files hook later ran successfully through `uv` by
pinning `UV_PROJECT_ENVIRONMENT` to that provisioned environment.

## GREEN evidence

- Added the minimal production check in `LocalUnitHead.forward`; rerunning the
  original RED selector -> `2 passed`.
- Full-enabled strict extraction, fingerprint, and production resume selector
  -> `3 passed`.
- After splitting the integration assertions down to the actual attention,
  MLP, query-token, local-preprocessing, and local-block parameter groups, the
  all-feature training test still passed.
- After final typing-only test edits, the fresh integration/checkpoint/control
  selector -> `9 passed`.
- The existing actor/learner replay and disabled numerical-control tests remain
  included in the recurrent-focused and repository-wide gates below.

## Files changed

- `tests/learn/test_toad_model_integration.py`: deterministic all-feature actor
  trajectory, active-head batches, component gradient/update gate, reset and
  segment-entry agreement, and optional-field serialization/fingerprint gate.
- `tests/learn/test_toad_checkpoint.py`: strict all-feature policy extraction
  and authoritative recurrent boundary checkpoint/resume integration.
- `tests/learn/test_toad_model.py`: clear standalone local-head channel mismatch
  regression and narrow type suppression for an intentional invalid float-size
  runtime test.
- `src/kaggriculture/learn/toad/model.py`: early `LocalUnitHead` channel-width
  validation.
- `.superpowers/sdd/2026-08-23-lux-ai3-recurrent-model/task-7-report.md`.

No change was needed in `tests/learn/test_toad_lightning.py`: its existing
production loss-done shift and actor/learner boundary replay proofs remain
green, while the new all-feature composition belongs in the dedicated Task 7
integration file.

## Final verification

- Full recurrent-focused gate:
  `pytest tests/learn/test_toad_*.py tests/learn/test_rollout.py -q`
  -> `278 passed, 1 deselected, 1 xfailed, 45 warnings in 82.32s`.
- Repository-wide pytest:
  `pytest -q`
  -> `935 passed, 5 skipped, 52 deselected, 1 xfailed, 50 warnings in 701.33s`.
- Fresh post-typing affected gate:
  integration file, strict extraction, boundary resume, local mismatch, and
  disabled control -> `9 passed, 11 warnings in 8.65s`.
- Touched-file `ruff check`: all checks passed.
- Touched-file `ruff format --check`: all files formatted.
- Full-source `ty check`: all checks passed.
- `git diff --check`: passed.
- All-files `pre-commit run -a`: AST, EOF, whitespace, merge-conflict,
  Prettier, Ruff format, ty, and full pytest passed. Ruff itself fixed and then
  reported zero remaining errors, but the hook returned nonzero because it
  mechanically reordered imports in the two known unrelated baseline files
  `src/kaggriculture/learn/scripts/selfplay.py` and
  `src/kaggriculture/scripts/tracking.py`; those unrelated hunks were inspected
  and restored and are not part of this task.

## Concerns

- The worktree-local uv environment cannot be freshly resolved without network
  access or a populated writable cache. Direct Python 3.11 verification and an
  explicitly pinned `UV_PROJECT_ENVIRONMENT` provide complete local evidence.
- The suites retain pre-existing expected xfail/skip markers and existing NVML,
  Lightning pytree/DataLoader/eval-mode, and scheduler-order warnings. No new
  numerical-control, masking, recurrence, belief-leakage, checkpoint, or
  deterministic-resume concern was found.
- The full suite and hook deliberately use the existing synthetic/reference
  gates; no unnecessary 720-turn rollout was added for this integration task.
