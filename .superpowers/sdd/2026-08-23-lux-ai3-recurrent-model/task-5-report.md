# Task 5 report: spatial attention and interaction value

## Outcome

Implemented the optional spatial transformer and interaction-aware value head,
wired both through the time-major `StatefulPolicy`, Lightning learner, and
reference worker, and relaxed only their production stage gates. The disabled
control continues to instantiate bare `Policy`; `local_patch` remains rejected.

The interaction head reuses the immutable, validated
`transformer_heads`/`transformer_mlp_ratio` fields. No interaction-specific
configuration knobs were added. Its learned value query attends to exactly one
projected global scalar token plus the 100 unpooled spatial tokens before the
MLP, scalar projection, and existing optional value bound.

## RED evidence

- Added model tests for the explicit `(1, 100, C)` learned positional tensor,
  shape preservation, finite component gradients, the 101-token interaction
  context, remote spatial/global sensitivity, value shape/bounds, supported
  recurrence/belief/attention combinations, and config/stage validation.
- Added Lightning and reference-worker tests requiring attention-only configs
  to construct `StatefulPolicy` and strictly restore every exact stateful key.
- First requested `uv run pytest ...` attempt stopped before collection because
  Python 3.14 tried to build `pygame==2.6.1` without `sdl2-config`, the known
  worktree hook environment issue.
- Re-ran with the repository's Python 3.11 environment and `PYTHONPATH=src`.
  Collection failed on the intended missing feature:
  `ImportError: cannot import name 'InteractionValueHead'`.

## GREEN evidence

- Attention model selector: `10 passed, 20 deselected`.
- Model + Lightning + data integration: `60 passed`.
- Required focused model/Lightning/belief/rollout/data/checkpoint/control gate:
  `119 passed, 1 xfailed`.
- Touched-file Ruff: passed.
- Full-source `ty check src`: passed.
- `git diff --check`: passed.

The expected xfail is the pre-existing rollout split marker. Test output also
retained existing environment warnings for unavailable NVML, Lightning data
loader worker count/pytree deprecation, and the scheduler-order fixture.

## Files changed

- `src/kaggriculture/learn/toad/model.py`
- `src/kaggriculture/learn/toad/config.py`
- `src/kaggriculture/learn/toad/lightning.py`
- `src/kaggriculture/learn/scripts/toad.py`
- `tests/learn/test_toad_model.py`
- `tests/learn/test_toad_lightning.py`
- `tests/learn/test_toad_data.py`
- `.superpowers/sdd/2026-08-23-lux-ai3-recurrent-model/task-5-report.md`

## Concerns

- The worktree-local Python 3.14 `uv` environment still cannot build pygame
  without SDL development tooling. Direct verification used the repository's
  established Python 3.11 environment instead.
- No model, checkpoint, numerical-control, belief-isolation, shifted-done,
  rollout-schema, or metric regression remains in the exercised suites.

## Fix round 1/5: reject degenerate width-one attention

Reviewer evidence showed that an accepted interaction-value topology with
`channels=1` and `transformer_heads=1` was input-independent. Reproduction with
two unrelated board/scalar batches returned the identical value tensor
`[0.3342, 0.3342]` both times. The root cause is mathematical: `LayerNorm(1)`
maps every token to zero before attention. The same normalization degeneracy is
disallowed for the spatial transformer even though its residual can retain
some input dependence.

RED added four independently failing cases: Pydantic rejected neither
transformer nor interaction-value width one, and both direct module constructors
also accepted it.
The fix requires attention-enabled `ModelConfig.channels >= 2` and enforces the
same lower bound in the shared direct-constructor validator. Existing head
divisibility checks remain unchanged.

The smallest valid `channels=2`, `heads=1` interaction head is covered by a
controlled sensitivity fixture: identity query/key/value and output
projections, a fixed `[1, -1]` query, zero MLP, and explicit scalar projection.
Changing only the remote bottom-right spatial token or only one global scalar
increases its value relative to the all-zero context, independent of random
initialization.

Fix-round verification:

- Targeted width/config gate: `14 passed`.
- Original focused model/Lightning/belief/rollout/data/checkpoint/control gate:
  `125 passed, 1 xfailed`.
- Touched-file Ruff: passed.
- Full-source `ty check src`: passed.
- `git diff --check`: passed.

No new concerns. The expected rollout xfail and pre-existing environment
warnings are unchanged.
