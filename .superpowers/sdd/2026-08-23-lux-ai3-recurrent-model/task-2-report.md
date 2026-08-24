# Task 2 report: terminal-aware ConvLSTM policy memory

## Delivered

- Added `ConvLSTMCell`, stacked `ConvLSTM`, and typed `ConvLSTMState` /
  `ConvLSTMOutput` sequence contracts.
- Enabled `StatefulPolicy` processes true time-major board, scalar, and
  position tensors through the residual trunk, recurrent stack, 1x1 merge,
  then the existing heads.
- A terminal row clears hidden state, cell state, and `prior_belief` before
  that row's observation. Nonterminal rows preserve all three carried values.
- `recurrent_layers` selects an independent hidden/cell map per layer. The
  one-layer `PolicyState` shape remains compatible with the Task 1 contract.
- Disabled configurations still delegate exactly to the existing `Policy`.
  The production `ToadConfig` recurrent gate is intentionally unchanged for
  Task 3.

## Test evidence

### RED

1. `uv run pytest tests/learn/test_toad_model.py -k "terminal_reset or two_chunks" -v`
   could not start because the sandbox could not create uv's cache lock.
2. The same command outside the sandbox reached dependency resolution but the
   worktree's Python 3.14 environment could not build `pygame==2.6.1` because
   SDL development tooling (`sdl2-config`) is unavailable.
3. `PYTHONPATH=src /home/will/projects/kaggriculture-2026/.venv/bin/python -m pytest tests/learn/test_toad_model.py -k "terminal_reset or two_chunks" -v`
   failed during collection with the expected error:
   `ImportError: cannot import name 'ConvLSTM'`.

### GREEN

- `PYTHONPATH=src /home/will/projects/kaggriculture-2026/.venv/bin/python -m pytest tests/learn/test_toad_model.py -v`
  — 22 passed.
- The Task 1/control fixture/checkpoint selection
  (`test_toad_model.py`, `test_toad_control_fixture.py`, `test_toad_config.py`,
  `test_toad_lightning.py`, and `test_toad_final_repairs.py`) — 68 passed,
  1 deselected.
- `python -m ruff format --check src/kaggriculture/learn/toad/model.py tests/learn/test_toad_model.py`
  — 2 files already formatted.
- `python -m ruff check src/kaggriculture/learn/toad/model.py tests/learn/test_toad_model.py`
  — all checks passed.
- `python -m ty check src` — all checks passed.
- `git diff --check` — clean.

## Files

- `src/kaggriculture/learn/toad/model.py`
- `tests/learn/test_toad_model.py`
- `.superpowers/sdd/2026-08-23-lux-ai3-recurrent-model/task-2-report.md`

## Concerns

- The requested `uv run` gate remains blocked by the worktree's Python 3.14
  dependency environment (`pygame` needs unavailable SDL build tooling). All
  tests and static checks above ran against the repository's existing Python
  3.11 virtual environment with `PYTHONPATH=src`, ensuring imports came from
  this worktree.
- Task 3 remains responsible for wiring recurrent state and done masks through
  the learner and collector; the `ToadConfig` production stage gate remains
  deliberately closed.

## Fix round 1/5: prior-belief temporal reset

### Delivered

- Added `PolicyState.reset_rows(dones)`, which clears hidden, cell, and prior
  belief for exactly the terminal batch rows.
- ConvLSTM now exposes a one-observation `step` transition. Its sequence
  forward resets state before every step, and the enabled policy uses the same
  reset-before-step order for its full `PolicyState`. This is the extension
  point Task 4 needs to replace a reset prior belief with each new prediction.
- Extended terminal coverage to compare hidden/cell sequences and final state
  with fresh processing, test the policy wrapper, and place a terminal in the
  middle of a sequence with a nonzero carried prior belief.

### RED

`PYTHONPATH=src /home/will/projects/kaggriculture-2026/.venv/bin/python -m pytest tests/learn/test_toad_model.py -k "terminal_reset or prior_belief or policy_state_reset" -v`
failed as intended: `PolicyState` had no `reset_rows` attribute (1 failed, 2
passed, 20 deselected).

### GREEN

- The same focused terminal-boundary selection — 4 passed.
- Original Task 2 plus Task 1/control-fixture/checkpoint selection — 70
  passed, 1 deselected.
- `python -m ruff format ...` reformatted the two changed files, then
  `python -m ruff check ...` passed.
- `python -m ty check src` — all checks passed.
- `git diff --check` — clean.
