# SDD ledger — plan: docs/superpowers/plans/2026-09-04-campaign-loop.md

Worktree: .claude/worktrees/campaign, branch campaign. Started after foundation Task 11; foundation Task 12 (phase-1 codex sessions) still in flight.
Carry-overs from the foundation ledger: harness.play raises on any crash; chdir sandbox only in kernel_watch.gate; held-out lynnsakurai_v5 near-duplicates indarkarhana (replace); seat-1 step injected by harness.
Task 1: complete (commits f5d98ad..d20f598, review clean)
Task 2: complete (commits d20f598..30c683c, review clean). Deferred (final wave): guard apply_weakness_pressure when old weight == 1.0 (single-opponent pool); add a test that a full pool with nobody crushed grows past cap (FAMOU behaviour); weighted() is a partial dot product when rates omit members.
Task 3: implemented (commit 2125836); review dispatched. Note for Task 5: task_prompt.md carries worktree-absolute harness commands; prompt.py must strip/replace them.
Task 3: complete (commits 30c683c..2125836, review clean). Implementer fixed two reference-code bugs (UCB visits, migration snapshot). Minor (deferred): id-uniqueness not asserted on insert/store; reset id second-precision.
Task 4: implemented (cdd1b86). Notes for Task 8: pool.save(config.POOL) BEFORE deep-evaluating against a new champion (roster.path reads the saved pool); DeepResult.field assumes >=1 vendored opponent remains in the pool.
Task 4: review — approved, 1 Important (plan-mandated formula): pooled Wilson on weighted score is overconfident under non-uniform weights. Ruling: fix now — low/high = weighted sums of per-opponent Wilson bounds. Fix round 1 dispatched. Minor (deferred): field ZeroDivisionError not in Raises; report text vs code filter mismatch.
Task 4: fix round 1/5 (1 addressed, 0 open — weighted per-opponent Wilson bounds; commits cdd1b86..597b4cf)
Task 4: complete (commits f35cb75..597b4cf, review clean).
Task 5: review — 1 Important: build_sandbox rmtree on unguarded program_id (traversal). Fix round 1 dispatched. Minor folded in: module-level basicConfig / root logger; no test for the added prompt line.
Task 5: fix round 1/5 (3 addressed, 0 open; commits ee11e09..f91dabb)
Task 5: complete (commits 597b4cf..f91dabb, review clean).
Task 6: implemented (92a0944; earlier 155f2a8 was amended/replaced). Real codex call: 50.7 s, 168283 input / 1836 output tokens on a trivial sandbox — flag per-call cost vs DAILY_CALL_BUDGET. Codex refuses untrusted dirs; sandboxes under the repo are trusted.
Task 6: review — 1 Important: timeout kills only the codex process, not its process group (orphaned shell children). Fix round 1 dispatched. Minor (deferred): exec_error discards a possibly usable child.py.
Task 6: fix round 1 implemented (1f35d6c, process-group kill); re-review dispatched.
Task 6: fix round 1/5 (1 addressed, 0 open; commits 92a0944..1f35d6c)
Task 6: complete (commits f91dabb..1f35d6c, review clean).
Task 7: implemented (b1b6946); review dispatched.
Task 7: review — approved; 1 Important: commit failure after floor/pool/epoch written would raise out of promote. Ruling: git is the record, not the truth — log and continue. Fix round 1 dispatched. Minor (deferred): boundary-value and champion_2 tests.
Task 7: fix round 1 implemented (4517f22); re-review dispatched.
Task 7: fix round 1/5 (1 addressed + minors; commits b1b6946..4517f22)
Task 7: complete (commits 1f35d6c..4517f22, review clean).
Task 8: implemented (0de19c7). Campaign-level notes: archive.append_eval has no caller (programs are fast-evaluated once at insert; FAMOU re-evaluates parents — consider re-evaluating UCB parents each selection to cut seed luck); budget-exhausted branch sleeps 60 s, untested.
Task 8: complete (commits 4517f22..0de19c7, review clean). Deferred to final wave: test must pass commit=False explicitly (add a commit flag to run/epoch); archive.top docstring misleading (n_evals never 0). Parked with ruling: deep scores not fed back into the archive — matches FAMOU (deep eval confirms champions only).
ALL LOOP TASKS COMPLETE. Final whole-branch review over 0be4293..0de19c7.
FINAL REVIEW: With fixes. 2 Critical (champion file aliasing; candidate vs champion on different pools), 11 Important, minors, ledger triage. One fix wave dispatched with rulings in final-fix-brief.md.
Final fix wave complete: 1772df6, 614d3ce (fix subagent), ab5bfbf (controller: re-review new defects #1-#6). Re-review: all 13 findings RESOLVED. Parked with rulings: opponent segfault surfaces as BrokenProcessPool and is charged to the candidate (rare; loud); champion.json is the last write in promote so a kill between pool.save and the rename can still double-promote (microsecond window); node_modules stays in .gitignore (prettier hook resolves its plugin there); validation spawns one child per candidate outside CORE_BUDGET accounting (watch on the real run).
