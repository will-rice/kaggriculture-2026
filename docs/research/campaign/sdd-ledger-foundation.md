# SDD ledger — plan: docs/superpowers/plans/2026-09-04-campaign-foundation.md

Worktree: .claude/worktrees/campaign, branch campaign, base 0be4293.

Task 1: ruling — pre-commit pytest hook fails on pre-existing tests in tests/market_residual, tests/search, tests/learn (missing gitignored checkpoints in the worktree; all three directories are deleted by Task 2). Task 1 commits with SKIP=pytest; formatting/lint/type hooks run. Baseline run with -x had hidden these.
Task 1: review — spec ✅, 1 Important (plan-mandated): summarize fixed-width slicing breaks when outcomes skips a failed game. Ruling: finding governs; plan snippet was a transcription oversight. Fix round 1 dispatched.
Task 1: fix round 1/5 (1 addressed, 0 open — summarize attributes by member; commits 542c5e7..82d13c8)
Task 1: complete (commits 0be4293..82d13c8, review clean). Note: summarize signature is (scores, league); seeds dropped.
Task 2: complete (commits 240c780..73fc602, review clean). ⚠️ resolved: hooks/ and rule_search.py were untracked in the main checkout, never committed. Minor (deferred): stale .gitignore patterns for learn/policy.pt and tests/learn/fixtures; README lists `campaign --help` before Task 6 creates it.
Task 3: complete (commits 73fc602..dd014f7, review clean). Note: sim.hpp carries no SPDX line; Apache-2.0 attribution rests on the kernel (NOTICE names the source) — flag to user. Minor (deferred): engine/**init**.py has a docstring (ruff D104).
Task 4: note for Task 6 — engine state (and archives) carry no `step` for seat 1; the runner injects shared `step` and `remainingOverageTime` at call time. Harness must add both to seat-1 observations before calling an agent.
Task 4: complete (commits dd014f7..62d8d0c, review clean). Important (report claim, not code): animal-tile branch not exercised by the cited replay; reviewer verified it by hand; Task 5 tapes cover it. Minor (deferred): rich pack_action paths not round-tripped through the engine in committed tests; comment the guarded/unguarded int() asymmetry.
Task 5: review — spec ✅, 2 Important (plan-mandated): collection-time sample(200) costs minutes/18GB on every pytest run incl. the hook; missing /data silently skips. Ruling: findings govern. Fix round 1 dispatched. Minor (deferred): sample(8)/sample(200) re-parse the same archives; comment STATUSES.
Task 5: fix round 1/5 (2 addressed, 0 open — lazy qualifying(i), loud FileNotFoundError; commits e4906b6..2f433f0)
Task 5: complete (commits 62d8d0c..2f433f0, review clean). 200 tapes bit-identical.
Task 6: implementer note — indarkarhana (training) and lynnsakurai_v5 (held-out) share 7/8 files; held-out set effectively one agent. Plan 2 roster must replace lynnsakurai_v5 with an independent held-out opponent (kernel_watch find). salemali7_2900 uses agent(obs) one-arg signature; harness resolves arity at load.
Task 6: review — spec ❌ on 3 Important: check() gives seat 1 no step; opponent exception leaks real path via pool traceback; \_verify_sample RNG seeded on public game count. Fix round 1 dispatched.
Task 6: note for Plan 2 — harness.play RAISES RuntimeError when any side crashes (sanitized); evaluator/validate/loop must treat that as a failed evaluation, never a 0 bank.
Task 6: fix round 1/5 (3 addressed, 0 open; commits 61fbcb3..e33e509)
Task 6: complete (commits 2f433f0..e33e509, review clean).
Task 7: review — spec ✅ interfaces; 2 Important: whitespace tokenizer defeated by spacing/quote reformatting (0.196 -> 0.010 on a 110-line lift); fixture is 44 lines not ~60. Fix round 1 dispatched.
Task 7: fix round 1/5 (2 addressed, 0 open — regex tokenizer, THRESHOLD 0.03 from six numbers; commits 3ca123f..4161435)
Task 7: complete (commits e33e509..4161435, review clean).
Task 8: review — 2 Important: **import**/importlib evade the import scan; last-callable rule misses class/lambda/async def. Fix round 1 dispatched. Minor (deferred): no test for the no-/data-path reason rule; relative-import reason text; ctypes in allowlist (plan-inherited).
Task 8: fix round 1/5 (2 addressed, 0 open — denied calls, wider last-callable + dynamic name check; commits 5f19785..6414792)
Task 8: complete (commits 4161435..6414792, review clean). Residual (accepted): dynamic import routes like getattr(**builtins**, ...) are not statically closable; harness.check executes the code anyway.
Task 9: review — 2 Important: container runs as root and leaves undeletable files in tmp_path; no --pull=never. Fix round 1 dispatched. Minor (deferred): .so not exercised in-episode (plan design); no --network none.
Task 9: fix round 1/5 (2 addressed, 0 open — --user, pre-created mountpoint, --pull=never, --network none; commits f4b7f89..777f282)
Task 9: complete (commits 6414792..777f282, review clean). Image: Python 3.12.13, kaggle_environments 1.29.3 (fallback mount used); .so loads.
Task 10: note for Plan 2 — kernel_watch.gate chdirs before running untrusted kernels; harness.\_play workers still run in the caller cwd and candidates may import pathlib. Plan 2 evaluator/harness should chdir workers into a scratch dir.
Task 10: complete (commits 777f282..fa25a33, review clean). Carry to Plan 2: (a) play_unsealed raises on any crash so score_field aborts whole — evaluator must map a candidate crash to a rejection, and kernel_watch drops a stranger kernel that crashes once; (b) chdir sandbox lives only in kernel_watch.gate — Plan 2 deep evaluator must chdir too (or move it into harness.\_play). Minor: no direct test that play_unsealed allows exam seeds.
Task 11: complete (commits fa25a33..f5d98ad, review clean). Skeleton banks ~3699. Minor (deferred): a harvest on a day-final turn can be weeded before replanting (0.5%/day); no weed-clearing branch.
Task 12: in progress — docs/campaign/phase1-brief.md written; run/campaign/phase1 set up; codex session 1 stopped for approval; resumed as session 2 (id 01a06ede-bbc9-7a50-97f4-e0c2397fa1ce) with approval; awaiting notes/game-model.md and task_prompt.md.
Task 12: complete (commit f35cb75). Phase 1: codex session 01a06ede wrote notes/game-model.md (295 lines), task_prompt.md (241 lines), 4 probes; 5 numbers spot-checked OK. Note: task_prompt.md hardcodes the worktree path in its harness commands — Plan 2 prompt.py must replace that section with its own HARNESS_SECTION.
ALL FOUNDATION TASKS COMPLETE. Final whole-branch review pending (to run after Plan 2, on the whole branch).
