#!/usr/bin/env bash
# Rebuild the corpus dataset from the day's replays, then re-measure every claim.
#
# Runs after `fetch_episodes.py` has landed the new archive. Both steps are
# idempotent and both rebuild from scratch: the dataset is replaced rather than
# appended to, because a half-built one that looks complete is worse than none,
# and the claims are re-measured over the whole corpus every time. A claim
# settled over one corpus can come undone over a bigger one, and the log keeps
# both readings -- that is the point of running it daily rather than once.
#
# It runs from its own worktree with its own virtualenv, deliberately. The
# campaign loop runs from a different worktree, and `uv run` without --no-sync
# reinstalls the package it is executing: sharing one checkout between a cron
# job and a running loop makes the loop's parent and its spawned workers
# diverge mid-game, which has killed a run before.
set -euo pipefail

WORKTREE=/home/will/projects/kaggriculture-2026/.worktrees/scheduled

cd "$WORKTREE"
git fetch --quiet origin main
git reset --quiet --hard origin/main

echo "=== $(date -u +%FT%TZ) extracting corpus ==="
uv run extract-corpus

echo "=== $(date -u +%FT%TZ) measuring claims ==="
uv run strategies

echo "=== $(date -u +%FT%TZ) rendering the report ==="
uv run report

echo "=== $(date -u +%FT%TZ) done ==="
