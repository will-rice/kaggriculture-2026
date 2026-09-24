#!/usr/bin/env bash
# Rebuild the corpus from the day's replays, re-measure every claim, re-render
# the report, and commit whatever changed.
#
# Runs after `fetch_episodes.py` has landed the new archive. Every step is
# idempotent and rebuilds from scratch: the dataset is replaced rather than
# appended to, because a half-built one that looks complete is worse than none,
# and the claims are re-measured over the whole corpus every time. A claim
# settled over one corpus can come undone over a bigger one -- eight of the
# first eleven cleared at sixty games and one survived sixteen thousand -- so
# re-measuring is not maintenance, it is the mechanism.
#
# It runs from its own worktree with its own virtualenv, deliberately. The
# campaign loop runs from another, and `uv run` reinstalls the package it
# executes: sharing a checkout between a cron job and a running loop makes the
# loop's parent and its spawned workers diverge mid-game, which has killed a
# run before.
set -euo pipefail

WORKTREE=/home/will/projects/kaggriculture-2026/.worktrees/scheduled
# The two files the run regenerates. Both are committed: the build order
# because the round prompt reads it out of the package, the report because it
# is what gets published and a version of it is worth keeping.
GENERATED=(
  src/kaggriculture/campaign/build_order.md
  docs/reports/ladder.html
)

cd "$WORKTREE"
git fetch --quiet origin main

# Discard what this job writes, and nothing else. `reset --hard` discarded
# both indiscriminately, unattended, at one in the morning -- and this worktree
# had `main` itself checked out, so that included any commit left on it. Four
# nightly resets have been fast-forwards that lost nothing, which is luck
# rather than a property of the command.
git restore --quiet --worktree --staged -- "${GENERATED[@]}"
intruder=$(git status --porcelain --untracked-files=no)
if [ -n "$intruder" ]; then
  echo "=== $(date -u +%FT%TZ) changes here are not this job's; not running ==="
  echo "$intruder"
  exit 1
fi

# Detached, because this job owns no branch. It needs a tree at origin/main and
# a commit to push from, and it pushes `HEAD:main` already -- holding the
# branch bought nothing and cost two things: every run moved it, and no other
# worktree in this repository could check main out while this one held it.
git checkout --quiet --detach origin/main

echo "=== $(date -u +%FT%TZ) extracting corpus ==="
uv run extract-corpus

echo "=== $(date -u +%FT%TZ) measuring claims ==="
uv run strategies

echo "=== $(date -u +%FT%TZ) rebuilding the build order ==="
uv run build-order

echo "=== $(date -u +%FT%TZ) rendering the report ==="
uv run report

# Formatted before the diff is taken, not after. The generators write plain
# output and prettier rewrites whitespace in it, so an unformatted file always
# differs from the formatted one in the repo -- and the job would commit every
# night whether or not a single number had moved.
echo "=== $(date -u +%FT%TZ) formatting ==="
uv run pre-commit run prettier-format --files "${GENERATED[@]}" || true

if git diff --quiet -- "${GENERATED[@]}"; then
  echo "=== $(date -u +%FT%TZ) nothing changed ==="
else
  echo "=== $(date -u +%FT%TZ) committing ==="
  git add -- "${GENERATED[@]}"
  git -c user.name="kaggriculture" -c user.email="wrice20@gmail.com" \
    commit -q -m "data: corpus through $(date -u +%F)" -- "${GENERATED[@]}"
  git push -q origin HEAD:main
  echo "pushed $(git rev-parse --short HEAD)"
fi

echo "=== $(date -u +%FT%TZ) done ==="
