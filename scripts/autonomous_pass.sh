#!/usr/bin/env bash
# One pass of the autonomous Kaggriculture worker.
#
# Does the mechanical half of competing: keep the corpus, kernel and discussion
# caches fresh, read where we stand on the ladder, gate the best local candidate
# against what we currently ship, and submit only when the gate says it is
# better. Everything requiring judgement -- designing an experiment, deciding a
# result means the approach is wrong -- is left to the agent, which reads this
# pass's status file rather than re-deriving the state.
#
# Runs from cron under `flock -n` so passes can never overlap. Every Kaggle call
# is bounded so a hung request cannot wedge the slot.
#
# The submission rule is deliberately strict, because a submission is public,
# spends one of five daily slots, and only the latest two are scored:
#
#   - the working tree must be clean and the archive built from a commit, so a
#     score can always be traced to code that exists
#   - `gate.py` must report the candidate beating the agent we currently serve
#   - at least RESERVE_SLOTS must remain afterwards, so a human always has room
#     to correct an automated mistake the same day
#
# Any one of those failing skips the submission and records why. The pass never
# submits to break a tie and never submits twice in a row without a gate.
set -u

ROOT="/home/will/projects/kaggriculture-2026"
SKILL="/home/will/.claude/plugins/cache/nvidia-kaggle/nvidia-kaggle/46eaa990e498/skills/nvidia-kaggle-skill"
COMPETITION="kaggriculture"
STATE="$ROOT/run/autonomous"
LOG="$STATE/pass.log"
STATUS="$STATE/status.json"
STEP_TIMEOUT_S="${KAGGRICULTURE_STEP_TIMEOUT_S:-600}"
RESERVE_SLOTS="${KAGGRICULTURE_RESERVE_SLOTS:-2}"
SUBMIT_ENABLED="${KAGGRICULTURE_SUBMIT:-0}"

# Cron starts with a minimal environment: pin HOME so the Kaggle client finds
# ~/.kaggle, and PATH so uv resolves.
export HOME="${HOME:-/home/will}"
export PATH="/home/will/.local/bin:/usr/bin:/bin"
mkdir -p "$STATE"
cd "$ROOT" || exit 1

ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=== $ts autonomous pass ===" >>"$LOG"
FAILS=0

step() {
  # step LABEL cmd... -- bounded, logged, never aborts the pass. A transient
  # Kaggle failure must not skip the steps after it.
  local label="$1"
  shift
  echo "--- $label" >>"$LOG"
  timeout "$STEP_TIMEOUT_S" "$@" >>"$LOG" 2>&1
  local status=$?
  if [[ "$status" -eq 124 ]]; then
    echo "TIMEOUT: $label exceeded ${STEP_TIMEOUT_S}s" >>"$LOG"
    FAILS=$((FAILS + 1))
  elif [[ "$status" -ne 0 ]]; then
    echo "FAILED($status): $label" >>"$LOG"
    FAILS=$((FAILS + 1))
  fi
  return 0
}

# 1. Fresh replay corpus. The ladder's agent mix changes daily and every
#    corpus statistic in this project is archive-dependent, so a stale corpus
#    silently reasons about a meta that has moved.
step "fetch episodes" "$ROOT/scripts/fetch_episodes.py" --days 2

# 2. Fresh kernels and discussions. A survey taken three days earlier was once
#    quoted here as current; the caches carry ingest dates for that reason.
step "ingest kernels" env PROJECT_ROOT="$ROOT" uv run --with httpx --with kaggle \
  --with kagglesdk --with nbformat --with pydantic --with python-dotenv --with rich \
  python "$SKILL/scripts/kernel_ingest.py" "$COMPETITION" --sort-by voteCount --max-pages 3
step "ingest discussions" env PROJECT_ROOT="$ROOT" uv run --with httpx --with kaggle \
  --with kagglesdk --with nbformat --with pydantic --with python-dotenv --with rich \
  python "$SKILL/scripts/discussion_ingest.py" "$COMPETITION" --max-pages 3 --sort-by votes

# 3. Where we stand, and whether a submission is warranted. All the judgement
#    lives in Python where it can be tested, not in shell.
step "assess and maybe submit" env \
  KAGGRICULTURE_RESERVE_SLOTS="$RESERVE_SLOTS" \
  KAGGRICULTURE_SUBMIT="$SUBMIT_ENABLED" \
  uv run python -m kaggriculture.scripts.autonomous --status "$STATUS"

if [[ "$FAILS" -eq 0 ]]; then
  date -u +%Y-%m-%dT%H:%M:%SZ >"$STATE/last_success_utc"
else
  echo "pass completed with $FAILS failed step(s)" >>"$LOG"
fi
echo "=== end $ts ($FAILS failures) ===" >>"$LOG"
