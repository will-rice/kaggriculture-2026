#!/usr/bin/env bash
# One pass of the autonomous competitor: a single agent, holding the only guard.
#
# There used to be two schedules. A cron entry ran the mechanical pass -- caches,
# ladder, submission guard -- while a session-bound loop did the judging, and
# each carried its own idea of when submitting was allowed. Two guards meant two
# things to keep correct, and a review found one of them would submit the
# incumbent against itself.
#
# So the mechanical work is now a *step this agent calls* rather than a parallel
# system: `autonomous_pass.sh` refreshes the caches and writes a status file, and
# the agent reads it, judges the training numbers, and decides. One
# decision-maker, one guard, one place a submission can happen.
#
# The agent's authority and its limits live in the prompt below rather than
# here, because they are judgement calls and shell is the wrong place to encode
# a judgement call.
set -u

ROOT="/home/will/projects/kaggriculture-2026"
LOG="$ROOT/run/autonomous/judgement.log"
TIMEOUT_S="${KAGGRICULTURE_JUDGEMENT_TIMEOUT_S:-3600}"

export HOME="${HOME:-/home/will}"
export PATH="/home/will/.local/bin:/usr/bin:/bin"
mkdir -p "$ROOT/run/autonomous"
cd "$ROOT" || exit 1

ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=== $ts judgement pass ===" >>"$LOG"

read -r -d '' PROMPT <<'PROMPT_END' || true
You are the autonomous competitor for the Kaggriculture Kaggle competition,
running unattended from cron. Work in /home/will/projects/kaggriculture-2026.

FIRST, refresh state and read it:
  ./scripts/autonomous_pass.sh
That refreshes the replay corpus and the kernel/discussion caches and writes
run/autonomous/status.json. Read that file. It carries our ladder standing,
today's spent submission slots, what main.py serves, and whether the submission
guard would pass and why not.

THEN read the training numbers: the newest wandb/run-*/files/output.log and any
scratchpad logs. Pull bank, margin, entropy, kl, kl_weight, value, illegal.
Confirm training is alive with:
  ps -eo pid,cmd --no-headers | grep "kaggriculture-2026/.venv/bin/python3 -m kaggriculture" | grep -v "bash -c"

JUDGE AGAINST BANKED COINS, NOT CURVE HEALTH. Smooth entropy, a falling value
loss and a closing margin have all coexisted here with a bank that never moved.
Margin closes as a side effect of selling into a shared market, so it is not
evidence of earning. Known baselines: the behaviour clone banks 17,675; a
warmed-up run plateaus around 21,000; the corpus median seat banks 125,773.

CHECK AGAINST WHAT IS ALREADY KNOWN before concluding anything:
  docs/research/2026-08-07-self-play-rl-findings.md   -- primary-source findings
  .superpowers/sdd/2026-08-07-self-play-rl/progress.md -- what has been tried
If the numbers contradict a finding, trust the numbers and say so.

THEN ACT. Message a running implementer with evidence, dispatch the next task,
stop a run that has been flat for more than ~30 iterations, or run the cheapest
diagnostic that could falsify the current diagnosis. Prefer falsifying to
confirming.

SUBMISSION happens through run/autonomous/status.json and nothing else. If it
says should_submit is false, that is the answer -- do not reason around it, and
do not submit by any other route. If it says true, the guard has already checked
that the tree is clean, that a gate verdict fingerprints the exact artifact set
that would ship, that the gate measured what main.py serves against what the
ladder currently carries, that those two differ, and that slots remain.

NEVER `git add -A` or `git add .`; stage only the paths your commit message
names. Before any commit run `git status --porcelain`, and if files you did not
write are modified, an implementer is mid-task -- do not commit at all.

Be brief. Report what the numbers say, what you did, and what you are waiting
for.
PROMPT_END

timeout "$TIMEOUT_S" claude -p "$PROMPT" >>"$LOG" 2>&1
status=$?
if [[ "$status" -eq 124 ]]; then
  echo "TIMEOUT: judgement pass exceeded ${TIMEOUT_S}s" >>"$LOG"
elif [[ "$status" -ne 0 ]]; then
  echo "FAILED($status): judgement pass" >>"$LOG"
else
  date -u +%Y-%m-%dT%H:%M:%SZ >"$ROOT/run/autonomous/last_judgement_utc"
fi
echo "=== end $ts ($status) ===" >>"$LOG"
