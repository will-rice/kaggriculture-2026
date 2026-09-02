#!/usr/bin/env bash
# Daily kernel scan, run after the episode archive lands.
#
# The meta rotates every two or three days -- a published kernel sweeps to
# 50-80% of top-band seats and then collapses -- so the lever with the only
# demonstrated track record is catching the next wave early. This runs the
# scan and leaves the submission decision to a person, because the final
# ranking is a Bradley-Terry tournament over the two agents standing at the
# deadline and a slot spent before then buys nothing.
set -euo pipefail
cd /home/will/projects/kaggriculture-2026
set -a; source .env; set +a
exec uv run python -m kaggriculture.scripts.kernel_watch --limit 8 --gate-seeds 16
