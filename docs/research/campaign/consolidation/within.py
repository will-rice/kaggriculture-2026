"""Among rounds that actually worked, does reading more mean measuring less?

The first cut compared rounds with a huge read against rounds without, and the
second group turned out to be rounds that did nothing at all -- so it compared
working rounds with dead ones and said nothing about the cost of the read. This
looks only at rounds that produced something, and asks whether the ones that
read more got fewer cycles out of the same budget.

Still observational. A round that reads more may be working on something harder.
"""

import json
import logging
import statistics

from kaggriculture.campaign import config

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Rank working rounds by bytes read and compare the halves."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = []
    for path in sorted(config.LIVE.rounds.glob("*/codex.jsonl")):
        read = 0
        edits = 0
        measured = 0
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "item.completed":
                continue
            item = event.get("item") or {}
            kind = item.get("type")
            if kind == "file_change":
                edits += len(item.get("changes") or [])
            elif kind == "command_execution":
                read += len(str(item.get("aggregated_output") or ""))
                if "measure.py" in str(item.get("command") or ""):
                    measured += 1
        if edits or measured:
            rows.append((read, edits, measured))

    rows.sort()
    half = len(rows) // 2
    LOGGER.info("%d rounds that produced something", len(rows))
    for name, part in (
        ("lighter readers", rows[:half]),
        ("heavier readers", rows[half:]),
    ):
        LOGGER.info(
            "  %-16s read %5.0f KB median, %4.1f measurements median, "
            "%4.1f file changes median",
            name,
            statistics.median(read for read, _, _ in part) / 1000,
            statistics.median(measured for _, _, measured in part),
            statistics.median(edits for _, edits, _ in part),
        )


if __name__ == "__main__":
    main()
