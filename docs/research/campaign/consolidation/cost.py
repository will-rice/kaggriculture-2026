"""Do the rounds that swallowed the blob end up with less to show for it?

A round that spends 165k of a 200k budget on one command has to finish the job
in what is left. The transcripts say how much each round read; they also say how
many files it changed and how often it measured, which is what it was there to
do. This puts the two beside each other.

Not a controlled comparison -- nothing assigned rounds to arms, and a round that
reads a lot may differ in other ways. It is the observation that motivates the
fix, not evidence the fix works.
"""

import json
import logging
import statistics

from kaggriculture.campaign import config

LOGGER = logging.getLogger(__name__)
BIG = 100_000


def main() -> None:
    """Split rounds by whether they had a huge read, and compare what they did."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    arms: dict[bool, list[tuple[int, int, int]]] = {False: [], True: []}
    for path in sorted(config.LIVE.rounds.glob("*/codex.jsonl")):
        read = 0
        edits = 0
        measured = 0
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("type") != "item.completed":
                continue
            item = row.get("item") or {}
            kind = item.get("type")
            if kind == "file_change":
                edits += len(item.get("changes") or [])
            elif kind == "command_execution":
                read += len(str(item.get("aggregated_output") or ""))
                if "measure.py" in str(item.get("command") or ""):
                    measured += 1
        arms[read >= BIG].append((read, edits, measured))

    for swallowed, rows in sorted(arms.items()):
        if not rows:
            continue
        LOGGER.info(
            "%s (%d rounds): read %.0f KB median, %.1f file changes median, "
            "%.1f measurements median",
            "swallowed a big read" if swallowed else "did not",
            len(rows),
            statistics.median(read for read, _, _ in rows) / 1000,
            statistics.median(edits for _, edits, _ in rows),
            statistics.median(measured for _, _, measured in rows),
        )


if __name__ == "__main__":
    main()
