"""Has a round ever swallowed the 441 KB line, and what did it cost?

The controller a round edits carries one line of 440,834 bytes at line 4,149, in
a file whose median line is 43 bytes. Any read spanning it returns the lot. This
asks the transcripts whether that happened, how often, and how much of a round's
budget went with it.
"""

import json
import logging

from kaggriculture.campaign import config

LOGGER = logging.getLogger(__name__)
# Above this, a single command's output is a meaningful share of a round's
# budget rather than a line of a table. Roughly 25k tokens at four bytes each.
BIG = 100_000


def main() -> None:
    """Report every command whose output was large, by round."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rounds = sorted(
        config.LIVE.rounds.glob("*/codex.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    hit = 0
    worst: list[tuple[int, str, str]] = []
    for path in rounds:
        spent = 0
        biggest = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("type") != "item.completed":
                continue
            item = row.get("item") or {}
            if item.get("type") != "command_execution":
                continue
            output = str(item.get("aggregated_output") or "")
            if len(output) >= BIG:
                spent += len(output)
                biggest.append((len(output), str(item.get("command") or "")[:70]))
        if biggest:
            hit += 1
            for size, command in biggest:
                worst.append((size, path.parent.name, command))
            LOGGER.info(
                "%s: %d big read(s), %.0f KB of output",
                path.parent.name,
                len(biggest),
                spent / 1000,
            )

    LOGGER.info("\n%d of %d rounds had a read over %d KB", hit, len(rounds), BIG // 1000)
    LOGGER.info("the largest single reads:")
    for size, name, command in sorted(worst, reverse=True)[:8]:
        LOGGER.info("  %7.0f KB  %s  %s", size / 1000, name, command)


if __name__ == "__main__":
    main()
