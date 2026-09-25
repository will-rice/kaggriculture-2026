"""What does a round actually spend its call on?

The claim to test is that a round's bottleneck is reaching the information it
needs rather than deciding what to change. If that is right, most of what it
does before its first edit is looking things up, and the lookups are a large
share of everything it does.

Read from the transcripts rather than reasoned about: every command a round ran,
in order, classified by what it was for, with the position of the first edit to
`plan.json` as the line between orienting and working.
"""

import collections
import json
import logging
from pathlib import Path

from kaggriculture.campaign import config

LOGGER = logging.getLogger(__name__)
# How a command is classified. First match wins, so the order is the specific
# before the general.
KINDS = (
    ("measure", ("measure.py", "./measure")),
    ("query", ("curl", "clickhouse", "select ")),
    ("read-plan", ("plan.json",)),
    ("read-source", ("child.py", "parent.py", "sed -n", "head -", "tail -", "wc -")),
    ("search", ("grep", "rg ", "find ", "ls ")),
    ("python", ("python", "uv run")),
)


def main() -> None:
    """Classify every command in the most recent rounds and report the shares."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rounds = sorted(
        config.LIVE.rounds.glob("*/codex.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:20]
    LOGGER.info("%d rounds, most recent first", len(rounds))

    totals: collections.Counter[str] = collections.Counter()
    before_edit = []
    commands_each = []
    for path in rounds:
        commands = list(_commands(path))
        commands_each.append(len(commands))
        totals.update(kind for kind, _ in commands)
        edits = [
            index
            for index, (kind, text) in enumerate(commands)
            if kind == "python" and "plan" in text and "json" in text
        ]
        before_edit.append(edits[0] if edits else len(commands))

    every = sum(totals.values())
    LOGGER.info("\n%d commands over %d rounds", every, len(rounds))
    for kind, count in totals.most_common():
        LOGGER.info("  %-12s %4d  %4.1f%%", kind, count, 100 * count / every)
    LOGGER.info(
        "\ncommands per round: median %d, max %d",
        sorted(commands_each)[len(commands_each) // 2],
        max(commands_each),
    )


def _commands(path: Path) -> list[tuple[str, str]]:
    """Every shell command one transcript ran, as ``(kind, text)``."""
    out = []
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
        text = str(item.get("command") or "")
        out.append((_kind(text), text))
    return out


def _kind(text: str) -> str:
    """Which sort of work one command was."""
    low = text.lower()
    for kind, needles in KINDS:
        if any(needle in low for needle in needles):
            return kind
    return "other"


if __name__ == "__main__":
    main()
