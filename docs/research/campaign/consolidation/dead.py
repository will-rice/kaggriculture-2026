"""The rounds that produced nothing: what happened instead?

Eight of forty-one transcripts hold no file change and no measurement. That is a
fifth of the calls the campaign spends, and unlike the size of the program it is
a number with an obvious ceiling on the gain: a round that does nothing cannot
have been slowed by anything.
"""

import json
import logging

from kaggriculture.campaign import config

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Print the last thing each empty round said or did."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for path in sorted(config.LIVE.rounds.glob("*/codex.jsonl")):
        items = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "item.completed":
                items.append(event.get("item") or {})
        worked = any(
            item.get("type") == "file_change"
            or "measure.py" in str(item.get("command") or "")
            for item in items
        )
        if worked:
            continue
        LOGGER.info(
            "%s  %d bytes, %d items", path.parent.name, path.stat().st_size, len(items)
        )
        for item in items[-2:]:
            text = (
                item.get("message")
                or item.get("text")
                or str(item.get("command") or "")
            )
            LOGGER.info(
                "    %-16s %s", item.get("type"), str(text)[:150].replace("\n", " ")
            )


if __name__ == "__main__":
    main()
