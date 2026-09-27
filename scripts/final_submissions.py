#!/usr/bin/env python
"""Submit the two most recent champions, once, shortly before the deadline.

Only the latest two submissions are scored and those same two are what the
final leaderboard uses, so the last thing this account does before the
deadline decides the result. A promotion can land at any hour and the
campaign runs unattended, so the choice of what to ship is made here from
what exists at the time rather than by a person picking a file.

Three guards, because an upload cannot be taken back:

* a window. Outside it the run refuses, so a timer that fires on the wrong
  day cannot spend a slot on a champion five days early.
* validation. Each candidate goes through `package.build`, which is
  `validate.validate` -- Kaggle's own loader and a full episode -- and a
  champion that fails is skipped for the next most recent rather than
  uploaded and discovered at the finale.
* a marker. A machine that reboots can fire a timer twice; the second run
  reads the marker and stops.

Run it with ``--dry-run`` to see exactly what it would send.
"""

import argparse
import datetime
import json
import logging
import re
import subprocess
from pathlib import Path

from kaggriculture.campaign import archive, config
from kaggriculture.constants import ENVIRONMENT
from kaggriculture.scripts import package

LOGGER = logging.getLogger(__name__)

# The deadline, verified against the Kaggle API on 2026-09-26 rather than
# taken from a note, and the machine runs on UTC.
DEADLINE = datetime.datetime(2026, 9, 30, 23, 59, tzinfo=datetime.UTC)
# How long before it this may run. Wide enough that a late promotion still
# gets in, narrow enough that a stray firing days early is refused: each
# submission is a local validation episode and an upload, about two minutes.
WINDOW = datetime.timedelta(hours=4)
# Written once the uploads are done, so a second firing does nothing.
MARKER = "final-submissions.json"
CHAMPION = re.compile(r"^champion_(?P<number>\d+)\.py$")


def main() -> None:
    """Pick the two newest champions, validate them, and send them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and package, report what would be sent, upload nothing",
    )
    parser.add_argument(
        "--count", type=int, default=2, help="how many to send, newest last"
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    marker = config.LIVE.root / MARKER
    if marker.exists() and not arguments.dry_run:
        LOGGER.warning("already sent, per %s; doing nothing", marker)
        return
    now = datetime.datetime.now(tz=datetime.UTC)
    if not arguments.dry_run and not DEADLINE - WINDOW <= now <= DEADLINE:
        raise SystemExit(
            f"{now:%Y-%m-%d %H:%M} UTC is outside the {WINDOW} before "
            f"{DEADLINE:%Y-%m-%d %H:%M} UTC; refusing to spend a slot"
        )

    sent = []
    for champion in newest(arguments.count):
        try:
            built = package.build(
                output=config.LIVE.root / f"{champion.stem}-submission.tar.gz",
                entrypoint=champion,
            )
        except RuntimeError as refused:
            # Skipped rather than fatal: the next most recent champion is a
            # promotion away from this one and the pair still has to be filled.
            LOGGER.error(
                "%s does not validate and will not ship: %s", champion, refused
            )
            continue
        message = describes(champion)
        LOGGER.info(
            "%s -> %s (%d KiB)", champion.name, built.name, built.stat().st_size // 1024
        )
        LOGGER.info("  %s", message)
        if arguments.dry_run:
            continue
        subprocess.run(
            [
                "kaggle",
                "competitions",
                "submit",
                ENVIRONMENT,
                "-f",
                str(built),
                "-m",
                message,
            ],
            check=True,
        )
        sent.append(champion.name)

    if arguments.dry_run:
        LOGGER.info("dry run: nothing uploaded")
        return
    if not sent:
        raise SystemExit("nothing validated; no submission was made")
    marker.write_text(
        json.dumps({"sent": sent, "at": f"{now:%Y-%m-%dT%H:%M:%SZ}"}, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("sent %s", ", ".join(sent))


def newest(count: int) -> list[Path]:
    """The ``count`` most recent champions, oldest first.

    Oldest first because only the latest submissions are scored, so the one
    uploaded last is the one that is certain to be in the pair.

    Args:
        count: How many to return. More than exist returns what exists.

    Returns:
        Paths to the champion programs, the newest last.

    Raises:
        SystemExit: The run holds no champions at all.
    """
    found = {}
    for path in config.LIVE.champions.glob("champion_*.py"):
        named = CHAMPION.match(path.name)
        if named:
            found[int(named.group("number"))] = path
    if not found:
        raise SystemExit(f"no champions under {config.LIVE.champions}")
    return [found[number] for number in sorted(found)[-count:]]


def describes(champion: Path) -> str:
    """What to tell Kaggle about this champion, from the campaign's own record.

    Defensive about the archive: a description is worth having and never worth
    failing an upload for, so anything it cannot find is left out.

    Args:
        champion: The champion program about to ship.

    Returns:
        The submission description.
    """
    said = [f"{champion.stem}: one of this campaign's final two submissions."]
    try:
        database = archive.Database(config.LIVE.archive, config.LIVE.programs)
        # By bytes, not by name. `promoted_as` records the champion's pool key,
        # which is `config.POOL_CHAMPION` for every one of them -- the slot,
        # not the agent -- so the only thing that ties `champion_44.py` to the
        # program that became it is that the gate copied those bytes. Newest
        # first because a champion is a recent program, and sized first because
        # these are 870 KB each.
        wanted = champion.read_bytes()
        for program in sorted(database.programs, key=lambda p: -p.created):
            stored = Path(program.source_path)
            if not stored.exists() or stored.stat().st_size != len(wanted):
                continue
            if stored.read_bytes() != wanted:
                continue
            said.append(
                f"Gate {program.fitness:.3f} over {len(program.rates)} opponents, "
                f"mean bank margin {archive.mean_margin(program):+,.0f} a game and "
                f"{archive.swept_margin(program):+,.0f} against the opponents it "
                f"sweeps."
            )
            if program.changed:
                said.append(f"The edit that earned it: {program.changed}")
            break
    except (OSError, ValueError, KeyError) as unreadable:
        LOGGER.warning("no record for %s: %s", champion.stem, unreadable)
    said.append(
        "Selection is the win rate first and, where two programs win the same "
        "games, the bank margin against the opponents both sweep -- which is "
        "where a rate pinned at 1.000 has nothing left to say."
    )
    return " ".join(said)


if __name__ == "__main__":
    main()
