"""Decide, without a human, whether to submit and what the agent should do next.

The mechanical half of competing is a decision procedure, not a judgement call:
read where we stand, check whether the best local candidate beats what we
already serve, and submit only if it does. This module is that procedure, kept
in Python rather than in the cron shell so it can be tested.

What it deliberately does not do is design experiments or decide that a result
means the approach is wrong. It writes a status file instead, which the agent
reads on its next pass so it starts from measured state rather than from
whatever it remembers.

The submission rule is strict on purpose. A submission is public, spends one of
five daily slots, and only the latest two are scored -- so an automated mistake
is not just wasted, it displaces a good agent from the scored pair. Three
conditions must all hold, and each is reported when it does not:

* the working tree is clean, so the score traces to code that exists
* the gate says the candidate beats the agent currently served by ``main.py``
* enough slots remain afterwards that a human can still correct the day

A tie is not an improvement. Absent evidence, the incumbent stays.
"""

import argparse
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

COMPETITION = "kaggriculture"
DAILY_SLOTS = 5


def main() -> None:
    """Assess the competition state and act, from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--status", type=Path, required=True, help="status file to write"
    )
    parser.add_argument(
        "--reserve",
        type=int,
        default=2,
        help="slots to leave unspent so a human can correct the day",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="actually submit when the gate passes; otherwise report only",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    state = assess(reserve=args.reserve)
    if args.submit and state["should_submit"]:
        state["submitted"] = submit(state["candidate"], state["reason"])
    else:
        state["submitted"] = None

    args.status.write_text(json.dumps(state, indent=2))
    LOGGER.info("wrote %s: %s", args.status, state["reason"])


def assess(reserve: int) -> dict[str, Any]:
    """Return the competition state and whether a submission is warranted.

    Args:
        reserve: Slots to leave unspent today.

    Returns:
        A mapping carrying the ladder standing, the local candidate, whether to
        submit, and the reason -- which is written whether the answer is yes or
        no, because "why not" is the part a later reader needs.
    """
    revision = working_revision()
    submissions = recent_submissions()
    spent = submissions_today(submissions)
    remaining = DAILY_SLOTS - spent

    state: dict[str, Any] = {
        "checked": datetime.now(timezone.utc).isoformat(),
        "revision": revision,
        "scored_pair": [
            {"description": row["description"][:80], "score": row["score"]}
            for row in submissions[:2]
        ],
        "submissions_today": spent,
        "slots_remaining": remaining,
        "candidate": served_agent(),
        "should_submit": False,
        "reason": "",
    }

    if revision is None:
        state["reason"] = (
            "working tree is dirty; a score could not be traced to a commit"
        )
        return state
    if remaining - 1 < reserve:
        state["reason"] = (
            f"{remaining} slot(s) left and {reserve} reserved for a human correction"
        )
        return state

    state["reason"] = (
        "gate has not been run by this pass; the agent decides what to gate and when"
    )
    return state


def working_revision() -> str | None:
    """Return the short commit if the tree is clean, otherwise None.

    A submission from a dirty tree is a measurement of code that exists nowhere,
    which is the same reason `tracking` refuses to record one.
    """
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    ).stdout.strip()
    if dirty:
        return None
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def served_agent() -> str:
    """Return the module `main.py` currently serves.

    Read from the entrypoint rather than assumed, because which agent we ship
    has changed four times in this project and a stale assumption here would
    gate the wrong thing.
    """
    for line in Path("main.py").read_text().splitlines():
        if line.startswith("from kaggriculture") and " import agent" in line:
            return line.split()[1]
    raise RuntimeError(
        "main.py does not import an agent; the entrypoint has changed shape"
    )


def recent_submissions() -> list[dict[str, Any]]:
    """Return our submissions, newest first, with parsed scores."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    listed = api.competition_submissions(COMPETITION)
    if listed is None:
        raise RuntimeError("Kaggle returned no submission list; the state is unknown")
    rows = []
    for item in listed:
        if item is None:
            raise RuntimeError(
                "Kaggle returned an empty submission entry; a partial list would "
                "undercount today's spent slots and could overspend them"
            )
        score = getattr(item, "public_score", None)
        rows.append(
            {
                "date": str(item.date),
                "description": item.description or "",
                "score": float(score) if score not in (None, "") else None,
                "status": str(item.status),
            }
        )
    return rows


def submissions_today(submissions: list[dict[str, Any]]) -> int:
    """Return how many of today's five slots are already spent."""
    today = datetime.now(timezone.utc).date().isoformat()
    return sum(1 for row in submissions if row["date"][:10] == today)


def submit(candidate: str, reason: str) -> str:
    """Build the archive and submit it, returning the description used.

    Args:
        candidate: The agent module being submitted.
        reason: Why the gate passed, recorded in the submission description so
            the ladder entry carries its own justification.

    Returns:
        The description sent to Kaggle.
    """
    description = f"autonomous: {candidate} -- {reason}"[:500]
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "kaggriculture.scripts.submit",
            description,
            "--yes",
        ],
        check=True,
    )
    return description


if __name__ == "__main__":
    main()
