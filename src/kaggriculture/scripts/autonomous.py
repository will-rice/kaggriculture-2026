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
import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

COMPETITION = "kaggriculture"
DAILY_SLOTS = 5

# What we last put on the ladder, recorded when we put it there. Kaggle's
# submission description is a human-readable label that does not affect
# scoring; reading machine state back out of it meant parsing our own prose,
# which picks the wrong module when a sentence names two and finds nothing when
# a human submits by hand. We control what we submit, so we record it.
LEDGER = Path("run/autonomous/submitted.json")

# Where `gate.py` records its verdict. The pass does not run the gate itself:
# 128 seeded episodes take minutes, and a cron tick that re-gates every half
# hour would spend the box on re-measuring something that has not changed. The
# gate is run deliberately -- by the agent, or by hand -- and leaves this file.
GATE_VERDICT = Path("run/autonomous/gate.json")

# A gate result is only evidence about the revision it was measured on. If the
# tree has moved since, the verdict describes code we are no longer shipping.
# Refusing on a stale verdict is the difference between a gate and a rubber
# stamp.
#
# Wilson lower bound rather than the raw win rate: 65 wins in 128 games is a
# 0.508 rate whose interval still straddles a coin flip, and submitting on that
# spends a slot and displaces the incumbent on noise.
MIN_WIN_RATE_LOWER_BOUND = 0.55


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

    verdict = gate_verdict(revision)
    state["gate"] = verdict
    if verdict is None:
        state["reason"] = (
            f"no gate verdict for {revision}; run gate.py and record it first"
        )
        return state
    defending = last_submitted(submissions)
    state["defending"] = defending
    if defending is None:
        state["reason"] = (
            "the newest submission carries no agent marker, so what is on the "
            "ladder is unknown; a gate cannot say whether to replace something "
            "we cannot name"
        )
        return state
    if verdict["opponent"] != defending:
        state["reason"] = (
            f"gate measured against {verdict['opponent']!r} but the ladder "
            f"carries {defending!r}; that is not the comparison that decides "
            "whether to replace it"
        )
        return state
    if verdict.get("candidate") != state["candidate"]:
        state["reason"] = (
            f"gate measured {verdict.get('candidate')!r} but main.py serves "
            f"{state['candidate']!r}; the archive would ship something the gate "
            "never played"
        )
        return state
    if verdict["candidate"] == verdict["opponent"]:
        state["reason"] = (
            f"gate pitted {verdict['candidate']!r} against itself; beating "
            "yourself is not evidence for replacing yourself, and submitting on "
            "it would re-submit the incumbent every tick"
        )
        return state
    if verdict["low"] <= MIN_WIN_RATE_LOWER_BOUND:
        state["reason"] = (
            f"gate win rate {verdict['win_rate']:.3f} "
            f"[{verdict['low']:.3f}, {verdict['high']:.3f}] against "
            f"{verdict['opponent']}; the lower bound does not clear "
            f"{MIN_WIN_RATE_LOWER_BOUND}, so this is not yet evidence of an improvement"
        )
        return state

    state["should_submit"] = True
    state["reason"] = (
        f"beats {verdict['opponent']} at {verdict['win_rate']:.3f} "
        f"[{verdict['low']:.3f}, {verdict['high']:.3f}] over {verdict['games']} seeded "
        f"episodes, banking {verdict['bank']:,.0f} to {verdict['opponent_bank']:,.0f}"
    )
    return state


def gate_verdict(revision: str) -> dict[str, Any] | None:
    """Return the recorded gate result, but only if it measured this revision.

    Args:
        revision: The commit the archive would be built from.

    Returns:
        The verdict, or None when none exists or it describes other code.
    """
    if not GATE_VERDICT.is_file():
        return None
    verdict = json.loads(GATE_VERDICT.read_text())
    if verdict.get("revision") != revision:
        LOGGER.info(
            "gate verdict is for %s, tree is at %s", verdict.get("revision"), revision
        )
        return None
    return verdict


def last_submitted(submissions: list[dict[str, Any]]) -> str | None:
    """Return the agent module we last put on the ladder, or None if unknown.

    Read from our own ledger rather than from Kaggle's description field, and
    cross-checked against Kaggle: if the newest submission there is newer than
    the one we recorded, something was submitted outside this system and our
    ledger describes an agent that is no longer on the leading edge.

    Returning None on that mismatch is the point. The caller refuses, because a
    gate measured against an opponent we are not actually defending says
    nothing about whether to replace what is there.

    Args:
        submissions: Our submissions from Kaggle, newest first.

    Returns:
        The module we last submitted, or None when unknown or superseded.
    """
    if not LEDGER.is_file():
        return None
    recorded = json.loads(LEDGER.read_text())
    if not submissions:
        return None
    if submissions[0]["date"] > recorded["date"]:
        LOGGER.info(
            "kaggle's newest submission is %s, our ledger records %s; something "
            "was submitted outside this system",
            submissions[0]["date"],
            recorded["date"],
        )
        return None
    return recorded["agent"]


def working_revision() -> str | None:
    """Return a fingerprint of what would ship, or None if the tree is dirty.

    Not `HEAD`. A commit is the wrong thing to stamp a gate verdict with, in
    both directions:

    * It changes when nothing that ships changed. This pass commits refreshed
      cache databases every tick, so stamping HEAD invalidated every verdict
      before the next pass could read it -- the system could only ever have
      submitted on a tick whose ingests happened to be byte-identical.
    * It does not change when something that ships *does* change. `package.py`
      copies `REQUIRED` artifacts that are deliberately gitignored, so a
      re-harvested prototype store alters the agent's behaviour while HEAD and
      a clean tree both stay exactly as they were.

    The fingerprint covers the package source, the entrypoint, and every
    artifact `package.py` copies in -- which is the set that decides what the
    archive does.

    Returns:
        A short hex digest of the shipped set, or None when the tree is dirty.
    """
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    ).stdout.strip()
    if dirty:
        return None

    from kaggriculture.scripts.package import ENTRYPOINT, PACKAGE_ROOT, REQUIRED

    digest = hashlib.sha256()
    tree = subprocess.run(
        ["git", "rev-parse", f"HEAD:{PACKAGE_ROOT.relative_to(Path.cwd())}"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    digest.update(tree.encode())
    digest.update(ENTRYPOINT.read_bytes())
    for artifact in sorted(REQUIRED):
        if not artifact.is_file():
            raise RuntimeError(
                f"{artifact} is required by package.py but absent; the archive "
                "cannot be built and a gate verdict about it would be fiction"
            )
        digest.update(artifact.read_bytes())
    return digest.hexdigest()[:12]


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
    description = f"{candidate}: {reason}"[:500]
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
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(
        json.dumps(
            {
                "agent": candidate,
                "date": datetime.now(timezone.utc).isoformat(sep=" "),
                "reason": reason,
            },
            indent=2,
        )
    )
    return description


if __name__ == "__main__":
    main()
