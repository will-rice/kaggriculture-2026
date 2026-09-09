"""The same calibration, with the trajectory the first attempt withheld.

Seven verdicts from a single observation each landed on a quantity the corpus
cannot separate the sides on. The reading was that local efficiency is all one
frame can show, and the corpus's levers -- land, quadrants, animals, pens, in
the first week -- are statements about two farms across days.

So this gives the model what it was missing: every quantity for both sides on
every day of the season, and the moments chosen where the two came apart
rather than at fixed steps. Same question, same vocabulary, same scoring.
"""

import asyncio
import json
import pathlib
import subprocess
import tempfile

from kaggriculture.campaign import strategy

GAME = pathlib.Path(
    "/tmp/claude-1000/-home-will-projects-kaggriculture-2026/"
    "521a4f31-d72e-47e3-adbb-c5944812e9ba/scratchpad/trajectory.json"
)
RULES = pathlib.Path("src/kaggriculture/campaign/task_prompt.md")
MODEL = "gpt-5.6-luna"
EFFORT = "medium"

ASK = """You are diagnosing a two-player farming game that our side lost.

`rules.md` is the game.
`trajectory.json` is the whole season: for every day, twenty-nine measured
quantities for our side and for theirs. This is where the game was won and
lost, and comparing the two columns day by day is the point of it.
`state.json` is our seat's complete observation at one moment, and the move we
actually played there. That moment is one of the days the gap moved against us.

Write `verdict.json` and nothing else:

{
  "mistake": true or false,
  "why": "at most two sentences, citing numbers from the trajectory",
  "claim": {
    "quantity": "one of the quantities listed below, exactly",
    "day": <the day number this is about>,
    "direction": "more" or "less"
  }
}

`claim` is the general lesson, stated so it can be checked against thousands of
other games: which quantity the stronger player should hold more or less of, on
which day. Read the trajectory before deciding what the claim is about -- the
quantity that decided this game is the one where the two columns diverge, which
is not always the quantity the move touched.

The quantity must be exactly one of:
{quantities}
"""


def main() -> None:
    """Ask about every turning point, then score the claims against the corpus."""
    store = strategy.Strategies(strategy.STORE)
    known = {(claim.form.quantity, claim.form.day): claim for claim in store.claims}
    game = json.loads(GAME.read_text(encoding="utf-8"))
    print(
        f"seed {game['seed']}: {game['final_ours']:,.0f} against "
        f"{game['final_theirs']:,.0f}\n"
    )
    verdicts = asyncio.run(_ask_all(game))

    print(f"{'day':>4} {'quantity':<16}{'says':<7}{'corpus':<9}{'games':>7}  agree?")
    scored = {"agree": 0, "disagree": 0, "unsettled": 0, "unusable": 0}
    for verdict in verdicts:
        claim = (verdict or {}).get("claim") or {}
        quantity, day, said = (
            claim.get("quantity"),
            claim.get("day"),
            claim.get("direction"),
        )
        if quantity not in strategy.QUANTITIES or said not in {"more", "less"}:
            scored["unusable"] += 1
            print(f"{'?':>4} {str(quantity)[:15]:<16}{str(said):<7}unusable")
            continue
        measured = known.get((quantity, int(day)))
        if measured is None or not measured.settled:
            scored["unsettled"] += 1
            print(
                f"{int(day):>4} {quantity:<16}{said:<7}{'open':<9}"
                f"{measured.support if measured else 0:>7}  --"
            )
            continue
        corpus = "more" if measured.status == "leads" else "less"
        hit = corpus == said
        scored["agree" if hit else "disagree"] += 1
        print(
            f"{int(day):>4} {quantity:<16}{said:<7}{corpus:<9}"
            f"{measured.support:>7}  {'yes' if hit else 'NO'}"
        )

    settled = scored["agree"] + scored["disagree"]
    print(f"\n{scored}")
    if settled:
        print(
            f"calibration on settled claims: {scored['agree']}/{settled} "
            f"= {scored['agree'] / settled:.0%}   (chance is 50%)"
        )
    else:
        print("still nothing the corpus settles: it is not finding the levers")


async def _ask_all(game: dict) -> list[dict | None]:
    """One call per turning point, all in flight together."""
    trajectory = json.dumps(game["trajectory"], indent=1)
    return await asyncio.gather(
        *(_ask(moment, trajectory) for moment in game["moments"])
    )


async def _ask(moment: dict, trajectory: str) -> dict | None:
    """Put one turning point, with the whole season beside it."""
    with tempfile.TemporaryDirectory(prefix="oracle-") as box:
        room = pathlib.Path(box)
        (room / "rules.md").write_text(
            RULES.read_text(encoding="utf-8"), encoding="utf-8"
        )
        (room / "trajectory.json").write_text(trajectory, encoding="utf-8")
        (room / "state.json").write_text(json.dumps(moment, indent=1), encoding="utf-8")
        process = await asyncio.create_subprocess_exec(
            "codex", "exec", "--skip-git-repo-check", "-s", "workspace-write",
            "-c", "approval_policy=never",
            "-c", f"model_reasoning_effort={EFFORT}",
            "-m", MODEL, "-C", str(room), "-",
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=room,
            start_new_session=True,
        )
        await process.communicate(
            ASK.replace("{quantities}", ", ".join(sorted(strategy.QUANTITIES))).encode()
        )
        written = room / "verdict.json"
        if not written.exists():
            return None
        try:
            return json.loads(written.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None


if __name__ == "__main__":
    main()
