"""How often is the oracle right, measured against what the corpus already knows?

The seven-call probe showed a model that reasons about the state it is given
and discriminates between moves. It also showed its most confident, most
repeated diagnosis -- that champion_65 under-hires -- to be unsupported: the
champion tracks the ladder's top twelve within seven-tenths of a hand at every
mark.

That is the shape that decides whether this can be feedback. Wrong at random is
survivable, because the gate turns away candidates that act on bad advice.
Wrong the same way every time, fluently and with arithmetic, is not: it pushes
every round in one direction with a persuasive reason.

So the question is calibration, and it is answerable without any new
measurement. `evidence.questions()` already crosses every quantity with every
day, so whatever the oracle claims has been measured over thousands of paired
games. Ask it to state its claim in that vocabulary, look the claim up, and
count how often the corpus agrees.
"""

import asyncio
import json
import pathlib
import subprocess
import tempfile

from kaggriculture.campaign import strategy

MOMENTS = pathlib.Path(
    "/tmp/claude-1000/-home-will-projects-kaggriculture-2026/"
    "521a4f31-d72e-47e3-adbb-c5944812e9ba/scratchpad/loss.json"
)
RULES = pathlib.Path("src/kaggriculture/campaign/task_prompt.md")
MODEL = "gpt-5.6-luna"
EFFORT = "medium"

ASK = """You are judging one move in a two-player farming game.

`rules.md` is the game. `state.json` holds one seat's complete observation at a
single step, and the move that seat actually played there.

That move was played by a strong scripted agent in a game it went on to lose
narrowly. It may still be a fine move: most moves in a lost game are.

Write `verdict.json` and nothing else:

{
  "mistake": true or false,
  "why": "at most two sentences, citing the numbers in this state",
  "claim": {
    "quantity": "one of the quantities listed below, exactly",
    "day": <the day number this is about>,
    "direction": "more" or "less"
  }
}

`claim` is the general lesson, stated so it can be checked against thousands of
other games: which measurable quantity the stronger player should hold more or
less of, on which day. If the played move was right, still give the claim your
reasoning rests on.

The quantity must be exactly one of:
{quantities}

Cite the state. A verdict that would read the same for any turn of any game is
worth nothing.
"""


def main() -> None:
    """Ask about every moment, then score each claim against the corpus."""
    store = strategy.Strategies(strategy.STORE)
    known = {(claim.form.quantity, claim.form.day): claim for claim in store.claims}
    moments = json.loads(MOMENTS.read_text(encoding="utf-8"))["moments"]
    verdicts = asyncio.run(_ask_all(moments))

    print(f"{'day':>4} {'quantity':<16}{'says':<7}{'corpus':<9}{'games':>7}  agree?")
    scored = {"agree": 0, "disagree": 0, "unsettled": 0, "unusable": 0}
    for moment, verdict in zip(moments, verdicts, strict=True):
        claim = (verdict or {}).get("claim") or {}
        quantity, day = claim.get("quantity"), claim.get("day")
        said = claim.get("direction")
        if quantity not in strategy.QUANTITIES or said not in {"more", "less"}:
            scored["unusable"] += 1
            print(f"{moment['day']:>4} {str(quantity)[:15]:<16}{str(said):<7}unusable")
            continue
        measured = known.get((quantity, int(day)))
        if measured is None or not measured.settled:
            scored["unsettled"] += 1
            support = measured.support if measured else 0
            print(
                f"{int(day):>4} {quantity:<16}{said:<7}{'open':<9}{support:>7}  --"
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
        print(f"calibration on settled claims: {scored['agree']}/{settled} "
              f"= {scored['agree'] / settled:.0%}")
    print("chance is 50%: a coin agrees half the time")


async def _ask_all(moments: list[dict]) -> list[dict | None]:
    """One call per moment, all in flight together."""
    return await asyncio.gather(*(_ask(moment) for moment in moments))


async def _ask(moment: dict) -> dict | None:
    """Put one moment to the model and read back its verdict."""
    with tempfile.TemporaryDirectory(prefix="oracle-") as box:
        room = pathlib.Path(box)
        (room / "rules.md").write_text(
            RULES.read_text(encoding="utf-8"), encoding="utf-8"
        )
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
