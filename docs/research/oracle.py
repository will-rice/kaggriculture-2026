"""Can a model judge a move in this game, given the state it was made in?

The falsification, before anything is built on it. A game champion_65 lost by
790 coins out of 94,000 is replayed to seven moments; at each, the model gets
the seat's whole observation and the move the champion actually played, and is
asked whether that move was a mistake and what it would play instead.

What this can establish is narrow and worth stating. It cannot prove the model
plays well -- judging one move is not playing a season. What it can show is
whether the model reasons about *this* state or produces game-flavoured filler:
whether it cites the seat's actual holdings, prices and cash, whether its
alternative move is legal, and whether seven independent verdicts are
consistent with each other. A model that fails that is not an oracle, and
finding out costs seven calls rather than seven hundred.

Run on the cheap model on purpose. If the cheap one can do it there is nothing
to discuss; if it cannot, that is a reason to try a better one rather than a
verdict on the idea.
"""

import asyncio
import json
import pathlib
import subprocess
import tempfile

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

That move was played by a strong scripted agent, in a game it went on to lose
by 790 coins out of 94,000 -- close enough that one decision could have
carried it. It may still be a fine move: most moves in a lost game are.

Write `verdict.json` and nothing else, as:

{
  "mistake": true or false,
  "confidence": "low" or "medium" or "high",
  "better_move": {"farmer": [...], "hands": [[...]], "market": [[...]]},
  "why": "at most three sentences, citing the numbers in this state",
  "what_it_costs": "what the difference is worth by the end of the season"
}

`better_move` uses the same shape as the played move and must be legal from
this exact state -- affordable at these prices, on tiles this farm holds. If
the played move is right, repeat it and set "mistake" to false.

Cite the state. A verdict that would read the same for any turn of any game is
worth nothing here.
"""


def main() -> None:
    """Ask the model about every kept moment, and report what came back."""
    moments = json.loads(MOMENTS.read_text(encoding="utf-8"))
    print(
        f"seed {moments['seed']}, lost {moments['final_ours']:,.0f} to "
        f"{moments['final_theirs']:,.0f} against {moments['opponent']}"
    )
    verdicts = asyncio.run(_ask_all(moments["moments"]))
    print(f"\n{'day':>4} {'hour':>5}  {'mistake':<9}{'conf':<8}why")
    for moment, verdict in zip(moments["moments"], verdicts, strict=True):
        if verdict is None:
            print(f"{moment['day']:>4} {moment['hour']:>5}  -- no verdict --")
            continue
        print(
            f"{moment['day']:>4} {moment['hour']:>5}  "
            f"{str(verdict.get('mistake')):<9}{verdict.get('confidence', '?'):<8}"
            f"{verdict.get('why', '')[:150]}"
        )
    kept = [v for v in verdicts if v]
    print(f"\n{len(kept)} of {len(verdicts)} calls returned a verdict")
    flagged = [v for v in kept if v.get("mistake")]
    print(f"{len(flagged)} called a mistake")
    for verdict in flagged:
        print(f"\n  costs: {verdict.get('what_it_costs', '')[:200]}")
        print(f"  plays: {json.dumps(verdict.get('better_move'))[:200]}")


async def _ask_all(moments: list[dict]) -> list[dict | None]:
    """One call per moment, all in flight together."""
    return await asyncio.gather(*(_ask(moment) for moment in moments))


async def _ask(moment: dict) -> dict | None:
    """Put one moment to the model and read back its verdict file."""
    with tempfile.TemporaryDirectory(prefix="oracle-") as box:
        room = pathlib.Path(box)
        (room / "rules.md").write_text(
            RULES.read_text(encoding="utf-8"), encoding="utf-8"
        )
        (room / "state.json").write_text(json.dumps(moment, indent=1), encoding="utf-8")
        process = await asyncio.create_subprocess_exec(
            "codex",
            "exec",
            "--skip-git-repo-check",
            "-s",
            "workspace-write",
            "-c",
            "approval_policy=never",
            "-c",
            f"model_reasoning_effort={EFFORT}",
            "-m",
            MODEL,
            "-C",
            str(room),
            "-",
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=room,
            start_new_session=True,
        )
        await process.communicate(ASK.encode())
        written = room / "verdict.json"
        if not written.exists():
            return None
        try:
            return json.loads(written.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None


if __name__ == "__main__":
    main()
