"""The pre-registered run: is the oracle finding levers, and getting them right?

Two earlier probes put thirteen verdicts to the corpus and none landed on a
form it settles. The second of them asked about days eight to twenty-six, where
only four to eighteen percent of forms settle at all, so most of that is the
question being aimed badly rather than the answer being wrong.

This asks where the corpus can adjudicate. Settled forms concentrate in the
first eight days -- thirty-one percent of them, against four percent at days
sixteen to twenty-two -- so the moments are the opening, across several lost
games. The model still chooses the quantity and the direction; only the day is
chosen for it, and it is chosen where evidence exists rather than where an
answer is wanted.

Registered before running:

    lands on a settled form   chance 31%
    agrees, given settled     chance 50%

Failing the first says it reasons about quantities that do not decide games.
Passing the first and failing the second is worse: checkable, confident and
wrong is the one shape that would poison a round rather than merely waste it.
"""

import asyncio
import json
import pathlib
import subprocess
import tempfile

from kaggriculture.campaign import dataset, harness, roster, strategy
from kaggriculture.campaign.engine.wrapper import Engine, render_private

OURS = pathlib.Path("run/campaign/champions/champion_65.py")
THEIRS = "router2929"
RULES = pathlib.Path("src/kaggriculture/campaign/task_prompt.md")
MODEL = "gpt-5.6-luna"
EFFORT = "medium"
SEEDS = list(range(9_400_001, 9_400_049))
DAYS = tuple(range(9))
GAMES = 4

ASK = """You are diagnosing a two-player farming game that our side lost.

`rules.md` is the game.
`trajectory.json` is the whole season: for every day, twenty-nine measured
quantities for our side and for theirs. Comparing those two columns day by day
is where the game was won and lost.
`state.json` is our seat's complete observation at one moment in the opening,
and the move we played there.

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
other games: which quantity the stronger player holds more or less of, on which
day. The quantity that decided the game is the one where the two columns
diverge, which is not always the quantity this move touched.

The quantity must be exactly one of:
{quantities}
"""


def main() -> None:
    """Gather losses, ask about their openings, and score what comes back."""
    opponent = str(pathlib.Path(roster.path(THEIRS)).resolve())
    games = harness.play(OURS, [THEIRS], SEEDS, workers=8, days=True)
    lost = sorted(
        (game for game in games if game.ours < game.theirs),
        key=lambda game: game.theirs - game.ours,
    )[:GAMES]
    print(f"{len(lost)} narrow losses of {len(games)} games")
    asked = []
    for game in lost:
        trajectory = json.dumps(
            [{"day": r.day, "ours": r.ours, "theirs": r.theirs} for r in game.days],
            indent=1,
        )
        for moment in _replay(game.seed, game.seat, opponent):
            asked.append((moment, trajectory))
    print(f"{len(asked)} moments across their openings\n")
    verdicts = asyncio.run(_ask_all(asked))
    _score(verdicts)


async def _ask_all(asked: list[tuple[dict, str]]) -> list[dict | None]:
    """Every moment in flight together."""
    return await asyncio.gather(*(_ask(moment, path) for moment, path in asked))


def _score(verdicts: list[dict | None]) -> None:
    """Set every claim beside what the corpus already measured."""
    store = strategy.Strategies(strategy.STORE)
    known = {(c.form.quantity, c.form.day): c for c in store.claims}
    tally = {"agree": 0, "disagree": 0, "unsettled": 0, "unusable": 0}
    picked: dict[str, int] = {}
    for verdict in verdicts:
        claim = (verdict or {}).get("claim") or {}
        quantity, day, said = (
            claim.get("quantity"),
            claim.get("day"),
            claim.get("direction"),
        )
        if quantity not in strategy.QUANTITIES or said not in {"more", "less"}:
            tally["unusable"] += 1
            continue
        picked[quantity] = picked.get(quantity, 0) + 1
        measured = known.get((quantity, int(day)))
        if measured is None or not measured.settled:
            tally["unsettled"] += 1
            continue
        corpus = "more" if measured.status == "leads" else "less"
        tally["agree" if corpus == said else "disagree"] += 1
        print(
            f"  day {int(day):>2} {quantity:<16}says {said:<5}corpus {corpus:<5}"
            f"{measured.support:>6} games  {'yes' if corpus == said else 'NO'}"
        )
    usable = sum(v for k, v in tally.items() if k != "unusable")
    landed = tally["agree"] + tally["disagree"]
    print(f"\n{tally}")
    if usable:
        print(f"landed on a settled form: {landed}/{usable} = {landed / usable:.0%}"
              f"   (chance 31%)")
    if landed:
        print(f"agreed, given settled:    {tally['agree']}/{landed} = "
              f"{tally['agree'] / landed:.0%}   (chance 50%)")
    print(f"\nquantities it chose: {dict(sorted(picked.items(), key=lambda kv: -kv[1]))}")


def _replay(seed: int, seat: int, opponent: str) -> list[dict]:
    """Replay one game, keeping the opening's first hour of each day."""
    sources = [str(OURS.resolve()), opponent]
    if seat == 1:
        sources.reverse()
    agents = [harness.load_agent(pathlib.Path(p)) for p in sources]
    arities = [harness.argument_count(agent) for agent in agents]
    engine = Engine(seed=seed)
    conf = harness.configuration()
    wanted = {day * dataset.HOURS for day in DAYS}
    kept = []
    while not engine.done:
        step = engine.step_index
        actions = []
        for player, agent in enumerate(agents):
            view = engine.observation(player)
            action = agent(view, conf) if arities[player] > 1 else agent(view)
            actions.append(action)
            if player == seat and step in wanted:
                kept.append(
                    {
                        "day": step // dataset.HOURS,
                        "observation": view,
                        "private": render_private(engine.state.farms[player]),
                        "played": action,
                    }
                )
        engine.step(actions[0], actions[1])
    return kept


async def _ask(moment: dict, trajectory: str) -> dict | None:
    """Put one opening moment, with its whole season beside it."""
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
