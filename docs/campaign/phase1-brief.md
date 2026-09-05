# Phase 1: how Kaggriculture works

You are in a workspace containing `engine/kaggriculture.py` (the exact engine,
kaggle-environments 1.32.7 — read it as ground truth), `docs-competition.md`
(notes, some stale; the engine wins any disagreement), and a harness. Run the
harness from this directory as:

    uv run --project /home/will/projects/kaggriculture-2026/.claude/worktrees/campaign campaign play AGENT --vs NAME... --seeds A-B --workers N
    uv run --project /home/will/projects/kaggriculture-2026/.claude/worktrees/campaign campaign check AGENT

Opponent names: router_v1, router2929, v54, v56, shopforge, indarkarhana.
You may measure opponents through the harness; you may not read their source.
For experiments against the engine directly, `uv run --project <same path> python`
gives you `kaggle_environments` (`from kaggle_environments import make; env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 1})`).

Write `notes/game-model.md`: how a season works (turns, days, hours, the
end-of-day transition), every unit action and what it costs and yields,
crops (seed cost, first yield day, interval, max yield, ongoing or not),
animals, structures, land, hiring (Fibonacci pricing), the market (base
prices, I0, the price curve shapes including the hinge, price impact of a
sale, town-centre and shop consumption, shop unlock schedule and the seed's
role), the shed cap and what it silently destroys, what the two players share
(market, town) and what each sees of the other (`farms[1-player]`), and what
the win condition rewards (relative bank at turn 720; win/loss only).

Every number must come from a script under `experiments/` that you ran
against the engine, and the note must cite the script. Then write
`task_prompt.md`: the same content compressed to what a policy author needs
in one read — objective, rules, interface (`agent(observation, configuration)`,
the last callable in the file is what runs, 1 s per step, the action dict
shape), verified economics, and the doctrine (measure opponents through the
harness; never read their source). Mirror the shape of a competition task
prompt: objective first, key insights second, rules third, interface last.
Do not build an agent yet.
