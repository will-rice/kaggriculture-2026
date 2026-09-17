<!--
The whole message a round is given. `prompt.compose` fills the placeholders
and nothing else composes a message, so what a codex call sees is this file
plus numbers.

Placeholders, all filled on every round:

  {task}         the game's rules, from `task_prompt.md`
  {imports}      the allowed-import list, rendered from the gate's own
                 whitelist so a round is never told a different set from the
                 one that rejects it
  {game}         the one game this round is feedback on
  {tried}        the edits already made to this program and what they scored,
                 or empty before any has been
  {failures}     the lineage's recent rejected attempts, or empty
  {schedule}     which days of the season the program has already decided
  {instruction}  what to do
-->

{task}

## Your program

`child.py` in your working directory is the program, and `plan.json` beside it
is the strategy it plays. Edit either, in place. Both are read: what the
campaign plays is `child.py` with `plan.json` packed back into it, as one
self-contained file whose last top-level callable is
`agent(observation, configuration)` -- that is what Kaggle loads, and a program
that crashes forfeits every game.

`plan.json` holds three things, and they decide what the agent does on the
board:

- `shops` maps the two shops a map happens to have to a route number. This is
  the smallest change that makes the agent play a different game, and it is 64
  lines.
- `routes` are the paths those numbers name, as tile indices.
- `actions` is the scripted season, step by step: the farmer's move, each
  hand's moves, and the market orders, for 3,982 steps.

It came from a solver, and no round before this one could read it -- it shipped
as a single line of base85 and was left untouched through eight promotions
while the controller around it was rewritten again and again. That is why
children keep drawing with the champion: the two share this file, so they play
the same game, and the gate cannot tell them apart. An edit here is an edit to
what the agent does; an edit to `child.py` alone is an edit to how it is
steered.

Both are worth doing. Measure either the same way.

The directory is yours and is thrown away after this call.

Nobody is reading this session. There is no human to answer a question,
approve a design or confirm anything, and nothing in your reply is read. Plan
as much as you like, then edit, and never stop to ask.

The rules above cite probes by filename. Those files are not here: take their
numbers as verified and do not go looking.

## What your program may import

It ships alone, so it may import only these. Anything else is rejected before
the program is scored.

{imports}

## Opponents

Building on published work is what this competition allows, and the field is
doing it in the open: the agents that beat this program carry Apache-2.0
notices and attribute a shared lineage of published kernels by name.

They are on disk at `/data/kaggriculture/opponents/<name>/main.py`, one
directory per pool opponent, and `{game}` below names the one you are losing
to. Read them. The ones that win are not doing something unguessable -- they
issue about as many sell orders as this program does and move fourteen times
the units through them -- but how they decide that is in the source and not in
any number we can hand you.

If you take Apache-2.0 code, the licence's terms come with it: keep the notice
and the attribution in `child.py`. That is what those kernels themselves do,
and it is the whole of the obligation.

{game}
{tried}
{failures}
{schedule}

## Your instruction

{instruction}
