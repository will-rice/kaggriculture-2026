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

`child.py` in your working directory is the program. Edit it in place: it is
the only file read, and whatever it holds when you finish is what the campaign
plays. It must stay one self-contained file whose last top-level callable is
`agent(observation, configuration)` -- that is what Kaggle loads, and a
program that crashes forfeits every game.

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
