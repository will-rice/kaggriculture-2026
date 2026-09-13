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

The program began as a published agent, and building on published work is what
this competition allows. Every other opponent is closed: never read one's
source, ask for it, or reconstruct it -- the gate rejects code resembling any
opponent your lineage did not start from.

{game}
{tried}
{failures}

## Your instruction

{instruction}
