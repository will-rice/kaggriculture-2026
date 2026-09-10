<!--
The whole message a round is given, in order, with the parts the campaign
measures left as placeholders. `prompt.compose` reads this file, renders the
tables, and fills them in; nothing else composes a message, so what a codex
call sees is this file plus numbers.

Placeholders, all filled on every round:

  {task}         the game's rules, from `task_prompt.md`
  {name}         the program's name -- a pool name or a database id, never a
                 path, and interpolated raw
  {imports}      the allowed-import list, rendered from the gate's own
                 whitelist so a round is never told a different set from the
                 one that rejects it
  {seeds}        how many seeds the verdict was measured over
  {rates}        one row per pool opponent: rate, mean margin, worst, best
  {verdict}      the gate's own sentence about where this program placed
  {placing}      one line on what that place means, promotion or progress
  {standings}    the tournament table, this program marked
  {states}       one day-by-day game against each opponent that took a game
  {siblings}     other programs written from this one, or empty
  {failures}     the lineage's recent rejected attempts, or empty
  {instruction}  what to do, and under stagnation why the start moved

The corpus-derived sections are gone. `pace`, the build order, and
`claims`, the settled statements about the ladder's winners, were both
true about the corpus and neither earned its place: with the build order
clustered the median candidate scored 0.275 and promotions ran about one
an hour; with the opening added as orders the median fell to 0.026 over
156 gates and nothing was promoted in ten hours. What the model did with
them is what three separate experiments did -- bolt another strategy's
orders onto this one, and break the economy underneath.

A section that would be empty is rendered as nothing at all, heading
included: a heading over an empty list is noise in a message read every
round.
-->

{task}

## The program

`child.py` in your working directory is `{name}`, and it is the only file
there. It is the program to improve.

Edit it in place and stop. Do not run anything and do not report anything back
in your reply: whatever `child.py` holds when you finish is what the campaign
plays, against every opponent below, and the result comes back to you as
another message like this one asking you to improve it again.

`child.py` must stay one self-contained file whose last top-level callable is
`agent(observation, configuration)` -- that is what Kaggle loads. Say in a
docstring at its top what you changed and why. There is no time limit on this
call: take as long as the work needs. Keep the file complete and runnable as
you go all the same, so that what it holds is always something that could be
scored.

Nobody is reading this session. There is no human here to answer a question,
approve a design, choose between options or confirm anything, and nothing you
write in your reply is read by anyone. Any skill or process that would have you
present something and wait for approval before writing code does not apply:
plan as much as you like, but plan and then edit, and never stop to ask. The
edited file is the only thing that leaves this call, and a call that ends
without one is a round the campaign spent on nothing.

The rules above cite probes by filename. Those files are not in your
directory: take their numbers as verified and do not go looking.

## What your program may import

Your program is one file, and that file ships alone: nothing is packaged
beside it. It may import only these modules, and an import of anything else is
rejected before the program is scored.

{imports}

## Doctrine

The program in front of you began as a published agent, and building on
published work is what this competition allows. It is yours to change however
far you like -- rewrite any part of it, or all of it.

Every other opponent is closed. Never read one's source, never ask for it,
never reconstruct it: the gate rejects code that resembles any opponent your
lineage did not start from. You are given their names and what your program
scored against them, and that is the whole of what you may know about them.

## The verdict on `{name}`

Played over {seeds} seeds, both seats, against every opponent in the pool. The
margin is your bank minus theirs at the final state.

| opponent | win rate | mean margin | worst | best |
| -------- | -------- | ----------- | ----- | ---- |

{rates}

It {verdict}.

The bar is a Bradley-Terry tournament, which is how the competition itself
ranks the field: every agent plays every other, one strength per agent is
fitted from all of it at once, and the ranking is what counts. Beating a
strong opponent is worth more than beating a weak one, and one bad matchup is
absorbed rather than fatal -- there is no opponent you must beat, only a field
you must finish above. {placing}

| rank | agent | rating |
| ---- | ----- | ------ |

{standings}

{states}
{siblings}
{failures}

## Your instruction

{instruction}
