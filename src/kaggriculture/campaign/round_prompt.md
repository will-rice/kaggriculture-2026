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

`plan.json` holds four things, and they decide what the agent does on the
board:

- `actions` is a pool of every distinct step any season plays: the farmer's
  move, each hand's move, and the market orders. Where a step sits in the pool
  means nothing. One step per line, so `actions[N]` is line `N + 3` of the file
  and you can go straight to it.
- `routes` is one whole season per route number: 719 indices into `actions`,
  one per step, so entry N says which pooled step the farm plays on step N.
  These are not board tiles. A step the farm plays forty times is one pooled
  entry cited forty times. One season per line.
- `shops` maps the two shops a map happens to have to a route number, so it
  chooses which season gets played. 64 lines, and the smallest change that
  makes the agent play a different game.
- `settings` switches the chassis's nine reactive layers on and off: selling
  ahead of the opponent, liquidating at the end, funding a block before it
  spends. These were in `child.py` until today, where no round editing the plan
  could reach them. All 512 combinations have now been played and the current
  one is the best, so this part is a settled question rather than somewhere to
  look.

So there are two kinds of edit. Changing a pooled step changes that step
everywhere every season cites it. Changing a season's indices changes the order
without touching a step. Both are real edits; neither is the other.

## How big an edit has to be to be worth making

The margins that decide these games are small next to the banks. The champion
banks around 100,000 and beats the field by between 279 and 3,242. So an edit
worth keeping has to be worth hundreds, and most edits are worth nothing:

- One quantity changed on one pooled step was measured at **+6**. Twenty-two of
  twenty-four such edits changed the score by nothing at all, because a game
  plays one route and any one route cites only 18% of the pool.
- The same change applied to _every_ pooled step matching a rule -- every
  `SELL MELON`, capped -- moved **1,269** on a single seed. A grep for
  `SELL MELON` reaches 61 pooled steps deciding 543 step-slots; `HIRE` reaches
  400 steps and 2,169 slots.

Edit by rule, not by point. Work out what is wrong from the game's rules --
which goods collapse on a glut, when a hire pays for itself -- then change every
step that does it. `plan.json` is ordinary JSON: read it, transform it with a
few lines of Python, write it back.

And measure enough seasons. `./measure.py` defaults to 64 of them, 128 games,
in about fifteen seconds, and it tells you in coins whether what you measured is
bigger than the noise in the measurement. Take the default. A round before you
ran `--seeds 4` four separate times, was told four times that the result said
nothing, and shipped anyway; the gate then spent twenty minutes establishing
what those four runs had already said for free.

Small samples do not merely say less, they mislead. The same edit measured +454
with a spread of 13 over four seasons, +314 over sixteen, and +255 over
sixty-four: the small sample was both wrong about the size and falsely confident
about it, because four seasons that happen to agree look like certainty.

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
