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
  {kept}         every plan change that has survived a gate, or empty
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
- `settings` switches the chassis's nine reactive layers on and off. The
  controller's own table names them and says what each one hooks into.

`attempts.jsonl` beside them is every program this campaign has ever written,
one JSON object per line: `id`, the `from` it was edited from, what it
`changed` in the plan, its `wins` and `margin` against the pool, whether it was
`promoted`, and its `rates` against each opponent by name. It is not read to
you and it is not a suggestion; it is the record, for the question you bring to
it -- whether an edit like the one you have in mind has been measured before,
and against which agent it helped. `grep` it.

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

So the useful unit is a rule and not a step: work out what this program is
getting wrong, and change every step that does it. `plan.json` is ordinary JSON; read it, transform it with a few
lines of Python, write it back.

Nothing limits you to one rule. Eight of them have measured positive
independently, and whether they add up is not known, because they have only ever
been tried one at a time. The market is shared and finite, so they may well
fight: selling more of two goods at once moves both prices against you. Stack
them and find out -- but measure after each one you add rather than at the end,
because a bundle that nets positive can carry a losing rule inside it, and the
gate promotes whole programs. An edit that rides in on a better one is inherited
by every program after it.

`./measure.py --against <episode>` answers both questions your edit has to
answer, in one run. `The game` section gives the episode key.

First, against the matchup: your child and the program you started from each
play that opponent, on the same seasons and both seats, and the difference
between them is reported. That matchup is the one taking the most games off us,
and beating it is the job.

Then, against the program you started from directly. That is the bar the job has
to clear on the way -- a program is promoted for beating what it replaces, so an
edit that helps the matchup and loses to its own parent does not get in.

You do not choose between these. For a long time only the second existed, which
is how this lineage spent twenty-nine promotions without closing a three-percent
gap: rounds were aimed at a matchup and graded against a sibling, so every edit
that helped the matchup measured neutral and was thrown away.

Each comes back in coins, with what your change is worth and how much noise is
in the figure. Read the second number. If it is larger than the first, that run
has told you nothing at all, and the two ways out are more seasons or a bigger
change.

Measurements are not instant and you have to wait for them. `--against` plays
both programs, so the default sixty-four seasons is 256 games and takes about
two minutes; `--seeds 8` is 32 games and takes about fifteen. Open with
`--seeds 8` on one edit, and spend the full block only on something that already
looks worth settling. Do not start several at once -- they share the same cores,
so three at a time is three times slower, not three times more evidence.

A turn that ends while a measurement is still running produces nothing: the
round is thrown away, the edit with it, and the next round starts from where
this one did. Run the command, wait for its output, and read it. If you find
yourself writing that you will review the results shortly, you are about to
waste the round -- wait instead.

Small samples do not merely say less, they mislead. The same edit measured +454
over four seasons, +314 over sixteen, and +255 over sixty-four: the small sample
was wrong about the size and confident about being wrong, because a handful of
seasons that happen to agree is indistinguishable from certainty.

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

{kept}

## Your instruction

{instruction}
