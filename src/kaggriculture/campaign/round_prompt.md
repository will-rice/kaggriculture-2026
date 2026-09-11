<!--
The whole message a round is given, in order, with the parts the campaign
measures left as placeholders. `prompt.compose` reads this file and fills them
in; nothing else composes a message, so what a codex call sees is this file
plus numbers, and one file beside `child.py` that this message points at.

Placeholders, all filled on every round:

  {task}         the game's rules, from `task_prompt.md`
  {imports}      the allowed-import list, rendered from the gate's own
                 whitelist so a round is never told a different set from the
                 one that rejects it
  {seeds}        how many seasons it was measured over
  {margin}       how far behind it finished on average, across all of them
  {states}       the index of `seasons.csv`: one line per matchup it played
  {failures}     the lineage's recent rejected attempts, or empty
  {instruction}  what to do

What is *not* here is the point of the shape. A round used to be told its own
name, its rank in a tournament of the pool, the gate's verdict on where it
placed, a row per opponent with a win rate against each, and what sibling
rounds had already tried. All of that framed the task as climbing a named
ladder, and the campaign optimised exactly that: the previous lineage evolved
opponent fingerprinting, recognising specific agents by their sheep and cow
counts, which is the right answer to "beat this pool" and worth nothing on a
ladder where the agent across the table has never been seen before.

The objective is a program that beats *any* opponent. The pool is a sample of
the field used to estimate that, never a set of targets, so no opponent is
named anywhere in this message and neither is the program itself.

The seasons themselves are a file, not a section. Rendered into the message
they were 74% of it, at 30 rows of fifteen columns per game, and the fifteen
were all a markdown table can be read at -- so cutting the message meant
cutting to one game, and cutting the evidence with it. Worse, the evaluation
had already dropped thirty-one of every thirty-two games before the message
was composed: a round was asked to improve a program on one game in
thirty-two of what it would be scored on.

Written to `seasons.csv` instead, every scored game is there at full width:
all 29 quantities `dataset.measures` defines, for both sides, plus the
per-crop breakdowns and the market. Two keys order it. A `matchup` is one
opponent and every season played against them -- fixed opponent, so what
varies between its seasons is the map, the prices and the seat, which is the
variation a general program has to hold up across. Between matchups the
adversary varies too, so a difference there says nothing about either.

The message carries the index of that, because how many seasons a matchup
holds and how they went is the part a round cannot work out for itself.

A section that would be empty is rendered as nothing at all, heading
included: a heading over an empty list is noise in a message read every
round.
-->

{task}

## The program

`child.py` in your working directory is the program to improve.

Edit it in place. Do not report anything back in your reply: whatever
`child.py` holds when you finish is what the campaign plays, against
opponents you never see, and the result comes back to you as another message
like this one asking you to improve it again.

## Work one season at a time

`parent.py` beside it is the same program before you touched it, and
`measure.py` plays one against the other. Do not work from the totals. A total
tells you whether an edit helped and never where, and a program is improved by
finding one thing it does badly on one day and fixing that. So take one season,
work it until it is better, and then take the next:

**1. Pick the season.** `python measure.py` plays all sixteen in both seats,
prints a line for each, and names the one this program does worst on. The
seasons the campaign scored are in `seasons.csv`, hardest matchup first.

**2. Read it day by day.** `python measure.py --replay 103` writes
`replay-103.csv` -- one row per day, the same columns as `seasons.csv` -- and
says which day the gap moved most against you. Open that day and the few
before it. What did the other side hold that you did not? Bare tiles, seed
sitting unplanted, a quadrant never bought, animals never fed: the columns are
there for both sides.

**3. Change one thing.** The one thing that day pointed at.

**4. Replay the same season.** `python measure.py --replay 103` again. The seed
is fixed, so it is the same map, the same prices and the same opponent, and
what moved is your edit and nothing else. Did the day you were aiming at get
better?

**5. Check it cost nothing elsewhere.** `python measure.py` over all sixteen.
Keep the edit if it is ahead by more than twice its error, put it back if it is
behind. Inside the error it has told you nothing -- play more seeds with
`--seeds`, or make a bigger change. An edit that wins one season by twenty
thousand and loses three by six is luck on one map, not an improvement; the
per-season lines are printed so you can see which you have.

Then go back to 1 with the next season. Several small changes, each measured on
the season that motivated it, beat one large change measured once.

The pairing is what makes these numbers worth reading. A fixed plan's bank
swings about 19.5% from season to season, so two programs played on different
seasons are mostly being compared on their luck -- telling apart a
five-thousand-coin difference that way takes about 114 games. Played on the
same seasons in both seats, the luck lands on both sides and cancels, and the
same difference shows up in about four. You are measuring more sharply here
than the gate that will judge you does.

What it prints is not the verdict. The campaign plays every scored game itself,
against opponents you never see, and that is what promotes a program. This is
for deciding whether an edit is worth submitting to it. Most ideas are worse
than what is already there, and finding that out here costs a few seconds
instead of a whole round.

## Check your work

The directory is yours and it is thrown away after this call, so write whatever
scratch files help -- a test, a probe, a script that plays a few turns -- and
leave them there; only `child.py` is read. Two things are worth doing every
round:

    ruff format child.py
    ruff check --select C,E,F,I,W,D,N,B,PTH,ANN --ignore D107 child.py

Both are on your path. The flags are spelled out because there is no config
file in this directory. Fix what `check` reports rather than silencing it: a
round is handed this whole program as text, so how readable it is decides how
much of the next call goes into reading it rather than improving it. `C901`
is the one that matters most and the one the program is worst at -- `agent`
arrived at 111 branches against a limit of 10. Splitting it up is welcome work
in its own right.

Then test what you changed. Write a small test beside `child.py`, run it with
`pytest`, and make it fail before you make it pass -- a test that passes
against the bug it was written for is worse than none, because it reports the
bug as fixed. Import `child` and call `agent` with an observation you build
yourself. A change that has never been executed is a guess, and a round that
ships a crash scores nothing at all: the program forfeits every game.

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
lineage did not start from. You are not told who they are, and nothing in this
message names one. You are writing a program that has to beat an opponent it
has never seen, so recognising a particular one would be worth nothing even if
you could.

## How this program played

It played {seeds} seasons in both seats against each opponent drawn from the
field, and finished {margin} behind on average. Every one of those games is in
your directory, day by day.

{states}
{failures}

## Your instruction

{instruction}
