# Campaign consolidation plan

2026-09-25. Five days before the deadline (2026-09-30), so every item below is
staged by what it risks against a loop that is currently producing champions.
The survey behind it read every module under `src/kaggriculture` (16,031 lines
in 45 files) rather than the ones touched this week; the scripts that produced
each number are named so the numbers can be re-run.

## 1. The charge, measured

The system reads as patches because it is one: 738 commits, 125 of them in the
last fourteen days, and 108 places in 22 modules where a comment records a
decision that was later reversed ("used to", "until 2026-", "is gone", "there
was a second") -- `survey.py`. The four most-churned files in thirty days are
`loop.py` (80 commits), `config.py` (61), `prompt.py` (50) and `gate.py` (34).

Churn is not itself the problem. The problem is what churn leaves behind when
each change adds a surface and none removes one. The seams below are where two
designs met and both survived.

## 2. Findings, by seam

Each carries the evidence, the cost today, and the consolidation. They are
ordered by how much the evidence supports acting, not by size.

### A. The persisted state carries twelve thousand games with their day tables

`Result.states` holds every game an evaluation played, with its thirty-row day
table (`evaluator.py:343-349`: all games, sorted by margin, per opponent). The
only reader is `games.record`, which writes them to ClickHouse
(`games.py:497,509`; `grep -rn "\.states\b"` finds nothing else). Yet the
`Result` rides inside `Champion.result`, which rides inside `State.champion`,
which `finish()` serialises after **every session**
(`loop.py:1653-1659`, `model_dump_json(indent=2)`).

Measured: `run/campaign/state.json` is **688 MB**, `champion.json` **646 MB**
(`ls -la run/campaign/*.json`). The games inside them are already in the
`games` database, which is where anything that wants a day table reads it.

Consolidation: exclude `states` from what is persisted -- record the games,
then drop the tables. `state.json` and `champion.json` become kilobytes; the
per-session write stops costing seconds and disk. No behaviour changes: the one
reader runs before persistence.

### B. A Bradley-Terry fit per candidate that nothing decides on

`loop.py:1382` says it plainly: "nothing reads that fit any more: the bar is a
win rate and a head-to-head." Yet `loop.py:1484` still fits
`rating.standings(rating.Field.load(...))` for every candidate measured;
`gate.refresh` and `gate.standing` maintain `field.json` for it
(`gate.py:139,155,203`); the fit lands in `Program.rating`, `Program.place`,
and four wandb series (`calls/rating`, `gate/rating`, `champion/rating`,
`database/top_rating`).

Consolidation: the fit stays as a diagnostic in `scripts/champions.py`, which
is the one place the lineage is rated on fixed files. The per-candidate fit,
`field.json`, the two `Program` fields and the four series go.

### C. Five measures of "good"; the gate uses two

A program carries `fitness` (pool win rate), `field` (win rate over 14 vendored
incumbents, `evaluator.VENDORED = list(roster.TRAINING)`), `rating`, `place`
and `margins`. The gate promotes on `rates`, `margins` and the head-to-head
(`gate.promotion`, three conditions). Over the last 60 programs `field` takes
**14 distinct values** where `fitness` takes 37 and mean margin 55
(`gradient.py`). The prompt shows a round `fitness` only, though margin is the
measure with resolution and the one the gate scores by.

Consolidation: `Program` keeps `rates` and `margins`; `field`, `rating`,
`place` go (with B). The prompt shows win rate _and_ margin and says which the
gate promotes on. `roster.TRAINING` -- a hand-kept dict of fourteen paths from
2026-09-01 -- survives only as the pool's cold-start seed.

### D. Three drivers, one shape

`mutate.py` is 1,652 lines: `CodexMutator`, `AgyMutator`, `OpenCodeMutator`,
each with `__init__`, `__call__`, `invocation`, `call` (~1,000 lines between
them), plus `Rotating` and `Spent`. Its own docstring says why: "they exist side
by side because an entitlement ran out rather than because either is better."
Twelve `config` constants are read by `mutate.py` alone (`constants.py`).

Consolidation: one driver with a per-provider table (binary, flags, transcript
globs, quota phrases, model catalogue); `Rotating` unchanged. This is the module
producing every candidate, so it waits for the deadline.

### E. Two harvesters and a second gate

`harvest.py` (278 lines) calls `kernel_watch.py` (659 lines) for four things:
`discover`, `remember`, `build_compiled`, `extract` (`harvest.py:173-274`).
`kernel_watch` also carries its own `main()` (not an entry point) and its own
`gate()` (`kernel_watch.py:617`), which scores against `roster.TRAINING` through
`field_gate.score_field` -- a 43-line module whose sole caller this is. That is
a second gate with a second field concept beside `gate.py`.

Consolidation: the four functions move into `harvest`; `kernel_watch.main`,
`kernel_watch.gate` and `field_gate` are deleted with their tests. About 500
lines.

### F. Three corpus modules and a superseded store

`tapes` reads archives; `dataset` parses them into the database; `games` owns
the schema and writer; `losses` reaches across the boundary to `dataset._rows`
(`losses.py:210`). `dataset.py`'s docstring still says "parsed once into
SQLite" and `scripts/extract_corpus.py` says the same and tells the reader to
open `corpus.sqlite` -- both call `games.create` and write ClickHouse
(`dataset.py:186,223`). SQLite appears nowhere else in the package.

Consolidation: `dataset.rows` becomes public; the two docstrings say ClickHouse;
`extract-corpus` is renamed for what it does or folded into `dataset.build`.
`tapes` stays.

### G. A research subsystem the campaign never reads

`evidence.py` (166), `strategy.py` (299) and `scripts/strategies.py` (80) write
`strategies.jsonl`, "which the round prompt reads" (`strategies.py:26`).
`prompt.py` does not read it (`grep strategies prompt.py`: one unrelated
comment); nothing in `campaign/` imports `evidence` or `strategy` except each
other. 545 lines plus `test_evidence` and `test_strategy`.

Consolidation: move to `docs/research/` as the record of what was learned, or
delete. Not part of the loop either way.

### H. Dead root modules

`actions.py`, `observation.py` and `replay.py` at the package root are imported
by nothing in the package (`survey.py`, import graph). They are the hand-written
agent's helpers from before the campaign. `served/main.py` is named by
`config.SERVED`, which no module reads; `gate.py:44` calls it the cold-start
seed. `seed/main.py` (925 lines) is a harvested agent with a tuning-note
docstring and no importer.

Consolidation: delete the three root modules and `tests/test_replay.py`. Keep
exactly one cold-start file and name it once.

### I. References to things that no longer exist

The copy gate was removed on 2026-09-15 (`validate.py:401`), and `copycheck` is
still cited as live in `loop.py:96`, `harvest.py:135` and `roster.py:8`.
`validate.validate` keeps a `seed` parameter "for callers that still name it;
nothing reads it." The 108 reversal marks are the same pattern at scale: the
losing design is documented beside the winner.

Consolidation: a docstring pass that keeps the _decision_ and drops the
_history_ -- git holds the history. Remove the dead parameter.

### J. The proposer's surfaces, which is where this week's patches landed

A round is told what it wants to know in up to four places each:

| a round wants to know | answered in                                                                                           |
| --------------------- | ----------------------------------------------------------------------------------------------------- |
| what has been tried   | `_tried_lines` (prompt), `tried_N.py` (files), `attempts.jsonl` (uncommitted)                         |
| what worked           | `_kept_lines` -- re-reads and re-splits every champion file, every round -- and now `Program.changed` |
| where a season went   | `--replay` (one season, CSV), `_walked` (pooled, new), `games.days`, `losses.deficit`                 |
| where we lose         | the prompt's live query, `losses.deficit`                                                             |

Consolidation: the archive is the record and `attempts.jsonl` is its only
round-facing projection. `_kept_lines` and `_tried_lines` render from it
instead of from files and recomputed diffs. `field` leaves the prompt (C).
`--replay` and `_walked` become one day table.

### K. `loop.py` holds what other modules own

1,781 lines. `Campaign` has 17 methods. Beside it: `_kept`/`_restored`
(workspace protection), `_packed` (repacking a plan -- `plan.gather`'s job),
`_program`/`_changed` (building an archive record -- `archive`'s job),
`_open_run` (wandb), `seasons`/`must_play` (pool sampling -- `pool`'s job).

Consolidation: move by ownership. `loop` keeps `drive`, `work`, `session`,
`round`, `keep`, `consider`, the two timers.

### L. `config.py`: 54 constants, 35 read by one module

`constants.py`: twelve are read only by `mutate.py`, twelve only by `loop.py`,
one (`SERVED`) by nothing. `config.py` is 766 lines, most of it the reasoning
behind a number.

Consolidation: a constant read by one module lives in that module. `config`
keeps paths and anything two modules share.

### M. One 441 KB line in the program a round edits

`_V92_P_BLOB` (the price-prototype library) is 440,834 bytes of a 754,659-byte
controller whose median line is 43 bytes (`bench_days.py` survey). `diff` and
`rg` return it whole: 20 of 41 rounds had a single command return over 100 KB,
the largest 1,021 KB (`blowouts.py`).

**Not shown to cost anything.** Rounds that read more made _more_ edits and
measurements, not fewer (`within.py`); rounds that read nothing were dead
rounds from an expired token (`dead.py`). This is a design fix, not a
throughput fix, and it is filed accordingly.

Consolidation: the same `split`/`join` the plan already uses -- `lay_out`
writes the blob beside `plan.json`, `gather` packs it back.

## 3. The plan

The rule that would have prevented most of section 2: **no new surface unless
it removes one.** Applied this week it would have stopped two of four additions.

### Tier 0 -- now. No behaviour changes; each is a deletion or a docstring.

|     | item                                                       | lines | verified by                                   |
| --- | ---------------------------------------------------------- | ----- | --------------------------------------------- |
| A   | `states` out of persisted state                            | ~10   | `state.json` < 1 MB; `test_loop` resume tests |
| I   | stale `copycheck` references, dead `seed` param            | ~20   | grep                                          |
| G   | move `evidence`/`strategy`/`strategies` out of the package | -545  | suite                                         |
| H   | delete `actions`, `observation`, `replay`, `test_replay`   | -340  | suite                                         |
| F   | `dataset.rows` public; two docstrings say ClickHouse       | ~10   | grep                                          |

### Tier 1 -- this week. Low risk, existing tests cover the seams.

|                 | item                                                                         | notes                                                           |
| --------------- | ---------------------------------------------------------------------------- | --------------------------------------------------------------- |
| J               | `attempts.jsonl` as the one projection; `_kept_lines`/`_tried_lines` read it | finishes the uncommitted work as a replacement, not an addition |
| C (prompt half) | win rate and margin in the message, `field` out                              | `test_prompt`                                                   |
| L               | single-reader constants move to their reader                                 | mechanical; `test_config`                                       |

### Tier 2 -- only behind a measurement.

|     | item                                                    | the measurement that would justify it                      |
| --- | ------------------------------------------------------- | ---------------------------------------------------------- |
| M   | split the blob out of `child.py`                        | a comparison where reading less produces more; none exists |
| E   | fold `kernel_watch` into `harvest`, delete `field_gate` | harvest is hourly and working; touch it when it fails      |

### Tier 3 -- after 2026-09-30.

|                 | item                                                     | why it waits                                                              |
| --------------- | -------------------------------------------------------- | ------------------------------------------------------------------------- |
| B               | drop the per-candidate BT fit, `field.json`, four series | touches `gate.refresh`; a gate change during scoring is how a day is lost |
| C (record half) | `Program` loses `field`, `rating`, `place`               | depends on B                                                              |
| D               | one driver in `mutate.py`                                | it is the module producing every candidate                                |
| K               | `loop.py` by ownership                                   | 2,711 lines of tests to move with it                                      |

## 4. What this plan does not do

Five hypotheses were tested on 2026-09-25 and refuted by the data
(`budget.py`, `cost.py`, `within.py`, `dead.py`, `retention.py`,
`quality.py`, `gradient.py`):

- rounds lack evidence (queries are 2.6% of what they do);
- big reads crowd out work (heavier readers did more);
- dead rounds are a current 20% loss (all eight were 2026-09-22, before re-auth);
- the search does not build on unpromoted work (272 of 489 programs do, to depth 19);
- fitness is pinned at 1.000 (0 of the last 60 are).

So nothing here is justified as a throughput fix. The candidates are fine (38%
beat their parent, `quality.py`); the gate is strict by design. The
consolidation is worth doing because a system with one surface per question is
one a person can hold in their head, and because 1.3 GB of JSON written per
session is a bug regardless of what it costs.

Nothing here gates on a day-indexed figure. `losses/gap_days20_29` and
`_walked` are locators; day-10 bank once trended across a selected chain of
champions without being the mechanism.

## 5. The tree as this was written

Uncommitted: `archive.Program.changed`, `archive.Database.attempts`, and
`loop._changed` with the parent threaded through `keep` -- 111 insertions. It
is Tier 1's foundation and is held until this plan is read, so that it lands as
a replacement for two prompt surfaces rather than a third beside them.
