# kaggriculture-2026

Agent for the [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture)
simulation competition: two players farm for 30 in-game days and the one with
the most coins banked wins.

See [docs/competition.md](docs/competition.md) for the format, the rules the code
relies on, and the places where the engine disagrees with the documentation.

## The campaign

The agent is no longer hand-written: a codex-driven search plays candidate
agents against a held-out field, keeps the ones that validate, and promotes
what wins to the campaign floor, `run/campaign/floor/agent/main.py`, which is
what `uv run package` ships. See
[docs/superpowers/specs/2026-09-05-campaign-script-design.md](docs/superpowers/specs/2026-09-05-campaign-script-design.md)
for the design and
[docs/superpowers/plans/2026-09-05-campaign-script.md](docs/superpowers/plans/2026-09-05-campaign-script.md)
for how it was built.

| path                                                           | responsibility                                                                                                           |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `src/kaggriculture/campaign/loop.py`                           | the script: the event loop, `SESSIONS` workers, a session's rounds, and the gate they fire                               |
| `src/kaggriculture/campaign/workspace.py`                      | the directory a round works in: laid out, guarded against edits to the campaign's own source, read back                  |
| `src/kaggriculture/campaign/mutate.py`                         | one `Driver` under codex, agy and opencode; a rotation that hands a call to the next program when one is out of quota    |
| `src/kaggriculture/campaign/prompt.py`                         | composes the message a round is given: the rules, its games, the edits already tried on this program, one instruction    |
| `src/kaggriculture/campaign/plan.py`                           | the plan a program carries, laid out as `plan.json` for a round and packed back afterwards                               |
| `src/kaggriculture/campaign/measure.py`                        | copied into every round as `measure.py`: plays `child.py` against its parent, one season at a time                       |
| `src/kaggriculture/campaign/validate.py`                       | `validate(agent) -> Verdict`: syntax, contract, imports, a full game through Kaggle's own loader                         |
| `src/kaggriculture/campaign/evaluator.py`                      | the one measurement: fresh seeds, both seats, against the whole pool; a program never plays itself                       |
| `src/kaggriculture/campaign/gate.py`                           | `promotion`: no worse against the field on rate and margin, and a decisive head-to-head. `promote`: tarball, copy, floor |
| `src/kaggriculture/campaign/pool.py`                           | the opponents, counting equally; harvested agents never leave, champions join                                            |
| `src/kaggriculture/campaign/roster.py`                         | opponent names → paths (never exposed): the vendored kernels the pool starts from                                        |
| `src/kaggriculture/campaign/archive.py`                        | the shared database: every program with its rates and margins, every failure, every promotion, as an append-only log     |
| `src/kaggriculture/campaign/harness.py`                        | `play`, `check`, `package`; the `campaign` CLI                                                                           |
| `src/kaggriculture/campaign/pools.py`                          | where somebody else's program runs and where it may write                                                                |
| `src/kaggriculture/campaign/arena.py`                          | reference-engine games between two agent files, both seats, over a process pool                                          |
| `src/kaggriculture/campaign/engine/`                           | the C++ engine port, its bridge, and the `Engine` wrapper                                                                |
| `src/kaggriculture/campaign/games.py`                          | the ClickHouse games database: the recorded ladder and every game the campaign plays                                     |
| `src/kaggriculture/campaign/dataset.py`, `tapes.py`            | archived episodes parsed into that database                                                                              |
| `src/kaggriculture/campaign/losses.py`                         | the games this lineage lost, kept current for the round prompt's live query                                              |
| `src/kaggriculture/campaign/harvest.py`, `kernel_watch.py`     | the ladder scan that vendors newly published kernels into the pool                                                       |
| `src/kaggriculture/campaign/rating.py`                         | a Bradley-Terry fit, used only to rate the public corpus and the champion lineage                                        |
| `src/kaggriculture/campaign/telemetry.py`                      | the wandb run, named for the git revision                                                                                |
| `src/kaggriculture/campaign/config.py`                         | the paths, and the constants two modules share                                                                           |
| `src/kaggriculture/campaign/task_prompt.md`, `round_prompt.md` | the game as a round is told it, and how a round is asked                                                                 |
| `src/kaggriculture/seed/main.py`                               | the cold-start seed the loop begins from; not what ships                                                                 |
| `src/kaggriculture/scripts/package.py`, `submit.py`            | shipping the floor                                                                                                       |
| `tests/campaign/*.py`                                          | one test module per source module                                                                                        |

### Running the campaign

```bash
uv run campaign dry-run --sessions 2      # fake mutator, proves the pipeline
nohup uv run campaign loop > run/campaign/loop.log 2>&1 &
tail -f run/campaign/loop.log            # metrics: wandb.ai/will-rice/kaggriculture-2026
```

The loop is one `asyncio` event loop running `SESSIONS` workers against one
database, one pool and one champion. A worker takes the champion and runs a
session on it: each round composes a message, hands codex a directory holding
one file, `child.py`, and the message on standard input, then validates what
codex wrote, plays it against every pool opponent, inserts it, and sends the
result back for another round. A round continues from the best program its
session has produced, not the latest one: a round that lost ground hands the
next one the program it started from, and that program's rejected children
reach the next round as `tried_1.py` and so on beside their scores. A new
session starts again from the champion. A session runs
`ROUNDS_PER_OPPONENT` consecutive rounds on each pool opponent in turn and
ends when a round clears the promotion gate.

The model is a mutation operator: it plays nothing and measures nothing, so
every game goes through the one pool that knows how many cores there are, and
the verdict it is sent is the same rule the promotion gate applies. There is
one measurement, over `GATE_SEEDS` fresh seeds, and a program becomes the
champion by clearing two bars on it: a higher mean win rate than the standing
champion over the opponents both played, by more than twice the error of the
difference, and a win against that champion itself whose Wilson lower bound is
above a half over at least `DECISIVE_GAMES` decided games. Either alone let the
chain move sideways — `champion_3` replaced a champion 0.041 better than it on
the field, and `champion_8` one 0.023 better — so both are required.
There were two — a cheap ranking to shortlist on and a sealed block to confirm
— and the cheap one selected the luckiest program rather than the best: 78 of
471 topped it and none survived the block. Re-measuring cannot fix that, so
there is now one gate deep enough to select on. `--sessions` is how many
sessions to run; whatever is in flight when the last one is taken is drained.

### What a round is told about the ladder

The pool is built from published kernels, and the agents at the top of the
leaderboard publish none — so the strongest play in the competition appears
nowhere in the pool and only in the public replay archive. Two commands put it
where a round can query it:

```bash
uv run extract-corpus     # every recorded game into the ClickHouse games database
uv run losses             # the games this lineage lost, from the ladder's own records
```

`extract-corpus` parses every recorded game once — one row per game, per
game-day, per market order and per command — so a question about the corpus
costs a query rather than a twenty-minute walk of the archives, and fits a
Bradley-Terry rating over the ladder so a query can ask about the strong
rather than about the lucky. The loop records every game it scores into the
same database, and the round prompt carries a live query over the games we
lost.

Nothing measured off other agents' games reaches a round. A `build-order`
command used to write a table of what the top-rated agents hold on each day,
and the round prompt carried it whole; both were removed on 2026-09-10 after
measurement, because supplying another strategy's schedule took the median
candidate from 0.275 to 0.026 and stopped promotions for ten hours.

`scripts/daily_corpus.sh` runs the extraction nightly from its own worktree.

`dry-run` swaps the codex call for one that copies the parent with a visible
edit, so it exercises validation, evaluation, insertion and the promotion gate
without spending quota. Both commands resume from `run/campaign/state.json`,
`run/campaign/champion.json` and the archive log, so a killed loop restarts
where it stopped, and it resumes the same wandb run, named for
the git revision; a dry run logs nothing. Each promotion uploads
the champion's tarball as a wandb artifact named after it. `--workers` is what
one evaluation may fan over, and every session in flight can be evaluating at
once, so keep `--workers * SESSIONS` inside `CORE_BUDGET`.

The commit hook runs the fast suite; run `uv run pytest -m slow` before
pushing, which is where the tests that play real games against the vendored
corpus live.

## The Rust engine

`rust/` is a Rust port of the reference interpreter, bit-for-bit: the same
rules in the same order, CPython's random stream for weeds and shop unlocks,
and observations laid out like the reference's. It steps about 2,000 times
faster than `env.step`. `rust/python/` wraps it as the `kaggriculture_engine`
Python module, and `uv run replay-corpus` replays recorded Kaggle episodes
through it and compares every recorded field. `tests/rust/` proves both
against the installed reference. See [rust/README.md](rust/README.md).

```bash
uv sync --group rust
uv run maturin develop --release --uv -m rust/python/Cargo.toml
uv run replay-corpus --episodes 50 -v     # needs the archives under /data/kaggriculture/episodes
```

## Quick Start

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install dependencies

```bash
uv sync
```

The project pins Python 3.12 because that is what Kaggle's image runs
(3.12.13). Developing on the runner's own version closes the gap in which a
candidate could use something the local interpreter accepts and the runner
does not.

### 3. Set up environment variables

Copy the example environment file and add your Kaggle API token:

```bash
cp .env.example .env
# Edit .env, or save the token to ~/.kaggle/access_token
```

### 4. Install pre-commit hooks

```bash
uv run pre-commit install
```

## Commands

```bash
uv sync                        # install dependencies
uv run campaign --help         # the campaign CLI (play / check / package / loop)
uv run pytest                  # run the test suite
uv run pre-commit run -a       # format, lint, type-check, test
uv run package                 # writes submission.tar.gz for the campaign floor
uv run submit "melon loop v1"  # packages, confirms, then uploads
```

Submitting spends one of the day's five slots, so `submit` asks before
uploading; pass `--yes` to skip the prompt.

## Development

### Running Tests

```bash
uv run pytest
```

### Type Checking

```bash
uv run ty check src/
```

### Linting and Formatting

```bash
uv run ruff check src/
uv run ruff format src/
```

### Pre-commit Hooks

Pre-commit hooks will automatically run on every commit to ensure code quality. To run manually:

```bash
uv run pre-commit run --all-files
```

## Dependencies

Core dependencies:

- **kaggle-environments**: The Kaggriculture simulator and its rules tables
- **kaggle**: Competition CLI used for submission
- **Pydantic**: Data validation, used by `replay.py`
- **python-dotenv**: Environment variable management
- **tqdm**: Progress bars over iterables

Development tools:

- **ruff**: Fast Python linter and formatter
- **ty**: Static type checker
- **pytest**: Testing framework
- **pre-commit**: Git hooks for code quality

## Build System

This project uses `uv_build` as the build backend. To build the project:

```bash
uv build
```

## License

See [LICENSE](LICENSE) file for details.

## Contributing

1. Create a new branch for your feature
2. Make your changes
3. Ensure all tests pass and pre-commit hooks succeed
4. Submit a pull request
