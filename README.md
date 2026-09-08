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

| path                                                                  | responsibility                                                                                                                                                  |
| --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/kaggriculture/campaign/__init__.py`                              | package marker                                                                                                                                                  |
| `src/kaggriculture/campaign/config.py`                                | every path and constant: `EXAM_SEEDS`, `ROOT`, `RUN`, `OPPONENTS`, `EPISODES`, `ENGINE_LIBRARY`, `CORE_BUDGET`, `ITEMS`, `SHOP_NAMES`, `UNIT_OPS`, `MARKET_OPS` |
| `src/kaggriculture/campaign/arena.py`                                 | reference-engine games between two agent files, both seats, over a process pool (from `search/arena.py`, stripped to file-path opponents)                       |
| `src/kaggriculture/campaign/engine/sim.hpp`, `pyrandom.hpp`, `NOTICE` | the adopted port, verbatim, with license                                                                                                                        |
| `src/kaggriculture/campaign/engine/bridge.cpp`                        | `extern "C"` surface: create, step, export packed state, free                                                                                                   |
| `src/kaggriculture/campaign/engine/build.py`                          | compiles `kaggriculture_engine.so` beside itself                                                                                                                |
| `src/kaggriculture/campaign/engine/wrapper.py`                        | `Engine` class: `reset`, `step`, `observation(player)`, `bank(player)`; `pack_action`, `render`                                                                 |
| `src/kaggriculture/campaign/tapes.py`                                 | iterate archived episodes: seed, actions per step, observations per step                                                                                        |
| `src/kaggriculture/campaign/roster.py`                                | opponent names → paths (never exposed), training pool and held-out set                                                                                          |
| `src/kaggriculture/campaign/harness.py`                               | `play`, `check`, `package`; the `campaign` CLI                                                                                                                  |
| `src/kaggriculture/campaign/copycheck.py`                             | token-shingle similarity against opponent sources                                                                                                               |
| `src/kaggriculture/campaign/validate.py`                              | `validate(agent) -> Verdict`: syntax, contract, imports, copy check, a full game; no latency check                                                              |
| `src/kaggriculture/campaign/archive.py`                               | the shared database: every program with its scores, its deep result and every failure, as an append-only log                                                    |
| `src/kaggriculture/campaign/prompt.py`                                | composes the message a call is given: the game, the program, its verdict, the day states, one instruction                                                       |
| `src/kaggriculture/campaign/mutate.py`                                | one `codex exec` call: a directory holding `child.py`, the message on stdin, a fallback model on a provider refusal                                             |
| `src/kaggriculture/campaign/evaluator.py`                             | `fast` on fresh seeds and `deep` on the sealed block; a program never plays itself                                                                              |
| `src/kaggriculture/campaign/pool.py`                                  | the opponents, counting equally; champions join, the crushed retire at `POOL_CAP`                                                                               |
| `src/kaggriculture/campaign/gate.py`                                  | `promotion`: beat every pool opponent. `promote`: the tarball, the champion file, the floor, `champion.json`                                                    |
| `src/kaggriculture/campaign/loop.py`                                  | the script: wandb, the event loop, `SESSIONS` workers, the gate they fire                                                                                       |
| `src/kaggriculture/campaign/field_gate.py`, `kernel_watch.py`         | the vendored-field rate, and the ladder scan that finds new held-out opponents                                                                                  |
| `src/kaggriculture/scripts/package.py`, `submit.py`                   | shipping, without the routes store                                                                                                                              |
| `src/kaggriculture/served/main.py`                                    | the cold-start seed the loop begins from; not what ships                                                                                                        |
| `src/kaggriculture/campaign/task_prompt.md`                           | the game as a call is told it: objective, rules, verified economics, the interface, the doctrine                                                                |
| `tests/campaign/*.py`                                                 | one test module per source module                                                                                                                               |

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
result back for another round. A round continues from its own previous
program; a new session starts again from the champion. A session ends when a
round beats every opponent, or at whichever of `ROUNDS_PER_SESSION` and
`SESSION_LIMIT_SECONDS` comes first.

The model is a mutation operator: it plays nothing and measures nothing, so
every game goes through the one pool that knows how many cores there are, and
the verdict it is sent is the same rule the promotion gate applies. There is
one measurement, over `GATE_SEEDS` fresh seeds, and a program that finishes top
of the Bradley-Terry standings on it becomes the champion and joins the pool.
There were two — a cheap ranking to shortlist on and a sealed block to confirm
— and the cheap one selected the luckiest program rather than the best: 78 of
471 topped it and none survived the block. Re-measuring cannot fix that, so
there is now one gate deep enough to select on. `--sessions` is how many
sessions to run; whatever is in flight when the last one is taken is drained.

### What a round is told about the ladder

The pool is built from published kernels, and the agents at the top of the
leaderboard publish none — so the strongest play in the competition appears
nowhere in the pool and only in the public replay archive. Two commands mine
it, and both write files the round prompt reads:

```bash
uv run extract-corpus     # /data/.../corpus.sqlite: every game as a table
uv run build-order        # campaign/build_order.md: what the strongest hold, by day
uv run strategies         # strategies.jsonl: what separates the strong from the rest
```

`extract-corpus` parses every recorded game once into SQLite — one row per
game, per game-day, per market order and per command — so a question about the
corpus costs a query rather than a twenty-minute walk of the archives. It also
fits a Bradley-Terry rating over all of them, and that rating is what the other
two are grouped by.

The grouping is the whole point. Read by _who won each game_, eleven quantities
over thirteen thousand games all came back between 45% and 60%: about half of a
ladder's winners are the weaker agent having a good day, and that noise swamps
everything. Read by _who is actually strong_, the same games separate at
90-100% — and reverse the sign of one of them.

`build-order` writes the table the round prompt carries whole: what the top
twenty-five rated agents hold on each day. `strategies` puts every quantity
crossed with every day to the corpus as a paired within-game comparison between
the stronger and the weaker agent, and keeps whatever settles; a round is shown
only the settled claims its own games put it on the wrong side of.

`scripts/daily_corpus.sh` runs the first and third nightly from its own
worktree. Run `build-order` when the ladder has moved. Everything reads every
game there is rather than a sample: measured over sixty games eight of the
first eleven claims cleared the bar, over four hundred six did, and over all
sixteen thousand one did. Selecting a claim for scoring highly on a sample is
the same winner's curse the promotion gate was rebuilt to avoid, and the log
keeps every measurement so a claim settled early and undone later keeps both.

`dry-run` swaps the codex call for one that copies the parent with a visible
edit, so it exercises validation, evaluation, insertion and the promotion gate
without spending quota. Both commands resume from `run/campaign/state.json`,
`run/campaign/champion.json` and the archive log, so a killed loop restarts
where it stopped, and it resumes the same wandb run, named
`<codex model>-<git revision>`; a dry run logs nothing. Each promotion uploads
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
