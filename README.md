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
[docs/superpowers/specs/2026-09-04-codex-campaign-design.md](docs/superpowers/specs/2026-09-04-codex-campaign-design.md)
for the design and
[docs/superpowers/plans/2026-09-04-campaign-foundation.md](docs/superpowers/plans/2026-09-04-campaign-foundation.md)
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
| `src/kaggriculture/campaign/validate.py`                              | `validate(agent) -> Verdict`: syntax, contract, imports, copy, load, latency                                                                                    |
| `src/kaggriculture/campaign/field_gate.py`, `kernel_watch.py`         | moved from `scripts/`, imports fixed; Plan 2 folds `field_gate` into the evaluator                                                                              |
| `src/kaggriculture/scripts/package.py`, `submit.py`                   | shipping, without the routes store                                                                                                                              |
| `src/kaggriculture/served/main.py`                                    | the cold-start seed the loop begins from; not what ships                                                                                                        |
| `src/kaggriculture/campaign/task_prompt.md`                           | phase 1's product; a first draft is written by hand in Task 12                                                                                                  |
| `tests/campaign/*.py`                                                 | one test module per source module                                                                                                                               |

### Running the campaign

```bash
uv run campaign dry-run --calls 2         # fake mutator, proves the pipeline
nohup uv run campaign loop > run/campaign/loop.log 2>&1 &
tail -f run/campaign/loop.log            # metrics: wandb.ai/will-rice/kaggriculture-2026
```

The loop is one `asyncio` event loop and nothing in it waits on anything it
does not need. A dispatcher keeps `--concurrency` codex sessions running at
once, and a session's slot is refilled the moment codex exits rather than
when its child finishes being scored; every child is validated,
fast-evaluated and inserted as soon as it exists. The schedule is counted in
completed calls, not rounds -- migration every `MIGRATION_INTERVAL` calls,
a deep-evaluation epoch every `EPOCH_INTERVAL`, an island reset every
`RESET_INTERVAL` -- and an epoch is a task like any other: it competes for
cores with the evaluations running beside it instead of stopping dispatch,
and a crossing that finds the previous epoch still running is skipped.
`--calls` is how many mutation calls to run; whatever is in flight when the
last one is planned is drained.

`dry-run` swaps the codex mutator for one that copies the parent with a
visible edit, so it exercises validation, evaluation, insertion and the
promotion gate without spending a call. Both commands resume from
`run/campaign/state.json` and the archive log, so a killed loop restarts
where it stopped, and it resumes the same wandb run (`config.WANDB_RUN_ID`),
so the curves continue, named `<codex model>-<git revision>` as of the launch;
a dry run logs nothing. Each promotion also uploads
the champion's file as a wandb artifact named after it. Keep `--workers * --concurrency` inside the core budget:
every session in flight can be evaluating at once, so the peak is their product.

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
