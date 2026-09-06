# kaggriculture-engine

A Rust port of the Kaggriculture reference interpreter,
[`kaggle_environments/envs/kaggriculture/kaggriculture.py`](https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/kaggriculture/kaggriculture.py)
(pinned to `kaggle-environments` 1.32.7, the version this repository installs).

The port is bit-for-bit faithful, not approximately so:

- the same rules resolved in the same order (unit actions, then the per-unit
  lockstep market, town consumption, plant decay, end of day);
- the same CPython `random.Random` stream for weed spawns and shop unlocks,
  via a port of MT19937 with CPython's integer seeding;
- the same half-to-even rounding on prices;
- observation JSON laid out key for key like the reference, including dict
  insertion order for per-unit inventories, which decides what survives an
  end-of-day drop into a nearly full shed.

`tests/rust/test_differential.py` at the repository root drives both engines
with one action tape and compares every observable field of both seats at
every step of full 719-step episodes, under the default configuration and a
variant that overrides most of `kaggriculture.json`.

`uv run replay-corpus` does the same against recorded Kaggle episodes: it
replays each archive's recorded actions from the recorded seed and compares
every recorded observation of every step (see **Corpus verification**).

## Layout

| file              | reference counterpart                                             |
| ----------------- | ----------------------------------------------------------------- |
| `src/tables.rs`   | `CROPS`, `ANIMALS`, `MARKET_PARAMS`, `SHOPS`, land prices, `_fib` |
| `src/pricing.rs`  | `_shape`, `market_price`                                          |
| `src/pyrandom.rs` | `random.Random`: `random()`, `choice()`, `getrandbits()`          |
| `src/config.rs`   | the `configuration` block of `kaggriculture.json`                 |
| `src/state.rs`    | farms, tiles, private sheds, market, town, and their JSON         |
| `src/action.rs`   | typed actions and the reference's lenient action parsing          |
| `src/engine.rs`   | `interpreter` and everything it calls                             |
| `src/agents.rs`   | `pass_agent`, `random_agent`, `starter_agent`                     |
| `src/render.rs`   | `renderer`                                                        |
| `src/main.rs`     | CLI: `run` a tape, `bench`, `render`                              |
| `python/`         | PyO3 bindings: the `kaggriculture_engine` Python module           |

## Python

```sh
uv sync --group rust
uv run maturin develop --release --uv -m rust/python/Cargo.toml
```

```python
import kaggriculture_engine as ke

engine = ke.Engine({"seed": 42})          # any kaggriculture.json key
while not engine.done:
    obs = engine.observation(0)           # the dict a Kaggle agent receives
    engine.step([my_agent(obs), None])    # reference action dicts; None passes
print(engine.rewards)

snapshot = engine.state()                 # both sheds included
later = ke.Engine.from_state(snapshot, seed=engine.seed)
branch = engine.copy()                    # cheap, for search
```

`Engine` also exposes `observations()`, `money`, `day`, `hour`, `step_count`,
`render()` and `reset()`; the module carries `market_price`, the three
reference agents, a CPython-exact `Random`, and the name tables. Stubs ship
with the wheel. `tests/rust/test_bindings.py` checks the bindings against the
reference the same way the CLI is checked.

## Corpus verification

Kaggle's daily replay archives record the seed, both seats' actions and the
world after every turn, so a replay through this engine must reproduce every
recorded field. With archives under `/data/kaggriculture/episodes` (see
`scripts/fetch_episodes.py`):

```sh
uv run replay-corpus --episodes 50 -v          # fixed-seed sample across archives
uv run replay-corpus --archive path/to/one.zip # every episode in one archive
uv run replay-corpus --version 1.32.7 --stop   # only this engine version; stop at first divergence
uv run pytest tests/rust -m slow --rust-episodes=50   # the same as a test
```

The tool exits non-zero on any divergence and names the first turn, seat and
field. `tests/rust/test_corpus.py` checks the tool itself on locally recorded
episodes in the same archive layout, including that a corrupted record is
caught at the right step.

## Use

```rust
use kaggriculture_engine::{Action, Config, Engine, Order, UnitAction, Crop};

let mut engine = Engine::new(Config::with_seed(42));
let buy = Action {
    farmer: UnitAction::Noop,
    hands: vec![],
    market: vec![Order::BuySeed { crop: Crop::Carrot, n: 1 }],
};
engine.step(&[buy, Action::pass()]);
while !engine.done() {
    engine.step(&[Action::pass(), Action::pass()]);
}
println!("{:?}", engine.rewards());
```

`Engine::observation(player)` returns the reference's observation dict as
`serde_json::Value`; `Action::from_json` accepts the reference's
`{"farmer": [...], "hands": [...], "market": [...]}` with the same silent
no-ops for anything malformed.

## CLI

```sh
cargo build --release
# Replay a tape and print one JSON state per line (reset state first).
echo '{"configuration": {"seed": 1}, "actions": [[{"farmer": ["PASS"]}, {}]]}' \
  | target/release/kaggriculture-engine run
# Throughput with the built-in starter and random agents.
target/release/kaggriculture-engine bench --episodes 50
# The text renderer after 100 steps.
target/release/kaggriculture-engine render --seed 3 --steps 100
```

## Tests

```sh
cargo test                       # unit tests and CPython/reference goldens
uv run pytest tests/rust         # differential campaign against the reference
uv run pytest tests/rust --rust-episodes=10 --rust-turns=719
```

`tests/data/cpython_golden.json` is generated by `tools/make_golden.py` from
the installed reference; regenerate it there rather than editing it.

## Lint

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
```

The workspace lint set (`[workspace.lints]` in `Cargo.toml`) forbids
`unsafe`, denies every default clippy lint and the pedantic group, and allows
only the cast and float-comparison lints that the port trips by mirroring
Python's arithmetic, each with its reason. The same two commands run as
pre-commit hooks and in CI, across every crate here.

## Deliberate differences

- A malformed count in a unit action (`["PICKUP", "WHEAT", "x"]`) raises
  inside the reference interpreter and aborts the episode; here it is a
  no-op. Market orders already treat it as invalid in the reference, and
  the port matches that.
- `random_agent` takes an explicit generator; the reference's is unseeded.
- The reference can be handed a non-integer `I0` or `base` override and
  will carry floats through the market; the port keeps inventories and
  prices integral.
