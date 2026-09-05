# kaggriculture-engine (Python bindings)

PyO3 bindings for the Rust port of the Kaggriculture reference interpreter in
the parent directory. See `../README.md` for what the engine is and how its
fidelity is proven.

```python
import kaggriculture_engine as ke

engine = ke.Engine({"seed": 42})           # any kaggriculture.json key
while not engine.done:
    obs = engine.observation(0)            # the dict a Kaggle agent receives
    engine.step([ke.starter_agent(engine, 0), None])   # None passes
print(engine.rewards)
```

`Engine.step` takes the same `{"farmer": ..., "hands": ..., "market": ...}`
dicts a Kaggle agent returns; `Engine.state()` / `Engine.from_state()` save
and resume a whole episode, both sheds included; `Engine.copy()` is cheap, for
search.

Build into the project environment from the repository root:

```sh
uv sync --group rust
uv run maturin develop --release --uv -m rust/python/Cargo.toml
```
