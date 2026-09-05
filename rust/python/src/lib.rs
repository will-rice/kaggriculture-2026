//! Python bindings for `kaggriculture_engine`.
//!
//! The Python surface speaks the reference engine's own dialect: actions are
//! the `{"farmer": [...], "hands": [...], "market": [...]}` dicts Kaggle
//! agents return, and observations are the dicts Kaggle agents receive.

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use serde_json::{Map, Value};

use engine::agents;
use engine::engine::day_rng;
use engine::tables::{ANIMALS, CROPS, ITEMS, PRODUCTS, SHOPS_SORTED};
use engine::{Action, Animal, Config, Crop, Engine, Item, PyRandom, Shop};

/// Convert a JSON value into the equivalent Python object.
fn to_py<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match value {
        Value::Null => py.None().into_bound(py),
        Value::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                PyInt::new(py, i).into_any()
            } else if let Some(u) = n.as_u64() {
                PyInt::new(py, u).into_any()
            } else {
                PyFloat::new(py, n.as_f64().unwrap_or(f64::NAN)).into_any()
            }
        }
        Value::String(s) => PyString::new(py, s).into_any(),
        Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(to_py(py, item)?)?;
            }
            list.into_any()
        }
        Value::Object(map) => {
            let dict = PyDict::new(py);
            for (key, item) in map {
                dict.set_item(key, to_py(py, item)?)?;
            }
            dict.into_any()
        }
    })
}

/// Convert a Python object into JSON, the way `json.dumps` would see it.
/// Objects with no JSON spelling become `null`, which the lenient action
/// parser then treats exactly as the reference treats a non-list.
fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(b) = obj.cast::<PyBool>() {
        return Ok(Value::Bool(b.is_true()));
    }
    if let Ok(i) = obj.cast::<PyInt>() {
        if let Ok(v) = i.extract::<i64>() {
            return Ok(Value::from(v));
        }
        if let Ok(v) = i.extract::<u64>() {
            return Ok(Value::from(v));
        }
        return Ok(Value::from(i.extract::<f64>()?));
    }
    if let Ok(f) = obj.cast::<PyFloat>() {
        return Ok(Value::from(f.value()));
    }
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(Value::String(s.to_str()?.to_owned()));
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        let mut map = Map::new();
        for (key, value) in dict.iter() {
            let key = match key.cast::<PyString>() {
                Ok(key) => key.to_str()?.to_owned(),
                Err(_) => key.str()?.to_str()?.to_owned(),
            };
            map.insert(key, from_py(&value)?);
        }
        return Ok(Value::Object(map));
    }
    if let Ok(list) = obj.cast::<PyList>() {
        return list
            .iter()
            .map(|item| from_py(&item))
            .collect::<PyResult<Vec<_>>>()
            .map(Value::Array);
    }
    if let Ok(tuple) = obj.cast::<PyTuple>() {
        return tuple
            .iter()
            .map(|item| from_py(&item))
            .collect::<PyResult<Vec<_>>>()
            .map(Value::Array);
    }
    // Anything dict-like (kaggle_environments' Struct is a dict subclass and
    // is caught above; other mappings land here).
    if let Ok(items) = obj.call_method0("items") {
        let mut map = Map::new();
        for pair in items.try_iter()? {
            let pair = pair?;
            let key: String = pair.get_item(0)?.str()?.to_str()?.to_owned();
            map.insert(key, from_py(&pair.get_item(1)?)?);
        }
        return Ok(Value::Object(map));
    }
    Ok(Value::Null)
}

fn config_from_py(configuration: Option<&Bound<'_, PyAny>>, seed: Option<i64>) -> PyResult<Config> {
    let mut config = match configuration {
        None => Config::default(),
        Some(obj) => {
            let value = from_py(obj)?;
            if !value.is_object() {
                return Err(PyValueError::new_err("configuration must be a mapping"));
            }
            Config::from_json(&value).map_err(|error| PyValueError::new_err(error.to_string()))?
        }
    };
    if seed.is_some() {
        config.seed = seed;
    }
    Ok(config)
}

/// A CPython-compatible `random.Random` stream.
#[pyclass(name = "Random", module = "kaggriculture_engine", skip_from_py_object)]
#[derive(Clone)]
struct PyRandomStream {
    inner: PyRandom,
}

#[pymethods]
impl PyRandomStream {
    /// `random.Random(seed)` for an integer seed.
    #[new]
    fn new(seed: i128) -> Self {
        PyRandomStream {
            inner: PyRandom::new(seed),
        }
    }

    /// The generator the engine uses for `day` of an episode with `seed`.
    #[staticmethod]
    fn for_day(seed: i64, day: i64) -> Self {
        PyRandomStream {
            inner: day_rng(seed, day),
        }
    }

    /// `random.random()`.
    fn random(&mut self) -> f64 {
        self.inner.random()
    }

    /// `random.getrandbits(k)` for `k <= 64`.
    fn getrandbits(&mut self, k: u32) -> PyResult<u64> {
        if k > 64 {
            return Err(PyValueError::new_err(
                "getrandbits supports at most 64 bits",
            ));
        }
        Ok(self.inner.getrandbits(k))
    }

    /// `random.randrange(n)`: an index below `n`.
    fn randbelow(&mut self, n: u64) -> PyResult<u64> {
        if n == 0 {
            return Err(PyValueError::new_err("randbelow needs n > 0"));
        }
        Ok(self.inner.randbelow(n))
    }

    /// `random.choice(seq)`.
    fn choice<'py>(&mut self, seq: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let len = seq.len()?;
        if len == 0 {
            return Err(PyIndexError::new_err(
                "cannot choose from an empty sequence",
            ));
        }
        seq.get_item(self.inner.choice_index(len))
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }
}

/// One Kaggriculture episode.
#[pyclass(name = "Engine", module = "kaggriculture_engine")]
struct PyEngine {
    inner: Engine,
}

#[pymethods]
impl PyEngine {
    /// Start an episode. `configuration` takes the keys of
    /// `kaggriculture.json`; `seed` overrides its `seed` entry. Without a
    /// seed one is drawn, as the reference does.
    #[new]
    #[pyo3(signature = (configuration=None, *, seed=None, players=2))]
    fn new(
        configuration: Option<&Bound<'_, PyAny>>,
        seed: Option<i64>,
        players: usize,
    ) -> PyResult<Self> {
        if players == 0 {
            return Err(PyValueError::new_err("players must be at least 1"));
        }
        let config = config_from_py(configuration, seed)?;
        Ok(PyEngine {
            inner: Engine::with_players(config, players),
        })
    }

    /// Resume from a state dict produced by `Engine.state`.
    #[staticmethod]
    #[pyo3(signature = (state, configuration=None, *, seed=None))]
    fn from_state(
        state: &Bound<'_, PyAny>,
        configuration: Option<&Bound<'_, PyAny>>,
        seed: Option<i64>,
    ) -> PyResult<Self> {
        let mut engine = PyEngine::new(configuration, seed, 2)?;
        engine.load_state(state)?;
        Ok(engine)
    }

    /// The resolved episode seed (`env.info["seed"]` in the reference).
    #[getter]
    fn seed(&self) -> i64 {
        self.inner.seed()
    }

    #[setter]
    fn set_seed(&mut self, seed: i64) {
        self.inner.set_seed(seed);
    }

    #[getter]
    fn players(&self) -> usize {
        self.inner.players()
    }

    /// The configuration in effect, with every default filled in.
    #[getter]
    fn configuration<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let value = serde_json::to_value(self.inner.config())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        to_py(py, &value)
    }

    #[getter]
    fn step_count(&self) -> i64 {
        self.inner.state().step
    }

    #[getter]
    fn day(&self) -> i64 {
        self.inner.state().day
    }

    #[getter]
    fn hour(&self) -> i64 {
        self.inner.state().hour
    }

    #[getter]
    fn done(&self) -> bool {
        self.inner.done()
    }

    /// Each player's bank once the episode is done; empty before that.
    #[getter]
    fn rewards(&self) -> Vec<f64> {
        self.inner.rewards().to_vec()
    }

    /// Each player's current bank.
    #[getter]
    fn money(&self) -> Vec<f64> {
        self.inner
            .state()
            .farms
            .iter()
            .map(|farm| farm.money)
            .collect()
    }

    /// Start over with the same configuration and seed.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Advance one turn. `actions` holds one entry per player in the
    /// reference's dict form; a missing or malformed entry passes.
    fn step(&mut self, actions: &Bound<'_, PyAny>) -> PyResult<()> {
        let parsed: Vec<Action> = if actions.is_none() {
            Vec::new()
        } else {
            actions
                .try_iter()?
                .map(|item| Ok(Action::from_json(&from_py(&item?)?)))
                .collect::<PyResult<Vec<_>>>()?
        };
        self.inner.step(&parsed);
        Ok(())
    }

    /// The observation dict `player` would receive from the reference.
    fn observation<'py>(&self, py: Python<'py>, player: usize) -> PyResult<Bound<'py, PyAny>> {
        if player >= self.inner.players() {
            return Err(PyIndexError::new_err(format!("no player {player}")));
        }
        to_py(py, &self.inner.observation(player))
    }

    /// Every player's observation, in seat order.
    fn observations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for player in 0..self.inner.players() {
            list.append(to_py(py, &self.inner.observation(player))?)?;
        }
        Ok(list)
    }

    /// The whole state, both privates included, as a dict.
    fn state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_py(py, &self.inner.state().to_json())
    }

    /// Replace the state with one from `Engine.state`.
    fn load_state(&mut self, state: &Bound<'_, PyAny>) -> PyResult<()> {
        let value = from_py(state)?;
        self.inner
            .load_state_json(&value)
            .map_err(PyValueError::new_err)
    }

    /// The reference's text renderer.
    fn render(&self) -> String {
        engine::render(self.inner.state())
    }

    /// An independent copy, for search.
    fn copy(&self) -> Self {
        PyEngine {
            inner: self.inner.clone(),
        }
    }

    fn __copy__(&self) -> Self {
        self.copy()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.copy()
    }

    fn __repr__(&self) -> String {
        let state = self.inner.state();
        format!(
            "Engine(seed={}, players={}, step={}, day={}, hour={}, done={})",
            self.inner.seed(),
            self.inner.players(),
            state.step,
            state.day,
            state.hour,
            state.done
        )
    }
}

/// `market_price(item, inventory)` against the default curve.
#[pyfunction]
fn market_price(item: &str, inventory: i64) -> PyResult<i64> {
    let item = Item::from_name(item)
        .filter(|item| item.is_product())
        .ok_or_else(|| PyValueError::new_err(format!("{item} is not a market product")))?;
    Ok(engine::market_price(item, inventory))
}

/// The reference `pass_agent`.
#[pyfunction]
fn pass_agent<'py>(
    py: Python<'py>,
    engine: &PyEngine,
    player: usize,
) -> PyResult<Bound<'py, PyAny>> {
    to_py(
        py,
        &agents::pass_agent(engine.inner.state(), player).to_json(),
    )
}

/// The reference `starter_agent`.
#[pyfunction]
fn starter_agent<'py>(
    py: Python<'py>,
    engine: &PyEngine,
    player: usize,
) -> PyResult<Bound<'py, PyAny>> {
    to_py(
        py,
        &agents::starter_agent(engine.inner.state(), player).to_json(),
    )
}

/// The reference `random_agent`, drawing from `rng`.
#[pyfunction]
fn random_agent<'py>(
    py: Python<'py>,
    engine: &PyEngine,
    player: usize,
    rng: &mut PyRandomStream,
) -> PyResult<Bound<'py, PyAny>> {
    to_py(
        py,
        &agents::random_agent(engine.inner.state(), player, &mut rng.inner).to_json(),
    )
}

#[pymodule]
fn kaggriculture_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add_class::<PyRandomStream>()?;
    m.add_function(wrap_pyfunction!(market_price, m)?)?;
    m.add_function(wrap_pyfunction!(pass_agent, m)?)?;
    m.add_function(wrap_pyfunction!(starter_agent, m)?)?;
    m.add_function(wrap_pyfunction!(random_agent, m)?)?;
    let names = |items: &[&str]| items.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
    m.add("PRODUCTS", names(&PRODUCTS.map(Item::name)))?;
    m.add("ITEMS", names(&ITEMS.map(Item::name)))?;
    m.add("CROPS", names(&CROPS.map(Crop::name)))?;
    m.add("ANIMALS", names(&ANIMALS.map(Animal::name)))?;
    m.add("SHOPS", names(&SHOPS_SORTED.map(Shop::name)))?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
