//! A Rust port of the Kaggriculture reference interpreter
//! (`kaggle_environments/envs/kaggriculture/kaggriculture.py`).
//!
//! The port is meant to be bit-for-bit faithful: the same rules, the same
//! order of resolution, the same CPython random stream for weeds and shop
//! unlocks, and observation JSON laid out exactly like the reference's. The
//! Python test in `tests/rust/` drives both engines with one action tape and
//! compares every observable field each step.
//!
//! ```
//! use kaggriculture_engine::{Action, Config, Engine};
//!
//! let mut engine = Engine::new(Config::with_seed(42));
//! while !engine.done() {
//!     engine.step(&[Action::pass(), Action::pass()]);
//! }
//! assert_eq!(engine.rewards(), &[3000.0, 3000.0]);
//! ```

pub mod action;
pub mod agents;
pub mod config;
pub mod engine;
pub mod pricing;
pub mod pyrandom;
pub mod render;
pub mod state;
pub mod tables;

pub use action::{Action, Direction, Order, UnitAction};
pub use config::Config;
pub use engine::Engine;
pub use pricing::{market_price, market_price_with, shape};
pub use pyrandom::PyRandom;
pub use render::render;
pub use state::{Farm, GameState, Inventory, Livestock, Market, Plant, Private, Tile, Town};
pub use tables::{Animal, Crop, Item, Quadrant, Shop, Structure};
