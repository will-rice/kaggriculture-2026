//! Typed turn actions, and the lenient JSON parsing the reference applies.
//!
//! The reference treats every malformed or illegal action as a silent no-op.
//! Parsing here folds the malformed cases into `UnitAction::Noop` and
//! `Order::Invalid`, which the engine then ignores in exactly the same places.
//! The one place a malformed order still matters is its position: orders are
//! matched by index across players and truncated by count, so an invalid
//! order keeps its slot.

use serde_json::Value;

use crate::tables::{Animal, Crop, Item};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    North,
    South,
    East,
    West,
}

impl Direction {
    /// `FARMER_MOVES`: (dx, dy), y grows downward.
    pub fn delta(self) -> (i64, i64) {
        match self {
            Direction::North => (0, -1),
            Direction::South => (0, 1),
            Direction::East => (1, 0),
            Direction::West => (-1, 0),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Direction::North => "NORTH",
            Direction::South => "SOUTH",
            Direction::East => "EAST",
            Direction::West => "WEST",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnitAction {
    /// Anything the reference silently ignores, including a bare `PASS`.
    Noop,
    Move(Direction),
    Drop,
    Pickup {
        item: Item,
        n: i64,
    },
    Plant(Crop),
    Water,
    Harvest,
    Fertilize,
    Dig,
    BuildCoop,
    BuildPasture,
    /// `PLACE <item> [n]`. `n` is `None` when the count could not be read as
    /// an integer; the reference would only fail on it in the shed path.
    Place {
        item: Item,
        n: Option<i64>,
    },
    Feed,
    CollectFertilizer,
    Care,
}

impl UnitAction {
    pub const PASS: UnitAction = UnitAction::Noop;

    /// Parse one `[op, ...args]` list the way `_apply_unit_action` reads it.
    pub fn from_json(value: &Value) -> UnitAction {
        let Some(items) = value.as_array() else {
            return UnitAction::Noop;
        };
        let Some(op) = items.first().and_then(Value::as_str) else {
            return UnitAction::Noop;
        };
        let item = || {
            items
                .get(1)
                .and_then(Value::as_str)
                .and_then(Item::from_name)
        };
        let count = || items.get(2).map(py_int);
        match op {
            "NORTH" => UnitAction::Move(Direction::North),
            "SOUTH" => UnitAction::Move(Direction::South),
            "EAST" => UnitAction::Move(Direction::East),
            "WEST" => UnitAction::Move(Direction::West),
            // `PASS` and every unknown op share the fallthrough arm.
            "DROP" => UnitAction::Drop,
            "PICKUP" => {
                if items.len() < 2 {
                    return UnitAction::Noop;
                }
                // An unknown item has 0 in the shed, so the pickup is a no-op.
                let Some(item) = item() else {
                    return UnitAction::Noop;
                };
                match count() {
                    None => UnitAction::Pickup { item, n: 1 },
                    Some(Some(n)) => UnitAction::Pickup { item, n },
                    Some(None) => UnitAction::Noop,
                }
            }
            "PLACE" => {
                if items.len() < 2 {
                    return UnitAction::Noop;
                }
                // Neither placement path can act on an item no unit can carry.
                let Some(item) = item() else {
                    return UnitAction::Noop;
                };
                let n = match count() {
                    None => Some(1),
                    Some(parsed) => parsed,
                };
                UnitAction::Place { item, n }
            }
            "PLANT" => match items
                .get(1)
                .and_then(Value::as_str)
                .and_then(Crop::from_name)
            {
                Some(crop) => UnitAction::Plant(crop),
                None => UnitAction::Noop,
            },
            "WATER" => UnitAction::Water,
            "HARVEST" => UnitAction::Harvest,
            "FERTILIZE" => UnitAction::Fertilize,
            "DIG" => UnitAction::Dig,
            "BUILD_COOP" => UnitAction::BuildCoop,
            "BUILD_PASTURE" => UnitAction::BuildPasture,
            "FEED" => UnitAction::Feed,
            "COLLECT_FERTILIZER" => UnitAction::CollectFertilizer,
            "CARE" => UnitAction::Care,
            _ => UnitAction::Noop,
        }
    }

    /// The `[op, ...args]` list form.
    pub fn to_json(&self) -> Value {
        match self {
            UnitAction::Noop => Value::Array(vec!["PASS".into()]),
            UnitAction::Move(direction) => Value::Array(vec![direction.name().into()]),
            UnitAction::Drop => Value::Array(vec!["DROP".into()]),
            UnitAction::Pickup { item, n } => {
                Value::Array(vec!["PICKUP".into(), item.name().into(), (*n).into()])
            }
            UnitAction::Plant(crop) => Value::Array(vec!["PLANT".into(), crop.name().into()]),
            UnitAction::Water => Value::Array(vec!["WATER".into()]),
            UnitAction::Harvest => Value::Array(vec!["HARVEST".into()]),
            UnitAction::Fertilize => Value::Array(vec!["FERTILIZE".into()]),
            UnitAction::Dig => Value::Array(vec!["DIG".into()]),
            UnitAction::BuildCoop => Value::Array(vec!["BUILD_COOP".into()]),
            UnitAction::BuildPasture => Value::Array(vec!["BUILD_PASTURE".into()]),
            UnitAction::Place { item, n } => Value::Array(vec![
                "PLACE".into(),
                item.name().into(),
                n.unwrap_or(1).into(),
            ]),
            UnitAction::Feed => Value::Array(vec!["FEED".into()]),
            UnitAction::CollectFertilizer => Value::Array(vec!["COLLECT_FERTILIZER".into()]),
            UnitAction::Care => Value::Array(vec!["CARE".into()]),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    /// Unparseable, or a buy/sell of something the market does not trade.
    /// Occupies a slot but never executes.
    Invalid,
    Hire,
    BuyLand,
    Sell {
        item: Item,
        n: i64,
    },
    BuySeed {
        crop: Crop,
        n: i64,
    },
    BuyProduct {
        item: Item,
        n: i64,
    },
    BuyAnimal {
        animal: Animal,
        n: i64,
    },
}

impl Order {
    /// `_parse_order`, plus the per-item validity the lockstep loop applies.
    pub fn from_json(value: &Value) -> Order {
        let Some(items) = value.as_array() else {
            return Order::Invalid;
        };
        let Some(op) = items.first().and_then(Value::as_str) else {
            return Order::Invalid;
        };
        match op {
            "HIRE" => Order::Hire,
            "BUY_LAND" => Order::BuyLand,
            "BUY_SEED" | "BUY_PRODUCT" | "BUY_ANIMAL" | "SELL" => {
                if items.len() < 3 {
                    return Order::Invalid;
                }
                let Some(n) = py_int(&items[2]) else {
                    return Order::Invalid;
                };
                if n <= 0 {
                    return Order::Invalid;
                }
                let name = items[1].as_str().unwrap_or("");
                match op {
                    "SELL" => match Item::from_name(name).filter(|item| item.is_product()) {
                        Some(item) => Order::Sell { item, n },
                        None => Order::Invalid,
                    },
                    "BUY_SEED" => match Crop::from_name(name) {
                        Some(crop) => Order::BuySeed { crop, n },
                        None => Order::Invalid,
                    },
                    "BUY_PRODUCT" => match Item::from_name(name) {
                        Some(item @ (Item::Wheat | Item::Fertilizer)) => {
                            Order::BuyProduct { item, n }
                        }
                        _ => Order::Invalid,
                    },
                    _ => match Animal::from_name(name) {
                        Some(animal) => Order::BuyAnimal { animal, n },
                        None => Order::Invalid,
                    },
                }
            }
            _ => Order::Invalid,
        }
    }

    pub fn to_json(&self) -> Value {
        match self {
            Order::Invalid => Value::Array(vec!["INVALID".into()]),
            Order::Hire => Value::Array(vec!["HIRE".into()]),
            Order::BuyLand => Value::Array(vec!["BUY_LAND".into()]),
            Order::Sell { item, n } => {
                Value::Array(vec!["SELL".into(), item.name().into(), (*n).into()])
            }
            Order::BuySeed { crop, n } => {
                Value::Array(vec!["BUY_SEED".into(), crop.name().into(), (*n).into()])
            }
            Order::BuyProduct { item, n } => {
                Value::Array(vec!["BUY_PRODUCT".into(), item.name().into(), (*n).into()])
            }
            Order::BuyAnimal { animal, n } => {
                Value::Array(vec!["BUY_ANIMAL".into(), animal.name().into(), (*n).into()])
            }
        }
    }
}

/// One player's turn: `{"farmer": [...], "hands": [[...], ...], "market": [[...], ...]}`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Action {
    pub farmer: UnitAction,
    pub hands: Vec<UnitAction>,
    pub market: Vec<Order>,
}

impl Default for Action {
    fn default() -> Action {
        Action::pass()
    }
}

impl Action {
    /// `{"farmer": ["PASS"], "hands": [], "market": []}`.
    pub fn pass() -> Action {
        Action {
            farmer: UnitAction::Noop,
            hands: Vec::new(),
            market: Vec::new(),
        }
    }

    /// Parse one player's action the way `interpreter` and `_process_market`
    /// read it: a non-object is an empty action, a missing farmer passes, and
    /// non-list hands or market fields are empty.
    pub fn from_json(value: &Value) -> Action {
        let Some(map) = value.as_object() else {
            return Action::pass();
        };
        let farmer = map
            .get("farmer")
            .map_or(UnitAction::Noop, UnitAction::from_json);
        let hands = map
            .get("hands")
            .and_then(Value::as_array)
            .map(|hands| hands.iter().map(UnitAction::from_json).collect())
            .unwrap_or_default();
        let market = map
            .get("market")
            .and_then(Value::as_array)
            .map(|orders| orders.iter().map(Order::from_json).collect())
            .unwrap_or_default();
        Action {
            farmer,
            hands,
            market,
        }
    }

    pub fn to_json(&self) -> Value {
        let mut map = serde_json::Map::new();
        map.insert("farmer".into(), self.farmer.to_json());
        map.insert(
            "hands".into(),
            Value::Array(self.hands.iter().map(UnitAction::to_json).collect()),
        );
        map.insert(
            "market".into(),
            Value::Array(self.market.iter().map(Order::to_json).collect()),
        );
        Value::Object(map)
    }
}

/// Python's `int(x)` for the JSON values an action can carry. Integers and
/// bools convert, floats truncate toward zero, numeric strings parse, and
/// anything else is `None` (a `TypeError`/`ValueError` in the reference).
pub fn py_int(value: &Value) -> Option<i64> {
    match value {
        Value::Bool(b) => Some(i64::from(*b)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(i)
            } else if let Some(u) = n.as_u64() {
                i64::try_from(u).ok()
            } else {
                let f = n.as_f64()?;
                if f.is_finite() && f.abs() < 9.2e18 {
                    Some(f.trunc() as i64)
                } else {
                    None
                }
            }
        }
        Value::String(s) => s.trim().replace('_', "").parse::<i64>().ok(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn unit_actions_parse_leniently() {
        assert_eq!(
            UnitAction::from_json(&json!(["NORTH"])),
            UnitAction::Move(Direction::North)
        );
        assert_eq!(UnitAction::from_json(&json!(["PASS"])), UnitAction::Noop);
        assert_eq!(UnitAction::from_json(&json!([])), UnitAction::Noop);
        assert_eq!(UnitAction::from_json(&json!("WATER")), UnitAction::Noop);
        assert_eq!(UnitAction::from_json(&json!(["PLANT"])), UnitAction::Noop);
        assert_eq!(
            UnitAction::from_json(&json!(["PLANT", "MELON"])),
            UnitAction::Plant(Crop::Melon)
        );
        assert_eq!(
            UnitAction::from_json(&json!(["PICKUP", "WHEAT"])),
            UnitAction::Pickup {
                item: Item::Wheat,
                n: 1
            }
        );
        assert_eq!(
            UnitAction::from_json(&json!(["PICKUP", "WHEAT", 2.9])),
            UnitAction::Pickup {
                item: Item::Wheat,
                n: 2
            }
        );
        assert_eq!(
            UnitAction::from_json(&json!(["PLACE", "GOOSE", "x"])),
            UnitAction::Place {
                item: Item::Goose,
                n: None
            }
        );
        assert_eq!(
            UnitAction::from_json(&json!(["PLACE", "BOGUS"])),
            UnitAction::Noop
        );
    }

    #[test]
    fn orders_parse_like_the_reference() {
        assert_eq!(Order::from_json(&json!(["HIRE", "extra"])), Order::Hire);
        assert_eq!(Order::from_json(&json!(["SELL", "MELON"])), Order::Invalid);
        assert_eq!(
            Order::from_json(&json!(["SELL", "MELON", 0])),
            Order::Invalid
        );
        assert_eq!(
            Order::from_json(&json!(["SELL", "MELON", "3"])),
            Order::Sell {
                item: Item::Melon,
                n: 3
            }
        );
        assert_eq!(
            Order::from_json(&json!(["SELL", "GOOSE", 1])),
            Order::Invalid
        );
        assert_eq!(
            Order::from_json(&json!(["BUY_PRODUCT", "MELON", 1])),
            Order::Invalid
        );
        assert_eq!(
            Order::from_json(&json!(["BUY_PRODUCT", "FERTILIZER", true])),
            Order::BuyProduct {
                item: Item::Fertilizer,
                n: 1
            }
        );
        assert_eq!(
            Order::from_json(&json!(["BUY_SEED", "WHEAT", null])),
            Order::Invalid
        );
    }

    #[test]
    fn actions_default_sensibly() {
        assert_eq!(Action::from_json(&json!(null)), Action::pass());
        let parsed = Action::from_json(&json!({"farmer": "bad", "hands": "bad", "market": 3}));
        assert_eq!(parsed, Action::pass());
        let parsed = Action::from_json(&json!({"hands": [["WATER"], 7], "market": [["HIRE"]]}));
        assert_eq!(parsed.farmer, UnitAction::Noop);
        assert_eq!(parsed.hands, vec![UnitAction::Water, UnitAction::Noop]);
        assert_eq!(parsed.market, vec![Order::Hire]);
    }
}
