//! Episode configuration, mirroring `kaggriculture.json`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tables::{MarketParams, Shape, MARKET_PARAMS, PRODUCTS, PRODUCT_COUNT};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", default)]
pub struct Config {
    pub episode_steps: i64,
    pub board_size: i64,
    pub starting_money: i64,
    pub max_market_orders_per_turn: i64,
    pub turns_per_day: i64,
    pub shed_capacity: i64,
    pub weed_spawn_chance: f64,
    pub town_shop_unlock_interval: i64,
    pub town_shop_sell_interval: i64,
    pub town_center_sell_interval: i64,
    /// Episode seed. `None` draws a random 31-bit seed at reset, as the
    /// reference does via `resolve_episode_seed`.
    pub seed: Option<i64>,
    pub farm_hand_cost_mult: i64,
    /// Sparse per-product overrides of `MARKET_PARAMS`, in the JSON shape the
    /// reference accepts. Kept verbatim so the observation can echo the
    /// resolved table exactly as the reference does.
    pub market_params: Value,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            episode_steps: 720,
            board_size: 10,
            starting_money: 3000,
            max_market_orders_per_turn: 10,
            turns_per_day: 24,
            shed_capacity: 100,
            weed_spawn_chance: 0.005,
            town_shop_unlock_interval: 3,
            town_shop_sell_interval: 4,
            town_center_sell_interval: 24,
            seed: None,
            farm_hand_cost_mult: 1,
            market_params: Value::Object(Default::default()),
        }
    }
}

impl Config {
    pub fn with_seed(seed: i64) -> Config {
        Config {
            seed: Some(seed),
            ..Config::default()
        }
    }

    /// Parse a configuration object; missing keys take their defaults.
    pub fn from_json(value: &Value) -> Result<Config, serde_json::Error> {
        serde_json::from_value(value.clone())
    }

    pub fn has_market_overrides(&self) -> bool {
        matches!(&self.market_params, Value::Object(map) if !map.is_empty())
    }

    /// `_resolve_market_params`: the numeric curve per product after
    /// overrides, plus the JSON the reference stores under `market["params"]`
    /// when overrides were supplied.
    pub fn resolve_market_params(&self) -> ([MarketParams; PRODUCT_COUNT], Option<Value>) {
        let mut resolved = MARKET_PARAMS;
        if !self.has_market_overrides() {
            return (resolved, None);
        }
        let overrides = self.market_params.as_object().expect("checked above");
        let mut echoed = serde_json::Map::new();
        for (index, product) in PRODUCTS.iter().enumerate() {
            let mut entry = params_to_json(&resolved[index]);
            if let Some(Value::Object(patch)) = overrides.get(product.name()) {
                for (key, value) in patch {
                    entry.insert(key.clone(), value.clone());
                }
                resolved[index] = params_from_json(&entry, &resolved[index]);
            }
            echoed.insert(product.name().to_string(), Value::Object(entry));
        }
        (resolved, Some(Value::Object(echoed)))
    }
}

fn params_to_json(params: &MarketParams) -> serde_json::Map<String, Value> {
    let mut map = serde_json::Map::new();
    map.insert("base".into(), number(params.base));
    map.insert("I0".into(), number(params.i0));
    map.insert("T".into(), number(params.t));
    map.insert(
        "below_func".into(),
        Value::String(params.below_func.name().into()),
    );
    map.insert("below_target".into(), Value::from(params.below_target));
    map.insert(
        "above_func".into(),
        Value::String(params.above_func.name().into()),
    );
    map.insert("above_target".into(), Value::from(params.above_target));
    map
}

/// Integers stay integers in the echoed table, as they are in the reference.
fn number(value: f64) -> Value {
    if value.fract() == 0.0 && value.abs() < 9.0e15 {
        Value::from(value as i64)
    } else {
        Value::from(value)
    }
}

fn params_from_json(map: &serde_json::Map<String, Value>, fallback: &MarketParams) -> MarketParams {
    let float = |key: &str, default: f64| map.get(key).and_then(Value::as_f64).unwrap_or(default);
    let shape = |key: &str, default: Shape| {
        map.get(key)
            .and_then(Value::as_str)
            .map(Shape::from_name)
            .unwrap_or(default)
    };
    MarketParams {
        base: float("base", fallback.base),
        i0: float("I0", fallback.i0),
        t: float("T", fallback.t),
        below_func: shape("below_func", fallback.below_func),
        below_target: float("below_target", fallback.below_target),
        above_func: shape("above_func", fallback.above_func),
        above_target: float("above_target", fallback.above_target),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn defaults_match_the_specification() {
        let config = Config::from_json(&json!({})).unwrap();
        assert_eq!(config, Config::default());
        assert_eq!(config.episode_steps, 720);
        assert_eq!(config.turns_per_day, 24);
    }

    #[test]
    fn overrides_merge_sparsely() {
        let config = Config::from_json(&json!({
            "seed": 7,
            "marketParams": {"MELON": {"base": 300, "above_func": "linear"}, "BOGUS": {"base": 1}}
        }))
        .unwrap();
        let (resolved, echoed) = config.resolve_market_params();
        assert_eq!(resolved[4].base, 300.0);
        assert_eq!(resolved[4].above_func, Shape::Linear);
        assert_eq!(resolved[4].above_target, 3.60);
        assert_eq!(resolved[0], MARKET_PARAMS[0]);
        let echoed = echoed.unwrap();
        assert_eq!(echoed["MELON"]["base"], json!(300));
        assert_eq!(echoed["WHEAT"]["below_func"], json!("sqrt"));
        assert!(echoed.get("BOGUS").is_none());
    }

    #[test]
    fn no_overrides_means_no_echo() {
        let (resolved, echoed) = Config::default().resolve_market_params();
        assert_eq!(resolved, MARKET_PARAMS);
        assert!(echoed.is_none());
    }
}
