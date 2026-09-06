//! The market price curve: `_shape` and `market_price`.

use crate::tables::{Item, MarketParams, Shape, HINGE_GAIN, MARKET_PARAMS, PRICE_FLOOR};

/// `_shape(func, x, T)`.
pub fn shape(func: Shape, x: f64, t: f64) -> f64 {
    let x = x.max(0.0);
    match func {
        Shape::Linear => x,
        Shape::Sq => x * x,
        Shape::Sqrt => x.sqrt(),
        Shape::Log => (1.0 + x).ln(),
        Shape::Log10 => (1.0 + x).log10(),
        Shape::Hinge => {
            // Degenerates to linear if T is missing or non-positive.
            if t.is_nan() || t <= 0.0 {
                return x;
            }
            let u = x / t;
            u + HINGE_GAIN * (u - 1.0).max(0.0).powi(2)
        }
    }
}

/// Python's `round()` on a float: half to even, then `int()`.
fn py_round_to_int(value: f64) -> i64 {
    // `f64::round_ties_even` is banker's rounding, like CPython's round().
    let rounded = value.round_ties_even();
    if rounded.is_nan() {
        0
    } else {
        rounded as i64
    }
}

/// `market_price(item, inventory, params)`: floored at `PRICE_FLOOR`.
pub fn market_price_with(params: &MarketParams, inventory: i64) -> i64 {
    let inventory = inventory as f64;
    let price = if inventory < params.i0 {
        let f = params.below_func;
        let amp = params.below_target * params.base / shape(f, params.t, params.t);
        params.base + amp * shape(f, params.i0 - inventory, params.t)
    } else {
        let f = params.above_func;
        let amp = params.above_target * params.base / shape(f, params.t, params.t);
        params.base - amp * shape(f, inventory - params.i0, params.t)
    };
    PRICE_FLOOR.max(py_round_to_int(price))
}

/// `market_price` against the default `MARKET_PARAMS` table.
pub fn market_price(item: Item, inventory: i64) -> i64 {
    market_price_with(&MARKET_PARAMS[item as usize], inventory)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_price_at_equilibrium() {
        assert_eq!(market_price(Item::Wheat, 10_000), 25);
        assert_eq!(market_price(Item::Melon, 10_000), 250);
    }

    #[test]
    fn floor_holds_in_a_glut() {
        assert_eq!(market_price(Item::Melon, 1_000_000), 1);
    }

    #[test]
    fn rounding_is_half_to_even() {
        assert_eq!(py_round_to_int(0.5), 0);
        assert_eq!(py_round_to_int(1.5), 2);
        assert_eq!(py_round_to_int(2.5), 2);
        assert_eq!(py_round_to_int(-0.5), 0);
    }
}
