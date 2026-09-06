//! Checks against values recorded from `CPython` and the reference module.
//!
//! `tests/data/cpython_golden.json` is written by `tools/make_golden.py`
//! against the installed `kaggle-environments`; regenerate it there rather
//! than editing by hand.

use serde_json::Value;

use kaggriculture_engine::pyrandom::PyRandom;
use kaggriculture_engine::tables::{fib, Item, Shape, SHOPS_SORTED};
use kaggriculture_engine::{market_price, shape};

fn golden() -> Value {
    let text = include_str!("data/cpython_golden.json");
    serde_json::from_str(text).expect("golden JSON")
}

#[test]
fn mt19937_words_match_cpython_for_every_recorded_seed() {
    let golden = golden();
    for (seed, record) in golden["rng"].as_object().unwrap() {
        let seed: i128 = seed.parse().unwrap();
        let mut rng = PyRandom::new(seed);
        let words: Vec<u64> = record["words"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| w.as_u64().unwrap())
            .collect();
        for (index, want) in words.iter().enumerate() {
            assert_eq!(u64::from(rng.next_u32()), *want, "seed {seed} word {index}");
        }
    }
}

#[test]
fn random_floats_match_cpython() {
    let golden = golden();
    for (seed, record) in golden["rng"].as_object().unwrap() {
        let seed: i128 = seed.parse().unwrap();
        let mut rng = PyRandom::new(seed);
        for (index, want) in record["random"].as_array().unwrap().iter().enumerate() {
            assert_eq!(
                rng.random(),
                want.as_f64().unwrap(),
                "seed {seed} random {index}"
            );
        }
    }
}

#[test]
fn choice_over_eight_shops_matches_cpython() {
    let golden = golden();
    for (seed, record) in golden["rng"].as_object().unwrap() {
        let seed: i128 = seed.parse().unwrap();
        let mut rng = PyRandom::new(seed);
        for (index, want) in record["choice8"].as_array().unwrap().iter().enumerate() {
            assert_eq!(
                rng.choice_index(8) as u64,
                want.as_u64().unwrap(),
                "seed {seed} choice {index}"
            );
        }
    }
}

#[test]
fn interleaved_calls_match_cpython() {
    let golden = golden();
    for (seed, record) in golden["rng"].as_object().unwrap() {
        let seed: i128 = seed.parse().unwrap();
        let mut rng = PyRandom::new(seed);
        let mixed = record["mixed"].as_array().unwrap();
        assert_eq!(rng.random(), mixed[0].as_f64().unwrap(), "seed {seed}");
        assert_eq!(
            rng.choice_index(8) as u64,
            mixed[1].as_u64().unwrap(),
            "seed {seed}"
        );
        assert_eq!(rng.random(), mixed[2].as_f64().unwrap(), "seed {seed}");
        assert_eq!(
            rng.choice_index(5) as u64,
            mixed[3].as_u64().unwrap(),
            "seed {seed}"
        );
        assert_eq!(
            rng.getrandbits(32),
            mixed[4].as_u64().unwrap(),
            "seed {seed}"
        );
        assert_eq!(rng.random(), mixed[5].as_f64().unwrap(), "seed {seed}");
    }
}

#[test]
fn market_prices_match_the_reference_curve() {
    let golden = golden();
    for (name, points) in golden["prices"].as_object().unwrap() {
        let item = Item::from_name(name).unwrap();
        for point in points.as_array().unwrap() {
            let inventory = point[0].as_i64().unwrap();
            let want = point[1].as_i64().unwrap();
            assert_eq!(market_price(item, inventory), want, "{name} at {inventory}");
        }
    }
    for (name, prices) in golden["dense"].as_object().unwrap() {
        let item = Item::from_name(name).unwrap();
        for (offset, want) in prices.as_array().unwrap().iter().enumerate() {
            let inventory = 9000 + offset as i64;
            assert_eq!(
                market_price(item, inventory),
                want.as_i64().unwrap(),
                "{name} at {inventory}"
            );
        }
    }
}

#[test]
fn shapes_match_the_reference() {
    let golden = golden();
    for (name, points) in golden["shape"].as_object().unwrap() {
        let func = Shape::from_name(name);
        for point in points.as_array().unwrap() {
            let x = point[0].as_f64().unwrap();
            let want = point[1].as_f64().unwrap();
            assert_eq!(shape(func, x, 200.0), want, "{name}({x})");
        }
    }
}

#[test]
fn fib_and_shop_order_match_the_reference() {
    let golden = golden();
    for (n, want) in golden["fib"].as_array().unwrap().iter().enumerate() {
        assert_eq!(fib(n as i64), want.as_i64().unwrap());
    }
    let shops: Vec<&str> = golden["sorted_shops"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap())
        .collect();
    let ours: Vec<&str> = SHOPS_SORTED.iter().map(|shop| shop.name()).collect();
    assert_eq!(ours, shops);
}
