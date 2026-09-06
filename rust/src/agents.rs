//! The three built-in agents from the reference module.
//!
//! `random_agent` in the reference draws from an unseeded `random.Random()`,
//! so no port can match its draws; this one takes an explicit generator.

use crate::action::{Action, Direction, Order, UnitAction};
use crate::pyrandom::PyRandom;
use crate::state::{GameState, Tile};
use crate::tables::{Crop, CROPS};

/// `pass_agent`.
pub fn pass_agent(_state: &GameState, _player: usize) -> Action {
    Action::pass()
}

const FARMER_OPS: [UnitAction; 7] = [
    UnitAction::Move(Direction::North),
    UnitAction::Move(Direction::South),
    UnitAction::Move(Direction::East),
    UnitAction::Move(Direction::West),
    UnitAction::Water,
    UnitAction::Harvest,
    UnitAction::Noop,
];

/// `random_agent`, drawing from `rng`.
pub fn random_agent(state: &GameState, player: usize, rng: &mut PyRandom) -> Action {
    let Some(farm) = state.farms.get(player) else {
        return Action::pass();
    };
    let private = &state.privates[player];
    let mut market = Vec::new();

    let affordable: Vec<Crop> = CROPS
        .iter()
        .copied()
        .filter(|crop| crop.data().seed as f64 <= farm.money)
        .collect();
    if !affordable.is_empty() && rng.random() < 0.1 {
        let crop = affordable[rng.choice_index(affordable.len())];
        market.push(Order::BuySeed { crop, n: 1 });
    }

    let available: Vec<Crop> = CROPS
        .iter()
        .copied()
        .filter(|crop| private.seeds_get(*crop) > 0)
        .collect();
    let farmer = if !available.is_empty() && rng.random() < 0.3 {
        UnitAction::Plant(available[rng.choice_index(available.len())])
    } else {
        FARMER_OPS[rng.choice_index(FARMER_OPS.len())]
    };

    let hands = farm
        .hands
        .iter()
        .map(|_| FARMER_OPS[rng.choice_index(FARMER_OPS.len())])
        .collect();
    Action {
        farmer,
        hands,
        market,
    }
}

/// `starter_agent`: buy a carrot seed, plant on the current tile, water, and
/// harvest at `max_yield_day`.
pub fn starter_agent(state: &GameState, player: usize) -> Action {
    let Some(farm) = state.farms.get(player) else {
        return Action::pass();
    };
    let private = &state.privates[player];
    let (fx, fy) = farm.farmer;
    let tile = farm.tile(fx, fy);
    let day = state.day;

    let mut market = Vec::new();
    let carrots = private.shed_get(crate::tables::Item::Carrot);
    if carrots > 0 {
        market.push(Order::Sell {
            item: crate::tables::Item::Carrot,
            n: carrots,
        });
    }
    if private.seeds_get(Crop::Carrot) == 0 && farm.money >= Crop::Carrot.data().seed as f64 {
        market.push(Order::BuySeed {
            crop: Crop::Carrot,
            n: 1,
        });
    }

    let mut farmer = UnitAction::Noop;
    match tile {
        Tile::Empty if private.seeds_get(Crop::Carrot) > 0 => {
            farmer = UnitAction::Plant(Crop::Carrot);
        }
        Tile::Plant(plant) if plant.crop == Crop::Carrot => {
            let age = day - plant.planted_day;
            if age >= Crop::Carrot.data().max_yield_day {
                farmer = UnitAction::Harvest;
            } else if !plant.watered_today {
                farmer = UnitAction::Water;
            }
        }
        _ => {}
    }
    Action {
        farmer,
        hands: Vec::new(),
        market,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::engine::Engine;

    #[test]
    fn starter_agent_makes_money() {
        let mut engine = Engine::new(Config::with_seed(1));
        while !engine.done() {
            let actions = [
                starter_agent(engine.state(), 0),
                pass_agent(engine.state(), 1),
            ];
            engine.step(&actions);
        }
        assert!(engine.rewards()[0] > 3000.0);
        assert_eq!(engine.rewards()[1], 3000.0);
    }

    #[test]
    fn random_agent_is_deterministic_for_a_seed() {
        let mut a = PyRandom::new(9);
        let mut b = PyRandom::new(9);
        let engine = Engine::new(Config::with_seed(1));
        for _ in 0..50 {
            assert_eq!(
                random_agent(engine.state(), 0, &mut a),
                random_agent(engine.state(), 0, &mut b)
            );
        }
    }
}
