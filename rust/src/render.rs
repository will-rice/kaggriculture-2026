//! The text `renderer` from the reference module.

use std::fmt::Write;

use crate::state::{GameState, Tile};
use crate::tables::{CROPS, ITEMS, PRODUCTS};

/// `_render_tile`.
pub fn render_tile(tile: Tile) -> char {
    match tile {
        Tile::Empty => '.',
        Tile::Locked => '#',
        Tile::Weed => 'x',
        Tile::Plant(plant) => plant
            .crop
            .name()
            .chars()
            .next()
            .unwrap()
            .to_ascii_lowercase(),
        Tile::Animal(livestock) => livestock.animal.name().chars().next().unwrap(),
        Tile::Structure(structure) => structure.name().chars().next().unwrap(),
    }
}

/// `renderer(state, env)`: the same text, modulo Python's dict/list spelling
/// of the shed and seed tables.
pub fn render(state: &GameState) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "Step {}  Day {}  Hour {}",
        state.step, state.day, state.hour
    );
    let shops: Vec<String> = state
        .town
        .unlocked_shops
        .iter()
        .map(|shop| format!("'{}'", shop.name()))
        .collect();
    let _ = writeln!(out, "Town shops: [{}]", shops.join(", "));
    let prices: Vec<String> = PRODUCTS
        .iter()
        .map(|product| format!("{}=${}", product.name(), state.market.price_of(*product)))
        .collect();
    let _ = writeln!(out, "Prices: {}", prices.join(", "));
    for (player, farm) in state.farms.iter().enumerate() {
        let private = &state.privates[player];
        let shed: Vec<String> = ITEMS
            .iter()
            .map(|item| format!("'{}': {}", item.name(), private.shed_get(*item)))
            .collect();
        let seeds: Vec<String> = CROPS
            .iter()
            .map(|crop| format!("'{}': {}", crop.name(), private.seeds_get(*crop)))
            .collect();
        let quadrants: Vec<String> = farm
            .unlocked_quadrants
            .iter()
            .map(|q| format!("'{}'", q.name()))
            .collect();
        let _ = writeln!(
            out,
            "Player {}: ${:.0}  farmer=[{}, {}]  hands={}  unlocked=[{}]  shed={{{}}}  seeds={{{}}}",
            player,
            farm.money,
            farm.farmer.0,
            farm.farmer.1,
            farm.hands.len(),
            quadrants.join(", "),
            shed.join(", "),
            seeds.join(", "),
        );
        for y in 0..farm.board_size {
            let row: Vec<String> = (0..farm.board_size)
                .map(|x| render_tile(farm.tile(x, y)).to_string())
                .collect();
            let _ = writeln!(out, "  {}", row.join(" "));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::engine::Engine;

    #[test]
    fn renders_the_opening_board() {
        let engine = Engine::new(Config::with_seed(3));
        let text = render(engine.state());
        assert!(text.starts_with("Step 0  Day 0  Hour 0\n"));
        assert!(text.contains("Player 0: $3000  farmer=[4, 4]  hands=0  unlocked=['NW']"));
        assert!(text.contains("  . . . . . # # # # #\n"));
    }
}
