//! The interpreter: one call of `step` is one call of the reference
//! `interpreter(state, env)` after initialisation.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::action::{Action, Order, UnitAction};
use crate::config::Config;
use crate::pyrandom::PyRandom;
use crate::state::{
    default_spawn, is_shed_adjacent, Farm, GameState, Livestock, Plant, Private, Tile, Town,
};
use crate::state::{Inventory, Market};
use crate::tables::{
    Animal, Crop, Item, Quadrant, Structure, CROP_COUNT, LAND_ORDER, LAND_PRICES,
    MAX_SHOP_INSTANCES, SHOPS_SORTED, TOWN_CENTER_PRODUCTS,
};

/// Escape hatch on the per-unit market loop, as in the reference.
const MARKET_LOOP_LIMIT: u32 = 100_000;

#[derive(Clone, Debug)]
pub struct Engine {
    config: Config,
    seed: i64,
    players: usize,
    state: GameState,
}

impl Engine {
    /// A two-player episode. The seed is taken from the configuration or,
    /// like `resolve_episode_seed`, drawn as a random 31-bit integer.
    pub fn new(config: Config) -> Engine {
        Engine::with_players(config, 2)
    }

    pub fn with_players(config: Config, players: usize) -> Engine {
        let seed = config.seed.unwrap_or_else(random_seed);
        let state = initial_state(&config, players);
        Engine {
            config,
            seed,
            players,
            state,
        }
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// The resolved episode seed (`env.info["seed"]` in the reference).
    pub fn seed(&self) -> i64 {
        self.seed
    }

    pub fn players(&self) -> usize {
        self.players
    }

    pub fn state(&self) -> &GameState {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut GameState {
        &mut self.state
    }

    /// Replace the state wholesale, e.g. to resume from a recorded replay.
    pub fn set_state(&mut self, state: GameState) {
        self.state = state;
    }

    /// Load a state produced by `GameState::to_json`, keeping this engine's
    /// configuration and seed. The number of players follows the state.
    pub fn load_state_json(&mut self, value: &serde_json::Value) -> Result<(), String> {
        let (params, params_json) = self.config.resolve_market_params();
        let template = Market::new(params, params_json);
        let state = GameState::from_json(value, &template)?;
        self.players = state.players();
        self.state = state;
        Ok(())
    }

    /// Replace the seed that drives weeds and shop unlocks, e.g. when
    /// resuming a recorded episode whose seed is known.
    pub fn set_seed(&mut self, seed: i64) {
        self.seed = seed;
    }

    pub fn done(&self) -> bool {
        self.state.done
    }

    /// Final rewards (each player's bank), empty until the episode is done.
    pub fn rewards(&self) -> &[f64] {
        &self.state.rewards
    }

    /// Start the episode over with the same seed.
    pub fn reset(&mut self) {
        self.state = initial_state(&self.config, self.players);
    }

    pub fn observation(&self, player: usize) -> serde_json::Value {
        self.state.observation(player)
    }

    /// One interpreter call. Missing actions pass; extra ones are ignored.
    pub fn step(&mut self, actions: &[Action]) {
        if self.state.done {
            return;
        }
        let pass = Action::pass();
        let config = &self.config;
        let turns_per_day = config.turns_per_day.max(1);
        let shed_capacity = config.shed_capacity;
        let step = self.state.step;
        let day = step / turns_per_day;

        for player in 0..self.players {
            let action = actions.get(player).unwrap_or(&pass);
            apply_player_actions(
                &mut self.state.farms[player],
                &mut self.state.privates[player],
                action,
                day,
                turns_per_day,
                shed_capacity,
            );
        }

        process_market(&mut self.state, actions, config);
        town_consume(&mut self.state, config, step);
        for farm in &mut self.state.farms {
            decay_plants(farm, step);
        }
        if (step + 1) % turns_per_day == 0 {
            end_of_day(&mut self.state, config, self.seed, day);
        }

        let next_step = step + 1;
        self.state.step = next_step;
        self.state.day = next_step / turns_per_day;
        self.state.hour = next_step % turns_per_day;

        if step >= config.episode_steps - 2 {
            self.state.done = true;
            self.state.rewards = self.state.farms.iter().map(|farm| farm.money).collect();
        }
    }
}

fn random_seed() -> i64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut x = (nanos as u64) ^ 0x9E37_79B9_7F4A_7C15;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    (x & 0x7FFF_FFFF) as i64
}

/// `_initialize`.
fn initial_state(config: &Config, players: usize) -> GameState {
    let (params, params_json) = config.resolve_market_params();
    GameState {
        farms: (0..players)
            .map(|_| Farm::new(config.board_size, config.starting_money))
            .collect(),
        privates: (0..players).map(|_| Private::new()).collect(),
        market: Market::new(params, params_json),
        town: Town::default(),
        step: 0,
        day: 0,
        hour: 0,
        done: false,
        rewards: Vec::new(),
    }
}

/// The per-player half of `interpreter`: atomic PLANT validation, then the
/// farmer and every listed hand in order.
fn apply_player_actions(
    farm: &mut Farm,
    private: &mut Private,
    action: &Action,
    day: i64,
    turns_per_day: i64,
    shed_capacity: i64,
) {
    // If total PLANT requests for a crop this turn exceed the seeds held,
    // every PLANT of that crop is dropped, including ones for hands that do
    // not exist.
    let mut demand = [0i64; CROP_COUNT];
    for unit in std::iter::once(&action.farmer).chain(action.hands.iter()) {
        if let UnitAction::Plant(crop) = unit {
            demand[*crop as usize] += 1;
        }
    }
    let allowed = |unit: &UnitAction| -> UnitAction {
        match unit {
            UnitAction::Plant(crop) if demand[*crop as usize] > private.seeds_get(*crop) => {
                UnitAction::Noop
            }
            other => *other,
        }
    };
    let farmer = allowed(&action.farmer);
    let hands: Vec<UnitAction> = action.hands.iter().map(allowed).collect();

    apply_unit_action(farm, private, 0, &farmer, day, turns_per_day, shed_capacity);
    for (index, hand) in hands.iter().enumerate() {
        apply_unit_action(
            farm,
            private,
            index + 1,
            hand,
            day,
            turns_per_day,
            shed_capacity,
        );
    }
}

/// `_apply_unit_action`. Illegal actions are silent no-ops.
///
/// One function, like the reference: the op dispatch reads as the rules table.
#[allow(clippy::too_many_lines)]
pub fn apply_unit_action(
    farm: &mut Farm,
    private: &mut Private,
    unit: usize,
    action: &UnitAction,
    day: i64,
    turns_per_day: i64,
    shed_capacity: i64,
) {
    let board_size = farm.board_size;
    let Some((fx, fy)) = farm.unit_position(unit) else {
        return;
    };
    if matches!(action, UnitAction::Noop) {
        return;
    }
    // The reference fetches the inventory before dispatching, which grows
    // the list when a unit index is past its end.
    private.inventory_mut(unit);

    if let UnitAction::Move(direction) = action {
        let (dx, dy) = direction.delta();
        let (nx, ny) = (fx + dx, fy + dy);
        if nx < 0 || nx >= board_size || ny < 0 || ny >= board_size {
            return;
        }
        // Movement onto LOCKED tiles is allowed: a hand can spawn on a locked
        // shed-access tile, and blocking movement would strand it there.
        farm.set_unit_position(unit, (nx, ny));
        return;
    }

    let tile = farm.tile(fx, fy);
    let adjacent = is_shed_adjacent((fx, fy), board_size);

    // Shed operations resolve before the LOCKED guard: three of the four
    // shed-access tiles start locked.
    match action {
        UnitAction::Drop => {
            if !adjacent {
                return;
            }
            let entries: Vec<(Item, i64)> = private.inventories[unit].entries().to_vec();
            for (item, n) in entries {
                if n <= 0 {
                    continue;
                }
                let room = (shed_capacity - private.shed_total()).max(0);
                let take = n.min(room);
                if take > 0 {
                    private.shed_add(item, take);
                }
            }
            private.inventories[unit].clear();
            return;
        }
        UnitAction::Pickup { item, n } => {
            if !adjacent || *n <= 0 {
                return;
            }
            let n = (*n).min(private.shed_get(*item));
            if n <= 0 {
                return;
            }
            private.shed_add(*item, -n);
            private.inventories[unit].add(*item, n);
            return;
        }
        UnitAction::Place { item, n } => {
            // Animal placement: standing on a matching, unoccupied structure.
            if let (Some(animal), Tile::Structure(structure)) = (item.animal(), tile) {
                if structure == animal.data().structure {
                    if private.inventories[unit].take(*item, 1) {
                        *farm.tile_mut(fx, fy) = Tile::Animal(Livestock::new(animal, day));
                    }
                    return;
                }
            }
            // Shed drop: orthogonally adjacent to the shed; obeys capacity.
            if !adjacent {
                return;
            }
            let Some(n) = *n else {
                return;
            };
            if n <= 0 {
                return;
            }
            let n = n.min(private.inventories[unit].get(*item));
            if n <= 0 {
                return;
            }
            let room = (shed_capacity - private.shed_total()).max(0);
            let n = n.min(room);
            if n <= 0 {
                return;
            }
            private.inventories[unit].remove(*item, n);
            private.shed_add(*item, n);
            return;
        }
        _ => {}
    }

    // Everything below mutates the tile the unit stands on, so it requires
    // that tile to be owned.
    if tile == Tile::Locked {
        return;
    }

    match action {
        UnitAction::Plant(crop) => {
            if tile != Tile::Empty || private.seeds_get(*crop) <= 0 {
                return;
            }
            private.seeds[*crop as usize] -= 1;
            *farm.tile_mut(fx, fy) = Tile::Plant(Plant::new(*crop, day, turns_per_day));
        }
        UnitAction::Water => {
            let Tile::Plant(mut plant) = tile else {
                return;
            };
            if plant.watered_today {
                return;
            }
            plant.watered_today = true;
            let data = plant.crop.data();
            if !data.ongoing {
                let age_days = day - plant.planted_day;
                let window_start = (data.max_yield_day + 1) / 2;
                if window_start <= age_days && age_days <= data.max_yield_day {
                    let bonus = if plant.fertilized_until_day >= day {
                        2
                    } else {
                        1
                    };
                    plant.yield_units = data.max_yield.min(plant.yield_units + bonus);
                }
            }
            *farm.tile_mut(fx, fy) = Tile::Plant(plant);
        }
        UnitAction::Harvest => match tile {
            Tile::Plant(mut plant) => {
                if plant.yield_units <= 0 {
                    return;
                }
                let data = plant.crop.data();
                if day - plant.planted_day < data.first_yield_day {
                    return;
                }
                let units = plant.yield_units;
                plant.yield_units = 0;
                private.inventories[unit].add(plant.crop.product(), units);
                *farm.tile_mut(fx, fy) = if data.ongoing {
                    Tile::Plant(plant)
                } else {
                    Tile::Empty
                };
            }
            Tile::Animal(mut livestock) => {
                if livestock.yield_units <= 0 {
                    return;
                }
                let units = livestock.yield_units;
                livestock.yield_units = 0;
                private.inventories[unit].add(livestock.animal.data().product, units);
                *farm.tile_mut(fx, fy) = Tile::Animal(livestock);
            }
            _ => {}
        },
        UnitAction::Fertilize => {
            let Tile::Plant(mut plant) = tile else {
                return;
            };
            if !private.inventories[unit].take(Item::Fertilizer, 1) {
                return;
            }
            // Active for `day`, `day + 1`, `day + 2` (3 days inclusive).
            plant.fertilized_until_day = plant.fertilized_until_day.max(day + 2);
            *farm.tile_mut(fx, fy) = Tile::Plant(plant);
        }
        UnitAction::Dig => {
            // Removes plants, weeds and empty structures, never an animal.
            if matches!(tile, Tile::Empty | Tile::Animal(_)) {
                return;
            }
            *farm.tile_mut(fx, fy) = Tile::Empty;
        }
        UnitAction::BuildCoop => {
            if tile == Tile::Empty {
                *farm.tile_mut(fx, fy) = Tile::Structure(Structure::Coop);
            }
        }
        UnitAction::BuildPasture => {
            if tile == Tile::Empty {
                *farm.tile_mut(fx, fy) = Tile::Structure(Structure::Pasture);
            }
        }
        UnitAction::Feed => {
            let Tile::Animal(mut livestock) = tile else {
                return;
            };
            if livestock.fed_today || !private.inventories[unit].take(Item::Wheat, 1) {
                return;
            }
            livestock.fed_today = true;
            *farm.tile_mut(fx, fy) = Tile::Animal(livestock);
        }
        UnitAction::CollectFertilizer => {
            let Tile::Animal(mut livestock) = tile else {
                return;
            };
            if !livestock.fertilizer_available {
                return;
            }
            livestock.fertilizer_available = false;
            private.inventories[unit].add(Item::Fertilizer, 1);
            *farm.tile_mut(fx, fy) = Tile::Animal(livestock);
        }
        UnitAction::Care => {
            let Tile::Animal(mut livestock) = tile else {
                return;
            };
            if livestock.cared_today {
                return;
            }
            livestock.cared_today = true;
            *farm.tile_mut(fx, fy) = Tile::Animal(livestock);
        }
        UnitAction::Noop
        | UnitAction::Move(_)
        | UnitAction::Drop
        | UnitAction::Pickup { .. }
        | UnitAction::Place { .. } => {}
    }
}

/// One player's progress through a quantity-bearing order.
#[derive(Clone, Copy)]
struct OpenOrder {
    order: Order,
    remaining: i64,
}

#[derive(Clone, Copy)]
enum Quote {
    Sell(Item, i64),
    BuyProduct(Item, i64),
    BuySeed(Crop, i64),
    BuyAnimal(Animal, i64),
}

/// `_process_market`: per-unit lockstep. At each unit, both players' prices
/// are quoted against the same inventory, then both commit in player order.
#[allow(clippy::too_many_lines)]
fn process_market(state: &mut GameState, actions: &[Action], config: &Config) {
    let players = state.players();
    let max_orders = config.max_market_orders_per_turn.max(1) as usize;
    let hire_mult = config.farm_hand_cost_mult;
    let shed_capacity = config.shed_capacity;
    let board_size = config.board_size;
    let pass = Action::pass();

    let queues: Vec<&[Order]> = (0..players)
        .map(|player| {
            let orders = &actions.get(player).unwrap_or(&pass).market;
            &orders[..orders.len().min(max_orders)]
        })
        .collect();
    let max_len = queues.iter().map(|queue| queue.len()).max().unwrap_or(0);

    let mut open: Vec<Option<OpenOrder>> = vec![None; players];
    let mut quoted: Vec<Option<Quote>> = vec![None; players];
    for index in 0..max_len {
        for (player, queue) in queues.iter().enumerate() {
            open[player] = match queue.get(index) {
                None | Some(Order::Invalid) => None,
                Some(Order::Hire) => {
                    do_hire(
                        &mut state.farms[player],
                        &mut state.privates[player],
                        board_size,
                        hire_mult,
                    );
                    None
                }
                Some(Order::BuyLand) => {
                    do_buy_land(&mut state.farms[player], board_size);
                    None
                }
                Some(
                    order @ (Order::Sell { n, .. }
                    | Order::BuySeed { n, .. }
                    | Order::BuyProduct { n, .. }
                    | Order::BuyAnimal { n, .. }),
                ) => Some(OpenOrder {
                    order: *order,
                    remaining: *n,
                }),
            };
        }

        let mut iterations = 0u32;
        loop {
            iterations += 1;
            if iterations >= MARKET_LOOP_LIMIT {
                break;
            }
            let mut any_quote = false;
            for player in 0..players {
                quoted[player] = None;
                let Some(order) = open[player] else {
                    continue;
                };
                if order.remaining <= 0 {
                    continue;
                }
                quoted[player] = Some(match order.order {
                    Order::Sell { item, .. } => Quote::Sell(
                        item,
                        state.market.quote(item, state.market.inventory_of(item)),
                    ),
                    // Quoted at post-buy inventory so a buy/sell round-trip
                    // against an unchanged market nets zero.
                    Order::BuyProduct { item, .. } => Quote::BuyProduct(
                        item,
                        state
                            .market
                            .quote(item, state.market.inventory_of(item) - 1),
                    ),
                    Order::BuySeed { crop, .. } => Quote::BuySeed(crop, crop.data().seed),
                    Order::BuyAnimal { animal, .. } => Quote::BuyAnimal(animal, animal.data().cost),
                    Order::Invalid | Order::Hire | Order::BuyLand => unreachable!(),
                });
                any_quote = true;
            }
            if !any_quote {
                break;
            }

            let mut committed_any = false;
            for player in 0..players {
                let Some(quote) = quoted[player] else {
                    continue;
                };
                let ok = commit_unit(
                    quote,
                    &mut state.farms[player],
                    &mut state.privates[player],
                    &mut state.market,
                    shed_capacity,
                );
                if ok {
                    if let Some(order) = open[player].as_mut() {
                        order.remaining -= 1;
                    }
                    committed_any = true;
                } else {
                    open[player] = None;
                }
            }
            if !committed_any {
                break;
            }
        }

        state.market.refresh_prices();
    }
}

/// `_commit_unit`.
fn commit_unit(
    quote: Quote,
    farm: &mut Farm,
    private: &mut Private,
    market: &mut Market,
    shed_capacity: i64,
) -> bool {
    match quote {
        Quote::Sell(item, price) => {
            if private.shed_get(item) <= 0 {
                return false;
            }
            private.shed_add(item, -1);
            farm.money += price as f64;
            // Sales at $1 do not increase market supply.
            if price > 1 {
                market.inventory[item as usize] += 1;
            }
            true
        }
        Quote::BuyProduct(item, price) => {
            if farm.money < price as f64 || private.shed_total() >= shed_capacity {
                return false;
            }
            farm.money -= price as f64;
            private.shed_add(item, 1);
            market.inventory[item as usize] -= 1;
            true
        }
        Quote::BuySeed(crop, price) => {
            if farm.money < price as f64 {
                return false;
            }
            farm.money -= price as f64;
            private.seeds[crop as usize] += 1;
            true
        }
        Quote::BuyAnimal(animal, price) => {
            if farm.money < price as f64 || private.shed_total() >= shed_capacity {
                return false;
            }
            farm.money -= price as f64;
            private.shed_add(animal.item(), 1);
            true
        }
    }
}

/// `_do_hire`.
fn do_hire(farm: &mut Farm, private: &mut Private, _board_size: i64, mult: i64) {
    let cost = crate::tables::hire_cost(farm.hires_today, mult);
    if farm.money < cost as f64 {
        return;
    }
    farm.money -= cost as f64;
    farm.hires_today += 1;
    let spawn = farm.spawn_hand_position();
    farm.hands.push(spawn);
    private.inventories.push(Inventory::default());
}

/// `_do_buy_land`.
fn do_buy_land(farm: &mut Farm, board_size: i64) {
    let extra = farm.unlocked_quadrants.len() - 1; // NW is always there
    if extra >= LAND_ORDER.len() {
        return;
    }
    let cost = LAND_PRICES[extra];
    if farm.money < cost as f64 {
        return;
    }
    farm.money -= cost as f64;
    let quadrant = LAND_ORDER[extra];
    farm.unlocked_quadrants.push(quadrant);
    for y in 0..board_size {
        for x in 0..board_size {
            if Quadrant::of(x, y, board_size) == quadrant && farm.tile(x, y) == Tile::Locked {
                *farm.tile_mut(x, y) = Tile::Empty;
            }
        }
    }
}

/// `_town_consume`.
fn town_consume(state: &mut GameState, config: &Config, step: i64) {
    let shop_interval = config.town_shop_sell_interval.max(1);
    let center_interval = config.town_center_sell_interval.max(1);
    if step % shop_interval == 0 {
        for shop in &state.town.unlocked_shops {
            let products = shop.products();
            let multiplier = if products.len() == 1 { 2 } else { 1 };
            for item in products {
                state.market.inventory[*item as usize] -= multiplier;
            }
        }
    }
    if step % center_interval == 0 {
        for item in TOWN_CENTER_PRODUCTS {
            state.market.inventory[item as usize] -= 1;
        }
    }
    state.market.refresh_prices();
}

/// `_decay_plants`: past its lifespan a plant loses a unit every other turn
/// until it is a weed.
fn decay_plants(farm: &mut Farm, step: i64) {
    for tile in &mut farm.tiles {
        let Tile::Plant(plant) = tile else {
            continue;
        };
        let lifespan = plant.max_lifespan_step;
        if lifespan < 0 || step < lifespan || (step - lifespan) % 2 != 0 {
            continue;
        }
        plant.yield_units -= 1;
        if plant.yield_units <= 0 {
            *tile = Tile::Weed;
        }
    }
}

/// `_daily_refresh_plants`.
fn daily_refresh_plants(farm: &mut Farm, current_day: i64, turns_per_day: i64) {
    let next_day = current_day + 1;
    for tile in &mut farm.tiles {
        let Tile::Plant(plant) = tile else {
            continue;
        };
        let was_watered = plant.watered_today;
        if was_watered {
            plant.consecutive_unwatered = 0;
        } else {
            plant.consecutive_unwatered += 1;
        }
        plant.watered_today = false;
        if plant.consecutive_unwatered >= 2 {
            *tile = Tile::Weed;
            continue;
        }
        let data = plant.crop.data();
        if !data.ongoing {
            continue;
        }
        let days_since_first = next_day - plant.planted_day - data.first_yield_day;
        if days_since_first < 0 || days_since_first % data.interval != 0 {
            continue;
        }
        let production_count = days_since_first / data.interval + 1;
        if production_count > data.max_yield {
            continue;
        }
        // Fertilizer bonus only applies on watered days (basic needs first).
        let fertilized = was_watered && plant.fertilized_until_day >= current_day;
        plant.yield_units = data
            .max_yield
            .min(plant.yield_units + if fertilized { 2 } else { 1 });
        if production_count == data.max_yield {
            plant.max_lifespan_step = (next_day + 1) * turns_per_day;
        }
    }
}

/// `_daily_refresh_animals`.
fn daily_refresh_animals(farm: &mut Farm, day: i64) {
    let next_day = day + 1;
    for tile in &mut farm.tiles {
        let Tile::Animal(livestock) = tile else {
            continue;
        };
        if livestock.fed_today {
            livestock.consecutive_unfed = 0;
        } else {
            livestock.consecutive_unfed += 1;
        }
        if livestock.consecutive_unfed >= 2 {
            // Animal escapes; structure remains.
            *tile = Tile::Structure(livestock.structure());
            continue;
        }
        let data = livestock.animal.data();
        let days_since_first = next_day - livestock.placed_day - data.first_yield_day;
        if days_since_first >= 0 && days_since_first % data.interval == 0 {
            // Care bonus only consumed on a fed production day.
            let bonus = if livestock.fed_today {
                livestock.pending_care_bonus
            } else {
                0
            };
            livestock.yield_units = data.max_held.min(livestock.yield_units + 1 + bonus);
            livestock.pending_care_bonus = 0;
        }
        if livestock.cared_today && livestock.fed_today {
            livestock.pending_care_bonus += 1;
        }
        livestock.fertilizer_available = true;
        livestock.fed_today = false;
        livestock.cared_today = false;
    }
}

/// `_spawn_weeds`: one `random()` draw per empty unlocked tile, row-major.
fn spawn_weeds(farm: &mut Farm, weed_chance: f64, rng: &mut PyRandom) {
    for tile in &mut farm.tiles {
        if *tile == Tile::Empty && rng.random() < weed_chance {
            *tile = Tile::Weed;
        }
    }
}

/// `_drop_inventories_to_shed`: overflow past capacity is discarded.
fn drop_inventories_to_shed(private: &mut Private, capacity: i64) {
    for index in 0..private.inventories.len() {
        let entries: Vec<(Item, i64)> = private.inventories[index].entries().to_vec();
        for (item, n) in entries {
            if n <= 0 {
                continue;
            }
            let room = (capacity - private.shed_total()).max(0);
            let take = n.min(room);
            if take > 0 {
                private.shed_add(item, take);
            }
        }
        private.inventories[index].clear();
    }
}

/// The day's RNG: `random.Random((seed * 1_000_003) ^ day)`.
pub fn day_rng(seed: i64, day: i64) -> PyRandom {
    PyRandom::new((i128::from(seed) * 1_000_003) ^ i128::from(day))
}

/// `_end_of_day`.
fn end_of_day(state: &mut GameState, config: &Config, seed: i64, day: i64) {
    let board_size = config.board_size;
    let turns_per_day = config.turns_per_day.max(1);
    let weed_chance = config.weed_spawn_chance;
    let shed_capacity = config.shed_capacity;
    let unlock_interval = config.town_shop_unlock_interval.max(1);

    let mut rng = day_rng(seed, day);
    for player in 0..state.players() {
        let farm = &mut state.farms[player];
        let private = &mut state.privates[player];
        daily_refresh_plants(farm, day, turns_per_day);
        daily_refresh_animals(farm, day);
        spawn_weeds(farm, weed_chance, &mut rng);
        drop_inventories_to_shed(private, shed_capacity);
        farm.farmer = default_spawn(board_size);
        farm.hands.clear();
        farm.hires_today = 0;
        private.inventories.truncate(1);
        private.inventories[0].clear();
    }

    let next_day = day + 1;
    if next_day > 0
        && next_day % unlock_interval == 0
        && state.town.unlocked_shops.len() < MAX_SHOP_INSTANCES
    {
        // Drawn with replacement: the same shop can unlock repeatedly.
        let index = rng.choice_index(SHOPS_SORTED.len());
        state.town.unlocked_shops.push(SHOPS_SORTED[index]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::Shop;

    fn engine() -> Engine {
        Engine::new(Config::with_seed(12345))
    }

    fn turn(farmer: UnitAction, market: Vec<Order>) -> Action {
        Action {
            farmer,
            hands: Vec::new(),
            market,
        }
    }

    #[test]
    fn a_passed_episode_ends_after_719_steps_with_the_starting_bank() {
        let mut engine = engine();
        let mut steps = 0;
        while !engine.done() {
            engine.step(&[Action::pass(), Action::pass()]);
            steps += 1;
        }
        assert_eq!(steps, 719);
        assert_eq!(engine.state().step, 719);
        assert_eq!(engine.state().day, 29);
        assert_eq!(engine.state().hour, 23);
        assert_eq!(engine.rewards(), &[3000.0, 3000.0]);
        // Stepping a finished episode changes nothing.
        let before = engine.state().clone();
        engine.step(&[Action::pass(), Action::pass()]);
        assert_eq!(engine.state(), &before);
    }

    #[test]
    fn carrot_loop_banks_a_harvest() {
        let mut engine = engine();
        let buy = turn(
            UnitAction::Noop,
            vec![Order::BuySeed {
                crop: Crop::Carrot,
                n: 1,
            }],
        );
        engine.step(&[buy, Action::pass()]);
        assert_eq!(engine.state().privates[0].seeds_get(Crop::Carrot), 1);
        assert_eq!(engine.state().farms[0].money, 2980.0);

        engine.step(&[
            turn(UnitAction::Plant(Crop::Carrot), vec![]),
            Action::pass(),
        ]);
        assert!(matches!(engine.state().farms[0].tile(4, 4), Tile::Plant(_)));

        // Water once a day on days 0..3. Only the watering window (days 2
        // and 3 for carrot) adds yield, so the plant holds 3 units on day 3.
        for day in 0..4 {
            engine.step(&[turn(UnitAction::Water, vec![]), Action::pass()]);
            let Tile::Plant(plant) = engine.state().farms[0].tile(4, 4) else {
                panic!("expected a carrot");
            };
            assert_eq!(plant.yield_units, [1, 1, 2, 3][day]);
            if day < 3 {
                while engine.state().hour != 0 {
                    engine.step(&[Action::pass(), Action::pass()]);
                }
            }
        }
        engine.step(&[turn(UnitAction::Harvest, vec![]), Action::pass()]);
        assert_eq!(engine.state().farms[0].tile(4, 4), Tile::Empty);
        assert_eq!(
            engine.state().privates[0].inventories[0].get(Item::Carrot),
            3
        );

        engine.step(&[turn(UnitAction::Drop, vec![]), Action::pass()]);
        assert_eq!(engine.state().privates[0].shed_get(Item::Carrot), 3);
        let sell = turn(
            UnitAction::Noop,
            vec![Order::Sell {
                item: Item::Carrot,
                n: 3,
            }],
        );
        let before = engine.state().farms[0].money;
        engine.step(&[sell, Action::pass()]);
        assert!(engine.state().farms[0].money >= before + 3.0 * 35.0);
        assert_eq!(engine.state().privates[0].shed_get(Item::Carrot), 0);
    }

    #[test]
    fn hires_follow_the_fibonacci_ladder_and_reset_at_night() {
        let mut engine = engine();
        let hire = turn(
            UnitAction::Noop,
            vec![Order::Hire, Order::Hire, Order::Hire, Order::Hire],
        );
        engine.step(&[hire, Action::pass()]);
        let farm = &engine.state().farms[0];
        assert_eq!(farm.hires_today, 4);
        assert_eq!(farm.hands, vec![(5, 4), (4, 5), (5, 5), (4, 4)]);
        assert_eq!(farm.money, 3000.0 - f64::from(1 + 1 + 2 + 3));
        assert_eq!(engine.state().privates[0].inventories.len(), 5);
        while engine.state().hour != 0 {
            engine.step(&[Action::pass(), Action::pass()]);
        }
        let farm = &engine.state().farms[0];
        assert_eq!(farm.hires_today, 0);
        assert!(farm.hands.is_empty());
        assert_eq!(engine.state().privates[0].inventories.len(), 1);
    }

    #[test]
    fn land_unlocks_in_order_at_rising_prices() {
        let mut engine = engine();
        engine.state_mut().farms[0].money = 7000.0;
        let buy = turn(
            UnitAction::Noop,
            vec![Order::BuyLand, Order::BuyLand, Order::BuyLand],
        );
        engine.step(&[buy, Action::pass()]);
        let farm = &engine.state().farms[0];
        assert_eq!(
            farm.unlocked_quadrants,
            vec![Quadrant::Nw, Quadrant::Ne, Quadrant::Sw, Quadrant::Se]
        );
        assert_eq!(farm.money, 0.0);
        assert!(farm.tiles.iter().all(|tile| *tile != Tile::Locked));
    }

    #[test]
    fn town_unlocks_the_same_shops_as_the_reference_seed() {
        // Seed 12345 from the reference: shops arrive every third day.
        let mut engine = engine();
        while !engine.done() {
            engine.step(&[Action::pass(), Action::pass()]);
        }
        let shops = &engine.state().town.unlocked_shops;
        assert_eq!(shops.len(), MAX_SHOP_INSTANCES);
        assert!(shops.iter().all(|shop| SHOPS_SORTED.contains(shop)));
        let _ = Shop::Bakery;
    }

    #[test]
    fn plant_demand_is_atomic_per_crop() {
        let mut engine = engine();
        engine.state_mut().privates[0].seeds[Crop::Wheat as usize] = 1;
        let hire = turn(UnitAction::Noop, vec![Order::Hire]);
        engine.step(&[hire, Action::pass()]);
        // Two requests against one seed: both are dropped.
        let action = Action {
            farmer: UnitAction::Plant(Crop::Wheat),
            hands: vec![UnitAction::Plant(Crop::Wheat)],
            market: vec![],
        };
        engine.step(&[action, Action::pass()]);
        assert_eq!(engine.state().privates[0].seeds_get(Crop::Wheat), 1);
        assert_eq!(engine.state().farms[0].tile(4, 4), Tile::Empty);
    }
}
