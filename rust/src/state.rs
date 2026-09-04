//! Game state: the public farms, the private sheds, the market and the town.
//!
//! `to_json` methods reproduce the exact dictionaries the reference engine
//! stores on each agent's observation, key for key, so a differential test
//! can compare the two engines field by field.

use serde_json::{Map, Value};

use crate::tables::{
    Animal, Crop, Item, MarketParams, Quadrant, Shop, Structure, CROPS, CROP_COUNT, ITEMS,
    ITEM_COUNT, PRODUCTS, PRODUCT_COUNT,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plant {
    pub crop: Crop,
    pub planted_day: i64,
    pub watered_today: bool,
    pub consecutive_unwatered: i64,
    pub yield_units: i64,
    pub max_lifespan_step: i64,
    pub fertilized_until_day: i64,
}

impl Plant {
    /// `_new_plant(crop, day, turns_per_day)`.
    pub fn new(crop: Crop, day: i64, turns_per_day: i64) -> Plant {
        let data = crop.data();
        Plant {
            crop,
            planted_day: day,
            watered_today: false,
            consecutive_unwatered: 1, // planting day counts as unwatered
            yield_units: if data.ongoing { 0 } else { 1 },
            max_lifespan_step: if data.ongoing {
                -1
            } else {
                (day + data.max_yield_day + 1) * turns_per_day
            },
            fertilized_until_day: -1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Livestock {
    pub animal: Animal,
    pub placed_day: i64,
    pub yield_units: i64,
    pub consecutive_unfed: i64,
    pub fed_today: bool,
    pub cared_today: bool,
    pub fertilizer_available: bool,
    pub pending_care_bonus: i64,
}

impl Livestock {
    /// `_new_animal(animal, day)`.
    pub fn new(animal: Animal, day: i64) -> Livestock {
        Livestock {
            animal,
            placed_day: day,
            yield_units: 0,
            consecutive_unfed: 0,
            fed_today: false,
            cared_today: false,
            fertilizer_available: false,
            pending_care_bonus: 0,
        }
    }

    pub fn structure(&self) -> Structure {
        self.animal.data().structure
    }
}

/// One farm tile. The reference stores `None`, the string `"LOCKED"`, or a
/// dict whose `kind` is `WEED`, `PLANT`, `COOP` or `PASTURE` (the latter two
/// with or without an `animal`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tile {
    Empty,
    Locked,
    Weed,
    Plant(Plant),
    Structure(Structure),
    Animal(Livestock),
}

impl Tile {
    pub fn to_json(&self) -> Value {
        match self {
            Tile::Empty => Value::Null,
            Tile::Locked => Value::String("LOCKED".into()),
            Tile::Weed => single("kind", "WEED"),
            Tile::Structure(structure) => single("kind", structure.name()),
            Tile::Plant(plant) => {
                let mut map = Map::new();
                map.insert("kind".into(), "PLANT".into());
                map.insert("crop".into(), plant.crop.name().into());
                map.insert("planted_day".into(), plant.planted_day.into());
                map.insert("watered_today".into(), plant.watered_today.into());
                map.insert(
                    "consecutive_unwatered".into(),
                    plant.consecutive_unwatered.into(),
                );
                map.insert("yield_units".into(), plant.yield_units.into());
                map.insert("max_lifespan_step".into(), plant.max_lifespan_step.into());
                map.insert(
                    "fertilized_until_day".into(),
                    plant.fertilized_until_day.into(),
                );
                Value::Object(map)
            }
            Tile::Animal(livestock) => {
                let mut map = Map::new();
                map.insert("kind".into(), livestock.structure().name().into());
                map.insert("animal".into(), livestock.animal.name().into());
                map.insert("placed_day".into(), livestock.placed_day.into());
                map.insert("yield_units".into(), livestock.yield_units.into());
                map.insert(
                    "consecutive_unfed".into(),
                    livestock.consecutive_unfed.into(),
                );
                map.insert("fed_today".into(), livestock.fed_today.into());
                map.insert("cared_today".into(), livestock.cared_today.into());
                map.insert(
                    "fertilizer_available".into(),
                    livestock.fertilizer_available.into(),
                );
                map.insert(
                    "pending_care_bonus".into(),
                    livestock.pending_care_bonus.into(),
                );
                Value::Object(map)
            }
        }
    }

    pub fn from_json(value: &Value) -> Result<Tile, String> {
        match value {
            Value::Null => Ok(Tile::Empty),
            Value::String(s) if s == "LOCKED" => Ok(Tile::Locked),
            Value::Object(map) => {
                let kind = map.get("kind").and_then(Value::as_str).unwrap_or("");
                if let Some(animal) = map.get("animal").and_then(Value::as_str) {
                    let animal = Animal::from_name(animal).ok_or("unknown animal")?;
                    return Ok(Tile::Animal(Livestock {
                        animal,
                        placed_day: int(map, "placed_day")?,
                        yield_units: int(map, "yield_units")?,
                        consecutive_unfed: int(map, "consecutive_unfed")?,
                        fed_today: boolean(map, "fed_today")?,
                        cared_today: boolean(map, "cared_today")?,
                        fertilizer_available: boolean(map, "fertilizer_available")?,
                        pending_care_bonus: map
                            .get("pending_care_bonus")
                            .and_then(Value::as_i64)
                            .unwrap_or(0),
                    }));
                }
                match kind {
                    "WEED" => Ok(Tile::Weed),
                    "COOP" => Ok(Tile::Structure(Structure::Coop)),
                    "PASTURE" => Ok(Tile::Structure(Structure::Pasture)),
                    "PLANT" => {
                        let crop = map.get("crop").and_then(Value::as_str).unwrap_or("");
                        let crop = Crop::from_name(crop).ok_or("unknown crop")?;
                        Ok(Tile::Plant(Plant {
                            crop,
                            planted_day: int(map, "planted_day")?,
                            watered_today: boolean(map, "watered_today")?,
                            consecutive_unwatered: int(map, "consecutive_unwatered")?,
                            yield_units: int(map, "yield_units")?,
                            max_lifespan_step: int(map, "max_lifespan_step")?,
                            fertilized_until_day: map
                                .get("fertilized_until_day")
                                .and_then(Value::as_i64)
                                .unwrap_or(-1),
                        }))
                    }
                    other => Err(format!("unknown tile kind {other:?}")),
                }
            }
            other => Err(format!("unrecognised tile {other}")),
        }
    }
}

fn single(key: &str, value: &str) -> Value {
    let mut map = Map::new();
    map.insert(key.into(), value.into());
    Value::Object(map)
}

fn int(map: &Map<String, Value>, key: &str) -> Result<i64, String> {
    map.get(key)
        .and_then(Value::as_i64)
        .ok_or_else(|| format!("missing integer {key}"))
}

fn boolean(map: &Map<String, Value>, key: &str) -> Result<bool, String> {
    map.get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("missing bool {key}"))
}

/// A unit's carried inventory: a Python dict in insertion order, which
/// decides which items make it into a nearly full shed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Inventory {
    entries: Vec<(Item, i64)>,
}

impl Inventory {
    pub fn new() -> Inventory {
        Inventory::default()
    }

    pub fn get(&self, item: Item) -> i64 {
        self.entries
            .iter()
            .find(|(key, _)| *key == item)
            .map_or(0, |(_, n)| *n)
    }

    /// `_inv_add`.
    pub fn add(&mut self, item: Item, n: i64) {
        match self.entries.iter_mut().find(|(key, _)| *key == item) {
            Some((_, count)) => *count += n,
            None => self.entries.push((item, n)),
        }
    }

    /// `_inv_take`: false, and untouched, if fewer than `n` are held.
    pub fn take(&mut self, item: Item, n: i64) -> bool {
        let Some(position) = self.entries.iter().position(|(key, _)| *key == item) else {
            return false;
        };
        if self.entries[position].1 < n {
            return false;
        }
        self.entries[position].1 -= n;
        if self.entries[position].1 == 0 {
            self.entries.remove(position);
        }
        true
    }

    /// `inv[item] -= n` followed by the zero-deletion, for a present item.
    pub fn remove(&mut self, item: Item, n: i64) {
        if let Some(position) = self.entries.iter().position(|(key, _)| *key == item) {
            self.entries[position].1 -= n;
            if self.entries[position].1 == 0 {
                self.entries.remove(position);
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[(Item, i64)] {
        &self.entries
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn to_json(&self) -> Value {
        let mut map = Map::new();
        for (item, n) in &self.entries {
            map.insert(item.name().into(), (*n).into());
        }
        Value::Object(map)
    }

    pub fn from_json(value: &Value) -> Result<Inventory, String> {
        let map = value.as_object().ok_or("inventory must be an object")?;
        let mut inventory = Inventory::new();
        for (key, n) in map {
            let item = Item::from_name(key).ok_or_else(|| format!("unknown item {key}"))?;
            inventory
                .entries
                .push((item, n.as_i64().ok_or("inventory count must be int")?));
        }
        Ok(inventory)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Farm {
    pub money: f64,
    pub board_size: i64,
    /// Row-major, `tiles[y * board_size + x]`.
    pub tiles: Vec<Tile>,
    pub farmer: (i64, i64),
    pub hands: Vec<(i64, i64)>,
    pub unlocked_quadrants: Vec<Quadrant>,
    pub hires_today: i64,
}

/// `_shed_access_tiles`: the four inner-corner tiles, NW, NE, SW, SE.
pub fn shed_access_tiles(board_size: i64) -> [(i64, i64); 4] {
    let half = board_size / 2;
    [
        (half - 1, half - 1),
        (half, half - 1),
        (half - 1, half),
        (half, half),
    ]
}

pub fn is_shed_adjacent(pos: (i64, i64), board_size: i64) -> bool {
    shed_access_tiles(board_size).contains(&pos)
}

/// `_default_spawn`: the first shed-access tile inside NW.
pub fn default_spawn(board_size: i64) -> (i64, i64) {
    shed_access_tiles(board_size)
        .into_iter()
        .find(|(x, y)| Quadrant::of(*x, *y, board_size) == Quadrant::Nw)
        .unwrap_or((0, 0))
}

impl Farm {
    /// `_new_farm(board_size, starting_money)`.
    pub fn new(board_size: i64, starting_money: i64) -> Farm {
        let mut tiles = Vec::with_capacity((board_size * board_size) as usize);
        for y in 0..board_size {
            for x in 0..board_size {
                tiles.push(if Quadrant::of(x, y, board_size) == Quadrant::Nw {
                    Tile::Empty
                } else {
                    Tile::Locked
                });
            }
        }
        Farm {
            money: starting_money as f64,
            board_size,
            tiles,
            farmer: default_spawn(board_size),
            hands: Vec::new(),
            unlocked_quadrants: vec![Quadrant::Nw],
            hires_today: 0,
        }
    }

    #[inline]
    pub fn tile(&self, x: i64, y: i64) -> Tile {
        self.tiles[(y * self.board_size + x) as usize]
    }

    #[inline]
    pub fn tile_mut(&mut self, x: i64, y: i64) -> &mut Tile {
        &mut self.tiles[(y * self.board_size + x) as usize]
    }

    /// `_farmer_position`: 0 is the farmer, 1.. the hands.
    pub fn unit_position(&self, unit: usize) -> Option<(i64, i64)> {
        if unit == 0 {
            Some(self.farmer)
        } else {
            self.hands.get(unit - 1).copied()
        }
    }

    pub fn set_unit_position(&mut self, unit: usize, pos: (i64, i64)) {
        if unit == 0 {
            self.farmer = pos;
        } else {
            self.hands[unit - 1] = pos;
        }
    }

    /// `_spawn_hand`: the least occupied shed-access tile, NWSE on ties.
    pub fn spawn_hand_position(&self) -> (i64, i64) {
        let access = shed_access_tiles(self.board_size);
        let mut occupants = [0usize; 4];
        for pos in std::iter::once(self.farmer).chain(self.hands.iter().copied()) {
            if let Some(index) = access.iter().position(|tile| *tile == pos) {
                occupants[index] += 1;
            }
        }
        let mut best = 0usize;
        for index in 1..4 {
            if occupants[index] < occupants[best] {
                best = index;
            }
        }
        access[best]
    }

    pub fn to_json(&self) -> Value {
        let mut map = Map::new();
        map.insert("money".into(), self.money.into());
        let rows: Vec<Value> = (0..self.board_size)
            .map(|y| {
                Value::Array(
                    (0..self.board_size)
                        .map(|x| self.tile(x, y).to_json())
                        .collect(),
                )
            })
            .collect();
        map.insert("tiles".into(), Value::Array(rows));
        map.insert("farmer".into(), pair(self.farmer));
        map.insert(
            "hands".into(),
            Value::Array(self.hands.iter().map(|p| pair(*p)).collect()),
        );
        map.insert(
            "unlocked_quadrants".into(),
            Value::Array(
                self.unlocked_quadrants
                    .iter()
                    .map(|q| q.name().into())
                    .collect(),
            ),
        );
        map.insert("hires_today".into(), self.hires_today.into());
        Value::Object(map)
    }

    pub fn from_json(value: &Value) -> Result<Farm, String> {
        let map = value.as_object().ok_or("farm must be an object")?;
        let rows = map
            .get("tiles")
            .and_then(Value::as_array)
            .ok_or("farm.tiles missing")?;
        let board_size = rows.len() as i64;
        let mut tiles = Vec::with_capacity(rows.len() * rows.len());
        for row in rows {
            let row = row.as_array().ok_or("tile row must be an array")?;
            if row.len() as i64 != board_size {
                return Err("farm tiles must be square".into());
            }
            for tile in row {
                tiles.push(Tile::from_json(tile)?);
            }
        }
        let money = map
            .get("money")
            .and_then(Value::as_f64)
            .ok_or("farm.money missing")?;
        let farmer = parse_pair(map.get("farmer").ok_or("farm.farmer missing")?)?;
        let hands = map
            .get("hands")
            .and_then(Value::as_array)
            .ok_or("farm.hands missing")?
            .iter()
            .map(parse_pair)
            .collect::<Result<Vec<_>, _>>()?;
        let unlocked_quadrants = map
            .get("unlocked_quadrants")
            .and_then(Value::as_array)
            .ok_or("farm.unlocked_quadrants missing")?
            .iter()
            .map(|q| {
                q.as_str()
                    .and_then(Quadrant::from_name)
                    .ok_or_else(|| format!("bad quadrant {q}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Farm {
            money,
            board_size,
            tiles,
            farmer,
            hands,
            unlocked_quadrants,
            hires_today: int(map, "hires_today")?,
        })
    }
}

fn pair(pos: (i64, i64)) -> Value {
    Value::Array(vec![pos.0.into(), pos.1.into()])
}

fn parse_pair(value: &Value) -> Result<(i64, i64), String> {
    let items = value.as_array().ok_or("position must be an array")?;
    if items.len() != 2 {
        return Err("position must have two coordinates".into());
    }
    let x = items[0].as_i64().ok_or("position must be integral")?;
    let y = items[1].as_i64().ok_or("position must be integral")?;
    Ok((x, y))
}

/// The per-player hidden state: shed, seeds and unit inventories.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Private {
    pub shed: [i64; ITEM_COUNT],
    pub seeds: [i64; CROP_COUNT],
    /// `inventories[0]` is the farmer; hands are appended on hire.
    pub inventories: Vec<Inventory>,
}

impl Default for Private {
    fn default() -> Private {
        Private::new()
    }
}

impl Private {
    /// `_new_private()`.
    pub fn new() -> Private {
        Private {
            shed: [0; ITEM_COUNT],
            seeds: [0; CROP_COUNT],
            inventories: vec![Inventory::new()],
        }
    }

    pub fn shed_total(&self) -> i64 {
        self.shed.iter().sum()
    }

    pub fn shed_get(&self, item: Item) -> i64 {
        self.shed[item as usize]
    }

    pub fn shed_add(&mut self, item: Item, n: i64) {
        self.shed[item as usize] += n;
    }

    pub fn seeds_get(&self, crop: Crop) -> i64 {
        self.seeds[crop as usize]
    }

    /// `_farmer_inventory`: grows the list if `unit` is past the end.
    pub fn inventory_mut(&mut self, unit: usize) -> &mut Inventory {
        while self.inventories.len() <= unit {
            self.inventories.push(Inventory::new());
        }
        &mut self.inventories[unit]
    }

    pub fn to_json(&self) -> Value {
        let mut shed = Map::new();
        for item in ITEMS {
            shed.insert(item.name().into(), self.shed[item as usize].into());
        }
        let mut seeds = Map::new();
        for crop in CROPS {
            seeds.insert(crop.name().into(), self.seeds[crop as usize].into());
        }
        let mut map = Map::new();
        map.insert("shed".into(), Value::Object(shed));
        map.insert("seeds".into(), Value::Object(seeds));
        map.insert(
            "inventories".into(),
            Value::Array(self.inventories.iter().map(Inventory::to_json).collect()),
        );
        Value::Object(map)
    }

    pub fn from_json(value: &Value) -> Result<Private, String> {
        let map = value.as_object().ok_or("private must be an object")?;
        let mut private = Private::new();
        let shed = map
            .get("shed")
            .and_then(Value::as_object)
            .ok_or("private.shed missing")?;
        for (key, n) in shed {
            let item = Item::from_name(key).ok_or_else(|| format!("unknown shed item {key}"))?;
            private.shed[item as usize] = n.as_i64().ok_or("shed count must be int")?;
        }
        let seeds = map
            .get("seeds")
            .and_then(Value::as_object)
            .ok_or("private.seeds missing")?;
        for (key, n) in seeds {
            let crop = Crop::from_name(key).ok_or_else(|| format!("unknown seed {key}"))?;
            private.seeds[crop as usize] = n.as_i64().ok_or("seed count must be int")?;
        }
        private.inventories = map
            .get("inventories")
            .and_then(Value::as_array)
            .ok_or("private.inventories missing")?
            .iter()
            .map(Inventory::from_json)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(private)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Market {
    pub inventory: [i64; PRODUCT_COUNT],
    pub prices: [i64; PRODUCT_COUNT],
    pub params: [MarketParams; PRODUCT_COUNT],
    /// Present only when the configuration supplied overrides; echoed as
    /// `market["params"]` exactly as the reference does.
    pub params_json: Option<Value>,
}

impl Market {
    /// `_new_market(params)`.
    pub fn new(params: [MarketParams; PRODUCT_COUNT], params_json: Option<Value>) -> Market {
        let mut inventory = [0i64; PRODUCT_COUNT];
        let mut prices = [0i64; PRODUCT_COUNT];
        for index in 0..PRODUCT_COUNT {
            inventory[index] = params[index].i0 as i64;
            prices[index] = params[index].base as i64;
        }
        Market {
            inventory,
            prices,
            params,
            params_json,
        }
    }

    #[inline]
    pub fn inventory_of(&self, item: Item) -> i64 {
        self.inventory[item as usize]
    }

    #[inline]
    pub fn price_of(&self, item: Item) -> i64 {
        self.prices[item as usize]
    }

    /// `market_price(item, inventory, params)` against this market's curve.
    pub fn quote(&self, item: Item, inventory: i64) -> i64 {
        crate::pricing::market_price_with(&self.params[item as usize], inventory)
    }

    /// `_refresh_prices`.
    pub fn refresh_prices(&mut self) {
        for index in 0..PRODUCT_COUNT {
            self.prices[index] =
                crate::pricing::market_price_with(&self.params[index], self.inventory[index]);
        }
    }

    pub fn to_json(&self) -> Value {
        let mut inventory = Map::new();
        let mut prices = Map::new();
        for product in PRODUCTS {
            inventory.insert(
                product.name().into(),
                self.inventory[product as usize].into(),
            );
            prices.insert(product.name().into(), self.prices[product as usize].into());
        }
        let mut map = Map::new();
        map.insert("inventory".into(), Value::Object(inventory));
        map.insert("prices".into(), Value::Object(prices));
        if let Some(params) = &self.params_json {
            map.insert("params".into(), params.clone());
        }
        Value::Object(map)
    }

    pub fn from_json(value: &Value, template: &Market) -> Result<Market, String> {
        let map = value.as_object().ok_or("market must be an object")?;
        let mut market = template.clone();
        let inventory = map
            .get("inventory")
            .and_then(Value::as_object)
            .ok_or("market.inventory missing")?;
        let prices = map
            .get("prices")
            .and_then(Value::as_object)
            .ok_or("market.prices missing")?;
        for product in PRODUCTS {
            market.inventory[product as usize] = inventory
                .get(product.name())
                .and_then(Value::as_i64)
                .ok_or_else(|| format!("market.inventory.{} missing", product.name()))?;
            market.prices[product as usize] = prices
                .get(product.name())
                .and_then(Value::as_i64)
                .ok_or_else(|| format!("market.prices.{} missing", product.name()))?;
        }
        Ok(market)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Town {
    /// Instances, with repeats: shops are drawn with replacement.
    pub unlocked_shops: Vec<Shop>,
}

impl Town {
    pub fn to_json(&self) -> Value {
        let mut map = Map::new();
        map.insert(
            "unlocked_shops".into(),
            Value::Array(
                self.unlocked_shops
                    .iter()
                    .map(|shop| shop.name().into())
                    .collect(),
            ),
        );
        Value::Object(map)
    }

    pub fn from_json(value: &Value) -> Result<Town, String> {
        let shops = value
            .get("unlocked_shops")
            .and_then(Value::as_array)
            .ok_or("town.unlocked_shops missing")?;
        let unlocked_shops = shops
            .iter()
            .map(|shop| {
                shop.as_str()
                    .and_then(Shop::from_name)
                    .ok_or_else(|| format!("unknown shop {shop}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Town { unlocked_shops })
    }
}

/// Everything the interpreter carries between turns.
#[derive(Clone, Debug, PartialEq)]
pub struct GameState {
    pub farms: Vec<Farm>,
    pub privates: Vec<Private>,
    pub market: Market,
    pub town: Town,
    pub step: i64,
    pub day: i64,
    pub hour: i64,
    pub done: bool,
    pub rewards: Vec<f64>,
}

impl GameState {
    pub fn players(&self) -> usize {
        self.farms.len()
    }

    /// The observation dict handed to `player`, as the reference lays it out.
    /// `step` is included, which the framework only writes for seat 0.
    pub fn observation(&self, player: usize) -> Value {
        let mut map = Map::new();
        map.insert("player".into(), (player as i64).into());
        map.insert(
            "farms".into(),
            Value::Array(self.farms.iter().map(Farm::to_json).collect()),
        );
        map.insert("private".into(), self.privates[player].to_json());
        map.insert("market".into(), self.market.to_json());
        map.insert("town".into(), self.town.to_json());
        map.insert("day".into(), self.day.into());
        map.insert("hour".into(), self.hour.into());
        map.insert("step".into(), self.step.into());
        Value::Object(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_farm_unlocks_only_nw() {
        let farm = Farm::new(10, 3000);
        assert_eq!(farm.tile(0, 0), Tile::Empty);
        assert_eq!(farm.tile(4, 4), Tile::Empty);
        assert_eq!(farm.tile(5, 4), Tile::Locked);
        assert_eq!(farm.tile(4, 5), Tile::Locked);
        assert_eq!(farm.tile(9, 9), Tile::Locked);
        assert_eq!(farm.farmer, (4, 4));
    }

    #[test]
    fn spawn_hand_prefers_least_occupied_then_nwse() {
        let mut farm = Farm::new(10, 3000);
        assert_eq!(farm.spawn_hand_position(), (5, 4));
        farm.hands.push((5, 4));
        assert_eq!(farm.spawn_hand_position(), (4, 5));
        farm.hands.push((4, 5));
        farm.hands.push((5, 5));
        assert_eq!(farm.spawn_hand_position(), (4, 4));
        farm.farmer = (0, 0);
        assert_eq!(farm.spawn_hand_position(), (4, 4));
    }

    #[test]
    fn inventory_keeps_insertion_order() {
        let mut inventory = Inventory::new();
        inventory.add(Item::Melon, 2);
        inventory.add(Item::Wheat, 1);
        inventory.add(Item::Melon, 1);
        assert_eq!(inventory.entries(), &[(Item::Melon, 3), (Item::Wheat, 1)]);
        assert!(!inventory.take(Item::Wheat, 2));
        assert!(inventory.take(Item::Wheat, 1));
        assert_eq!(inventory.entries(), &[(Item::Melon, 3)]);
        assert!(!inventory.take(Item::Egg, 1));
    }

    #[test]
    fn tiles_round_trip_through_json() {
        let tiles = [
            Tile::Empty,
            Tile::Locked,
            Tile::Weed,
            Tile::Structure(Structure::Coop),
            Tile::Structure(Structure::Pasture),
            Tile::Plant(Plant::new(Crop::Tomato, 3, 24)),
            Tile::Animal(Livestock::new(Animal::Sheep, 5)),
        ];
        for tile in tiles {
            assert_eq!(Tile::from_json(&tile.to_json()).unwrap(), tile);
        }
    }
}
