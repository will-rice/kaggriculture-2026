//! Rules tables, transcribed from `kaggriculture.py`.
//!
//! Everything the reference engine keys by name is keyed by a small `Copy`
//! enum here; the string names only appear at the JSON boundary.

/// Crops, in the order of the reference `CROPS` table (also the seed order).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Crop {
    Wheat = 0,
    Carrot = 1,
    Tomato = 2,
    Strawberry = 3,
    Melon = 4,
}

pub const CROP_COUNT: usize = 5;

pub const CROPS: [Crop; CROP_COUNT] = [
    Crop::Wheat,
    Crop::Carrot,
    Crop::Tomato,
    Crop::Strawberry,
    Crop::Melon,
];

#[derive(Clone, Copy, Debug)]
pub struct CropData {
    pub seed: i64,
    pub first_yield_day: i64,
    pub max_yield_day: i64,
    pub interval: i64,
    pub max_yield: i64,
    pub ongoing: bool,
}

const CROP_DATA: [CropData; CROP_COUNT] = [
    CropData {
        seed: 10,
        first_yield_day: 2,
        max_yield_day: 4,
        interval: 0,
        max_yield: 6,
        ongoing: false,
    },
    CropData {
        seed: 20,
        first_yield_day: 2,
        max_yield_day: 3,
        interval: 0,
        max_yield: 4,
        ongoing: false,
    },
    CropData {
        seed: 50,
        first_yield_day: 8,
        max_yield_day: 8,
        interval: 1,
        max_yield: 4,
        ongoing: true,
    },
    CropData {
        seed: 100,
        first_yield_day: 10,
        max_yield_day: 10,
        interval: 2,
        max_yield: 4,
        ongoing: true,
    },
    CropData {
        seed: 80,
        first_yield_day: 10,
        max_yield_day: 12,
        interval: 0,
        max_yield: 6,
        ongoing: false,
    },
];

impl Crop {
    pub fn name(self) -> &'static str {
        match self {
            Crop::Wheat => "WHEAT",
            Crop::Carrot => "CARROT",
            Crop::Tomato => "TOMATO",
            Crop::Strawberry => "STRAWBERRY",
            Crop::Melon => "MELON",
        }
    }

    pub fn from_name(name: &str) -> Option<Crop> {
        CROPS.iter().copied().find(|crop| crop.name() == name)
    }

    pub fn data(self) -> &'static CropData {
        &CROP_DATA[self as usize]
    }

    /// The product a harvested plant of this crop yields.
    pub fn product(self) -> Item {
        match self {
            Crop::Wheat => Item::Wheat,
            Crop::Carrot => Item::Carrot,
            Crop::Tomato => Item::Tomato,
            Crop::Strawberry => Item::Strawberry,
            Crop::Melon => Item::Melon,
        }
    }
}

/// Animals, in the order of the reference `ANIMALS` table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Animal {
    Goose = 0,
    Cow = 1,
    Sheep = 2,
}

pub const ANIMAL_COUNT: usize = 3;

pub const ANIMALS: [Animal; ANIMAL_COUNT] = [Animal::Goose, Animal::Cow, Animal::Sheep];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Structure {
    Coop,
    Pasture,
}

impl Structure {
    pub fn name(self) -> &'static str {
        match self {
            Structure::Coop => "COOP",
            Structure::Pasture => "PASTURE",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AnimalData {
    pub cost: i64,
    pub structure: Structure,
    pub first_yield_day: i64,
    pub interval: i64,
    pub max_held: i64,
    pub product: Item,
}

const ANIMAL_DATA: [AnimalData; ANIMAL_COUNT] = [
    AnimalData {
        cost: 300,
        structure: Structure::Coop,
        first_yield_day: 4,
        interval: 1,
        max_held: 4,
        product: Item::Egg,
    },
    AnimalData {
        cost: 400,
        structure: Structure::Pasture,
        first_yield_day: 8,
        interval: 2,
        max_held: 6,
        product: Item::Milk,
    },
    AnimalData {
        cost: 500,
        structure: Structure::Pasture,
        first_yield_day: 6,
        interval: 3,
        max_held: 6,
        product: Item::Wool,
    },
];

impl Animal {
    pub fn name(self) -> &'static str {
        match self {
            Animal::Goose => "GOOSE",
            Animal::Cow => "COW",
            Animal::Sheep => "SHEEP",
        }
    }

    pub fn from_name(name: &str) -> Option<Animal> {
        ANIMALS.iter().copied().find(|animal| animal.name() == name)
    }

    pub fn data(self) -> &'static AnimalData {
        &ANIMAL_DATA[self as usize]
    }

    /// The shed slot this animal occupies while it is carried or stored.
    pub fn item(self) -> Item {
        match self {
            Animal::Goose => Item::Goose,
            Animal::Cow => Item::Cow,
            Animal::Sheep => Item::Sheep,
        }
    }
}

/// Every key of the private shed: the nine products followed by the three
/// animals, in reference order (`PRODUCTS + list(ANIMALS)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Item {
    Wheat = 0,
    Carrot = 1,
    Tomato = 2,
    Strawberry = 3,
    Melon = 4,
    Egg = 5,
    Milk = 6,
    Wool = 7,
    Fertilizer = 8,
    Goose = 9,
    Cow = 10,
    Sheep = 11,
}

pub const PRODUCT_COUNT: usize = 9;
pub const ITEM_COUNT: usize = 12;

pub const ITEMS: [Item; ITEM_COUNT] = [
    Item::Wheat,
    Item::Carrot,
    Item::Tomato,
    Item::Strawberry,
    Item::Melon,
    Item::Egg,
    Item::Milk,
    Item::Wool,
    Item::Fertilizer,
    Item::Goose,
    Item::Cow,
    Item::Sheep,
];

/// The reference `PRODUCTS` list: everything the market trades.
pub const PRODUCTS: [Item; PRODUCT_COUNT] = [
    Item::Wheat,
    Item::Carrot,
    Item::Tomato,
    Item::Strawberry,
    Item::Melon,
    Item::Egg,
    Item::Milk,
    Item::Wool,
    Item::Fertilizer,
];

/// `TOWN_CENTER_PRODUCTS`: every product except fertilizer.
pub const TOWN_CENTER_PRODUCTS: [Item; PRODUCT_COUNT - 1] = [
    Item::Wheat,
    Item::Carrot,
    Item::Tomato,
    Item::Strawberry,
    Item::Melon,
    Item::Egg,
    Item::Milk,
    Item::Wool,
];

impl Item {
    pub fn name(self) -> &'static str {
        match self {
            Item::Wheat => "WHEAT",
            Item::Carrot => "CARROT",
            Item::Tomato => "TOMATO",
            Item::Strawberry => "STRAWBERRY",
            Item::Melon => "MELON",
            Item::Egg => "EGG",
            Item::Milk => "MILK",
            Item::Wool => "WOOL",
            Item::Fertilizer => "FERTILIZER",
            Item::Goose => "GOOSE",
            Item::Cow => "COW",
            Item::Sheep => "SHEEP",
        }
    }

    pub fn from_name(name: &str) -> Option<Item> {
        ITEMS.iter().copied().find(|item| item.name() == name)
    }

    pub fn from_index(index: usize) -> Option<Item> {
        ITEMS.get(index).copied()
    }

    pub fn is_product(self) -> bool {
        (self as usize) < PRODUCT_COUNT
    }

    /// The animal this shed slot holds, if it is an animal slot.
    pub fn animal(self) -> Option<Animal> {
        match self {
            Item::Goose => Some(Animal::Goose),
            Item::Cow => Some(Animal::Cow),
            Item::Sheep => Some(Animal::Sheep),
            _ => None,
        }
    }
}

/// Price-curve shapes. Unknown names fall back to linear in the reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    Linear,
    Sq,
    Sqrt,
    Log,
    Log10,
    Hinge,
}

impl Shape {
    pub fn from_name(name: &str) -> Shape {
        // `linear` and every unknown name share the fallthrough arm.
        match name {
            "sq" => Shape::Sq,
            "sqrt" => Shape::Sqrt,
            "log" => Shape::Log,
            "log10" => Shape::Log10,
            "hinge" => Shape::Hinge,
            _ => Shape::Linear,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Shape::Linear => "linear",
            Shape::Sq => "sq",
            Shape::Sqrt => "sqrt",
            Shape::Log => "log",
            Shape::Log10 => "log10",
            Shape::Hinge => "hinge",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MarketParams {
    pub base: f64,
    pub i0: f64,
    pub t: f64,
    pub below_func: Shape,
    pub below_target: f64,
    pub above_func: Shape,
    pub above_target: f64,
}

pub const MARKET_I0: i64 = 10_000;
pub const PRICE_FLOOR: i64 = 1;
pub const HINGE_GAIN: f64 = 8.0;

const fn params(
    base: f64,
    t: f64,
    below_func: Shape,
    below_target: f64,
    above_func: Shape,
    above_target: f64,
) -> MarketParams {
    MarketParams {
        base,
        i0: MARKET_I0 as f64,
        t,
        below_func,
        below_target,
        above_func,
        above_target,
    }
}

/// `MARKET_PARAMS`, indexed by product.
pub const MARKET_PARAMS: [MarketParams; PRODUCT_COUNT] = [
    params(25.0, 400.0, Shape::Sqrt, 0.80, Shape::Log, 0.20),
    params(35.0, 450.0, Shape::Hinge, 1.00, Shape::Sqrt, 0.70),
    params(60.0, 200.0, Shape::Hinge, 0.40, Shape::Sqrt, 0.60),
    params(120.0, 100.0, Shape::Sqrt, 0.70, Shape::Linear, 1.60),
    params(250.0, 300.0, Shape::Log, 0.20, Shape::Sq, 3.60),
    params(50.0, 332.0, Shape::Hinge, 0.40, Shape::Log, 0.20),
    params(160.0, 122.0, Shape::Sqrt, 0.60, Shape::Linear, 1.60),
    params(200.0, 105.0, Shape::Log, 0.20, Shape::Sq, 3.20),
    params(100.0, 200.0, Shape::Linear, 0.40, Shape::Linear, 0.40),
];

/// Quadrants of the farm. NW is always unlocked; the rest unlock in
/// `LAND_ORDER`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Quadrant {
    Nw = 0,
    Ne = 1,
    Sw = 2,
    Se = 3,
}

impl Quadrant {
    pub fn name(self) -> &'static str {
        match self {
            Quadrant::Nw => "NW",
            Quadrant::Ne => "NE",
            Quadrant::Sw => "SW",
            Quadrant::Se => "SE",
        }
    }

    pub fn from_name(name: &str) -> Option<Quadrant> {
        [Quadrant::Nw, Quadrant::Ne, Quadrant::Sw, Quadrant::Se]
            .into_iter()
            .find(|quadrant| quadrant.name() == name)
    }

    pub fn of(x: i64, y: i64, board_size: i64) -> Quadrant {
        let half = board_size / 2;
        match (y < half, x < half) {
            (true, true) => Quadrant::Nw,
            (true, false) => Quadrant::Ne,
            (false, true) => Quadrant::Sw,
            (false, false) => Quadrant::Se,
        }
    }
}

pub const LAND_ORDER: [Quadrant; 3] = [Quadrant::Ne, Quadrant::Sw, Quadrant::Se];
pub const LAND_PRICES: [i64; 3] = [1000, 2000, 4000];

pub const FARM_HAND_COST_MULT: i64 = 1;

/// Town shops in `sorted(SHOPS)` order, which is the order the reference
/// draws from at unlock time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Shop {
    Bakery = 0,
    BrunchSpot = 1,
    FarmersMarket = 2,
    IceCreamShop = 3,
    PetCafe = 4,
    PizzaShop = 5,
    SmoothieShop = 6,
    YarnStore = 7,
}

pub const SHOP_COUNT: usize = 8;

pub const SHOPS_SORTED: [Shop; SHOP_COUNT] = [
    Shop::Bakery,
    Shop::BrunchSpot,
    Shop::FarmersMarket,
    Shop::IceCreamShop,
    Shop::PetCafe,
    Shop::PizzaShop,
    Shop::SmoothieShop,
    Shop::YarnStore,
];

impl Shop {
    pub fn name(self) -> &'static str {
        match self {
            Shop::Bakery => "BAKERY",
            Shop::BrunchSpot => "BRUNCH_SPOT",
            Shop::FarmersMarket => "FARMERS_MARKET",
            Shop::IceCreamShop => "ICE_CREAM_SHOP",
            Shop::PetCafe => "PET_CAFE",
            Shop::PizzaShop => "PIZZA_SHOP",
            Shop::SmoothieShop => "SMOOTHIE_SHOP",
            Shop::YarnStore => "YARN_STORE",
        }
    }

    pub fn from_name(name: &str) -> Option<Shop> {
        SHOPS_SORTED
            .iter()
            .copied()
            .find(|shop| shop.name() == name)
    }

    /// Products the shop consumes, in reference order.
    pub fn products(self) -> &'static [Item] {
        match self {
            Shop::Bakery => &[Item::Egg, Item::Wheat],
            Shop::PizzaShop => &[Item::Milk, Item::Tomato, Item::Wheat],
            Shop::BrunchSpot => &[Item::Egg, Item::Wheat, Item::Strawberry],
            Shop::YarnStore => &[Item::Wool],
            Shop::IceCreamShop => &[Item::Strawberry, Item::Milk, Item::Wheat],
            Shop::PetCafe => &[Item::Carrot],
            Shop::SmoothieShop => &[Item::Strawberry, Item::Milk],
            Shop::FarmersMarket => &[Item::Wheat, Item::Carrot, Item::Tomato, Item::Strawberry],
        }
    }
}

/// Maximum number of shop instances the town ever unlocks.
pub const MAX_SHOP_INSTANCES: usize = 8;

/// `_fib`, indexed so that fib(0) = fib(1) = 1, fib(2) = 2, fib(3) = 3, ...
pub fn fib(n: i64) -> i64 {
    let (mut a, mut b) = (1i64, 1i64);
    for _ in 0..n.max(0) {
        let next = a.saturating_add(b);
        a = b;
        b = next;
    }
    a
}

pub fn hire_cost(hires_today: i64, mult: i64) -> i64 {
    mult.saturating_mul(fib(hires_today))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fib_matches_reference_indexing() {
        let expected = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55];
        for (n, want) in expected.iter().enumerate() {
            assert_eq!(fib(n as i64), *want);
        }
    }

    #[test]
    fn names_round_trip() {
        for item in ITEMS {
            assert_eq!(Item::from_name(item.name()), Some(item));
        }
        for crop in CROPS {
            assert_eq!(Crop::from_name(crop.name()), Some(crop));
            assert_eq!(crop.product().name(), crop.name());
        }
        for animal in ANIMALS {
            assert_eq!(Animal::from_name(animal.name()), Some(animal));
            assert_eq!(animal.item().animal(), Some(animal));
        }
        for shop in SHOPS_SORTED {
            assert_eq!(Shop::from_name(shop.name()), Some(shop));
        }
    }

    #[test]
    fn shops_are_in_python_sorted_order() {
        let names: Vec<&str> = SHOPS_SORTED.iter().map(|shop| shop.name()).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted);
    }
}
