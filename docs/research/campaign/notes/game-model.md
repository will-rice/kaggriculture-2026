# Kaggriculture engine game model

This is a measured reference for the supplied engine, byte-for-byte identical to
`kaggle-environments==1.32.7`; the engine overrides the older competition notes
where they disagree. [Verified by the common engine check used by every probe.](../experiments/_common.py)

## Objective and season clock

The game has two farms. Each begins with `$3000`, and the higher bank at the
final recorded state wins; equal banks tie. The engine writes each player's raw
final bank into `reward`, but match value is only the relative outcome: a win is
not worth more because its bank margin is larger. [Season/outcome probe.](../experiments/season_visibility.py)

The nominal season is `720` recorded states, numbered `0..719`, spanning days
`0..29` and hours `0..23`. There is an important runner detail: state `0` is the
initialized observation, policies are called on states `0..718`, and state
`719` is final, so each policy is actually called `719` times. The final state
is day `29`, hour `23`. [Season/outcome probe.](../experiments/season_visibility.py)

An ordinary rollover follows the action on hour `23` and appears in the next
recorded state at hour `0`; for example, the first rollover is produced by the
action at step `23` and observed at step `24`. Because play stops at recorded
state `719`, only `29` end-of-day transitions run; the last appears at step
`696`, and day `29` never receives its closing transition. [Timing probe.](../experiments/season_visibility.py)

Within each action step, resolution order is:

1. every main-farmer and supplied hand action, player by player;
2. market order slots, with per-unit transactions lockstepped between players;
3. town-shop and town-centre consumption, followed by a price refresh;
4. per-turn crop decay;
5. when the action closes a day: plant refresh, animal refresh, random weeds,
   inventory auto-drop, unit reset, and a possible shop unlock. [Season probe.](../experiments/season_visibility.py)
   [Market/town probe.](../experiments/market_town.py)

The transition boundary and ordering are exercised by the same probes.

At the end of a day, carried goods are auto-dropped subject to the shed cap; the
main farmer returns to `(4,4)`, all hired hands disappear, today's hire count
returns to `0`, and the per-unit inventory list becomes one empty main-farmer
inventory. [End-of-day probe.](../experiments/season_visibility.py)

Each empty unlocked tile independently attempts a weed spawn with probability
`0.005` during that transition. [Configuration probe.](../experiments/season_visibility.py)

## Unit actions

The main farmer and every currently hired hand can each issue one unit action
per policy call. Illegal actions—including malformed arguments, missing items,
wrong tile types, repeated daily actions, and off-board movement—silently do
nothing. Unit actions do not directly spend money; their costs below are items,
seeds, or the action opportunity. [All-verb probe.](../experiments/actions_economy.py)

| Action                           | Legal effect, cost, and yield                                                                                                                                                                                                                         | Source                                                      |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `NORTH`, `SOUTH`, `EAST`, `WEST` | Move one orthogonal tile if still on the board. Units may move onto locked land; collisions do not block movement.                                                                                                                                    | [Action probe](../experiments/actions_economy.py)           |
| `PASS`                           | No state change.                                                                                                                                                                                                                                      | [Action probe](../experiments/actions_economy.py)           |
| `DROP`                           | At a shed-access corner, attempts to deposit everything carried. It empties the unit inventory even when the shed has no room, silently destroying overflow.                                                                                          | [Action probe](../experiments/actions_economy.py)           |
| `PICKUP item [n]`                | At a shed-access corner, moves up to `n` units from shed to the acting unit; omitted `n` defaults to `1`. Carrying capacity is unbounded. Seeds cannot be picked up because they are stored separately.                                               | [Action probe](../experiments/actions_economy.py)           |
| `PLACE item [n]`                 | For a carried animal, places one on an empty matching structure under the unit. Otherwise, at a shed-access corner, deposits up to `n` carried items; omitted `n` defaults to `1`. Unlike `DROP`, excess remains carried rather than being destroyed. | [Action probe](../experiments/actions_economy.py)           |
| `PLANT crop`                     | On an empty unlocked tile, consumes `1` separately stored seed and creates the crop. If all units' same-turn requests for a crop exceed that crop's seed stock, all those requests are cancelled atomically.                                          | [Action probe](../experiments/actions_economy.py)           |
| `WATER`                          | Marks the plant watered for the day. A second watering that day is a no-op. In a one-time crop's productive window, it adds `1` yield, or `2` while fertilizer is active, up to the crop cap.                                                         | [Action and crop probes](../experiments/crops_animals.py)   |
| `HARVEST`                        | Once mature, moves every held yield unit into the acting unit's inventory. A one-time plant disappears; an ongoing plant or animal remains with held yield reset to `0`.                                                                              | [Action probe](../experiments/actions_economy.py)           |
| `FERTILIZE`                      | On a plant, consumes `1` carried fertilizer. It is active on the current day and the next `2` days—`3` days inclusive—and extends rather than stacks when reapplied.                                                                                  | [Action and crop probes](../experiments/actions_economy.py) |
| `DIG`                            | Clears a plant, weed, empty coop, or empty pasture. It cannot clear an occupied animal structure.                                                                                                                                                     | [Action probe](../experiments/actions_economy.py)           |
| `BUILD_COOP`                     | Creates a free empty coop on an empty unlocked tile.                                                                                                                                                                                                  | [Action probe](../experiments/actions_economy.py)           |
| `BUILD_PASTURE`                  | Creates a free empty pasture on an empty unlocked tile.                                                                                                                                                                                               | [Action probe](../experiments/actions_economy.py)           |
| `FEED`                           | On an animal, consumes `1` carried wheat and marks it fed; further feeds that day are no-ops.                                                                                                                                                         | [Animal/action probe](../experiments/actions_economy.py)    |
| `COLLECT_FERTILIZER`             | If the animal's boolean fertilizer flag is set, clears it and adds `1` fertilizer to the unit inventory. Repeating before another night yields nothing.                                                                                               | [Animal/action probe](../experiments/actions_economy.py)    |
| `CARE`                           | Marks the animal cared for that day; further care that day is a no-op. It consumes no item.                                                                                                                                                           | [Animal/action probe](../experiments/actions_economy.py)    |

On the default `10×10` board, the shed is not a tile. Its only access positions
are the four inner corners `(4,4)`, `(5,4)`, `(4,5)`, and `(5,5)`. `DROP`,
`PICKUP`, and shed-form `PLACE` work from those positions even while the tile is
locked; other tile-mutating actions still fail there until its quadrant is
owned. [Locked-corner action probe.](../experiments/actions_economy.py)

Unit actions precede the market. Thus a shed-adjacent unit can `DROP` and sell
the deposited goods in the same action step, but a seed bought by the market
cannot be planted until the next policy call. `SELL` never reads carried
inventory. [Ordering and sale-source probes.](../experiments/actions_economy.py)
[Market probe.](../experiments/market_town.py)

## Crops

Age is measured in day boundaries after planting. `interval=0` denotes a
one-time crop rather than a repeating interval. “Ongoing” means harvesting does
not remove the plant; it does not mean unlimited production. [Crop lifecycle probe.](../experiments/crops_animals.py)

| Crop       |   Seed | First harvest age | Max-yield day | Interval | Yield cap | Ongoing | Source                                        |
| ---------- | -----: | ----------------: | ------------: | -------: | --------: | :-----: | --------------------------------------------- |
| Wheat      |  `$10` |               `2` |           `4` |      `0` |       `6` |   no    | [Crop probe](../experiments/crops_animals.py) |
| Carrot     |  `$20` |               `2` |           `3` |      `0` |       `4` |   no    | [Crop probe](../experiments/crops_animals.py) |
| Tomato     |  `$50` |               `8` |           `8` |  `1` day |       `4` |   yes   | [Crop probe](../experiments/crops_animals.py) |
| Strawberry | `$100` |              `10` |          `10` | `2` days |       `4` |   yes   | [Crop probe](../experiments/crops_animals.py) |
| Melon      |  `$80` |              `10` |          `12` |      `0` |       `6` |   no    | [Crop probe](../experiments/crops_animals.py) |

One-time crops start with `1` latent yield but cannot be harvested before their
first-yield age. Plain watering changes wheat at ages `2,3,4`, carrot at ages
`2,3`, and melon at ages `6,7,8,9,10`. Their resulting plain caps are therefore
`4`, `3`, and `6`; fertilizer can reach the configured caps of `6`, `4`, and
`6` at ages `4`, `3`, and `8`. Although melon's configured max-yield day is
`12`, plain watering at ages `11` and `12` is dead because it already hit its
cap at age `10`. [One-time crop probe.](../experiments/crops_animals.py)

An unharvested one-time crop begins losing `1` held unit every `2` action steps
starting at global step `(planted_day + max_yield_day + 1) × 24`; it becomes a
weed when held yield reaches `0`. For a day-`0` planting those start steps are
`96` for carrot, `120` for wheat, and `312` for melon. [Crop decay probe.](../experiments/crops_animals.py)

With daily water and a day-`0` planting, tomato produces `1` unit at day ages
`8,9,10,11`; strawberry produces `1` at ages `10,12,14,16`. Each has exactly
`4` scheduled productions. Harvesting between them frees held capacity but does
not reset that lifetime count. Their post-production decay starts at global
steps `288` and `408`; an untouched full crop takes `4` decay ticks to become a
weed. [Ongoing crop probe.](../experiments/crops_animals.py)

Every new plant starts with an unwatered count of `1`, so failing to water on
the planting day makes it a weed that night. After a watered night resets the
counter to `0`, the plant tolerates one dry night and weeds on the second
consecutive dry night. [Plant-health probe.](../experiments/crops_animals.py)

## Animals and structures

| Animal | Purchase | Required structure | Product | First production age | Interval | Max held | Source                                          |
| ------ | -------: | ------------------ | ------- | -------------------: | -------: | -------: | ----------------------------------------------- |
| Goose  |   `$300` | Coop               | Egg     |             `4` days |  `1` day |      `4` | [Animal probe](../experiments/crops_animals.py) |
| Cow    |   `$400` | Pasture            | Milk    |             `8` days | `2` days |      `6` | [Animal probe](../experiments/crops_animals.py) |
| Sheep  |   `$500` | Pasture            | Wool    |             `6` days | `3` days |      `6` | [Animal probe](../experiments/crops_animals.py) |

Animals are bought into the shed, picked up, and placed on a matching empty
structure. Coops and pastures cost `$0`; only the animal costs money. Animals
have ongoing production and no fixed production-count limit while they remain,
but unharvested product stops accumulating at the max-held value. [Animal and structure probes.](../experiments/actions_economy.py)
[Animal schedule probe.](../experiments/crops_animals.py)

An uncared production adds `1` product. Every surviving animal makes its
fertilizer flag available each night, representing at most `1` uncollected
fertilizer rather than a stack. A first unfed production night still produces
the base `1`, but after `2` consecutive unfed nights the animal escapes and its
empty structure remains. Feeding costs `1` wheat per day. [Animal state-transition probe.](../experiments/crops_animals.py)

`CARE` pays later. A day that is both cared and fed banks a `+1` bonus after
that night's production check. Banked bonuses accumulate until a later
production night; they are consumed only if the animal is fed then and the
result is capped by max held. In the probe, `2` banked care actions make the
later goose production add `3` units rather than the base `1`. Care on the
production day itself is too late for that production. [Care probe.](../experiments/crops_animals.py)

## Land, labour, and storage

Each farm is `10×10`, split into four `5×5` quadrants. The northwest `25` tiles
start owned and the other `75` locked. `BUY_LAND` unlocks northeast, southwest,
then southeast for `$1000`, `$2000`, and `$4000`, yielding all `100` tiles.
[Land probe.](../experiments/actions_economy.py)

`HIRE` buys a hand for the rest of the current day. With the default multiplier
`1`, hire number `n` that day costs `fib(n)` under the engine's zero-based
sequence `1,1,2,3,5,8,13,21,34,55,…`; the first `7` hires total `$33`. The
sequence and hand roster reset at end of day. [Hiring probe.](../experiments/actions_economy.py)
[Daily reset probe.](../experiments/season_visibility.py)

Hands spawn by least occupancy across the four shed-access corners, with fixed
northwest-to-southeast tie-breaking. With the main farmer initially at `(4,4)`,
the first four hand spawns are `(5,4)`, `(4,5)`, `(5,5)`, `(4,4)`. Locked
spawns are not trapped because movement across locked tiles is legal. [Hiring/action probe.](../experiments/actions_economy.py)

The shed cap is `100` total non-seed items across crops, products, fertilizer,
and unplaced animals. Seeds are separate and do not consume it. `BUY_PRODUCT`
and `BUY_ANIMAL` fail when full; explicit shed-form `PLACE` fills only available
room and leaves excess carried. In contrast, `DROP` and the end-of-day automatic
drop erase every carried entry after moving what fits, silently destroying the
overflow. A controlled end-of-day example with `99` stored and `3` carried
ended at `100` and destroyed `2`. [Storage probe.](../experiments/actions_economy.py)

## Shared market

Both players use one market inventory and one price per product. The initial
inventory is `I0=10000` for every product. A price is an integer rounded from a
curve and floored at `$1`. [Market curve probe.](../experiments/market_town.py)

For distance `x = |inventory-I0|`, shape `f`, base `b`, scale distance `T`, and
target `q`, the engine uses `amp = q×b/f(T)`. Below `I0`, price is
`max(1, round(b + amp×f(x)))`; at or above `I0`, it is
`max(1, round(b - amp×f(x)))`. Thus `q` is the fraction of base added or removed
at distance `T`, before the floor. [Formula and sample probe.](../experiments/market_town.py)

The shape functions are `linear(x)=x`, `sq(x)=x²`, `sqrt(x)=√x`,
`log(x)=ln(1+x)`, and `log10(x)=log10(1+x)`. For the scarcity-only hinge,
`u=x/T` and `hinge(x)=u + 8×max(0,u-1)²`; it is linear through the knee at
`x=T`, then rises quadratically. [Shape/hinge probe.](../experiments/market_town.py)

| Product    |   Base |      I0 |     T | Below shape / target | Above shape / target | Price at I0−T / I0 / I0+T | Source                                        |
| ---------- | -----: | ------: | ----: | -------------------- | -------------------- | ------------------------: | --------------------------------------------- |
| Wheat      |  `$25` | `10000` | `400` | sqrt / `0.80`        | log / `0.20`         |         `$45 / $25 / $20` | [Market probe](../experiments/market_town.py) |
| Carrot     |  `$35` | `10000` | `450` | hinge / `1.00`       | sqrt / `0.70`        |         `$70 / $35 / $10` | [Market probe](../experiments/market_town.py) |
| Tomato     |  `$60` | `10000` | `200` | hinge / `0.40`       | sqrt / `0.60`        |         `$84 / $60 / $24` | [Market probe](../experiments/market_town.py) |
| Strawberry | `$120` | `10000` | `100` | sqrt / `0.70`        | linear / `1.60`      |        `$204 / $120 / $1` | [Market probe](../experiments/market_town.py) |
| Melon      | `$250` | `10000` | `300` | log / `0.20`         | sq / `3.60`          |        `$300 / $250 / $1` | [Market probe](../experiments/market_town.py) |
| Egg        |  `$50` | `10000` | `332` | hinge / `0.40`       | log / `0.20`         |         `$70 / $50 / $40` | [Market probe](../experiments/market_town.py) |
| Milk       | `$160` | `10000` | `122` | sqrt / `0.60`        | linear / `1.60`      |        `$256 / $160 / $1` | [Market probe](../experiments/market_town.py) |
| Wool       | `$200` | `10000` | `105` | log / `0.20`         | sq / `3.20`          |        `$240 / $200 / $1` | [Market probe](../experiments/market_town.py) |
| Fertilizer | `$100` | `10000` | `200` | linear / `0.40`      | linear / `0.40`      |       `$140 / $100 / $60` | [Market probe](../experiments/market_town.py) |

The hinge's post-knee acceleration is large: at scarcity `2T`, carrot, tomato,
and egg quote `$385`, `$300`, and `$250`, versus `$52`, `$72`, and `$60` at
scarcity `T/2`. [Hinge sample probe.](../experiments/market_town.py)

### Market orders and price impact

The action accepts at most `10` market-order entries per player per policy call;
later entries are silently dropped. The verbs are `BUY_SEED`, `BUY_PRODUCT`,
`BUY_ANIMAL`, `SELL`, `HIRE`, and `BUY_LAND`. The first four take
`item, quantity`; the last two are atomic and unquantified. [Order-limit probe.](../experiments/market_town.py)

- `BUY_SEED` charges the fixed crop seed price and adds to the separate seed
  store. It neither reads nor changes market inventory.
- `BUY_ANIMAL` charges the fixed animal price and deposits into the shed if
  there is room. It neither reads nor changes product-market inventory.
- `BUY_PRODUCT` is accepted only for wheat and fertilizer. Each unit is quoted
  at the price for post-buy inventory, then inventory falls by `1`.
- `SELL` can sell any market product, but draws only from the shed. Each unit is
  paid at the current-inventory quote. If that quote exceeds `$1`, market
  inventory rises by `1`; a `$1` sale pays the coin but does not add supply.
  [Purchase, sale, and floor probe.](../experiments/market_town.py)

All transaction effects in this list were exercised directly.

Quantities execute one unit at a time, so a sale changes later units' quotes.
Starting carrot inventory at `9100`, a three-unit sale paid `$385`, `$384`, then
`$382`, and ended at `9103`. At the same order position, both players are quoted
against the same pre-commit inventory before either unit commits: from inventory
`9550`, two simultaneous one-unit carrot sellers both received `$70`, after
which inventory was `9552`. [Sale-impact probe.](../experiments/market_town.py)

Because buys quote post-buy inventory, buying and immediately selling one wheat
against an otherwise unchanged market had net cost `$0` and restored inventory.
[Round-trip probe.](../experiments/market_town.py)

## Shared town demand and the episode seed

Town consumption happens after player market orders, can drive inventory below
its initial level, and therefore raises subsequent scarcity prices. The town
centre consumes `1` unit of every non-fertilizer product on action steps
divisible by `24`, beginning at action step `0`. It fires `30` times in the
episode; with shops disabled and no player trades, each such inventory ends at
`9970` while fertilizer remains `10000`. [Town-centre probe.](../experiments/market_town.py)

Every unlocked shop instance consumes on action steps divisible by `4`.
Multi-product shops take `1` of each listed item; a single-product shop takes
`2`. Duplicate instances each consume independently. In a probe with two yarn
stores and one bakery, one tick took `4` wool plus `1` egg and `1` wheat.
[Shop-demand probe.](../experiments/market_town.py)

| Shop           | Products consumed per tick by one instance        | Source                                      |
| -------------- | ------------------------------------------------- | ------------------------------------------- |
| Bakery         | `1` egg, `1` wheat                                | [Shop probe](../experiments/market_town.py) |
| Pizza shop     | `1` milk, `1` tomato, `1` wheat                   | [Shop probe](../experiments/market_town.py) |
| Brunch spot    | `1` egg, `1` wheat, `1` strawberry                | [Shop probe](../experiments/market_town.py) |
| Yarn store     | `2` wool                                          | [Shop probe](../experiments/market_town.py) |
| Ice cream shop | `1` strawberry, `1` milk, `1` wheat               | [Shop probe](../experiments/market_town.py) |
| Pet cafe       | `2` carrot                                        | [Shop probe](../experiments/market_town.py) |
| Smoothie shop  | `1` strawberry, `1` milk                          | [Shop probe](../experiments/market_town.py) |
| Farmers market | `1` wheat, `1` carrot, `1` tomato, `1` strawberry | [Shop probe](../experiments/market_town.py) |

A shop instance unlocks after the end-of-day transitions into days
`3,6,9,12,15,18,21,24`, for a maximum of `8` instances. Draws are with
replacement, so neither variety nor uniqueness is guaranteed. A shop unlocked
at day `3` is already present for the market/town processing of action step
`72`. [Unlock probe.](../experiments/market_town.py)

The input episode seed is resolved once, moved to `env.info`, and the
configuration's `seed` visible to agents is reset to `None`; the resolved value
is absent from observations. Each end-of-day RNG is initialized from
`(resolved_seed × 1000003) XOR current_day`. Weed attempts for both farms consume
that RNG stream before the shop draw, so the shop sequence is deterministic for
the seed and evolving farm occupancy, not the seed alone. In the controlled
pass-agent probe, repeating seed `1` reproduced the same eight-shop sequence,
while seed `2` produced a different sequence. [Seed/unlock probe.](../experiments/market_town.py)

## Shared and private observations

Each seat's `player` is `0` or `1`. Both receive equal current values for
`farms`, `market`, `town`, `day`, and `hour`. `farms[player]` and
`farms[1-player]` expose public money, all tiles and structures, farmer and hand
positions, unlocked quadrants, and today's hire count. Thus the opponent's bank,
layout, crops, animals, labour, and positions are observable. [Visibility probe.](../experiments/season_visibility.py)

Each seat receives only its own `private`: shed counts, seed counts, and every
unit's carried inventory. No opponent `private`, shed, seeds, or carried goods
appear inside `farms[1-player]`. [Visibility probe.](../experiments/season_visibility.py)

One framework quirk matters to portable policies: seat `0` receives a `step`
key, while seat `1` does not. Derive the action step as
`day × configuration.turnsPerDay + hour` instead. [Two-seat visibility probe.](../experiments/season_visibility.py)
