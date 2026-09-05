# Kaggriculture policy task

## Objective

Author a fast two-player policy that maximizes match wins: finish with more bank
than the opponent at the final recorded state. The engine exposes raw final bank
as reward, but evaluation cares about win/loss/tie, not winning margin. A season
has `720` recorded states but only `719` policy calls per seat. [Verified season probe.](experiments/season_visibility.py)

Treat `engine/kaggriculture.py` as ground truth for
`kaggle-environments==1.32.7`; older notes lose every disagreement. [Engine identity check.](experiments/_common.py)

Opponent doctrine is strict: measure behavior only through the harness; never
read opponent source.

```bash
uv run --project /home/will/projects/kaggriculture-2026/.claude/worktrees/campaign campaign play AGENT --vs NAME... --seeds A-B --workers N
uv run --project /home/will/projects/kaggriculture-2026/.claude/worktrees/campaign campaign check AGENT
```

Valid measured opponents are `router_v1`, `router2929`, `v54`, `v56`,
`shopforge`, and `indarkarhana`.

## Key insights

- Optimize win probability, not nominal profit margin. The final observation is
  recorded step `719` at day `29`, hour `23`; policies act on steps `0..718`.
  The last completed daily refresh appears at recorded step `696`, so the final
  day never closes. [Timing probe.](experiments/season_visibility.py)
- Labour is deliberately cheap and temporary. With multiplier `1`, daily hires
  cost `1,1,2,3,5,8,13,…` and disappear at day-end; the first `7` cost only
  `$33`. [Hiring probe.](experiments/actions_economy.py)
- The shed is a hard shared inventory bottleneck at `100` non-seed items.
  End-of-day auto-drop and `DROP` silently erase overflow; seeds are outside the
  cap. [Storage probe.](experiments/actions_economy.py)
- The product market and town demand are shared. Your sales lower later prices;
  opponent sales do too. Same-position transactions are lockstepped, so both
  players receive the same pre-commit quote for that unit. [Sale probe.](experiments/market_town.py)
- Seed purchases resolve after unit actions and cannot fund same-step planting.
  Harvest enters carried inventory, while `SELL` reads only the shed; route
  units through a shed corner before selling. [Ordering probes.](experiments/actions_economy.py)
- Locked land blocks tile operations but not movement or shed operations from
  the four inner corners `(4,4)`, `(5,4)`, `(4,5)`, `(5,5)`. [Action probe.](experiments/actions_economy.py)
- Water a seed on planting day: its initial dry count is already `1`, so it
  weeds on the first night if ignored. After a watered night, two consecutive
  dry nights kill it. [Plant-health probe.](experiments/crops_animals.py)
- “Ongoing” crops are finite producers. Tomato yields on ages `8,9,10,11` and
  strawberry on `10,12,14,16`, then stop after `4` scheduled productions.
  [Crop schedule probe.](experiments/crops_animals.py)
- Melon has the largest base quote at `$250`, but its quadratic glut curve hits
  the `$1` floor by `I0+T`; wheat's logarithmic glut curve is much gentler and
  wheat also feeds every animal. [Market probe.](experiments/market_town.py)
- Read the opponent's public farm and bank. You cannot see its shed, seeds, or
  carried inventory. [Visibility probe.](experiments/season_visibility.py)

## Rules

### Clock and resolution

The board defaults to `10×10`; both players start with `$3000`. Days and hours
are zero-based, with `24` hour labels per day. An ordinary day-end follows the
action at hour `23` and is visible at the next hour `0`. [Configuration/timing probe.](experiments/season_visibility.py)

Each action step resolves unit actions, market orders, town demand, crop decay,
then—when applicable—the daily refresh. Daily refresh updates plants and
animals, attempts weeds at probability `0.005` per empty owned tile, auto-drops
inventory, resets labour and positions, and may unlock a shop. [Season probe.](experiments/season_visibility.py)

### Crops

`interval=0` means one-time. Ongoing means harvest leaves the plant, not that it
produces forever. [Crop probe.](experiments/crops_animals.py)

| Crop       |   Seed | First yield age | Max-yield day | Interval | Cap | Ongoing |
| ---------- | -----: | --------------: | ------------: | -------: | --: | :-----: |
| Wheat      |  `$10` |             `2` |           `4` |      `0` | `6` |   no    |
| Carrot     |  `$20` |             `2` |           `3` |      `0` | `4` |   no    |
| Tomato     |  `$50` |             `8` |           `8` |      `1` | `4` |   yes   |
| Strawberry | `$100` |            `10` |          `10` |      `2` | `4` |   yes   |
| Melon      |  `$80` |            `10` |          `12` |      `0` | `6` |   no    |

All crop-table values are asserted by the [crop lifecycle probe](experiments/crops_animals.py).

One-time plants begin with `1` latent unit. Plain productive water ages are
wheat `2,3,4`, carrot `2,3`, and melon `6,7,8,9,10`; resulting plain yields are
`4`, `3`, and `6`. Fertilized water adds `2` rather than `1` and lasts `3` days
inclusive. Melon is already capped at age `10`, despite max-yield day `12`.
[One-time crop probe.](experiments/crops_animals.py)

Unharvested plants eventually lose `1` unit every `2` action steps and turn to
weeds at zero. Harvest clears one-time plants; it resets held yield but not the
finite production count of ongoing plants. [Decay probe.](experiments/crops_animals.py)

### Animals and structures

| Animal |   Cost | Structure | Product | First yield age | Interval | Max held |
| ------ | -----: | --------- | ------- | --------------: | -------: | -------: |
| Goose  | `$300` | Coop      | Egg     |             `4` |      `1` |      `4` |
| Cow    | `$400` | Pasture   | Milk    |             `8` |      `2` |      `6` |
| Sheep  | `$500` | Pasture   | Wool    |             `6` |      `3` |      `6` |

All animal-table values are asserted by the [animal lifecycle probe](experiments/crops_animals.py).

Coops and pastures cost `$0`. Buy an animal into the shed, pick it up, and place
it on a matching empty structure. An uncared production adds `1` product.
Feeding costs `1` carried wheat per day; after `2` consecutive unfed nights the
animal escapes and leaves the structure. Each surviving night makes at most `1`
fertilizer available. [Animal/action probes.](experiments/crops_animals.py)

Feed plus care banks a `+1` bonus after that night's production check. Bonuses
can accumulate and are consumed by a later fed production, subject to max held;
care never boosts the same night's production. [Care probe.](experiments/crops_animals.py)

### Land, hands, and shed

The northwest `5×5` quadrant begins unlocked. `BUY_LAND` then unlocks northeast,
southwest, southeast for `$1000`, `$2000`, `$4000`. [Land probe.](experiments/actions_economy.py)

The daily hire price is `farmHandCostMult × fib(hires_today)`, with the default
multiplier `1` and sequence `1,1,2,3,5,8,13,21,34,55,…`. Each hand gets one
unit action per policy call, then disappears at day-end. [Hiring probe.](experiments/actions_economy.py)

The shed holds `100` total crops, products, fertilizer, and unplaced animals.
`BUY_PRODUCT`/`BUY_ANIMAL` fail when full. Shed-form `PLACE` preserves unplaced
excess, but `DROP` and automatic day-end deposit discard overflow. Carrying has
no cap. [Storage probe.](experiments/actions_economy.py)

### Market

All products begin at inventory `I0=10000`. Prices are integer-rounded and
floored at `$1`. At distance `x`, amplitude is `target×base/f(T)`; price adds
`amp×f(x)` below `I0` and subtracts it at or above `I0`. [Curve probe.](experiments/market_town.py)

Shapes are linear, square, square root, natural log of `1+x`, base-ten log of
`1+x`, and hinge. For hinge, `u=x/T` and
`f(x)=u+8×max(0,u−1)²`, so scarcity accelerates beyond `T`. [Hinge probe.](experiments/market_town.py)

| Product    |   Base |     T | Scarcity shape/target | Glut shape/target | Price at I0−T / I0 / I0+T |
| ---------- | -----: | ----: | --------------------- | ----------------- | ------------------------: |
| Wheat      |  `$25` | `400` | sqrt / `0.80`         | log / `0.20`      |         `$45 / $25 / $20` |
| Carrot     |  `$35` | `450` | hinge / `1.00`        | sqrt / `0.70`     |         `$70 / $35 / $10` |
| Tomato     |  `$60` | `200` | hinge / `0.40`        | sqrt / `0.60`     |         `$84 / $60 / $24` |
| Strawberry | `$120` | `100` | sqrt / `0.70`         | linear / `1.60`   |        `$204 / $120 / $1` |
| Melon      | `$250` | `300` | log / `0.20`          | square / `3.60`   |        `$300 / $250 / $1` |
| Egg        |  `$50` | `332` | hinge / `0.40`        | log / `0.20`      |         `$70 / $50 / $40` |
| Milk       | `$160` | `122` | sqrt / `0.60`         | linear / `1.60`   |        `$256 / $160 / $1` |
| Wool       | `$200` | `105` | log / `0.20`          | square / `3.20`   |        `$240 / $200 / $1` |
| Fertilizer | `$100` | `200` | linear / `0.40`       | linear / `0.40`   |       `$140 / $100 / $60` |

All market-table parameters and samples are asserted by the [market curve probe](experiments/market_town.py).

Market order semantics:

- `BUY_SEED crop n`: fixed seed price; separate seed store; no market impact.
- `BUY_ANIMAL animal n`: fixed animal price; shed capacity required; no product-market impact.
- `BUY_PRODUCT item n`: only wheat or fertilizer; quote post-buy inventory, then
  subtract `1` inventory per unit.
- `SELL item n`: only from shed; quote current inventory, then add `1` supply per
  unit unless the quote is exactly `$1`, in which case supply does not change.
- `HIRE` and `BUY_LAND`: atomic, unquantified orders.

These unit effects and the zero-cost unchanged-market buy/sell round trip were
verified directly. [Transaction probe.](experiments/market_town.py)

At most `10` market-order entries per player are processed per action step;
extras vanish. Quantities within an entry execute per unit, and the public price
refreshes after each order entry and again after town consumption. [Order probe.](experiments/market_town.py)

### Town and seed

The town centre takes `1` of every non-fertilizer product every `24` action
steps, starting at step `0`; it fires `30` times. Shops consume every `4` action
steps. A multi-product shop takes `1` of each ingredient; yarn store and pet
cafe are single-product shops and take `2` wool or carrot. Duplicate shop
instances consume independently. [Town probes.](experiments/market_town.py)

Shop recipes are:

- bakery: egg, wheat;
- pizza shop: milk, tomato, wheat;
- brunch spot: egg, wheat, strawberry;
- yarn store: wool;
- ice cream shop: strawberry, milk, wheat;
- pet cafe: carrot;
- smoothie shop: strawberry, milk;
- farmers market: wheat, carrot, tomato, strawberry.

One shop instance unlocks on days `3,6,9,12,15,18,21,24`, capped at `8` and
drawn with replacement. The resolved episode seed initializes daily weed/shop
randomness, but the resolved value is hidden from agents; farm occupancy affects
how far the RNG advances before the shop draw. [Unlock/seed probe.](experiments/market_town.py)

### Visibility and win condition

Both seats see shared `farms`, `market`, `town`, `day`, and `hour`. Each public
farm contains money, tiles, main-farmer/hand positions, unlocked quadrants, and
today's hire count. Each seat sees only its own `private` shed, seeds, and unit
inventories. Compare your bank with `farms[1-player].money`; a strict lead at the
final state wins and equality ties. [Visibility/outcome probe.](experiments/season_visibility.py)

## Interface

Submit a Python file whose policy is callable as:

```python
def agent(observation, configuration):
    return {"farmer": ["PASS"], "hands": [], "market": []}
```

The loader executes the file and runs the **last callable left in its global
namespace**, regardless of its name. Keep `agent` as that last callable; in
particular, do not import or define another callable after it.

The action dict is:

```text
{
  "farmer": [UNIT_OP, ...args],
  "hands": [[UNIT_OP, ...args], ...],
  "market": [[MARKET_OP, ...args], ...]
}
```

Accepted unit operations are `NORTH`, `SOUTH`, `EAST`, `WEST`, `PASS`, `DROP`,
`PICKUP item [n]`, `PLACE item [n]`, `PLANT crop`, `WATER`, `HARVEST`,
`FERTILIZE`, `DIG`, `BUILD_COOP`, `BUILD_PASTURE`, `FEED`,
`COLLECT_FERTILIZER`, and `CARE`. `DROP` is real even though older published
action schemas omitted it. [Every unit verb was exercised.](experiments/actions_economy.py)

Accepted market operations are `BUY_SEED crop n`, `BUY_PRODUCT item n`,
`BUY_ANIMAL animal n`, `SELL item n`, `HIRE`, and `BUY_LAND`. [Order catalogue probe.](experiments/market_town.py)

The engine silently no-ops invalid actions. Return one hand action per hand you
intend to operate. Keep each call below the `1` second `actTimeout`.
[Configuration probe.](experiments/season_visibility.py)

Do not depend on `observation.step`: the framework supplies it to seat `0` but
not seat `1`. Use
`observation.day * configuration.turnsPerDay + observation.hour` instead.
[Two-seat observation probe.](experiments/season_visibility.py)
