# Kaggriculture — competition notes

Reference notes for <https://www.kaggle.com/competitions/kaggriculture>. Rules
tables live in the environment package (`kaggle_environments/envs/kaggriculture/`,
`README.md` and `AGENTS.md`); this file records what the code depends on and
what the community has found that the docs get wrong.

## Format

- Two-player turn-based farming sim: 30 days × 24 turns = **720 turns**, most
  coins banked wins. Margin never matters, only win/loss/tie.
- Featured competition, **$50,000** (ten $5,000 prizes), started 29 Jul 2026.
- **Final submission deadline 30 Sep 2026.** Episodes keep running for ~2 weeks
  afterwards, then a single **Bradley-Terry tournament** over those episodes
  produces the final leaderboard — deliberately different from other Kaggle
  simulations, to damp hot streaks.
- Up to 5 submissions/day; only the **latest 2** are scored and matched.
- Leaderboard positions are skill ratings (~2,900 at the top today), not coins.
- **Joining the competition on the website is required before submitting.**

## What the agent must produce

`main.py` at the archive root exposing an `agent(obs) -> action` callable. Kaggle
`exec`s that file with an empty globals dict and takes the **last callable
defined in it**, so nothing may be imported after `agent`, and `__file__` is not
available. The archive is unpacked to `/kaggle_simulations/agent/`, which is on
`sys.path` while the module executes — hence `main.py` beside a flat copy of the
`kaggriculture` package.

`actTimeout` is **1 second per turn**, so the policy has to stay cheap; there is
no room for per-turn search.

## Engine vs documentation

The engine is the authority where the two disagree
([discussion 732450](https://www.kaggle.com/competitions/kaggriculture/discussion/732450),
confirmed against the interpreter source):

- A seed is created with `consecutive_unwatered = 1`, so **an unwatered seed
  becomes a weed the same night**.
- One-time crops stop gaining yield once `yield_units` hits `max_yield`. Melon
  caps at 6 units at **age 10**, so the documented 6–12 watering window has two
  dead days at the end — harvesting on cap rather than on `max_yield_day` is
  worth roughly 45% more bank in local play.
- The shed is not a tile. Its access points are exactly `(4,4)`, `(5,4)`,
  `(4,5)`, `(5,5)`, and `DROP`/`PICKUP` only work from an **unlocked** one.
- Hands spawn on those four tiles even when locked; a hand landing on `(5,5)`
  with only NW unlocked is walled in and wastes its wage for the day.
- `DIG` fails on an occupied coop or pasture, only clearing empty structures.
- `SELL` draws from the **shed**, never from a unit's carried inventory.
- Strawberry is capped at 4 productions, not an indefinite producer.

## Economics the policy leans on

- **Labour is cheap.** The n-th hire of a day costs `fib(n)` coins (1, 1, 2, 3,
  5, 8, …) and resets daily, while a worked tile returns tens of coins a day.
  The binding constraint is unit-turns, not money.
- **Melon dominates single-crop play**: 6 units × $250 base per tile, against a
  seed cost of 80. Locally it banks ~50k versus ~9k for wheat.
- **Melon also crashes hardest.** Its glut curve is `sq` with `above_target 3.6`,
  so ~150 melons sold drives the price from $250 to ~$25. The market is shared
  with the opponent, so a melon mirror-match collapses the price for both — a
  reason to expect the crop mix to matter more as the ladder matures.
- Wheat is the opposite: a `log` glut curve barely moves, and wheat is also the
  feed for every animal.

## Useful resources

- [Daily top-episode replay dataset](https://www.kaggle.com/datasets/kaggle/kaggriculture-episodes-index)
  — up to 20 GB/day of high-rated replays, for imitation learning or statistics.
- [Crop price crash analysis — why MELON dominates](https://www.kaggle.com/competitions/kaggriculture/discussion/732623)
- [Comment on the final evaluation](https://www.kaggle.com/competitions/kaggriculture/discussion/731587)

## Not yet exploited

- Animals (goose/cow/sheep), fertilizer, and `CARE` bonuses — the policy grows
  crops only.
- Reading the opponent's public farm to time sales against their harvests.
- Mixed cropping to stay off the melon price floor.
