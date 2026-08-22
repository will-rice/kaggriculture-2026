"""Toad's ``StatefulMultiReward``, mapped onto our farm economy.

Their phase 1 is shaped and their phases 2-5 are sparse; this is the shaped one,
transcribed from ``toad/reward_spaces_lux.py`` lines 138-247. Theirs is a
gather -> deposit economy scored on city tiles, ours is
plant -> water -> harvest -> pick up -> carry -> drop -> sell scored on coins
banked, so each of their five components is mapped onto the nearest thing our
game counts. No component is added that they did not have, and none is dropped.

Three properties of theirs are easy to lose in translation and are reproduced
exactly:

* Every component is a **per-turn delta**, not a level (reward_spaces_lux.py:196-207).
  The space carries the previous turn's counts itself.
* ``fuel`` is clamped non-negative -- "Don't penalize losing fuel at night"
  (:201). Ours is clamped for the matching reason: selling stock must not read
  as a loss, since the sale is already paid for through ``game_result``.
* The whole sum is divided by **500** (:227), which the analysis this
  reproduction was briefed from omitted entirely. Without it the shaped signal
  arrives 500x too large -- the same class of scale bug that already cost this
  project a run.

``game_result`` rides inside the shaped reward at 10x from the first turn (:219),
firing only on the last one. It is not a separate phase.

Deliberately absent: anything that rewards selling *well*. Toad had no such
component, and price x quantity is what decides our game. That gap is the
reproduction's most important open question and is left standing so phase 1 can
measure it rather than paper over it.
"""

import dataclasses
import os
from collections.abc import Mapping
from typing import Any

import torch

from kaggriculture.learn.encoding import ANIMAL_NAMES
from kaggriculture.observation import is_plant

# reward_spaces_lux.py:166-177, verbatim. `step` defaults to 0. there and phase 1
# overrides it to 0.005 via reward_space_kwargs
# (conf/conv_phase1_shaped_reward.yaml:29). `fuel` is 0.005 by default. They are
# different knobs that happen to share a value, which is a good way to get this
# wrong.
GAME_RESULT_WEIGHT = 10.0
CITY_WEIGHT = 1.0
UNIT_WEIGHT = 0.5
RESEARCH_WEIGHT = 0.1
FUEL_WEIGHT = 0.005
STEP_WEIGHT = 0.005
# Present in their weights dict but disabled, with the live value commented out
# at -0.01 (:172-174). Kept disabled, as the winning recipe had it.
FULL_WORKERS_WEIGHT = 0.0

# OURS, NOT THEIRS. Deviation D1 made explicit: Toad's component set pays for
# every link of their chain, and ours is missing exactly one -- converting shed
# stock into coins. Nothing in their five components rewards selling, and price
# times quantity is what decides our game.
#
# The weight is anchored on their own ratios rather than picked. Their
# score-deciding component is `city` at 1.0, so one city tile is worth
# 1.0 / 500 = 0.002. Measured over a real 719-turn episode (kaito vs kaito, seed
# 7, banking 128,341 against the 125,773 corpus median): 129 of 719 turns bank
# anything, the median sale is 648 coins and the season gains 155,526.
#
#   one median sale  648 * 0.001 / 500 = 0.0013  = 0.65 city tiles
#   season total  155,526 * 0.001 / 500 = 0.311  vs their ~0.2 for city
#   ratio to game_result             = 15.6x     vs their ~10x
#
# All three land within a small factor of theirs, and 0.001 is a round number in
# the style of their 1.0 / 0.5 / 0.1 / 0.005. Erring slightly high is deliberate:
# coins are our *sole* score-decider, where theirs splits between city and units.
#
# Off by default. The baseline reward must stay exactly as it was measured.
# A MAPPING CORRECTION, not a new component. Toad's `city` at 1.0 is their
# COMPOUNDING, SCORE-DECIDING asset. We pointed that weight at our terrain --
# unlocked tiles and plants -- and left our actual economic engine, the animals,
# in `fuel` at 0.005, where the shed sums them in with fertilizer. A COW and one
# unit of fertilizer have been earning the identical 0.005 all along, which is
# why the livestock potential read zero in every logged iteration.
#
# THE WEIGHT IS NOT THEIR 1.0, and the reason is measured rather than argued.
# This project has produced four pump-shaped defects and all four share one
# shape: a term paying for something free or reversible. So the question is
# asked before the weight is set -- what does this pay for that costs nothing,
# or can be undone and redone? -- and the answer here is "nothing", by three
# engine facts and two inequalities.
#
# COSTS NOTHING? No. `_commit_unit`'s BUY_ANIMAL charges `ANIMALS[item]["cost"]`
# -- a fixed 300/400/500, not a market quote that can be walked down. Contrast
# the count this term deliberately is NOT: `BUILD_COOP` and `BUILD_PASTURE`
# write a tile for zero coins on any empty square, so a structure count would
# pay for roughly a hundred free tiles a season. See `_hosts_animal`.
#
# UNDONE AND REDONE? No, and three separate engine rules close it.
#   * `_process_market` quotes SELL only for `item in PRODUCTS`, and ANIMALS is
#     disjoint from PRODUCTS, so an animal can never be turned back into coins.
#   * DIG returns early on a tile holding an animal ("Does NOT remove a placed
#     animal").
#   * The one cycle that does exist -- PICKUP an animal out of the shed, PLACE
#     it on its structure -- moves it between `herd` and `placed`, and `capital`
#     sums both, so the round trip is exactly zero. It is also unclamped, unlike
#     `fuel`: the negative leg lands first and is not forgiven. Measured over a
#     reference season, gross positive capital deltas are 29 against a net 15,
#     which is that cycle showing up and cancelling.
#   Capital falls only when an animal escapes after two consecutive unfed days,
#   which destroys the asset with no refund.
#
# SO THE SEASON TOTAL IS BOUNDED BY COINS IRREVERSIBLY SPENT:
#
#   capital <= 3000/300 + earned/300         (startingMoney over the cheapest animal)
#   reward  <= 0.05 * (10 + earned/300)/500 = 0.0010 + earned * 3.3e-7
#
# An agent that sells nothing all season caps at 0.0010, one twentieth of the
# terminal `game_result` at 10/500 = 0.020. Every animal past the tenth has to
# be paid for by running the whole production chain first.
#
# AND ACQUIRING MUST NOT OUT-PAY OPERATING, which is what fixes the magnitude.
# Measured on 1.32.6, economic_policy mirror, seeds 7 and 11: it ends the season
# holding 15 animals, its `fuel` term totals 0.0092-0.0098 (the ~950 units of
# produce its fields and herd actually yielded) and its whole shaped reward
# excluding capital is 0.146-0.157.
#
#   capital at 0.05 * 15 / 500 = 0.0015   <- 15% of the produce it enabled, 1.0%
#                                            of the shaped total
#   capital at 1.00 * 15 / 500 = 0.0300   <- 3x every unit of produce harvested
#                                            all season, and 1.5x game_result
#
# Requiring the term to stay under the produce it enables caps the weight at
# 0.0098 * 500 / 15 = 0.33. Toad's published 1.0 for this slot fails that by 3x,
# so it is not reused; 0.05 sits an order of magnitude inside the bound and is
# the value the margin arm already ran animals at. PRE-REGISTERED: if this arm
# underperforms, this weight is the first knob and 0.33 is its ceiling -- not
# something to tune mid-run.
#
# Land needs no change: quadrants already enter `city` through unlocked tiles.
CAPITAL_WEIGHT = 0.05

MONEY_WEIGHT = 0.001

# OURS, AND NOT A SHAPING TERM AT ALL. The competition's win condition, verbatim,
# is "having the most coins in the bank at the end of 720 turns", so the quantity
# to maximise is our bank MINUS the opponent's, and a coin denied is worth exactly
# a coin earned. Every arm this project has run maximised our own bank instead;
# corpus mining over 1,350 seats then measured bank against ladder rating at
# Pearson -0.043, and our 98.8th-percentile banker scores ~1,014.
#
# `margin` below is not shaping added to an objective -- it IS the objective,
# decomposed onto the turns that produced it. The per-turn deltas telescope to
# the terminal margin exactly (both farms open on the same `startingMoney`, so
# the opening margin is zero and no constant leaks in).
#
# THE WEIGHT IS THE MONEY WEIGHT, deliberately. `margin` and `money` are both
# denominated in coins, so reusing 0.001 says the two rewards put a coin of
# margin and a coin of bank on the same scale, and the arm's whole diff is which
# coin it counts. It also keeps the episode total in the band the value head has
# already been trained over: a 160,000-coin margin reads
# 160,000 * 0.001 / 500 = 0.32, against the 0.13-0.31 the shaped reward spanned,
# and well inside VALUE_BOUND.
MARGIN_WEIGHT = 0.001

# OURS, AND NOT OPTIONAL. A purely differential reward is a fixed point of mirror
# self-play: two copies of a policy that banks nothing bankrupt each other
# identically, R_1 = -R_0, and the advantage vanishes. Measured here over 30
# iterations -- mean bank 0, best episode 4 coins of a possible 200,000,
# advantage -0.0008 +- 0.0128. FLG kept `+0.3 PER FACTORY` alongside their
# differentials for exactly this reason, and note what that is: a COUNT of a
# durable asset, not revenue.
#
# WHAT IT IS: the per-turn change in producing capital -- animals owned, whether
# still in the shed or already standing on their structure. It is a floor under
# the differential, NOT a co-objective, and it is deliberately not denominated in
# coins.
#
# WHY NOT OUR OWN BANK, which was the obvious choice and is wrong. A seat's final
# bank correlates +0.98 with its OPPONENT'S final bank (1.32.5: +0.977, 1.32.6:
# +0.976, measured across the corpus by build). The shared order book sets the
# level for both players and the seat contributes only the ~2.2% margin on top,
# so a bank-proportional reward is 98% a measurement of the market draw. That is
# why four bank-shaped arms went nowhere. Sizing makes it concrete -- at the
# competitive operating point (1.32.6 median bank 80,660, median winning margin
# 2,561) a per-coin absolute term large enough to matter anywhere is ruinous
# here:
#
#   margin term    0.001 * 2,561              = +2.56
#   terminal rank  10.0                       = +10.0
#   own bank at 0.0002 * 77,660               = +15.53   <- SIX TIMES the margin
#   own bank at 0.00002 * 77,660              = +1.55    <- still 61% of it
#   capital at 0.05 * 15 structures           = +0.75    <- 6% of the objective
#
# A COUNT does not scale with the draw, so its share stays ~6% whether the market
# paid 80,000 or 160,000. A coin-denominated term cannot have that property.
#
# WHY 0.05, AND WHY IT CANNOT PUMP. An un-priced count is exactly the shape that
# gave this project three "shaped up, bank down" failures, so the weight is set
# by a structural inequality rather than by taste: the margin term already
# charges the full purchase price of anything bought, so acquiring capital is
# reward-NEGATIVE unless the animal later earns its keep through the margin.
#
# That inequality only holds because the count is of ANIMALS, not of the
# structures they stand on. `BUILD_COOP` and `BUILD_PASTURE` cost nothing at all
# in this engine, so a structure count would have been free reward -- roughly a
# hundred buildable tiles at 0.05 is 0.01 after the normaliser, against an
# objective worth 0.025 at the competitive point. See `_hosts_animal`.
#
#   GOOSE  cost 300 -> margin -0.300, capital +0.050, net -0.250
#   COW    cost 400 -> margin -0.400, capital +0.050, net -0.350
#   SHEEP  cost 500 -> margin -0.500, capital +0.050, net -0.450
#
# Every animal in the game clears that test with 6x to 9x of headroom, so the
# weight would have to rise above 0.30 before a buying spree paid for itself.
# 0.05 is an order of magnitude below the cheapest animal's charge. Anchored on
# the reference: economic_policy finishes a season with ~15 structures.
#
# If the margin improves while capital and bank both collapse, THIS WEIGHT IS TOO
# SMALL. That read is pre-registered rather than something to tune around mid-run.
ABSOLUTE_WEIGHT = 0.05

# Arm W escalates the weight tenfold. Read from the environment rather than
# passed down, because the rollout runs in worker subprocesses that inherit the
# environment but not our arguments -- a weight threaded through the parent
# alone would leave every worker silently on the default and the arm would
# measure nothing.
MONEY_WEIGHT_ENV = "TOAD_MONEY_WEIGHT"
MONEY_SIGNED_ENV = "TOAD_MONEY_SIGNED"


def money_signed() -> bool:
    """Whether the money component keeps both signs.

    Arm W clamped the delta at zero, mirroring their ``fuel`` term, and at a
    weight of 0.01 that produced a money pump: buying in order to sell back
    earns shaped reward because the clamp forgives the purchase while the spread
    quietly eats the coins. Measured over updates 1-34, ``money_term`` climbed
    0.0064 -> 0.028 with ``gross_purchases`` tracking it 3320 -> 4414 and
    ``bank_mean`` pinned at zero.

    Unclamped, the per-turn deltas telescope to exactly the net coins banked
    over the episode, so a round trip that loses to the spread earns
    net-negative reward and the pump is unprofitable by construction. Genuine
    investment -- seeds, hires, land -- goes negative on the turn it is paid and
    recoups at the sale, a gap gamma=0.999 comfortably spans.

    This also makes the component potential-based with Phi = money (Ng, Harada
    and Russell 1999): a potential-based shaping term cannot change the optimal
    policy's ranking of whole episodes, only guide exploration toward it. That is
    a strictly stronger safety property than the clamped version had.

    Returns:
        True when the environment asks for the signed form.
    """
    return os.environ.get(MONEY_SIGNED_ENV, "") == "1"


def money_weight() -> float:
    """Return the money component's weight for this run.

    Returns:
        ``MONEY_WEIGHT`` unless the environment overrides it.
    """
    return float(os.environ.get(MONEY_WEIGHT_ENV, MONEY_WEIGHT))


# reward_spaces_lux.py:227. Tuned against their 360-turn game; ours runs 719
# decisions. Kept verbatim rather than rescaled, because faithfulness is the
# point of this pass -- see the deviation list in the task report.
NORMALISER = 500.0


@dataclasses.dataclass(frozen=True)
class Counts:
    """The five things a shaped reward is a difference of, for one seat.

    Attributes:
        city: Unlocked tiles plus standing plants. Their city tile is the
            durable asset that both compounds and scores; ours is worked land.
        unit: The farmer plus hired hands, our exact analogue of their workers.
        research: Shops the town has unlocked. Their research points are an
            irreversible one-way capability unlock and this is our only other
            monotone progress counter.
        capital: Producing animals -- those standing on the board and those
            still in the shed. Our compounding asset, in Toad's `city` slot.
            Empty structures are excluded: building one is free, so counting it
            would pay for nothing.
        fuel: Product stock in the shed. Animals are excluded; they are capital,
            not consumable stock. Their fuel is stored spendable resource
            in hand; ours is harvested stock not yet sold.
        money: Coins banked. Not a shaped component -- it decides
            ``game_result`` alone, exactly as their city-tile count does.
        opponent: The other seat's coins banked. Public, unlike the shed, so
            unlike ``fuel`` it is legible from either seat's observation. Held
            here rather than recomputed at the call site so that one series of
            ``Counts`` is sufficient for every reward in this module -- the
            margin reward cannot then be handed a series that silently lacks
            the quantity it is a difference of.
    """

    city: int
    unit: int
    research: int
    fuel: int
    capital: int
    money: float
    opponent: float


_ANIMALS = frozenset(ANIMAL_NAMES)


def _hosts_animal(tile: object) -> bool:
    """Whether a tile has a producing animal standing on it.

    THE TEST IS THE ANIMAL, NEVER THE STRUCTURE, and the difference is a pump.
    ``BUILD_COOP`` and ``BUILD_PASTURE`` cost NOTHING -- they need only an empty
    tile -- so a count of ``kind in {"COOP", "PASTURE"}`` pays for up to a
    hundred free structures a season, which at any weight worth having swamps
    the margin term. Counting the animal instead prices the same asset at the
    300-500 coins it actually costs, which is what makes the capital term
    unpumpable.

    The engine writes ``{"kind": "PASTURE"}`` for an empty structure and
    replaces it wholesale with ``_new_animal(...)`` -- a dict carrying an
    ``"animal"`` key -- when one is placed, so the two states are disjoint and
    nothing is counted twice. Tiles are also sometimes ``None`` or the string
    ``"LOCKED"``, hence the type check rather than a bare ``in``.
    """
    return isinstance(tile, dict) and "animal" in tile


def counts(observation: Mapping[str, Any], seat: int) -> Counts:
    """Read the five counts for one seat from one observation.

    Args:
        observation: The observation as handed to that seat's agent.
        seat: Which seat to read. ``fuel`` is only legible for the seat the
            observation belongs to, because the shed lives in the unindexed
            ``private`` mapping and no opponent equivalent is exposed.

    Returns:
        That seat's counts.
    """
    farm = observation["farms"][seat]
    tiles = farm["tiles"]
    unlocked = sum(tile != "LOCKED" for row in tiles for tile in row)
    plants = sum(is_plant(tile) for row in tiles for tile in row)
    shed = observation["private"]["shed"]
    herd = sum(count for good, count in shed.items() if good in _ANIMALS)
    placed = sum(1 for row in tiles for tile in row if _hosts_animal(tile))
    return Counts(
        city=unlocked + plants,
        unit=1 + len(farm["hands"]),
        research=len(observation["town"]["unlocked_shops"]),
        fuel=sum(count for good, count in shed.items() if good not in _ANIMALS),
        capital=herd + placed,
        money=farm["money"],
        opponent=float(observation["farms"][1 - seat]["money"]),
    )


class StatefulMultiReward:
    """Toad's shaped reward for one seat, carrying the previous turn's counts.

    One instance per seat per episode. Their space holds both players' counts in
    a length-2 array; ours holds one seat's, because our ``fuel`` analogue is
    private and an opponent's shed is not observable. The arithmetic is
    otherwise theirs.
    """

    def __init__(self, observation: Mapping[str, Any], seat: int) -> None:
        """Seed the previous counts from the opening position.

        Their ``_reset`` seeds city and unit at 1 and research and fuel at 0
        (reward_spaces_lux.py:242-246), which *is* the Lux starting state, so
        their first delta is zero. Seeding from the opening observation is the
        same statement about our game.

        Args:
            observation: The first observation of the episode.
            seat: Which seat this instance scores.
        """
        self.seat = seat
        self.previous = counts(observation, seat)

    def step(self, observation: Mapping[str, Any], done: bool) -> float:
        """Return the shaped reward for the turn that produced ``observation``.

        Follows reward_spaces_lux.py:190-227. On the final turn ``game_result``
        fires as a rank in {-1, +1} and is weighted 10x, exactly as theirs does.

        Args:
            observation: The observation after the action was taken.
            done: Whether the episode ended on this turn.

        Returns:
            The scalar shaped reward, already divided by ``NORMALISER``.
        """
        current = counts(observation, self.seat)
        items = {
            "city": float(current.city - self.previous.city),
            "unit": float(current.unit - self.previous.unit),
            "research": float(current.research - self.previous.research),
            # Clamped at zero, their line 201.
            "fuel": float(max(current.fuel - self.previous.fuel, 0)),
            "step": 1.0,
        }
        weights = {
            "city": CITY_WEIGHT,
            "unit": UNIT_WEIGHT,
            "research": RESEARCH_WEIGHT,
            "fuel": FUEL_WEIGHT,
            "step": STEP_WEIGHT,
        }
        total = sum(items[key] * weight for key, weight in weights.items())

        if done:
            # Their GameResultReward ranks the players and maps to {-1, +1}
            # (reward_spaces_lux.py:111-112). Ours ranks on coins banked, which
            # is our terminal objective the way city tiles are theirs.
            other = observation["farms"][1 - self.seat]["money"]
            total += GAME_RESULT_WEIGHT * rank(current.money, other)
        else:
            self.previous = current

        return total / NORMALISER


def shaped(series: list[Counts], won: float, money_weight: float = 0.0) -> torch.Tensor:
    """Return the per-turn shaped reward for a counts series.

    The stateful class above is the faithful transcription of their per-step
    space; this is the same arithmetic over a whole episode at once, so it can
    sit beside ``rollout._differences`` and be read the same way. The series is
    one longer than the number of decisions -- one entry per state, terminal
    state included -- so the differences are one per decision, and a series the
    same length as the turns raises rather than silently shifting every reward
    onto the turn after the one that earned it.

    Args:
        series: The counts at each state, terminal state included.
        won: The terminal result in {-1., 0., +1.}, added on the last turn at
            ``GAME_RESULT_WEIGHT`` exactly as their ``game_result`` component
            does.
        money_weight: Weight on the per-turn coin delta. Zero -- the default --
            reproduces Toad's component set exactly and is what the baseline
            run measures. ``MONEY_WEIGHT`` enables phase-1b, the one deliberate
            addition. Clamped non-negative like their ``fuel``, so that spending
            coins on seeds, hands or land is not punished: investment is how the
            chain advances, and the return on it is already paid when the goods
            are sold.

    Returns:
        ``(turns,)`` float32 shaped rewards, already divided by ``NORMALISER``.

    Raises:
        ValueError: If ``series`` holds fewer than two states.
    """
    if len(series) < 2:
        raise ValueError(
            f"a shaped series needs one state per turn plus the terminal one, "
            f"got {len(series)}"
        )
    rewards = []
    for before, after in zip(series[:-1], series[1:], strict=True):
        total = (
            CITY_WEIGHT * (after.city - before.city)
            + UNIT_WEIGHT * (after.unit - before.unit)
            + RESEARCH_WEIGHT * (after.research - before.research)
            # DELIBERATELY UNCLAMPED, unlike `fuel` directly below. `capital`
            # sums animals in the shed and animals on their structures, so
            # PICKUP then PLACE is a round trip through both halves. Clamping
            # would forgive the negative leg and pay for every cycle -- an
            # unbounded pump out of one animal. Unclamped it is exactly zero.
            + CAPITAL_WEIGHT * (after.capital - before.capital)
            # Clamped at zero, their line 201: selling stock must not read as a
            # loss, because the sale is already paid for through game_result.
            + FUEL_WEIGHT * max(after.fuel - before.fuel, 0)
            + STEP_WEIGHT
            # Ours. Zero unless phase-1b enables it; see MONEY_WEIGHT.
            + money_weight
            * (
                (after.money - before.money)
                if money_signed()
                else max(after.money - before.money, 0.0)
            )
        )
        rewards.append(total / NORMALISER)
    rewards[-1] += GAME_RESULT_WEIGHT * won / NORMALISER
    return torch.tensor(rewards, dtype=torch.float32)


def margin(series: list[Counts], won: float) -> torch.Tensor:
    """Return the per-turn margin reward for a counts series.

    The win condition, decomposed onto the turns that produced it, plus the
    smallest absolute term that survives mirror self-play. Three components and
    no others:

    * ``MARGIN_WEIGHT`` on the per-turn change in (our bank - theirs). Telescopes
      to the terminal margin exactly.
    * ``ABSOLUTE_WEIGHT`` on the per-turn change in producing capital. Present
      only so that the reward is not identically zero-sum, which is a fixed
      point self-play falls straight into. A count rather than coins, because a
      seat's bank correlates +0.98 with its opponent's and a coin-denominated
      floor is therefore 98% a reading of the shared market draw.
    * ``GAME_RESULT_WEIGHT`` on the terminal rank, on the last turn alone --
      Toad's own component, unmodified, and the literal statement that what is
      being maximised is *winning* rather than the size of the win. At 10/500 =
      0.02 it dominates the margin term for the near-parity episodes the corpus
      says decide this ladder (a 2,561-coin median margin reads 0.005).

    **No un-zeroed potential survives to the horizon.** The Grzes (AAMAS 2017)
    condition is that a potential must be zero at a trajectory's stopping state,
    or its ``g^N Phi(s_N)`` term modifies the policy. This project shipped that
    bug once, by carrying the value of *held produce* into turn 719: at
    gamma 0.999 holding from turn 619 returned 0.905 of its shaped value, so in a
    market both farms were pushing below base, refusing to sell was
    shaped-optimal, and "shed fills, bank flat" was built into the reward.

    Two of the three components here cannot express that mistake, being coins or
    a terminal rank. The capital term is the one that does carry a state to the
    horizon, and it is made safe STRUCTURALLY rather than by zeroing: the margin
    term already charges the full purchase price of an animal, which is 6x to 9x
    what the capital term pays for it, so ending the season holding capital is
    only ever profitable if that capital *earned* through the margin. There is no
    turn late enough to make a buying spree pay. Held produce, meanwhile, is
    worth exactly nothing here -- ``fuel`` never enters this reward.

    Args:
        series: The counts at each state, terminal state included, so the
            differences are one per decision. One longer than the turns.
        won: The terminal result in {-1., 0., +1.}, from ``rank``.

    Returns:
        ``(turns,)`` float32 rewards, already divided by ``NORMALISER``.

    Raises:
        ValueError: If ``series`` holds fewer than two states.
    """
    if len(series) < 2:
        raise ValueError(
            f"a margin series needs one state per turn plus the terminal one, "
            f"got {len(series)}"
        )
    rewards = [
        (
            MARGIN_WEIGHT
            * ((after.money - after.opponent) - (before.money - before.opponent))
            + ABSOLUTE_WEIGHT * (after.capital - before.capital)
        )
        / NORMALISER
        for before, after in zip(series[:-1], series[1:], strict=True)
    ]
    rewards[-1] += GAME_RESULT_WEIGHT * won / NORMALISER
    return torch.tensor(rewards, dtype=torch.float32)


def sparse(series: list[Counts], won: float) -> torch.Tensor:
    """Return the per-turn ``GameResultReward`` for a counts series.

    Phases 2 onward train on this alone: +1 terminal for a win, -1 for a loss,
    0 for a draw, and nothing else. Zero on every turn but the last is not an
    approximation of sparse -- it is what sparse means. ``margin`` above
    decomposes the same win condition onto the turns that produced it, plus a
    small signed own-bank term; this reward decomposes it onto nothing, on
    purpose, so the phase boundary that switches to it means what the recipe
    says it means.

    Args:
        series: The counts at each state, terminal state included. The counts
            themselves are unused -- only the length sets the turn count --
            kept as the argument so this reward shares its call site with
            ``shaped`` and ``margin`` and an arm remains a choice of field
            rather than a re-run.
        won: The terminal result in {-1., 0., +1.}, from ``rank``.

    Returns:
        ``(turns,)`` float32 rewards, zero everywhere but the last turn.

    Raises:
        ValueError: If ``series`` holds fewer than two states.
    """
    if len(series) < 2:
        raise ValueError(
            f"a sparse series needs one state per turn plus the terminal one, "
            f"got {len(series)}"
        )
    rewards = torch.zeros(len(series) - 1, dtype=torch.float32)
    rewards[-1] = won
    return rewards


def rank(ours: float, theirs: float) -> float:
    """Return +1 for a win, -1 for a loss, 0 for a draw.

    ``scipy.rankdata`` on two players maps to ``(rank - 1) * 2 - 1``, which is
    -1 and +1 for a decisive result and 0 for both on a tie
    (reward_spaces_lux.py:112). Reproduced without the scipy dependency.
    """
    if ours > theirs:
        return 1.0
    if ours < theirs:
        return -1.0
    return 0.0
