"""What a farm is worth mid-chain, so the reward stops paying only for the last link.

A coin is banked by a seven-step, five-day chain: buy a seed, walk to a free
tile, plant it, water it on each of the days that add yield, walk back, harvest
it, and sell what the harvest put in the shed. The reward this loop trained on
paid for the last of those alone. Random masked exploration completes the whole
chain approximately never -- a freshly initialised policy reached a state where
``SELL`` was legal on 0 of 719 turns -- so the reward was never observed and the
gradient never pointed at production. That is the measured barrier this module
exists to remove.

**It removes it by paying for the pipeline, not by paying for the actions.**
Every sub-goal reward here is the change in one number, ``potential``: what the
farm is holding that is on its way to becoming money. A seed in the store, a
quadrant that has been unlocked, a plant standing in the ground with three units
of yield on it, a cow in a pasture, a crate of wheat in a farmer's arms, a shed
with produce in it. The reward for a transition is how much that number moved.

That formulation is chosen over a list of "+3 for planting, +5 for harvesting"
because of a theorem rather than a taste. Ng, Harada and Russell prove that
``F(s, a, s') = gamma * P(s') - P(s)``, for any function ``P`` of the state
alone, is *necessary and sufficient* for a shaping term to leave the optimal
policy unchanged. Two consequences, and both of them are the difference between
this working and this producing another agent that games its own reward:

**A shaped reward in this form cannot be farmed, structurally.** The shaping
collected over any sequence of transitions depends only on its endpoints, so
every cycle in state space pays exactly zero (and slightly less than zero at
``gamma < 1``, which is a small carrying cost on hoarded stock). An agent that
digs a plant up and replants it recovers nothing and is out the seed. An agent
that picks produce out of the shed and puts it back is out nothing and gains
nothing. This project has already produced two agents that maximised a shaped
term without banking -- one that suppressed the opponent, one that hired on
nearly every turn -- and neither failure is available to a potential.

That guarantee has one condition, and getting it wrong would have produced the
same failure a third time: **the potential at the state a trajectory stops in
must be zero**, because the episode's endpoint is the one endpoint the agent
chooses. ``progress_reward`` enforces it by construction rather than by
convention; its docstring carries the citation and the arithmetic.

**Nothing has to be decided about whether a link pays once or every time.** The
question does not arise: a link pays for the state it leaves behind. Planting
pays once because the tile is only planted once; watering pays every day
*because the engine adds a yield unit every day*, and pays nothing on a day
outside the crop's bonus window because nothing was added.

## The valuations, and why each is what it is

Everything is priced in coins, on the same scale as the bank, so the weight
against the terminal reward is 1 by construction and there is no relative weight
to tune. The prices are the engine's ``base`` figures rather than the live
market quote: a live quote moves when the *opponent* trades and when the town
consumes, which would pay and charge us for events no action of ours caused, and
would make the potential a function of a market both farms are pushing around.

``seeds``, ``land`` and the purchase price of livestock are carried at **cost**.
That makes every capital purchase exactly neutral -- money leaves, potential
arrives -- so the shaped reward never punishes an investment for being an
expense. Without it, buying the 1,000-coin NE quadrant reads as a 1,000-coin
mistake at the instant it happens and the agent learns to stay on 25 tiles.

Produce is carried at **full price whether it is in a farmer's arms or in the
shed**, which makes ``DROP``, ``PLACE`` and ``PICKUP`` neutral. That is not a
concession, it is the engine: ``_end_of_day`` runs ``_drop_inventories_to_shed``
over every inventory every 24 turns, so carrying produce to the shed is a thing
the farm does for free overnight and is not a link the agent has to discover.
The two are still counted separately, because "produce is being harvested but
never reaches the shed" and "produce reaches the shed but is never sold" are
different failures and a single total hides both.

Yield still standing in a field or an animal is carried at ``GROWING`` of its
price, and that fraction is the one genuine judgement call in the module.

Args and returns for each function are below; ``POTENTIAL_COMPONENTS`` is the
order every array in this module uses.
"""

import logging
from typing import Any, Mapping, Sequence

import torch

from kaggriculture.constants import ANIMALS, CROPS, LAND_PRICES, MARKET_PARAMS, PRODUCTS

LOGGER = logging.getLogger(__name__)

# What one unit of each tradeable thing is worth to the potential, in coins.
#
# Produce at the engine's `base` price -- the price the market quotes when its
# inventory sits at `I0`, which is where every episode opens -- and animals at
# what they cost to buy. Deliberately a table of engine constants and not a
# reading of `market["prices"]`: the live quote moves on the town's consumption
# schedule and on the opponent's sales, so a potential built from it would hand
# us a reward for the opponent flooding the market and a penalty for the town
# eating breakfast. It would also stop being a function of *our* state, which is
# the property the whole construction rests on.
VALUE = {
    **{item: float(MARKET_PARAMS[item]["base"]) for item in PRODUCTS},
    **{animal: float(ANIMALS[animal]["cost"]) for animal in ANIMALS},
}

# What a unit of yield is worth while it is still on the plant or the animal,
# as a fraction of what the same unit is worth once it has been harvested.
#
# It is less than 1 because standing yield is contingent on actions that have
# not happened yet. A plant that misses two consecutive waterings becomes a WEED
# and takes everything on it (`_daily_refresh_plants`); a one-time crop past
# `max_lifespan_step` sheds a unit every other turn until it is a weed
# (`_decay_plants`); an animal unfed for two days escapes and leaves its
# structure behind (`_daily_refresh_animals`). And in every case a unit still
# has to be walked to and harvested.
#
# 0.4 rather than 0.5 so that the *majority* of a crop's value is paid on the
# HARVEST transition, which is the link random exploration is least likely to
# stumble into and therefore the one that most needs to be worth finding, while
# leaving 0.4 * price on each watering, which is the link the agent has to
# repeat for five days running and therefore the one that most needs to not be
# free. Concretely, on wheat at base 25: watering inside the bonus window pays
# 10, and harvesting five units pays 0.6 * 25 * 5 - 10 = 65 net of the seed the
# tile gives back.
GROWING = 0.4

# The order every potential array in this module uses. Six numbers rather than
# one, because the sum is what the reward needs and the split is what a human
# needs: `growing` rising while `stored` stays flat is a farm that plants and
# never harvests, and `stored` rising while the bank stays flat is a farm that
# harvests and never sells. Both look identical in the total.
POTENTIAL_COMPONENTS = (
    "seeds",
    "land",
    "growing",
    "livestock",
    "carried",
    "stored",
)


def potential(observation: Mapping[str, Any]) -> list[float]:
    """Return what this seat's farm is holding on its way to the bank, by component.

    The seat is read from the observation rather than passed, exactly as
    ``rollout._bank`` reads it: the engine stamps each seat's own index into the
    observation it hands that seat, and ``observation["private"]`` -- the shed,
    the unplanted seeds and each unit's arms -- is legible only in that seat's
    own observation. A potential computed from someone else's observation would
    silently be a potential over an empty private mapping.

    Args:
        observation: One seat's own observation, mid-episode or terminal.

    Returns:
        One coin figure per name in ``POTENTIAL_COMPONENTS``, in that order.
        Sums to the whole farm's pipeline value; never negative.
    """
    seat = int(observation["player"])
    farm = observation["farms"][seat]
    private = observation["private"]

    seeds = sum(count * CROPS[crop]["seed"] for crop, count in private["seeds"].items())
    land = sum(LAND_PRICES[: len(farm["unlocked_quadrants"]) - 1])
    growing, livestock = _standing(farm["tiles"])
    carried = sum(
        count * VALUE[item]
        for held in private["inventories"]
        for item, count in held.items()
    )
    stored = sum(count * VALUE[item] for item, count in private["shed"].items())
    return [float(seeds), float(land), growing, livestock, float(carried), stored]


def _standing(tiles: Sequence[Sequence[Any]]) -> tuple[float, float]:
    """Return the coin value standing in the fields and in the structures.

    Split because they behave differently and a run needs to see which one is
    moving: a crop is a five-day loop the agent drives every turn, an animal is
    a 400-coin asset that produces on the engine's own schedule and escapes if
    it is not fed.

    A live plant is worth its seed *plus* the discounted yield on it, and a
    placed animal its purchase price plus the discounted yield it is holding.
    The cost term is what makes PLANT and PLACE neutral rather than a loss: an
    ongoing crop opens at ``yield_units == 0`` (``_new_plant``), so a tomato
    seed valued only by its yield would read as a 50-coin mistake at the moment
    it went into the ground and the agent would learn to plant nothing but
    wheat. It is also what prices the losses correctly -- a plant left unwatered
    becomes a WEED and takes the seed with it, an unfed animal escapes and takes
    400.

    An empty COOP or PASTURE is worth nothing, which is right: the engine
    charges nothing to build one.

    Args:
        tiles: ``farm["tiles"]``, indexed ``[y][x]``. Entries are ``None`` for
            bare soil, the string ``"LOCKED"``, or a structure mapping.

    Returns:
        ``(growing, livestock)`` in coins.
    """
    growing = 0.0
    livestock = 0.0
    for row in tiles:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            animal = tile.get("animal")
            if animal is not None:
                # `str` on the product because the engine's rules tables are
                # heterogeneous dicts -- `ANIMALS["COW"]` holds ints and strings
                # side by side -- so the item name comes back typed as a number
                # and cannot index `VALUE` without it.
                product = str(ANIMALS[animal]["product"])
                livestock += (
                    ANIMALS[animal]["cost"]
                    + GROWING * tile["yield_units"] * VALUE[product]
                )
            elif tile.get("kind") == "PLANT":
                growing += (
                    CROPS[tile["crop"]]["seed"]
                    + GROWING * tile["yield_units"] * VALUE[tile["crop"]]
                )
    return growing, livestock


def progress_reward(potentials: torch.Tensor, gamma: float) -> torch.Tensor:
    """Return the per-turn shaped reward for one whole episode.

    ``gamma * P(s') - P(s)``, with the ``gamma`` that the advantage is actually
    discounted by. The factor is not decoration: it is the whole of Ng, Harada
    and Russell's condition, and dropping it is the same defect this project
    already documented in its own differential reward -- an undiscounted
    per-turn delta optimised under a discounted objective is not
    policy-invariant, so it quietly asks for something other than the terminal
    objective.

    **The potential of the state after the last turn is zero, and that is not a
    detail.** The shaped return over a whole episode telescopes to
    ``gamma**N * P(s_N) - P(s_0)``. The second term is a constant and cannot
    change a policy. The first is a function of the state the agent chose to
    end in, and Grze&#347; (AAMAS 2017) shows it therefore *does* change the policy
    unless the potential at a trajectory's stopping state is set to zero.

    Left un-zeroed here, that term would pay the agent to finish the season
    holding stock. The arithmetic at our ``gamma`` of 0.999 and a 719-turn
    season: a unit held to the horizon from turn 619 returns ``0.999 ** 100``,
    or 90.5% of its base price, and from turn 700 it returns 98.1% -- so with
    both farms pushing the market price below base by selling into it, refusing
    to sell would be the *shaped-optimal* play for the last quarter of every
    episode. "The shed fills and the bank stays flat" would be built into the
    reward rather than being a risk of it, which is the third instance of a
    failure this project has already shipped twice.

    Zeroed, unsold stock is confiscated at the horizon -- which is not a
    penalty invented here but the game's own scoring rule: ``interpreter`` ends
    the season by setting each seat's reward to its ``money``, and a shed full
    of melons is worth nothing at that moment.

    That is why this function takes the *acting* states and appends the
    terminal zero itself, rather than accepting a state-major array with the
    terminal row already in it. The caller cannot supply a non-zero terminal
    because there is nowhere to put one. It also means this function is
    specifically for a whole episode: a truncated trajectory would need a
    bootstrap in that slot instead of a zero, and would need a different
    function that says so.

    What remains true after the zeroing is what the term was introduced for:
    the shaped return over any *cycle* of states still depends only on its
    endpoints, so no reversible loop pays anything, and holding stock still
    costs ``(1 - gamma)`` of its value per turn.

    Args:
        potentials: ``(turns, len(POTENTIAL_COMPONENTS))`` potentials, one row
            per state the policy acted from. The terminal state is not one of
            them and must not be included.
        gamma: The discount the advantage uses.

    Returns:
        ``(turns,)`` shaped reward in coins. The last entry is
        ``-P(s_last_acting)``: everything still in the pipeline at the horizon
        is handed back.

    Raises:
        ValueError: If ``potentials`` is not turn-major over the components --
            in particular, an array carrying the terminal row would be one turn
            too long, and would pair every reward with the wrong turn.
    """
    if potentials.ndim != 2 or potentials.shape[1] != len(POTENTIAL_COMPONENTS):
        raise ValueError(
            f"potentials must be (turns, {len(POTENTIAL_COMPONENTS)}) with one row "
            f"per acting state and no terminal row: got {tuple(potentials.shape)}"
        )
    total = potentials.sum(dim=1)
    after = torch.cat([total[1:], torch.zeros(1, dtype=total.dtype)])
    return gamma * after - total
