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
from collections.abc import Mapping
from typing import Any

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
        fuel: Total goods in the shed. Their fuel is stored spendable resource
            in hand; ours is harvested stock not yet sold.
        money: Coins banked. Not a shaped component -- it decides
            ``game_result`` alone, exactly as their city-tile count does.
    """

    city: int
    unit: int
    research: int
    fuel: int
    money: float


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
    return Counts(
        city=unlocked + plants,
        unit=1 + len(farm["hands"]),
        research=len(observation["town"]["unlocked_shops"]),
        fuel=sum(observation["private"]["shed"].values()),
        money=farm["money"],
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
            total += GAME_RESULT_WEIGHT * _rank(current.money, other)
        else:
            self.previous = current

        return total / NORMALISER


def _rank(ours: float, theirs: float) -> float:
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
