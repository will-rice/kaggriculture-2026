"""Tests for route replay: retrieval, hysteresis, realignment and the fallback.

Fixtures build observations the way ``tests/routes/test_signature.py`` and
``tests/routes/test_store.py`` do -- through the engine's own constructors
where one exists (``engine._new_private``, ``engine._new_plant``) -- so a test
cannot quietly agree with code that reads a key the real game never writes.

A prototype here is identified by a ``seed``, which is simply how many melon
tiles stand on the board it recorded. That makes "the board this route was
recorded from" and "a board one tile away from it" things a test can state
exactly, rather than approximately.

None of these tests read the real store: they are fast tests, and a fast test
that needs ``/data`` is a fast test that fails on every machine but one.
"""

import json
import subprocess
import sys
import zipfile
from typing import Any

import pytest
from kaggle_environments.envs.kaggriculture import kaggriculture as engine

from kaggriculture import economic_policy
from kaggriculture.constants import BOARD_SIZE, EPISODE_STEPS, PRODUCTS, TURNS_PER_DAY
from kaggriculture.learn.corpus import CORPUS, Sample
from kaggriculture.routes.play import RouteAgent, realign, route_units
from kaggriculture.routes.scripts.harvest import harvest
from kaggriculture.routes.signature import HANDS_SCALE, SIGNATURE_FIELDS, signature
from kaggriculture.routes.store import Prototype

_ARCHIVE = "kaggriculture-episodes-2026-08-03.zip"

# How far the marginal-improvement board sits from the prototype it is nearest
# to. Both prototypes are then far from it and one is a single melon tile
# nearer, which is what "marginally closer" has to mean for the hysteresis
# test to be about a margin rather than about a rout.
_MARGINAL_GAP = 18

_TURNS = EPISODE_STEPS - 1


def _farm(melons: int = 0, money: float = 3_000.0, hands: int = 0) -> dict[str, Any]:
    """Return a fresh, independent farm carrying ``melons`` planted tiles.

    Each call builds its own tile grid: two farms in one observation must
    never share one mutable list, or mutating one player's board silently
    mutates the other's.
    """
    tiles: list[list[Any]] = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    for index in range(melons):
        tiles[index // 5][index % 5] = engine._new_plant("MELON", 0, TURNS_PER_DAY)
    return {
        "tiles": tiles,
        "money": money,
        "farmer": [4, 4],
        "hands": [[4, 4] for _ in range(hands)],
        "unlocked_quadrants": ["NW"],
        "hires_today": hands,
    }


def _observation(
    melons: int = 0,
    money: float = 3_000.0,
    day: int = 0,
    hour: int = 0,
    hands: int = 0,
) -> dict[str, Any]:
    """Return a minimal well-formed observation for one seat.

    ``private`` comes from the engine's own ``_new_private`` so the shed is
    dense -- every product keyed at zero -- and code that reaches for a
    missing key fails here rather than only on real data.
    """
    return {
        "player": 0,
        "day": day,
        "hour": hour,
        "farms": [_farm(melons, money, hands), _farm()],
        "market": {
            "prices": dict.fromkeys(PRODUCTS, 100),
            "inventory": dict.fromkeys(PRODUCTS, 10_000),
        },
        "town": {"unlocked_shops": []},
        "private": engine._new_private(),
    }


def _prototype(seed: int = 1, market: list[list[Any]] | None = None) -> Prototype:
    """Return a route that recorded ``seed`` melon tiles on every one of its turns.

    Every turn carries the same signature and the same do-nothing field plan,
    so a test that moves the live board is moving the only thing under test.
    ``market`` is the market half of the recorded action -- the half this
    design throws away and recomputes.
    """
    action: dict[str, Any] = {
        "farmer": ["PASS"],
        "hands": [],
        "market": [] if market is None else market,
    }
    return Prototype(
        bank=150_000.0,
        opponent_bank=100_000.0,
        rating=2_700.0,
        actions=[dict(action) for _ in range(_TURNS)],
        signatures=[signature(_observation(melons=seed), 0)] * _TURNS,
    )


def _observation_matching(seed: int) -> dict[str, Any]:
    """Return the board ``_prototype(seed)`` recorded, exactly."""
    return _observation(melons=seed)


def _observation_with_empty_shed() -> dict[str, Any]:
    """Return the board ``_prototype()`` recorded, with nothing in the shed.

    An empty shed and empty carried inventories are what make the market half
    independent of the field half: no plan can place anything into the shed
    this turn, so the only thing that can move the market orders is which
    policy produced them.
    """
    return _observation_matching(seed=1)


def _observation_marginally_closer_to(seed: int) -> dict[str, Any]:
    """Return a board far from every prototype and one tile nearer to ``seed``'s."""
    return _observation(melons=seed + _MARGINAL_GAP)


def _observation_unlike(seed: int) -> dict[str, Any]:
    """Return a board no route recorded: another phase, another farm, another bank."""
    assert seed < 25
    return _observation(melons=25, money=100_000.0, day=20, hour=12)


def economic_policy_market(observation: dict[str, Any]) -> list[list[Any]]:
    """Return the market orders the vendored policy would send this turn."""
    return economic_policy.agent(observation)["market"]


def test_the_market_orders_come_from_the_live_policy_not_the_route() -> None:
    """Prices depend on both players' cumulative sales.

    A replayed SELL quantity is priced for a market that no longer exists, so
    the production plan is replayed and the market is recomputed. This is the
    single most important division in this design.
    """
    prototype = _prototype(market=[["SELL", "WHEAT", 40]])

    action = RouteAgent([prototype]).act(_observation_with_empty_shed())

    assert action["market"] != [["SELL", "WHEAT", 40]]
    assert action["market"] == economic_policy_market(_observation_with_empty_shed())


def test_an_unmatched_board_falls_back_to_the_whole_policy() -> None:
    """The floor of this design is the agent we already ship, not zero."""
    agent = RouteAgent([_prototype(seed=1)])

    action = agent.act(_observation_unlike(seed=1))

    assert agent.fallbacks == 1
    assert action == economic_policy.agent(_observation_unlike(seed=1))


def test_hysteresis_keeps_the_current_route_unless_another_is_clearly_better() -> None:
    """Thrashing between routes yields a sequence no route would ever play.

    Each prototype is individually coherent; alternating between them is not.
    A new prototype must beat the incumbent by a margin, not by a hair.
    """
    agent = RouteAgent([_prototype(seed=1), _prototype(seed=2)])
    agent.act(_observation_matching(seed=1))

    agent.act(_observation_marginally_closer_to(seed=2))

    assert agent.current == 0


def test_realignment_issues_orders_to_the_units_we_actually_have() -> None:
    """A replayed action assumes a unit stands where the prototype's unit stood.

    When the boards diverge the action is a no-op at best and an order to the
    wrong hand at worst. Orders map onto our nearest unit, and orders for units
    we do not have are dropped rather than issued to somebody else.
    """
    prototype_units = [(4, 4), (7, 1), (0, 9)]
    ours = [(4, 4), (7, 2)]

    realigned = realign(
        {"farmer": ["WATER"], "hands": [["DIG"], ["HARVEST"]]}, prototype_units, ours
    )

    assert realigned["farmer"] == ["WATER"]
    assert len(realigned["hands"]) == 1


def test_a_unit_standing_off_the_route_walks_onto_it_before_acting() -> None:
    """A stationary order executed one tile away operates on the wrong tile.

    ``WATER`` at (7, 2) waters whatever grows at (7, 2), not the crop the route
    meant. The order is only issued verbatim once our unit stands where the
    route's unit stood; until then it is a step toward that tile.
    """
    realigned = realign({"farmer": ["WATER"], "hands": []}, [(7, 1)], [(7, 2)])

    assert realigned["farmer"] == ["NORTH"]


def test_route_positions_are_replayed_from_the_actions_not_guessed() -> None:
    """The store records no unit positions, so replay has to derive them.

    Units start each day on the shed-access spawn and move only when the
    recorded action moves them, so folding the day's recorded actions forward
    reconstructs where every unit stood -- and a hand hired during the day
    appears at the engine's own spawn tile.
    """
    prototype = _prototype()
    prototype.actions[0] = {"farmer": ["NORTH"], "hands": [], "market": [["HIRE"]]}
    prototype.actions[1] = {"farmer": ["WEST"], "hands": [["SOUTH"]], "market": []}
    hired = list(prototype.signatures[0])
    hired[SIGNATURE_FIELDS.index("HANDS")] = 1.0 / HANDS_SCALE
    prototype.signatures[1] = tuple(hired)
    prototype.signatures[2] = tuple(hired)

    assert route_units(prototype, 0) == [(4, 4)]
    assert route_units(prototype, 1) == [(4, 3), (4, 4)]
    assert route_units(prototype, 2) == [(3, 3), (4, 5)]


@pytest.mark.slow
@pytest.mark.skipif(
    not (CORPUS / _ARCHIVE).exists(), reason="replay corpus not present on this machine"
)
def test_derived_route_positions_match_the_ones_the_engine_recorded() -> None:
    """The derivation is exact or the whole realignment is aimed at a guess.

    Every recorded episode carries the positions the engine actually had, so
    the reconstruction can be checked against them rather than argued for.
    Both seats of one episode are walked turn by turn: a hire, a day boundary
    or a blocked move landing one tile out would show here.
    """
    with zipfile.ZipFile(CORPUS / _ARCHIVE) as bundle:
        name = next(
            member
            for member in bundle.namelist()
            if member.endswith(".json") and "manifest" not in member
        )
        with bundle.open(name) as member:
            steps = json.load(member)["steps"]

    for seat in (0, 1):
        [route] = harvest(
            [Sample(archive=_ARCHIVE, name=name, seat=seat, rating=2_600.0)], 0.0
        )
        recorded = [
            [tuple(farm["farmer"]), *(tuple(hand) for hand in farm["hands"])]
            for farm in (
                step[seat]["observation"]["farms"][seat] for step in steps[:-1]
            )
        ]

        derived = [route_units(route, step) for step in range(len(route.actions))]

        assert derived == recorded


@pytest.mark.parametrize("forbidden", ["torch", "wandb", "kaggriculture.learn.corpus"])
def test_the_agent_path_imports_neither_torch_nor_the_corpus(forbidden: str) -> None:
    """The submission ships no corpus module and cannot afford torch.

    ``package.py`` drops ``corpus.py`` from the archive, so an agent that
    reaches it through an import chain raises ``ModuleNotFoundError`` on turn
    zero; torch costs 10.7 seconds of a 60-second overage pool. Both failures
    are invisible until a submission scores zero, so they are pinned here.
    """
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, kaggriculture.routes.play; "
            f"print({forbidden!r} in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert probe.stdout.strip() == "False"
