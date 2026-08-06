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
    mutates the other's. The grid, the farmer's spawn and every hand's spawn
    all come from the engine's own constructors, so tiles outside NW are
    ``LOCKED`` exactly as they are in a real game -- which is the whole
    subject of ``_shed_masked`` and cannot be tested on a board of ``None``.
    ``melons`` fill NW, which is the only quadrant that starts unlocked.
    """
    tiles: list[list[Any]] = [
        [engine._initial_tile(x, y, BOARD_SIZE) for x in range(BOARD_SIZE)]
        for y in range(BOARD_SIZE)
    ]
    for index in range(melons):
        tiles[index // 5][index % 5] = engine._new_plant("MELON", 0, TURNS_PER_DAY)
    farm = {
        "tiles": tiles,
        "money": money,
        "farmer": list(engine._default_spawn(BOARD_SIZE)),
        "hands": [],
        "unlocked_quadrants": ["NW"],
        "hires_today": hands,
    }
    for _ in range(hands):
        farm["hands"].append(engine._spawn_hand(farm, BOARD_SIZE))
    return farm


def _observation(
    melons: int = 0,
    money: float = 3_000.0,
    day: int = 0,
    hour: int = 0,
    hands: int = 0,
    shed: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Return a minimal well-formed observation for one seat.

    ``private`` comes from the engine's own ``_new_private`` so the shed is
    dense -- every product keyed at zero -- and code that reaches for a
    missing key fails here rather than only on real data. Every price is 100,
    so a clamp's arithmetic is something a test can state in one line.
    """
    private = engine._new_private()
    private["shed"].update(shed or {})
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
        "private": private,
    }


def _prototype(seed: int = 1, market: list[list[Any]] | None = None) -> Prototype:
    """Return a route that recorded ``seed`` melon tiles on every one of its turns.

    Every turn carries the same signature and the same do-nothing field plan,
    so a test that moves the live board is moving the only thing under test.
    ``market`` is the market half of the recorded action, which is replayed
    with the rest of the turn and clamped to what our own state supports.
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


def _observation_backing(market: list[list[Any]]) -> dict[str, Any]:
    """Return the board ``_prototype()`` recorded, stocked to back ``market``.

    A clamp that fires would hide whether the orders came from the route at
    all, so the board this is played on can afford the recorded order outright.
    """
    return _observation(melons=1, shed={"WHEAT": sum(order[2] for order in market)})


def _observation_marginally_closer_to(seed: int) -> dict[str, Any]:
    """Return a board far from every prototype and one tile nearer to ``seed``'s.

    An hour later than the board the incumbent was chosen on, because that is
    what a next turn is: two calls at turn zero would be two episodes, and
    ``act`` clears its incumbent there on purpose.
    """
    return _observation(melons=seed + _MARGINAL_GAP, hour=1)


def _observation_unlike(seed: int) -> dict[str, Any]:
    """Return a board no route recorded: another phase, another farm, another bank."""
    assert seed < 25
    return _observation(melons=25, money=100_000.0, day=20, hour=12)


def economic_policy_market(observation: dict[str, Any]) -> list[list[Any]]:
    """Return the market orders the vendored policy would send this turn."""
    return economic_policy.agent(observation)["market"]


def test_the_market_orders_come_from_the_route_not_the_live_policy() -> None:
    """A route's buys and its production are one plan, not two.

    The route buys the cow on the turn its units are walking to the pasture
    that will hold it. Substituting an independent market policy buys
    livestock the replayed units never place, so money leaves and nothing is
    produced: measured over one episode, 10,312 banked against 181,321. This
    is the single most important division in this design, and it took banking
    the wrong side of it to learn which way round it went.
    """
    recorded = [["SELL", "WHEAT", 40]]
    prototype = _prototype(market=recorded)

    action = RouteAgent([prototype]).act(_observation_backing(recorded))

    assert action["market"] == recorded
    assert action["market"] != economic_policy_market(_observation_backing(recorded))


def test_a_replayed_order_is_cut_down_to_what_our_own_state_backs() -> None:
    """A recorded order is a claim about the farm that recorded it.

    We reach the same turn with a different shed and a different bank, so the
    quantities have to be cut to what we can actually back -- and cut in
    order, since the engine credits a sale before it charges a later purchase.
    Five sacks at 100 pays for exactly one 400-coin cow, not the two recorded.
    """
    prototype = _prototype(market=[["SELL", "WHEAT", 40], ["BUY_ANIMAL", "COW", 2]])

    action = RouteAgent([prototype]).act(
        _observation(melons=1, money=0.0, shed={"WHEAT": 5})
    )

    assert action["market"] == [["SELL", "WHEAT", 5], ["BUY_ANIMAL", "COW", 1]]


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


def test_a_transfer_the_engine_would_refuse_is_not_counted_against_the_shed() -> None:
    """The shed model has to place goods where the engine would, not where asked.

    The engine spawns a farm's first hand on ``(5, 4)``, a shed-access tile in
    a quadrant nobody has bought yet, and ``_apply_unit_action`` refuses every
    shed transfer from a ``LOCKED`` tile. A model without that guard predicts
    the ``PICKUP`` emptying six sacks out of the shed, and clamps the sale of
    ten that follows down to four -- revenue given away for goods we hold,
    with nothing raised and nothing logged.
    """
    prototype = _prototype(market=[["SELL", "WHEAT", 10]])
    for action in prototype.actions:
        action["hands"] = [["PICKUP", "WHEAT", 6]]
    hired = list(prototype.signatures[0])
    hired[SIGNATURE_FIELDS.index("HANDS")] = 1.0 / HANDS_SCALE
    prototype.signatures[:] = [tuple(hired)] * len(prototype.signatures)

    observation = _observation(melons=1, hands=1, shed={"WHEAT": 10})

    assert observation["farms"][0]["hands"] == [[5, 4]]
    assert observation["farms"][0]["tiles"][4][5] == "LOCKED"
    assert RouteAgent([prototype]).act(observation)["market"] == [["SELL", "WHEAT", 10]]


def test_turn_zero_starts_a_new_episode_rather_than_continuing_the_last() -> None:
    """One process plays many episodes in a league evaluation.

    An agent that carried its incumbent route across a season boundary would
    replay the wrong opening and report one fallback count for two games.
    """
    agent = RouteAgent([_prototype(seed=1)])
    agent.act(_observation_unlike(seed=1))

    agent.act(_observation_matching(seed=1))

    assert agent.fallbacks == 0


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
