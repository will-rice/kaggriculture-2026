"""Build an opponent out of the public games: the strongest opening, as orders.

The gate has nothing left to learn from. Fourteen of champion_69's twenty-four
opponents are saturated and every one of them is ours; the twelve publics it
does not saturate it still beats at 0.90. We cannot play the agents above us on
the ladder -- they publish nothing -- but we hold 17,617 of their games.

The first pacer chased the averaged build order and could not fire: it was told
to hold 1.2 quadrants on day three with a bank of 200, against a land price of
1,000. That target was a blend of two openings and belonged to neither.

This one takes the orders the strongest opening actually sends, mined from the
corpus and whole: seven hires on day zero, five cows, land on day three. Those
are things a policy can do, and the cash to do them is what the same games show
them generating -- banks of 626 on day one and 996 on day two, against 119 and
163 for the rest of the top twelve.

Aggregate over a group and naming no agent, the footing the build order and the
report already stand on.
"""

import pathlib

from kaggriculture.campaign import dataset

OUT = pathlib.Path(
    "/tmp/claude-1000/-home-will-projects-kaggriculture-2026/"
    "521a4f31-d72e-47e3-adbb-c5944812e9ba/scratchpad/pacer2.py"
)
BASE = pathlib.Path("run/campaign/champions/champion_69.py")

PACER = '''

# --- the pacer's addition ------------------------------------------------
# The champion above, unchanged. Below: the orders the ladder's strongest
# opening sends, mined from {games:,} public games and applied as a floor
# rather than as an addition -- top the champion up to what they send, never
# stack on top of it, or the hires compound into something nobody plays.
# `_BASE_AGENT` is bound *before* the wrapper is defined, and that ordering is
# the whole thing. The runner takes `[v for v in env.values() if callable(v)][-1]`
# -- the last callable by insertion order -- and rebinding `agent` at the end
# does not move `agent` in that order. Bound after the def, `_BASE_AGENT` is
# itself the last callable inserted, so the runner picks the champion and the
# wrapper never runs. It did not, twice, and both pacer results were void.
_BASE_AGENT = agent
_HIRES = {hires!r}
_PENS = {pens!r}
_LAND_DAY = {land_day!r}
_LAND_PRICE = (1000.0, 2000.0, 4000.0)


def _pacer_agent(observation, configuration=None):
    """The champion's move, topped up to the strongest opening's orders."""
    action = _BASE_AGENT(observation, configuration)
    try:
        me = observation["player"]
        farm = observation["farms"][me]
        day = int(observation["day"])
        money = float(farm["money"])
        market = list(action.get("market") or [])
        tiles = [t for row in farm["tiles"] for t in row if isinstance(t, dict)]
        # Hires are per day and reset, so the floor is what has gone out today
        # plus what this move already asks for.
        want = _HIRES.get(day, 0)
        have = int(farm.get("hires_today") or 0)
        have += sum(1 for order in market if order and order[0] == "HIRE")
        market += [["HIRE"]] * max(0, want - have)
        # Animals are cumulative, so the floor is what stands on the farm.
        pens = len([t for t in tiles if t.get("animal")])
        short = _PENS.get(day, 0) - pens
        if short > 0:
            market.append(["BUY_ANIMAL", "COW", short])
        # And the land, on the day they take it, if it can be paid for.
        quadrants = len(farm.get("unlocked_quadrants") or [])
        if day >= _LAND_DAY and 1 <= quadrants <= 2:
            price = _LAND_PRICE[quadrants - 1]
            if money >= price and not any(
                order and order[0] == "BUY_LAND" for order in market
            ):
                market.append(["BUY_LAND"])
        action = dict(action)
        action["market"] = market
    except Exception:
        return action
    return action


agent = _pacer_agent
'''


def main() -> None:
    """Mine the opening and write the champion with it bolted on."""
    sent = dataset.opening_orders()
    hires = {
        order.day: order.count for order in sent if order.verb == "HIRE"
    }
    land = next(
        (order.day for order in sent if order.verb == "BUY_LAND"), 3
    )
    order = dataset.build_order(teams=dataset.openings()[0].teams)
    pens = {
        day: round(value)
        for day, value in zip(dataset.MARKS, order["pens"], strict=False)
        if day <= 4
    }
    games = dataset.counts()["episodes"]
    print(f"hires by day : {hires}")
    print(f"pens by day  : {pens}")
    print(f"land on day  : {land}")
    OUT.write_text(
        BASE.read_text(encoding="utf-8")
        + PACER.format(games=games, hires=hires, pens=pens, land_day=land),
        encoding="utf-8",
    )
    print(f"\\nwrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
