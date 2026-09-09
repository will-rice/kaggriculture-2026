"""Write a pacer: champion_67, forced to hit the corpus's build order.

The gate has stopped measuring anything. Fourteen of twenty-four opponents are
saturated and every one of them is ours -- a candidate beats the whole lineage
almost always, so nothing about it is learned. The only unsaturated opponents
are the twelve harvested publics, and we beat those at 0.75 to 0.94 while
sitting 1,557 places below the top of the ladder.

We cannot play the agents above us; they publish nothing. But the corpus holds
sixteen thousand of their games, and champion_67 meets 3 of 43 settled claims
about how they play. So build an opponent out of what they do rather than what
they are: the same champion, with the capacity purchases the corpus says the
top twelve make, on the days it says they make them.

Written as a wrapper rather than a new agent on purpose. A fresh pacer would
differ from champion_67 in every respect and its result would attribute to
nothing; this differs in exactly one, so if it wins, the build order caused it.

Aggregate statistics only. No opponent's source is read and no agent is named,
which is the same footing the build order and the report already stand on.
"""

import pathlib

from kaggriculture.campaign import dataset

OUT = pathlib.Path(
    "/tmp/claude-1000/-home-will-projects-kaggriculture-2026/"
    "521a4f31-d72e-47e3-adbb-c5944812e9ba/scratchpad/pacer.py"
)
BASE = pathlib.Path("run/campaign/champions/champion_67.py")

PACER = '''
# --- the pacer's addition ------------------------------------------------
# champion_67 above, unchanged. Below: the capacity the corpus says the top
# twelve of the ladder hold, on the days it says they hold it. The champion
# meets 3 of 43 settled claims; these are the two it misses that a single
# market order can close.
_QUADRANTS = {quadrants!r}
_PENS = {pens!r}
_LAND_PRICE = (1000.0, 2000.0, 4000.0)
# What the corpus's top twelve hold, read at the days it tabulates and held
# flat between them: a target is a floor to reach, not a curve to trace.
def _target(table, day):
    """The top twelve's holding on `day`, from the nearest tabulated day at or below."""
    best = None
    for mark, value in sorted(table.items()):
        if mark <= day:
            best = value
    return best if best is not None else 0.0


def _pacer_agent(observation, configuration=None):
    """The champion's own move, plus the purchases the build order implies."""
    action = _BASE_AGENT(observation, configuration)
    try:
        me = observation["player"]
        farm = observation["farms"][me]
        day = int(observation["day"])
        money = float(farm["money"])
        tiles = [t for row in farm["tiles"] for t in row if isinstance(t, dict)]
        quadrants = len(farm.get("unlocked_quadrants") or [])
        pens = len([t for t in tiles if t.get("animal")])
        market = list(action.get("market") or [])
        # Land first: a quadrant is the capacity everything else needs, and it
        # is the claim the champion misses earliest -- 1.0 against 1.2 on day
        # three, 2.0 against 2.5 on day ten.
        if quadrants < _target(_QUADRANTS, day) and quadrants <= 3:
            price = _LAND_PRICE[min(quadrants - 1, 2)]
            if money >= price:
                market.append(["BUY_LAND"])
                money -= price
        if pens < _target(_PENS, day):
            market.append(["BUY_ANIMAL", "CHICKEN", 1])
        action = dict(action)
        action["market"] = market
    except Exception:
        return action
    return action


_BASE_AGENT = agent
agent = _pacer_agent
'''


def main() -> None:
    """Read the build order and write the champion with a pacer bolted on."""
    order = dataset.build_order(best=dataset.BEST)
    quadrants = {
        day: value
        for day, value in zip(dataset.MARKS, order["quadrants"], strict=False)
    }
    pens = {day: value for day, value in zip(dataset.MARKS, order["pens"], strict=False)}
    print("targets the pacer holds itself to:")
    print(f"  quadrants {({k: round(v, 1) for k, v in quadrants.items()})}")
    print(f"  pens      {({k: round(v, 1) for k, v in pens.items()})}")
    OUT.write_text(
        BASE.read_text(encoding="utf-8")
        + PACER.format(quadrants=quadrants, pens=pens),
        encoding="utf-8",
    )
    print(f"\nwrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
