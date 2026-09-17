"""The plan a program carries, taken out for a round and put back afterwards.

A champion is a controller wrapped around a plan: 3,982 scripted actions, 41
routes, and the table that picks a route from the shops a map happens to have.
The plan is the part that decides what the agent does on the board, and it
ships as one line of base85'd, zlib'd JSON -- 94,490 characters, 73,306 tokens,
and unreadable to anything.

Which is why no round has ever changed it. Across champions 17 to 24 the blob
is byte-identical: eight promotions that edited the controller and left the
strategy exactly as it was harvested. It also explains what the gate keeps
reporting -- children draw with the champion because they inherit its tape, so
the two play the same game and the promotion bar of eight decided games is
unreachable. Self-play had the same hole: a pool of champions that all share
one plan is one plan playing itself.

So a round is handed the two apart. `split` writes the controller with a line
that loads `plan.json` beside it, and the plan as readable JSON the round can
open, grep and edit like any other file. `join` puts them back into the single
self-contained program every other part of the campaign expects -- the gate,
the archive, the pool, the validator and the submission all still see one file.

`join(split(source))` returns the original source byte for byte when the plan
is untouched, which is asserted rather than hoped: the encoding is
`separators=(",", ":")` at level 9, and that reproduces what the agent already
carries.
"""

import base64
import json
import re
import zlib
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kaggriculture.campaign import config

# The plan as a program carries it: a generated name, then the three calls.
# The name is captured so it can be put back; it differs between programs.
PACKED = re.compile(
    r"^(?P<name>\w+)\s*=\s*json\.loads\(\s*zlib\.decompress\(\s*"
    r"base64\.b85decode\(\s*'(?P<payload>[^']+)'\s*\)\s*\)\s*\)\s*$",
    re.MULTILINE,
)

# And as a round sees it: one import, in place of 94,490 characters.
LOADER = "{name}=__import__('{module}').DATA"
UNPACKED = re.compile(
    r"^(?P<name>\w+)=__import__\('(?P<module>[\w]+)'\)\.DATA\s*$",
    re.MULTILINE,
)

PLAN_FILE = "plan.json"
PLAN_MODULE = "_plan"
# The three lines that stand between the agent and its plan. They are a module
# rather than a path read because the agent is exec'd without a `__file__` --
# `kaggle_environments` compiles it and runs it in a bare dict -- while it does
# put the agent's own directory on `sys.path`, so an import from there resolves
# and the module it reaches has the `__file__` the agent does not.
PLAN_LOADER_SOURCE = """import json
import pathlib

DATA = json.loads(
    (pathlib.Path(__file__).with_name({file!r})).read_text(encoding="utf-8")
)
"""

# The engine's own enums, as types rather than as checks. `sim.hpp`'s `Op` and
# `MOp` are the vocabularies, and pydantic renders an enum into the schema as
# `enum: [...]` -- so a generator handed this can only spell a verb the engine
# has, which a validator could report only after the writing was done.
ItemName = StrEnum("ItemName", {name: name for name in config.ITEMS})
ProductName = StrEnum("ProductName", {name: name for name in config.PRODUCTS})
CropName = StrEnum("CropName", {name: name for name in config.CROPS})
AnimalName = StrEnum("AnimalName", {name: name for name in config.ANIMALS})
ShopName = StrEnum("ShopName", {name: name for name in config.SHOP_NAMES})

# Every unit op that is a verb on its own. `action.rs` gives `PICKUP`, `PLACE`
# and `PLANT` arguments and reads the rest as the verb alone -- and `PASS`
# shares its arm with every op the engine does not know, so an unknown verb is
# a pass rather than an error.
PLAIN_UNIT_OPS = [
    op for op in config.UNIT_OPS if op not in ("PICKUP", "PLACE", "PLANT")
]
PlainUnitOp = StrEnum("PlainUnitOp", {op: op for op in PLAIN_UNIT_OPS})

# A count. Not `gt=0`, though `action.rs` drops an order whose count is zero or
# less: the champion carries `["SELL", "STRAWBERRY", 0]` at step 449, the engine
# runs that plan without complaint, and a schema that refused it would refuse a
# legal plan. So the rule is said where a writer reads it rather than enforced
# where it would reject what already plays.
Count = Annotated[
    int,
    Field(description="How many. An order with a count of zero or less is dropped."),
]

# One command, shaped as the parser reads it.
UnitCommand = Annotated[
    tuple[PlainUnitOp]
    | tuple[Literal["PICKUP"], ItemName]
    | tuple[Literal["PICKUP"], ItemName, Count]
    | tuple[Literal["PLACE"], ItemName]
    | tuple[Literal["PLACE"], ItemName, Count]
    | tuple[Literal["PLANT"], CropName],
    Field(description="A verb the farmer or a hand performs, and its arguments."),
]

# And one order. `HIRE` and `BUY_LAND` are verbs alone; the four that move goods
# each take the names their own arm accepts. An empty order is legal and
# common: the engine drops it, and 283 of the champion's steps place one.
MarketCommand = Annotated[
    tuple[()]
    | tuple[Literal["HIRE"]]
    | tuple[Literal["BUY_LAND"]]
    | tuple[Literal["SELL"], ProductName, Count]
    | tuple[Literal["BUY_SEED"], CropName, Count]
    | tuple[Literal["BUY_PRODUCT"], Literal["WHEAT", "FERTILIZER"], Count]
    | tuple[Literal["BUY_ANIMAL"], AnimalName, Count],
    Field(description="An order to place this step, or nothing."),
]


class Action(BaseModel):
    """One step of the season: what the farmer does, the hands, and the market.

    Attributes:
        farmer: The farmer's command for this step. It always acts.
        hands: One command per hired hand, up to the twelve the champion
            reaches, and none at the start of a season.
        market: The orders placed this step.
    """

    model_config = ConfigDict(extra="forbid")

    farmer: UnitCommand = Field(
        description=(
            "The farmer's own command for this step. The farmer always acts, so "
            "this is never empty; `['PASS']` is how it does nothing."
        ),
        examples=[["PASS"], ["NORTH"], ["PLANT", "WHEAT"], ["PICKUP", "COW", 2]],
    )
    # `sim.hpp` keeps `MAX_UNITS` slots for the farmer and its hands, so a
    # farm can work thirty-nine. The first schema said twelve, which is
    # what this champion hires -- and would have refused a legal plan.
    hands: list[UnitCommand] = Field(
        max_length=config.MAX_UNITS - 1,
        description=(
            "One command for each hand hired so far, in the order they were "
            "hired: the first entry is the first hand. Empty at the start of a "
            "season, before anything has hired, and never longer than the number "
            "of hands the farm actually has -- a command addressed to a hand that "
            "does not exist is ignored."
        ),
        examples=[[], [["PASS"]], [["NORTH"], ["PICKUP", "COW"]]],
    )
    market: list[MarketCommand] = Field(
        description=(
            "The orders to place this step, in the order they are placed. Both "
            "players' orders resolve in per-unit lockstep, so position matters "
            "when both sides reach for the same goods. May be empty."
        ),
        examples=[[], [["HIRE"]], [["SELL", "WHEAT", 30], ["BUY_SEED", "MELON", 2]]],
    )


class Route(BaseModel):
    """One path, as the tile indices it walks in order.

    A list rather than the dict the program keeps, because strict structured
    output refuses an object whose keys are the writer's to choose, and a plan
    that cannot be a schema cannot be generated against one. The numbering is
    not a range -- the champion's routes are 0 to 12 and 100 to 128 -- so the
    id travels with the path rather than being its position.

    Attributes:
        id: The number `shops` refers to this route by.
        tiles: The tiles it walks, in order.
    """

    model_config = ConfigDict(extra="forbid")

    id: int = Field(
        description=(
            "The number `shops` refers to this route by. The numbering is not a "
            "range and does not have to be contiguous: the champion's routes are "
            "0 to 12 and 100 to 128."
        ),
        examples=[0, 101, 128],
    )
    tiles: list[int] = Field(
        description=(
            "The tiles this route walks, in order, as indices into the board. "
            "Repeats are meaningful -- standing still is walking the same tile "
            "twice -- and every route in the champion is the same length as the "
            "season it is walked over."
        ),
        examples=[[0, 1, 2, 3, 4]],
    )


class ShopRoute(BaseModel):
    """Which route to walk when the map unlocks these two shops.

    This is the table worth editing. It is 64 entries, and it is the whole of
    how the plan adapts to a map: change a route here and the agent plays a
    different season.

    Attributes:
        shops: The two shops the map has, by the engine's own names.
        route: The route to walk, which has to be one `routes` holds.
    """

    model_config = ConfigDict(extra="forbid")

    shops: tuple[ShopName, ShopName] = Field(
        description=(
            "The two shops this map unlocks, in the engine's own order. The pair "
            "is what a map is identified by, and both may be the same shop."
        ),
        examples=[["BAKERY", "BAKERY"], ["BAKERY", "YARN_STORE"]],
    )
    route: int = Field(
        description=(
            "Which route to walk on a map with those two shops. This has to be "
            "the `id` of a route the plan holds, or the agent has nowhere to walk."
        ),
        examples=[101, 112],
    )


class Plan(BaseModel):
    """The strategy a program plays: the season, the paths, and the lookup.

    Attributes:
        actions: The scripted season, step by step.
        routes: The paths those steps walk, as tile indices, by route number.
        shops: Which route to walk for the shops a map happens to unlock.
    """

    model_config = ConfigDict(extra="forbid")

    actions: list[Action] = Field(
        description=(
            "The season, scripted step by step and walked in order: entry N is "
            "what the farm does on step N. This is what the agent actually does "
            "on the board, and it is the largest part of a plan by far -- the "
            "champion's is 3,982 steps."
        ),
    )
    routes: list[Route] = Field(
        description=(
            "The paths the shop lookup chooses between, each with the number it "
            "is chosen by."
        ),
    )
    shops: list[ShopRoute] = Field(
        description=(
            "Which route to walk for each pair of shops a map can unlock. This is "
            "the whole of how the plan adapts to the map it is dealt: everything "
            "else is fixed, and this decides which fixed thing gets played. The "
            "champion carries all 64 ordered pairs of the eight shops."
        ),
    )

    @model_validator(mode="after")
    def _routes_exist(self) -> "Plan":
        """Every route the lookup names has to be a route that is there.

        The likeliest way to break this file is to point a shop pair at a route
        number that was never in it, and the agent would find that out by
        failing mid-season.
        """
        known = {route.id for route in self.routes}
        missing = sorted({entry.route for entry in self.shops} - known)
        if missing:
            have = ", ".join(str(one) for one in sorted(known)[:6])
            raise ValueError(
                f"shops point at routes {missing} that `routes` does not hold "
                f"(it has {have}, ...)"
            )
        return self


def validated(data: dict) -> dict:
    """The plan, checked, or a refusal that says what is wrong with it.

    Args:
        data: A plan as read from `plan.json`.

    Returns:
        The same plan, once it is known to be one.

    Raises:
        ValueError: It is not, with the first thing wrong named.
    """
    Plan.model_validate(data)
    return data


def _routes_as_list(routes: dict) -> list[dict]:
    """The program's dict of paths, as the list a schema can describe."""
    return [{"id": int(number), "tiles": tiles} for number, tiles in routes.items()]


def _routes_as_dict(routes: list) -> dict:
    """And back, as the program reads them."""
    return {str(route["id"]): route["tiles"] for route in routes}


def carries(source: str) -> bool:
    """Whether this program has a packed plan to take out.

    Args:
        source: A program's source.

    Returns:
        True when there is a plan; programs before champion_17 have none.
    """
    return PACKED.search(source) is not None


def split(source: str) -> tuple[str, dict]:
    """The controller with a loader in place of the plan, and the plan.

    Args:
        source: A self-contained program.

    Returns:
        The controller's source, and the plan as a dict.

    Raises:
        ValueError: The program carries no packed plan.
    """
    found = PACKED.search(source)
    if found is None:
        raise ValueError("this program carries no packed plan")
    packed = json.loads(zlib.decompress(base64.b85decode(found.group("payload"))))
    plan = {**packed, "routes": _routes_as_list(packed["routes"])}
    line = LOADER.format(name=found.group("name"), module=PLAN_MODULE)
    return source[: found.start()] + line + source[found.end() :], plan


def join(controller: str, plan: dict) -> str:
    """One self-contained program again, packed as a program carries one.

    Args:
        controller: A controller carrying the loader line.
        plan: The plan to pack back into it.

    Returns:
        The program, with the plan inline.

    Raises:
        ValueError: The controller has no loader line to put the plan back
            into, or the packed plan holds a quote that would break the line.
    """
    found = UNPACKED.search(controller)
    if found is None:
        raise ValueError("this controller has no plan loader to pack back into")
    carried = {**plan, "routes": _routes_as_dict(plan["routes"])}
    packed = base64.b85encode(
        zlib.compress(json.dumps(carried, separators=(",", ":")).encode(), 9)
    ).decode()
    if "'" in packed:
        raise ValueError("the packed plan holds a quote")
    line = (
        f"{found.group('name')}=json.loads(zlib.decompress("
        f"base64.b85decode('{packed}')))"
    )
    return controller[: found.start()] + line + controller[found.end() :]


def lay_out(source: str, box: Path, name: str = "child.py") -> None:
    """Write a program into a round's directory, plan alongside.

    A program with no plan is written as it is, so the older lineage still
    runs.

    Args:
        source: The self-contained program.
        box: The round's directory.
        name: What to call the program there.
    """
    if not carries(source):
        (box / name).write_text(source, encoding="utf-8")
        return
    controller, plan = split(source)
    (box / name).write_text(controller, encoding="utf-8")
    (box / PLAN_FILE).write_text(
        json.dumps(plan, indent=1, sort_keys=True), encoding="utf-8"
    )
    (box / f"{PLAN_MODULE}.py").write_text(
        PLAN_LOADER_SOURCE.format(file=PLAN_FILE), encoding="utf-8"
    )


def gather(box: Path, name: str = "child.py") -> str:
    """Read a round's program back as one self-contained file.

    Args:
        box: The round's directory.
        name: The program's name there.

    Returns:
        The program with its plan packed back in, or as written when the round
        was never given a plan to edit.

    """
    source = (box / name).read_text(encoding="utf-8")
    plan_file = box / PLAN_FILE
    if not plan_file.exists() or UNPACKED.search(source) is None:
        # No loader left: the round rewrote the program around its own data,
        # and what it wrote is what it meant. The plan file is a leftover.
        return source
    # Checked here, where the round can still be told it wrote nothing usable.
    # Unchecked, a plan with a verb the engine has no op for, or a shop pair
    # pointing at a route that is not there, is a program that forfeits every
    # game -- and the campaign would read that as a bad idea rather than a
    # broken file.
    return join(source, validated(json.loads(plan_file.read_text(encoding="utf-8"))))
