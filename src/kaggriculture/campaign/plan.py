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
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

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


class Action(BaseModel):
    """One step of the season: what the farmer does, the hands, and the market.

    A command is a verb and its arguments -- `["PASS"]`, `["PICKUP", "COW"]`,
    `["BUY_PRODUCT", "WHEAT", 13]` -- and the vocabularies are the engine's own
    enums, so a verb this rejects is one `sim.hpp` would not have understood
    either.

    Attributes:
        farmer: The farmer's command for this step.
        hands: One command per hired hand, up to twelve, and none at the start.
        market: The orders placed this step. An empty command is allowed here
            and occurs 283 times in the champion's own plan; it is a step that
            places nothing.
    """

    model_config = ConfigDict(extra="forbid")

    farmer: list[str | int]
    hands: list[list[str | int]]
    market: list[list[str | int]]

    @field_validator("farmer")
    @classmethod
    def _farmer_acts(cls, command: list[str | int]) -> list[str | int]:
        """The farmer acts every step, and the verb is one the engine has."""
        return _command(command, config.UNIT_OPS, "farmer", empty=False)

    @field_validator("hands")
    @classmethod
    def _hands_act(cls, commands: list[list[str | int]]) -> list[list[str | int]]:
        """Every hand that exists acts, with the same vocabulary."""
        for command in commands:
            _command(command, config.UNIT_OPS, "a hand", empty=False)
        return commands

    @field_validator("market")
    @classmethod
    def _orders_are_orders(
        cls, commands: list[list[str | int]]
    ) -> list[list[str | int]]:
        """Market orders, or nothing at all."""
        for command in commands:
            _command(command, config.MARKET_OPS, "a market order", empty=True)
        return commands


def _command(
    command: list[str | int], vocabulary: list[str], whose: str, empty: bool
) -> list[str | int]:
    """Check one verb and its arguments.

    Args:
        command: The command to check.
        vocabulary: The verbs the engine accepts here.
        whose: What to call it if it is wrong.
        empty: Whether an empty command means "do nothing" or is a mistake.

    Returns:
        The command, unchanged.

    Raises:
        ValueError: The command is empty where it may not be, too long, or
            names a verb the engine does not have.
    """
    if not command:
        if empty:
            return command
        raise ValueError(f"{whose} has no verb")
    verb = command[0]
    if verb not in vocabulary:
        raise ValueError(f"{whose} says {verb!r}, which the engine has no op for")
    if len(command) > 3:
        raise ValueError(f"{whose} has {len(command)} parts; the engine reads three")
    return command


class ShopRoute(BaseModel):
    """Which route to walk when the map unlocks these two shops.

    This is the table worth editing. It is 64 lines, and it is the whole of
    how the plan adapts to a map: change a route here and the agent plays a
    different season.

    Attributes:
        shops: The two shops the map has, by the engine's own names.
        route: The route to walk, which has to be one `routes` holds.
    """

    model_config = ConfigDict(extra="forbid")

    shops: list[str]
    route: int

    @field_validator("shops")
    @classmethod
    def _two_known_shops(cls, shops: list[str]) -> list[str]:
        """Two of them, both names the engine unlocks."""
        if len(shops) != 2:
            raise ValueError(f"a map unlocks two shops, not {len(shops)}")
        for shop in shops:
            if shop not in config.SHOP_NAMES:
                raise ValueError(f"{shop!r} is not a shop the engine has")
        return shops


class Plan(BaseModel):
    """The strategy a program plays: the season, the paths, and the lookup.

    Attributes:
        actions: The scripted season, step by step.
        routes: The paths those steps walk, as tile indices, by route number.
        shops: Which route to walk for the shops a map happens to unlock.
    """

    model_config = ConfigDict(extra="forbid")

    actions: list[Action]
    routes: dict[str, list[int]]
    shops: list[ShopRoute]

    @model_validator(mode="after")
    def _routes_exist(self) -> "Plan":
        """Every route the lookup names has to be a route that is there.

        The likeliest way to break this file is to point a shop pair at a route
        number that was never in it, and the agent would find that out by
        failing mid-season.
        """
        missing = sorted(
            {entry.route for entry in self.shops if str(entry.route) not in self.routes}
        )
        if missing:
            have = ", ".join(sorted(self.routes, key=int)[:6])
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
    plan = json.loads(zlib.decompress(base64.b85decode(found.group("payload"))))
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
    packed = base64.b85encode(
        zlib.compress(json.dumps(plan, separators=(",", ":")).encode(), 9)
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
