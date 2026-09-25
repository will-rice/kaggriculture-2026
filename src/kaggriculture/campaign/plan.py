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

Commands are models rather than the positional lists a program carries, because
a fixed-length heterogeneous array is not something every provider's schema
subset can say: it was eighty violations of OpenAI's, and nothing else in the
schema broke a rule. As `{"verb": "PLANT", "crop": "WHEAT"}` the same thing is
said in a shape they all take, and said more precisely -- one model per verb,
each carrying only the arguments that verb accepts. The program still reads
`["PLANT", "WHEAT"]`; that form is a serialization, converted at the boundary.

For OpenAI's strict mode, hand the model to
`openai.lib._pydantic.to_strict_json_schema`, which closes the objects and
inlines the `$ref`s that carry descriptions. DeepSeek and GLM take
`model_json_schema()` as it is -- both were asked.

`join(split(source))` returns the same plan and the same controller. It also
happens to return the same bytes for the champions in play -- the models are
declared in the order those programs write their keys -- but nothing depends on
that. A round's verdict compares `gather` before against `gather` after, and
both sides are packed by this module in the same process, so key order and
separators cancel out.
"""

import ast
import base64
import json
import re
import zlib
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from kaggriculture.campaign import config

# `sim.hpp`: `constexpr int MAX_UNITS = 40; // farmer + hands`. So a farm can
# work thirty-nine hands beside its farmer, whatever any one champion happens
# to hire.
MAX_UNITS = 40

# `kaggriculture.json` sets `"episodeSteps": 720`, and the interpreter fires
# DONE at `step >= cfg.episodeSteps - 2`, on the step whose actions it just
# read. So step 718 is the last one a unit acts on and a season is 719 acting
# steps -- which is exactly how long every route in a plan is.
SEASON = 719

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
# The second payload a program carries, and the second reason a round cannot
# read its own program: `_V92_P_BLOB` is the price-prototype library, 440,834
# bytes of base85 on one line of a controller whose median line is 43 bytes.
# Any `diff` or `rg` that crosses it returns the lot -- 20 of 41 rounds had one
# command return over 100 KB, the largest 1,021 KB, against a budget of about
# 800 KB -- and it is data no round has any reason to read. It goes beside the
# program the way the plan does, and comes back the way the plan does.
BLOB = re.compile(
    r"^(?P<name>_V92_P_BLOB)\s*=\s*'(?P<payload>[^']+)'\s*$", re.MULTILINE
)
BLOB_FILE = "prototypes.b85"
BLOB_MODULE = "_prototypes"
BLOB_UNPACKED = re.compile(
    r"^(?P<name>_V92_P_BLOB)=__import__\('_prototypes'\)\.DATA\s*$", re.MULTILINE
)
BLOB_LOADER_SOURCE = """import pathlib

DATA = (pathlib.Path(__file__).with_name({file!r})).read_text(encoding="utf-8")
"""
# The third payload a program carries, and 12,939 of the 13,626 characters of
# champion_41's header: the whole Apache-2.0 licence body, pasted into the file
# as comments by the published kernel this lineage derives from. Two hundred
# lines before the module docstring, re-read on every round, about 3,200
# tokens of it, and not one line about this program -- so a round reading the
# top of its own file learns nothing and pays for the privilege.
#
# The licence's terms are met without it. Section 4 asks that a recipient be
# given a copy of the licence and that the copyright and attribution notices
# be retained in the source: `harness.package` ships `LICENSE` -- the same two
# hundred lines -- beside `main.py` in every archive, and the notices proper
# are the twenty-three lines above the body, which stay where they are.
#
# Shelving it rather than asking a round to leave it alone settles both halves
# at once: the round cannot spend tokens on it and cannot strip it. The
# obligation stops depending on what a model decides to rewrite.
NOTICE = re.compile(
    r"^#[^\n]*Apache License[^\n]*\n(?:#[^\n]*\n)*?#[^\n]*"
    r"limitations under the License\.[^\n]*\n",
    re.MULTILINE,
)
NOTICE_FILE = "licence.txt"
NOTICE_MARK = (
    "# The Apache-2.0 licence body sits in `licence.txt` and is put back into"
    " this\n# file when the campaign packs it. Leave this line where it is.\n"
)
# What the farm does on a step no season names: nothing. A written season may
# stop short of the year, and this is what the rest of it is.
IDLE: dict = {"farmer": ["PASS"], "hands": [], "market": []}
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
UnitOp = StrEnum("UnitOp", {op: op for op in config.UNIT_OPS})
ItemName = StrEnum("ItemName", {name: name for name in config.ITEMS})
ShopName = StrEnum("ShopName", {name: name for name in config.SHOP_NAMES})

# A command is `[verb]`, `[verb, name]` or `[verb, name, count]` -- the shape
# `action.rs` reads, and the shape the program carries. Said as a tuple per
# arity, because that is the only way a schema constrains a *position*: given a
# bounded list of tokens instead, GLM produced `["FERTILIZER", "HARVEST"]` and
# an order beginning with an integer, both of which validated and neither of
# which the engine would act on. Given these, the same model wrote
# `["PLANT", "WHEAT"]` first time.
#
# The cost is that a tuple becomes `prefixItems`, which OpenAI's strict mode
# refuses -- so OpenAI is not a generator for this plan. DeepSeek and GLM take
# it, and they are the ones with the output length a whole plan needs.
CropName = StrEnum("CropName", {name: name for name in config.CROPS})
ProductName = StrEnum("ProductName", {name: name for name in config.PRODUCTS})
AnimalName = StrEnum("AnimalName", {name: name for name in config.ANIMALS})

UnitCommand = Annotated[
    tuple[UnitOp]
    | tuple[Literal["PICKUP", "PLACE"], ItemName]
    | tuple[Literal["PICKUP", "PLACE"], ItemName, int]
    | tuple[Literal["PLANT"], CropName],
    Field(
        description=(
            "One command: the verb first, then what it acts on. Most verbs are "
            "the verb alone. `PICKUP` and `PLACE` take an item and may take a "
            "count; `PLANT` takes one of the five crops and no count."
        ),
        examples=[["PASS"], ["NORTH"], ["PLANT", "WHEAT"], ["PICKUP", "COW", 2]],
    ),
]

MarketCommand = Annotated[
    tuple[()]
    | tuple[Literal["HIRE", "BUY_LAND"]]
    | tuple[Literal["SELL"], ProductName, int]
    | tuple[Literal["BUY_SEED"], CropName, int]
    | tuple[Literal["BUY_PRODUCT"], Literal["WHEAT", "FERTILIZER"], int]
    | tuple[Literal["BUY_ANIMAL"], AnimalName, int],
    Field(
        description=(
            "One order: the verb, what it moves, and how many. `HIRE` and "
            "`BUY_LAND` stand alone. Each moving verb takes the names its own "
            "arm accepts, and an order whose count is zero or less is dropped. "
            "The empty list places nothing, which is legal and common."
        ),
        examples=[[], ["HIRE"], ["SELL", "WHEAT", 30], ["BUY_ANIMAL", "COW", 2]],
    ),
]


class Action(BaseModel):
    """One step of the season: what the farmer does, the hands, and the market."""

    model_config = ConfigDict(extra="forbid")

    farmer: UnitCommand = Field(
        description=(
            "The farmer's own command for this step. The farmer always acts, so "
            "this is never empty; `['PASS']` is how it does nothing."
        ),
    )
    hands: list[UnitCommand] = Field(
        max_length=MAX_UNITS - 1,
        description=(
            "One command for each hand hired so far, in the order they were "
            "hired: the first entry is the first hand. Empty at the start of a "
            "season, before anything has hired, and never longer than the number "
            "of hands the farm actually has -- a command addressed to a hand that "
            "does not exist is ignored. A farm can work at most "
            f"{MAX_UNITS - 1}, which is every unit slot the engine keeps "
            "beside the farmer."
        ),
    )
    market: list[MarketCommand] = Field(
        description=(
            "The orders to place this step, in the order they are placed. Both "
            "players' orders resolve in per-unit lockstep, so position matters "
            "when both sides reach for the same goods. May be empty."
        ),
    )


# The program keys its routes with `int(k)`, so a key that is not a number is a
# crash before the first move. A generator reaches for one: asked for a plan,
# GLM wrote a route called "season" beside route "100".
RouteId = Annotated[
    str,
    Field(
        pattern=r"^(0|[1-9][0-9]*)$",
        description="The route's number, written as a string of digits.",
    ),
]

Season = Annotated[
    list[Annotated[int, Field(ge=0)]],
    Field(
        min_length=SEASON,
        max_length=SEASON,
        description=(
            f"One index into `actions` for each of the {SEASON} steps a "
            "season has. Shorter and the farm stands idle for the rest of the "
            "year; longer and the tail is never reached."
        ),
    ),
]


class ShopRoute(BaseModel):
    """Which route to walk when the map unlocks these two shops.

    This is the table worth editing. It is 64 entries, and it is the whole of
    how the plan adapts to a map: change a route here and the agent plays a
    different season.
    """

    model_config = ConfigDict(extra="forbid")

    # Declared in the order a program writes them, so an untouched plan packs
    # back into the same bytes and diffs between champions stay legible.
    route: int = Field(
        description=(
            "Which route to walk on a map with these two shops. This has to be "
            "the `id` of a route the plan holds, or the agent has nowhere to walk."
        ),
        examples=[101, 112],
    )
    shops: list[ShopName] = Field(
        min_length=2,
        max_length=2,
        description=(
            "The two shops this map unlocks, in the engine's own order. Both may "
            "be the same shop, and the champion carries all sixty-four pairs."
        ),
        examples=[["BAKERY", "BAKERY"], ["BAKERY", "YARN_STORE"]],
    )


# The controller's own lines, which `join` regenerates from the plan. Each is a
# single top-level assignment of a literal, which is what makes projecting into
# them safe: nothing about the controller's code has to change.
CARRIED_SETTINGS = re.compile(r"^_SETTINGS\s*=\s*(\{[^\n]*\})$", re.MULTILINE)
CARRIED_OPENING = re.compile(r"^_R42_OPENING\s*=\s*(\[[^\n]*\])$", re.MULTILINE)
CARRIED_DEFAULTS = re.compile(
    r"^DEFAULT_SETTINGS\s*=\s*(\{.*?^\})$", re.MULTILINE | re.DOTALL
)
CARRIED_YARN = re.compile(r"^_R110_OLD_SHOPS\s*=\s*(\{[^\n]*\})$", re.MULTILINE)


class Settings(BaseModel):
    """Which of the chassis's reactive layers the program is built with.

    The chassis replays a season and wraps it in layers that each watch the
    board, and every one is switchable. They are strategy rather than
    mechanism: whether to sell ahead of the opponent, whether to liquidate at
    the end, whether to fund a block's purchases before it makes them.

    Described from the chassis's own table, because `dead_stock` and
    `room_guard` say nothing on their own.

    All 512 combinations were played on 2026-09-18. The champion's ranked fifth
    and the best in the whole space did not survive a wider block -- it lost by
    37 a game over 400 -- so these are a settled question rather than a place to
    look for gains.
    """

    model_config = ConfigDict(extra="forbid")

    hand_align: bool = Field(
        description="Pad or truncate the hand commands to the hands actually hired."
    )
    weed_repair: bool = Field(
        description="DIG a weed blocking a PLANT or BUILD, then replay the step."
    )
    sell_lead: bool = Field(description="Sell the next step's lots one step early.")
    front_run: bool = Field(
        description=(
            "Sell before the opponent's own scheduled sale, taking the price "
            "first. Needs their plan to predict from."
        )
    )
    budget_guard: bool = Field(
        description="Fund each block's purchases before the block makes them."
    )
    room_guard: bool = Field(
        description="Keep the shed at or under 99 at the last hour of the day."
    )
    clamp_sells: bool = Field(
        description="Trim a SELL down to what the shed is projected to hold."
    )
    dead_stock: bool = Field(
        description="Sell stock the rest of the season is never going to sell."
    )
    terminal_liquidation: bool = Field(
        description="On the last step, sell the whole projected shed."
    )

    block_turns: int = Field(
        ge=1,
        le=SEASON,
        description=(
            "How many steps ahead the budget guard funds. The season runs in "
            "blocks of this many, and each block's purchases are paid for "
            "before it makes them."
        ),
        examples=[72],
    )
    min_sell_price: int = Field(
        ge=1,
        description=(
            "Refuse to sell anything priced below this. The engine floors a "
            "price at $1, so 2 means never sell into the floor."
        ),
        examples=[2],
    )


class Plan(BaseModel):
    """The strategy a program plays: the season, the paths, and the lookup.

    Attributes:
        actions: The pool of distinct steps a route can cite.
        routes: One season each, as indices into `actions`, by route number.
        shops: Which route to walk for the shops a map happens to unlock.
    """

    model_config = ConfigDict(extra="forbid")

    actions: list[Action] = Field(
        min_length=1,
        description=(
            "Every distinct step any route can play, in no particular order. A "
            "step is not owned by the position it sits at here: routes cite "
            "entries by index, and two routes that do the same thing on some "
            "step cite the same entry. So this is a pool, and it is smaller "
            "than the seasons it spells out -- the champion's 41 routes are "
            f"{SEASON} steps each, {41 * SEASON:,} step slots in "
            "all, drawn from 3,982 entries here."
        ),
    )
    routes: dict[RouteId, Season] = Field(
        min_length=1,
        description=(
            "One whole season per route, keyed by the number the shop lookup "
            "chooses it by. A season is not a path across the board: it is "
            f"{SEASON} indices into `actions`, one per step, so entry N "
            "is which pooled step the farm plays on step N. Repeats are normal "
            "and are what makes the pool small -- a step the farm plays forty "
            "times is one entry in `actions` cited forty times here. The "
            "numbering is not a range and need not be contiguous: the "
            "champion's are 0 to 12 and 100 to 128."
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

    settings: Settings = Field(
        description=(
            "Which of the chassis's reactive layers to switch on. These live in "
            "the program's own source as `_SETTINGS`, and are written back there "
            "when the plan is packed, so changing one here changes the agent."
        ),
    )

    @model_validator(mode="after")
    def _steps_exist(self) -> "Plan":
        """Every index a season cites has to name a step the pool holds.

        An index past the end is the one way a plan can be internally
        well-formed and still unplayable: the program subscripts `actions` with
        it while building the tape and dies before the first move, which the
        scoreboard shows as a forfeit rather than as an error.
        """
        last = len(self.actions)
        over = {
            f"route {at} cites {max(past)}"
            for at, season in self.routes.items()
            if (past := [step for step in season if step >= last])
        }
        if over:
            raise ValueError(
                f"seasons cite steps `actions` does not hold -- it has "
                f"{last}, so the last index is {last - 1}: "
                f"{', '.join(sorted(over))}"
            )
        return self

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


class Stretch(BaseModel):
    """One step, and how many steps in a row the farm plays it."""

    model_config = ConfigDict(extra="forbid")

    step: Action = Field(
        description="What the farm does on each step of this stretch.",
    )
    steps: int = Field(
        ge=1,
        le=SEASON,
        description=(
            "How many steps in a row to play it. One means this step alone. "
            "Use a longer stretch for a rhythm the farm holds -- watering the "
            "same ground, feeding the same animals -- and one for a step that "
            "is its own decision."
        ),
        examples=[1, 24],
    )


Stretches = Annotated[
    list[Stretch],
    Field(
        min_length=1,
        max_length=SEASON,
        description=(
            f"A season, as stretches in the order they are played. The season "
            f"is {SEASON} steps long; the counts may add up to less "
            f"than that, and the farm then does nothing for the rest of the "
            f"year, but they may not add up to more, because a step past the "
            f"end is never played. There is no minimum number of stretches -- "
            f"one step held for the whole season is legal, and so is "
            f"{SEASON} stretches of one."
        ),
    ),
]


class Written(BaseModel):
    """A plan as it is written, before it is packed into one.

    The difference from `Plan` is that a season here is the steps themselves,
    in stretches, rather than indices into a pool. Two of our own constraints
    had to go for that to be writable. Indices, because no schema can say an
    index points at anything in particular, so a model filled the array by
    counting. And the fixed length of 719, because a model that has worked out
    a good year still has to reach 719 entries, and `PASS` is the cheapest
    thing to put in the rest -- which is what three models did.

    Attributes:
        routes: One written season per route number.
        shops: Which route to walk for the shops a map happens to unlock.
    """

    model_config = ConfigDict(extra="forbid")

    routes: dict[RouteId, Stretches] = Field(
        min_length=1,
        description=(
            "A season for each route, keyed by the number the shop lookup "
            "chooses it by. Writing one is a whole plan: a route the program "
            "asks for and this does not name plays the lowest-numbered season "
            "here, so the farm always has something to do."
        ),
    )
    shops: list[ShopRoute] = Field(
        description=(
            "Which route to walk for each pair of shops a map can unlock. Only "
            "pairs worth treating differently need an entry -- anything not "
            "named here plays the lowest-numbered season."
        ),
    )

    @model_validator(mode="after")
    def _seasons_fit(self) -> "Written":
        """No season may run past the end of the year.

        A season that stops short is fine and means the farm is idle for the
        rest of it -- that is what an unstated step means, and it is what both
        models wrote out longhand when they had nothing more to say. One that
        runs long is a different thing: the steps past the end cannot be played
        at all, so the writer has lost count of the year rather than finished
        early with it.

        Requiring an exact total was the same mistake as requiring 719 entries.
        A model wrote five stretches covering 699 steps, twenty short, and the
        whole strategy was refused over the arithmetic.
        """
        over = {
            f"route {route} covers {total}"
            for route, stretches in self.routes.items()
            if (total := sum(stretch.steps for stretch in stretches)) > SEASON
        }
        if over:
            raise ValueError(
                f"a season is {SEASON} steps and these run past the end "
                f"of it: {', '.join(sorted(over))}"
            )
        return self


def fold(written: Written, over: Plan) -> Plan:
    """Pack written seasons into the shape a program expects.

    A stretch becomes as many citations as it is long, and steps that are the
    same step become one pooled entry however many stretches reach for it.
    Every route `over` holds gets a season, so the program can ask for any of
    them: the written one where there is one, and the lowest-numbered written
    season otherwise.

    Args:
        written: The seasons as they were written.
        over: The plan being replaced, which says what routes are asked for.

    Returns:
        A plan in the program's own form.
    """
    pool: list[dict] = []
    at_index: dict[str, int] = {}
    seasons: dict[str, list[int]] = {}
    for route, stretches in written.routes.items():
        season: list[int] = []
        for stretch in stretches:
            body = stretch.step.model_dump(mode="json")
            key = json.dumps(body, sort_keys=True, separators=(",", ":"))
            if key not in at_index:
                at_index[key] = len(pool)
                pool.append(body)
            # A stretch held for twenty steps is one pooled entry cited twenty
            # times, which is what the pool is for.
            season.extend([at_index[key]] * stretch.steps)
        # A season that stopped short is idle for the rest of the year. The
        # program indexes a season by step number and every step has to be
        # there, so the idleness is written out rather than left implied.
        if len(season) < SEASON:
            key = json.dumps(IDLE, sort_keys=True, separators=(",", ":"))
            if key not in at_index:
                at_index[key] = len(pool)
                pool.append(IDLE)
            season.extend([at_index[key]] * (SEASON - len(season)))
        seasons[route] = season

    spare = seasons[min(seasons, key=int)]
    routes = {route: seasons.get(route, spare) for route in over.routes}
    named = {entry.route for entry in written.shops}
    shops = list(written.shops) + [
        entry for entry in over.shops if entry.route not in named
    ]
    return Plan(
        actions=pool,
        # A writer supplies seasons, not a chassis, so the layers stay as the
        # plan being replaced had them.
        settings=over.settings,
        routes=routes | seasons,
        shops=[entry for entry in shops if str(entry.route) in routes | seasons],
    )


def unfold(plan: Plan) -> Written:
    """A plan's seasons written out, which is what a generator is given to beat.

    Consecutive steps that are the same step collapse into one stretch, so a
    season that holds a rhythm comes back as few entries and one that decides
    every step comes back as many. The champion decides nearly every step, so
    it round-trips at close to one stretch per step.

    Args:
        plan: A plan in the program's own form.

    Returns:
        The same seasons, as stretches.
    """
    routes: dict[str, list[Stretch]] = {}
    for route, season in plan.routes.items():
        stretches: list[Stretch] = []
        for step in season:
            playing = plan.actions[step]
            if stretches and stretches[-1].step == playing:
                stretches[-1].steps += 1
            else:
                stretches.append(Stretch(step=playing, steps=1))
        routes[route] = stretches
    return Written(routes=routes, shops=plan.shops)


def described(before: dict, after: dict) -> str:
    """What changed between two plans, in the terms a round would change them.

    A round is told what the rounds before it scored and never what they did.
    The edits this lineage has kept were a rule across the pool -- every
    `SELL WHEAT` tripled, 395 orders at once -- and two orders added to one step
    played forty times; a score alone carries neither forward.

    Args:
        before: The plan that was, as `split` returns one.
        after: The plan that replaced it.

    Returns:
        One line saying what moved, or that nothing did.
    """
    said = []

    was = dict(enumerate(before["actions"]))
    now = dict(enumerate(after["actions"]))
    moved = [at for at in was if at in now and was[at] != now[at]]
    if len(now) != len(was):
        said.append(f"the pool went from {len(was):,} steps to {len(now):,}")
    if moved:
        # How often the changed steps are actually played, which is the size of
        # the edit rather than the size of the diff.
        cited = sum(
            season.count(at) for season in after["routes"].values() for at in moved
        )
        label = _one_rule(was, now, moved) or (
            f"{len(moved)} pooled step{'' if len(moved) == 1 else 's'}"
        )
        said.append(f"{label} changed, deciding {cited:,} step-slots")

    repointed = [
        at
        for at in before["routes"]
        if at in after["routes"] and before["routes"][at] != after["routes"][at]
    ]
    if repointed:
        said.append(
            f"{len(repointed)} season{'' if len(repointed) == 1 else 's'} "
            f"repointed: {', '.join(repointed[:4])}"
        )

    theirs = {tuple(one["shops"]): one["route"] for one in before["shops"]}
    ours = {tuple(one["shops"]): one["route"] for one in after["shops"]}
    lookup = [pair for pair in theirs if ours.get(pair) != theirs[pair]]
    if lookup:
        shown = ", ".join(f"{' + '.join(pair)} -> {ours[pair]}" for pair in lookup[:2])
        said.append(
            f"{len(lookup)} shop pair{'' if len(lookup) == 1 else 's'} "
            f"repointed: {shown}"
        )

    switches = [
        name
        for name, value in before.get("settings", {}).items()
        if after.get("settings", {}).get(name) != value
    ]
    if switches:
        said.append(
            "settings changed: "
            + ", ".join(f"{name}={after['settings'][name]}" for name in switches)
        )
    return "; ".join(said) if said else "the plan is unchanged"


def _one_rule(was: dict, now: dict, moved: list[int]) -> str:
    """The rule a change follows, when every changed order follows one.

    A round that edits by rule -- every `SELL MELON`, capped -- changes hundreds
    of steps with one transformation, and saying "389 pooled steps changed" hides
    that. This returns the rule when there is one and nothing when there is not,
    because a diff that is not a rule should not be described as one.
    """
    rules = set()
    ratios = set()
    for at in moved:
        old, new = was[at], now[at]
        if old["farmer"] != new["farmer"] or old["hands"] != new["hands"]:
            return ""
        if len(old["market"]) != len(new["market"]):
            return ""
        # The same length by the check above it, which is what `strict` says.
        for one, other in zip(old["market"], new["market"], strict=True):
            if one == other:
                continue
            if len(one) != 3 or len(other) != 3 or one[:2] != other[:2]:
                return ""
            rules.add((one[0], one[1]))
            ratios.add(round(other[2] / one[2], 3) if one[2] else None)
    if len(rules) != 1:
        return ""
    ((verb, item),) = rules
    if len(ratios) == 1 and (factor := next(iter(ratios))):
        return f"every `{verb} {item}` x{factor:g} ({len(moved)} steps)"
    return f"every `{verb} {item}` ({len(moved)} steps)"


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
    carried = json.loads(zlib.decompress(base64.b85decode(found.group("payload"))))

    # The strategy the controller keeps in its own source. Read here so the
    # plan is the whole of it, and written back by `join`.
    carried["settings"] = carried_settings(source)
    # `_R42_OPENING` replaces step 0's market orders in every route, so the
    # plan's own `actions[0]` never runs. Every route cites entry 0 there and
    # nothing cites it elsewhere, so putting the opening in it is exact.
    carried["actions"][0]["market"] = ast.literal_eval(
        _carried(source, CARRIED_OPENING)
    )
    # One lookup. The controller's table decides a map that unlocked a
    # YARN_STORE and the packed one decides the rest, so each carries a dead
    # half; this takes every pair from whichever table actually decides it.
    theirs = ast.literal_eval(_carried(source, CARRIED_YARN))
    carried["shops"] = [
        dict(entry, route=theirs[tuple(entry["shops"])])
        if "YARN_STORE" in entry["shops"]
        else entry
        for entry in carried["shops"]
    ]

    plan = Plan.model_validate(carried).model_dump(mode="json")
    line = LOADER.format(name=found.group("name"), module=PLAN_MODULE)
    return source[: found.start()] + line + source[found.end() :], plan


def _carried(source: str, pattern: re.Pattern[str]) -> str:
    """The literal a controller keeps on one of its own lines.

    Args:
        source: The program's source.
        pattern: The line to find.

    Returns:
        The literal's text.

    Raises:
        ValueError: The program does not carry that line.
    """
    found = pattern.search(source)
    if found is None:
        raise ValueError(f"this program carries no {pattern.pattern.split(chr(92))[0]}")
    return found.group(1)


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
    whole = Plan.model_validate(plan)
    carried = whole.model_dump(mode="json")

    # Back into the lines the controller reads them from. The settings and the
    # opening are not packed with the rest: the controller takes them from its
    # own source, and a copy in the blob would be a second answer to the same
    # question.
    settings = carried.pop("settings")
    opening = carried["actions"][0]["market"]
    # Only the pairs that table is consulted for. It decides a map that
    # unlocked a YARN_STORE and nothing else, so its other 49 entries were
    # never read and are not carried back.
    shops = {
        tuple(entry["shops"]): entry["route"]
        for entry in carried["shops"]
        if "YARN_STORE" in entry["shops"]
    }

    packed = base64.b85encode(
        zlib.compress(json.dumps(carried, separators=(",", ":")).encode(), 9)
    ).decode()
    if "'" in packed:
        raise ValueError("the packed plan holds a quote")
    line = (
        f"{found.group('name')}=json.loads(zlib.decompress("
        f"base64.b85decode('{packed}')))"
    )
    source = controller[: found.start()] + line + controller[found.end() :]
    for pattern, value in (
        (CARRIED_SETTINGS, settings),
        (CARRIED_OPENING, [list(order) for order in opening]),
        (CARRIED_YARN, shops),
    ):
        if pattern.search(source) is None:
            raise ValueError(f"this controller has no {pattern.pattern[1:12]} line")
        written = f"{pattern.pattern[1:].split(chr(92))[0]}={value!r}"
        source = pattern.sub(lambda _, w=written: w, source, count=1)
    return source


# `actions[N]` is line `N + FIRST_STEP_LINE` of the file `readable` writes: line
# 1 opens the object, line 2 opens `actions`, so the first step is line 3. That
# is what lets a round go from an index in a season to the step it names.
FIRST_STEP_LINE = 3


def readable(plan: dict) -> str:
    """The plan as a file a round can address, one part of it per line.

    `json.dumps(indent=1)` renders the champion's plan as 213,302 lines and
    1.93MB, and 29,479 of those lines hold a single digit -- every index of
    every season on a line of its own. Nothing reads that and nothing greps it
    usefully, which is the likeliest reason no round has ever edited the plan
    despite being told to. One step per line is 4,095 lines, and it makes an
    index into `actions` a line number.

    Args:
        plan: The plan, as `split` returns it.

    Returns:
        The file's text, which is still ordinary JSON.
    """
    compact = (",", ":")
    steps = ",\n".join(
        f"  {json.dumps(step, separators=compact)}" for step in plan["actions"]
    )
    seasons = ",\n".join(
        f"  {json.dumps(at)}: {json.dumps(plan['routes'][at], separators=compact)}"
        for at in sorted(plan["routes"], key=int)
    )
    shops = ",\n".join(
        f"  {json.dumps(entry, separators=compact)}" for entry in plan["shops"]
    )
    return (
        f'{{\n "actions": [\n{steps}\n ],\n'
        f' "routes": {{\n{seasons}\n }},\n'
        f' "shops": [\n{shops}\n ],\n'
        f' "settings": {json.dumps(plan["settings"], separators=compact)}\n}}\n'
    )


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
        (box / name).write_text(shelve_notice(source, box), encoding="utf-8")
        return
    controller, plan = split(source)
    controller = shelve_notice(shelve_blob(controller, box), box)
    (box / name).write_text(controller, encoding="utf-8")
    (box / PLAN_FILE).write_text(readable(plan), encoding="utf-8")
    (box / f"{PLAN_MODULE}.py").write_text(
        PLAN_LOADER_SOURCE.format(file=PLAN_FILE), encoding="utf-8"
    )


def shelve_blob(controller: str, box: Path) -> str:
    """Move the prototype blob out of the controller into a file beside it.

    Args:
        controller: The program's source, plan already taken out.
        box: The round's directory.

    Returns:
        The controller with a loader line where the blob was, or unchanged
        when it carries no blob.
    """
    found = BLOB.search(controller)
    if found is None:
        return controller
    (box / BLOB_FILE).write_text(found.group("payload"), encoding="utf-8")
    (box / f"{BLOB_MODULE}.py").write_text(
        BLOB_LOADER_SOURCE.format(file=BLOB_FILE), encoding="utf-8"
    )
    line = LOADER.format(name=found.group("name"), module=BLOB_MODULE)
    return controller[: found.start()] + line + controller[found.end() :]


def shelve_notice(controller: str, box: Path) -> str:
    """Move the inlined licence body out of the program into a file beside it.

    Args:
        controller: The program's source, plan already taken out.
        box: The round's directory.

    Returns:
        The controller with one line where the licence body was, or unchanged
        when it carries none -- the scratch lineage starts from a program that
        derives from nothing and has no notice to keep.
    """
    found = NOTICE.search(controller)
    if found is None:
        return controller
    (box / NOTICE_FILE).write_text(found.group(0), encoding="utf-8")
    return controller[: found.start()] + NOTICE_MARK + controller[found.end() :]


def unshelve_notice(source: str, box: Path) -> str:
    """Put the licence body back, from the file `shelve_notice` wrote.

    Args:
        source: The program, plan and blob already packed back in.
        box: The round's directory.

    Returns:
        The program with the licence body inline, or unchanged when it never
        had a mark.

    Raises:
        ValueError: The mark is there and the file behind it is not. The
            program would ship a derivative work of Apache-2.0 kernels with
            the licence body gone, which is the one thing this may not do
            quietly.
    """
    if NOTICE_MARK not in source:
        return source
    shelf = box / NOTICE_FILE
    if not shelf.exists():
        raise ValueError(
            f"the program's licence body was shelved to {NOTICE_FILE} and the "
            "round removed it"
        )
    return source.replace(NOTICE_MARK, shelf.read_text(encoding="utf-8"), 1)


def unshelve_blob(source: str, box: Path) -> str:
    """Put the prototype blob back on its line, from the file `shelve_blob` wrote.

    Args:
        source: The program, plan already packed back in.
        box: The round's directory.

    Returns:
        The program with the blob literal inline, or unchanged when it never
        had a loader line.

    Raises:
        ValueError: The loader line is there and the file behind it is not, so
            the program would load nothing where it expects a library.
    """
    found = BLOB_UNPACKED.search(source)
    if found is None:
        return source
    shelf = box / BLOB_FILE
    if not shelf.exists():
        raise ValueError(f"the controller loads {BLOB_FILE} and the round removed it")
    payload = shelf.read_text(encoding="utf-8").strip()
    if "'" in payload:
        raise ValueError(f"{BLOB_FILE} holds a quote and cannot go back on its line")
    line = f"{found.group('name')} = '{payload}'"
    return source[: found.start()] + line + source[found.end() :]


class BrokenPlanError(ValueError):
    """A round wrote a `plan.json` the schema refuses.

    Raised rather than allowed out as `ValidationError` because the two want
    different handling: a pydantic error anywhere else in the campaign is a bug
    in the campaign, and this one is a model writing something malformed, which
    is an ordinary outcome of asking a model to edit a file. The round is told
    and the run goes on.
    """


def carried_settings(source: str) -> dict:
    """The chassis settings a controller keeps in its own source.

    The chassis builds its config as the defaults with the overrides applied,
    so this reads it the same way rather than guessing which line a value is
    on. Only the keys the schema names are taken; the rest of
    `DEFAULT_SETTINGS` restates the engine.

    Read by `split`, which lifts it into the plan so a round sees the whole
    strategy in one file, and by `gather`, which puts it back when a round
    rewrote `plan.json` without it.

    Args:
        source: A controller, packed or unpacked -- both carry the lines.

    Returns:
        The settings the schema names, defaults under overrides.

    Raises:
        ValueError: The controller carries no settings lines.
    """
    chassis = ast.literal_eval(_carried(source, CARRIED_DEFAULTS))
    chassis.update(ast.literal_eval(_carried(source, CARRIED_SETTINGS)))
    return {name: chassis[name] for name in Settings.model_fields if name in chassis}


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
        # and what it wrote is what it meant. The plan file is a leftover --
        # the licence body is not, and goes back whatever else was rewritten.
        return unshelve_notice(source, box)
    # Checked here, where the round can still be told it wrote nothing usable.
    # Unchecked, a plan with a verb the engine has no op for, or a shop pair
    # pointing at a route that is not there, is a program that forfeits every
    # game -- and the campaign would read that as a bad idea rather than a
    # broken file.
    try:
        written = json.loads(plan_file.read_text(encoding="utf-8"))
        # `settings` is the controller's, not the plan file's: `split` lifts it
        # out of `_SETTINGS` so a round can see the whole strategy in one
        # place, and `join` writes it back to the same line. A round that
        # regenerates `plan.json` and omits it has changed nothing -- the
        # values are still in the controller this is about to pack -- so take
        # them from there rather than failing the round over a key it never
        # meant to touch. Thirteen of twenty-two rounds died on exactly this
        # overnight on 2026-09-22.
        #
        # `isinstance` because a round that wrote a list would otherwise raise
        # `TypeError` here, which is not a `ValueError` and would go straight
        # past the clause below and up into the loop -- the shape of crash that
        # killed the main lineage for ten hours the day before.
        if isinstance(written, dict) and "settings" not in written:
            written["settings"] = carried_settings(source)
        return unshelve_notice(unshelve_blob(join(source, written), box), box)
    except ValueError as broken:
        # `ValueError` rather than the two the plan can fail with, because the
        # controller can fail too and does: `join` raises it from eight places
        # -- no loader to pack back into, no `_SETTINGS` line, a packed plan
        # holding a quote -- and a round that rewrites `child.py` instead of
        # `plan.json` hits those. One of them killed the main lineage for ten
        # hours on 2026-09-21. Both `ValidationError` and `JSONDecodeError`
        # subclass this, so the wider clause is also the shorter one.
        raise BrokenPlanError(
            f"what this round wrote does not pack back into a program, so it "
            f"produced nothing runnable: {broken}"
        ) from broken
