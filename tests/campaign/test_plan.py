"""Taking a program's plan out for a round, and putting it back.

A champion is a controller around a plan, and the plan ships as one line of
base85'd zlib'd JSON. Every test here is about the same property: what a round
works on can be two files, and what everything else sees is always one.
"""

import asyncio
import base64
import json
import zlib
from pathlib import Path

import pytest

from kaggriculture.campaign import config, harness, mutate, plan, roster
from kaggriculture.campaign.plan import Plan

# A program shaped like a champion: imports, a packed plan under a generated
# name, and an agent that reads it.
# The plan exactly as a program carries it. The schema describes this and does
# not reshape it: a schema that differs from its data needs a conversion at
# every boundary, and the one that was missing handed every round an agent that
# forfeited every game it played.
PLAN = {
    # A pool of distinct steps. Which position a step sits at here means
    # nothing; the seasons below are what put them in an order.
    "actions": [
        {"farmer": ["PASS"], "hands": [], "market": []},
        {"farmer": ["NORTH"], "hands": [], "market": [["SELL", "WHEAT", 3]]},
    ],
    # Two whole seasons, each one index per step. The route the lookup names has
    # to be one that is here and every index has to name a step the pool holds,
    # which are the two things the schema checks and the two things this fixture
    # got wrong -- it cited steps 1, 2 and 3 of a one-step pool.
    "routes": {
        "101": [step % 2 for step in range(config.SEASON)],
        "112": [0] * config.SEASON,
    },
    "shops": [{"route": 101, "shops": ["BAKERY", "BAKERY"]}],
}


def packed(data: dict) -> str:
    """A program carrying `data` the way a champion carries its plan.

    The schema is the plan's own shape, so there is nothing to convert: what
    these tests build is what a champion actually is.
    """
    carried = Plan.model_validate(data).model_dump(mode="json")
    blob = base64.b85encode(
        zlib.compress(json.dumps(carried, separators=(",", ":")).encode(), 9)
    ).decode()
    return (
        "import base64\nimport json\nimport zlib\n"
        f"_R108_DATA=json.loads(zlib.decompress(base64.b85decode('{blob}')))\n"
        "def agent(observation, configuration=None):\n"
        "    return _R108_DATA['actions'][0]\n"
    )


def test_a_plan_comes_out_and_goes_back_unchanged() -> None:
    """The plan survives the trip, and the controller is not touched on the way.

    What matters is the plan as data and the Python around it, not the bytes the
    packing happens to produce: key order and separators are the serialiser's
    business, and a program that comes back with its keys in another order plays
    exactly the same season.

    Nothing downstream compares these bytes either. A round's verdict is
    `gather` before against `gather` after, and both sides of that are packed by
    the same code in the same process -- so a round that changed nothing reads as
    nothing whatever order the keys come out in.
    """
    source = packed(PLAN)

    controller, data = plan.split(source)

    assert data == PLAN
    assert "b85decode" not in controller

    whole = plan.join(controller, data)
    again, back = plan.split(whole)
    assert back == PLAN, "the plan is the same plan"
    assert again == controller, "and the controller was never touched"


def test_the_controller_is_what_is_left_when_the_plan_goes() -> None:
    """What a round reads is the program minus the part it cannot read."""
    source = packed(PLAN)

    controller, _ = plan.split(source)

    assert len(controller) < len(source)
    assert plan.UNPACKED.search(controller) is not None
    # The imports above it do not move: the line is replaced in place.
    assert controller.splitlines()[:3] == source.splitlines()[:3]


def test_a_round_is_given_the_plan_as_json_it_can_edit(tmp_path: Path) -> None:
    """The box holds the controller, the plan, and the three lines that join them."""
    plan.lay_out(packed(PLAN), tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "_plan.py",
        "child.py",
        "plan.json",
    ]
    # Readable, which is the entire point.
    assert json.loads((tmp_path / "plan.json").read_text()) == PLAN
    assert "b85decode" not in (tmp_path / "child.py").read_text()


def test_a_program_with_no_plan_is_laid_out_as_it_is(tmp_path: Path) -> None:
    """The lineage before champion_17 has no plan, and still has to run."""
    source = "def agent(o, c=None):\n    return {}\n"

    plan.lay_out(source, tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["child.py"]
    assert (tmp_path / "child.py").read_text() == source
    assert plan.gather(tmp_path) == source


def test_editing_only_the_plan_changes_the_program(tmp_path: Path) -> None:
    """A round that touched no Python has still changed what the agent does.

    This is the property the whole split turns on. `child.py` is untouched, so
    anything that judged a round by that file alone would call this nothing --
    which is exactly how eight promotions went by with the strategy frozen.
    """
    source = packed(PLAN)
    plan.lay_out(source, tmp_path)
    before = (tmp_path / "child.py").read_text()

    edited = json.loads((tmp_path / "plan.json").read_text())
    edited["shops"][0]["route"] = 112
    (tmp_path / "plan.json").write_text(json.dumps(edited), encoding="utf-8")

    assert (tmp_path / "child.py").read_text() == before, "no Python was edited"
    whole = plan.gather(tmp_path)
    assert whole != source, "but the program is not the one we started with"
    _, again = plan.split(whole)
    assert again["shops"][0]["route"] == 112


def test_a_round_that_drops_the_loader_keeps_what_it_wrote(tmp_path: Path) -> None:
    """A round may restructure its program; then what it wrote is the program."""
    plan.lay_out(packed(PLAN), tmp_path)
    (tmp_path / "child.py").write_text("def agent(o, c=None):\n    return {}\n")

    assert plan.gather(tmp_path) == "def agent(o, c=None):\n    return {}\n"


def test_a_mutator_calls_a_plan_only_edit_a_child(tmp_path: Path) -> None:
    """End to end: the verdict is about the program, not about one of its files.

    The stand-in edits `plan.json` and leaves `child.py` alone, which is what a
    round changing the strategy looks like from outside.
    """
    plan.lay_out(packed(PLAN), tmp_path)
    edit = (
        "python3 - <<'EOF'\n"
        "import json, pathlib\n"
        "p = pathlib.Path('plan.json')\n"
        "d = json.loads(p.read_text())\n"
        "d['shops'][0]['route'] = 112\n"
        "p.write_text(json.dumps(d))\n"
        "EOF\n"
        "printf '%s' "
        '\'{"type":"result","result":{"status":"SUCCESS","response":"done",'
        '"usage":{"input_tokens":5,"output_tokens":5}}}\''
    )

    class Stand:
        """An opencode that edits the plan and nothing else."""

        SKILLS_DIR = Path(".agents") / "skills"
        COMMAND = ["bash", "-c", edit]
        POLICY: dict = {}

        def invocation(self, workspace: Path, message: str, model: str) -> list[str]:
            """The stand-in, unchanged."""
            return list(self.COMMAND)

    mutator = mutate.OpenCodeMutator(model="m", fallback="")
    mutator.COMMAND = Stand.COMMAND

    result = asyncio.run(mutator(tmp_path, "improve it", "p40"))

    assert result.status == "ok", result.reason
    _, data = plan.split(plan.gather(tmp_path))
    assert data["shops"][0]["route"] == 112


def test_a_round_that_changes_nothing_is_still_no_output(tmp_path: Path) -> None:
    """The split must not turn every round into a success.

    `child.py` in the box is the controller and the program is the controller
    plus the plan, so anything comparing the file against the program finds them
    different every time -- and a round that did nothing at all reads as one
    that rewrote the agent. That failure is silent and it flatters: the campaign
    would archive its own parent once per round and call it a child.
    """
    plan.lay_out(packed(PLAN), tmp_path)

    mutator = mutate.OpenCodeMutator(model="m", fallback="")
    mutator.COMMAND = ["true"]  # an opencode that does nothing whatsoever

    result = asyncio.run(mutator(tmp_path, "improve it", "p41"))

    assert result.status == "no_output" and result.child is None


def test_the_schema_names_what_is_wrong_with_a_plan() -> None:
    """A bad edit is refused here, where it is still a sentence rather than a loss.

    Unchecked, each of these ships a program that forfeits every game it plays,
    and the campaign reads that as a bad idea instead of a broken file.
    """
    for wrong, says in (
        # The relational rules, which no field constraint can express: one
        # field has to agree with another.
        (
            {"shops": [{"route": 999, "shops": ["BAKERY", "BAKERY"]}]},
            "does not hold",
        ),
        # A season citing a step past the end of the pool. This is the shape a
        # generator reaches for first, because an index is just a number and
        # nothing local to it is wrong.
        (
            {"routes": {"101": [len(PLAN["actions"])] * config.SEASON}},
            "cites",
        ),
        # A season that is not a season. Short and the farm idles out the year
        # on `PASS`, long and the tail is never read -- either way the plan is
        # not the thing it claims to be.
        (
            {"routes": {"101": [0] * (config.SEASON - 1)}},
            f"at least {config.SEASON}",
        ),
        (
            {"routes": {"101": [0] * (config.SEASON + 1)}},
            f"at most {config.SEASON}",
        ),
        # A route key the program cannot turn into a number. `int(k)` on it is
        # a crash at load, before a move is made, and a generator asked for a
        # plan reached for exactly this: a route called "season".
        (
            {"routes": {"season": [0] * config.SEASON}},
            "string_pattern_mismatch",
        ),
        # And the rest, which the types carry, so the message names the path.
        (
            {"actions": [{"farmer": {"verb": "TELEPORT"}, "hands": [], "market": []}]},
            "farmer",
        ),
        (
            {"shops": [{"route": 101, "shops": ["NOT_A_SHOP", "BAKERY"]}]},
            "shops",
        ),
        # A crop the engine does not grow. A bare `PLANT` with no crop at all is
        # not here: `action.rs` reads one as a no-op and plays the season, so the
        # schema says it too.
        (
            {
                "actions": [
                    {
                        "farmer": {"verb": "PLANT", "crop": "EGG"},
                        "hands": [],
                        "market": [],
                    }
                ]
            },
            "crop",
        ),
    ):
        with pytest.raises(ValueError, match=says):
            plan.Plan.model_validate({**PLAN, **wrong})


def test_the_schema_carries_the_rules_rather_than_checking_them_after() -> None:
    """A generator reads the schema, not the validators.

    `model_json_schema` is the whole of what a structured call is told, so a
    rule kept in a `field_validator` is a rule the writer never sees: it emits
    a verb the engine has no op for and finds out by being refused. As types,
    the vocabulary and the arities are in the schema, and strict decoding
    cannot spell anything else.
    """
    schema = plan.Plan.model_json_schema()
    defs = schema["$defs"]

    # The vocabularies are named types the fields point at, which is how
    # pydantic renders an enum: `{"enum": [...]}` under `$defs`.
    assert "HARVEST" in defs["UnitOp"]["enum"], "the engine's verbs"
    assert "TELEPORT" not in defs["UnitOp"]["enum"]
    # The whole of the engine's list, not a subset kept by subtraction: a bare
    # `PLANT` is a no-op the engine runs, so the schema can say it.
    assert defs["UnitOp"]["enum"] == config.UNIT_OPS
    assert "BAKERY" in defs["ShopName"]["enum"], "and the shop names"
    market = json.dumps(defs["Action"]["properties"]["market"])
    for verb in config.MARKET_OPS:
        if verb != "NONE":
            assert verb in market, f"{verb} is a shape the market can take"

    # A command is a bounded list whose tokens are those vocabularies, which is
    # the program's own shape and a shape every provider's strict mode takes --
    # a tuple would be `prefixItems`, and that is the one OpenAI refuses.
    farmer = json.dumps(defs["Action"]["properties"]["farmer"])
    # One shape per arity, which is the only way a schema constrains a
    # *position*: given a bounded list of tokens instead, GLM wrote
    # `["FERTILIZER", "HARVEST"]` and it validated.
    assert "prefixItems" in farmer
    assert "#/$defs/UnitOp" in farmer and "#/$defs/CropName" in farmer
    # And a map's two shops are said as a length.
    shops = defs["ShopRoute"]["properties"]["shops"]
    assert shops["minItems"] == 2 and shops["maxItems"] == 2
    # Routes are the mapping the program keeps, described rather than reshaped.
    routes = schema["properties"]["routes"]
    assert routes["type"] == "object", "the shape the plan has, not a tidier one"


def test_the_schema_takes_the_plan_the_champion_actually_carries() -> None:
    """Whatever it rejects, it has to accept the one in play.

    The shapes are irregular in ways a schema written from imagination would
    forbid: the farmer's command is one, two or three parts; a market order can
    be empty, and 283 of them are; a step has between zero and twelve hands.
    """
    champion = Path("run/campaign/floor/agent/main.py")
    if not champion.exists():
        pytest.skip("no live champion in this checkout")

    _, data = plan.split(champion.read_text(encoding="utf-8"))
    checked = plan.Plan.model_validate(data)

    assert len(checked.actions) == 3982
    assert any(not action.market for action in checked.actions)
    assert max(len(action.hands) for action in checked.actions) == 12


def test_the_real_champion_survives_the_round_trip() -> None:
    """The property, on the program the campaign is actually standing on."""
    champion = Path("run/campaign/floor/agent/main.py")
    if not champion.exists():
        pytest.skip("no live champion in this checkout")
    source = champion.read_text(encoding="utf-8")

    controller, data = plan.split(source)

    assert {*data} == {"actions", "routes", "shops"}
    # The plan, not the bytes. This champion happens to come back byte for byte
    # -- the models are declared in the order it writes its keys -- but that is
    # a tidy archive rather than a property anything depends on, and a champion
    # that ordered them differently would still be the same program.
    _, back = plan.split(plan.join(controller, data))
    assert back == data


def step(**parts: object) -> dict:
    """One action, defaulting to a farmer that passes and an empty board."""
    return {"farmer": ["PASS"], "hands": [], "market": [], **parts}


def test_the_schema_accepts_everything_the_engine_acts_on() -> None:
    """Whatever it refuses, it may not refuse a plan that plays.

    These are shapes `action.rs` parses and acts on, and the first schema --
    written from one champion's habits rather than from the parser -- refused
    two of them. A schema that rejects a legal plan costs a round for nothing.
    """
    legal = [
        # A farm can work thirty-nine hands: `sim.hpp` keeps MAX_UNITS slots for
        # the farmer and its hands, and the first schema capped this at twelve.
        step(hands=[["PASS"]] * (config.MAX_UNITS - 1)),
        # `PICKUP` and `PLACE` take an item, and the count is optional.
        step(farmer=["PICKUP", "COW"]),
        step(farmer=["PICKUP", "COW", 2]),
        step(farmer=["PLACE", "WHEAT"]),
        # `PLANT` takes a crop and no count at all.
        step(farmer=["PLANT", "STRAWBERRY"]),
        # Verbs that stand alone.
        step(farmer=["COLLECT_FERTILIZER"]),
        step(market=[["HIRE"], ["BUY_LAND"]]),
        # An empty order: the engine drops it, and 283 of the champion's steps
        # place one.
        step(market=[[]]),
        # A count of zero is dropped rather than refused, and the champion
        # carries `["SELL", "STRAWBERRY", 0]` at step 449.
        step(market=[["SELL", "STRAWBERRY", 0]]),
        # Each moving verb with the names its own arm accepts.
        step(market=[["SELL", "WOOL", 3]]),
        step(market=[["BUY_SEED", "MELON", 1]]),
        step(market=[["BUY_PRODUCT", "FERTILIZER", 5]]),
        step(market=[["BUY_ANIMAL", "SHEEP", 2]]),
    ]

    plan.Plan.model_validate({**PLAN, "actions": legal})


def test_the_schema_refuses_what_it_can_and_describes_the_rest() -> None:
    """What a bounded list of tokens can say, and what it cannot.

    It can say the vocabulary and the arity: a verb the engine has no op for, a
    command with no verb at all, a command longer than three parts, a map with
    three shops, more hands than the engine keeps slots for.

    It cannot say which verb takes which name -- `["SELL", "COW", 1]` is three
    tokens from the right vocabularies and the schema has no way to object, even
    though `action.rs` drops it. Saying that would mean one shape per verb, which
    means `prefixItems`, which is the one thing OpenAI's strict mode refuses. So
    those rules live in the field descriptions, where a writer reads them, and
    the cost is that a generator can still spell an order the engine will drop.
    """
    for wrong in (
        # A verb the engine has no arm for reads as a pass; it is a mistake in
        # the writing rather than a strategy.
        step(farmer=["TELEPORT"]),
        # The farmer always acts, so its command is never empty.
        step(farmer=[]),
        # Three parts is all the engine reads.
        step(farmer=["PICKUP", "COW", 2, "EXTRA"]),
        # A map unlocks two shops.
        step(),
        # And a farm cannot work more hands than the engine keeps slots for.
        step(hands=[["PASS"]] * config.MAX_UNITS),
    ):
        with pytest.raises(ValueError):
            if wrong == step():
                plan.Plan.model_validate(
                    {**PLAN, "shops": [{"route": 101, "shops": ["BAKERY"]}]}
                )
            else:
                plan.Plan.model_validate({**PLAN, "actions": [wrong]})


def test_every_field_tells_a_writer_what_to_produce() -> None:
    """The schema is the whole of what a structured call is told.

    A docstring's `Attributes:` section documents the source and reaches
    nothing else: `model_json_schema()` carried nine of eleven fields with no
    description at all, which is a bare `array` and a title pydantic invented
    from the attribute name. A writer handed that has to guess what `shops`
    means, what order `hands` is in, and what `route` has to agree with.
    """
    schema = plan.Plan.model_json_schema()

    described: list[str] = []
    bare: list[str] = []
    for where, properties in (
        ("Plan", schema["properties"]),
        *(
            (name, schema["$defs"][name]["properties"])
            for name in ("Action", "ShopRoute")
        ),
    ):
        for field, spec in properties.items():
            (described if spec.get("description") else bare).append(f"{where}.{field}")

    assert not bare, f"fields a writer is told nothing about: {bare}"
    # Three on the plan, three on a step, two on a lookup.
    assert len(described) == 8, described

    # And the descriptions carry the things that cannot be read off a type:
    # what the field means, what order it is in, what it has to agree with.
    assert (
        "order they were hired"
        in (schema["$defs"]["Action"]["properties"]["hands"]["description"])
    )
    assert "id" in schema["$defs"]["ShopRoute"]["properties"]["route"]["description"]
    assert "not a range" in schema["properties"]["routes"]["description"]


@pytest.mark.slow
def test_the_plan_a_round_is_handed_still_plays(tmp_path: Path) -> None:
    """The split form has to play the season the whole one plays.

    Every other test here reads files and compares data, and all of them passed
    while a round was being handed an agent that forfeited every game it played:
    `plan.json` keeps routes as a list, the program indexes them by number, and
    nothing noticed because a crashed agent simply loses. Only playing catches
    that, so this plays.
    """
    champion = config.LIVE.floor / "main.py"
    if not champion.exists():
        pytest.skip("no live champion in this checkout")
    source = champion.read_text(encoding="utf-8")

    whole = tmp_path / "whole"
    apart = tmp_path / "apart"
    whole.mkdir()
    apart.mkdir()
    (whole / "main.py").write_text(source, encoding="utf-8")
    plan.lay_out(source, apart, name="main.py")

    opponent = roster.path("ours_20", config.POOL)
    entire = harness.game(whole / "main.py", opponent, seed=11, seat=0)
    split = harness.game(apart / "main.py", opponent, seed=11, seat=0)

    assert entire.ours > 0, "the champion banks something"
    assert (split.ours, split.theirs) == (entire.ours, entire.theirs)


def test_a_season_written_out_packs_back_to_the_same_play() -> None:
    """A plan is its play, not its pool: unfold and fold change one, not the other.

    The pool is a dedupe, so the entry a step lands at depends on the order the
    seasons are walked. What has to survive is which step each route plays on
    each of its steps, which is what this compares.
    """
    champion = plan.Plan.model_validate(PLAN)
    packed = plan.fold(plan.unfold(champion), champion)

    assert plan.unfold(packed).routes == plan.unfold(champion).routes
    assert sorted(packed.routes, key=int) == sorted(champion.routes, key=int)


def test_one_written_season_is_a_whole_plan() -> None:
    """A writer supplies a season; the program may ask for any route it knows.

    The program's shop table sends some maps to routes a writer never named,
    and a route it cannot find is a `KeyError` before the first move. So every
    route the plan being replaced holds comes back, playing the written season
    where there is nothing better.
    """
    champion = plan.Plan.model_validate(PLAN)
    written = plan.unfold(champion)
    one = plan.Written(routes={"101": written.routes["101"]}, shops=[])

    folded = plan.fold(one, champion)

    assert sorted(folded.routes, key=int) == sorted(champion.routes, key=int)
    assert len({tuple(season) for season in folded.routes.values()}) == 1
    assert plan.unfold(folded).routes["112"] == plan.unfold(champion).routes["101"]


def test_a_written_season_is_steps_and_never_an_index() -> None:
    """The schema a generator reads has no integer to put in a season.

    Handed the pooled form, the model filled the array by counting: 715 of 718
    transitions were +1, into a pool of one entry. Nothing in that form could
    have said otherwise, because no schema coupled an index to the pool.
    """
    schema = plan.Written.model_json_schema()
    # The key carries a pattern, so a season sits under `patternProperties`
    # rather than `additionalProperties`.
    (season,) = schema["properties"]["routes"]["patternProperties"].values()

    assert "#/$defs/Stretch" in json.dumps(season["items"]), "a step, not a number"
    assert set(schema["$defs"]["Stretch"]["properties"]) == {"step", "steps"}
    assert "#/$defs/Action" in json.dumps(schema["$defs"]["Stretch"]["properties"])
    assert "actions" not in plan.Written.model_fields, "the pool is not written"


def test_a_written_season_need_not_be_seven_hundred_entries_long() -> None:
    """The floor of 719 was ours, and it is what forced the padding.

    A model that has worked out a year still had to reach 719 array entries, and
    the cheapest thing to put in the rest is `PASS`. Three models did exactly
    that: 719, 719 and 713 of their steps were the same step. A season is
    stretches now, so a plan for the year can be stated in as many entries as it
    actually has decisions in it.
    """
    schema = plan.Written.model_json_schema()
    (season,) = schema["properties"]["routes"]["patternProperties"].values()

    assert season["minItems"] == 1, "a short season has to be writable"
    assert season["maxItems"] == config.SEASON, "and no longer than a season"

    # Three stretches covering the year, which the old schema refused.
    hold = {"farmer": ["WATER"], "hands": [], "market": []}
    short = plan.Written.model_validate(
        {
            "routes": {
                "100": [
                    {
                        "step": {"farmer": ["PASS"], "hands": [], "market": []},
                        "steps": 1,
                    },
                    {
                        "step": {
                            "farmer": ["PLANT", "WHEAT"],
                            "hands": [],
                            "market": [],
                        },
                        "steps": 1,
                    },
                    {"step": hold, "steps": config.SEASON - 2},
                ]
            },
            "shops": [],
        }
    )
    assert len(short.routes["100"]) == 3
    # And it is a whole season once packed: one pooled step per distinct step,
    # cited as many times as its stretch is long.
    champion = plan.Plan.model_validate(PLAN)
    folded = plan.fold(short, champion)
    assert len(folded.routes["100"]) == config.SEASON
    assert len(folded.actions) == 3


def test_a_written_plan_names_what_is_wrong_with_it() -> None:
    """Refused here, where it is still a sentence rather than a lost game."""
    pause = {"farmer": ["PASS"], "hands": [], "market": []}
    whole = [{"step": pause, "steps": config.SEASON}]
    for wrong, says in (
        # A season that runs past the end of the year. Stopping short is legal
        # and means the farm is idle for the rest, which is what both models
        # wrote out longhand; running over cannot be played at all.
        (
            {
                "routes": {
                    "100": [
                        {"step": pause, "steps": 400},
                        {
                            "step": {"farmer": ["WATER"], "hands": [], "market": []},
                            "steps": 400,
                        },
                    ]
                }
            },
            "run past the end",
        ),
        (
            {"routes": {"100": [{"step": pause, "steps": config.SEASON + 1}]}},
            "less than or equal",
        ),
        ({"routes": {"season": whole}}, "string_pattern_mismatch"),
        # A crop the engine does not grow, in one stretch out of two.
        (
            {
                "routes": {
                    "100": [
                        {"step": pause, "steps": config.SEASON - 1},
                        {
                            "step": {
                                "farmer": ["PLANT", "EGG"],
                                "hands": [],
                                "market": [],
                            },
                            "steps": 1,
                        },
                    ]
                }
            },
            "step",
        ),
    ):
        with pytest.raises(ValueError, match=says):
            plan.Written.model_validate({"routes": {}, "shops": [], **wrong})


@pytest.mark.slow
def test_a_folded_season_plays(tmp_path: Path) -> None:
    """Written out and packed back, the champion banks what the champion banks.

    Every data comparison above passed once while the program they described
    forfeited every game it played, because a crashed agent simply loses. So
    this plays, and it compares banks rather than asking whether one was made:
    a fold that sent every step to the pool's first entry still banked money,
    because that step is a wheat trade and 719 of them still trade.
    """
    source = config.LIVE.floor / "main.py"
    if not source.exists():
        pytest.skip("no live champion in this checkout")
    controller, carried = plan.split(source.read_text(encoding="utf-8"))
    whole = plan.Plan.model_validate(carried)

    packed = plan.fold(plan.unfold(whole), whole)
    agent = tmp_path / "main.py"
    agent.write_text(
        plan.join(controller, packed.model_dump(mode="json")), encoding="utf-8"
    )

    opponent = roster.path("ours_24", config.POOL)
    champion = harness.game(source, opponent, seed=11, seat=0)
    folded = harness.game(agent, opponent, seed=11, seat=0)

    assert champion.ours > 0, "the champion banks something to compare against"
    assert (folded.ours, folded.theirs) == (champion.ours, champion.theirs)


def test_the_plan_a_round_opens_is_addressable_by_line() -> None:
    """One step per line, so an index into `actions` is a line number.

    `json.dumps(indent=1)` puts every index of every season on a line of its
    own: 213,302 lines for the champion, 29,479 of them a single digit. A round
    is told to edit that file and never has. What makes it editable is not that
    it is smaller but that the two halves address each other -- `actions[N]` is
    line `N + FIRST_STEP_LINE`, so a season's index is somewhere to go.
    """
    text = plan.readable(PLAN)
    lines = text.splitlines()

    assert json.loads(text) == PLAN, "still the same plan, still ordinary JSON"
    # One line per step, per season and per lookup entry, and eight lines of
    # structure: the brace, the three keys, their three closers, and the brace.
    assert len(lines) == (
        len(PLAN["actions"]) + len(PLAN["routes"]) + len(PLAN["shops"]) + 8
    )
    for at, step in enumerate(PLAN["actions"]):
        line = lines[at + plan.FIRST_STEP_LINE - 1].strip().rstrip(",")
        assert json.loads(line) == step, f"actions[{at}] is not on its own line"
    # And a whole season on one line, so a round can read one at a time.
    # Indexed through the model rather than the fixture: a bare dict literal's
    # value type is the union of all three fields, so `PLAN["routes"]["101"]`
    # is a subscript the checker cannot call sound.
    (season,) = [line for line in lines if line.strip().startswith('"101"')]
    written = json.loads(season.split(": ", 1)[1].rstrip(","))
    assert written == Plan.model_validate(PLAN).routes["101"]


def test_the_plan_a_round_opens_is_not_one_line_per_index() -> None:
    """The layout it replaced, so the reason for this one stays measured.

    This is the whole defect in one assertion: the same plan, written the way
    `lay_out` used to write it, is more than ten times the lines and most of
    them hold one digit.
    """
    theirs = json.dumps(PLAN, indent=1, sort_keys=True).splitlines()
    ours = plan.readable(PLAN).splitlines()

    assert len(theirs) > 10 * len(ours)
    digits = sum(1 for line in theirs if line.strip().rstrip(",").isdigit())
    assert digits > len(ours), "most of the old file was one number per line"
    assert not [line for line in ours if line.strip().rstrip(",").isdigit()]


def test_a_season_that_stops_short_leaves_the_farm_idle() -> None:
    """A writer that runs out of plan is not a writer that has to be refused.

    The stretches had to sum to exactly 719 for one commit, which is the same
    mistake as requiring 719 entries in a smaller size: a model wrote five
    stretches covering 699 steps, twenty short, and the whole strategy was
    thrown away over the arithmetic. It then tried twice more and gave up.

    So a short season is legal and the rest of the year is idle -- which is what
    the models were writing out longhand anyway, both ending on a long `PASS` --
    and `fold` writes that idleness into the season, because the program indexes
    a season by step number and every step has to be there.
    """
    champion = plan.Plan.model_validate(PLAN)
    working = {"farmer": ["WATER"], "hands": [], "market": []}
    short = plan.Written.model_validate(
        {
            "routes": {"100": [{"step": working, "steps": config.SEASON - 20}]},
            "shops": [],
        }
    )

    folded = plan.fold(short, champion)
    season = [folded.actions[at] for at in folded.routes["100"]]

    assert len(season) == config.SEASON, "the program needs every step"
    assert all(step.farmer[0] == "WATER" for step in season[: config.SEASON - 20])
    assert all(step.model_dump(mode="json") == plan.IDLE for step in season[-20:])
    # Two pooled entries: the step it named, and doing nothing.
    assert len(folded.actions) == 2
