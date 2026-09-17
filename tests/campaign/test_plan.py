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

from kaggriculture.campaign import config, mutate, plan
from kaggriculture.campaign.plan import Plan

# A program shaped like a champion: imports, a packed plan under a generated
# name, and an agent that reads it.
# A command is the positional list the program itself carries -- said in the
# schema as a bounded list of tokens, because a tuple becomes `prefixItems` and
# that is the one shape OpenAI's strict mode refuses. Routes are the exception:
# they are a list of `{id, tiles}` where the program keeps a dict, because an
# object with writer-chosen keys is the other thing it refuses.
PLAN = {
    "actions": [{"farmer": ["PASS"], "hands": [], "market": []}],
    # The route the lookup names has to be one that is here, which is the first
    # thing the schema checks and the first thing this fixture got wrong.
    "routes": [{"id": 101, "tiles": [1, 2, 3]}, {"id": 112, "tiles": [3, 2, 1]}],
    "shops": [{"route": 101, "shops": ["BAKERY", "BAKERY"]}],
}


def packed(data: dict) -> str:
    """A program carrying `data` the way a champion carries its plan.

    The program keeps routes as a dict keyed by number; the round is handed the
    list a schema can describe. This writes the program's form, so what these
    tests build is what a champion actually is.
    """
    carried = Plan.model_validate(data).model_dump(
        mode="json", context={"packed": True}
    )
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
        # The relational rule, which no field constraint can express: one field
        # has to agree with another.
        (
            {"shops": [{"route": 999, "shops": ["BAKERY", "BAKERY"]}]},
            "does not hold",
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
    assert defs["MarketOp"]["enum"] == config.MARKET_OPS
    assert "BAKERY" in defs["ShopName"]["enum"], "and the shop names"

    # A command is a bounded list whose tokens are those vocabularies, which is
    # the program's own shape and a shape every provider's strict mode takes --
    # a tuple would be `prefixItems`, and that is the one OpenAI refuses.
    farmer = defs["Action"]["properties"]["farmer"]
    assert farmer["minItems"] == 1 and farmer["maxItems"] == 3
    assert "#/$defs/UnitOp" in json.dumps(farmer["items"])
    assert "#/$defs/ItemName" in json.dumps(farmer["items"])
    # And a map's two shops are said as a length, for the same reason.
    shops = defs["ShopRoute"]["properties"]["shops"]
    assert shops["minItems"] == 2 and shops["maxItems"] == 2


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


def test_the_schema_cannot_say_which_verb_takes_which_name() -> None:
    """The gap, asserted so nobody discovers it by trusting the schema.

    `action.rs` drops each of these -- an animal cannot be sold, only wheat and
    fertilizer can be bought, eggs are not planted -- and a bounded list of
    tokens has no way to refuse them. The descriptions say so; the types cannot.
    """
    for dropped in (
        step(market=[["SELL", "COW", 1]]),
        step(market=[["BUY_PRODUCT", "STRAWBERRY", 1]]),
        step(market=[["BUY_SEED", "EGG", 1]]),
        step(farmer=["PLANT", "EGG"]),
        step(market=[["BUY_ANIMAL", "WHEAT", 1]]),
    ):
        plan.Plan.model_validate({**PLAN, "actions": [dropped]})

    # And the rule is in the text a writer is handed instead.
    market = plan.Plan.model_json_schema()["$defs"]["Action"]["properties"]["market"]
    said = json.dumps(market)
    assert "a product" in said and "only" in said


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
            for name in ("Action", "Route", "ShopRoute")
        ),
    ):
        for field, spec in properties.items():
            (described if spec.get("description") else bare).append(f"{where}.{field}")

    assert not bare, f"fields a writer is told nothing about: {bare}"
    # Three on the plan, three on a step, two on a route, two on a lookup.
    assert len(described) == 10, described

    # And the descriptions carry the things that cannot be read off a type:
    # what the field means, what order it is in, what it has to agree with.
    assert (
        "order they were hired"
        in (schema["$defs"]["Action"]["properties"]["hands"]["description"])
    )
    assert "id" in schema["$defs"]["ShopRoute"]["properties"]["route"]["description"]
