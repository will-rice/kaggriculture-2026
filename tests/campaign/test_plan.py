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

# A program shaped like a champion: imports, a packed plan under a generated
# name, and an agent that reads it.
# Routes are a list of `{id, tiles}` rather than the dict the program keeps,
# because strict structured output refuses an object whose keys are the
# writer's to choose -- and a plan that cannot be a schema cannot be generated
# against one. `split` and `join` convert at the boundary.
PLAN = {
    "actions": [{"farmer": ["PASS"], "hands": [], "market": []}],
    # The route the lookup names has to be one that is here, which is the first
    # thing the schema checks and the first thing this fixture got wrong.
    "routes": [{"id": 101, "tiles": [1, 2, 3]}, {"id": 112, "tiles": [3, 2, 1]}],
    "shops": [{"shops": ["BAKERY", "BAKERY"], "route": 101}],
}


def packed(data: dict) -> str:
    """A program carrying `data` the way a champion carries its plan.

    The program keeps routes as a dict keyed by number; the round is handed the
    list a schema can describe. This writes the program's form, so what these
    tests build is what a champion actually is.
    """
    carried = {
        **data,
        "routes": {str(route["id"]): route["tiles"] for route in data["routes"]},
    }
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
    """`join(split(x))` is `x`, to the byte.

    It has to be exact, because the packed line is what the archive keeps and
    the submission ships: a program that came back merely equivalent would make
    every stored hash disagree with itself.
    """
    source = packed(PLAN)

    controller, data = plan.split(source)

    assert data == PLAN
    assert "b85decode" not in controller
    assert plan.join(controller, data) == source


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
        ({"shops": [{"shops": ["BAKERY", "BAKERY"], "route": 999}]}, "does not hold"),
        # And the rest, which the types carry, so the message names the path.
        ({"actions": [{"farmer": ["TELEPORT"], "hands": [], "market": []}]}, "farmer"),
        (
            {"shops": [{"shops": ["BAKERY", "BAKERY", "BAKERY"], "route": 101}]},
            "shops",
        ),
        ({"actions": [{"farmer": [], "hands": [], "market": []}]}, "farmer"),
    ):
        with pytest.raises(ValueError, match=says):
            plan.validated({**PLAN, **wrong})


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
    assert "HARVEST" in defs["PlainUnitOp"]["enum"], "the verbs that stand alone"
    assert "TELEPORT" not in defs["PlainUnitOp"]["enum"]
    assert defs["CropName"]["enum"] == config.CROPS, "the engine's own crops"
    assert defs["ProductName"]["enum"] == config.PRODUCTS
    assert "BAKERY" in defs["ShopName"]["enum"], "and the shop names"
    # The verbs that take arguments are shapes rather than members, so they are
    # not in the plain list: `PLANT` only ever appears beside a crop.
    assert "PLANT" not in defs["PlainUnitOp"]["enum"]

    # And the fields reach them, so a generator reading the schema is held to
    # the vocabulary rather than told about it afterwards.
    farmer = json.dumps(defs["Action"]["properties"]["farmer"])
    assert "#/$defs/PlainUnitOp" in farmer and "#/$defs/CropName" in farmer
    assert '"const": "PLANT"' in farmer or '"PLANT"' in farmer
    shops = json.dumps(defs["ShopRoute"]["properties"]["shops"])
    assert "#/$defs/ShopName" in shops
    # Two shops, said in the schema rather than by a validator afterwards.
    assert '"minItems": 2' in shops and '"maxItems": 2' in shops


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
    assert plan.join(controller, data) == source


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
        # the farmer and its hands. The first schema capped this at twelve.
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

    plan.validated({**PLAN, "actions": legal})


def test_the_schema_refuses_what_the_engine_would_never_act_on() -> None:
    """The per-verb argument rules, which `action.rs` applies one arm at a time.

    Each of these parses to `Invalid` or a no-op: the order occupies its slot
    and never executes. A generator that can spell them wastes a season finding
    out, and nothing in the champion's plan does.
    """
    for wrong in (
        # `SELL` takes a product, and an animal is not one.
        step(market=[["SELL", "COW", 1]]),
        # `BUY_PRODUCT` takes wheat or fertiliser, whatever else is an item.
        step(market=[["BUY_PRODUCT", "STRAWBERRY", 1]]),
        # `BUY_SEED` and `PLANT` take crops; eggs are not planted.
        step(market=[["BUY_SEED", "EGG", 1]]),
        step(farmer=["PLANT", "EGG"]),
        # `BUY_ANIMAL` takes an animal.
        step(market=[["BUY_ANIMAL", "WHEAT", 1]]),
        # A verb the engine has no arm for reads as a pass; it is a mistake in
        # the writing rather than a strategy.
        step(farmer=["TELEPORT"]),
        # The moving verbs need their count: two parts is `Invalid`.
        step(market=[["SELL", "WHEAT"]]),
        # And a farm cannot work more hands than the engine has slots for.
        step(hands=[["PASS"]] * config.MAX_UNITS),
    ):
        with pytest.raises(ValueError):
            plan.validated({**PLAN, "actions": [wrong]})
