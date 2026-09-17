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

from kaggriculture.campaign import mutate, plan

# A program shaped like a champion: imports, a packed plan under a generated
# name, and an agent that reads it.
PLAN = {
    "actions": [{"farmer": ["PASS"], "hands": [], "market": []}],
    "routes": {"0": [1, 2, 3]},
    "shops": [{"shops": ["BAKERY", "BAKERY"], "route": 101}],
}


def packed(data: dict) -> str:
    """A program carrying `data` the way a champion carries its plan."""
    blob = base64.b85encode(
        zlib.compress(json.dumps(data, separators=(",", ":")).encode(), 9)
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


def test_the_real_champion_survives_the_round_trip() -> None:
    """The property, on the program the campaign is actually standing on."""
    champion = Path("run/campaign/floor/agent/main.py")
    if not champion.exists():
        pytest.skip("no live champion in this checkout")
    source = champion.read_text(encoding="utf-8")

    controller, data = plan.split(source)

    assert {*data} == {"actions", "routes", "shops"}
    assert plan.join(controller, data) == source
