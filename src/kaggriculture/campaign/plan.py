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
    return join(source, json.loads(plan_file.read_text(encoding="utf-8")))
