"""Everything that rejects a candidate before a game is scored."""

import time
from pathlib import Path

import pytest

from kaggriculture.campaign import harness, roster, validate

GOOD = """
import math

def agent(observation, configuration=None):
    return {"farmer": ["PASS"], "hands": [], "market": []}
"""

SLOW_AGENT = """
import time
def agent(o, c=None):
    time.sleep(0.6)
    return {'farmer': ['PASS'], 'hands': [], 'market': []}
"""

# Loading a module runs its top-level code. Nothing about this file is
# syntactically wrong and `agent` is still the last callable defined, so every
# static check passes and the wedge only shows up when the file is loaded.
WEDGING_AGENT = """
def agent(o, c=None):
    return {'farmer': ['PASS'], 'hands': [], 'market': []}


while True:
    pass
"""

# The candidate writes at import time, where a scratch cwd is the only thing
# between it and the caller's working directory.
WRITING_AGENT = """
from pathlib import Path

Path('scribble.txt').write_text('x', encoding='utf-8')


def agent(o, c=None):
    return {'farmer': ['PASS'], 'hands': [], 'market': []}
"""

# Exits while loading: a Python-level failure with no verdict to report, which
# must read as a crash the moment the child is gone, not as slowness after the
# whole cap has elapsed.
EXITING_AGENT = """
def agent(o, c=None):
    return {'farmer': ['PASS'], 'hands': [], 'market': []}


raise SystemExit(3)
"""


def write(tmp_path: Path, source: str) -> Path:
    """Write `source` as `main.py` under `tmp_path` and return its path."""
    path = tmp_path / "main.py"
    path.write_text(source, encoding="utf-8")
    return path


def test_a_good_agent_is_ok(tmp_path: Path) -> None:
    """A well-formed, original, fast agent passes every check."""
    verdict = validate.validate(write(tmp_path, GOOD))
    assert verdict.status == "ok", verdict.reason


def test_syntax_error(tmp_path: Path) -> None:
    """A file that doesn't parse fails before anything else runs."""
    assert validate.validate(write(tmp_path, "def agent(:\n")).status == "syntax"


def test_missing_entrypoint_is_a_contract_failure(tmp_path: Path) -> None:
    """No top-level function named `agent` at all is a contract failure."""
    assert validate.validate(write(tmp_path, "x = 1\n")).status == "contract"


def test_agent_must_be_the_last_top_level_callable(tmp_path: Path) -> None:
    """A trailing helper would be the one Kaggle's runner actually calls."""
    source = GOOD + "\ndef zzz():\n    pass\n"
    verdict = validate.validate(write(tmp_path, source))
    assert verdict.status == "contract" and "last" in verdict.reason


def test_forbidden_import(tmp_path: Path) -> None:
    """An import outside `ALLOWED_IMPORTS` is rejected by name."""
    verdict = validate.validate(write(tmp_path, "import socket\n" + GOOD))
    assert verdict.status == "imports" and "socket" in verdict.reason


def test_the_kaggriculture_package_is_not_importable(tmp_path: Path) -> None:
    """Only `main.py` ships, so a program importing our package dies on Kaggle.

    It used to be allowed, back when the archive carried four plumbing
    modules. A candidate that still imports one would pass every check here
    and fail on the ladder, which is worse than failing in the campaign.
    """
    source = "from kaggriculture.actions import PASS\n" + GOOD
    verdict = validate.validate(write(tmp_path, source))
    assert verdict.status == "imports" and "kaggriculture.actions" in verdict.reason


def test_ctypes_is_not_importable(tmp_path: Path) -> None:
    """`ctypes` was for the engine library, and the archive no longer has one."""
    verdict = validate.validate(write(tmp_path, "import ctypes\n" + GOOD))
    assert verdict.status == "imports" and "ctypes" in verdict.reason


def test_dunder_import_bypasses_the_import_statement_check(tmp_path: Path) -> None:
    """`__import__` reaches a module without ever naming it in an import."""
    source = GOOD + '\nsock = __import__("socket")\n'
    verdict = validate.validate(write(tmp_path, source))
    assert verdict.status == "imports" and "__import__" in verdict.reason


def test_eval_is_a_forbidden_call(tmp_path: Path) -> None:
    """`eval` can build and run an `__import__` call from a string."""
    source = GOOD + "\nresult = eval('1')\n"
    verdict = validate.validate(write(tmp_path, source))
    assert verdict.status == "imports" and "eval" in verdict.reason


def test_trailing_class_shadows_the_entrypoint(tmp_path: Path) -> None:
    """A trailing top-level `class` is also a callable that would run instead."""
    source = GOOD + "\n\nclass Zzz:\n    pass\n"
    verdict = validate.validate(write(tmp_path, source))
    assert verdict.status == "contract" and "last" in verdict.reason


def test_trailing_lambda_shadows_the_entrypoint(tmp_path: Path) -> None:
    """A trailing name bound to a `lambda` shadows `agent` the same way a def would."""
    source = GOOD + "\nhelper = lambda o, c=None: {'farmer': ['PASS']}\n"
    verdict = validate.validate(write(tmp_path, source))
    assert verdict.status == "contract" and "last" in verdict.reason


def test_trailing_async_def_shadows_the_entrypoint(tmp_path: Path) -> None:
    """A trailing top-level `async def` is invisible to a `FunctionDef`-only check."""
    source = GOOD + "\nasync def zzz():\n    pass\n"
    verdict = validate.validate(write(tmp_path, source))
    assert verdict.status == "contract" and "last" in verdict.reason


@pytest.mark.local_data
def test_copied_opponent(tmp_path: Path) -> None:
    """A wholesale copy of an opponent is caught, and named, before the harness.

    v56 also fails the import and shadowed-entrypoint checks, so this only
    holds if the copy check runs first.
    """
    verdict = validate.validate(
        write(tmp_path, roster.path("v56").read_text(encoding="utf-8"))
    )
    assert verdict.status == "copy" and "v56" in verdict.reason


@pytest.mark.local_data
def test_copied_opponent_reason_never_names_a_path(tmp_path: Path) -> None:
    """The copy verdict names the roster key, never the file it lives at."""
    verdict = validate.validate(
        write(tmp_path, roster.path("v56").read_text(encoding="utf-8"))
    )
    assert "/data" not in verdict.reason


def test_crash(tmp_path: Path) -> None:
    """An agent that raises on its first call is `crashed`, not `ok`."""
    verdict = validate.validate(
        write(tmp_path, "def agent(o, c=None):\n    raise ValueError('x')\n")
    )
    assert verdict.status == "crashed"


def test_too_slow(tmp_path: Path) -> None:
    """An agent slower than the latency budget is `too_slow`, not `ok`.

    The one clock the campaign still keeps, and it is not the campaign's: it
    is `harness.LATENCY_BUDGET`, half of the one second `actTimeout` Kaggle
    enforces per call. A program over it does not run on Kaggle, so promoting
    one would ship an agent that cannot compete. How long validation takes
    overall is not bounded; how long a single step takes is.
    """
    verdict = validate.validate(write(tmp_path, SLOW_AGENT), steps=3)
    assert verdict.status == "too_slow"
    assert verdict.worst_step_seconds > harness.LATENCY_BUDGET


def test_what_a_candidate_writes_while_loading_lands_in_a_scratch_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loading a candidate runs it, so validation is an execution path too."""
    agent = write(tmp_path, WRITING_AGENT)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)

    assert validate.validate(agent, steps=3).status == "ok"
    assert list(workspace.iterdir()) == []


def test_a_candidate_that_exits_while_loading_is_a_crash_not_a_wait(
    tmp_path: Path,
) -> None:
    """A dead child is reported at once and as what it was.

    Nothing caps how long validation may take any more, so this is the only
    thing between a child that will never report and a worker waiting on a
    queue for the life of the campaign: the poll notices the process is gone
    and says what happened to it.
    """
    started = time.monotonic()

    verdict = validate.validate(write(tmp_path, EXITING_AGENT))

    assert verdict.status == "crashed" and "exited with code 3" in verdict.reason
    assert time.monotonic() - started < 15
