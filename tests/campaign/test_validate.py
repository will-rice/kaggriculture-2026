"""Everything that rejects a candidate before a game is scored."""

from pathlib import Path

from kaggriculture.campaign import roster, validate

GOOD = """
import math
from kaggriculture.constants import CROPS

def agent(observation, configuration=None):
    return {"farmer": ["PASS"], "hands": [], "market": []}
"""

SLOW_AGENT = """
import time
def agent(o, c=None):
    time.sleep(0.6)
    return {'farmer': ['PASS'], 'hands': [], 'market': []}
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


def test_copied_opponent(tmp_path: Path) -> None:
    """A wholesale copy of an opponent is caught, and named, before the harness.

    v56 also fails the import and shadowed-entrypoint checks, so this only
    holds if the copy check runs first.
    """
    verdict = validate.validate(
        write(tmp_path, roster.path("v56").read_text(encoding="utf-8"))
    )
    assert verdict.status == "copy" and "v56" in verdict.reason


def test_crash(tmp_path: Path) -> None:
    """An agent that raises on its first call is `crashed`, not `ok`."""
    verdict = validate.validate(
        write(tmp_path, "def agent(o, c=None):\n    raise ValueError('x')\n")
    )
    assert verdict.status == "crashed"


def test_too_slow(tmp_path: Path) -> None:
    """An agent slower than the latency budget is `too_slow`, not `ok`."""
    verdict = validate.validate(write(tmp_path, SLOW_AGENT), steps=3)
    assert verdict.status == "too_slow"
