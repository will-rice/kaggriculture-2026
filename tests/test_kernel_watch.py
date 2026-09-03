"""The daily scan's own failures are silent, so they are pinned here.

Every defect these cover shipped and went unnoticed for days, because the scan
reports a skip and a low score the same calm way it reports a real result: a
kernel it could not read logged "no verifiable agent payload", and an agent it
could not beat logged a number. Nothing ever raised.
"""

import json
from pathlib import Path

import pytest

from kaggriculture.scripts import kernel_watch

WRITES_A_FILE = """
from pathlib import Path

Path("escaped.csv").write_text("a scanned notebook wrote this")


def agent(observation, configuration=None):
    return {"farmer": ["PASS"], "hands": [], "market": []}
"""


def test_resolving_a_stranger_cannot_write_into_our_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Published notebooks write files when they run, and one of ours did.

    A scan wrote a `submission.csv` into the repository root through the one
    call site that resolved source outside the sandbox; an earlier harvest
    overwrote `main.py`. Resolving is executing, so the guarantee has to hold
    for every caller, not the one that was noticed first.
    """
    monkeypatch.chdir(tmp_path)
    agent_file = tmp_path / "candidate.py"
    agent_file.write_text(WRITES_A_FILE, encoding="utf-8")

    assert kernel_watch.resolved_entrypoint(agent_file) == "agent"
    assert kernel_watch.loadable(WRITES_A_FILE)
    assert not (tmp_path / "escaped.csv").exists()


def test_a_kernel_that_calls_sys_exit_cannot_end_the_scan() -> None:
    """A published kernel must not be able to terminate the scanner.

    One that cannot find its dataset prints a note and calls ``sys.exit``.
    ``SystemExit`` derives from ``BaseException``, so the scan's
    ``except Exception`` did not catch it and a single such kernel killed a
    sweep at 80 of 343 -- with no traceback, and no failed run to show for it.
    """
    source = 'import sys\n\nprint("dataset not attached")\nsys.exit(0)\n'

    assert not kernel_watch.loadable(source)


def test_a_kernel_published_as_a_script_is_read_like_any_other(
    tmp_path: Path,
) -> None:
    """Kaggle publishes scripts as well as notebooks.

    Globbing only for ``*.ipynb`` skipped a whole competitor's v28/v29/v30/v31
    series, reporting it as a missing payload rather than a missing extension.
    """
    (tmp_path / "published.py").write_text(WRITES_A_FILE, encoding="utf-8")

    cells = kernel_watch.published_cells(tmp_path)

    assert cells is not None
    assert "def agent(" in "".join(cells[0]["source"])


def test_a_notebook_is_still_preferred_and_non_code_cells_are_dropped(
    tmp_path: Path,
) -> None:
    """The script path must not shadow a notebook sitting beside it."""
    notebook = {
        "cells": [
            {"cell_type": "markdown", "source": ["prose"]},
            {"cell_type": "code", "source": ["x = 1\n"]},
        ]
    }
    (tmp_path / "book.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
    (tmp_path / "published.py").write_text("y = 2\n", encoding="utf-8")

    cells = kernel_watch.published_cells(tmp_path)

    assert cells is not None
    assert [c["cell_type"] for c in cells] == ["code"]
    assert "".join(cells[0]["source"]) == "x = 1\n"


def test_a_directory_with_neither_form_reads_as_nothing(tmp_path: Path) -> None:
    """A pull that produced no source is not an empty kernel, it is no kernel."""
    assert kernel_watch.published_cells(tmp_path) is None


def test_an_import_of_a_module_the_notebook_writes_is_inlined() -> None:
    """The strongest agent measured here splits itself across two written files.

    Its controller does ``from fieldbook_tapes import PLAN_SCRIPTS`` and a
    second cell writes that module, so concatenating cells leaves an import
    naming a module that exists only in the notebook's own directory.
    """
    source = "from tapes import PLAN\n\ndef agent(observation):\n    return PLAN\n"

    inlined = kernel_watch.inline_written(source, {"tapes": "PLAN = [1, 2]"})

    assert "PLAN = [1, 2]" in inlined
    assert "from tapes import" not in inlined


def test_an_unwritten_import_is_left_alone() -> None:
    """Only the notebook's own modules are inlined; real imports must survive."""
    source = "import json\n\ndef agent(observation):\n    return json\n"

    assert kernel_watch.inline_written(source, {"tapes": "PLAN = []"}) == source


def test_an_entrypoint_bound_at_module_level_is_recognised() -> None:
    """An entrypoint need not be a def.

    The engine's loader takes the last callable a file leaves behind, and the
    strongest agent measured here ends with ``agent = make_agent()``. A
    ``def agent(`` test alone reported it as having no agent at all.
    """
    import re

    assert re.search(kernel_watch.ENTRYPOINT, "agent = make_agent()", re.M)
    assert re.search(kernel_watch.ENTRYPOINT, "def agent(obs):\n    pass", re.M)
    assert not re.search(kernel_watch.ENTRYPOINT, "agents = []", re.M)


WRITES_ON_LOAD = """
from pathlib import Path

Path("main.py").write_text("# a scanned kernel overwrote this")


def agent(observation, configuration=None):
    return {"farmer": ["PASS"], "hands": [], "market": []}
"""


def test_gating_a_stranger_cannot_overwrite_our_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Playing an agent executes it, and a gated candidate is usually a stranger.

    This is not hypothetical: a sweep gated from the repository root overwrote
    `main.py`, our own competition entrypoint, and left four agent files and a
    tarball behind. The scan sandboxes resolution, but gating runs the engine,
    and the engine had our working directory.
    """
    from kaggriculture.scripts import field_gate
    from kaggriculture.search.arena import OutcomeScores

    monkeypatch.chdir(tmp_path)
    ours = tmp_path / "main.py"
    ours.write_text("# our real entrypoint", encoding="utf-8")
    candidate = tmp_path / "candidate.py"
    candidate.write_text(WRITES_ON_LOAD, encoding="utf-8")
    opponent = tmp_path / "opponent.py"
    opponent.write_text(WRITES_ON_LOAD, encoding="utf-8")
    played: list[str] = []

    def record(
        path: str, league: dict[str, str], seeds: object, workers: int
    ) -> OutcomeScores:
        """Stand in for the arena, executing the agents the way it would."""
        played.append(path)
        for source in (path, *league.values()):
            exec(compile(Path(source).read_text(), source, "exec"), {})
        scores = OutcomeScores()
        scores.append(0.5)
        scores.margins.append(0)
        scores.normalized_margins.append(0.0)
        return scores

    monkeypatch.setattr(field_gate, "FIELD", {"only": (str(opponent), 1.0)})
    monkeypatch.setattr("kaggriculture.search.arena.outcomes", record)

    field_gate.score_field(candidate, 1, 1, set())

    assert played and Path(played[0]).is_absolute()
    assert ours.read_text(encoding="utf-8") == "# our real entrypoint"


EMBEDS_AN_AGENT = '''
AGENT_SOURCE = """
def agent(observation, configuration=None):
    return {"farmer": ["PASS"], "hands": [], "market": []}
"""

print(f"Embedded agent: {len(AGENT_SOURCE):,} characters")
open("agent.py", "w").write(AGENT_SOURCE)
'''


def test_an_agent_carried_as_a_plain_string_is_recovered() -> None:
    """Some kernels neither compress their agent nor define it inline.

    One holds it in a raw string, prints its size and writes it out. Taking the
    surrounding cell gets the wrapper, whose last callable is not a policy, so
    the candidate resolved and then failed on turn one -- which the gate
    reported as a crash rather than as a kernel we had misread.
    """
    cells = [{"cell_type": "code", "source": EMBEDS_AN_AGENT}]

    recovered = kernel_watch.embedded_source(cells)

    assert recovered is not None
    assert "def agent(" in recovered
    assert "Embedded agent:" not in recovered


def test_a_cell_with_no_embedded_agent_yields_nothing() -> None:
    """A string that merely mentions an agent is not an agent."""
    cells = [{"cell_type": "code", "source": 'NOTE = "the agent is elsewhere"\n'}]

    assert kernel_watch.embedded_source(cells) is None


def test_a_notebooks_own_compiler_line_is_read_as_a_literal() -> None:
    """The top agent in this competition is C++ and must be built to be gated.

    Its notebook assigns the command as a list of string constants and hands it
    to subprocess. Reading the literal keeps the build the author's own while
    still never evaluating a cell to discover it.
    """
    cell = (
        "import subprocess\n"
        'command = ["g++", "-O3", "-shared", "-fPIC", "-o", "agent.so", "p.cpp"]\n'
        "subprocess.run(command, check=True)\n"
    )

    assert kernel_watch.compiler_command(cell) == [
        "g++",
        "-O3",
        "-shared",
        "-fPIC",
        "-o",
        "agent.so",
        "p.cpp",
    ]


def test_a_compiler_line_with_a_computed_argument_is_declined() -> None:
    """A partially-literal command is the dangerous case, not the obvious one.

    A list holding a name among its constants still parses as a list, and
    reading it would silently drop that argument and build with a command the
    author never wrote. Dropping ``-o agent.so``'s target, or an include path,
    yields a binary that is not the published agent.
    """
    cell = 'command = ["g++", flag, "-o", "agent.so", "p.cpp"]\n'

    assert kernel_watch.compiler_command(cell) is None


def test_a_command_assembled_by_concatenation_is_declined() -> None:
    """A cell that builds its arguments cannot be read without running it."""
    cell = 'flags = ["-O3"]\ncommand = ["g++"] + flags\n'

    assert kernel_watch.compiler_command(cell) is None


def test_a_cell_that_runs_no_compiler_yields_nothing() -> None:
    """Only a compiler invocation counts; ordinary lists must not match."""
    cell = 'names = ["alpha", "beta"]\n'

    assert kernel_watch.compiler_command(cell) is None
