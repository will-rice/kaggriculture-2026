"""The mutation call, with a fake for every test that is not about codex itself."""

import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import IO

import pytest

from kaggriculture.campaign import mutate


def sandbox(tmp_path: Path) -> Path:
    """A minimal sandbox: a parent, a prompt, and standing agent rules."""
    (tmp_path / "parent.py").write_text(
        "LIMIT = 1\n"
        "def agent(o, c=None):\n"
        "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
    )
    (tmp_path / "PROMPT.md").write_text(
        "Write child.py implementing def agent(observation, configuration=None).\n"
    )
    (tmp_path / "AGENTS.md").write_text(
        "Write child.py implementing def agent(observation, configuration=None).\n"
        "Keep it a complete, self-contained agent file.\n"
    )
    return tmp_path


def test_fake_mutator_writes_a_child_with_the_edit_applied(tmp_path: Path) -> None:
    """FakeMutator copies parent.py to child.py through the caller's edit."""
    box = sandbox(tmp_path)
    mutator = mutate.FakeMutator(edit=lambda s: s.replace("LIMIT = 1", "LIMIT = 2"))
    result = mutator(box, "p1")
    assert result.status == "ok" and result.child == box / "child.py"
    assert result.child is not None
    assert "LIMIT = 2" in result.child.read_text()


def test_codex_mutator_reports_no_output_when_nothing_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A codex that exits clean but writes no child.py is "no_output", not "ok"."""
    box = sandbox(tmp_path)
    monkeypatch.setattr(
        mutate.CodexMutator, "COMMAND", ["true"]
    )  # a codex that says nothing
    result = mutate.CodexMutator()(box, "p2")
    assert result.status == "no_output"
    assert (box / "codex.jsonl").exists()


def test_codex_mutator_reports_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A codex call that outlives its timeout is reported as "timeout"."""
    box = sandbox(tmp_path)
    monkeypatch.setattr(mutate.CodexMutator, "COMMAND", ["sleep", "5"])
    result = mutate.CodexMutator(timeout=1)(box, "p3")
    assert result.status == "timeout"
    assert (box / "codex.jsonl").exists()


def test_a_timed_out_session_still_yields_the_child_it_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cap bounds the session's wall clock, not whether its work counts.

    Live sessions write a complete child within minutes and then spend as
    long again testing it; killing the session must not throw that away.
    """
    box = sandbox(tmp_path)
    monkeypatch.setattr(
        mutate.CodexMutator,
        "COMMAND",
        [
            "bash",
            "-c",
            "printf 'def agent(o, c=None):\\n    return {}\\n' > child.py; sleep 30",
        ],
    )
    result = mutate.CodexMutator(timeout=2)(box, "p8")
    assert result.status == "timeout"
    assert result.child == box / "child.py"
    assert "def agent" in (box / "child.py").read_text()


def test_codex_mutator_reports_exec_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A codex process that exits non-zero is reported as "exec_error"."""
    box = sandbox(tmp_path)
    monkeypatch.setattr(
        mutate.CodexMutator, "COMMAND", ["sh", "-c", "echo boom 1>&2; exit 1"]
    )
    result = mutate.CodexMutator()(box, "p6")
    assert result.status == "exec_error"
    assert "boom" in result.reason
    assert (box / "codex.jsonl").exists()


def test_codex_mutator_no_output_reason_carries_the_last_agent_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A no_output reason names the last thing codex said, and tokens still parse."""
    box = sandbox(tmp_path)

    def fake_popen(
        command: list[str],
        stdin: int,
        stdout: IO[str],
        stderr: int,
        text: bool,
        cwd: Path,
        start_new_session: bool,
    ) -> SimpleNamespace:
        stdout.write(
            '{"type":"thread.started","thread_id":"t"}\n'
            '{"type":"item.completed","item":{"id":"i1","type":"agent_message",'
            '"text":"I have a question about the interface."}}\n'
            '{"type":"turn.completed","usage":{"input_tokens":10,"cached_input_tokens":0,'
            '"cache_write_input_tokens":0,"output_tokens":5,"reasoning_output_tokens":0}}\n'
        )
        return SimpleNamespace(
            pid=99999, returncode=0, communicate=lambda input, timeout: (None, "")
        )

    monkeypatch.setattr(mutate.subprocess, "Popen", fake_popen)
    result = mutate.CodexMutator()(box, "p5")
    assert result.status == "no_output"
    assert "question about the interface" in result.reason
    assert result.input_tokens == 10 and result.output_tokens == 5


def test_codex_mutator_kills_the_whole_process_group_on_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A timeout kills the process group, not just codex, so no orphan survives."""
    box = sandbox(tmp_path)
    monkeypatch.setattr(
        mutate.CodexMutator, "COMMAND", ["bash", "-c", "sleep 30 & sleep 30"]
    )
    result = mutate.CodexMutator(timeout=1)(box, "p7")
    assert result.status == "timeout"
    pgid = int(result.reason.split("pgid ")[1].rstrip(")"))
    for _ in range(20):
        survivors = subprocess.run(
            ["pgrep", "-g", str(pgid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if survivors.returncode != 0:
            break
        time.sleep(0.1)
    else:
        pytest.fail(f"process group {pgid} still has a member after the timeout kill")


@pytest.mark.skipif(
    not os.environ.get("CAMPAIGN_CODEX"),
    reason="set CAMPAIGN_CODEX=1 to spend a real codex call",
)
def test_real_codex_writes_a_child(tmp_path: Path) -> None:
    """One real codex call turns a trivial parent into a working child.py."""
    box = sandbox(tmp_path)
    # codex refuses to run outside a trusted directory; production sandboxes
    # already live inside this repo's own trusted worktree, so make this
    # tmp_path one too rather than relaxing the mutator's real command.
    subprocess.run(["git", "init", "-q"], cwd=box, check=True)
    result = mutate.CodexMutator()(box, "p4")
    assert result.status == "ok"
    assert result.child is not None
    assert "def agent" in (box / "child.py").read_text()
    assert result.input_tokens > 0 and result.output_tokens > 0
