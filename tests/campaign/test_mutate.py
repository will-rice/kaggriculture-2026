"""The mutation call, with a fake for every test that is not about codex itself.

Every codex stand-in here is a real subprocess -- ``true``, ``sleep``, a
``bash`` line -- because what these tests are about is what the mutator does
to a live process group, which no fake process could show.
"""

import asyncio
import os
import subprocess
import time
from pathlib import Path

import pytest

from kaggriculture.campaign import mutate

# A codex that says nothing but a well-formed transcript, so the no-output
# path can be read without spending a call.
TRANSCRIPT = (
    '{"type":"thread.started","thread_id":"t"}\n'
    '{"type":"item.completed","item":{"id":"i1","type":"agent_message",'
    '"text":"I have a question about the interface."}}\n'
    '{"type":"turn.completed","usage":{"input_tokens":10,"cached_input_tokens":0,'
    '"cache_write_input_tokens":0,"output_tokens":5,"reasoning_output_tokens":0}}\n'
)


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
    result = asyncio.run(mutator(box, "p1"))
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
    result = asyncio.run(mutate.CodexMutator()(box, "p2"))
    assert result.status == "no_output"
    assert (box / "codex.jsonl").exists()


def test_codex_mutator_reports_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A codex call that outlives its timeout is reported as "timeout"."""
    box = sandbox(tmp_path)
    monkeypatch.setattr(mutate.CodexMutator, "COMMAND", ["sleep", "5"])
    result = asyncio.run(mutate.CodexMutator(timeout=1)(box, "p3"))
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
    result = asyncio.run(mutate.CodexMutator(timeout=2)(box, "p8"))
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
    result = asyncio.run(mutate.CodexMutator()(box, "p6"))
    assert result.status == "exec_error"
    assert "boom" in result.reason
    assert (box / "codex.jsonl").exists()


def test_codex_mutator_no_output_reason_carries_the_last_agent_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A no_output reason names the last thing codex said, and tokens still parse."""
    box = sandbox(tmp_path)
    # `printf %s` writes its argument verbatim, so the transcript reaches
    # `codex.jsonl` exactly as a real session would have written it.
    monkeypatch.setattr(mutate.CodexMutator, "COMMAND", ["printf", "%s", TRANSCRIPT])
    result = asyncio.run(mutate.CodexMutator()(box, "p5"))
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
    result = asyncio.run(mutate.CodexMutator(timeout=1)(box, "p7"))
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


def test_cancelling_a_call_kills_the_whole_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shutting the loop down takes the session with it, grandchildren included.

    A cancelled call that only closed its pipes would leave codex running
    against a sandbox nothing owns, holding cores the budget counts as free;
    a restart would then race eight orphans.
    """
    box = sandbox(tmp_path)
    pgid_file = tmp_path / "pgid"
    monkeypatch.setattr(
        mutate.CodexMutator,
        "COMMAND",
        [
            "bash",
            "-c",
            f"ps -o pgid= -p $$ | tr -d ' ' > {pgid_file}; sleep 30 & sleep 30",
        ],
    )

    async def cancel_mid_session() -> None:
        """Start the call, wait for the group to exist, then cancel it."""
        call = asyncio.ensure_future(mutate.CodexMutator(timeout=30)(box, "p9"))
        while not pgid_file.exists():
            await asyncio.sleep(0.05)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call

    asyncio.run(cancel_mid_session())

    pgid = int(pgid_file.read_text().strip())
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
        pytest.fail(f"process group {pgid} survived the cancellation")


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
    result = asyncio.run(mutate.CodexMutator()(box, "p4"))
    assert result.status == "ok"
    assert result.child is not None
    assert "def agent" in (box / "child.py").read_text()
    assert result.input_tokens > 0 and result.output_tokens > 0
