"""What a sandbox contains, and what it must never contain."""

import re
from pathlib import Path

import pytest

from kaggriculture.campaign import config, prompt


def test_sandbox_has_every_required_file_and_no_opponent_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every required file exists, and no opponent path leaks anywhere in it."""
    monkeypatch.setattr(config, "SANDBOXES", tmp_path)
    parent = tmp_path / "p.py"
    parent.write_text(
        "def agent(o, c=None):\n"
        "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
    )
    box = prompt.build_sandbox(
        "abc",
        "full",
        parent,
        None,
        {"v54": 0.1, "router_v1": 0.0},
        "router_v1",
        {"v54": 0.5, "router_v1": 0.5},
        ["syntax: bad"],
    )
    for name in (
        "AGENTS.md",
        "parent.py",
        "feedback.md",
        "engine/kaggriculture.py",
        "PROMPT.md",
    ):
        assert (box / name).exists(), name
    everything = "".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in box.rglob("*")
        if p.is_file()
    )
    assert "/data/kaggriculture" not in everything
    feedback = (box / "feedback.md").read_text()
    assert "router_v1" in feedback and "weakest" in feedback.lower()
    assert "syntax: bad" in feedback


def test_cross_sandbox_carries_the_inspiration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cross mutation's sandbox carries inspiration.py and names it in PROMPT.md."""
    monkeypatch.setattr(config, "SANDBOXES", tmp_path)
    parent = tmp_path / "p.py"
    parent.write_text("# parent\n")
    other = tmp_path / "q.py"
    other.write_text("# inspiration\n")
    box = prompt.build_sandbox("xyz", "cross", parent, other, {}, "", {}, [])
    assert (box / "inspiration.py").read_text() == "# inspiration\n"
    assert "inspiration.py" in (box / "PROMPT.md").read_text()


def test_agents_md_is_the_task_prompt_plus_harness_and_doctrine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AGENTS.md carries the harness commands and the never-read-source doctrine."""
    monkeypatch.setattr(config, "SANDBOXES", tmp_path)
    parent = tmp_path / "p.py"
    parent.write_text("# parent\n")
    box = prompt.build_sandbox("q", "full", parent, None, {}, "", {}, [])
    text = (box / "AGENTS.md").read_text()
    assert "campaign play" in text and "never read their source" in text.lower()
    assert "700000" not in text


def test_agents_md_strips_the_stale_harness_block_and_rewrites_the_project_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 1's harness block is hard-coded to this worktree; it must not leak."""
    monkeypatch.setattr(config, "SANDBOXES", tmp_path)
    parent = tmp_path / "p.py"
    parent.write_text("# parent\n")
    box = prompt.build_sandbox("r", "full", parent, None, {}, "", {}, [])
    text = (box / "AGENTS.md").read_text()
    root = str(config.ROOT)
    assert text.count(f"uv run --project {root} campaign check") == 1
    assert text.count(f"uv run --project {root} campaign play") == 1
    # No other absolute path appears anywhere in AGENTS.md.
    sanitized = text.replace(root, "<ROOT>")
    assert not re.search(r"/(?:home|data|Users|tmp)/\S*", sanitized)
    assert "Valid measured opponents" not in text
    assert "700000" not in text
