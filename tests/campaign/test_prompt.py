"""What a sandbox contains, and what it must never contain."""

import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from kaggriculture.campaign import config, prompt

IMPROVE = prompt.INSTRUCTIONS[0][1]


@pytest.fixture(autouse=True)
def restore_sandboxes() -> Iterator[None]:
    """Put ``config.SANDBOXES`` back after a test has pointed it at a tmp_path."""
    original = config.SANDBOXES
    yield
    config.SANDBOXES = original


def monkeypatched_sandbox(tmp_path: Path) -> Path:
    """Point ``config.SANDBOXES`` at a directory under ``tmp_path``.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        The sandboxes directory the campaign will build under.
    """
    config.SANDBOXES = tmp_path / "sandboxes"
    config.SANDBOXES.mkdir(exist_ok=True)
    return config.SANDBOXES


def test_the_sandbox_holds_the_champion_as_child_py(tmp_path: Path) -> None:
    """A session edits the champion in place; there is no parent.py."""
    monkeypatched_sandbox(tmp_path)
    champion = tmp_path / "champion.py"
    champion.write_text("def agent(o, c=None):\n    return {'x': 1}\n")

    box = prompt.build_sandbox(
        "p1",
        champion,
        IMPROVE,
        None,
        bar={"fitness": 0.42},
        rates={"v54": 0.3},
        weights={"v54": 1.0},
        weakest="v54",
        failures=[],
        started_from="champion_1",
    )

    assert (box / "child.py").read_text() == champion.read_text()
    assert not (box / "parent.py").exists()


def test_the_feedback_states_the_bar(tmp_path: Path) -> None:
    """The session is told the number it has to beat."""
    monkeypatched_sandbox(tmp_path)
    champion = tmp_path / "champion.py"
    champion.write_text("def agent(o, c=None):\n    return {}\n")

    box = prompt.build_sandbox(
        "p2",
        champion,
        IMPROVE,
        None,
        bar={"fitness": 0.42},
        rates={"v54": 0.3},
        weights={"v54": 1.0},
        weakest="v54",
        failures=[],
        started_from="champion_1",
    )

    feedback = (box / "feedback.md").read_text()
    assert "0.42" in feedback and "v54" in feedback


def test_there_are_five_full_instructions_and_they_differ() -> None:
    """FAMOU C.2's variants, so eight sessions on one champion diverge."""
    names = [name for name, _ in prompt.INSTRUCTIONS]
    texts = [text for _, text in prompt.INSTRUCTIONS]
    assert len(prompt.INSTRUCTIONS) == 5
    assert len(set(names)) == 5 and len(set(texts)) == 5
    assert all("child.py" in text for text in texts)


def test_sandbox_has_every_required_file_and_no_opponent_path(tmp_path: Path) -> None:
    """Every required file exists, and no opponent path leaks anywhere in it."""
    monkeypatched_sandbox(tmp_path)
    champion = tmp_path / "p.py"
    champion.write_text(
        "def agent(o, c=None):\n"
        "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
    )
    box = prompt.build_sandbox(
        "abc",
        champion,
        IMPROVE,
        None,
        bar={"fitness": 0.5},
        rates={"v54": 0.1, "router_v1": 0.0},
        weights={"v54": 0.5, "router_v1": 0.5},
        weakest="router_v1",
        failures=["syntax: bad"],
        started_from="champion_1",
    )
    for name in (
        "AGENTS.md",
        "child.py",
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


def test_cross_sandbox_carries_the_inspiration(tmp_path: Path) -> None:
    """A crossover session's sandbox carries inspiration.py and names it."""
    monkeypatched_sandbox(tmp_path)
    champion = tmp_path / "p.py"
    champion.write_text("# champion\n")
    other = tmp_path / "q.py"
    other.write_text("# inspiration\n")
    box = prompt.build_sandbox(
        "xyz",
        champion,
        prompt.CROSS,
        other,
        bar={"fitness": 0.5},
        rates={},
        weights={},
        weakest="",
        failures=[],
        started_from="champion_1",
    )
    assert (box / "inspiration.py").read_text() == "# inspiration\n"
    assert "inspiration.py" in (box / "PROMPT.md").read_text()


def test_prompt_states_the_instruction_the_bar_and_the_budget(tmp_path: Path) -> None:
    """PROMPT.md records the drawn instruction and the session's budget."""
    monkeypatched_sandbox(tmp_path)
    champion = tmp_path / "p.py"
    champion.write_text("# champion\n")
    box = prompt.build_sandbox(
        "drawn",
        champion,
        prompt.INSTRUCTIONS[4][1],
        None,
        bar={"fitness": 0.5},
        rates={},
        weights={},
        weakest="",
        failures=[],
        started_from="champion_1",
    )
    text = (box / "PROMPT.md").read_text()
    assert prompt.INSTRUCTIONS[4][1] in text
    assert prompt.INSTRUCTIONS[0][1] not in text
    assert f"{config.SESSION_LIMIT_SECONDS // 60} minutes" in text


def test_agents_md_is_the_task_prompt_plus_harness_and_doctrine(tmp_path: Path) -> None:
    """AGENTS.md carries the harness commands and the never-read-source doctrine."""
    monkeypatched_sandbox(tmp_path)
    champion = tmp_path / "p.py"
    champion.write_text("# champion\n")
    box = prompt.build_sandbox(
        "q",
        champion,
        IMPROVE,
        None,
        bar={"fitness": 0.5},
        rates={},
        weights={},
        weakest="",
        failures=[],
        started_from="champion_1",
    )
    text = (box / "AGENTS.md").read_text()
    assert "campaign play" in text and "never read their source" in text.lower()
    assert "700000" not in text


def test_agents_md_strips_the_stale_harness_block_and_rewrites_the_project_path(
    tmp_path: Path,
) -> None:
    """Phase 1's harness block is hard-coded to this worktree; it must not leak."""
    monkeypatched_sandbox(tmp_path)
    champion = tmp_path / "p.py"
    champion.write_text("# champion\n")
    box = prompt.build_sandbox(
        "r",
        champion,
        IMPROVE,
        None,
        bar={"fitness": 0.5},
        rates={},
        weights={},
        weakest="",
        failures=[],
        started_from="champion_1",
    )
    text = (box / "AGENTS.md").read_text()
    root = str(config.ROOT)
    assert text.count(f"uv run --project {root} campaign check") == 1
    assert text.count(f"uv run --project {root} campaign play") == 1
    # No other absolute path appears anywhere in AGENTS.md.
    sanitized = text.replace(root, "<ROOT>")
    assert not re.search(r"/(?:home|data|Users|tmp)/\S*", sanitized)
    assert "Valid measured opponents" not in text
    assert "700000" not in text


def test_every_instruction_forbids_reading_opponent_source(tmp_path: Path) -> None:
    """The doctrine is wrapped around every instruction, crossover included."""
    monkeypatched_sandbox(tmp_path)
    champion = tmp_path / "p.py"
    champion.write_text("# champion\n")
    other = tmp_path / "q.py"
    other.write_text("# inspiration\n")
    forbidden = "do not read, request, or reconstruct any opponent's source"

    for index, (name, text) in enumerate(prompt.INSTRUCTIONS):
        box = prompt.build_sandbox(
            f"full-{index}",
            champion,
            text,
            None,
            bar={"fitness": 0.5},
            rates={},
            weights={},
            weakest="",
            failures=[],
            started_from="champion_1",
        )
        assert forbidden in (box / "PROMPT.md").read_text().lower(), name

    cross_box = prompt.build_sandbox(
        "cross-id",
        champion,
        prompt.CROSS,
        other,
        bar={"fitness": 0.5},
        rates={},
        weights={},
        weakest="",
        failures=[],
        started_from="champion_1",
    )
    assert forbidden in (cross_box / "PROMPT.md").read_text().lower()


def test_program_id_path_traversal_is_rejected(tmp_path: Path) -> None:
    """A crafted program_id can never resolve outside the sandboxes directory."""
    monkeypatched_sandbox(tmp_path)
    champion = tmp_path / "p.py"
    champion.write_text("# champion\n")
    sentinel = tmp_path / "sentinel-must-survive.txt"
    sentinel.write_text("still here\n")
    for bad_id in ("..", "../x", "a/b", ""):
        with pytest.raises(ValueError):
            prompt.build_sandbox(
                bad_id,
                champion,
                IMPROVE,
                None,
                bar={"fitness": 0.5},
                rates={},
                weights={},
                weakest="",
                failures=[],
                started_from="champion_1",
            )
    assert sentinel.read_text() == "still here\n"


def test_the_child_is_writable_even_when_the_champion_is_not(tmp_path: Path) -> None:
    """The gate writes a champion read-only; the file a session edits cannot be.

    `gate.promote` chmods `CHAMPIONS/<name>.py` to 0o444 so nothing can edit
    what the pool plays, and `shutil.copy` carries the mode across. A session
    handed a read-only `child.py` cannot do the one thing it is asked to do.
    """
    champion = tmp_path / "champion_1.py"
    champion.write_text("def agent(o, c=None):\n    return {}\n", encoding="utf-8")
    champion.chmod(0o444)

    box = prompt.build_sandbox(
        "p-readonly",
        champion,
        prompt.INSTRUCTIONS[0][1],
        None,
        bar={"fitness": 0.5},
        rates={"pass": 0.5},
        weights={"pass": 1.0},
        weakest="pass",
        failures=[],
        started_from="champion_1",
    )

    child = box / "child.py"
    child.write_text("edited\n", encoding="utf-8")
    assert child.read_text() == "edited\n"
