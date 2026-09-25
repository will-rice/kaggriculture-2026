"""The guard on the campaign's own source while a round runs."""

from pathlib import Path

from kaggriculture.campaign import mutate, workspace


def guarded(tmp_path: Path) -> Path:
    """A checkout-shaped tree with one source file and one prompt in it."""
    root = tmp_path / "checkout"
    (root / "src" / "kaggriculture" / "campaign").mkdir(parents=True)
    here = root / "src" / "kaggriculture" / "campaign"
    (here / "harness.py").write_text("REFERENCE_SAMPLE = 0.02\n", encoding="utf-8")
    (here / "roster.py").write_text("    raise KeyError(name)\n", encoding="utf-8")
    (here / "round_prompt.md").write_text("# how to ask\n", encoding="utf-8")
    (here / "engine.so").write_bytes(b"\x00compiled")
    return root


def a_call(program_id: str = "pdeadbeef") -> mutate.Mutation:
    """A successful call, for the guard to turn down."""
    return mutate.Mutation(
        program_id=program_id,
        child=Path("/tmp/child.py"),
        status="ok",
        reason="",
        seconds=1.0,
        input_tokens=1,
        output_tokens=1,
        model="test",
    )


def test_the_two_edits_a_round_actually_made_are_put_back(tmp_path: Path) -> None:
    """The 2026-09-20 tampering, reproduced exactly.

    A round set `REFERENCE_SAMPLE` to 0.0, turning off the reference-engine
    cross-check, and turned `roster.path`'s `raise KeyError` into a constructed
    path so any string resolves to a file. Both weaken a guard in the direction
    that makes the round's own job easier.
    """
    root = guarded(tmp_path)
    here = root / "src" / "kaggriculture" / "campaign"
    before = workspace.kept(root)
    (here / "harness.py").write_text("REFERENCE_SAMPLE = 0.0\n", encoding="utf-8")
    (here / "roster.py").write_text(
        "    return Path('/data/' + name)\n", encoding="utf-8"
    )

    verdict = workspace.restored(before, a_call(), "pdeadbeef")

    assert (here / "harness.py").read_text(
        encoding="utf-8"
    ) == "REFERENCE_SAMPLE = 0.02\n"
    assert (here / "roster.py").read_text(
        encoding="utf-8"
    ) == "    raise KeyError(name)\n"
    assert verdict.status == "no_output"
    assert verdict.child is None
    assert "harness.py" in verdict.reason and "roster.py" in verdict.reason


def test_the_message_a_round_is_asked_with_is_guarded_too(tmp_path: Path) -> None:
    """A round that rewrites its own instructions has rewritten the objective."""
    root = guarded(tmp_path)
    prompt_file = root / "src" / "kaggriculture" / "campaign" / "round_prompt.md"
    before = workspace.kept(root)
    prompt_file.write_text("# anything goes\n", encoding="utf-8")

    verdict = workspace.restored(before, a_call(), "pdeadbeef")

    assert prompt_file.read_text(encoding="utf-8") == "# how to ask\n"
    assert verdict.status == "no_output"


def test_a_round_that_changes_nothing_is_left_alone(tmp_path: Path) -> None:
    """The guard must not fail every round, which is how it would be noticed."""
    root = guarded(tmp_path)

    verdict = workspace.restored(workspace.kept(root), a_call(), "pdeadbeef")

    assert verdict.status == "ok"
    assert verdict.child is not None


def test_a_deleted_source_file_comes_back(tmp_path: Path) -> None:
    """Removing a guard is as effective as editing it."""
    root = guarded(tmp_path)
    gone = root / "src" / "kaggriculture" / "campaign" / "harness.py"
    before = workspace.kept(root)
    gone.unlink()

    verdict = workspace.restored(before, a_call(), "pdeadbeef")

    assert gone.read_text(encoding="utf-8") == "REFERENCE_SAMPLE = 0.02\n"
    assert verdict.status == "no_output"


def test_the_guard_can_only_write_back_what_it_read(tmp_path: Path) -> None:
    """The bound the git version did not have.

    That one asked git what had changed and reverted the answer, which under a
    pre-commit hook was all 250 tracked files. This holds bytes, so a file it
    never snapshotted -- anything compiled, anything outside `src` -- is one it
    cannot touch however it is called.
    """
    root = guarded(tmp_path)
    before = workspace.kept(root)
    binary = root / "src" / "kaggriculture" / "campaign" / "engine.so"
    outside = tmp_path / "not_in_the_snapshot.py"
    outside.write_text("untouched\n", encoding="utf-8")
    binary.write_bytes(b"\x00changed")

    verdict = workspace.restored(before, a_call(), "pdeadbeef")

    assert binary.read_bytes() == b"\x00changed"
    assert outside.read_text(encoding="utf-8") == "untouched\n"
    assert verdict.status == "ok"
    assert set(before) == {
        root / "src" / "kaggriculture" / "campaign" / name
        for name in ("harness.py", "roster.py", "round_prompt.md")
    }
