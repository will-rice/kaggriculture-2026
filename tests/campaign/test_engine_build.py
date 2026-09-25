"""The engine library builds, exports the ABI, and lands whole or not at all."""

import ctypes
import multiprocessing
import subprocess
from pathlib import Path

import pytest

from kaggriculture.campaign.engine import build


def test_build_produces_a_loadable_library_with_abi_version_1() -> None:
    """The compiled bridge loads and reports ABI version 1 and the pinned engine."""
    path = build.build()
    assert path == build.ENGINE_LIBRARY and path.exists()
    library = ctypes.CDLL(str(path))
    library.kag_abi_version.restype = ctypes.c_uint32
    assert library.kag_abi_version() == 1
    library.kag_engine_version.restype = ctypes.c_char_p
    assert library.kag_engine_version() == b"1.32.7"


def test_a_compile_is_never_visible_at_the_target_until_it_is_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The compiler writes to a private path; the target moves into place at once.

    ``g++`` fills its output file incrementally, so a worker that ``dlopen``ed
    the target while another worker was compiling straight to it would load a
    truncated library. The compiler here is a stand-in that writes twice, and
    what is asserted is that the target does not exist between the two
    writes -- which only holds if the real compile goes somewhere else first.
    """
    target = tmp_path / "engine.so"
    monkeypatch.setattr(build, "ENGINE_LIBRARY", target)
    midway: list[bytes] = []

    def fake_compile(command: list[str], check: bool) -> subprocess.CompletedProcess:
        """Write the output in two halves, looking at the target in between."""
        out = Path(command[command.index("-o") + 1])
        assert out != target, "compiled straight to the target"
        out.write_bytes(b"partial")
        midway.append(target.read_bytes() if target.exists() else b"")
        out.write_bytes(b"a whole library")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(build.subprocess, "run", fake_compile)

    assert build.build() == target
    assert target.read_bytes() == b"a whole library"
    assert midway == [b""]
    assert list(tmp_path.glob("*.tmp")) == []


def test_a_failed_compile_leaves_no_scratch_library_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The private output path is cleaned up when the compiler fails."""
    target = tmp_path / "engine.so"
    monkeypatch.setattr(build, "ENGINE_LIBRARY", target)

    def failing_compile(command: list[str], check: bool) -> subprocess.CompletedProcess:
        """Write a partial output, then fail as g++ would."""
        Path(command[command.index("-o") + 1]).write_bytes(b"partial")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(build.subprocess, "run", failing_compile)

    with pytest.raises(subprocess.CalledProcessError):
        build.build()
    assert list(tmp_path.iterdir()) == []


def _build_into(target: Path) -> None:
    """Point this process's config at ``target`` and build. Runs after a fork."""
    build.ENGINE_LIBRARY = target
    build.build()


def test_two_processes_building_at_once_both_succeed(tmp_path: Path) -> None:
    """Every evaluation forks workers that each load the library on a cold tree."""
    target = tmp_path / "kaggriculture_engine.so"
    context = multiprocessing.get_context("fork")
    processes = [context.Process(target=_build_into, args=(target,)) for _ in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(600)

    assert [process.exitcode for process in processes] == [0, 0]
    library = ctypes.CDLL(str(target))
    library.kag_abi_version.restype = ctypes.c_uint32
    assert library.kag_abi_version() == 1
    assert list(tmp_path.glob("*.tmp")) == []
