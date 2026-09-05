"""The one packager: what `uv run package` and `uv run submit` actually write."""

import tarfile
from pathlib import Path

import pytest

from kaggriculture.scripts.package import build

# One file ships, and the repository licence rides with it. Nothing else: no
# engine library, no `kaggriculture` package, no attribution for a binary the
# archive no longer carries.
EXPECTED = {"main.py", "LICENSE"}

PASS_AGENT = (
    "def agent(observation, configuration=None):\n"
    "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
)


@pytest.fixture
def entrypoint(tmp_path: Path) -> Path:
    """A self-contained agent to package, standing in for the campaign floor."""
    path = tmp_path / "main.py"
    path.write_text(PASS_AGENT, encoding="utf-8")
    return path


def names(archive: Path) -> list[str]:
    """Return the member names of ``archive``, duplicates included."""
    with tarfile.open(archive) as bundle:
        return bundle.getnames()


def test_the_archive_is_exactly_the_agent_and_the_licence(
    entrypoint: Path, tmp_path: Path
) -> None:
    """Every member is expected, and each appears exactly once.

    A duplicated member is invisible to a set of names and to ``tar -x``,
    which simply overwrites; it doubles the upload and reads as a corrupt
    archive.
    """
    listed = names(build(tmp_path / "submission.tar.gz", entrypoint=entrypoint))

    assert sorted(listed) == sorted(set(listed))
    assert set(listed) == EXPECTED


def test_the_archive_ships_no_engine_and_no_package(
    entrypoint: Path, tmp_path: Path
) -> None:
    """A program is one file, so nothing it could import travels with it.

    The engine library and the four plumbing modules used to ship, for a
    lookahead option no evolved agent ever took. Shipping them made the
    search and the ladder disagree: a session could import a helper it was
    not allowed to edit, and a weakness in that helper was invisible to the
    search that was supposed to find it.
    """
    listed = names(build(tmp_path / "submission.tar.gz", entrypoint=entrypoint))

    assert not any(name.endswith(".so") for name in listed)
    assert not any(name == "kaggriculture" for name in listed)
    assert not any(name.startswith("kaggriculture/") for name in listed)
    assert "NOTICE" not in listed
    assert not any("__pycache__" in name for name in listed)


def test_build_accepts_a_self_contained_alternate_entrypoint(
    entrypoint: Path, tmp_path: Path
) -> None:
    """The bytes handed to the packager are the bytes that land at the root."""
    archive = build(tmp_path / "alternate.tar.gz", entrypoint=entrypoint)

    with tarfile.open(archive) as bundle:
        packaged_main = bundle.extractfile("main.py")
        assert packaged_main is not None
        source = packaged_main.read().decode()
    assert source == entrypoint.read_text()
