"""The one packager: what `uv run package` and `uv run submit` actually write."""

import tarfile
from pathlib import Path

from kaggriculture.campaign import harness
from kaggriculture.scripts.package import build

# `main.py`, the engine library, the attribution that has to travel with it,
# and the four plumbing modules the served agent imports. Nothing else.
EXPECTED = {
    "main.py",
    "NOTICE",
    "LICENSE",
    "kaggriculture_engine.so",
    "kaggriculture",
    *(f"kaggriculture/{module}" for module in harness.PACKAGE_MODULES),
}


def names(archive: Path) -> list[str]:
    """Return the member names of ``archive``, duplicates included."""
    with tarfile.open(archive) as bundle:
        return bundle.getnames()


def test_the_archive_is_exactly_the_agent_the_engine_and_the_plumbing(
    tmp_path: Path,
) -> None:
    """Every member is expected, and each appears exactly once.

    A duplicated member is invisible to a set of names and to ``tar -x``,
    which simply overwrites; it doubles the upload and reads as a corrupt
    archive.
    """
    listed = names(build(tmp_path / "submission.tar.gz"))

    assert sorted(listed) == sorted(set(listed))
    assert set(listed) == EXPECTED


def test_the_archive_ships_the_engine_and_its_attribution(tmp_path: Path) -> None:
    """The library is a port of Apache-2.0 kernel source; its NOTICE rides along."""
    archive = build(tmp_path / "submission.tar.gz")

    with tarfile.open(archive) as bundle:
        notice = bundle.extractfile("NOTICE")
        assert notice is not None
        text = notice.read().decode()
    assert "Apache License 2.0" in text
    assert "kaggriculture_engine.so" in names(archive)


def test_the_archive_does_not_ship_offline_tooling(tmp_path: Path) -> None:
    """The scripts and campaign packages have no place in a 4 MB agent archive.

    ``campaign`` plays hundreds of games against a league to search for a
    better agent, work the submitted agent never does at play time, and the
    guard also closes the path by which a later edit could reintroduce a heavy
    import behind it.
    """
    listed = names(build(tmp_path / "submission.tar.gz"))

    assert not any(name.startswith("kaggriculture/scripts") for name in listed)
    assert not any(name.startswith("kaggriculture/campaign") for name in listed)
    assert not any("__pycache__" in name for name in listed)


def test_build_accepts_a_self_contained_alternate_entrypoint(tmp_path: Path) -> None:
    """Candidate packaging can be tested without changing the served default."""
    entrypoint = tmp_path / "main.py"
    entrypoint.write_text(
        "def agent(observation, configuration=None):\n"
        "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
    )

    archive = build(tmp_path / "alternate.tar.gz", entrypoint=entrypoint)

    with tarfile.open(archive) as bundle:
        packaged_main = bundle.extractfile("main.py")
        assert packaged_main is not None
        source = packaged_main.read().decode()
    assert source == entrypoint.read_text()
