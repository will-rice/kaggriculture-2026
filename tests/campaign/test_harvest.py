"""Taking a newly published kernel into the pool."""

from pathlib import Path

import pytest

from kaggriculture.campaign import harness, harvest, kernel_watch, pool

PASS = (
    "def agent(observation, configuration=None):\n"
    "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
)
# Loads, and then takes longer per call than Kaggle allows. An opponent like
# this is not a hard opponent, it is a tax on every evaluation for the rest of
# the campaign.
SLOW = (
    "import time\n"
    "def agent(observation, configuration=None):\n"
    "    time.sleep(0.6)\n"
    "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
)
CRASHER = (
    "def agent(observation, configuration=None):\n"
    "    raise ZeroDivisionError('not an opponent')\n"
)


def test_a_ref_becomes_a_name_and_never_a_path() -> None:
    """The name is what a copy verdict says out loud, so it carries no path."""
    name = harvest.opponent_name("boatlee/v16-rc5-high-score-8c-4s-premium-market")

    assert name.startswith("boatlee_")
    assert "/" not in name and "." not in name
    assert len(name) <= harvest.NAME_LIMIT
    # Two kernels by one author stay distinguishable.
    assert harvest.opponent_name("a/one") != harvest.opponent_name("a/two")


def test_a_crashing_kernel_is_refused(tmp_path: Path) -> None:
    """A broken opponent is a broken pool: the loop halts rather than score on."""
    entry = tmp_path / "main.py"
    entry.write_text(CRASHER, encoding="utf-8")

    assert harvest.playable(entry, steps=3)


def test_a_slow_kernel_is_refused(tmp_path: Path) -> None:
    """Every later candidate plays this one thousands of times."""
    entry = tmp_path / "main.py"
    entry.write_text(SLOW, encoding="utf-8")

    problem = harvest.playable(entry, steps=3)

    assert "over the" in problem and str(harness.LATENCY_BUDGET) in problem


def test_a_playable_kernel_passes(tmp_path: Path) -> None:
    """The check has to admit something, or nothing is ever harvested."""
    entry = tmp_path / "main.py"
    entry.write_text(PASS, encoding="utf-8")

    assert harvest.playable(entry, steps=30) == ""


def test_vendoring_takes_the_whole_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A kernel is not always one file.

    The compiled ones carry headers, a tape and the ``agent.so`` built beside
    their ``main.py``; copying the entrypoint alone gives an agent that loads
    and then fails on turn one.
    """
    built = tmp_path / "built"
    built.mkdir()
    (built / "main.py").write_text(PASS, encoding="utf-8")
    (built / "agent.so").write_bytes(b"\x7fELF not really")
    (built / "tape.bin").write_bytes(b"tape")
    monkeypatch.setattr(harvest.config, "OPPONENTS", tmp_path / "opponents")

    entry = harvest.vendor(built / "main.py", "someone_kernel")

    assert entry.name == "main.py"
    assert {path.name for path in entry.parent.iterdir()} == {
        "main.py",
        "agent.so",
        "tape.bin",
    }


def test_a_kernel_that_plays_joins_the_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discovery, extraction and the build are stood in for; the rest is real.

    Nothing here decides whether it stays: the next gate measures its pairings
    and `Pool.trim` keeps the highest-rated, so a weak harvest leaves on the
    first promotion. Joining is the whole of what this does.
    """
    monkeypatch.setattr(harvest.config, "OPPONENTS", tmp_path / "opponents")
    monkeypatch.setattr(kernel_watch, "SEEN", tmp_path / "seen.json")
    monkeypatch.setattr(
        harvest.kernel_watch, "discover", lambda author, limit: ["a/good", "a/broken"]
    )
    monkeypatch.setattr(harvest.kernel_watch, "build_compiled", lambda ref: None)
    monkeypatch.setattr(
        harvest.kernel_watch,
        "extract",
        lambda ref: PASS if ref == "a/good" else CRASHER,
    )
    monkeypatch.setattr(harvest.kernel_watch, "WORK", tmp_path / "work")
    registry = tmp_path / "pool.json"
    pool.Pool(opponents={"pass": str(tmp_path / "pass.py")}).save(registry)
    (tmp_path / "pass.py").write_text(PASS, encoding="utf-8")

    added = harvest.harvest(2, registry)

    assert added == ["a_good"]
    saved = pool.Pool.load(registry)
    assert set(saved.opponents) == {"pass", "a_good"}
    # The broken one is remembered too: it will not build tomorrow either, and
    # re-checking every failure daily is how a daily job becomes an hourly one.
    assert "a/broken" in kernel_watch.SEEN.read_text(encoding="utf-8")


def test_a_kernel_already_in_the_pool_is_not_taken_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-harvesting an author must not replace the opponent mid-campaign.

    The pool's copy is what candidates have been measured against and what the
    field's pairings were played on; swapping the file underneath would leave
    every cached pairing describing an agent that is no longer there.
    """
    monkeypatch.setattr(harvest.config, "OPPONENTS", tmp_path / "opponents")
    monkeypatch.setattr(kernel_watch, "SEEN", tmp_path / "seen.json")
    monkeypatch.setattr(
        harvest.kernel_watch, "discover", lambda author, limit: ["a/good"]
    )
    monkeypatch.setattr(harvest.kernel_watch, "build_compiled", lambda ref: None)
    monkeypatch.setattr(harvest.kernel_watch, "extract", lambda ref: PASS)
    monkeypatch.setattr(harvest.kernel_watch, "WORK", tmp_path / "work")
    registry = tmp_path / "pool.json"
    pool.Pool(opponents={"a_good": "/somewhere/main.py"}).save(registry)

    added = harvest.harvest(1, registry)

    assert added == []
    assert pool.Pool.load(registry).opponents["a_good"] == "/somewhere/main.py"
