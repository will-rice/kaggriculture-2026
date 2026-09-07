"""Archived episodes as replayable tapes.

Each daily archive under ``config.EPISODES`` is a zip of episode JSONs from
the ladder. ``info.seed`` is the seed the engine was created with; the
action recorded at ``steps[t]`` is what each seat submitted while looking at
``steps[t-1]``'s observation, and ``steps[t]``'s observation is the state it
produced.
"""

import functools
import json
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from kaggriculture.campaign import config

STATUSES = {"ACTIVE", "DONE", "INACTIVE"}
# The engine version every measurement here is about, and the first day an
# archive holds a game played on it. Only that version is read: 1.32.6 and
# 1.32.7 each rewrote prices, so a number averaged across them is about two
# different games.
#
# The date is not a second filter -- the version check below is what decides --
# it is where a walk starts, so that reading "every game" does not mean parsing
# three weeks of archives that hold none. Measured per archive: 1.32.2 through
# 1.32.6 run to 2026-08-14, 2026-08-15 is the changeover, and every archive
# after it is 1.32.7 throughout. This was 2026-08-24 and silently dropped nine
# days of qualifying games -- about six thousand of them, more than the corpus
# it was reading.
ENGINE = "1.32.7"
SINCE = "2026-08-15"


class Episode(BaseModel):
    """One recorded episode."""

    seed: int
    engine_version: str
    steps: list[list[dict[str, Any]]]

    @property
    def clean(self) -> bool:
        """Whether every recorded status is one the engine itself assigns.

        Any other status means a seat crashed and the framework substituted
        PASS for it; the recorded action may not be what was actually
        applied, so such an episode cannot be replayed as a differential
        oracle.
        """
        return all(seat["status"] in STATUSES for step in self.steps for seat in step)


def archives(since: str = SINCE) -> list[Path]:
    """Every daily archive dated ``since`` or later, oldest first.

    The unit a whole-corpus walk divides over: one archive is a day, it is
    the only thing a worker needs to be handed, and nothing in one depends on
    another.

    Raises:
        FileNotFoundError: There are none, which is a missing dataset rather
            than an empty corpus and must fail rather than measure nothing.
    """
    found = sorted(
        p
        for p in config.EPISODES.glob("kaggriculture-episodes-*.zip")
        if p.stem[-10:] >= since
    )
    if not found:
        raise FileNotFoundError(
            f"no archives dated {since} or later under {config.EPISODES}"
        )
    return found


def _episode(raw: dict[str, Any]) -> Episode:
    """One episode from its parsed JSON."""
    return Episode(
        seed=int(raw["info"]["seed"]),
        engine_version=str(raw["module_version"]),
        steps=raw["steps"],
    )


def iter_archive(path: Path) -> Iterator[Episode]:
    """Yield every episode in one daily archive, in name order."""
    with zipfile.ZipFile(path) as archive:
        for name in sorted(n for n in archive.namelist() if n.endswith(".json")):
            yield _episode(json.loads(archive.read(name)))


def _members(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    """Each clean 1.32.7 member of one archive, with its parsed JSON.

    The one place a game is decided to qualify, and the one place its JSON is
    parsed. Both callers below used to parse it themselves, which is why
    reading the corpus by index parsed every episode twice: once to decide it
    counted and again to return it.
    """
    with zipfile.ZipFile(path) as archive:
        for name in sorted(n for n in archive.namelist() if n.endswith(".json")):
            raw = json.loads(archive.read(name))
            if str(raw["module_version"]) == ENGINE and _is_clean(raw["steps"]):
                yield name, raw


def qualifying_episodes(path: Path) -> Iterator[Episode]:
    """Every clean 1.32.7 episode in one archive, parsed once.

    What a whole-corpus walk wants. ``qualifying`` addresses one episode by
    index, which is what a parametrized test wants and what nothing reading
    every game should use.
    """
    for _, raw in _members(path):
        yield _episode(raw)


def sample(n: int, since: str = SINCE) -> list[Episode]:
    """Return the first ``n`` clean 1.32.7 episodes from ``since`` or later archives.

    For scripts. Tests must use ``qualifying`` instead: calling this from a
    ``pytest.mark.parametrize`` decorator parses every episode up to ``n`` at
    collection time, before ``-m 'not slow'`` has deselected anything.
    """
    out: list[Episode] = []
    for archive in archives(since):
        for episode in qualifying_episodes(archive):
            out.append(episode)
            if len(out) == n:
                return out
    return out


def _is_clean(steps: list[list[dict[str, Any]]]) -> bool:
    """Whether every recorded status is one the engine itself assigns.

    Mirrors ``Episode.clean`` but works from the raw JSON so the discovery
    walk below need not build a full ``Episode`` (and pay its pydantic
    validation) just to decide whether one qualifies.
    """
    return all(seat["status"] in STATUSES for step in steps for seat in step)


class _Index:
    """Qualifying ``(archive, member)`` references, discovered lazily and kept.

    Extends only as far as the highest index anyone has asked for, and never
    re-walks archives it has already covered.
    """

    def __init__(self, since: str) -> None:
        self._since = since
        self._refs: list[tuple[Path, str]] = []
        self._walker = self._walk()

    def _walk(self) -> Iterator[tuple[Path, str]]:
        for archive in archives(self._since):
            for name, _ in _members(archive):
                yield archive, name

    def ref(self, index: int) -> tuple[Path, str]:
        """Return the ``index``-th qualifying reference, walking further if needed."""
        while len(self._refs) <= index:
            try:
                self._refs.append(next(self._walker))
            except StopIteration as exhausted:
                raise FileNotFoundError(
                    f"fewer than {index + 1} qualifying episodes dated {self._since} "
                    f"or later under {config.EPISODES}"
                ) from exhausted
        return self._refs[index]


@functools.lru_cache(maxsize=None)
def _index(since: str) -> _Index:
    """Return the process-wide index for ``since``, shared across every call."""
    return _Index(since)


def qualifying(i: int, since: str = SINCE) -> Episode:
    """Return the ``i``-th clean 1.32.7 episode from ``since`` or later archives.

    Safe to call from inside a test body parametrized over plain integers
    (``range(n)``): unlike ``sample``, nothing here parses an archive until a
    test actually calls it, and the ``(archive, member)`` references found
    along the way are cached per process, so parametrizing many tests over
    ``qualifying`` costs one lazy walk, not one per test.

    Args:
        i: The zero-based index among qualifying episodes.
        since: Only archives dated this day or later are walked.

    Raises:
        FileNotFoundError: If ``config.EPISODES`` has no matching archive, or
            fewer than ``i + 1`` qualifying episodes exist there — a missing
            dataset must fail the proof, not silently skip it.
    """
    archive, name = _index(since).ref(i)
    with zipfile.ZipFile(archive) as opened:
        raw = json.loads(opened.read(name))
    return Episode(
        seed=int(raw["info"]["seed"]),
        engine_version=str(raw["module_version"]),
        steps=raw["steps"],
    )
