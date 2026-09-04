"""Archived episodes as replayable tapes.

Each daily archive under ``config.EPISODES`` is a zip of episode JSONs from
the ladder. ``info.seed`` is the seed the engine was created with; the
action recorded at ``steps[t]`` is what each seat submitted while looking at
``steps[t-1]``'s observation, and ``steps[t]``'s observation is the state it
produced.
"""

import json
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from kaggriculture.campaign import config

STATUSES = {"ACTIVE", "DONE", "INACTIVE"}


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


def iter_archive(path: Path) -> Iterator[Episode]:
    """Yield every episode in one daily archive, in name order."""
    with zipfile.ZipFile(path) as archive:
        for name in sorted(n for n in archive.namelist() if n.endswith(".json")):
            raw = json.loads(archive.read(name))
            yield Episode(
                seed=int(raw["info"]["seed"]),
                engine_version=str(raw["module_version"]),
                steps=raw["steps"],
            )


def sample(n: int, since: str = "2026-08-24") -> list[Episode]:
    """Return the first ``n`` clean 1.32.7 episodes from ``since`` or later archives."""
    archives = sorted(
        p
        for p in config.EPISODES.glob("kaggriculture-episodes-*.zip")
        if p.stem[-10:] >= since
    )
    out: list[Episode] = []
    for archive in archives:
        for episode in iter_archive(archive):
            if episode.engine_version == "1.32.7" and episode.clean:
                out.append(episode)
                if len(out) == n:
                    return out
    return out
