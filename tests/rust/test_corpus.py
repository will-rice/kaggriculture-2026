"""Recorded episodes replay through the Rust engine to every recorded field.

The fast test records episodes with the installed reference, packs them the
way Kaggle's daily archives are packed, and checks that the replay tool calls
them identical and that it can tell when one is not. The slow test runs the
same tool over a fixed-seed sample of the real corpus, when it is on disk.
"""
# ruff: noqa: D103

import csv
import importlib.metadata
import io
import json
import random
import zipfile
from pathlib import Path
from typing import Any

import pytest
from kaggle_environments import make

from kaggriculture.scripts import replay_corpus
from tests.rust.policies import biased_random, husbandry

pytest.importorskip("kaggriculture_engine", reason="build rust/python first")

ENGINE = importlib.metadata.version("kaggle-environments")


def _record(seed: int, agents: list[Any]) -> dict[str, Any]:
    environment = make("kaggriculture", configuration={"seed": seed}, debug=False)
    environment.run(agents)
    episode = environment.toJSON()
    episode["info"]["EpisodeId"] = seed
    return episode


def _pack(archive: Path, episodes: dict[str, dict[str, Any]]) -> Path:
    manifest = io.StringIO()
    writer = csv.DictWriter(
        manifest, fieldnames=["episode_id", "avg_score", "min_score", "agent_count"]
    )
    writer.writeheader()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, episode in episodes.items():
            bundle.writestr(name, json.dumps(episode))
            writer.writerow(
                {
                    "episode_id": episode["info"]["EpisodeId"],
                    "avg_score": 1000,
                    "min_score": 1000,
                    "agent_count": 2,
                }
            )
        bundle.writestr("manifest.csv", manifest.getvalue())
    return archive


@pytest.fixture(scope="module")
def local_archive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rng = random.Random(5)
    episodes = {
        "starter-vs-random.json": _record(301, ["starter", "random"]),
        "husbandry-vs-biased.json": _record(
            302, [lambda obs: husbandry(obs), lambda obs: biased_random(rng, obs)]
        ),
    }
    return _pack(
        tmp_path_factory.mktemp("corpus") / "kaggriculture-episodes-local.zip", episodes
    )


def test_locally_recorded_episodes_replay_identically(local_archive: Path) -> None:
    chosen = replay_corpus.sample([local_archive], 0, 1)
    assert len(chosen) == 2
    verdicts = replay_corpus.verify(chosen, version=ENGINE)
    assert len(verdicts) == 2
    for verdict in verdicts:
        assert verdict.identical, str(verdict)
        assert verdict.steps == 719
        assert verdict.engine_banks == verdict.recorded_banks
    assert "2 identical, 0 diverged" in replay_corpus.summarise(verdicts)


def test_the_tool_names_a_corrupted_step(local_archive: Path, tmp_path: Path) -> None:
    episode = replay_corpus.load_episode(local_archive, "starter-vs-random.json")
    episode["steps"][400][1]["observation"]["farms"][0]["money"] += 1
    verdict = replay_corpus.replay(episode, archive="corrupt", name="x")
    assert not verdict.identical
    assert verdict.divergence is not None
    assert verdict.divergence.step == 400
    assert verdict.divergence.seat == 1
    assert verdict.divergence.path == "observation.farms[0].money"
    assert "1 diverged" in replay_corpus.summarise([verdict])

    # A recorded action that the engine did not see would show up one step later.
    episode = replay_corpus.load_episode(local_archive, "starter-vs-random.json")
    episode["steps"][10][0]["action"] = {
        "farmer": ["PASS"],
        "hands": [],
        "market": [["HIRE"]],
    }
    verdict = replay_corpus.replay(episode)
    assert verdict.divergence is not None
    assert verdict.divergence.step == 10
    assert verdict.divergence.path.startswith("observation.farms[0]")

    archive = _pack(tmp_path / "bad.zip", {"bad.json": episode})
    assert replay_corpus.main(["--archive", str(archive)]) == 1
    assert (
        replay_corpus.main(["--archive", str(local_archive), "--version", ENGINE]) == 0
    )


@pytest.mark.slow
def test_a_corpus_sample_replays_identically(request: pytest.FixtureRequest) -> None:
    corpus = replay_corpus.CORPUS
    archives = (
        sorted(corpus.glob("kaggriculture-episodes-*.zip")) if corpus.exists() else []
    )
    if not archives:
        pytest.skip(f"no replay archives under {corpus}")
    count = int(request.config.getoption("--rust-episodes"))
    chosen = replay_corpus.sample(archives, max(count, 8), 20260905)
    verdicts = replay_corpus.verify(chosen, version=ENGINE)
    assert verdicts, f"no {ENGINE}-version episodes in the sample"
    diverged = [verdict for verdict in verdicts if not verdict.identical]
    assert not diverged, replay_corpus.summarise(verdicts)


def test_an_episode_with_a_failing_seat_is_compared_to_the_end() -> None:
    def crashes_on_day_two(observation: Any) -> Any:  # noqa: ANN401
        if observation["day"] >= 2:
            raise RuntimeError("agent bug")
        return {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 1]]}

    episode = _record(303, ["starter", crashes_on_day_two])
    # The framework does not end the episode: the seat is marked ERROR and
    # passes for the rest of the season, and every step is still recorded.
    assert len(episode["steps"]) == 720
    errored = [
        turn
        for turn, agents in enumerate(episode["steps"])
        if agents[1]["status"] == "ERROR"
    ]
    assert errored and errored[0] == 49

    verdict = replay_corpus.replay(episode)
    assert verdict.identical, str(verdict)
    assert verdict.steps == 719
    assert verdict.failure == (49, 1, "ERROR")
    assert "seat 1 ERROR from step 49" in str(verdict)
