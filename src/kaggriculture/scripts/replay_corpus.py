"""Replay recorded Kaggle episodes through the Rust engine, step by step.

A replay archive holds the world after every turn, both seats' actions, and
the episode seed. The engine is deterministic given those, so replaying the
recorded actions must reproduce every recorded observation exactly: every
tile, every inventory, every price, every turn. This tool does that for a
whole archive, or a fixed-seed sample across several, and names the first
turn and field where the engine and the record disagree.

Run from the repository root with the bindings built (``rust/python``)::

    uv run replay-corpus --dir /data/kaggriculture/episodes --episodes 50

Exit status is non-zero if any episode diverged.
"""

import argparse
import json
import logging
import random
import sys
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

CORPUS = Path("/data/kaggriculture/episodes")


@dataclass(frozen=True)
class Divergence:
    """Where a replay first disagreed with the record."""

    step: int
    seat: int
    path: str
    expected: Any
    actual: Any

    def __str__(self) -> str:
        """One line naming the turn, seat and field."""
        return (
            f"step {self.step} seat {self.seat} {self.path}: "
            f"recorded {self.expected!r}, engine {self.actual!r}"
        )


@dataclass(frozen=True)
class Verdict:
    """The outcome of replaying one episode."""

    archive: str
    name: str
    episode_id: int | None
    version: str | None
    seed: int
    steps: int
    divergence: Divergence | None
    recorded_banks: list[float]
    engine_banks: list[float]
    #: The first non-playing status (ERROR, TIMEOUT, INVALID) a seat recorded,
    #: as ``(step, seat, status)``. The framework keeps such an episode
    #: running with that seat passing, so every recorded step is compared.
    failure: tuple[int, int, str] | None = None

    @property
    def identical(self) -> bool:
        """True when every recorded field of every step was reproduced."""
        return self.divergence is None

    def __str__(self) -> str:
        """One line per episode."""
        label = (
            f"{self.archive}:{self.name} (episode {self.episode_id}, seed {self.seed})"
        )
        if self.identical:
            note = ""
            if self.failure is not None:
                step, seat, status = self.failure
                note = f", seat {seat} {status} from step {step}"
            return (
                f"OK       {label}: {self.steps} steps{note}, banks {self.engine_banks}"
            )
        return f"DIVERGED {label}: {self.divergence}"


PLAYING = frozenset({"ACTIVE", "INACTIVE", "DONE"})


def compare(expected: Any, actual: Any, path: str) -> Divergence | None:  # noqa: ANN401
    """Return the first difference between two observation trees, if any.

    Dict key order is ignored, including inside per-unit inventories. It was
    compared there until 2026-09-14, on the reasoning that insertion order
    decides what survives an end-of-day drop -- which is true of the engine:
    `UnitAction::Drop` walks the inventory in order filling the shed to
    `shed_capacity`, so order decides which goods get the last slots.

    The recording cannot witness it. Every archive is written with sorted keys
    -- checked across a whole episode, 0 dicts of any kind out of alphabetical
    order -- so the recorded order is alphabetical by construction and the
    engine's is insertion order, and comparing them reports a difference
    whenever the true order is not alphabetical. That is most of the time: it
    failed 8 of 8 recent episodes on an engine that is otherwise exact, so the
    tool could not pass and was gating nothing. With order ignored those same
    8 replay identically, every field of every observation of every turn.

    What it was guarding is already covered where it can be, and is unreachable
    where it cannot. The shed's contents are compared field by field like every
    other measure, and the eight episodes that replay identically contain 14
    drops with the shed at 90 or more of its 100 capacity. The case where order
    decides -- a drop that overflows while the unit carries two or more items --
    happens 0 times in 2,422 drops over 30 episodes. So there is nothing here to
    pin that a synthetic input would not be inventing, and both implementations
    walk their own insertion order, which the reference's plain-dict `_inv_add`
    makes identical to ours by construction.
    """
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        left: dict[Any, Any] = dict(expected)
        right: dict[Any, Any] = dict(actual)
        if set(left) != set(right):
            return Divergence(-1, -1, f"{path} keys", sorted(left), sorted(right))
        for key in left:
            found = compare(left[key], right[key], f"{path}.{key}")
            if found is not None:
                return found
        return None
    if (
        isinstance(expected, Sequence)
        and isinstance(actual, Sequence)
        and not isinstance(expected, str)
        and not isinstance(actual, str)
    ):
        if len(expected) != len(actual):
            return Divergence(-1, -1, f"{path} length", len(expected), len(actual))
        for index, (left_item, right_item) in enumerate(
            zip(expected, actual, strict=True)
        ):
            found = compare(left_item, right_item, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    if expected != actual or type(expected) is not type(actual):
        return Divergence(-1, -1, path, expected, actual)
    return None


def _recorded_observation(agent: Mapping[str, Any]) -> dict[str, Any]:
    observation = dict(agent["observation"])
    observation.pop("remainingOverageTime", None)
    return observation


def _engine_observation(
    engine: Any,  # noqa: ANN401
    seat: int,
    template: Mapping[str, Any],
) -> dict[str, Any]:
    observation = engine.observation(seat)
    # The framework writes ``step`` onto seat 0 only.
    if "step" not in template:
        observation.pop("step", None)
    return observation


def replay(episode: Mapping[str, Any], *, archive: str = "", name: str = "") -> Verdict:
    """Replay one recorded episode and compare every step.

    Args:
        episode: The replay JSON: ``configuration``, ``info`` (with ``seed``)
            and ``steps``.
        archive: Label for the archive the episode came from.
        name: Label for the episode within the archive.

    Returns:
        A verdict naming the first divergence, or none.
    """
    import kaggriculture_engine  # noqa: PLC0415  (built separately; see rust/python)

    configuration = dict(episode.get("configuration", {}))
    seed = int(episode["info"]["seed"])
    steps = episode["steps"]
    engine = kaggriculture_engine.Engine(
        configuration, seed=seed, players=len(steps[0])
    )
    info = episode.get("info", {})
    episode_id = info.get("EpisodeId")
    version = episode.get("module_version")

    failure: tuple[int, int, str] | None = None

    def verdict(divergence: Divergence | None, steps_done: int) -> Verdict:
        return Verdict(
            archive=archive,
            name=name,
            episode_id=int(episode_id) if episode_id is not None else None,
            version=str(version) if version is not None else None,
            seed=seed,
            steps=steps_done,
            divergence=divergence,
            recorded_banks=[
                float(farm["money"]) for farm in steps[-1][0]["observation"]["farms"]
            ],
            engine_banks=list(engine.money),
            failure=failure,
        )

    for turn, agents in enumerate(steps):
        if turn > 0:
            engine.step([agent.get("action") for agent in agents])
        for seat, agent in enumerate(agents):
            expected = _recorded_observation(agent)
            actual = _engine_observation(engine, seat, expected)
            found = compare(expected, actual, "observation")
            if found is not None:
                return verdict(
                    Divergence(turn, seat, found.path, found.expected, found.actual),
                    turn,
                )
        statuses = [str(agent["status"]) for agent in agents]
        # A seat that errors, times out or returns something unusable keeps
        # its non-playing status while the episode runs on with that seat
        # passing, so the observations stay comparable. Only the framework's
        # done flag is not: it is not DONE for that seat until the end.
        for seat, status in enumerate(statuses):
            if status not in PLAYING and failure is None:
                failure = (turn, seat, status)
        if any(status not in PLAYING for status in statuses):
            continue
        recorded_done = all(status == "DONE" for status in statuses)
        if recorded_done != engine.done:
            return verdict(
                Divergence(turn, -1, "done", recorded_done, engine.done), turn
            )
    return verdict(None, len(steps) - 1)


def episode_names(archive: Path) -> list[str]:
    """The replay JSON members of an archive, in archive order."""
    with zipfile.ZipFile(archive) as bundle:
        return [
            name
            for name in bundle.namelist()
            if name.endswith(".json") and not name.endswith("/")
        ]


def load_episode(archive: Path, name: str) -> dict[str, Any]:
    """Decode one replay from an archive."""
    with zipfile.ZipFile(archive) as bundle:
        return json.loads(bundle.read(name))


def iter_episodes(
    archive: Path, names: Sequence[str] | None = None
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(name, episode)`` for the chosen members of an archive."""
    with zipfile.ZipFile(archive) as bundle:
        chosen = list(names) if names is not None else episode_names(archive)
        for name in chosen:
            yield name, json.loads(bundle.read(name))


def sample(archives: Sequence[Path], count: int, seed: int) -> list[tuple[Path, str]]:
    """Choose ``count`` episodes uniformly across every archive, reproducibly.

    Sampling across the whole population rather than taking the head of one
    archive is deliberate: idioms such as ``SELL X 999`` are spread through
    the ranking, and a top-only slice can miss a whole class of behaviour.
    """
    population = [
        (archive, name) for archive in archives for name in episode_names(archive)
    ]
    if count <= 0 or count >= len(population):
        return population
    return random.Random(seed).sample(population, count)


def verify(
    chosen: Sequence[tuple[Path, str]],
    *,
    version: str | None = None,
    stop_on_divergence: bool = False,
) -> list[Verdict]:
    """Replay the chosen episodes and return one verdict each.

    Args:
        chosen: ``(archive, member)`` pairs, as ``sample`` returns.
        version: If set, skip episodes recorded by a different engine version.
        stop_on_divergence: Return as soon as one episode diverges.

    Returns:
        Verdicts in the order given, minus any skipped for their version.
    """
    verdicts: list[Verdict] = []
    for archive, name in chosen:
        episode = load_episode(archive, name)
        recorded_by = episode.get("module_version")
        if version is not None and str(recorded_by) != version:
            LOGGER.info(
                "skipping %s:%s recorded by %s", archive.name, name, recorded_by
            )
            continue
        verdict = replay(episode, archive=archive.name, name=name)
        LOGGER.info("%s", verdict)
        verdicts.append(verdict)
        if stop_on_divergence and not verdict.identical:
            break
    return verdicts


def summarise(verdicts: Sequence[Verdict]) -> str:
    """A one-paragraph summary for the log."""
    diverged = [verdict for verdict in verdicts if not verdict.identical]
    steps = sum(verdict.steps for verdict in verdicts)
    lines = [
        f"{len(verdicts)} episodes, {steps} steps replayed, "
        f"{len(verdicts) - len(diverged)} identical, {len(diverged)} diverged"
    ]
    lines.extend(str(verdict) for verdict in diverged)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--dir", type=Path, default=CORPUS, help="directory of daily .zip archives"
    )
    parser.add_argument(
        "--archive", type=Path, action="append", help="a specific archive (repeatable)"
    )
    parser.add_argument(
        "--episodes", type=int, default=0, help="sample size (0 = every episode)"
    )
    parser.add_argument("--seed", type=int, default=20260905, help="sampling seed")
    parser.add_argument(
        "--version",
        default=None,
        help="only episodes recorded by this kaggle-environments version",
    )
    parser.add_argument(
        "--stop", action="store_true", help="stop at the first divergence"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )

    archives = args.archive or sorted(args.dir.glob("*.zip"))
    if not archives:
        print(f"no archives under {args.dir}", file=sys.stderr)
        return 2
    chosen = sample(archives, args.episodes, args.seed)
    verdicts = verify(chosen, version=args.version, stop_on_divergence=args.stop)
    print(summarise(verdicts))
    return 0 if all(verdict.identical for verdict in verdicts) else 1


if __name__ == "__main__":
    sys.exit(main())
