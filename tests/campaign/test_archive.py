"""Islands, UCB, replacement, migration, reset — each on a hand-built archive."""

import random
from pathlib import Path

from kaggriculture.campaign import archive, config


def make(tmp_path: Path) -> archive.Archive:
    """A fresh archive backed by files under `tmp_path`."""
    return archive.Archive(
        path=tmp_path / "archive.jsonl", programs_dir=tmp_path / "programs"
    )


def test_seed_places_one_copy_on_every_island(tmp_path: Path) -> None:
    """seed() writes one program with the given fitness to each island."""
    a = make(tmp_path)
    seed = tmp_path / "seed.py"
    seed.write_text(
        "def agent(o, c=None):\n"
        "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
    )
    a.seed(seed, fitness=0.1)
    assert [len(a.island(i)) for i in range(config.ISLANDS)] == [1] * config.ISLANDS
    assert all(
        p.kind == "seed" and p.mean == 0.1
        for i in range(config.ISLANDS)
        for p in a.island(i)
    )


def test_ucb_prefers_the_under_sampled_program_when_means_tie(tmp_path: Path) -> None:
    """With tied means, UCB's exploration bonus favors the fewer-eval program."""
    a = make(tmp_path)
    many = a.insert(
        archive.Program(
            id="many",
            island=0,
            source_path="x",
            parents=[],
            kind="full",
            fitness_sum=5.0,
            n_evals=10,
            status="ok",
            reason="",
            created=0.0,
        )
    )
    few = a.insert(
        archive.Program(
            id="few",
            island=0,
            source_path="y",
            parents=[],
            kind="full",
            fitness_sum=0.5,
            n_evals=1,
            status="ok",
            reason="",
            created=0.0,
        )
    )
    assert many is None and few is None
    assert a.ucb_parent(0, random.Random(0)).id == "few"


def test_insert_replaces_the_worst_when_the_island_is_full(tmp_path: Path) -> None:
    """A better child on a full island evicts the current worst program."""
    a = make(tmp_path)
    for i in range(config.ISLAND_SIZE):
        a.insert(
            archive.Program(
                id=f"p{i}",
                island=1,
                source_path="x",
                parents=[],
                kind="full",
                fitness_sum=i / 100,
                n_evals=1,
                status="ok",
                reason="",
                created=0.0,
            )
        )
    replaced = a.insert(
        archive.Program(
            id="new",
            island=1,
            source_path="x",
            parents=[],
            kind="full",
            fitness_sum=0.5,
            n_evals=1,
            status="ok",
            reason="",
            created=0.0,
        )
    )
    assert replaced is not None and replaced.id == "p0"
    assert len(a.island(1)) == config.ISLAND_SIZE and "new" in {
        p.id for p in a.island(1)
    }


def test_a_weaker_child_does_not_evict_anyone_from_a_full_island(
    tmp_path: Path,
) -> None:
    """insert() on a full island drops a not-better child instead of evicting."""
    a = make(tmp_path)
    for i in range(config.ISLAND_SIZE):
        a.insert(
            archive.Program(
                id=f"p{i}",
                island=2,
                source_path="x",
                parents=[],
                kind="full",
                fitness_sum=0.5,
                n_evals=1,
                status="ok",
                reason="",
                created=0.0,
            )
        )
    replaced = a.insert(
        archive.Program(
            id="weak",
            island=2,
            source_path="x",
            parents=[],
            kind="full",
            fitness_sum=0.1,
            n_evals=1,
            status="ok",
            reason="",
            created=0.0,
        )
    )
    assert (
        replaced is not None and replaced.id == "weak"
    )  # the child itself is what was dropped
    assert "weak" not in {p.id for p in a.island(2)}


def test_migration_copies_the_top_two_to_the_next_island_in_a_ring(
    tmp_path: Path,
) -> None:
    """migrate() copies each island's top MIGRANTS to the next island, ring-wise."""
    a = make(tmp_path)
    for island in range(config.ISLANDS):
        for j in range(3):
            a.insert(
                archive.Program(
                    id=f"i{island}p{j}",
                    island=island,
                    source_path="x",
                    parents=[],
                    kind="full",
                    fitness_sum=(island + 1) * (j + 1) / 20,
                    n_evals=1,
                    status="ok",
                    reason="",
                    created=0.0,
                )
            )
    a.migrate()
    last = config.ISLANDS - 1
    ids_on_0 = {p.id for p in a.island(0)}
    assert {f"i{last}p2", f"i{last}p1"} <= {
        p.parents[0] for p in a.island(0) if p.kind == "migrant"
    }
    assert len(ids_on_0) == 5


def test_reset_reseeds_the_worst_island_from_the_champion(tmp_path: Path) -> None:
    """reset_worst_island() clears the weakest island, reseeds it from the champion."""
    a = make(tmp_path)
    for island in range(config.ISLANDS):
        a.insert(
            archive.Program(
                id=f"i{island}",
                island=island,
                source_path="x",
                parents=[],
                kind="full",
                fitness_sum=island / 10,
                n_evals=1,
                status="ok",
                reason="",
                created=0.0,
            )
        )
    champion = a.top(1)[0]
    reset = a.reset_worst_island(champion)
    assert reset == 0
    assert [p.kind for p in a.island(0)] == ["reset"] and a.island(0)[0].parents == [
        champion.id
    ]


def test_the_log_replays_to_the_same_state(tmp_path: Path) -> None:
    """A fresh Archive replaying the log reaches the same state as the original."""
    a = make(tmp_path)
    a.insert(
        archive.Program(
            id="p",
            island=0,
            source_path="x",
            parents=[],
            kind="full",
            fitness_sum=0.3,
            n_evals=1,
            status="ok",
            reason="",
            created=0.0,
        )
    )
    a.append_eval("p", 0.7)
    a.record_failure(0, ["p"], "full", "syntax")
    b = archive.Archive(
        path=tmp_path / "archive.jsonl", programs_dir=tmp_path / "programs"
    )
    assert [p.model_dump() for p in b.island(0)] == [
        p.model_dump() for p in a.island(0)
    ]
    assert a.island(0)[0].mean == 0.5 and b.failures()[0].reason == "syntax"
