"""The opponent pool: who is in it, who joins, and who a gate draws from it."""

from pathlib import Path

from kaggriculture.campaign import config, pool, roster


def five() -> pool.Pool:
    """A five-opponent pool, for tests that mutate it."""
    return pool.Pool(opponents={name: f"/x/{name}.py" for name in "abcde"})


def test_initial_pool_is_the_training_roster() -> None:
    """Pool.initial() mirrors roster.TRAINING."""
    p = pool.Pool.initial()

    assert p.names() == list(roster.TRAINING)
    assert p.opponents["v54"] == str(roster.TRAINING["v54"])


def test_a_champion_joins_and_nothing_leaves_with_it() -> None:
    """Every champion is an opponent the next candidate has to finish above.

    Who leaves is `trim`'s decision, taken on rating, so joining is only ever
    an addition. They were one method once, and the retirement rule inside it
    could keep an opponent nobody had crushed but everybody beat.
    """
    p = five()

    p.add_champion("/x/champ.py")

    # One key, `config.POOL_CHAMPION`, whatever the champion's file is called.
    # It used to take the name too and add `champion_N`, which collided with the
    # tape lineage's `champion_1` already in the pool and replaced it.
    assert p.names() == [*"abcde", config.POOL_CHAMPION]
    assert p.opponents[config.POOL_CHAMPION] == "/x/champ.py"
    assert p.history[-1]["action"] == "add_champion"

    # And a second promotion overwrites rather than accumulating.
    p.add_champion("/x/next.py")
    assert p.names() == [*"abcde", config.POOL_CHAMPION]
    assert p.opponents[config.POOL_CHAMPION] == "/x/next.py"


def test_round_trips_through_json(tmp_path: Path) -> None:
    """save() then load() reproduces the same pool."""
    p = five()
    p.add_champion("/x/champ.py")

    p.save(tmp_path / "pool.json")

    assert pool.Pool.load(tmp_path / "pool.json") == p


def test_a_beaten_champion_stays_in_the_pool() -> None:
    """A promotion moves the old champion aside; it does not delete it.

    The gate's field comparison excludes the champion itself, so the pool has
    to hold something the champion does not already beat or a candidate has
    nothing to fail against. On 2026-09-16 it held nothing of the sort:
    champion_19 beat all 67 harvested agents, its rate pinned at 1.000, and no
    program that could ever exist scores higher than that. Fifty candidates
    were evaluated and refused in the six hours after it, every one of them at
    0.993 or 0.994 "where ours scored 1.000".

    So every rung stays. `ours` is the current champion and is overwritten;
    the one it displaces is kept as `ours_N`, numbered from its own file, and
    nothing removes it.

    `ours_`, not `champion_`. That prefix is overloaded -- `champion_1` and
    `champion_65` here are the abandoned tape lineage, opponents rather than
    rungs of ours -- and on 2026-09-12 a promotion numbering itself from an
    empty champions directory took the name `champion_1` and overwrote the
    tape agent outright. Harvested names are an author and a kernel slug, so
    nothing but a promotion can produce `ours_`.
    """
    opponents = {
        "champion_1": "/tape/1.py",
        "champion_65": "/tape/65.py",
        "router_v1": "/public/a.py",
        "shopforge": "/public/b.py",
    }
    subject = pool.Pool(opponents=dict(opponents))

    subject.add_champion("/champions/champion_3.py")

    assert subject.opponents[config.POOL_CHAMPION] == "/champions/champion_3.py"
    assert {k: v for k, v in subject.opponents.items() if k in opponents} == opponents
    assert len(subject.opponents) == len(opponents) + 1, "the first has nothing to keep"

    subject.add_champion("/champions/champion_4.py")

    assert subject.opponents[config.POOL_CHAMPION] == "/champions/champion_4.py"
    assert subject.opponents["ours_3"] == "/champions/champion_3.py"
    assert len(subject.opponents) == len(opponents) + 2
    # And the tape lineage is still exactly where it was, which is the whole
    # reason the key is not `champion_3`.
    assert {k: v for k, v in subject.opponents.items() if k in opponents} == opponents

    subject.add_champion("/champions/champion_5.py")

    assert subject.opponents["ours_3"] == "/champions/champion_3.py"
    assert subject.opponents["ours_4"] == "/champions/champion_4.py"
    assert len(subject.opponents) == len(opponents) + 3, "nothing is trimmed"


def vendored(root: Path, *names: str) -> Path:
    """An opponents directory holding ``names``, each with a ``main.py``."""
    for name in names:
        (root / name).mkdir(parents=True)
        (root / name / "main.py").write_text("def agent(obs, cfg): ...", "utf-8")
    return root


def test_an_agent_on_disk_the_pool_never_held_is_adopted(tmp_path: Path) -> None:
    """The gap `tape_opponents` left open for sixty-one agents.

    Two scripts write the opponents directory and only `harvest` ever wrote
    membership, so the families rebuilt from recorded ladder episodes were
    never played by anything. Membership comes from the directory now, which
    a new writer cannot forget the way it could forget a call.
    """
    root = vendored(tmp_path / "opponents", "family_ymg_aq", "kaito_v56")
    held = pool.Pool(opponents={})

    adopted = held.adopt(root)

    assert adopted == ["family_ymg_aq", "kaito_v56"]
    assert held.opponents["family_ymg_aq"] == str(root / "family_ymg_aq" / "main.py")


def test_an_agent_the_roster_holds_under_an_alias_is_not_adopted_twice(
    tmp_path: Path,
) -> None:
    """Matched on the path, because the roster renames a dozen of these.

    `v56` is the roster's name for `kaito_v56`. Adopting by directory name
    would enrol the same file a second time, and a candidate would play that
    agent twice with both copies counting toward its rate.
    """
    root = vendored(tmp_path / "opponents", "kaito_v56")
    held = pool.Pool(opponents={"v56": str(root / "kaito_v56" / "main.py")})

    adopted = held.adopt(root)

    assert adopted == []
    assert held.names() == ["v56"]


def test_a_directory_without_an_agent_in_it_is_not_an_opponent(
    tmp_path: Path,
) -> None:
    """The opponents directory also holds a README and loose scripts."""
    root = tmp_path / "opponents"
    (root / "notes").mkdir(parents=True)
    (root / "notes" / "README.md").write_text("not an agent", "utf-8")
    vendored(root, "real_agent")
    held = pool.Pool(opponents={})

    assert held.adopt(root) == ["real_agent"]


def test_adopting_twice_adds_nothing_the_second_time(tmp_path: Path) -> None:
    """Every launch calls this, so it has to be idempotent."""
    root = vendored(tmp_path / "opponents", "one", "two")
    held = pool.Pool(opponents={})

    assert held.adopt(root) == ["one", "two"]
    assert held.adopt(root) == []
    assert len(held.names()) == 2
