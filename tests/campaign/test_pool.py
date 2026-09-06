"""The opponent pool: who is in it, who joins, and who makes way."""

from pathlib import Path

from kaggriculture.campaign import config, pool, roster


def five() -> pool.Pool:
    """A five-opponent pool, for tests that mutate it."""
    return pool.Pool(opponents={name: f"/x/{name}.py" for name in "abcde"})


def filled() -> pool.Pool:
    """`five()` topped up with champions until it stands exactly at the cap.

    Derived from `config.POOL_CAP` rather than written out, because the cap
    moves with the vendored roster it has to clear: a test that spelled out
    ten stopped meaning "a full pool" the moment six opponents were added.
    """
    p = five()
    for number in range(config.POOL_CAP - len(p.names())):
        p.add_champion(f"c{number}", f"/x/c{number}.py", rates={})
    assert len(p.names()) == config.POOL_CAP
    return p


def test_initial_pool_is_the_training_roster() -> None:
    """Pool.initial() mirrors roster.TRAINING."""
    p = pool.Pool.initial()

    assert p.names() == list(roster.TRAINING)
    assert p.opponents["v54"] == str(roster.TRAINING["v54"])


def test_a_champion_joins_a_pool_that_is_not_full_and_nobody_leaves() -> None:
    """Every champion is a gatekeeper the next candidate has to get past."""
    p = five()

    retired = p.add_champion("champ", "/x/champ.py", rates={})

    assert retired is None
    assert p.names() == [*"abcde", "champ"]


def test_a_full_pool_retires_the_opponent_the_champion_beats_most_decisively() -> None:
    """At the cap, the pool keeps the opponents that still separate programs.

    An opponent the incoming champion beats nine times in ten no longer
    tells one candidate from another, and it is the most beaten of those --
    not the least weighted, there being no weights -- that makes way.
    """
    p = filled()

    retired = p.add_champion(
        "new", "/x/new.py", rates={"a": 0.96, "b": 0.99, "c": 0.5, "d": 0.97}
    )

    assert retired == "b"
    assert "b" not in p.names() and "new" in p.names()
    assert len(p.names()) == config.POOL_CAP


def test_a_full_pool_of_opponents_that_still_matter_keeps_them_all() -> None:
    """Nothing is retired to make room; the pool grows past the cap instead.

    A cap that evicted an opponent nobody had crushed would throw away the
    hardest matchups first, which are the ones a gate exists to ask about.
    """
    p = filled()

    retired = p.add_champion("new", "/x/new.py", rates=dict.fromkeys("abcde", 0.6))

    assert retired is None
    assert len(p.names()) == config.POOL_CAP + 1


def test_round_trips_through_json(tmp_path: Path) -> None:
    """save() then load() reproduces the same pool."""
    p = five()
    p.add_champion("champ", "/x/champ.py", rates={})

    p.save(tmp_path / "pool.json")

    assert pool.Pool.load(tmp_path / "pool.json") == p


def test_a_vendored_incumbent_is_never_retired() -> None:
    """`field` is a mean over the vendored incumbents, and it must stay one.

    Retiring one changes what that mean is taken over without changing its
    name, and a curve that rose afterwards would be a program getting better
    or the field getting easier with no way to tell which. It is the number
    the campaign is steered by, so a full pool retires a champion or nobody.
    """
    vendored = list(roster.TRAINING)
    champions = [f"champion_{n}" for n in range(1, config.POOL_CAP - len(vendored) + 1)]
    assert champions, "the cap must leave room for at least one champion"
    p = pool.Pool(opponents={n: f"/x/{n}.py" for n in vendored + champions})
    assert len(p.names()) == config.POOL_CAP

    # It crushes a vendored incumbent harder than any champion, and the
    # champion it beat most decisively is the one that makes way.
    rates = dict.fromkeys(vendored, 0.97) | dict.fromkeys(champions, 0.96)
    rates["v56"] = 1.0
    rates[champions[1]] = 0.99
    retired = p.add_champion("champion_new", "/x/champion_new.py", rates)

    assert retired == champions[1]
    assert set(vendored) <= set(p.names())
