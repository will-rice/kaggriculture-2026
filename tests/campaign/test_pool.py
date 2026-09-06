"""The opponent pool: who is in it, who joins, and who makes way."""

from pathlib import Path

from kaggriculture.campaign import pool, roster


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

    p.add_champion("champ", "/x/champ.py")

    assert p.names() == [*"abcde", "champ"]
    assert p.opponents["champ"] == "/x/champ.py"
    assert p.history[-1]["action"] == "add_champion"


def test_trim_keeps_the_highest_rated_and_names_who_left() -> None:
    """The pool is the top of the tournament, so the weakest make way.

    An opponent every candidate already beats separates two candidates no
    better than a coin does, and it costs a game a round to learn that.
    """
    p = five()
    standings = {"a": 2.0, "b": -1.0, "c": 1.0, "d": -3.0, "e": 0.0}

    dropped = p.trim(standings, size=3)

    assert p.names() == ["a", "c", "e"]
    # Weakest first, which is the order they would have gone in one at a time.
    assert dropped == ["d", "b"]
    assert p.history[-1] == {**p.history[-1], "action": "trim", "dropped": ["b", "d"]}


def test_trim_keeps_everything_when_the_pool_is_already_small_enough() -> None:
    """No churn for its own sake: a pool under the size is left alone."""
    p = five()

    dropped = p.trim(dict.fromkeys("abcde", 0.0), size=8)

    assert dropped == []
    assert p.names() == list("abcde")
    assert not [entry for entry in p.history if entry["action"] == "trim"]


def test_an_opponent_with_no_rating_is_the_first_to_go() -> None:
    """A pool member the tournament did not rank cannot be defended.

    It happens when a champion joins between a tournament and the trim that
    follows it: the standings name the champion under its own id, not its
    pool name, and anything else unrated was not in that tournament at all.
    """
    p = five()

    dropped = p.trim({"a": 1.0, "b": 0.5}, size=2)

    assert p.names() == ["a", "b"]
    assert set(dropped) == {"c", "d", "e"}


def test_round_trips_through_json(tmp_path: Path) -> None:
    """save() then load() reproduces the same pool."""
    p = five()
    p.add_champion("champ", "/x/champ.py")

    p.save(tmp_path / "pool.json")

    assert pool.Pool.load(tmp_path / "pool.json") == p
