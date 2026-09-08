"""The opponent pool: who is in it, who joins, and who a gate draws from it."""

import random
from pathlib import Path

import pytest

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

    p.add_champion("champ", "/x/champ.py")

    assert p.names() == [*"abcde", "champ"]
    assert p.opponents["champ"] == "/x/champ.py"
    assert p.history[-1]["action"] == "add_champion"


def test_the_draw_always_contains_every_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anchors are what a rating is calibrated against, so they are not a draw.

    Every candidate needs direct edges to fixed points spanning the range. Left
    to chance, a champion is rated through a chain of overlapping pool eras --
    and that chain is measurably wrong: it put champion_37 at 0.994 against
    champion_1, which beats it 0.729 in the games themselves.
    """
    monkeypatch.setattr(config, "GATE_ANCHORS", ("a", "e"))
    monkeypatch.setattr(config, "GATE_OPPONENTS", 3)
    monkeypatch.setattr(config, "GATE_CONTENDERS", 1)
    p = five()

    for seed in range(20):
        drawn = p.sample({"b": 5.0}, random.Random(seed))
        assert {"a", "e"} <= set(drawn)
        assert len(drawn) == 3


def test_the_draw_contains_the_highest_rated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Topping the field still means beating the best of it.

    A candidate that never played the leaders cannot be said to have out-rated
    them, however well it did against the rest.
    """
    monkeypatch.setattr(config, "GATE_ANCHORS", ())
    monkeypatch.setattr(config, "GATE_OPPONENTS", 2)
    monkeypatch.setattr(config, "GATE_CONTENDERS", 2)
    p = five()

    drawn = p.sample(
        {"a": -9.0, "b": 5.0, "c": 4.0, "d": -1.0, "e": 0.0}, random.Random(1)
    )

    assert set(drawn) == {"b", "c"}


def test_the_rest_of_the_draw_is_random_and_reaches_the_whole_pool() -> None:
    """Coverage, and the only way a counter is ever found.

    The pool used to drop its lowest-rated members, which is how champion_1 --
    the one agent that counters our current champion -- stopped being played
    thirty promotions ago. Rating low against the field and beating *us* are
    different facts, and only the second one matters.
    """
    p = five()
    seen: set[str] = set()

    for seed in range(60):
        seen |= set(p.sample({}, random.Random(seed)))

    assert seen == set("abcde")


def test_a_pool_smaller_than_the_draw_is_played_whole() -> None:
    """A cold start has twelve opponents and asks for sixteen."""
    p = five()

    drawn = p.sample({}, random.Random(0))

    assert sorted(drawn) == list("abcde")


def test_the_candidate_is_never_drawn_against_itself() -> None:
    """A champion is a member of the pool it is scored on.

    Its own bytes in the other seat are a structural 0.5 that says nothing
    about the program and would enter the rate, the rating and the gate.
    """
    p = five()

    drawn = p.sample({}, random.Random(0), exclude="c")

    assert "c" not in drawn


def test_round_trips_through_json(tmp_path: Path) -> None:
    """save() then load() reproduces the same pool."""
    p = five()
    p.add_champion("champ", "/x/champ.py")

    p.save(tmp_path / "pool.json")

    assert pool.Pool.load(tmp_path / "pool.json") == p
