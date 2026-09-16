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


def test_the_leader_is_drawn_however_the_dice_fall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Topping the field means beating the best of it.

    Left to the draw, a candidate could be turned away for never having met
    the leader -- rejected for the sampler's luck rather than for anything it
    did. The floor rides along for the same reason: the promotion bar is a
    rating gap over that one named agent.
    """
    monkeypatch.setattr(config, "GATE_ANCHORS", ())
    monkeypatch.setattr(config, "GATE_OPPONENTS", 2)
    monkeypatch.setattr(config, "GATE_CONTENDERS", 0)
    p = five()

    for seed in range(30):
        drawn = p.sample({}, random.Random(seed), always=["a", "e"])
        assert drawn[:2] == ["a", "e"]


def test_an_always_name_the_pool_does_not_hold_is_ignored() -> None:
    """A cold start has no floor and no leader, and asks for both."""
    p = five()

    drawn = p.sample({}, random.Random(0), always=["nobody", "a"])

    assert "nobody" not in drawn and "a" in drawn


def test_the_candidate_is_not_drawn_against_itself_even_when_always_named() -> None:
    """A champion is the floor *and* a pool member, and would be drawn twice.

    Its own bytes in the other seat are a structural 0.5 that says nothing
    about the program, and the gate would then read a rating gap of zero
    against itself.
    """
    p = five()

    drawn = p.sample({}, random.Random(0), exclude="c", always=["c", "a"])

    assert "c" not in drawn and "a" in drawn


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
    p.add_champion("/x/champ.py")

    p.save(tmp_path / "pool.json")

    assert pool.Pool.load(tmp_path / "pool.json") == p


def test_every_vendored_opponent_is_drawn_into_every_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public agents are the only evidence about the field we are scored on.

    Left to the sampler they starved. Measured 2026-09-09: the two vendored
    opponents that happen to be anchors held 33 and 31 pairings, four harvested
    on 09-06 held one between them, and one had never been played at all.
    Champions took every contender slot, because the lineage rates itself
    highest -- so the draw fed itself and the outsiders were squeezed into a
    random remainder of about three slots.
    """
    monkeypatch.setattr(roster, "TRAINING", {"a": "/x/a.py", "b": "/x/b.py"})
    monkeypatch.setattr(config, "GATE_ANCHORS", ())
    monkeypatch.setattr(config, "GATE_CONTENDERS", 1)
    monkeypatch.setattr(config, "GATE_OPPONENTS", 4)
    p = five()

    for seed in range(20):
        drawn = p.sample({"c": 9.0, "d": 8.0}, random.Random(seed))
        assert {"a", "b"} <= set(drawn), f"vendored opponent missed on seed {seed}"


def test_the_floor_survives_a_draw_too_small_to_hold_everyone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Truncation may drop a contender or a vendored agent, never the floor.

    The draw is cut to `GATE_OPPONENTS` at the end, and a promotion is a rating
    gap over the floor -- a gap against an agent never played is not measurable
    at all. So `always` is ordered first, and the cut can only reach past it.
    """
    monkeypatch.setattr(roster, "TRAINING", {n: f"/x/{n}.py" for n in "abcde"})
    monkeypatch.setattr(config, "GATE_ANCHORS", ("a", "b"))
    monkeypatch.setattr(config, "GATE_CONTENDERS", 2)
    # Smaller than the vendored set alone, so something must be dropped.
    monkeypatch.setattr(config, "GATE_OPPONENTS", 2)
    p = five()

    drawn = p.sample({"c": 1.0}, random.Random(0), always=("d",))

    assert "d" in drawn
    assert len(drawn) == 2


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
