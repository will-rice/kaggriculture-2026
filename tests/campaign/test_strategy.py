"""A claim is what could be wrong; the form is what decides whether it is."""

from pathlib import Path

import pytest

from kaggriculture.campaign import strategy


def form(quantity: str = "planted", day: int = 20, leads: bool = True) -> strategy.Form:
    """A measurable statement, defaulted to one that is true."""
    return strategy.Form(quantity=quantity, day=day, winner_leads=leads)


def test_a_game_that_ties_on_the_quantity_is_not_evidence() -> None:
    """Neither for nor against: it is left out of the count, not halved.

    Counting ties as disagreement would refuse every claim about a quantity
    the two sides usually match on, and counting them as agreement would
    confirm any claim at all about one.
    """
    rule = form()

    assert rule.holds(winner=10, loser=4) is True
    assert rule.holds(winner=4, loser=10) is False
    assert rule.holds(winner=7, loser=7) is None


def test_a_claim_can_be_that_the_winner_trails() -> None:
    """Often the interesting one.

    Winners hold *less* bank on day eight and issue *fewer* hire orders than
    the side that loses; a store that could only say "more" could not hold
    either finding.
    """
    rule = form("bank", day=8, leads=False)

    assert rule.holds(winner=3_731, loser=8_187) is True
    assert rule.holds(winner=8_187, loser=3_731) is False


def test_a_claim_selects_the_program_that_is_not_doing_it() -> None:
    """What makes a claim worth showing is not that it is true.

    It is that this program is on the wrong side of it. That is what keeps
    the message the size of the gap rather than the size of the store.
    """
    rule = form("planted", day=20, leads=True)

    assert rule.wrong_side(ours=12, theirs=58) is True
    assert rule.wrong_side(ours=58, theirs=12) is False
    assert rule.wrong_side(ours=30, theirs=30) is False


def test_a_claim_about_something_unmeasurable_is_refused(tmp_path: Path) -> None:
    """The store is claims with evidence; one that cannot be checked is prose.

    Refused at the point of writing rather than stored and quietly never
    measured, because a claim nobody can test still reads like a finding to
    whoever is handed it.
    """
    store = strategy.Strategies(tmp_path / "strategies.jsonl")

    with pytest.raises(ValueError, match="unmeasurable"):
        store.propose(form("vibes"), "winners have better vibes", [1])
    with pytest.raises(ValueError, match="outside the season"):
        store.propose(form(day=44), "on day forty-four", [1])


def test_evidence_decides_the_status_and_prose_never_does(tmp_path: Path) -> None:
    """The form decides. The sentence is commentary on it.

    Written the other way round, a model that says "winners keep planting to
    the last day" would have produced a rule; measured, that is a 69%
    tendency over 43 games. Both are worth having and they are not the same.
    """
    store = strategy.Strategies(tmp_path / "strategies.jsonl")
    claim = store.propose(form(), "winners keep planting to the last day", [11, 22])

    assert claim.status == "proposed"  # nothing measured yet
    store.record(claim.id, support=10, agreement=1.0)
    assert claim.status == "proposed"  # ten games is not evidence
    store.record(claim.id, support=43, agreement=0.69)
    assert claim.status == "confirmed"
    store.record(claim.id, support=400, agreement=0.31)
    assert claim.status == "refuted"
    assert "69%" not in claim.says() and "31%" in claim.says()


def test_the_same_form_twice_is_one_claim_with_two_sources(tmp_path: Path) -> None:
    """Two readers arriving at one statement is more evidence, not more work."""
    store = strategy.Strategies(tmp_path / "strategies.jsonl")

    first = store.propose(form(), "expand early", [1])
    again = store.propose(form(), "worded quite differently", [2])

    assert again.id == first.id
    assert len(store.claims) == 1


def test_the_log_survives_a_restart_and_keeps_both_measurements(
    tmp_path: Path,
) -> None:
    """A claim confirmed at sixty games and refuted at four hundred keeps both.

    The second does not erase the first: what changed is the evidence, and a
    store that overwrote it would lose the fact that the corpus once said
    otherwise.
    """
    path = tmp_path / "strategies.jsonl"
    store = strategy.Strategies(path)
    claim = store.propose(form(), "expand early", [1])
    store.record(claim.id, support=60, agreement=0.80)
    store.record(claim.id, support=400, agreement=0.30)

    again = strategy.Strategies(path)

    assert again.get(claim.id).status == "refuted"
    assert path.read_text(encoding="utf-8").count('"evidence"') == 2


def test_confirmed_claims_come_back_strongest_first(tmp_path: Path) -> None:
    """A round sees the best-evidenced first, because the message has a budget."""
    store = strategy.Strategies(tmp_path / "strategies.jsonl")
    weak = store.propose(form("hands", 12), "hands", [1])
    strong = store.propose(form("planted", 12), "planted", [1])
    unmeasured = store.propose(form("shed", 12), "shed", [1])
    store.record(weak.id, support=40, agreement=0.70)
    store.record(strong.id, support=40, agreement=0.95)

    assert [claim.id for claim in store.confirmed()] == [strong.id, weak.id]
    assert unmeasured.id not in {claim.id for claim in store.confirmed()}
