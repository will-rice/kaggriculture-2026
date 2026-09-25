"""A claim is a question put to the corpus; the corpus supplies the answer.

The direction is the thing this module exists to keep out of a caller's
hands. The first version let a form carry its own answer, so a plausible
sentence could be written and then confirmed -- and eleven of them were,
before a bigger corpus refuted ten.
"""

from pathlib import Path

import pytest

from kaggriculture.campaign import strategy


def form(quantity: str = "planted", day: int = 20) -> strategy.Form:
    """A question: what to compare, and when. No answer attached."""
    return strategy.Form(quantity=quantity, day=day)


def claim(
    store: strategy.Strategies,
    quantity: str = "planted",
    day: int = 20,
    support: int = 4000,
    agreement: float = 0.8,
) -> strategy.Claim:
    """A claim the corpus has already spoken about."""
    made = store.propose(form(quantity, day))
    store.record(made.id, support=support, agreement=agreement)
    return made


def test_a_form_carries_no_direction(tmp_path: Path) -> None:
    """The flaw the whole rewrite is about.

    "Winners carry more in the shed" was written, then measured, then
    confirmed at 82% of 49 games -- and over thirteen thousand games it is
    50%. Writing the hypothesis and grading it are the same act when the
    direction is proposed, so a form names a quantity and a day and stops.
    """
    assert set(form().model_dump()) == {"quantity", "day"}


def test_the_corpus_decides_which_way_it_goes(tmp_path: Path) -> None:
    """Above the bar the stronger side leads; below it, the stronger trails.

    Both are findings, and the second is often the interesting one: the
    strong agents sell *less* than the weak ones, which no one would have
    proposed.
    """
    store = strategy.Strategies(tmp_path / "strategies.jsonl")

    leads = claim(store, "planted", 5, agreement=0.80)
    trails = claim(store, "sold_units", 20, agreement=0.20)
    neither = claim(store, "bank", 8, agreement=0.52)

    assert leads.status == "leads" and leads.ahead() is True
    assert trails.status == "trails" and trails.ahead() is False
    assert neither.status == "open"
    with pytest.raises(ValueError, match="is open"):
        neither.ahead()


def test_a_claim_nobody_has_measured_enough_says_nothing(tmp_path: Path) -> None:
    """Support is a bar, not a formality.

    Eight of eleven claims cleared at sixty games and one survived fourteen
    thousand. A share without a count behind it reads exactly like a finding.
    """
    store = strategy.Strategies(tmp_path / "strategies.jsonl")

    thin = claim(store, support=strategy.SUPPORT - 1, agreement=0.95)

    assert thin.status == "open"
    assert not thin.settled
    store.record(thin.id, support=strategy.SUPPORT, agreement=0.95)
    assert thin.settled


def test_an_open_claim_never_puts_a_program_on_a_side(tmp_path: Path) -> None:
    """With no direction there is no wrong side to be on.

    A claim that cannot say which way it goes must not be shown to a round as
    though it could.
    """
    store = strategy.Strategies(tmp_path / "strategies.jsonl")
    open_claim = claim(store, agreement=0.5)
    settled = claim(store, "shed", 12, agreement=0.8)

    assert open_claim.wrong_side(ours=1, theirs=99) is False
    # The stronger side holds more shed, and this program holds less.
    assert settled.wrong_side(ours=1, theirs=99) is True
    assert settled.wrong_side(ours=99, theirs=1) is False
    assert settled.wrong_side(ours=5, theirs=5) is False


def test_a_trailing_claim_puts_the_program_that_does_more_on_the_wrong_side(
    tmp_path: Path,
) -> None:
    """The direction has to reach selection, or half the findings invert.

    The strong agents sell less. A program that sells more is the one that
    needs telling, and reading the direction off the agreement rather than
    off a stored flag is what makes that come out right.
    """
    store = strategy.Strategies(tmp_path / "strategies.jsonl")
    sells_less = claim(store, "sold_units", 20, agreement=0.15)

    assert sells_less.wrong_side(ours=5000, theirs=2000) is True
    assert sells_less.wrong_side(ours=2000, theirs=5000) is False


def test_a_claim_about_something_the_dataset_cannot_measure_is_refused(
    tmp_path: Path,
) -> None:
    """The store is claims with evidence; one that cannot be checked is prose.

    Refused at the point of writing rather than stored and quietly never
    measured, because a claim nobody can test still reads like a finding to
    whoever is handed it.
    """
    store = strategy.Strategies(tmp_path / "strategies.jsonl")

    with pytest.raises(ValueError, match="unmeasurable"):
        store.propose(form("vibes"))
    with pytest.raises(ValueError, match="outside the season"):
        store.propose(form(day=44))


def test_the_quantities_are_the_datasets_own_columns() -> None:
    """One list, read off the schema, so the two cannot drift apart.

    A column added to the extraction is a question that can be asked about
    it, with nothing to remember to update -- and a claim can never name a
    column that is not there.
    """
    assert "sold_units" in strategy.QUANTITIES
    assert "watered" in strategy.QUANTITIES
    assert "bank" in strategy.QUANTITIES
    # The keys of a day row are not quantities to compare sides on.
    assert not {"episode", "seat", "day", "team"} & strategy.QUANTITIES


def test_prose_is_attached_after_the_measurement_and_never_decides(
    tmp_path: Path,
) -> None:
    """A sentence is commentary on a finding, not a way of making one.

    A claim with no sentence is still shown, because the measurement is the
    finding; a sentence can only ever be added to one the corpus settled.
    """
    store = strategy.Strategies(tmp_path / "strategies.jsonl")
    made = claim(store, "shed", 12, support=4102, agreement=0.78)

    assert made.reads() == made.says()
    assert "78% of 4102 games" in made.says()

    store.describe(made.id, "the strong agents keep a fuller shed")
    assert made.reads().startswith("the strong agents keep a fuller shed -- ")
    assert made.says() in made.reads()


def test_the_same_question_asked_twice_is_one_claim(tmp_path: Path) -> None:
    """The grammar is enumerated every run, so re-proposing is the normal case."""
    store = strategy.Strategies(tmp_path / "strategies.jsonl")

    first = store.propose(form())
    again = store.propose(form())

    assert again.id == first.id
    assert len(store.claims) == 1


def test_the_log_survives_a_restart_and_keeps_both_measurements(
    tmp_path: Path,
) -> None:
    """A claim settled over one corpus and undone over a bigger one keeps both.

    What changed is the evidence, and a store that overwrote it would lose
    the fact that the corpus once said otherwise -- which is the single most
    useful thing this store has recorded.
    """
    path = tmp_path / "strategies.jsonl"
    store = strategy.Strategies(path)
    made = claim(store, support=600, agreement=0.80)
    store.record(made.id, support=13431, agreement=0.50)

    again = strategy.Strategies(path)

    assert again.get(made.id).status == "open"
    assert path.read_text(encoding="utf-8").count('"evidence"') == 2


def test_settled_claims_come_back_by_how_far_they_separate_the_sides(
    tmp_path: Path,
) -> None:
    """A round has a budget, so the clearest separation is shown first.

    Distance from a coin flip, not raw agreement: a claim at 15% separates
    the sides exactly as sharply as one at 85%, and reading it as the weaker
    of the two would bury every finding about doing less of something.
    """
    store = strategy.Strategies(tmp_path / "strategies.jsonl")
    mild = claim(store, "planted", 5, agreement=0.70)
    sharp_low = claim(store, "sold_units", 20, agreement=0.10)
    sharp_high = claim(store, "shed", 12, agreement=0.93)
    unmeasured = store.propose(form("weeds", 29))

    order = [c.id for c in store.settled()]

    assert order == [sharp_high.id, sharp_low.id, mild.id]
    assert unmeasured.id not in order
