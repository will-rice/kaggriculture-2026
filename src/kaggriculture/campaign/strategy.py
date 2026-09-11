"""Claims about how the ladder's strongest agents play, and the evidence.

The tables in ``build_order.md`` are a snapshot: true when they were measured
and unattached to any statement about why. A claim is the other thing -- a
statement that could be wrong, with a form that says how to find out.

Two things decide whether such a statement means anything, and the first
version of this got the second one wrong.

**Paired.** A claim is measured inside single games, comparing the two sides
at the same day's close. Same map, same prices, same opponent, so what is left
when they differ is what the two players did. An average over the corpus would
compare farms that never met.

**Rated.** The comparison is between the stronger and the weaker *agent*, not
between the winner and the loser of that game. Half of every ladder's winners
are the weaker side having a good day, and measured that way eleven claims
over thirteen thousand games all came back between 45% and 60% -- noise, and
one of them was the reverse of what the same corpus says when the sides are
compared by rating instead.

**The direction is measured, not proposed.** A form names a quantity and a
day; it does not say which way it goes. That is what the corpus answers, and
the reason is the whole history of this module: a person writing "winners
carry more in the shed" and then confirming it has proposed a hypothesis and
graded their own paper. A form is a question, an agreement above `CONFIRM`
says the stronger side leads, one below `REFUTE` says it trails, and anything
between says the quantity does not separate them.

The log is append-only and replayed, like the program database: a claim
confirmed over sixty games and refuted over four hundred keeps both records,
and the second does not erase the first.
"""

import json
import time
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from kaggriculture.campaign import config, dataset

# What a claim may be about: every column of a day row that is a number, which
# is every quantity the extraction measures for both sides. A claim naming
# anything else cannot be checked and is refused rather than stored
# unverifiable.
#
# Taken from `dataset.COLUMNS` rather than listed here, so a quantity added to
# what the campaign measures is a claim that can be made about it, and there is
# no second list to keep in step. It used to be parsed out of a CREATE TABLE,
# which is what there was to read before the measures had a list of their own.
QUANTITIES: frozenset[str] = frozenset(dataset.COLUMNS)
DAYS = dataset.DAYS
# The store the round prompt reads, beside the corpus it is measured from.
#
# Not inside the package, though it started there next to ``build_order.md``.
# That works for a file committed once and read by whoever has the checkout,
# and breaks the moment it is rebuilt daily: the job runs from its own
# worktree with its own copy of the package, so it would rewrite a store in
# one checkout while the campaign read a stale one from another. A measurement
# of the public ladder belongs with the ladder's other measurements.
STORE = config.EPISODES.parent / "strategies.jsonl"
# Share of differing games at which a claim is settled one way or the other.
# Between them it says nothing: 60% over twenty games is not evidence, and
# saying so is the point of keeping the count.
CONFIRM = 0.65
REFUTE = 0.35
SUPPORT = 200


class Form(BaseModel):
    """The measurable half of a claim: what to compare, and when.

    Deliberately without a direction. `winner_leads` used to live here and it
    was the flaw: a form carrying its own answer lets a plausible sentence be
    written first and confirmed second, which is how "winners keep planting to
    the last day" became a finding at 69% and noise at 47%.

    Attributes:
        quantity: One of `QUANTITIES`, compared for both sides at a day's end.
        day: The day the comparison is made on.
    """

    quantity: str
    day: int

    def leads(self, ours: float, theirs: float) -> bool | None:
        """Whether we are ahead on this quantity, or None when level.

        Level is not evidence either way and is left out of the count rather
        than scored as a half.
        """
        if ours == theirs:
            return None
        return ours > theirs


class Claim(BaseModel):
    """One statement about strong play, with what has been measured of it.

    Attributes:
        id: Stable identifier, so evidence can be recorded against it later.
        form: What is compared. This is what decides.
        prose: What a reader said about it, or "" for a form nobody has
            written up. Commentary on the measurement, never the other way
            round.
        support: Games measured in which the two sides differed and one was
            rated clearly above the other.
        agreement: Share of those in which the stronger side led.
        created: Unix timestamp.
    """

    id: str
    form: Form
    prose: str = ""
    support: int = 0
    agreement: float = 0.0
    created: float

    @property
    def status(self) -> Literal["open", "leads", "trails"]:
        """What the corpus says, or "open" while it has not said anything."""
        if self.support < SUPPORT:
            return "open"
        if self.agreement >= CONFIRM:
            return "leads"
        if self.agreement <= REFUTE:
            return "trails"
        return "open"

    @property
    def settled(self) -> bool:
        """Whether the corpus separates the two sides on this at all."""
        return self.status != "open"

    def ahead(self) -> bool:
        """Whether the stronger side is the one with more of this.

        Raises:
            ValueError: The claim is open, so there is no direction to give
                and a caller guessing one would be inventing a finding.
        """
        if not self.settled:
            raise ValueError(f"{self.form.quantity} on day {self.form.day} is open")
        return self.status == "leads"

    def wrong_side(self, ours: float, theirs: float) -> bool:
        """Whether a program being judged plays this the other way round.

        What makes a claim worth showing a round: not that it is true, but
        that this program is not doing it.
        """
        if ours == theirs or not self.settled:
            return False
        return (ours > theirs) is not self.ahead()

    def reads(self) -> str:
        """The claim as one cell: the measurement, and any sentence about it.

        The measurement first and always. A claim with nobody's sentence
        attached is still a finding; a sentence without the measurement is
        what this module exists to stop being one.
        """
        return f"{self.prose} -- {self.says()}" if self.prose else self.says()

    def says(self) -> str:
        """Exactly what was compared, and what the corpus said.

        Deliberately not a sentence: it sits beside the prose wherever it is
        shown, and two readable sentences saying the same thing read as
        emphasis rather than as a claim and its evidence.
        """
        side = "more" if self.status == "leads" else "less"
        share = self.agreement if self.status == "leads" else 1 - self.agreement
        return (
            f"day {self.form.day}, `{self.form.quantity}`, stronger side has "
            f"{side}: {share:.0%} of {self.support} games"
        )


class Strategies:
    """Every claim and every measurement of one, backed by an append-only log.

    Each change appends one event and folds that same event into memory, so
    the log is the source of truth and replaying it rebuilds the state.
    """

    def __init__(self, path: Path) -> None:
        """Open the store at ``path``, replaying it if it exists."""
        self.path = path
        self._claims: dict[str, Claim] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                self._apply(json.loads(line), persist=False)

    def _apply(self, event: dict, persist: bool = True) -> None:
        """Fold one event into state and, unless replaying, append it.

        The fold comes first, so an event that cannot be applied never
        reaches the log and a restart replays only what this same code has
        already accepted.

        Raises:
            ValueError: An unknown event, or a duplicate claim id.
            KeyError: Evidence for a claim the store does not hold.
        """
        kind = event["event"]
        if kind == "claim":
            claim = Claim.model_validate(event["claim"])
            if claim.id in self._claims:
                raise ValueError(f"duplicate claim id: {claim.id!r}")
            if claim.form.quantity not in QUANTITIES:
                raise ValueError(f"unmeasurable quantity: {claim.form.quantity!r}")
            if not 0 <= claim.form.day < DAYS:
                raise ValueError(f"day outside the season: {claim.form.day}")
            self._claims[claim.id] = claim
        elif kind == "evidence":
            claim = self.get(event["id"])
            claim.support = int(event["support"])
            claim.agreement = float(event["agreement"])
        elif kind == "prose":
            self.get(event["id"]).prose = str(event["prose"]).strip()
        else:
            raise ValueError(f"unknown strategy event: {kind!r}")
        if persist:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")

    @property
    def claims(self) -> list[Claim]:
        """Every claim, in the order it was proposed."""
        return list(self._claims.values())

    def get(self, claim_id: str) -> Claim:
        """The claim with that id.

        Raises:
            KeyError: No claim has it.
        """
        if claim_id not in self._claims:
            raise KeyError(claim_id)
        return self._claims[claim_id]

    def propose(self, form: Form, prose: str = "") -> Claim:
        """Record a question to put to the corpus, unmeasured.

        Returns the claim, or the one already stored when the same form has
        been proposed before: the same comparison arrived at twice is one
        question, not two to measure separately.
        """
        for claim in self._claims.values():
            if claim.form == form:
                return claim
        claim = Claim(
            id=uuid.uuid4().hex[:12],
            form=form,
            prose=prose.strip(),
            created=time.time(),
        )
        self._apply({"event": "claim", "claim": claim.model_dump()})
        # The stored one, not the one built here: `_apply` validates a copy
        # into the log, so returning this object would hand the caller a claim
        # that never sees its own evidence.
        return self._claims[claim.id]

    def record(self, claim_id: str, support: int, agreement: float) -> None:
        """Attach a measurement to a claim, replacing whatever it had.

        Replacing rather than accumulating, because the measurement is over
        the whole corpus as it now stands and a later one is simply a better
        estimate. The log keeps both.
        """
        self._apply(
            {
                "event": "evidence",
                "id": claim_id,
                "support": support,
                "agreement": agreement,
            }
        )

    def describe(self, claim_id: str, prose: str) -> None:
        """Attach a sentence to a claim the corpus has already settled.

        Written after the measurement and never before it, which is the whole
        arrangement: a sentence cannot become a finding by being persuasive.
        """
        self._apply({"event": "prose", "id": claim_id, "prose": prose})

    def settled(self) -> list[Claim]:
        """Every claim the corpus separates the sides on, strongest first."""
        return sorted(
            (claim for claim in self._claims.values() if claim.settled),
            key=lambda claim: (-abs(claim.agreement - 0.5), -claim.support),
        )
