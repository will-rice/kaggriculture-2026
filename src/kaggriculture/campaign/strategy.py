"""Claims about how the ladder's winners play, and the evidence for each.

The tables in ``winning_pace.md`` are a snapshot: true when they were measured
and unattached to any statement about why. A claim is the other thing -- a
sentence that could be wrong, with a form that says how to find out.

Every claim carries both. The prose is what a reader wrote about a game; the
form is what gets measured, and it is the form that decides. That order
matters, because a model asked what a winner did differently will write
something plausible whether or not it is true, and today it wrote that winners
keep planting to the last day. Measured over 43 paired games that is a 69%
tendency and not a rule -- which is a useful thing to know and a different
thing from what the sentence said.

A form is deliberately small: a quantity, a day, and which side leads. That is
enough to do three jobs at once. It verifies a claim, by counting the games
that agree. It selects one, by asking whether the program being judged is on
the wrong side of it. And it ranks what is selected, by how far off that
program is. Retrieval is arithmetic rather than a search over sentences, so
the store can grow without the message growing.

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

# What a claim may be about: the quantities the paired walk measures for both
# sides of a game. A claim naming anything else cannot be checked, and is
# refused rather than stored unverifiable.
QUANTITIES: frozenset[str] = frozenset(
    {
        "bank",
        "planted",
        "ripe",
        "pens",
        "hands",
        "quadrants",
        "shed",
        "seeds",
        "sells",
        "hires",
        "land",
    }
)
DAYS = 30
# Share of differing games a claim must reach to be confirmed, and the share
# below which it is refuted. Between them it stays proposed: 60% over twenty
# games is not evidence, and saying so is the point of keeping the count.
CONFIRM = 0.65
REFUTE = 0.35
SUPPORT = 25

# The store the round prompt reads, kept in the package beside
# ``winning_pace.md`` and for the same reason: both are measurements of the
# public ladder rather than of any one run, so they belong to the code that
# reads them and not to a run directory that a fresh campaign starts without.
# Rebuilt and re-measured by ``uv run strategies``.
STORE = Path(__file__).with_name("strategies.jsonl")


class Form(BaseModel):
    """The measurable half of a claim: who leads on what, and when.

    Attributes:
        quantity: One of `QUANTITIES`, measured for both sides at a day's end.
        day: The day the comparison is made on.
        winner_leads: Whether the claim is that the winner is ahead on it.
            False is a real claim and often the interesting one -- winners
            hold *less* bank on day eight and issue *fewer* hire orders.
    """

    quantity: str
    day: int
    winner_leads: bool

    def holds(self, winner: float, loser: float) -> bool | None:
        """Whether one game agrees, or None when it cannot say.

        A game where both sides are equal on this quantity is not evidence
        either way and is left out of the count rather than scored as a half.
        """
        if winner == loser:
            return None
        return (winner > loser) is self.winner_leads

    def wrong_side(self, ours: float, theirs: float) -> bool:
        """Whether a program being judged is on the losing side of this claim.

        What makes a claim worth showing a round: not that it is true, but
        that this program is not doing it.
        """
        if ours == theirs:
            return False
        return (ours > theirs) is not self.winner_leads


class Claim(BaseModel):
    """One statement about winning play, with what has been measured of it.

    Attributes:
        id: Stable identifier, so evidence can be recorded against it later.
        form: The measurable statement. This is what decides.
        prose: What a reader said about it. Commentary on the form, never
            the other way round.
        source: Episode seeds the claim was proposed from.
        support: Games measured in which the two sides differed.
        agreement: Share of those that agreed with the form.
        created: Unix timestamp.
    """

    id: str
    form: Form
    prose: str
    source: list[int] = []
    support: int = 0
    agreement: float = 0.0
    created: float

    @property
    def status(self) -> Literal["proposed", "confirmed", "refuted"]:
        """Confirmed, refuted, or still waiting on evidence."""
        if self.support < SUPPORT:
            return "proposed"
        if self.agreement >= CONFIRM:
            return "confirmed"
        if self.agreement <= REFUTE:
            return "refuted"
        return "proposed"

    def says(self) -> str:
        """Exactly what was compared, and what the corpus said.

        Deliberately not a sentence: it sits beside the prose wherever it is
        shown, and two readable sentences saying the same thing read as
        emphasis rather than as a claim and its evidence.
        """
        side = "ahead" if self.form.winner_leads else "behind"
        return (
            f"day {self.form.day}, `{self.form.quantity}`, winner {side}: "
            f"{self.agreement:.0%} of {self.support} games"
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

    def propose(self, form: Form, prose: str, source: list[int]) -> Claim:
        """Record a new claim, unmeasured.

        Returns the claim, or the one already stored when an identical form
        has been proposed before: the same statement arrived at twice is one
        statement with more sources, not two claims to measure separately.
        """
        for claim in self._claims.values():
            if claim.form == form:
                return claim
        claim = Claim(
            id=uuid.uuid4().hex[:12],
            form=form,
            prose=prose.strip(),
            source=source,
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

    def confirmed(self) -> list[Claim]:
        """Confirmed claims, strongest evidence first."""
        return sorted(
            (claim for claim in self._claims.values() if claim.status == "confirmed"),
            key=lambda claim: (-claim.agreement, -claim.support),
        )
