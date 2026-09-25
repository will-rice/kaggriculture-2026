"""Putting the claims to the corpus, over the corpus they are claims about.

The nightly job measures every claim against the recorded ladder and rebuilds
the build order the round prompt reads. It stopped doing either on 2026-09-14,
the first run after `one games database, no SQLite, no users` reached main:
`evidence` still opened `sqlite3.connect(database)` while its caller's default
had become the ClickHouse database *name*, so the job created an empty file
called `games` in its worktree and died on `no such table: days`. Nothing in
the suite called `measure`, so nothing said so.
"""

import pytest

from kaggriculture.campaign import evidence, games, strategy

from .fixtures import live

# A rating gap wider than `evidence.MARGIN`, so every pair below counts as a
# comparison between a stronger agent and a weaker one.
STRONGER, WEAKER = 2.0, 0.0


@pytest.mark.local_data
def test_every_claim_is_measured_against_the_games_database() -> None:
    """`measure` answers, at all, from the database the campaign actually has.

    The regression above was a `sqlite3` call left in the one module every
    other caller had moved off, and it survived because no test made the call.
    This one does, over the real corpus, and asserts the shape of what comes
    back rather than any particular number: the corpus grows nightly and a
    pinned agreement would be stale by morning.
    """
    store = strategy.Strategies(strategy.STORE)
    for form in evidence.questions()[:8]:
        store.propose(form)

    measured = evidence.measure(store)

    assert measured, "no claim was measured at all"
    for claim, (support, agreement) in measured.items():
        assert support > 0, f"{claim} recorded with no games behind it"
        assert 0.0 <= agreement <= 1.0, f"{claim} agreed {agreement}"


@live
def test_a_claim_is_measured_over_the_ladder_and_not_our_own_games(
    scratch: str,
) -> None:
    """The campaign's own games must not count as the field's.

    Both writers share `days` and the campaign owns 18.6M of its 19.8M rows, so
    a claim measured over the table rather than over the ladder is a claim
    about this lineage playing itself.

    The two games here are identical but for `source`, and they disagree: the
    stronger side leads on the ladder and trails in the campaign. Filtered,
    that is one comparison the stronger side wins; unfiltered it is one of two.

    They share their team names and carry the same date deliberately, because
    that is the collision the filter is for and nothing else here would catch
    it. Against the real corpus two unrelated things already exclude our games
    -- `teams` holds 60 ladder display names and none of the campaign's 294,
    and campaign episodes carry no date to pass the window -- but neither says
    so, and both could stop being true: our own agents are rated elsewhere
    already, and the campaign plays harvested kernels under slugs built from
    the same accounts those display names belong to.
    """
    games.query(
        f"INSERT INTO {scratch}.teams (team, games, wins, rating, place) VALUES "
        f"('strong', 2, 2, {STRONGER}, 1), ('weak', 2, 0, {WEAKER}, 2)"
    )
    games.query(
        f"INSERT INTO {scratch}.episodes "
        "(episode, played, team_0, team_1, source, version) VALUES "
        "('e_ladder', '2026-09-10', 'strong', 'weak', 'ladder', 1), "
        "('e_campaign', '2026-09-10', 'strong', 'weak', 'campaign', 1)"
    )
    # Day ten of each game: on the ladder the stronger side banks more, in the
    # campaign game it banks less. One claim, two answers, and which one comes
    # back is the whole question.
    games.query(
        f"INSERT INTO {scratch}.days "
        "(episode, seat, day, team, source, bank, version) VALUES "
        "('e_ladder', 0, 10, 'strong', 'ladder', 900, 1), "
        "('e_ladder', 1, 10, 'weak', 'ladder', 100, 1), "
        "('e_campaign', 0, 10, 'strong', 'campaign', 100, 1), "
        "('e_campaign', 1, 10, 'weak', 'campaign', 900, 1)"
    )

    ahead, differing = evidence._day(10, ["bank"], "2026-09-01", scratch)["bank"]

    assert (ahead, differing) == (1, 1), (
        "the campaign's own game was counted as the ladder's: "
        f"{ahead} of {differing} rather than 1 of 1"
    )
