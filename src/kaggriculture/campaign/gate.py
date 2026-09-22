"""The one place a candidate becomes the floor.

The gate is a tournament. Every agent plays every other -- the candidate
against the pool and the pool against itself -- one Bradley-Terry fit ranks
them all, and a candidate is promoted when it comes out top. That is the
competition's own reading of better: the finale is a single Bradley-Terry
tournament over the episodes that keep running past the deadline, and a
leaderboard position is a skill rating.

It used to ask something else -- beat *every* opponent -- which is a minimum
where the ladder takes a strength-weighted view. Measured on the pool as it
stood on 2026-09-06, exactly one published agent cleared that bar, and the
second-strongest agent in the whole field was turned away for a single
matchup at 0.062 while winning 78.6% of everything else.

The pool is dynamic -- a champion joins and the weakest opponent makes way --
and the tournament is over the pool as it stands. What is not replayed is the
pool's games against itself: those are constants, so they are measured once
and kept, and only a new member's pairings ever run. The candidate's own row
is always played fresh, because a candidate has no history.

Champions accumulating in the pool are what makes the chain a ratchet: each
promotion came top of a field that already held every champion before it,
on seeds drawn fresh for that tournament and never seen by the lineage
being judged.

A promotion also produces the artefact a cut uploads: the program is
packaged into `champions/<name>.tar.gz`, so a cut is one command -- upload
`champion.tarball` -- and nothing is built at cut time. What proves a
candidate runs is validation, which loads it through Kaggle's own loader and
plays a full episode; everything else in the tarball is the same bytes for
every candidate, so proving the packaging is the packager's own tests' job
and not something to re-run on each promotion.

A promotion is two halves, because the campaign runs one event loop over
eight workers. `promote` is the file work -- packaging, the champion's own
copy, the floor -- and goes in a thread. `enroll` and `record` are the
shared state -- the pool the champion joins, and the file that says it is
the floor -- and stay on the loop thread, because `Pool` is one object the
other seven workers are reading while this one promotes.

Nothing here touches version control, and every write is under
`run/campaign`. In particular nothing writes into `src/`: the committed
`served/main.py` is the seed a cold start begins from, not the floor a
campaign produces, and a promotion that dirtied a tracked file would leave
a tree the next launch refuses to start on.
"""

import logging
import math
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

import scipy.stats
from pydantic import BaseModel

from kaggriculture.campaign import config, harness, rating, roster
from kaggriculture.campaign.archive import Program
from kaggriculture.campaign.evaluator import Result
from kaggriculture.campaign.pool import Pool
from kaggriculture.report import wilson_interval

LOGGER = logging.getLogger(__name__)


class Champion(BaseModel):
    """The promoted floor, as the runs ``champion.json`` records it.

    ``record`` writes it the moment the champion is in the pool, and it is
    preferred over ``state.json`` on restart: ``state.json`` is written after
    every completed session, so a kill in between would otherwise lose the
    champion the pool file already names.

    Attributes:
        name: The champion's pool name, e.g. "champion_3".
        path: The immutable copy under the runs ``champions`` the pool plays.
        tarball: The archive a cut uploads, written by this promotion. Empty
            for champion zero, which is the seed enthroned at startup so that
            there is no pre-champion regime: nothing would ever submit it, and
            packaging needs the licence and the served skeleton, which a bare
            run directory does not have.
        result: The measurement it was promoted on, which is also what a
            session is shown of the program it starts from.
    """

    name: str
    path: str
    tarball: str
    result: Result


def refresh(
    name: str,
    against: list[str],
    seeds: Sequence[int],
    workers: int,
    paths: config.Run,
) -> list[tuple[str, str]]:
    """Play one agent's missing edges against ``against``, and keep them.

    `standing` fits a rating over pairings that already exist and plays
    nothing. That is what makes a verdict cheap enough to give every round --
    and it means a pairing nobody has played is simply absent from the fit.

    Absent is fine for an opponent that has been around; it is wrong for a
    champion. A champion joins the pool the moment it is promoted with no
    pairings at all, so until they are played it sits in every fit on a
    single edge: the row of whichever candidate is being judged against it.
    Its rating is then inferred almost entirely from that one result -- beat
    it, and its rating falls far enough to make topping the field easy. Each
    promotion would buy the next one cheaply, which is the ratchet running
    backwards.

    It plays *one agent's* edges rather than every missing pair in the pool.
    Nothing leaves the pool now, so "every missing pair" grows with its
    square: sixty opponents is one thousand seven hundred and seventy
    pairings, and a promotion cannot cost forty thousand games. A new
    champion needs edges to the agents it will be compared against, and
    `Pool.sample` already says which those are.

    Args:
        name: The agent whose edges are wanted, normally a new champion.
        against: The opponents to connect it to.
        seeds: Episode seeds; each pairing is played on all of them, both
            seats, so a pairing is ``2 * len(seeds)`` games.
        workers: Processes to fan the games over.
        paths: The run whose field the pairings are kept in.

    Returns:
        The pairings measured, empty when the field already held them all.

    Raises:
        OpponentCrash: An opponent raised in its own seat.
    """
    games = 2 * len(seeds)
    field = rating.Field.load(paths.field)
    absent = [
        other
        for other in against
        if other != name and other not in field.rates.get(name, {})
    ]
    if not absent:
        return []
    LOGGER.info("field: %d pairing(s) for %s, measuring them", len(absent), name)
    for other in absent:
        field.record(
            name,
            other,
            _rate(roster.path(name, paths.pool), other, seeds, workers, paths.pool),
            games,
        )
    field.save(paths.field)
    return [(name, other) for other in absent]


def _rate(
    agent: Path,
    opponent: str,
    seeds: Sequence[int],
    workers: int,
    pool: Path | None = None,
) -> float:
    """``agent``'s win rate against ``opponent`` over ``seeds``, both seats."""
    played = harness.play(agent, [opponent], list(seeds), workers, pool=pool)
    return sum(
        1.0 if game.ours > game.theirs else 0.5 if game.ours == game.theirs else 0.0
        for game in played
    ) / len(played)


def standing(
    name: str,
    rates: dict[str, float],
    games: int,
    paths: config.Run,
) -> dict[str, float]:
    """The same tournament, over games already played: no new ones.

    A round has just measured its program against the opponents it drew, and
    every pairing anyone has ever played is kept, so the standings that
    verdict needs are a fit and nothing more. That is what lets the loop tell
    a model where it ranks after every round rather than only at the gate.

    Fitted over the whole record rather than over the pool as it stands. That
    distinction is what makes a sampled gate work at all: a candidate draws
    sixteen opponents out of dozens, and it is the pairings among the agents
    it did *not* draw that place it against them. Restricted to the drawn few,
    every candidate would be rated in a private tournament and the numbers
    would not compare.

    Args:
        name: The program's name in the standings.
        rates: Its win rate against each opponent it played.
        games: Games behind each rate.
        paths: The run whose field the pairings are kept in.

    Returns:
        A rating per agent, the program included.
    """
    field = rating.Field.load(paths.field)
    results = field.everything()
    results += [(name, two, rate, games) for two, rate in rates.items()]
    return rating.standings(results)


def promotion(
    result: Result,
    champion: "Champion | None",
    *,
    decisive_bar: int = config.DECISIVE_GAMES,
) -> tuple[bool, str]:
    """Whether the candidate is better than the champion, on every count.

    Three conditions, and all of them are required:

    1. **No worse against the field**, on two measures -- the win rate over the
       opponents both were measured against, and the mean final bank margin over
       the same ones. Behind on either by more than twice the error of the
       difference is a refusal; anything from level upward passes.
    2. **No matchup thrown away**: no single opponent the candidate has fallen
       behind the champion against by more than the joint noise, corrected for
       the number of opponents tested. `_slipped` carries why a mean cannot do
       this job -- on a field the champion sweeps, one matchup discarded moves
       the mean by a fraction of its own bar.
    3. **Beating the champion head-to-head**: the Wilson lower bound of its rate
       against the champion above 0.5, over at least ``decisive_bar`` decided
       games. That pairing is played at `config.DUEL_SEEDS` rather than at the
       sweep's `GATE_SEEDS`, because it is the only comparison here whose games
       decide anything: read off the sweep's 32 this condition needed a rate
       near 0.675 to clear, which is a rout rather than an improvement, and on
       2026-09-21 it turned away a candidate winning 0.594 of them.

    The margin joined the first condition on 2026-09-16, on a measurement
    published in the competition's own discussions: a round-robin of fourteen
    public implementations spanning a month, over 96 fresh seeds and both seats,
    found no intransitive triple, the newer implementation winning 86 of 91
    chronological pairs, and "the ordering tracks average final money remarkably
    closely". A stronger economy is a stronger agent here, which makes the
    margin evidence about strength rather than a consolation prize -- and it has
    no ceiling, where the win rate reached 1.000 on this pool the same day and
    stopped separating anything.

    Neither implies the other, and the campaign has now produced both failures.
    Condition 2 alone promoted `champion_3` at a field rate of 0.114 over a
    champion at 0.155, and `champion_8` at 0.221 over one at 0.244: two of nine
    promotions handing back field ground while winning the pairing decisively.
    Condition 1 alone would promote an agent that beats the field on average and
    loses to the specific program it replaces, which is not a ratchet.

    Condition 1 demanded a *higher* rate until 2026-09-16, and that stopped
    being a question a program could answer. champion_19 beat every agent in
    the pool: its rate was 1.000, nothing scores higher than 1.000, and the
    fifty candidates evaluated in the six hours after it were refused without
    the second condition ever being reached. Two of them, `p0d46cde4bbcd` and
    `p3c94d98c9d60`, beat champion_19 head-to-head 31-1 and 30-2 while sitting
    in the database as rejects, and seven programs stood at fitness 1.0000 with
    up to 1.000-against-0.000 between them -- a gate that called them identical
    was discarding the only signal that separated them.

    So the field rate keeps the job it can still do, which is refusing a
    regression, and the pairing does the discriminating. That is also the
    competition's own objective: wins against the agent you are matched with,
    not a coin total and not an average over a field you already beat.

    The rate is compared on common opponents because the two were measured in
    different seed blocks against a pool that harvest grows, so their own
    `fitness` figures are not the same question. Restricting to the opponents
    both played is the same discipline that turns a 0.331-against-24 and a
    0.221-against-33 into a comparable 0.283 and 0.214.

    It is still not paired -- different blocks means different maps, and the
    difference carries both measurements' noise. Twice the error of the
    difference is what stands in for that: at 41 opponents and 32 games each the
    standard error of a rate near 0.22 is 0.011, so the bar is about 0.023, and
    it tightens or loosens with the evidence rather than being a constant nobody
    checked against the noise.

    Args:
        result: The candidate's evaluation, which played the champion.
        champion: The champion it must beat, or None when there is none.
        decisive_bar: Decided games required before the pairing is read.

    Returns:
        Whether to promote, and a reason either way.
    """
    if champion is None:
        return False, "no champion to beat"
    name = champion.name
    if name not in result.rates:
        return False, f"did not play {name}"

    common = sorted((set(result.rates) & set(champion.result.rates)) - {name})
    if not common:
        return False, f"no opponent in common with {name}"
    # The mean is taken over the opponents the champion does not already
    # sweep. An opponent at 1.000 has no room above it, so it says nothing
    # about whether a candidate improved, and on 2026-09-21 there were 145 of
    # them against 31 that could move -- the difference between a mean of
    # 0.9661 and one of 0.8075, which is the difference between a number
    # nobody can read and one that tracks the only games left to win.
    #
    # It is not more sensitive, and that was measured rather than assumed: an
    # opponent at 1.000 contributes `p(1-p) = 0` variance as well as no room,
    # so dropping those terms shrinks the signal and the noise by the same
    # factor and the bar lands near one opponent-unit either way. What this
    # buys is legibility, not power. Power is what `_gained` below is for.
    #
    # Chosen by the *champion's* rates and never the candidate's, so a
    # candidate cannot pick its own denominator. A champion that sweeps
    # everything leaves it empty and the mean falls back to every shared
    # opponent -- the champion_19 case, where no program that could exist
    # scores higher and this condition has nothing left to say.
    held, why = _field(result, champion, common)
    if not held:
        return False, why
    keen = contested(champion, common)
    mine = sum(result.rates[one] for one in keen) / len(keen)
    theirs = sum(champion.result.rates[one] for one in keen) / len(keen)

    lead, coins = _margin_gap(result, champion, common)
    if -lead > coins:
        return False, (
            f"{mine:.3f} against the field, level with {name}, but "
            f"{lead:+,.0f} coins a game over {len(common)} shared opponents is "
            f"behind by more than twice its error of {coins:,.0f}"
        )

    slipped = _slipped(result, champion, common)
    if slipped:
        one, mine_one, theirs_one = slipped[0]
        return False, (
            f"{mine:.3f} against the field, level with {name} on the mean, but "
            f"{one} took it from {theirs_one:.3f} to {mine_one:.3f}"
            + (
                f" and {len(slipped) - 1} more opponent(s) with it"
                if len(slipped) > 1
                else ""
            )
            + ": a mean over the field cannot see one matchup thrown away"
        )

    gained = _gained(result, champion, common)
    if gained:
        one, mine_one, theirs_one = gained[0]
        return True, (
            f"{mine:.3f} against the field over {name}'s {theirs:.3f} on "
            f"{len(keen)} contested opponents, nothing given back, and {one} "
            f"went from {theirs_one:.3f} to {mine_one:.3f}"
            + (f" with {len(gained) - 1} more opponent(s)" if len(gained) > 1 else "")
        )

    decided = result.decisive.get(name, 0)
    rate = result.rates[name]
    if decided < decisive_bar:
        return False, (
            f"{mine:.3f} against the field over {name}'s {theirs:.3f}, but only "
            f"{decided} of its games against {name} were decided and the bar is "
            f"{decisive_bar}: the two play the same game"
        )
    low, _ = wilson_interval(rate * decided, decided)
    if low > 0.5:
        return True, (
            f"{mine:.3f} against the field over {name}'s {theirs:.3f}, and beat "
            f"{name} at {rate:.3f} over {decided} decided, lower bound {low:.3f}"
        )
    return False, (
        f"{mine:.3f} against the field over {name}'s {theirs:.3f}, but "
        f"{rate:.3f} against {name} over {decided} decided has lower bound "
        f"{low:.3f}: not shown to beat it"
    )


def _gained(
    result: Result, champion: "Champion", common: list[str]
) -> list[tuple[str, float, float]]:
    """Opponents the candidate pulled ahead on, beyond the joint noise.

    The upward half of `_slipped`, and the reason the gate has an objective
    rather than only a ratchet. Every other condition refuses a candidate for
    getting worse; this one promotes it for getting better at a specific
    opponent, which is where the rating that is left to win actually lives.

    Measured 2026-09-21 on champion_30: of 176 opponents it sweeps 145 at
    1.000, and of the twelve contested outsiders one --
    `haideptry_the_2950_peak_farm` -- takes all 32 games. Fixing that is the
    single most valuable change available and no condition here could see it.
    The field mean could not: moving it to an even 0.500 shifts the mean by
    less than half the bar, because a mean spreads one opponent's evidence
    across every denominator. `_slipped` could not either, by construction --
    its family needs the champion to have something to lose, and against this
    opponent the champion has nothing.

    Per opponent the same change is decisive. At 32 games and a family of 31
    the bar is 2.95 standard errors, which a candidate clears by taking seven
    games off an opponent the champion takes none from -- about a fifth of an
    opponent-unit, where the mean needs nine tenths of one.

    The family is the opponents with room above them, the champion's rate
    below 1.000, which is a count the candidate does not influence. The
    correction matters as much here as in `_slipped` and in the same
    direction: at 1.96 apiece over thirty-one opponents a candidate clears by
    chance better than one time in four, which is a gate that opens on noise.

    Args:
        result: The candidate's evaluation.
        champion: The champion it is being measured against.
        common: Opponents both were measured against, the champion excluded.

    Returns:
        ``(name, candidate rate, champion rate)`` per gain, largest first.
    """
    family = [one for one in common if champion.result.rates[one] < 1.0]
    if not family:
        return []
    # One-sided, like the guard: a candidate falling behind on some opponent
    # is `_slipped`'s business, and it has already refused before this runs.
    bar = float(scipy.stats.norm.ppf(1 - 0.05 / len(family)))
    mine_games = max(1, result.games)
    theirs_games = max(1, champion.result.games)
    gained = []
    for one in family:
        mine = result.rates[one]
        theirs = champion.result.rates[one]
        if mine <= theirs:
            continue
        error = math.sqrt(
            mine * (1 - mine) / mine_games + theirs * (1 - theirs) / theirs_games
        )
        # Zero error with a gain is a pairing that went 0.000 to 1.000 on
        # every game of both measurements, which is evidence rather than the
        # absence of it -- the same reading `_slipped` gives a certain loss.
        if error == 0.0 or mine - theirs > bar * error:
            gained.append((one, mine, theirs))
    return sorted(gained, key=lambda row: row[1] - row[2], reverse=True)


def contested(champion: "Champion", common: Sequence[str]) -> list[str]:
    """The opponents the champion does not already sweep.

    An opponent at 1.000 has no room above it, so it can neither show a
    candidate improving nor decide the mean -- it contributes `p(1-p) = 0`
    variance along with no room. Measured 2026-09-22 there were 126 of those
    against 58 that could move.

    Chosen by the champion's rates and never the candidate's, so a candidate
    cannot pick its own denominator. Empty means the champion sweeps
    everything -- the champion_19 state -- and the caller falls back to every
    shared opponent, because there is nothing this can say there.

    Args:
        champion: The champion the candidate is measured against.
        common: Opponents both were measured against.

    Returns:
        Those with room above them, or every one of ``common`` when none has.
    """
    return [one for one in common if champion.result.rates[one] < 1.0] or list(common)


def _field(
    result: Result, champion: "Champion", common: Sequence[str]
) -> tuple[bool, str]:
    """Whether the candidate holds the field mean over the contested opponents.

    Condition 1, on its own, because `screened` asks exactly this and nothing
    else. One copy rather than two: a screen that drifted from the condition it
    stands in for would start refusing candidates the gate would have promoted,
    which is the one thing a screen must never do.
    """
    keen = contested(champion, common)
    mine = sum(result.rates[one] for one in keen) / len(keen)
    theirs = sum(champion.result.rates[one] for one in keen) / len(keen)
    played = max(1, len(keen) * max(1, result.games))
    error = math.sqrt(mine * (1 - mine) / played + theirs * (1 - theirs) / played)
    bar = 2 * error
    if theirs - mine > bar:
        return False, (
            f"{mine:.3f} against the field where {champion.name} has "
            f"{theirs:.3f} over {len(keen)} contested opponents: "
            f"{mine - theirs:+.3f} is behind by more than twice its error "
            f"of {bar:.3f}"
        )
    return True, f"{mine:.3f} against {champion.name}'s {theirs:.3f}"


def screened(result: Result, champion: "Champion | None") -> tuple[bool, str]:
    """Whether a candidate is worth the whole sweep.

    The same test as the gate's first condition on the same seasons, played
    against the contested opponents alone. That is deliberate and is the whole
    safety argument: identical criterion on identical evidence, so a candidate
    this refuses is one `promotion` would refuse for the same reason and in the
    same words. Nothing promotable is thrown away.

    What it saves is the rest. A sweep is 184 opponents where 126 sit at 1.000
    and cannot move a verdict; both gates run on 2026-09-22 spent an hour each
    to refuse on a number the first third of the games had already settled.

    A candidate it passes plays the full pool, so every promotion still carries
    the per-opponent regression check over every opponent, swept ones included
    -- which is where a collapse has the most room to hide.

    Args:
        result: The candidate measured against the contested opponents.
        champion: The champion it must not be behind, or None at champion zero.

    Returns:
        Whether to spend the full sweep, and why not when not.
    """
    if champion is None:
        return True, "no champion to screen against"
    common = sorted(set(result.rates) & set(champion.result.rates) - {champion.name})
    if not common:
        return True, "no opponent in common to screen on"
    return _field(result, champion, common)


def _slipped(
    result: Result, champion: "Champion", common: list[str]
) -> list[tuple[str, float, float]]:
    """Opponents the candidate lost ground against, beyond the joint noise.

    The field condition is a mean over every shared opponent, and a mean is
    the wrong instrument for the question it is asked. Measured 2026-09-19 on
    champion_29: 89 of its 108 opponents sit at 1.000 and only five take more
    than a game off it. Throwing the worst of those away outright -- 0.156 to
    0.000 -- moves the mean by 0.0014 against a bar of 0.0096, so the gate
    cannot see a candidate discarding one of the few matchups it still has to
    lose. Repairing that same opponent is 0.33 of the bar, so it cannot see
    the repair either. Per opponent, that change is 3.1 standard errors: the
    evidence is there and averaging over a field of ceilings is what destroys
    it.

    So the regression guard is asked per opponent, where the evidence is, and
    the mean keeps only the job it can still do.

    The bar carries a Bonferroni correction. At 1.96 apiece over a hundred and
    seventy opponents a candidate would be refused four times over by chance,
    so a per-opponent test at the mean's confidence is a gate that never
    opens. The family is the opponents where a slip is possible at all -- the
    champion has to have something to lose -- which is a count the candidate
    does not influence, so the correction is not chosen after seeing it. It
    cannot be narrowed to the contested few: an opponent sitting at 1.000 is
    exactly where a collapse has the most room to happen, so excluding it
    would blind the guard to the case it exists for.

    What that buys, at a family of about 170 and 32 games an opponent: one
    opponent falling from 1.000 by 0.30 or more is refused, and a fall of
    0.25 is not. Smaller slips are left to the mean, which sees them once
    there are enough of them to matter -- five opponents each losing 0.25
    moves it 0.0074 against a bar of 0.0043. A single matchup quietly
    discarded is the hole this closes; a broad sag was never the hole.

    Args:
        result: The candidate's evaluation.
        champion: The champion it must not fall behind.
        common: Opponents both were measured against, the champion excluded.

    Returns:
        ``(name, candidate rate, champion rate)`` per slip, worst first.
    """
    family = [one for one in common if champion.result.rates[one] > 0]
    if not family:
        return []
    # One-sided: a candidate pulling ahead of the champion on some opponent is
    # not a regression, and spending half the alpha on that tail would only
    # make the guard harder to trip.
    bar = float(scipy.stats.norm.ppf(1 - 0.05 / len(family)))
    mine_games = max(1, result.games)
    theirs_games = max(1, champion.result.games)
    slipped = []
    for one in family:
        mine = result.rates[one]
        theirs = champion.result.rates[one]
        if mine >= theirs:
            continue
        error = math.sqrt(
            mine * (1 - mine) / mine_games + theirs * (1 - theirs) / theirs_games
        )
        # Both sides deterministic and different is a certain slip, not a
        # sampling question: 1.000 against 0.000 has no error to clear.
        if error == 0.0 or theirs - mine > bar * error:
            slipped.append((one, mine, theirs))
    return sorted(slipped, key=lambda row: row[1] - row[2])


def _margin_gap(
    result: Result, champion: "Champion", common: list[str]
) -> tuple[float, float]:
    """The candidate's lead in coins a game over the champion, and its noise.

    Mean final bank margin over the opponents both were measured against, one
    minus the other, against twice the standard error of the difference. Each
    opponent's margin carries its own error from the games behind it, so the
    error of the mean is the root of the summed squares over the count -- the
    same test the rate gets, on the measure that still has room to move.

    Args:
        result: The candidate's evaluation.
        champion: The champion it must not be worse than.
        common: The opponents both played, already cut by the caller.

    Returns:
        The lead in coins a game, and twice its error.
    """
    mine = [result.margins[one] for one in common if one in result.margins]
    theirs = [
        champion.result.margins[one] for one in common if one in champion.result.margins
    ]
    if not mine or not theirs:
        return 0.0, 0.0
    lead = sum(m.mean for m in mine) / len(mine) - sum(m.mean for m in theirs) / len(
        theirs
    )
    noise = math.sqrt(
        sum(m.error**2 for m in mine) + sum(m.error**2 for m in theirs)
    ) / max(1, len(common))
    return lead, 2 * noise


def promote(
    program: Program,
    result: Result,
    paths: config.Run,
    *,
    package: bool = True,
) -> Champion:
    """The file half of a promotion: the tarball, the champion's copy, the floor.

    Every file this writes is one nothing else owns, so it is safe to call
    in a thread; the pool and ``champion.json`` are ``enroll`` and
    ``record``, on the loop thread.

    Each champion is written once to ``CHAMPIONS/<name>.py`` and it is that
    path the pool registers, so a pool holding N champions holds N different
    programs. ``FLOOR/main.py`` is the current floor and is overwritten every
    promotion; registering it instead would make every pool entry an alias
    for the newest agent and silently erase the history the pool exists to
    keep.

    Args:
        program: The archive entry being promoted.
        result: Its measurement.
        paths: The run the champion is written into.
        package: Whether to build the submission tarball. Champion zero passes
            False: it is the seed being enthroned at startup rather than
            something a round won, so there is nothing to submit yet, and
            packaging it fails on the skeleton's `LICENSE`.

    Returns:
        The champion record, for ``enroll`` and ``record`` to act on. Its
        ``tarball`` is empty when ``package`` is False.

    Raises:
        FileExistsError: The champions directory already holds this name.
    """
    # Numbered off the champions directory, which only ever grows. Counting
    # the pool's `champion_` members instead would renumber after a
    # retirement and hand the sixth promotion a name the fifth already has.
    # The file is numbered for history; the pool key is not. They were the same
    # string until 2026-09-12, when numbering from this run's empty champions
    # directory produced `champion_1` and the pool already had one of those --
    # the tape lineage's, which `enroll` then overwrote.
    number = 1 + sum(1 for _ in paths.champions.glob("champion_*.py"))
    name = config.POOL_CHAMPION
    kept = paths.champions / f"champion_{number}.py"
    if kept.exists():
        raise FileExistsError(
            f"{kept} already exists: something other than a promotion has "
            "written to the champions directory"
        )
    source = Path(program.source_path)

    with tempfile.TemporaryDirectory() as scratch:
        # The tarball is built in here and moved into place whole. Packaging
        # writes a file at a time and can fail part way through; a half-built
        # archive under `champions/` would be a cut waiting to upload it.
        paths.champions.mkdir(parents=True, exist_ok=True)
        tarball = paths.champions / f"champion_{number}.tar.gz"
        if package:
            built = harness.package(source, Path(scratch) / f"champion_{number}.tar.gz")
            shutil.move(str(built), str(tarball))

    code = source.read_text(encoding="utf-8")
    kept.write_text(code, encoding="utf-8")
    kept.chmod(0o444)

    paths.floor.mkdir(parents=True, exist_ok=True)
    floor = paths.floor / "main.py"
    if floor.exists():
        floor.chmod(0o644)
    floor.write_text(code, encoding="utf-8")
    floor.chmod(0o444)

    LOGGER.info("promoted %s to %s", program.id, name)
    return Champion(
        name=name,
        path=str(kept),
        tarball=str(tarball) if package else "",
        result=result,
    )


def enroll(champion: Champion, pool: Pool, paths: config.Run) -> None:
    """Put ``champion`` in the pool, as a gatekeeper every later candidate faces.

    Mutates ``pool`` in place, so it belongs on whichever thread owns it --
    for the campaign, the event loop. The save follows immediately, so
    nothing between here and it can resolve the champion's name through a
    pool file that does not have it yet.

    Args:
        champion: The record ``promote`` returned.
        pool: The opponent pool, updated and saved in place.
        paths: The run whose pool file it is saved to.
    """
    # No rating is fitted here any more. It decided which of our champions to
    # retire, and there is one champion now: the key it occupies is overwritten
    # and nothing else in the pool is touched.
    pool.add_champion(champion.path)
    pool.save(paths.pool)
    LOGGER.info(
        "%s holds the %s slot; pool is %d",
        Path(champion.path).stem,
        champion.name,
        len(pool.opponents),
    )


def record(champion: Champion, paths: config.Run) -> Champion:
    """Write the run's ``champion.json`` atomically: a temp file, then a rename.

    A reader never sees a half-written record, and the file exists in full or
    not at all -- which is what lets ``loop.run`` trust it over ``state.json``.

    Args:
        champion: The record to write.
        paths: The run it is written into.

    Returns:
        ``champion``, so a caller can write and keep it in one expression.
    """
    paths.champion.parent.mkdir(parents=True, exist_ok=True)
    scratch = paths.champion.with_name(f"{paths.champion.name}.{os.getpid()}.tmp")
    scratch.write_text(champion.model_dump_json(indent=2), encoding="utf-8")
    scratch.replace(paths.champion)
    return champion


def load_champion(paths: config.Run) -> Champion | None:
    """The promoted champion on disk, or None if nothing has been promoted."""
    if not paths.champion.exists():
        return None
    return Champion.model_validate_json(paths.champion.read_text(encoding="utf-8"))
