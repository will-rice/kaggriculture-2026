"""One evaluation, which is also the gate.

``score`` plays fresh seeds through the harness, both seats, against every
pool opponent, and reports the per-opponent rates with Wilson intervals.
Every opponent counts the same -- the gate asks whether a candidate beats
each of them, not what it averages -- so the fitness is the plain mean of
the per-opponent rates, and the promotion turns on the tournament fitted
over them rather than on that mean.

There were two evaluations here: a cheap ranking on eight seeds and a sealed
sixty-four-seed block that decided promotions. The cheap one did not rank.
Over 471 programs it named 78 of them the best in the tournament and the
block promoted none. The seeds were redrawn every call, so nothing was being
fitted; the ranking was simply the maximum of an estimator with a standard
error of 0.125, and the maximum of a noisy estimator is the luckiest program
rather than the best one. A second measurement cannot undo that -- only games
can -- so there is one measurement, deep enough to select on.

It never plays a program against itself. A champion is a member of the pool
it is re-scored on, and its own bytes in the other seat are a structural 0.5
that no metric should carry: ``opponents`` drops that entry before any game
is played, so the fitness, the rates, ``field`` and the intervals are all
over real opponents and nothing downstream has to know the mirror existed.
"""

import random
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from kaggriculture.campaign import harness, roster
from kaggriculture.campaign.pool import Pool
from kaggriculture.report import wilson_interval

VENDORED = list(roster.TRAINING)


class Result(BaseModel):
    """The measurement, and the only one there is.

    There were two: a cheap ranking on eight random seeds and a sealed block
    of sixty-four that decided promotions. The cheap one selected the luckiest
    program rather than the best -- 78 of 471 topped it and none survived the
    block -- and the block could not fix that, because selecting the maximum
    of a noisy estimator is biased however you re-measure it afterwards. So
    there is one measurement now, deep enough to select on, and the promotion
    is decided on it.

    It is also the whole of what a model is told about the program it is asked
    to improve: the rates and margins are the verdict, and ``states`` is the
    game behind it.

    Attributes:
        program_id: The program this measures.
        fitness: Mean win rate over the pool, which grows as champions join
            it, so it ranks the database but does not compare across time.
        field: Mean win rate over the vendored incumbents alone. They never
            change, so this is the one number that means the same thing on
            the first session and the thousandth -- and None once the pool
            has trimmed the last of them away.
        rates: Win rate per pool opponent, ties as half.
        margins: Bank margin per pool opponent.
        intervals: Wilson interval per pool opponent, over the decisive games
            alone, so a rate is read with the width of the evidence actually
            behind it rather than of the games played.
        decisive: Games against each opponent that were not draws. The
            denominator any claim about a difference has to use, and the one
            thing that separates a candidate from a copy of its parent.
        games: Games played against each opponent: ``2 * GATE_SEEDS``.
        seeds: The seeds drawn for this call. Fresh every time, which is what
            makes every measurement a held-out one.
        hardest: The opponent with the lowest win rate. The log line names it;
            what a round is *shown* is chosen from the standings instead,
            because the gate is a tournament and the agent to study is the one
            directly above, not the one furthest away.
        states: Every game played, day by day, grouped by the opponent it was
            played against and narrowest first within each group.

            All of them, because a round can only improve a game it is shown
            and the campaign scores every one of them. It used to keep the
            single narrowest game against each opponent and drop the other
            thirty-one, which quietly decided on a round's behalf that the
            other thirty-one held nothing -- and the games are played either
            way, with their days already rendered, so keeping them costs
            nothing but the write.

            Grouped, because the grouping is what makes two seasons
            comparable. Within one group the opponent is fixed and what varies
            between seasons is the world: the map, the prices, the seat. That
            is the axis a general program has to hold up across. Across groups
            the adversary varies too, so a difference between two of them says
            nothing about either -- and the only reading that survives is
            "that opponent does this", which is the fitting the message exists
            not to encourage.
    """

    program_id: str
    fitness: float
    field: float | None
    rates: dict[str, float]
    margins: dict[str, harness.Margin]
    intervals: dict[str, tuple[float, float]] = {}
    decisive: dict[str, int] = {}
    games: int = 0
    seeds: list[int]
    hardest: str
    states: dict[str, list[list[harness.Day]]]


def _mean(rates: dict[str, float]) -> float:
    """The mean of ``rates``; every opponent counts the same."""
    return sum(rates.values()) / len(rates)


def _rates(games: list[harness.Game], names: list[str]) -> dict[str, float]:
    """Win rate per opponent over the games played against it, ties as half."""
    out = {}
    for name in names:
        mine = [g for g in games if g.opponent == name]
        out[name] = sum(
            1.0 if g.ours > g.theirs else 0.5 if g.ours == g.theirs else 0.0
            for g in mine
        ) / len(mine)
    return out


def _contested(
    games: list[harness.Game], names: list[str]
) -> dict[str, tuple[int, int]]:
    """Wins and decisive games against each opponent; draws count in neither.

    A draw says the two sides played the same game, not that they were evenly
    matched, and counting it as half a win makes thirty draws and two wins
    read exactly like seventeen wins and fifteen losses -- 0.53125 either way.
    The first is one agent measured against a copy of itself; the second is
    two agents that genuinely trade games. Everything downstream that asks
    "is this different from what we have" needs to tell them apart, and this
    is the only number that does.
    """
    out = {}
    for name in names:
        decided = [
            game for game in games if game.opponent == name and game.ours != game.theirs
        ]
        won = sum(1 for game in decided if game.ours > game.theirs)
        out[name] = (won, len(decided))
    return out


def opponents(pool: Pool, program_id: str, agent: Path) -> Pool:
    """``pool`` without ``program_id``; ``pool`` itself if it is not in it.

    A champion is a member of the pool it is scored on, and its own bytes in
    the other seat are a structural 0.5 that says nothing about the program.
    The match is by name, which is enough because the only caller that scores
    a pool member passes the pool's own key as the ``program_id``, and the
    pool registers that key against the very file being scored.

    Args:
        pool: The opponents as they stand.
        program_id: The program being scored.
        agent: The file being scored.

    Returns:
        The opponents to play.

    Raises:
        ValueError: The pool holds that name against a different file, which
            is a disagreement this cannot settle by looking.
    """
    if program_id not in pool.opponents:
        return pool
    if Path(pool.opponents[program_id]) != agent:
        raise ValueError(
            f"the pool's {program_id} is {pool.opponents[program_id]}, not the "
            f"{agent} being scored"
        )
    return Pool(opponents={n: p for n, p in pool.opponents.items() if n != program_id})


def score(
    agent: Path,
    program_id: str,
    pool: Pool,
    rng: random.Random,
    seeds: Sequence[int],
    workers: int,
    pool_file: Path | None = None,
    standings: dict[str, float] | None = None,
    always: Sequence[str] = (),
) -> Result:
    """Mean win rate over ``GATE_SEEDS`` seasons, both seats.

    The seeds come from the caller so that every candidate in one round plays
    the same seasons, and are redrawn between rounds. Both halves matter and
    they pull in opposite directions.

    Redrawing is what a held-out set was for: no program is ever measured on
    maps it or its ancestors were selected on, and getting that continuously
    is why there is no reserved block any more.

    Sharing is what makes two candidates comparable. Drawn per call, they were
    never ranked on the same seasons, and a season is most of what a rating
    measures: one unchanged agent put through this five times, opponents held
    fixed and only the seeds moving, produced fitted ratings from -3.466 to
    -2.226 -- a standard deviation of 0.491 against a promotion bar that used
    to be 0.15. Holding the seeds and varying the *opponents* instead moved it
    0.070. The maps are seven times the draw, which was the opposite of what
    this looked like before it was measured.

    Every game is played with its day table recorded, because one of them is
    what the loop shows a model of how its program played. Which one is not
    decided here: the gate is a tournament, so the game worth studying is the
    one against the agent directly above in the standings, and those are not
    fitted until these rates exist. So the narrowest game against every
    opponent is kept and the caller picks.

    Args:
        agent: The candidate's ``main.py``.
        program_id: The program being scored, so it is not among them.
        pool: The opponents to measure against.
        rng: The generator the opponent draw comes from.
        seeds: The seasons to play, shared by every candidate in this round.
        workers: Processes to fan the games over.
        pool_file: Where a champion's name resolves from, since the
            roster only knows the vendored opponents.
        standings: Ratings the contenders are chosen by. Without them the
            draw is anchors plus a random remainder, which is what a cold
            start has and is enough to fit the first ratings from.
        always: Opponents to draw whatever the dice say -- the top-ranked
            agent, which topping the field means beating, and the floor,
            which a promotion is measured as a gap over.

    Returns:
        The mean fitness, the per-opponent rates and margins, the seeds
        drawn, and one game against the hardest opponent day by day.

    Raises:
        RuntimeError: A side raised during a game. A crashed candidate is a
            failed evaluation, never a zero score, so this propagates.
    """
    measured = opponents(pool, program_id, agent)
    # A sample, not the pool. Nothing leaves the pool any more, so it is
    # everything the campaign has produced or harvested and a candidate cannot
    # play all of it -- sixty opponents is nearly four thousand games for one
    # verdict. A rating does not need it to: the fit spans every pairing on
    # the record, so a candidate adds sixteen edges and is placed against the
    # rest through them.
    names = measured.sample(standings or {}, rng, program_id, always)
    games = harness.play(agent, names, list(seeds), workers, days=True, pool=pool_file)
    rates = _rates(games, names)
    contested = _contested(games, names)
    # Ties on the rate are broken by the margin, because before the first win
    # every rate is 0.0 and `min` would otherwise always name the first pool
    # key -- so every session would be shown the same opponent rather than the
    # one it came closest to beating.
    margins = harness.margins(games, names)
    hardest = min(rates, key=lambda name: (rates[name], margins[name].mean))
    # Every game, grouped by opponent, narrowest first within each group. The
    # narrowest is the one a small change would have flipped, so it leads; the
    # widest shows the failure at its starkest and the least reachable, so it
    # does not. This used to keep the narrowest alone and drop the other
    # thirty-one against each opponent, which is a round being told about one
    # thirty-second of what it is scored on.
    states = {
        name: [
            game.days
            for game in sorted(
                (game for game in games if game.opponent == name),
                key=lambda game: abs(game.ours - game.theirs),
            )
        ]
        for name in names
    }
    played = 2 * len(seeds)
    return Result(
        program_id=program_id,
        fitness=_mean(rates),
        field=vendored_field(rates),
        rates=rates,
        margins=margins,
        intervals={
            # Over the decisive games, not over every game played. The guard
            # this width exists to provide was computed on `played`, which
            # claimed thirty-two games of confidence for a pairing that
            # decided two. `wilson_interval` returns the whole unit interval
            # at zero games, which is the right answer for a pairing that
            # drew every time: no evidence either way.
            name: wilson_interval(won, decided)
            for name, (won, decided) in contested.items()
        },
        decisive={name: decided for name, (_, decided) in contested.items()},
        games=played,
        seeds=seeds,
        hardest=hardest,
        states=states,
    )


def vendored_field(rates: dict[str, float]) -> float | None:
    """Mean win rate over the vendored incumbents in ``rates``.

    The pool grows as champions join it, so a mean over the pool moves for
    reasons that have nothing to do with a program improving. The vendored
    kernels are fixed, so this number is comparable across the whole
    campaign.

    Args:
        rates: Win rate per opponent name.

    Returns:
        The mean over the vendored names present, or None when the pool
        holds none of them.

    None rather than an error, because a pool of nothing but champions
    is where this campaign is going: they join on every promotion and
    the weakest opponent makes way, so the published agents leave one
    at a time and the last of them leaves for good. Raising there would
    kill the run at its most successful moment -- and did nearly:
    `ValueError` is not the `RuntimeError` a round catches, so it would
    have gone up through the task group and stopped the campaign.

    What is lost is a number, not the gate. `field` is the one metric
    comparable across the whole campaign because the vendored agents
    never change; without them there is nothing fixed to compare to,
    and the promotion chain is what says the search is moving.
    """
    vendored = {name: rate for name, rate in rates.items() if name in VENDORED}
    return _mean(vendored) if vendored else None
