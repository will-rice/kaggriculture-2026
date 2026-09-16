"""Hill-climb a season plan against opponents that can fight back.

The plan is a step-to-action table, seeded from a recorded season of a strong
team and perturbed. The seed matters: the space of 719-step plans is
overwhelmingly plans that do nothing, and a hand-written schedule of market
orders banks the 3,000 it started with. A recorded season is the one thing the
teams above us have handed over -- 1.2M of SpaTaro's decisions among them.

The opponents are real programs, and that is not an efficiency compromise but
the whole correctness of the thing. Scored against blind tapes this search found
a plan worth +22,955 coins a season, which then lost 8 of 8 to champion_24 at
-77,355 and 8 of 8 to a public kernel at -36,097: every coin of the gain was a
market manipulation that an opponent reading the board would have punished and a
tape could not. A fast evaluator and a strawman are the same object here.

It costs what it costs. A season against a real agent is about 1.1s -- 98% of it
handing observations across to Python, 39us each and 1,438 a season -- so a
candidate scored on 24 seasons is 27 seconds of work, which is 0.7s spread over
forty cores. Two thousand rounds is half an hour.

What it cannot do: a blind plan does not adapt, and the top of this ladder does.
Freezing champion_19 into its own recorded actions lost 6 of 6 against itself
while keeping 90% of the bank. So this is not a route to a champion on its own;
it asks whether a better *economy* than any recorded one exists, on the measure
another team's round-robin found tracks the ordering -- average final money.

    uv run plan-search --seasons 6 --rounds 2000
"""

import argparse
import copy
import json
import logging
import random
import statistics
import tempfile
from collections.abc import Sequence
from pathlib import Path

from kaggriculture.campaign import config, games, harness
from kaggriculture.campaign.harness import OpponentCrash
from kaggriculture.scripts.tape_opponents import tape, write_agent

LOGGER = logging.getLogger(__name__)

# What an agent returns when the plan has nothing for this step.
IDLE: dict = {"farmer": ["PASS"], "hands": [], "market": []}

Plan = dict[int, dict]

# What a candidate is scored against, in order. The standing champion because a
# plan has to beat what we already have, then the public kernel that took nine
# games in ten off champion_16, because the pool's worst matchup is where a new
# economy would show first.
OPPONENTS = ["ours", "ahmedberatozer_kaggriculture_v40_plans_t"]


def main() -> None:
    """Search for a plan that banks more than the recording it started from."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--team",
        default="SpaTaro",
        help="whose recorded season to start from (default: %(default)s)",
    )
    parser.add_argument(
        "--seasons",
        type=int,
        default=12,
        help="seeds each candidate is scored on (default: %(default)s)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=2000,
        help="perturbations to try (default: %(default)s)",
    )
    parser.add_argument(
        "--opponents",
        type=int,
        default=3,
        help="taped opponents to score against (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=config.RUN / "plan-search.json",
        help="where the best plan is written (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=0, help="the search's own rng")
    arguments = parser.parse_args()

    seed_plan, _ = corpus(arguments.team, 1)
    against = OPPONENTS[: arguments.opponents]
    home = Path(tempfile.mkdtemp(prefix="plan-search-"))
    LOGGER.info(
        "seeded from %s (%d steps), scored against %s",
        arguments.team,
        len(seed_plan),
        ", ".join(against),
    )
    seasons = random.Random(arguments.seed).sample(
        range(1, 1_000_000), arguments.seasons
    )

    best = seed_plan
    score = evaluate(best, against, seasons, home, config.CORE_BUDGET)
    LOGGER.info(
        "the recording banks %s coins a season against the tapes",
        f"{score:+,.0f}",
    )

    rng = random.Random(arguments.seed)
    kept = 0
    for round_number in range(1, arguments.rounds + 1):
        candidate = perturb(best, rng)
        got = evaluate(candidate, against, seasons, home, config.CORE_BUDGET)
        if got > score:
            best, score, kept = candidate, got, kept + 1
            LOGGER.info(
                "round %d: %s coins a season (kept %d)",
                round_number,
                f"{got:+,.0f}",
                kept,
            )
        if round_number % 250 == 0:
            LOGGER.info(
                "round %d: best %s, %d kept", round_number, f"{score:+,.0f}", kept
            )

    arguments.out.write_text(
        json.dumps(
            {
                "score": score,
                "seasons": seasons,
                "plan": {str(k): v for k, v in best.items()},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    LOGGER.info("wrote %s at %s coins a season", arguments.out, f"{score:+,.0f}")


def corpus(team: str, count: int) -> tuple[Plan, list[Plan]]:
    """One strong recorded season to start from, and some others to play.

    The opponents are recordings too, because a blind plan is what makes this
    fast: an agent that reads the board costs 39us an observation and 1,438 of
    them a season, and two tables cost neither.

    Args:
        team: Whose seasons to draw on.
        count: How many opponents to take.

    Returns:
        The seed plan, and the opponents.
    """
    quoted = team.replace("'", "''")
    rows = games.query(
        "select played, episode, if(team_0 = '" + quoted + "', 0, 1) as seat, "
        "greatest(bank_0, bank_1) as best from games.episodes "
        "where source = 'ladder' and played != '' "
        f"and (team_0 = '{quoted}' or team_1 = '{quoted}') "
        f"order by best desc limit {count + 1} format TabSeparated"
    ).splitlines()
    plans = []
    for line in rows:
        played, episode, seat, _best = line.split("\t")
        try:
            recorded = tape(played, episode, int(seat))
        except (FileNotFoundError, KeyError) as error:
            LOGGER.info("%s: %s", episode, error)
            continue
        plans.append({int(step): action for step, action in recorded.items()})
    if not plans:
        raise SystemExit(f"no archived seasons for {team!r}")
    return plans[0], plans[1:] or [plans[0]]


def evaluate(
    plan: Plan,
    against: Sequence[str],
    seasons: Sequence[int],
    home: Path,
    workers: int,
) -> float:
    """Mean coins a season ahead of real opponents, every seed, both seats.

    Through `harness.play`, which is the tested fan-out: it resolves pool names,
    spreads the games over workers, and refuses to score a game an agent crashed
    in rather than counting the untouched opening money as a loss.

    Both seats because the map is not symmetric between them and a plan that
    only works from one side is measuring the seat. The mean of the margin
    rather than a win rate: the rate saturated on this pool the same week, and
    another team's round robin of fourteen public implementations found the
    ordering tracks average final money closely.

    Args:
        plan: The candidate.
        against: Pool names to play.
        seasons: The seeds, fixed across the search so rounds compare.
        home: Where the candidate is written for the harness to load.
        workers: Worker processes.

    Returns:
        Mean coins a season ahead, or minus infinity if nothing was scored.
    """
    # `write_agent` keys by step as a string, the way the archives do; the
    # search keys by int so a perturbation can pick a neighbour arithmetically.
    agent = write_agent({str(k): v for k, v in plan.items()}, home / "main.py")
    try:
        played = harness.play(
            agent, list(against), list(seasons), workers, pool=config.POOL
        )
    except (RuntimeError, OpponentCrash):
        return float("-inf")
    margins = [game.ours - game.theirs for game in played if game.error is None]
    return statistics.fmean(margins) if margins else float("-inf")


def perturb(plan: Plan, rng: random.Random) -> Plan:
    """One small change to a plan: a step's action moved, dropped or repeated.

    Deliberately crude. The engine no-ops an action a state cannot support --
    a `PLANT` onto an occupied tile does nothing and says nothing -- so an
    illegal perturbation costs a season's worth of nothing rather than an
    error, and the search does not need to know the rules to respect them.
    """
    changed = copy.deepcopy(plan)
    steps = sorted(changed)
    if not steps:
        return changed
    step = rng.choice(steps)
    move = rng.random()
    if move < 0.4:
        # Repeat a neighbour's action here.
        changed[step] = copy.deepcopy(changed[rng.choice(steps)])
    elif move < 0.7:
        # Do nothing at this step.
        changed[step] = copy.deepcopy(IDLE)
    else:
        # Keep the step's own actions and resize what it trades.
        action = changed[step]
        market = action.get("market") or []
        if market:
            order = rng.choice(market)
            if len(order) >= 3 and isinstance(order[2], int):
                order[2] = max(1, order[2] + rng.choice((-2, -1, 1, 2, 5)))
    return changed
