"""Play a program by opponent name, check it as Kaggle loads it, package it.

Games run on the engine port; a 2% sample is replayed on the reference engine
and any bank disagreement fails the call, so drift between the two engines is
a harness error rather than a quietly wrong fitness.
"""

import argparse
import hashlib
import logging
import random
import shutil
import statistics
import sys
import tarfile
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

from kaggle_environments import make
from kaggle_environments.agent import get_last_callable
from kaggle_environments.core import Environment
from kaggle_environments.utils import Struct, structify
from pydantic import BaseModel

from kaggriculture.campaign import arena, config, dataset, pools, roster
from kaggriculture.campaign.engine.wrapper import Engine, render_private
from kaggriculture.constants import ENVIRONMENT, EPISODE_STEPS

LOGGER = logging.getLogger(__name__)
REFERENCE_SAMPLE = 0.02
# Kaggle's own per-call limit. Not ours, and not a budget we chose: a program
# that exceeds it does not run on the ladder at all.
ACT_TIMEOUT = 1.0
LATENCY_BUDGET = ACT_TIMEOUT / 2
# What `campaign play` allows a person at a terminal, enforced on the CLI
# only: the library function is what the evaluator calls, and it plays the
# whole gate. Nothing in the campaign calls the command -- the loop
# plays every game anything is scored on -- so these bound a hand-run check
# beside a live campaign, and nothing else.
SANDBOX_GAME_CAP = 16
SANDBOX_WORKER_CAP = 4
# The framework, not the interpreter, writes these onto every seat's
# observation at call time. The port exports neither, so `_one` injects both.
OVERAGE_SECONDS = 60
# One game for a worker process: the candidate, the opponent's roster name and
# file, the seed, the candidate's seat, and whether to record a day table.
Work = tuple[str, str, str, int, int, bool]
# The last hour label of a day; the step taken on it runs the day-end refresh.
LAST_HOUR = 23


class OpponentCrash(RuntimeError):  # noqa: N818 - a crash, not our error
    """An opponent, not the candidate, raised during a game.

    Attributing this to the candidate would write a failure into its lineage's
    feedback for something it did not do, and the next prompt would chase it.
    A broken opponent is a broken pool, so the loop lets this one out (spec
    section 8): it halts loudly rather than scoring around the hole.
    """


class Day(BaseModel):
    """One day of a game as it closed, from the candidate's point of view.

    The row for day ``d`` is that day at hour 23: what both programs were
    looking at when they made their last decision in it, and for the final
    day the terminal state the match is decided on. Hour 23 rather than the
    next day's hour zero because the day-end refresh in between clears the
    hands: a table read after it says every day was worked alone. A season is
    thirty days, so a game is thirty rows.

    Both sides appear, the shed a player cannot see at runtime included. That
    is deliberate: this is what the author of a program is shown after the
    game, never what the program may read while playing one.

    The banks say a program fell behind; the tile counts say what it was
    doing instead. A row without them shows a shed filling and emptying with
    no sight of the acreage that filled it, which is the half of the game a
    policy actually decides. Tiles are public for both seats, so the
    opponent's are here on the same footing as ours; its seeds are not, and
    are not shown.

    Attributes:
        day: The day this row closed.
        ours_bank: The candidate's money.
        theirs_bank: The opponent's money.
        ours_plants: The candidate's growing tiles, counted by crop.
        theirs_plants: The opponent's growing tiles, counted by crop.
        ours_animals: The candidate's animals, counted by species.
        theirs_animals: The opponent's animals, counted by species.
        ours_weeds: The candidate's tiles lost to weeds.
        theirs_weeds: The opponent's tiles lost to weeds.
        ours_seeds: The candidate's unplanted seed, by crop. Private, so the
            opponent's has no counterpart here.
        ours_shed: The candidate's shed, zero counts dropped.
        theirs_shed: The opponent's shed, zero counts dropped.
        ours_hands: Hands the candidate holds, the farmer aside.
        theirs_hands: Hands the opponent holds, the farmer aside.
        ours_quadrants: Quadrants the candidate has unlocked, of four.
        theirs_quadrants: The opponent's, on the same terms.
        ours_fertilised: Growing tiles of the candidate's still under
            fertilizer on this day.
        theirs_fertilised: The opponent's, on the same terms.
        ours: Every quantity `dataset.measures` defines, for the candidate.
        theirs: The same, for the opponent.
        prices: The shared market's price per product.

    The named fields are the ones the message renders as a table, and they
    are chosen for a reader: sixty columns is not a table anyone can follow.
    `ours` and `theirs` carry all thirty quantities the corpus measures, which
    is what the claim selector reads -- so a finding can be put to a program
    whether or not there is room to print it.

    That split exists because the two were confused once. Quadrants and
    fertilizer were added as named fields and rendered in the table, which
    made them visible to a reader and left them invisible to the selector, and
    the two sharpest separations in the whole corpus went on being measured,
    stored and shown to nobody. `fertilised` on day five separates the
    stronger side from the weaker in 100% of 313 paired games and quadrants on
    day three in 99% of 208.
    """

    day: int
    ours_bank: float
    theirs_bank: float
    ours_plants: dict[str, int]
    theirs_plants: dict[str, int]
    ours_animals: dict[str, int]
    theirs_animals: dict[str, int]
    ours_weeds: int
    theirs_weeds: int
    ours_seeds: dict[str, int]
    ours_shed: dict[str, int]
    theirs_shed: dict[str, int]
    ours_hands: int
    theirs_hands: int
    # Defaulted, so a result stored before these existed still loads.
    ours_quadrants: int = 0
    theirs_quadrants: int = 0
    ours_fertilised: int = 0
    theirs_fertilised: int = 0
    ours: dict[str, float] = {}
    theirs: dict[str, float] = {}
    prices: dict[str, int]


class Margin(BaseModel):
    """How far apart the banks finished, over the games against one opponent.

    A win rate says how often; this says by how much, which is what separates
    an opponent a candidate nearly beats from one it is nowhere near.

    Attributes:
        mean: Mean of ``ours - theirs`` over those games.
        worst: The lowest such difference.
        best: The highest.
        error: The standard error on ``mean``, so it can be read as a
            measurement rather than a number. Promotion turns on this: a
            candidate replaces the champion when its margin over it is larger
            than twice this, which is a question with a statistical answer
            rather than a constant somebody chose. Defaulted, so a result
            stored before it existed still loads -- and zero there means
            "unknown", which no comparison can clear.
    """

    mean: float
    worst: float
    best: float
    error: float = 0.0


class Game(BaseModel):
    """One episode's outcome from the candidate's point of view.

    Attributes:
        opponent: The roster name played, never a path.
        seed: The episode seed.
        seat: The seat the candidate held.
        ours: The candidate's final bank.
        theirs: The opponent's final bank.
        worst_step_seconds: The slowest single call to the candidate.
        error: Which side raised and what it raised, or None if the game
            finished. Names the side by roster name and the failure by type,
            because the traceback behind it names the opponent's real file.
        culprit: "candidate" or the opponent's roster name when ``error`` is
            set, else None. Carried as its own field rather than parsed back
            out of ``error``, because who raised decides whether the loop
            records a lineage failure or halts.
        days: One row per day when the game was played with ``days``, and
            empty otherwise: recording costs a render per day, and only the
            one game the loop shows a model is ever read.
    """

    opponent: str
    seed: int
    seat: int
    ours: float
    theirs: float
    worst_step_seconds: float
    error: str | None = None
    culprit: str | None = None
    days: list[Day] = []


class CheckReport(BaseModel):
    """Whether an agent loads and runs the way the Kaggle runner would load it.

    Attributes:
        loaded: Whether ``get_last_callable`` produced a callable.
        bank: Seat zero's money when the run stopped.
        worst_step_seconds: The slowest single call to the agent.
        error: The failure, or None if the run finished.
        fingerprint: Seat zero's whole action sequence, hashed. Two agents
            that play this identically are one agent under two names, which
            the published field produces constantly -- the same work reposted,
            a notebook and its fork, a Python agent and its C++ build. Free
            here because the check already emits every action to play the
            game, and worth having because a duplicate opponent costs a gate
            real time and tells it nothing new. Empty when the run failed,
            since a crash is not an identity.
    """

    loaded: bool
    bank: float
    worst_step_seconds: float
    error: str | None
    fingerprint: str = ""


def main() -> None:
    """``campaign play|check|package|harvest|loop|dry-run``."""
    if len(sys.argv) > 1 and sys.argv[1] in ("loop", "dry-run"):
        # The loop owns its own flags, so its arguments are forwarded rather
        # than restated here. The import is local because `loop` imports this
        # module, through the evaluator and the validator.
        from kaggriculture.campaign import loop

        arguments = sys.argv[2:]
        if sys.argv[1] == "dry-run":
            arguments.append("--dry-run")
        loop.main(arguments)
        return
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "campaign loop [--sessions N] [--workers N] [--seed-agent PATH] "
            "runs the campaign; campaign dry-run takes the same flags and "
            "runs it on a fake session."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    play_parser = commands.add_parser("play")
    play_parser.add_argument("agent", type=Path)
    play_parser.add_argument("--vs", nargs="+", required=True, help="opponent names")
    play_parser.add_argument("--seeds", required=True, help="A-B inclusive")
    play_parser.add_argument("--workers", type=int, default=4)
    check_parser = commands.add_parser("check")
    check_parser.add_argument("agent", type=Path)
    package_parser = commands.add_parser("package")
    package_parser.add_argument("agent", type=Path)
    package_parser.add_argument(
        "--output", type=Path, default=Path("submission.tar.gz")
    )
    harvest_parser = commands.add_parser("harvest")
    harvest_parser.add_argument(
        "--limit", type=int, default=5, help="new kernels to take"
    )
    harvest_parser.add_argument(
        "--author", default=None, help="only this kernel author"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.command == "harvest":
        # Local because `harvest` imports this module for `check`.
        from kaggriculture.campaign import harvest as harvesting

        harvesting.harvest(args.limit, config.LIVE.pool, args.author)
        return
    if args.command == "play":
        first, last = (int(part) for part in args.seeds.split("-"))
        seeds = range(first, last + 1)
        games = len(args.vs) * len(seeds) * 2
        if games > SANDBOX_GAME_CAP:
            raise SystemExit(
                f"{games} games requested; campaign play allows at most "
                f"{SANDBOX_GAME_CAP} (opponents x seeds x 2 seats). "
                "The evaluator measures for real after you finish."
            )
        if args.workers > SANDBOX_WORKER_CAP:
            raise SystemExit(
                f"--workers {args.workers} exceeds the {SANDBOX_WORKER_CAP} "
                "a sandbox may take; the rest of the box is running the loop."
            )
        for game in play(args.agent, args.vs, seeds, args.workers):
            LOGGER.info("%s", game.model_dump_json())
    elif args.command == "check":
        LOGGER.info("%s", check(args.agent).model_dump_json())
    else:
        LOGGER.info("wrote %s", package(args.agent, args.output))


def load_agent(path: Path) -> Callable[..., Any]:
    """Load an agent file the way the Kaggle runner does: the last callable."""
    return get_last_callable(path.read_text(encoding="utf-8"), path=str(path))


def configuration() -> Struct:
    """Return the configuration the reference engine hands agents."""
    return make(
        ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS}
    ).configuration


def argument_count(agent: Callable[..., Any]) -> int:
    """How many of ``(observation, configuration)`` the Kaggle runner passes.

    ``Agent.act`` truncates that pair to the callable's ``co_argcount``, so an
    agent declaring a single parameter -- ``salemali7_2900`` is one -- must be
    called with one argument here too or it raises on its very first turn. A
    callable with no ``__code__`` gets both, exactly as the runner leaves the
    pair untruncated. Resolved once at load, never by catching the mismatch
    and retrying: these agents carry state across turns, so a retry would run
    the first turn twice.
    """
    code = getattr(agent, "__code__", None)
    return min(2, getattr(code, "co_argcount", 2))


def shared_step(environment: Environment) -> int:
    """Return the step counter every seat is handed, as the runner hands it.

    ``Environment.__get_shared_state`` copies each property the schema marks
    ``shared`` from seat zero onto every other seat at call time, and ``step``
    is the one this environment shares. Driving ``environment.state`` directly
    skips that copy, so seat one's observation has no ``step`` at all --
    ``remainingOverageTime`` is per-seat and already present on both, so it is
    left exactly as the framework maintains it. Without this an agent that
    reads ``observation["step"]`` raises as seat one under ``check`` while
    running fine under ``play`` and on Kaggle.
    """
    return int(environment.state[0].observation.step)


def margins(games: list[Game], names: list[str]) -> dict[str, Margin]:
    """Mean, worst and best bank margin per opponent, over the games against it.

    Args:
        games: The games played, in any order.
        names: The opponents to report on; each must have been played.

    Returns:
        One ``Margin`` per name, over ``ours - theirs``.
    """
    out = {}
    for name in names:
        gaps = [game.ours - game.theirs for game in games if game.opponent == name]
        mean = sum(gaps) / len(gaps)
        # The games against one opponent are the same seasons played in both
        # seats, so this is a paired comparison and the error is small: the
        # episode's own swing lands on both sides and cancels. Measured
        # 2026-09-10: a plan's bank varies by 19.5% across seasons, and the
        # gap between two plans on the same seasons by a quarter of that.
        spread = statistics.stdev(gaps) if len(gaps) > 1 else 0.0
        out[name] = Margin(
            mean=mean,
            worst=min(gaps),
            best=max(gaps),
            error=spread / len(gaps) ** 0.5,
        )
    return out


def _failed(work: Work, player: int, error: Exception, worst: float) -> Game:
    """Record a crash as a name and an exception type, and nothing else.

    Opponents are loaded with their real path as ``__code__.co_filename``, so
    letting the exception itself cross back out of the pool would hand a
    sandbox the path in a remote traceback -- and a ``SyntaxError`` carries the
    filename in its own message besides. Only the type name travels.
    """
    _, opponent_name, _, seed, seat, _ = work
    who = "candidate" if player == seat else opponent_name
    return Game(
        opponent=opponent_name,
        seed=seed,
        seat=seat,
        ours=0.0,
        theirs=0.0,
        worst_step_seconds=worst,
        error=f"seat {player} ({who}) raised {type(error).__name__}",
        culprit=who,
    )


def _fertilised(tiles: list, day: int) -> int:
    """Growing tiles still under fertilizer on ``day``.

    A tile carries the day its fertilizer runs out, so "fertilised" is a
    question about now rather than about what was ever applied.
    """
    return sum(
        1
        for row in tiles
        for tile in row
        if isinstance(tile, dict)
        and tile.get("crop")
        and int(tile.get("fertilized_until_day", -1)) > day
    )


def _worked(tiles: list) -> tuple[dict[str, int], dict[str, int], int]:
    """What a board is growing: crops by kind, animals by species, weeds.

    A tile is ``None`` when it is owned and bare, the string ``"LOCKED"``
    when it is not owned yet, and otherwise a dict with a ``kind``. Only
    ``PLANT`` carries a crop and only a stocked pen carries an animal, so an
    empty coop counts as neither and is not a row this needs to explain.

    Args:
        tiles: One farm's ``tiles`` grid, as the observation renders it.

    Returns:
        Crop counts, animal counts, and the number of tiles lost to weeds.
    """
    plants: dict[str, int] = {}
    animals: dict[str, int] = {}
    weeds = 0
    for row in tiles:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            if tile["kind"] == "WEED":
                weeds += 1
            elif crop := tile.get("crop"):
                plants[crop] = plants.get(crop, 0) + 1
            elif animal := tile.get("animal"):
                animals[animal] = animals.get(animal, 0) + 1
    return plants, animals, weeds


def _tally(traded: dict[int, dict[str, list[int]]], actions: Sequence[Any]) -> None:
    """Fold one turn's market orders, both sides, into their running totals.

    Counted before the day's row is taken, not after. The order a side submits
    while looking at the close of a day is that day's last decision, and the
    extraction attributes it the same way -- an order at step ``t`` belongs to
    the state at ``t-1`` it was chosen from. Counted after, every day's trading
    would be reported one day late.

    Args:
        traded: Running totals by seat, ``{verb: [orders, units]}``.
        actions: What each player returned this turn. A candidate's action is
            whatever its code produced, so nothing here assumes a shape.
    """
    for player, action in enumerate(actions):
        orders = action.get("market") if isinstance(action, dict) else None
        for order in orders or ():
            parts = list(order) if isinstance(order, list | tuple) else [order]
            # An empty order has no verb to count. The engine drops one
            # silently, so a program can emit it and play a perfectly good
            # game -- `ahmedberatozer_notebook07b5f4563e` submits
            # `[['HIRE'], []]` on step 121 and four harvested opponents did
            # the same. This used to read `parts[0]` regardless, so the
            # bookkeeping raised `IndexError` where the game itself had no
            # complaint, and an opponent doing something legal took the whole
            # campaign down on the next gate.
            if not parts:
                continue
            running = traded[player].setdefault(str(parts[0]), [0, 0])
            running[0] += 1
            running[1] += int(parts[2]) if len(parts) > 2 else 0


def _day(
    engine: Engine, seat: int, day: int, traded: dict[int, dict[str, list[int]]]
) -> Day:
    """The row for ``day``, read off the engine as that day closed.

    Args:
        engine: The episode, standing at the state that closed ``day``.
        seat: The seat the candidate holds.
        day: The day this row is for.
        traded: Each seat's running market totals, by engine player index.

    Returns:
        One ``Day``: both banks, both farms, both sheds, our seed, both hand
        counts, the prices.
    """
    ours = engine.observation(seat)
    mine_private = render_private(engine.state.farms[seat])
    yours_private = render_private(engine.state.farms[1 - seat])
    theirs = yours_private["shed"]
    mine = ours["farms"][seat]
    yours = ours["farms"][1 - seat]
    ours_plants, ours_animals, ours_weeds = _worked(mine["tiles"])
    their_plants, their_animals, their_weeds = _worked(yours["tiles"])
    return Day(
        day=day,
        ours_bank=engine.bank(seat),
        theirs_bank=engine.bank(1 - seat),
        ours_plants=ours_plants,
        theirs_plants=their_plants,
        ours_animals=ours_animals,
        theirs_animals=their_animals,
        ours_weeds=ours_weeds,
        theirs_weeds=their_weeds,
        ours_quadrants=len(mine.get("unlocked_quadrants") or []),
        theirs_quadrants=len(yours.get("unlocked_quadrants") or []),
        ours_fertilised=_fertilised(mine["tiles"], day),
        theirs_fertilised=_fertilised(yours["tiles"], day),
        ours=dataset.measures(
            mine, mine_private, ours.get("town") or {}, day, traded[seat]
        ),
        theirs=dataset.measures(
            yours, yours_private, ours.get("town") or {}, day, traded[1 - seat]
        ),
        ours_seeds={crop: n for crop, n in ours["private"]["seeds"].items() if n},
        ours_shed={item: n for item, n in ours["private"]["shed"].items() if n},
        theirs_shed={item: n for item, n in theirs.items() if n},
        ours_hands=len(ours["farms"][seat]["hands"]),
        theirs_hands=len(ours["farms"][1 - seat]["hands"]),
        prices=ours["market"]["prices"],
    )


def _one(work: Work) -> Game:
    """Play one game on the engine port. Runs in a fresh process per game.

    Playing a candidate executes it, and a candidate is evolved source that
    may write files, so the game runs with the working directory moved into a
    scratch tree that is removed afterwards. This is the one place every
    execution path -- fast, deep, and a sandbox's own ``campaign play`` --
    passes through, so isolating here isolates all of them. Both sources are
    resolved to absolute paths before the move, because a relative one stops
    resolving the moment it happens, and the cwd is restored before the
    scratch tree is removed so nothing is left standing in a deleted
    directory.
    """
    agent_path, opponent_name, opponent_path, seed, seat, days = work
    resolved = (
        str(Path(agent_path).resolve()),
        opponent_name,
        str(Path(opponent_path).resolve()),
        seed,
        seat,
        days,
    )
    try:
        return _play_one(resolved)
    except Exception as error:
        # A process pool sends the exception back without the frames that
        # raised it, so a failure here reaches the loop as a bare `IndexError`
        # or `FileNotFoundError` with nothing to chase. Three separate crashes
        # on 2026-09-11 each cost a reproduction to find out which opponent
        # and which line, so the game says so itself.
        raise RuntimeError(
            f"{type(error).__name__} playing {opponent_name} "
            f"on seed {seed} from seat {seat}: {error}"
        ) from error


def _play_one(work: Work) -> Game:
    """Play one game; the caller has moved to a scratch cwd and resolved the paths."""
    agent_path, opponent_name, opponent_path, seed, seat, days = work
    sources = [agent_path, opponent_path]
    if seat == 1:
        sources.reverse()
    agents = []
    for player, source in enumerate(sources):
        try:
            agents.append(load_agent(Path(source)))
        except Exception as error:  # noqa: BLE001 - a failure, never a path
            return _failed(work, player, error, 0.0)
    arities = [argument_count(agent) for agent in agents]
    engine = Engine(seed=seed)
    conf = configuration()
    worst = 0.0
    rows: list[Day] = []
    traded: dict[int, dict[str, list[int]]] = {0: {}, 1: {}}
    while not engine.done:
        actions = []
        for player, agent in enumerate(agents):
            observation = structify(
                {
                    **engine.observation(player),
                    "step": engine.step_index,
                    "remainingOverageTime": OVERAGE_SECONDS,
                }
            )
            started = perf_counter()
            try:
                actions.append(agent(*(observation, conf)[: arities[player]]))
            except Exception as error:  # noqa: BLE001 - a failure, never a path
                return _failed(work, player, error, worst)
            elapsed = perf_counter() - started
            if player == seat:
                worst = max(worst, elapsed)
        _tally(traded, actions)
        # The last hour of the day, before the step that ends it takes the
        # hands away with it. The final day never reaches this branch: its
        # hour 23 is the terminal state, which is the row below.
        if days and engine.state.hour == LAST_HOUR:
            rows.append(_day(engine, seat, engine.state.day, traded))
        engine.step(actions[0], actions[1])
    if days:
        rows.append(_day(engine, seat, engine.state.day, traded))
    ours, theirs = engine.bank(seat), engine.bank(1 - seat)
    return Game(
        opponent=opponent_name,
        seed=seed,
        seat=seat,
        ours=ours,
        theirs=theirs,
        worst_step_seconds=worst,
        days=rows,
    )


def play(
    agent: Path,
    opponents: Sequence[str],
    seeds: Sequence[int],
    workers: int,
    days: bool = False,
    pool: Path | None = None,
) -> list[Game]:
    """Play every (opponent, seed, seat) on the port, sampling the reference engine.

    There was a second entry point here, ``play_unsealed``, and a check that
    refused to play a reserved block of seeds through this one. Both are gone
    with the block itself: every seed is drawn fresh now, so there is nothing
    to keep anything away from.

    Args:
        agent: The candidate's ``main.py``.
        opponents: Roster names; a path here is a KeyError.
        seeds: Episode seeds.
        workers: Processes to fan the games over, at most ``config.CORE_BUDGET``.
        days: Record each game's day table. The gate asks for them because one
            of these games is what a model is shown of how its program played.
        pool: The pool file a champion's name resolves through; the
            roster alone when it is None.

    Returns:
        One ``Game`` per opponent, seed and seat, in that order.

    Raises:
        ValueError: More workers than the core budget.
        KeyError: An opponent name not in the roster.
        OpponentCrash: Only opponent seats raised; the pool is broken.
        RuntimeError: The candidate raised during a game, or the
            reference-engine sample disagreed with the port.
    """
    if workers > config.CORE_BUDGET:
        raise ValueError(f"workers exceeds CORE_BUDGET ({config.CORE_BUDGET})")
    paths = {name: roster.path(name, pool) for name in opponents}
    work = [
        (str(agent), name, str(paths[name]), seed, seat, days)
        for name in opponents
        for seed in seeds
        for seat in (0, 1)
    ]
    # Named `executor` rather than `pool`, which is what this was: the
    # opponent pool arrived as a parameter and quietly shadowed it.
    with pools.workers(workers) as executor:
        games = list(executor.map(_one, work))
    # A crash is a failure, never a score: an agent that raised banked its
    # untouched opening money and would otherwise read as an ordinary loss,
    # which is the same rule `arena.run_banks` enforces on the reference
    # engine. `from None` because the only chained context available here is
    # the one that names the opponent's file. Which side raised decides the
    # exception type: a candidate's crash is the candidate's failure, while a
    # crash confined to opponent seats is a broken pool the loop must not
    # charge to the lineage under test.
    broken = [game for game in games if game.error]
    failures = [f"seed {game.seed}: {game.error}" for game in broken]
    if any(game.culprit == "candidate" for game in broken):
        raise RuntimeError(
            "agents raised during play -- " + "; ".join(failures)
        ) from None
    if broken:
        culprits = sorted({game.culprit for game in broken if game.culprit})
        raise OpponentCrash(
            f"opponent(s) {', '.join(culprits)} raised during play -- "
            + "; ".join(failures)
        ) from None
    _verify_sample(agent, paths, games)
    return games


def _verify_sample(agent: Path, paths: dict[str, Path], games: list[Game]) -> None:
    """Replay roughly 2% of the games on the reference engine and compare banks.

    The draw comes from the system entropy source, not a seeded generator: the
    thing being sampled is an evolved agent's own source, and a seed derived
    from the run's shape -- the opponents and seeds named on the command line
    -- would make the audited games predictable to whatever is being audited.
    """
    rng = random.SystemRandom()
    for game in games:
        if rng.random() >= REFERENCE_SAMPLE:
            continue
        opponent = str(paths[game.opponent])
        seating = (str(agent), opponent) if game.seat == 0 else (opponent, str(agent))
        banks = _replay(*seating, game.seed)
        ours, theirs = banks if game.seat == 0 else banks[::-1]
        if (float(ours), float(theirs)) != (game.ours, game.theirs):
            raise RuntimeError(
                f"engine drift: seed {game.seed} seat {game.seat} vs {game.opponent}: "
                f"port {game.ours}/{game.theirs}, reference {ours}/{theirs}"
            )


def _replay(seat_zero: str, seat_one: str, seed: int) -> tuple[int, int]:
    """Play one audited game on the reference engine, away from the caller.

    ``arena.run_banks`` hands both files to ``kaggle_environments``, which
    execs them in the calling process -- so an audited candidate would run
    wherever its caller stands, which is the one thing ``_one`` exists to
    prevent. The audit therefore gets a process of its own, and the same
    scratch directory every other game gets.

    Args:
        seat_zero: The file playing seat zero.
        seat_one: The file playing seat one.
        seed: The episode seed.

    Returns:
        Both seats' final banks, seat zero first.
    """
    with pools.workers(1) as pool:
        return pool.submit(_reference, (seat_zero, seat_one, seed)).result()


def _reference(work: tuple[str, str, int]) -> tuple[int, int]:
    """Run one reference-engine game in a scratch directory. Runs in a child."""
    seat_zero, seat_one, seed = work
    sources = (str(Path(seat_zero).resolve()), str(Path(seat_one).resolve()))
    return arena.run_banks(*sources, seed)


def check(agent: Path, steps: int = EPISODE_STEPS) -> CheckReport:
    """Load as Kaggle does and play ``steps`` turns against itself on the reference.

    Runs in a scratch directory, because loading a program runs its top-level
    code and playing it runs the rest. `_one` and `_reference` have both moved
    the working directory for this reason all along; this did not, and it is
    the path the *least* trusted code in the system takes -- `harvest` calls it
    on a kernel downloaded from the competition minutes earlier, before that
    kernel is anything but a file we fetched.

    Found on 2026-09-11, after a harvested agent overwrote the repository's own
    `main.py` with a 158KB replay agent and left a `teacher_bootstrap.tar.gz`
    beside it. Nothing was lost -- `main.py` is three tracked lines -- and the
    same write against an untracked file would not have been noticed at all.

    Args:
        agent: The candidate's ``main.py``.
        steps: How many turns to play before stopping.

    Returns:
        What happened: whether it loaded, its bank, its worst call, any failure.
    """
    with pools.workers(1) as executor:
        return executor.submit(_checked, str(Path(agent).resolve()), steps).result()


def _checked(agent: str, steps: int) -> CheckReport:
    """Move to a scratch directory and check there. Runs in a child.

    A child, because `os.chdir` is the whole process's and this one is its
    own. The first version of this moved the *caller's* directory, which was
    briefly correct and then catastrophic: a spawned worker inherits the cwd
    of whoever spawned it, so every game the loop started while a harvest was
    checking a kernel began life inside that check's scratch tree -- and when
    the check finished and removed it, those workers were standing in a
    directory that no longer existed. The campaign died on
    `FileNotFoundError: /tmp/campaign-check-56ludna7` an hour after the
    isolation was added to stop a harvested agent writing into the repository.

    `_one` and `_reference` chdir freely because each already has a process to
    itself. This is the same trick paid for honestly.
    """
    return _check(Path(agent), steps)


def _check(agent: Path, steps: int) -> CheckReport:
    """The check itself; the caller has moved to a scratch cwd."""
    try:
        policy = load_agent(agent)
    except Exception as error:  # noqa: BLE001 - the report is the point
        return CheckReport(
            loaded=False,
            bank=0.0,
            worst_step_seconds=0.0,
            error=f"{type(error).__name__}: {error}",
        )
    arity = argument_count(policy)
    environment = make(ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS})
    environment.reset()
    worst = 0.0
    # Seat zero's actions as they are emitted, so the identity below costs
    # nothing beyond the game this already plays.
    played = hashlib.sha256()
    try:
        for _ in range(steps):
            if environment.done:
                break
            actions = []
            for state in environment.state:
                observation = structify(
                    {**dict(state.observation), "step": shared_step(environment)}
                )
                arguments = (observation, environment.configuration)[:arity]
                started = perf_counter()
                actions.append(policy(*arguments))
                worst = max(worst, perf_counter() - started)
            played.update(repr(actions[0]).encode())
            environment.step(actions)
    except Exception as error:  # noqa: BLE001
        return CheckReport(
            loaded=True,
            bank=0.0,
            worst_step_seconds=worst,
            error=f"{type(error).__name__}: {error}",
        )
    return CheckReport(
        loaded=True,
        bank=float(environment.state[0].observation["farms"][0]["money"]),
        worst_step_seconds=worst,
        error=None,
        fingerprint=played.hexdigest()[:16],
    )


def package(agent: Path, output: Path) -> Path:
    """Write ``main.py`` and the repository ``LICENSE`` into a tarball.

    The only packager: ``uv run package`` and ``uv run submit`` call this too.
    A program is one file whose last top-level callable is ``agent``, so that
    file and the licence are the whole archive -- no engine library, no
    ``kaggriculture`` package. Shipping either would let a candidate depend on
    something the search can edit but the ladder cannot see, and what evolves
    would stop being what ships.

    Args:
        agent: The self-contained ``main.py`` to ship.
        output: Where to write the tarball.

    Returns:
        ``output``, unchanged.
    """
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        shutil.copy(agent, root / "main.py")
        shutil.copy(config.ROOT / "LICENSE", root / "LICENSE")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output, "w:gz") as tar:
            for item in sorted(root.rglob("*")):
                # `rglob` already yields every child, so an added directory
                # must not recurse or each module lands in the archive twice.
                tar.add(item, arcname=str(item.relative_to(root)), recursive=False)
    return output


if __name__ == "__main__":
    main()
