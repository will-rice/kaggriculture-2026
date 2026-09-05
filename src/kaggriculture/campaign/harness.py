"""What a mutation sandbox may run: play by name, check as Kaggle loads, package.

Games run on the engine port; a 2% sample is replayed on the reference engine
and any bank disagreement fails the call, so drift between the two engines is
a harness error rather than a quietly wrong fitness.
"""

import argparse
import logging
import os
import random
import shutil
import sys
import tarfile
import tempfile
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

from kaggle_environments import make
from kaggle_environments.agent import get_last_callable
from kaggle_environments.core import Environment
from kaggle_environments.utils import Struct, structify
from pydantic import BaseModel

from kaggriculture.campaign import arena, config, roster
from kaggriculture.campaign.engine.wrapper import Engine
from kaggriculture.constants import ENVIRONMENT, EPISODE_STEPS

LOGGER = logging.getLogger(__name__)
REFERENCE_SAMPLE = 0.02
LATENCY_BUDGET = 0.5  # half of actTimeout
PACKAGE_MODULES = ("__init__.py", "constants.py", "observation.py", "actions.py")
# What `campaign play` promises a sandbox in AGENTS.md, enforced on the CLI
# only: the library function is what the evaluator calls, and it plays the
# whole exam block.
SANDBOX_GAME_CAP = 16
SANDBOX_WORKER_CAP = 8
# The framework, not the interpreter, writes these onto every seat's
# observation at call time. The port exports neither, so `_one` injects both.
OVERAGE_SECONDS = 60


class OpponentCrash(RuntimeError):  # noqa: N818 - a crash, not our error
    """An opponent, not the candidate, raised during a game.

    Attributing this to the candidate would write a failure into its lineage's
    feedback for something it did not do, and the next prompt would chase it.
    A broken opponent is a broken pool, so the loop lets this one out (spec
    section 8): it halts loudly rather than scoring around the hole.
    """


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
    """

    opponent: str
    seed: int
    seat: int
    ours: float
    theirs: float
    worst_step_seconds: float
    error: str | None = None
    culprit: str | None = None


class CheckReport(BaseModel):
    """Whether an agent loads and runs the way the Kaggle runner would load it.

    Attributes:
        loaded: Whether ``get_last_callable`` produced a callable.
        bank: Seat zero's money when the run stopped.
        worst_step_seconds: The slowest single call to the agent.
        error: The failure, or None if the run finished.
    """

    loaded: bool
    bank: float
    worst_step_seconds: float
    error: str | None


def main() -> None:
    """``campaign play|check|package|loop|dry-run``."""
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
            "campaign loop [--iterations N] [--workers N] [--concurrency N] "
            "[--seed-agent PATH] runs the evolution loop; campaign dry-run "
            "takes the same flags and runs it on a fake mutator."
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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
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


def _failed(
    work: tuple[str, str, str, int, int], player: int, error: Exception, worst: float
) -> Game:
    """Record a crash as a name and an exception type, and nothing else.

    Opponents are loaded with their real path as ``__code__.co_filename``, so
    letting the exception itself cross back out of the pool would hand a
    sandbox the path in a remote traceback -- and a ``SyntaxError`` carries the
    filename in its own message besides. Only the type name travels.
    """
    _, opponent_name, _, seed, seat = work
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


def _one(work: tuple[str, str, str, int, int]) -> Game:
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
    agent_path, opponent_name, opponent_path, seed, seat = work
    resolved = (
        str(Path(agent_path).resolve()),
        opponent_name,
        str(Path(opponent_path).resolve()),
        seed,
        seat,
    )
    origin = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="campaign-game-") as scratch:
        os.chdir(scratch)
        try:
            return _play_one(resolved)
        finally:
            os.chdir(origin)


def _play_one(work: tuple[str, str, str, int, int]) -> Game:
    """Play one game; the caller has moved to a scratch cwd and resolved the paths."""
    agent_path, opponent_name, opponent_path, seed, seat = work
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
        engine.step(actions[0], actions[1])
    ours, theirs = engine.bank(seat), engine.bank(1 - seat)
    return Game(
        opponent=opponent_name,
        seed=seed,
        seat=seat,
        ours=ours,
        theirs=theirs,
        worst_step_seconds=worst,
    )


def play(
    agent: Path, opponents: Sequence[str], seeds: Sequence[int], workers: int
) -> list[Game]:
    """Play every (opponent, seed, seat) on the port, sampling the reference engine.

    Args:
        agent: The candidate's ``main.py``.
        opponents: Roster names; a path here is a KeyError.
        seeds: Episode seeds; an exam seed here is a ValueError.
        workers: Processes to fan the games over, at most ``config.CORE_BUDGET``.

    Returns:
        One ``Game`` per opponent, seed and seat, in that order.

    Raises:
        ValueError: An exam seed, or more workers than the core budget.
        KeyError: An opponent name not in the roster.
        OpponentCrash: Only opponent seats raised; the pool is broken.
        RuntimeError: The candidate raised during a game, or the
            reference-engine sample disagreed with the port.
    """
    return _play(agent, opponents, seeds, workers, sealed=True)


def play_unsealed(
    agent: Path, opponents: Sequence[str], seeds: Sequence[int], workers: int
) -> list[Game]:
    """``play`` for the gate: the exam seeds are allowed. Never exposed on the CLI.

    Args:
        agent: The candidate's ``main.py``.
        opponents: Roster names; a path here is a KeyError.
        seeds: Episode seeds, exam seeds included.
        workers: Processes to fan the games over, at most ``config.CORE_BUDGET``.

    Returns:
        One ``Game`` per opponent, seed and seat, in that order.

    Raises:
        ValueError: More workers than the core budget.
        KeyError: An opponent name not in the roster.
        OpponentCrash: Only opponent seats raised; the pool is broken.
        RuntimeError: The candidate raised during a game, or the
            reference-engine sample disagreed with the port.
    """
    return _play(agent, opponents, seeds, workers, sealed=False)


def _play(
    agent: Path,
    opponents: Sequence[str],
    seeds: Sequence[int],
    workers: int,
    sealed: bool,
) -> list[Game]:
    """Shared body of ``play`` and ``play_unsealed``; only the exam check differs."""
    if sealed and any(seed in config.EXAM_SEEDS for seed in seeds):
        raise ValueError("exam seeds are sealed; the harness will not play them")
    if workers > config.CORE_BUDGET:
        raise ValueError(f"workers exceeds CORE_BUDGET ({config.CORE_BUDGET})")
    paths = {name: roster.path(name) for name in opponents}
    work = [
        (str(agent), name, str(paths[name]), seed, seat)
        for name in opponents
        for seed in seeds
        for seat in (0, 1)
    ]
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as pool:
        games = list(pool.map(_one, work))
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
    with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as pool:
        return pool.submit(_reference, (seat_zero, seat_one, seed)).result()


def _reference(work: tuple[str, str, int]) -> tuple[int, int]:
    """Run one reference-engine game in a scratch directory. Runs in a child."""
    seat_zero, seat_one, seed = work
    sources = (str(Path(seat_zero).resolve()), str(Path(seat_one).resolve()))
    origin = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="campaign-reference-") as scratch:
        os.chdir(scratch)
        try:
            return arena.run_banks(*sources, seed)
        finally:
            os.chdir(origin)


def check(agent: Path, steps: int = EPISODE_STEPS) -> CheckReport:
    """Load as Kaggle does and play ``steps`` turns against itself on the reference.

    Args:
        agent: The candidate's ``main.py``.
        steps: How many turns to play before stopping.

    Returns:
        What happened: whether it loaded, its bank, its worst call, any failure.
    """
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
    )


def package(agent: Path, output: Path) -> Path:
    """Write ``main.py`` + the engine library + the plumbing package into a tarball.

    The only packager: ``uv run package`` and ``uv run submit`` call this too,
    so the artefact ``kaggle_image.load_test`` proves is the artefact that
    ships. The engine's ``NOTICE`` and the repository ``LICENSE`` ride along
    at the root because the library in the archive is a port of Apache-2.0
    kernel source and the attribution has to travel with the binary.

    Args:
        agent: The self-contained ``main.py`` to ship.
        output: Where to write the tarball.

    Returns:
        ``output``, unchanged.
    """
    from kaggriculture.campaign.engine import build

    library = build.build()
    package_root = Path(config.ROOT / "src" / "kaggriculture")
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        shutil.copy(agent, root / "main.py")
        shutil.copy(library, root / library.name)
        shutil.copy(library.parent / "NOTICE", root / "NOTICE")
        license_file = config.ROOT / "LICENSE"
        if license_file.exists():
            shutil.copy(license_file, root / "LICENSE")
        (root / "kaggriculture").mkdir()
        for module in PACKAGE_MODULES:
            shutil.copy(package_root / module, root / "kaggriculture" / module)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output, "w:gz") as tar:
            for item in sorted(root.rglob("*")):
                # `rglob` already yields every child, so an added directory
                # must not recurse or each module lands in the archive twice.
                tar.add(item, arcname=str(item.relative_to(root)), recursive=False)
    return output


if __name__ == "__main__":
    main()
