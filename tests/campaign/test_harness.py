"""What a sandbox may run, and what it may not."""

import traceback
from pathlib import Path

import pytest

from kaggriculture.campaign import config, harness, roster
from kaggriculture.constants import EPISODE_STEPS

# An episode records `EPISODE_STEPS` states and so takes one fewer transition;
# the terminal state is never acted on, so the last counter an agent sees is
# two below the configured length. Measured on the reference engine, not
# derived: `make(...).reset()` then stepping PASS to `done` last reports 718.
LAST_ACTED_STEP = EPISODE_STEPS - 2

PASS_AGENT = """
def agent(observation, configuration=None):
    return {"farmer": ["PASS"], "hands": [], "market": []}
"""

SLOW_AGENT = """
import time
def agent(observation, configuration=None):
    time.sleep(0.6)
    return {"farmer": ["PASS"], "hands": [], "market": []}
"""

# The Kaggle runner truncates (observation, configuration) to the callable's
# arity, so an agent declaring one parameter is called with one. Held-out
# opponent salemali7_2900 is such an agent and raised on its first turn until
# the harness matched the runner.
ONE_ARGUMENT_AGENT = """
def agent(observation):
    return {"farmer": ["PASS"], "hands": [], "market": []}
"""


# The port exports `step` on seat zero's observation only; the framework
# writes it onto every seat at call time, and vendored agents index their
# opening book by it. An agent that saw a frozen 0 would replay turn one for
# the whole episode without ever raising.
STEP_RECORDING_AGENT = """
from pathlib import Path

DIRECTORY = Path("{directory}")


def agent(observation, configuration=None):
    seat = observation["player"]
    (DIRECTORY / f"seat{{seat}}.txt").write_text(
        str(observation["step"]), encoding="utf-8"
    )
    return {{"farmer": ["PASS"], "hands": [], "market": []}}
"""


# An opponent is loaded with its real path as `__code__.co_filename`, so its
# traceback names a file the sandbox is never allowed to learn.
CRASHING_AGENT = """
def agent(observation, configuration=None):
    raise ZeroDivisionError("the message itself must not travel either")
"""

# A candidate is evolved source, so it may write files. It must not write them
# where the caller lives.
WRITING_AGENT = """
from pathlib import Path


def agent(observation, configuration=None):
    Path("scribble.txt").write_text("x", encoding="utf-8")
    return {"farmer": ["PASS"], "hands": [], "market": []}
"""


@pytest.fixture
def step_recording_agent(tmp_path: Path) -> Path:
    """Write an agent that records ``observation["step"]`` per seat it plays."""
    path = tmp_path / "main.py"
    path.write_text(STEP_RECORDING_AGENT.format(directory=tmp_path), encoding="utf-8")
    return path


@pytest.fixture
def one_argument_agent(tmp_path: Path) -> Path:
    """Write an agent whose signature declares only ``observation``."""
    path = tmp_path / "main.py"
    path.write_text(ONE_ARGUMENT_AGENT, encoding="utf-8")
    return path


@pytest.fixture
def pass_agent(tmp_path: Path) -> Path:
    """Write the agent that does nothing at all, on any observation."""
    path = tmp_path / "main.py"
    path.write_text(PASS_AGENT, encoding="utf-8")
    return path


def test_play_refuses_exam_seeds(pass_agent: Path) -> None:
    """The exam seeds belong to the deep evaluation; the harness will not spend them."""
    with pytest.raises(ValueError, match="exam"):
        harness.play(pass_agent, ["v54"], [config.EXAM_SEEDS[0]], workers=1)


def test_play_refuses_a_path_where_a_name_belongs(pass_agent: Path) -> None:
    """A sandbox may name an opponent, never point at one."""
    with pytest.raises(KeyError):
        harness.play(
            pass_agent,
            ["/data/kaggriculture/opponents/kaito_v54/main.py"],
            [1],
            workers=1,
        )


@pytest.mark.local_data
def test_play_reports_both_seats_and_latency(pass_agent: Path) -> None:
    """Every seed is played from both seats, timing the candidate's own calls."""
    games = harness.play(pass_agent, ["v54"], [1, 2], workers=2)
    assert [(g.seed, g.seat) for g in games] == [(1, 0), (1, 1), (2, 0), (2, 1)]
    assert all(g.ours == 3000.0 for g in games)
    assert all(g.worst_step_seconds < 0.5 for g in games)


@pytest.mark.local_data
def test_play_passes_each_side_the_arguments_its_signature_declares(
    one_argument_agent: Path,
) -> None:
    """Both the candidate and the opponent here take one argument."""
    games = harness.play(one_argument_agent, ["salemali7_2900"], [11], workers=2)
    assert all(game.ours == 3000.0 and game.theirs > 3000.0 for game in games)


@pytest.mark.local_data
def test_play_gives_both_seats_the_step_counter(
    step_recording_agent: Path, tmp_path: Path
) -> None:
    """Seat one gets the counter the framework, not the interpreter, writes."""
    harness.play(step_recording_agent, ["v54"], [11], workers=2)
    seen = {
        seat: int((tmp_path / f"seat{seat}.txt").read_text(encoding="utf-8"))
        for seat in (0, 1)
    }
    assert seen == {0: LAST_ACTED_STEP, 1: LAST_ACTED_STEP}


def test_a_crashing_opponent_is_a_failure_that_never_names_its_file(
    pass_agent: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The crash is reported by roster name and exception type, and by nothing else."""
    hidden = tmp_path / "secret_dir"
    hidden.mkdir()
    (hidden / "main.py").write_text(CRASHING_AGENT, encoding="utf-8")
    monkeypatch.setitem(roster.TRAINING, "crasher", hidden / "main.py")

    with pytest.raises(RuntimeError) as caught:
        harness.play(pass_agent, ["crasher"], [11], workers=2)

    message = str(caught.value)
    assert "crasher" in message and "ZeroDivisionError" in message
    rendered = "".join(traceback.format_exception(caught.value))
    assert "secret_dir" not in message
    assert "secret_dir" not in rendered
    assert "the message itself must not travel either" not in rendered


def test_a_crash_confined_to_opponent_seats_is_its_own_exception(
    pass_agent: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken opponent is a broken pool, not the candidate's failure.

    The loop writes a candidate's failure into its lineage's feedback and the
    next prompt chases it, so an opponent's crash must arrive as a different
    exception -- one the loop does not catch.
    """
    hidden = tmp_path / "secret_dir"
    hidden.mkdir()
    (hidden / "main.py").write_text(CRASHING_AGENT, encoding="utf-8")
    monkeypatch.setitem(roster.TRAINING, "crasher", hidden / "main.py")

    with pytest.raises(harness.OpponentCrash) as caught:
        harness.play(pass_agent, ["crasher"], [11], workers=2)

    assert "crasher" in str(caught.value)
    assert "secret_dir" not in str(caught.value)


@pytest.mark.local_data
def test_a_crashing_candidate_is_the_candidates_own_failure(
    tmp_path: Path,
) -> None:
    """The candidate raised, so the loop must be free to record it and go on."""
    candidate = tmp_path / "main.py"
    candidate.write_text(CRASHING_AGENT, encoding="utf-8")

    with pytest.raises(RuntimeError) as caught:
        harness.play(candidate, ["v54"], [11], workers=2)

    assert not isinstance(caught.value, harness.OpponentCrash)
    assert "candidate" in str(caught.value)


@pytest.mark.local_data
def test_a_candidate_writing_files_leaves_nothing_in_the_callers_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Playing a candidate runs it; every path that runs one is a scratch directory.

    ``REFERENCE_SAMPLE`` is forced to 1 so the reference-engine audit runs on
    every game: it hands the candidate's file to ``kaggle_environments``,
    which execs it, and at 2% a hole there shows up as a test that fails one
    run in twelve.
    """
    monkeypatch.setattr(harness, "REFERENCE_SAMPLE", 1.0)
    candidate = tmp_path / "main.py"
    candidate.write_text(WRITING_AGENT, encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)

    harness.play(candidate, ["v54"], [11], workers=2)

    assert list(workspace.iterdir()) == []
    assert Path.cwd() == workspace


@pytest.mark.local_data
def test_play_accepts_a_relative_candidate_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scratch move happens after the paths resolve, or nothing would load."""
    (tmp_path / "main.py").write_text(PASS_AGENT, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    games = harness.play(Path("main.py"), ["v54"], [11], workers=2)

    assert all(game.error is None for game in games)


def test_the_reference_sample_is_drawn_from_the_system_entropy_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A seeded draw would be reproducible from the run shape the sandbox chose."""
    games = [
        harness.Game(
            opponent="v54",
            seed=seed,
            seat=0,
            ours=1.0,
            theirs=2.0,
            worst_step_seconds=0.0,
        )
        for seed in (1, 2, 3, 4)
    ]
    built: list[object] = []
    # Below REFERENCE_SAMPLE selects the game; at or above it skips.
    draws = iter([0.9, 0.0, 0.9, 0.0])

    class Recording:
        """Stand in for ``random.SystemRandom`` and count its construction."""

        def __init__(self) -> None:
            built.append(self)

        def random(self) -> float:
            """Return the next scripted draw."""
            return next(draws)

    audited: list[int] = []

    def spy(seat_zero: str, seat_one: str, seed: int) -> tuple[int, int]:
        audited.append(seed)
        return 1, 2

    monkeypatch.setattr(harness.random, "SystemRandom", Recording)
    monkeypatch.setattr(harness, "_replay", spy)
    harness._verify_sample(tmp_path / "main.py", {"v54": tmp_path / "v54.py"}, games)

    assert len(built) == 1
    assert audited == [2, 4]


def test_play_caps_workers_at_the_core_budget(pass_agent: Path) -> None:
    """The box's other tenants keep the cores the budget reserves for them."""
    with pytest.raises(ValueError, match="CORE_BUDGET"):
        harness.play(pass_agent, ["v54"], [1], workers=config.CORE_BUDGET + 1)


def test_the_cli_refuses_more_than_sixteen_games(
    pass_agent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AGENTS.md promises a sandbox at most 16 games; the command keeps the promise."""
    monkeypatch.setattr(
        harness.sys,
        "argv",
        ["campaign", "play", str(pass_agent), "--vs", "v54", "--seeds", "1-9"],
    )
    with pytest.raises(SystemExit, match="16"):
        harness.main()


def test_the_cli_refuses_more_workers_than_a_sandbox_may_take(
    pass_agent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rest of the box is running the loop."""
    monkeypatch.setattr(
        harness.sys,
        "argv",
        [
            "campaign",
            "play",
            str(pass_agent),
            "--vs",
            "v54",
            "--seeds",
            "1-1",
            "--workers",
            "9",
        ],
    )
    with pytest.raises(SystemExit, match="workers"):
        harness.main()


def test_check_loads_as_kaggle_does_and_times_steps(pass_agent: Path) -> None:
    """A well-formed agent loads, keeps its opening bank and reports its latency."""
    report = harness.check(pass_agent)
    assert report.loaded and report.error is None and report.bank == 3000.0
    assert report.worst_step_seconds < 0.5


def test_check_calls_a_one_argument_agent_the_way_kaggle_does(
    one_argument_agent: Path,
) -> None:
    """An agent declaring one parameter is called with one."""
    report = harness.check(one_argument_agent, steps=5)
    assert report.loaded and report.error is None


def test_check_gives_both_seats_the_step_counter(
    step_recording_agent: Path, tmp_path: Path
) -> None:
    """`check` drives the state directly, so it owes seat one the shared copy."""
    report = harness.check(step_recording_agent, steps=5)
    assert report.loaded and report.error is None
    seen = {
        seat: int((tmp_path / f"seat{seat}.txt").read_text(encoding="utf-8"))
        for seat in (0, 1)
    }
    assert seen == {0: 4, 1: 4}


def test_check_flags_a_slow_agent(tmp_path: Path) -> None:
    """A per-call cost over the budget is reported rather than averaged away."""
    slow = tmp_path / "main.py"
    slow.write_text(SLOW_AGENT, encoding="utf-8")
    report = harness.check(slow, steps=5)
    assert report.worst_step_seconds >= 0.5


def test_package_places_main_and_the_engine_library_at_the_root(
    pass_agent: Path, tmp_path: Path
) -> None:
    """What Kaggle unpacks: entrypoint, library, plumbing, no campaign."""
    import tarfile

    archive = harness.package(pass_agent, tmp_path / "submission.tar.gz")
    with tarfile.open(archive) as tar:
        listed = tar.getnames()
    names = set(listed)
    assert "main.py" in names and "kaggriculture_engine.so" in names
    assert "kaggriculture/constants.py" in names
    assert not any(name.startswith("kaggriculture/campaign") for name in names)
    # A duplicated member is invisible to a set of names and to `tar -x`, which
    # simply overwrites; it doubles the upload and reads as a corrupt archive.
    assert sorted(listed) == sorted(names)
