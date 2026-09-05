"""Fast eval on fresh seeds; deep eval on the exam block. Both on the port."""

import random
from pathlib import Path

import pytest

from kaggriculture.campaign import config, evaluator, field_gate, harness, pool
from kaggriculture.report import wilson_interval

# Fits the smallest box the suite runs on: CORE_BUDGET clamps to 1 on a
# 4-core CI runner.
WORKERS = min(4, config.CORE_BUDGET)

PASS = (
    "def agent(observation, configuration=None):\n"
    "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
)
WRITER = (
    "from pathlib import Path\n"
    "\n"
    "def agent(observation, configuration=None):\n"
    "    Path('scribble.txt').write_text('x', encoding='utf-8')\n"
    "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
)
CRASHER = (
    "def agent(observation, configuration=None):\n"
    "    raise ValueError('this candidate is broken')\n"
)


@pytest.mark.local_data
def test_fast_draws_fresh_non_exam_seeds_and_averages_the_pool(
    tmp_path: Path,
) -> None:
    """Fast draws its own seeds from outside the exam block; every opponent counts."""
    agent = tmp_path / "main.py"
    agent.write_text(PASS, encoding="utf-8")
    p = pool.Pool(opponents={"v54": str(config.OPPONENTS / "kaito_v54" / "main.py")})
    result = evaluator.fast(agent, "prog", p, random.Random(1), workers=WORKERS)
    assert len(result.seeds) == config.FAST_SEEDS and not set(result.seeds) & set(
        config.EXAM_SEEDS
    )
    assert result.rates == {"v54": 0.0} and result.fitness == 0.0


@pytest.mark.local_data
def test_two_fast_calls_draw_different_seeds(tmp_path: Path) -> None:
    """Ranking must not reuse one seed block, or drift becomes overfitting."""
    agent = tmp_path / "main.py"
    agent.write_text(PASS, encoding="utf-8")
    p = pool.Pool(opponents={"v54": str(config.OPPONENTS / "kaito_v54" / "main.py")})
    rng = random.Random(2)
    assert (
        evaluator.fast(agent, "prog", p, rng, workers=WORKERS).seeds
        != evaluator.fast(agent, "prog", p, rng, workers=WORKERS).seeds
    )


@pytest.mark.local_data
def test_deep_scores_the_exam_block_with_intervals_and_held_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate reports every pool opponent and every held-out opponent."""
    agent = tmp_path / "main.py"
    agent.write_text(PASS, encoding="utf-8")
    monkeypatch.setattr(config, "EXAM_SEEDS", config.EXAM_SEEDS[:2])
    p = pool.Pool(opponents={"v54": str(config.OPPONENTS / "kaito_v54" / "main.py")})
    result = evaluator.deep(agent, "prog", p, workers=WORKERS)
    assert result.program_id == "prog" and result.score == 0.0 and result.games == 4
    assert result.intervals["v54"][0] == 0.0 and result.intervals["v54"][1] < 1.0
    assert set(result.held_out) == set(evaluator.HELD_OUT)
    assert set(result.rates) == set(p.names())  # held-out never enters the score


def test_a_program_in_the_pool_is_never_played_against_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A champion is a member of the pool it is re-scored on.

    Its own bytes in the other seat are a structural 0.5 that means nothing
    about the program: it would enter the score, the per-opponent rates,
    the intervals, and every rule downstream that reads them -- including
    a gate that asks whether it beat every opponent, which a mirror match
    at exactly one half answers no to. The entry is dropped before a game
    is played.
    """
    champion = tmp_path / "champion_1.py"
    champion.write_text(PASS, encoding="utf-8")
    other = tmp_path / "other.py"
    other.write_text(PASS, encoding="utf-8")
    monkeypatch.setattr(config, "EXAM_SEEDS", config.EXAM_SEEDS[:2])
    monkeypatch.setattr(config, "FAST_SEEDS", 1)
    monkeypatch.setattr(evaluator, "VENDORED", ["other"])
    monkeypatch.setattr(evaluator, "HELD_OUT", [])
    # Names resolve to paths through the saved pool, so it has to be on disk.
    monkeypatch.setattr(config, "POOL", tmp_path / "pool.json")
    p = pool.Pool(opponents={"other": str(other), "champion_1": str(champion)})
    p.save(config.POOL)

    result = evaluator.deep(champion, "champion_1", p, workers=WORKERS)
    ranked = evaluator.fast(champion, "champion_1", p, random.Random(4), WORKERS)

    assert set(result.rates) == {"other"} and set(result.intervals) == {"other"}
    assert set(ranked.rates) == {"other"}
    # One opponent left, so the mean over them is that opponent's rate.
    assert result.score == result.rates["other"]
    assert ranked.fitness == ranked.rates["other"]


def test_a_pool_name_against_a_different_file_is_an_error(tmp_path: Path) -> None:
    """Two programs cannot both be `champion_1`; nothing here can pick one."""
    champion = tmp_path / "champion_1.py"
    champion.write_text(PASS, encoding="utf-8")
    p = pool.Pool(opponents={"champion_1": str(tmp_path / "elsewhere.py")})

    with pytest.raises(ValueError, match="elsewhere.py"):
        evaluator.opponents(p, "champion_1", champion)


def test_deep_bounds_are_the_mean_of_the_per_opponent_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Games against different opponents are not one binomial.

    Two hundred games split between an opponent this beats nine times in ten
    and one it loses to nine times in ten is not the same measurement as two
    hundred coin flips at one half, and treating it as one would report an
    interval narrower than the truth. The rates are fixed here so the
    arithmetic, not the engine, is what is under test.
    """
    agent = tmp_path / "main.py"
    agent.write_text(PASS, encoding="utf-8")
    monkeypatch.setattr(config, "EXAM_SEEDS", config.EXAM_SEEDS[:8])
    fixed = {"v54": 0.9, "v56": 0.1} | dict.fromkeys(evaluator.HELD_OUT, 0.5)
    monkeypatch.setattr(
        field_gate,
        "score_field",
        lambda candidate, seeds, workers, opponents: (
            {n: fixed[n] for n in opponents},
            {n: harness.Margin(mean=0.0, worst=0.0, best=0.0) for n in opponents},
        ),
    )
    p = pool.Pool(
        opponents={
            "v54": str(config.OPPONENTS / "kaito_v54" / "main.py"),
            "v56": str(config.OPPONENTS / "kaito_v56" / "main.py"),
        }
    )
    result = evaluator.deep(agent, "prog", p, workers=WORKERS)
    games = 2 * 8
    strong = wilson_interval(0.9 * games, games)
    weak = wilson_interval(0.1 * games, games)
    assert result.games == games
    assert result.score == pytest.approx(0.5)
    assert result.low == pytest.approx((strong[0] + weak[0]) / 2)
    assert result.high == pytest.approx((strong[1] + weak[1]) / 2)
    assert result.low < result.score < result.high


@pytest.mark.local_data
def test_deep_diverts_what_a_candidate_writes_away_from_the_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Playing a candidate runs it, so its writes land in a scratch directory."""
    agent = tmp_path / "main.py"
    agent.write_text(WRITER, encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(config, "EXAM_SEEDS", config.EXAM_SEEDS[:1])
    p = pool.Pool(opponents={"v54": str(config.OPPONENTS / "kaito_v54" / "main.py")})
    evaluator.deep(agent, "prog", p, workers=WORKERS)
    assert list(workspace.iterdir()) == []
    assert Path.cwd() == workspace


def test_deep_lets_a_crash_propagate_and_still_restores_the_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crashed candidate is a failed evaluation, never a zero score."""
    agent = tmp_path / "main.py"
    agent.write_text(CRASHER, encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(config, "EXAM_SEEDS", config.EXAM_SEEDS[:1])
    p = pool.Pool(opponents={"v54": str(config.OPPONENTS / "kaito_v54" / "main.py")})
    with pytest.raises(RuntimeError):
        evaluator.deep(agent, "prog", p, workers=WORKERS)
    assert Path.cwd() == workspace


def test_deep_names_an_empty_field_instead_of_dividing_by_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pool of nothing but champions has no vendored field to average."""
    agent = tmp_path / "main.py"
    agent.write_text(PASS, encoding="utf-8")
    monkeypatch.setattr(config, "EXAM_SEEDS", config.EXAM_SEEDS[:1])
    monkeypatch.setattr(evaluator, "HELD_OUT", [])
    monkeypatch.setattr(
        field_gate,
        "score_field",
        lambda candidate, seeds, workers, opponents: (
            dict.fromkeys(opponents, 0.5),
            {n: harness.Margin(mean=0.0, worst=0.0, best=0.0) for n in opponents},
        ),
    )
    p = pool.Pool(opponents={"champion_1": str(tmp_path / "champ.py")})

    with pytest.raises(ValueError, match="no vendored opponent"):
        evaluator.deep(agent, "prog", p, workers=1)


@pytest.mark.local_data
def test_fast_diverts_what_a_candidate_writes_away_from_the_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ranking evaluation executes candidates too, so it is isolated too."""
    import random as random_module

    agent = tmp_path / "main.py"
    agent.write_text(WRITER, encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(config, "FAST_SEEDS", 1)
    p = pool.Pool(opponents={"v54": str(config.OPPONENTS / "kaito_v54" / "main.py")})

    evaluator.fast(agent, "prog", p, random_module.Random(3), workers=WORKERS)

    assert list(workspace.iterdir()) == []
    assert Path.cwd() == workspace
