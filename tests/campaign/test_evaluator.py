"""Fast eval on fresh seeds; deep eval on the exam block. Both on the port."""

import random
from pathlib import Path

import pytest

from kaggriculture.campaign import config, evaluator, field_gate, pool
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
def test_fast_draws_fresh_non_exam_seeds_and_weights_by_the_pool(
    tmp_path: Path,
) -> None:
    """Fast draws its own seeds from outside the exam block and weights by the pool."""
    agent = tmp_path / "main.py"
    agent.write_text(PASS, encoding="utf-8")
    p = pool.Pool(
        opponents={"v54": str(config.OPPONENTS / "kaito_v54" / "main.py")},
        weights={"v54": 1.0},
    )
    result = evaluator.fast(agent, p, random.Random(1), workers=WORKERS)
    assert len(result.seeds) == config.FAST_SEEDS and not set(result.seeds) & set(
        config.EXAM_SEEDS
    )
    assert result.rates == {"v54": 0.0} and result.fitness == 0.0


@pytest.mark.local_data
def test_two_fast_calls_draw_different_seeds(tmp_path: Path) -> None:
    """Ranking must not reuse one seed block, or drift becomes overfitting."""
    agent = tmp_path / "main.py"
    agent.write_text(PASS, encoding="utf-8")
    p = pool.Pool(
        opponents={"v54": str(config.OPPONENTS / "kaito_v54" / "main.py")},
        weights={"v54": 1.0},
    )
    rng = random.Random(2)
    assert (
        evaluator.fast(agent, p, rng, workers=WORKERS).seeds
        != evaluator.fast(agent, p, rng, workers=WORKERS).seeds
    )


@pytest.mark.local_data
def test_deep_scores_the_exam_block_with_intervals_and_held_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate reports every pool opponent and every held-out opponent."""
    agent = tmp_path / "main.py"
    agent.write_text(PASS, encoding="utf-8")
    monkeypatch.setattr(config, "EXAM_SEEDS", config.EXAM_SEEDS[:2])
    p = pool.Pool(
        opponents={"v54": str(config.OPPONENTS / "kaito_v54" / "main.py")},
        weights={"v54": 1.0},
    )
    result = evaluator.deep(agent, "prog", p, workers=WORKERS)
    assert result.program_id == "prog" and result.score == 0.0 and result.games == 4
    assert result.intervals["v54"][0] == 0.0 and result.intervals["v54"][1] < 1.0
    assert set(result.held_out) == set(evaluator.HELD_OUT)
    assert set(result.rates) == set(p.names())  # held-out never enters the score


def test_deep_bounds_are_the_pool_weighted_sum_of_the_per_opponent_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-uniform pool is not one binomial, so the bounds combine per opponent.

    Treating the weighted score as a single binomial over every game would
    report an interval narrower than the truth, and the gate promotes on the
    lower bound. The rates are fixed here so the arithmetic, not the engine,
    is what is under test.
    """
    agent = tmp_path / "main.py"
    agent.write_text(PASS, encoding="utf-8")
    monkeypatch.setattr(config, "EXAM_SEEDS", config.EXAM_SEEDS[:8])
    fixed = {"v54": 0.9, "v56": 0.1} | dict.fromkeys(evaluator.HELD_OUT, 0.5)
    monkeypatch.setattr(
        field_gate,
        "score_field",
        lambda candidate, seeds, workers, opponents: {n: fixed[n] for n in opponents},
    )
    p = pool.Pool(
        opponents={
            "v54": str(config.OPPONENTS / "kaito_v54" / "main.py"),
            "v56": str(config.OPPONENTS / "kaito_v56" / "main.py"),
        },
        weights={"v54": 0.8, "v56": 0.2},
    )
    result = evaluator.deep(agent, "prog", p, workers=WORKERS)
    games = 2 * 8
    strong = wilson_interval(0.9 * games, games)
    weak = wilson_interval(0.1 * games, games)
    assert result.games == games
    assert result.low == pytest.approx(0.8 * strong[0] + 0.2 * weak[0])
    assert result.high == pytest.approx(0.8 * strong[1] + 0.2 * weak[1])
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
    p = pool.Pool(
        opponents={"v54": str(config.OPPONENTS / "kaito_v54" / "main.py")},
        weights={"v54": 1.0},
    )
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
    p = pool.Pool(
        opponents={"v54": str(config.OPPONENTS / "kaito_v54" / "main.py")},
        weights={"v54": 1.0},
    )
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
        lambda candidate, seeds, workers, opponents: dict.fromkeys(opponents, 0.5),
    )
    p = pool.Pool(
        opponents={"champion_1": str(tmp_path / "champ.py")},
        weights={"champion_1": 1.0},
    )

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
    p = pool.Pool(
        opponents={"v54": str(config.OPPONENTS / "kaito_v54" / "main.py")},
        weights={"v54": 1.0},
    )

    evaluator.fast(agent, p, random_module.Random(3), workers=WORKERS)

    assert list(workspace.iterdir()) == []
    assert Path.cwd() == workspace
