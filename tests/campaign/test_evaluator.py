"""Fast eval on fresh seeds; deep eval on the exam block. Both on the port."""

import random
from pathlib import Path

import pytest

from kaggriculture.campaign import config, evaluator, pool

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
    result = evaluator.fast(agent, p, random.Random(1), workers=4)
    assert len(result.seeds) == config.FAST_SEEDS and not set(result.seeds) & set(
        config.EXAM_SEEDS
    )
    assert result.rates == {"v54": 0.0} and result.fitness == 0.0


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
        evaluator.fast(agent, p, rng, workers=4).seeds
        != evaluator.fast(agent, p, rng, workers=4).seeds
    )


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
    result = evaluator.deep(agent, "prog", p, workers=4)
    assert result.program_id == "prog" and result.score == 0.0 and result.games == 4
    assert result.intervals["v54"][0] == 0.0 and result.intervals["v54"][1] < 1.0
    assert set(result.held_out) == set(evaluator.HELD_OUT)
    assert set(result.rates) == set(p.names())  # held-out never enters the score


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
    evaluator.deep(agent, "prog", p, workers=4)
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
        evaluator.deep(agent, "prog", p, workers=4)
    assert Path.cwd() == workspace
