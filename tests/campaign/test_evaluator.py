"""The one evaluation, on fresh seeds, through the port."""

import random
from pathlib import Path

import pytest

from kaggriculture.campaign import config, evaluator, harness, pool

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
def test_score_draws_fresh_seeds_and_averages_the_pool(
    tmp_path: Path,
) -> None:
    """Seeds are drawn per call, and every opponent counts the same."""
    agent = tmp_path / "main.py"
    agent.write_text(PASS, encoding="utf-8")
    p = pool.Pool(opponents={"v54": str(config.OPPONENTS / "kaito_v54" / "main.py")})
    result = evaluator.score(agent, "prog", p, random.Random(1), workers=WORKERS)
    assert len(result.seeds) == config.GATE_SEEDS
    assert result.rates == {"v54": 0.0} and result.fitness == 0.0


@pytest.mark.local_data
def test_two_calls_draw_different_seeds(tmp_path: Path) -> None:
    """Ranking must not reuse one seed block, or drift becomes overfitting."""
    agent = tmp_path / "main.py"
    agent.write_text(PASS, encoding="utf-8")
    p = pool.Pool(opponents={"v54": str(config.OPPONENTS / "kaito_v54" / "main.py")})
    rng = random.Random(2)
    assert (
        evaluator.score(agent, "prog", p, rng, workers=WORKERS).seeds
        != evaluator.score(agent, "prog", p, rng, workers=WORKERS).seeds
    )


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
    monkeypatch.setattr(config, "GATE_SEEDS", 1)
    monkeypatch.setattr(evaluator, "VENDORED", ["other"])
    # Names resolve to paths through the saved pool, so it has to be on disk
    # and the evaluation has to be told where it is -- there is no module-level
    # pool to fall back on, which is the point.
    registry = tmp_path / "pool.json"
    p = pool.Pool(opponents={"other": str(other), "champion_1": str(champion)})
    p.save(registry)

    ranked = evaluator.score(
        champion, "champion_1", p, random.Random(4), WORKERS, registry
    )

    assert set(ranked.rates) == {"other"} and set(ranked.intervals) == {"other"}
    # One opponent left, so the mean over them is that opponent's rate.
    assert ranked.fitness == ranked.rates["other"]


def test_a_pool_name_against_a_different_file_is_an_error(tmp_path: Path) -> None:
    """Two programs cannot both be `champion_1`; nothing here can pick one."""
    champion = tmp_path / "champion_1.py"
    champion.write_text(PASS, encoding="utf-8")
    p = pool.Pool(opponents={"champion_1": str(tmp_path / "elsewhere.py")})

    with pytest.raises(ValueError, match="elsewhere.py"):
        evaluator.opponents(p, "champion_1", champion)


@pytest.mark.local_data
def test_score_diverts_what_a_candidate_writes_away_from_the_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ranking evaluation executes candidates too, so it is isolated too."""
    import random as random_module

    agent = tmp_path / "main.py"
    agent.write_text(WRITER, encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(config, "GATE_SEEDS", 1)
    p = pool.Pool(opponents={"v54": str(config.OPPONENTS / "kaito_v54" / "main.py")})

    evaluator.score(agent, "prog", p, random_module.Random(3), workers=WORKERS)

    assert list(workspace.iterdir()) == []
    assert Path.cwd() == workspace


@pytest.mark.local_data
def test_score_reports_an_interval_beside_every_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rate is read with the width of the thing behind it.

    The gate promotes on 64 games an opponent, which is a real interval and
    not a rounding of one, so the interval travels with the rate rather than
    being recomputed by whoever reads it.
    """
    agent = tmp_path / "main.py"
    agent.write_text(PASS, encoding="utf-8")
    monkeypatch.setattr(config, "GATE_SEEDS", 2)
    p = pool.Pool(opponents={"v54": str(config.OPPONENTS / "kaito_v54" / "main.py")})

    result = evaluator.score(agent, "prog", p, random.Random(1), WORKERS)

    assert result.program_id == "prog" and result.games == 4
    assert result.rates["v54"] == 0.0
    assert result.intervals["v54"][0] == 0.0 and result.intervals["v54"][1] < 1.0


@pytest.mark.local_data
def test_score_lets_a_crash_propagate_and_still_restores_the_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crashed candidate is a failed evaluation, never a zero score."""
    agent = tmp_path / "main.py"
    agent.write_text(CRASHER, encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(config, "GATE_SEEDS", 1)
    p = pool.Pool(opponents={"v54": str(config.OPPONENTS / "kaito_v54" / "main.py")})

    with pytest.raises(RuntimeError):
        evaluator.score(agent, "prog", p, random.Random(1), WORKERS)

    assert Path.cwd() == workspace


def test_a_pool_of_champions_has_no_field_and_that_is_not_fatal() -> None:
    """A pool of nothing but champions has no vendored field to average.

    That is where this campaign is going: champions join on every promotion
    and the weakest opponent makes way, so the published agents leave one at a
    time and the last leaves for good. It used to raise there -- and
    `ValueError` is not the `RuntimeError` a round catches, so it would have
    gone up through the task group and stopped the campaign at its most
    successful moment. Measured live: the pool reached seven champions and one
    published agent, tied last, in two hours.

    What is lost is a number, not the gate.
    """
    assert evaluator.vendored_field({"champion_1": 0.5, "champion_2": 0.9}) is None
    # And it is still the mean while any of them remain.
    assert evaluator.vendored_field({"champion_1": 0.5, "v54": 0.8}) == 0.8


def played(opponent: str, banks: list[tuple[float, float]]) -> list[harness.Game]:
    """Games against one opponent, as (ours, theirs) final banks."""
    return [
        harness.Game(
            opponent=opponent,
            seed=seed,
            seat=seed % 2,
            ours=ours,
            theirs=t,
            worst_step_seconds=0.0,
        )
        for seed, (ours, t) in enumerate(banks)
    ]


def test_a_draw_is_not_half_a_win_when_the_question_is_whether_anything_differs() -> (
    None
):
    """The two shapes that share a rate of 0.53125, told apart.

    Champion_55 against champion_54, measured 2026-09-08: two wins by five
    units and thirty exact draws. It scores the same as seventeen wins and
    fifteen losses, and only the second is two agents trading games. The rate
    cannot separate them, so the count of games that ended with a winner has
    to.
    """
    twin = played("t", [(5.0, 0.0)] * 2 + [(7.0, 7.0)] * 30)
    rival = played("r", [(5.0, 0.0)] * 17 + [(0.0, 5.0)] * 15)

    assert evaluator._rates(twin, ["t"])["t"] == evaluator._rates(rival, ["r"])["r"]

    assert evaluator._contested(twin, ["t"]) == {"t": (2, 2)}
    assert evaluator._contested(rival, ["r"]) == {"r": (17, 32)}


def test_a_pairing_that_drew_every_game_has_no_evidence_either_way() -> None:
    """Not a narrow interval around a half -- the whole unit interval.

    Thirty-two draws say the two programs played the same game, which is no
    evidence about which is stronger. The width has to say so, or the guard
    built on it reports certainty exactly where there is none.
    """
    assert evaluator._contested(played("t", [(7.0, 7.0)] * 32), ["t"]) == {"t": (0, 0)}
    assert evaluator.wilson_interval(0, 0) == (0.0, 1.0)
