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
def test_score_plays_the_block_and_averages_the_pool(
    tmp_path: Path,
) -> None:
    """It plays the seasons it is given, and every opponent counts the same.

    The seeds assertion was `len(result.seeds) == config.GATE_SEEDS` until
    2026-09-13, which contradicted both this docstring and the call right above
    it: `score` stopped drawing its own block and started taking one from the
    caller, so a test handing it `[5, 6]` was asserting that two seeds were
    sixteen. It reports back the block it was given.
    """
    agent = tmp_path / "main.py"
    agent.write_text(PASS, encoding="utf-8")
    p = pool.Pool(opponents={"v54": str(config.OPPONENTS / "kaito_v54" / "main.py")})
    result = evaluator.score(
        agent, "prog", p, random.Random(1), [5, 6], workers=WORKERS
    )
    assert sorted(result.seeds) == [5, 6]
    # Both seats of both seasons, which is what makes a rate over them a rate.
    assert result.games == 4
    assert result.rates == {"v54": 0.0} and result.fitness == 0.0


@pytest.mark.local_data
def test_a_call_plays_the_seasons_it_is_handed(tmp_path: Path) -> None:
    """The caller owns the seasons, so two candidates can share a block.

    They used to be drawn per call, which meant no two programs were ever
    ranked on the same seasons -- and a season is most of what the rating
    measures. One unchanged agent through the gate five times, opponents held
    fixed and only the seeds moving, gave fitted ratings from -3.466 to -2.226:
    a standard deviation of 0.491, where varying the opponents instead moved it
    0.070.

    The drift the old draw protected against is now `loop.Campaign.seasons`,
    which rotates the block; it is not this function's to decide.
    """
    agent = tmp_path / "main.py"
    agent.write_text(PASS, encoding="utf-8")
    p = pool.Pool(opponents={"v54": str(config.OPPONENTS / "kaito_v54" / "main.py")})
    block = [7, 8]

    first = evaluator.score(agent, "prog", p, random.Random(2), block, workers=WORKERS)
    again = evaluator.score(agent, "prog", p, random.Random(3), block, workers=WORKERS)

    assert sorted(first.seeds) == sorted(block)
    assert first.seeds == again.seeds


@pytest.mark.local_data
def test_the_champion_pairing_is_played_deeper_than_the_rest_of_the_sweep(
    tmp_path: Path,
) -> None:
    """The one pairing that decides a promotion gets its own seasons.

    Every other opponent is played at the sweep's depth, which is set by what
    a field of mostly-beaten agents needs. The champion decides the gate on
    its own, so it is played on those seasons and the duel block as well --
    and `config.POOL_CHAMPION` is found here rather than passed in, so no
    caller can deepen the wrong pairing.
    """
    agent = tmp_path / "main.py"
    agent.write_text(PASS, encoding="utf-8")
    # The champion second, not first. `Pool.names` is insertion order, so a
    # champion at the head would let "deepen whichever pairing comes first"
    # pass this unnoticed -- which it did, until the two were swapped.
    p = pool.Pool(
        opponents={
            "v56": str(config.OPPONENTS / "kaito_v56" / "main.py"),
            config.POOL_CHAMPION: str(config.OPPONENTS / "kaito_v54" / "main.py"),
        }
    )

    # `ours` is not a roster name, so it only resolves through the pool file.
    p.save(tmp_path / "pool.json")

    result = evaluator.score(
        agent,
        "prog",
        p,
        random.Random(1),
        [5, 6],
        workers=WORKERS,
        pool_file=tmp_path / "pool.json",
        duel_seeds=[7, 8, 9],
    )

    # Two seasons in both seats for an ordinary opponent; five for the
    # champion, because the duel block is played on top of the sweep's.
    assert len(result.states["v56"]) == 4
    assert len(result.states[config.POOL_CHAMPION]) == 10
    # And the sweep's own depth is untouched, because condition 1 reads it as
    # the games behind every field rate.
    assert result.games == 4


@pytest.mark.local_data
def test_without_a_duel_block_the_champion_is_played_at_the_sweep_depth(
    tmp_path: Path,
) -> None:
    """The deepening is the duel block's doing, not the champion's name.

    Mutating the guard in `score` to deepen unconditionally has to turn
    something red, or the test above is only asserting that two numbers differ
    for some reason it never pinned down.
    """
    agent = tmp_path / "main.py"
    agent.write_text(PASS, encoding="utf-8")
    # The champion second, not first. `Pool.names` is insertion order, so a
    # champion at the head would let "deepen whichever pairing comes first"
    # pass this unnoticed -- which it did, until the two were swapped.
    p = pool.Pool(
        opponents={
            "v56": str(config.OPPONENTS / "kaito_v56" / "main.py"),
            config.POOL_CHAMPION: str(config.OPPONENTS / "kaito_v54" / "main.py"),
        }
    )

    p.save(tmp_path / "pool.json")

    result = evaluator.score(
        agent,
        "prog",
        p,
        random.Random(1),
        [5, 6],
        workers=WORKERS,
        pool_file=tmp_path / "pool.json",
    )

    assert len(result.states[config.POOL_CHAMPION]) == 4
    assert len(result.states["v56"]) == 4


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
        champion, "champion_1", p, random.Random(4), [5, 6], WORKERS, registry
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

    evaluator.score(agent, "prog", p, random_module.Random(3), [5, 6], workers=WORKERS)

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

    result = evaluator.score(agent, "prog", p, random.Random(1), [5, 6], WORKERS)

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
        evaluator.score(agent, "prog", p, random.Random(1), [5, 6], WORKERS)

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


@pytest.mark.local_data
def test_every_game_it_scores_is_a_game_the_round_can_improve(tmp_path: Path) -> None:
    """Every game, grouped by opponent, narrowest first.

    This used to keep the single narrowest game against each opponent and drop
    the rest, so a round was asked to improve a program on one game in
    thirty-two of what it was about to be scored on -- and the thirty-one it
    could not see were decided by the same policy for the same reasons. The
    days are already rendered by the time this runs, so keeping them costs the
    write and nothing else.

    Narrowest first because that is the game a small change would have turned;
    the widest shows the failure at its starkest and the least reachable, so it
    goes last rather than first.
    """
    agent = tmp_path / "main.py"
    agent.write_text(PASS, encoding="utf-8")
    p = pool.Pool(opponents={"v54": str(config.OPPONENTS / "kaito_v54" / "main.py")})

    result = evaluator.score(
        agent, "prog", p, random.Random(4), [11, 12], workers=WORKERS
    )

    seasons = result.states["v54"]
    assert len(seasons) == result.games, "one recorded season per game scored"
    assert all(len(game.days) == len(seasons[0].days) for game in seasons)
    assert len(seasons[0].days) > 1, "a season is its days, not one terminal row"
    # Whole games, because the seat is what lets the days be written out the
    # way the corpus writes them, and days alone do not carry it.
    assert {game.seat for game in seasons} == {0, 1}
    assert {game.seed for game in seasons} == {11, 12}
    gaps = [abs(game.ours - game.theirs) for game in seasons]
    assert gaps == sorted(gaps), "narrowest first, so season 1 is the reachable one"


def test_a_result_serialises_without_its_games() -> None:
    """The games are for the record, not for the file the state is kept in.

    A champion's result rode inside `State`, and `state.json` was 688 MB
    rewritten after every session -- twelve thousand games with their day
    tables, every one of them already in the games database. What a result
    carries through the disk is what the gate reads: the rates, the margins
    and the rest. What it loads back with is no games, and a caller that wants
    them asks `games.recorded`.
    """
    seasons = [
        harness.Game(
            opponent="v54",
            seed=11,
            seat=0,
            ours=5.0,
            theirs=3.0,
            worst_step_seconds=0.1,
        )
    ]
    result = evaluator.Result(
        program_id="p",
        fitness=1.0,
        field=None,
        rates={"v54": 1.0},
        margins={"v54": harness.Margin(mean=2.0, worst=2.0, best=2.0)},
        games=1,
        seeds=[11],
        hardest="v54",
        states={"v54": seasons},
    )

    assert result.states == {"v54": seasons}, "in memory, the games are there"
    assert "states" not in result.model_dump()
    back = evaluator.Result.model_validate_json(result.model_dump_json())
    assert back.states == {}
    assert back.rates == result.rates and back.margins == result.margins


def rated(program_id: str, rates: dict[str, float]) -> evaluator.Result:
    """A result carrying nothing but the rates a comparison reads."""
    return evaluator.Result(
        program_id=program_id,
        fitness=sum(rates.values()) / len(rates),
        field=None,
        rates=rates,
        margins={name: harness.Margin(mean=0.0, worst=0.0, best=0.0) for name in rates},
        games=2,
        seeds=[1],
        hardest=next(iter(rates)),
        states={},
    )


def test_a_result_is_compared_over_the_opponents_both_programs_played() -> None:
    """A pool that grew between two evaluations must not decide the comparison.

    Harvest enrolled an opponent at 14:01 on 2026-09-13, mid-session, and a mean
    over a pool that gained an agent is not the same number as a mean over the
    pool before it. Here the child faces one extra opponent it beats outright:
    on `fitness` it looks ahead, and over what both actually played it is behind.
    """
    parent = rated("parent", {"a": 0.40, "b": 0.40})
    child = rated("child", {"a": 0.30, "b": 0.30, "harvested": 1.0})

    assert child.fitness > parent.fitness, "the whole-pool mean favours the child"
    assert not child.beats(parent), "over the shared two it is 0.30 against 0.40"
    assert parent.beats(child)


def test_a_tie_is_not_an_improvement() -> None:
    """Two programs that draw every game are the same program.

    Equality keeps the incumbent, so a round that changed nothing measurable
    does not become what the next round builds on.
    """
    held = rated("parent", {"a": 0.5, "b": 0.5})
    same = rated("child", {"a": 0.5, "b": 0.5})

    assert not same.beats(held)
    assert not held.beats(same)


def test_two_results_with_no_opponent_in_common_are_not_a_comparison() -> None:
    """No shared opponent is no evidence, and no evidence keeps the incumbent."""
    parent = rated("parent", {"a": 0.1})
    child = rated("child", {"b": 0.9})

    assert not child.beats(parent)
    assert not parent.beats(child)
