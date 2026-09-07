"""The promotion rule on hand-built numbers, the tarball, and the floor write."""

import tarfile
from pathlib import Path

import pytest

from kaggriculture.campaign import (
    archive,
    config,
    evaluator,
    gate,
    harness,
    pool,
    rating,
)

PASS = (
    "def agent(observation, configuration=None):\n"
    "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
)


def result(
    pid: str, score: float, rates: dict[str, float], width: float = 0.05
) -> evaluator.Result:
    """A hand-built Result with symmetric per-opponent intervals."""
    return evaluator.Result(
        program_id=pid,
        fitness=score,
        field=score,
        rates=rates,
        margins={n: harness.Margin(mean=0.0, worst=0.0, best=0.0) for n in rates},
        intervals={
            n: (max(0.0, r - width), min(1.0, r + width)) for n, r in rates.items()
        },
        games=64,
        seeds=[1, 2],
        hardest=min(rates, default=""),
        states={},
    )


def test_a_field_measured_over_other_games_is_thrown_away_not_extended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One game count covers the whole cache, so a changed one invalidates it.

    `Field.games` is a single number for every pairing in the file, and
    recording a new pairing sets it. Extend a field measured over one count
    with a pairing played over another and every old rate is relabelled as
    having been played over games it never was -- silently, because nothing
    fails, and consequentially, because the Bradley-Terry fit weights each
    rate by that number. Only changing `FAST_SEEDS` can reach it, which is
    exactly when nobody would be looking.

    The cached rate here is a lie: a PASS agent draws with a PASS agent, so
    the truth is 0.5 and the file claims 1.0. Surviving the tournament is what
    proves the cache was reused.
    """
    monkeypatch.setattr(config, "POOL", tmp_path / "pool.json")
    agents = {}
    for name in ("one", "two"):
        path = tmp_path / f"{name}.py"
        path.write_text(PASS, encoding="utf-8")
        agents[name] = str(path)
    opponents = pool.Pool(opponents=agents)
    opponents.save(config.POOL)

    kept = tmp_path / "field.json"
    stale = rating.Field(games=2)
    stale.record("one", "two", 1.0)
    stale.save(kept)

    gate.tournament(
        Path(agents["one"]), "candidate", opponents, seeds=[1, 2], workers=1, kept=kept
    )

    field = rating.Field.load(kept)
    assert field.games == 4
    assert field.rates["one"]["two"] == 0.5


def test_the_top_of_the_tournament_is_promoted() -> None:
    """The gate is a place, not a clean sweep.

    The finale is a single Bradley-Terry tournament and a leaderboard position
    is a skill rating, so coming out top of the pool is what promotion means.
    """
    ok, why = gate.promotion({"c": 1.2, "a": 0.4, "b": -0.9}, "c")

    assert ok
    assert "top of the tournament" in why and "+1.200" in why
    # Named, so the log says what it had to get past.
    assert "a" in why and "+0.400" in why


def test_second_place_is_not_promoted_however_close() -> None:
    """A floor that rose on anything but the best would not be the best."""
    ok, why = gate.promotion({"a": 0.9001, "c": 0.9000, "b": -0.5}, "c")

    assert not ok
    assert "2 of 3" in why and "below a" in why


def test_the_reason_says_where_it_placed_not_who_it_failed_to_beat() -> None:
    """A place is something a next round can aim at; a list of losses is not.

    The absolute rule this replaced named every opponent a program did not
    beat, which for a program that beat none of them was the whole pool and
    told it nothing about which to attack first.
    """
    ok, why = gate.promotion({"a": 2.0, "b": 1.0, "c": 0.0, "mine": -1.0}, "mine")

    assert not ok
    assert why.startswith("4 of 4")
    assert "below a" in why


def test_one_bad_matchup_does_not_sink_a_top_rating() -> None:
    """Where the tournament and the old absolute rule actually disagree.

    Measured on the pool of 2026-09-06, the second-strongest published agent
    wins 78.6% of everything and loses one matchup at 0.062. The rule this
    replaced turned that agent away; a tournament ranks it where it belongs.
    """
    # `mine` is rated top despite the standings being the only thing the gate
    # reads -- the rate that would have failed an absolute rule is inside the
    # fit, not beside it.
    assert gate.promotion({"mine": 1.5, "a": 1.0, "b": 0.2}, "mine")[0]


def test_the_standings_are_over_the_pool_and_nothing_else() -> None:
    """The gate is a place in one tournament, not a score against a set.

    There is no held-out set any more and no second measurement: seeds
    are drawn fresh every evaluation, so every rate here was already
    measured on maps this program was never selected on.
    """
    candidate = result("c", 0.8, {"a": 0.9, "b": 0.8})

    assert gate.promotion({"c": 1.0, "a": 0.1, "b": -0.4}, "c")[0]
    assert set(candidate.rates) == {"a", "b"}


def _program(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str = "PASS"
) -> archive.Program:
    """Patch every path a promotion writes into tmp_path and build a candidate.

    The live campaign's own ``run/campaign`` and pool file must never be a
    destination of the suite, so all five are redirected before a promotion
    is attempted.
    """
    monkeypatch.setattr(config, "FLOOR", tmp_path / "floor" / "agent")
    monkeypatch.setattr(config, "CHAMPIONS", tmp_path / "champions")
    monkeypatch.setattr(config, "CHAMPION", tmp_path / "champion.json")
    monkeypatch.setattr(config, "POOL", tmp_path / "pool.json")
    # Not a destination either, and redirected so that a promotion which
    # wrongly wrote there would write here instead of into the checkout.
    monkeypatch.setattr(config, "SERVED", tmp_path / "src" / "served" / "main.py")
    source = tmp_path / f"prog-{body}.py"
    source.write_text(
        f"def agent(o, c=None):\n"
        f"    return {{'farmer': ['{body}'], 'hands': [], 'market': []}}\n"
    )
    return archive.Program(
        id="p9",
        source_path=str(source),
        started_from="",
        instruction="improve it",
        model="gpt-5.6-luna",
        fitness=0.5,
        field=0.5,
        rates={"a": 0.5},
        created=0.0,
    )


def _result() -> evaluator.Result:
    """The deep result the promotion tests promote on."""
    return result("p9", 0.7, {"a": 0.9, "b": 0.5})


def test_promote_leaves_a_tarball_the_champion_record_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cut is uploading this file; nothing is built at cut time."""
    program = _program(tmp_path, monkeypatch)
    p = pool.Pool(opponents={"a": "/x/a.py"})

    champion = gate.record(gate.promote(program, _result()))
    gate.enroll(champion, p)

    tarball = Path(champion.tarball)
    assert tarball.exists() and tarball.parent == config.CHAMPIONS
    with tarfile.open(tarball) as tar:
        names = tar.getnames()
    assert sorted(names) == ["LICENSE", "main.py"]


def test_gate_runs_no_version_control() -> None:
    """The loop never touches git; provenance is champion.json."""
    assert not hasattr(gate, "commit_floor")
    assert "subprocess" not in gate.__dict__


def test_promote_writes_a_read_only_floor_and_updates_the_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The floor is read-only, the champion keeps its own copy, the pool is saved.

    The floor is what ships and what a later promotion overwrites; the
    champion's own copy is what the pool plays, and it is never written twice.
    """
    program = _program(tmp_path, monkeypatch)
    source = Path(program.source_path)
    p = pool.Pool(opponents={"a": "/x/a.py", "b": "/x/b.py"})
    champion = gate.record(gate.promote(program, _result()))
    gate.enroll(champion, p)
    assert champion.name == "champion_1"
    floor = config.FLOOR / "main.py"
    assert (
        floor.read_text() == source.read_text()
        and (floor.stat().st_mode & 0o777) == 0o444
    )
    kept = config.CHAMPIONS / "champion_1.py"
    assert (
        kept.read_text() == source.read_text()
        and (kept.stat().st_mode & 0o777) == 0o444
    )
    saved = pool.Pool.load(config.POOL)
    assert "champion_1" in saved.names()


def test_promote_refuses_a_champion_name_the_directory_already_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A champion file nothing wrote is a disagreement, not something to overwrite."""
    program = _program(tmp_path, monkeypatch)
    config.CHAMPIONS.mkdir(parents=True)
    stale = config.CHAMPIONS / "champion_1.py"
    stale.write_text("# an older champion\n", encoding="utf-8")
    stale.chmod(0o444)
    third = config.CHAMPIONS / "champion_3.py"
    third.write_text("# and a third\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="champion_3.py already exists"):
        gate.promote(program, _result())
    assert stale.read_text() == "# an older champion\n"
    assert not (config.FLOOR / "main.py").exists()


def test_the_pool_registers_each_champion_own_file_not_the_shared_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two promotions leave two different programs in the pool, not two aliases.

    The floor is one path that every promotion overwrites. Registering it
    would make `champion_1` and `champion_2` both resolve to whatever was
    promoted last, so the pool would hold N copies of the newest agent and
    every earlier champion would be gone.
    """
    first = _program(tmp_path, monkeypatch, body="PASS")
    p = pool.Pool(opponents={"a": "/x/a.py", "b": "/x/b.py"})
    gate.enroll(gate.promote(first, _result()), p)

    second = _program(tmp_path, monkeypatch, body="WATER")
    reloaded = pool.Pool.load(config.POOL)
    gate.enroll(gate.promote(second, result("p9", 0.8, {"a": 0.9, "b": 0.6})), reloaded)

    saved = pool.Pool.load(config.POOL)
    one, two = saved.opponents["champion_1"], saved.opponents["champion_2"]
    assert one != two
    assert Path(one).read_text() == Path(first.source_path).read_text()
    assert Path(two).read_text() == Path(second.source_path).read_text()
    floor = config.FLOOR / "main.py"
    assert floor.read_text() == Path(second.source_path).read_text()


def test_promote_writes_the_champion_record_before_it_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`champion.json` is what a restart trusts, so `record` writes it whole."""
    program = _program(tmp_path, monkeypatch)
    deep = _result()

    champion = gate.record(gate.promote(program, deep))

    assert gate.load_champion() == champion
    assert champion.result == deep
    assert Path(champion.path).read_text() == Path(program.source_path).read_text()


def test_champion_numbering_survives_a_pool_retirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retired champion must not free its number for the next promotion.

    `Pool.add_champion` retires the lowest-weight opponent the new champion
    crushes once the pool is at `POOL_CAP`, so the pool's `champion_` count
    can fall. Numbering off that count hands the next promotion a name whose
    file already exists, and the `FileExistsError` that follows escapes into
    the campaign's task group and stops the run.
    """
    program = _program(tmp_path, monkeypatch)
    config.CHAMPIONS.mkdir(parents=True)
    for number in (1, 2, 3):
        (config.CHAMPIONS / f"champion_{number}.py").write_text("# past\n")
    # What a retirement leaves behind: three champions promoted, one of them
    # still in the pool.
    retired = pool.Pool(opponents={"a": "/x/a.py", "champion_3": "/x/c3.py"})
    retired.save(config.POOL)

    champion = gate.promote(program, result("p9", 0.9, {"a": 0.99, "champion_3": 0.99}))
    gate.enroll(gate.record(champion), retired)

    assert champion.name == "champion_4"
    assert sorted(p.name for p in config.CHAMPIONS.glob("champion_*.py")) == [
        "champion_1.py",
        "champion_2.py",
        "champion_3.py",
        "champion_4.py",
    ]


def test_a_promotion_writes_nothing_outside_the_run_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing under `src/` is a promotion's business.

    `served/main.py` is the committed seed a cold start begins from, not the
    floor a campaign produces. A promotion that wrote there would dirty a
    tracked file, and `loop._open_run` refuses to start a run whose `src/`
    has uncommitted changes -- so the first promotion would have been the
    last thing that campaign ever did.
    """
    program = _program(tmp_path, monkeypatch)
    p = pool.Pool(opponents={"a": "/x/a.py", "b": "/x/b.py"})

    gate.enroll(gate.record(gate.promote(program, _result())), p)

    assert not config.SERVED.exists()
    assert not (tmp_path / "src").exists()
    assert "SERVED" not in gate.__dict__


def test_a_second_promotion_on_the_saved_pool_yields_champion_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The champion counter reads the pool that was just saved, not a stale one."""
    program = _program(tmp_path, monkeypatch)
    p = pool.Pool(opponents={"a": "/x/a.py", "b": "/x/b.py"})
    gate.enroll(gate.record(gate.promote(program, _result())), p)
    reloaded = pool.Pool.load(config.POOL)
    champion = gate.promote(program, result("p9", 0.8, {"a": 0.9, "b": 0.6}))
    gate.enroll(gate.record(champion), reloaded)
    assert champion.name == "champion_2"
