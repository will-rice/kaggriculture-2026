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


def test_a_field_measured_at_another_depth_is_kept_and_extended(
    tmp_path: Path,
) -> None:
    """The count is kept per pairing, so changing the gate's depth costs nothing.

    It used to be one number for the whole field, on the reasoning that every
    pairing is played on the same seeds. That could not be extended: measuring
    anything at a new depth relabelled every cached rate as having been played
    over games it was not, and the only safe response was to discard the lot.

    Discarding the lot now means discarding the campaign's history -- two
    hundred and eighty pairings -- and the rating everything is steered by is
    fitted over exactly that history.
    """
    paths = _paths(tmp_path)
    agents = {}
    for name in ("one", "two", "three"):
        path = tmp_path / f"{name}.py"
        path.write_text(PASS, encoding="utf-8")
        agents[name] = str(path)
    pool.Pool(opponents=agents).save(paths.pool)

    older = rating.Field(games=64)
    older.record("one", "two", 1.0, 64)
    older.save(paths.field)

    gate.refresh("three", ["one", "two"], seeds=[1], workers=1, paths=paths)

    field = rating.Field.load(paths.field)
    # The old pairing survives at its own depth; the new ones carry theirs.
    assert field.rates["one"]["two"] == 1.0
    assert field.depth("one", "two") == 64
    assert field.depth("three", "one") == 2


def test_a_new_champion_has_its_own_edges_played(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A champion joins with no pairings, and `standing` plays nothing.

    `Field` returns only the pairings it holds, so an agent with none is
    absent from the fit rather than an error. For an opponent that has been
    around that never matters. For a champion it is the whole ratchet: until
    its own pairings exist it sits in every fit on the single edge of whoever
    is being judged against it, and beating it drops its rating far enough to
    make topping the field easy. Each promotion would buy the next cheaply.
    """
    paths = _paths(tmp_path)
    agents = {}
    for name in ("one", "two", "champ"):
        path = tmp_path / f"{name}.py"
        path.write_text(PASS, encoding="utf-8")
        agents[name] = str(path)
    pool.Pool(opponents=agents).save(paths.pool)

    measured = gate.refresh("champ", ["one", "two"], seeds=[1], workers=1, paths=paths)

    assert measured == [("champ", "one"), ("champ", "two")]
    # Its own edges and nobody else's: what the pool's other members owe each
    # other is not a promotion's business, and with nothing ever leaving the
    # pool "every missing pair" grows with its square.
    field = rating.Field.load(paths.field)
    assert set(field.rates["champ"]) == {"one", "two"}
    assert "two" not in field.rates.get("one", {})


def test_edges_already_on_the_record_are_not_played_again(
    tmp_path: Path,
) -> None:
    """A pairing is a constant: fixed files on fixed seeds, measured once."""
    paths = _paths(tmp_path)
    agents = {}
    for name in ("one", "champ"):
        path = tmp_path / f"{name}.py"
        path.write_text(PASS, encoding="utf-8")
        agents[name] = str(path)
    pool.Pool(opponents=agents).save(paths.pool)

    gate.refresh("champ", ["one"], seeds=[1], workers=1, paths=paths)
    again = gate.refresh("champ", ["one"], seeds=[1], workers=1, paths=paths)

    assert again == []


def beat(rate: float, decided: int, champion: str = "floor") -> evaluator.Result:
    """An evaluation in which the candidate took ``rate`` of ``decided`` games.

    Only the three fields the bar reads are meaningful: who was played, the rate
    against the champion, and how many of those games ended with a winner.
    """
    return evaluator.Result(
        program_id="mine",
        fitness=rate,
        field=rate,
        rates={champion: rate, "other": 0.5},
        margins={
            name: harness.Margin(mean=0.0, worst=0.0, best=0.0)
            for name in (champion, "other")
        },
        decisive={champion: decided, "other": 32},
        seeds=[1, 2, 3, 4],
        hardest=champion,
        states={},
    )


def test_a_decisive_head_to_head_win_promotes() -> None:
    """The bar is beating the champion, measured in the games against it.

    Directly measured and paired: the champion is a pool opponent, so the two
    programs are in the same games on the same seeds in both seats, and the seat
    swap cancels position. No rating, no tournament, no stored pairings.

    22 of 32 is 0.688, which is where the Wilson lower bound crosses 0.5.
    """
    clear, why = gate.promotion(beat(22 / 32, 32), "floor")

    assert clear, why
    assert "beat floor" in why and "lower bound" in why


def test_a_narrow_head_to_head_win_is_not_shown_to_be_better() -> None:
    """Above half is not the bar; above half beyond the interval is.

    21 of 32 is 0.656 and a real edge on the face of it, and the lower bound is
    0.483. The interval is what a fixed margin could not be: it tightens when
    the measurement is good and refuses when it is not.
    """
    close, why = gate.promotion(beat(21 / 32, 32), "floor")

    assert not close
    assert "not shown to be better" in why


def test_a_candidate_that_mostly_draws_is_not_promoted() -> None:
    """Two programs that draw are the same program, whatever the rate says.

    champion_55 was promoted over champion_54 on two wins and thirty exact draws
    in 32 games. A rate cannot tell that from seventeen wins and fifteen losses,
    and 2/2 reads as 1.000.

    `DECISIVE_GAMES` is what covers the band the interval does not: 4 to 7
    decided out of 32 is a candidate drawing 78% to 88% of its games with the
    champion. Below 4 the interval refuses on its own, since 3/3 is 0.438.
    """
    drawn, why = gate.promotion(beat(1.0, 2), "floor")

    assert not drawn
    assert "only 2 of its games were decided" in why
    assert "the same game" in why

    # Seven decided is a perfect record and still refused, because seven of
    # thirty-two decided is two programs playing the same season.
    thin, why = gate.promotion(beat(1.0, 7), "floor")
    assert not thin and "the bar is 8" in why


def test_an_unplayed_champion_is_not_a_measurement() -> None:
    """A champion that moved under a running evaluation was never played.

    Eight sessions run at once, so a promotion by one lands while the others are
    mid-flight. Their games are against the champion it replaced, and a bar read
    off those would replace champion N with something that beat champion N-1.
    """
    missing, why = gate.promotion(beat(1.0, 32, champion="someone_else"), "floor")

    assert not missing
    assert "did not play floor" in why


def _paths(tmp_path: Path) -> config.Run:
    """The run a test promotes into: its own directory, its own pool."""
    return config.Run(root=tmp_path, pool=tmp_path / "pool.json")


def _program(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str = "PASS"
) -> archive.Program:
    """A run of its own under tmp_path, and a candidate to promote into it.

    The live campaign's ``run/campaign`` and pool file must never be a
    destination of the suite. They cannot be: a promotion writes into the run
    it is handed, and this hands it one made here.
    """
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
    paths = _paths(tmp_path)
    program = _program(tmp_path, monkeypatch)
    p = pool.Pool(opponents={"a": "/x/a.py"})

    champion = gate.record(
        paths=paths,
        champion=gate.promote(paths=paths, program=program, result=_result()),
    )
    gate.enroll(champion, p, paths)

    tarball = Path(champion.tarball)
    assert tarball.exists() and tarball.parent == paths.champions
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
    paths = _paths(tmp_path)
    program = _program(tmp_path, monkeypatch)
    source = Path(program.source_path)
    p = pool.Pool(opponents={"a": "/x/a.py", "b": "/x/b.py"})
    champion = gate.record(
        paths=paths,
        champion=gate.promote(paths=paths, program=program, result=_result()),
    )
    gate.enroll(champion, p, paths)
    # The pool key is fixed -- there is one champion -- and the numbering is on
    # the file, which is what keeps a history without colliding with the tape
    # lineage's `champion_1` already in the pool.
    assert champion.name == config.POOL_CHAMPION
    assert Path(champion.path).name == "champion_1.py"
    floor = paths.floor / "main.py"
    assert (
        floor.read_text() == source.read_text()
        and (floor.stat().st_mode & 0o777) == 0o444
    )
    kept = paths.champions / "champion_1.py"
    assert (
        kept.read_text() == source.read_text()
        and (kept.stat().st_mode & 0o777) == 0o444
    )
    saved = pool.Pool.load(paths.pool)
    assert config.POOL_CHAMPION in saved.names()


def test_promote_refuses_a_champion_name_the_directory_already_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A champion file nothing wrote is a disagreement, not something to overwrite."""
    paths = _paths(tmp_path)
    program = _program(tmp_path, monkeypatch)
    paths.champions.mkdir(parents=True)
    stale = paths.champions / "champion_1.py"
    stale.write_text("# an older champion\n", encoding="utf-8")
    stale.chmod(0o444)
    third = paths.champions / "champion_3.py"
    third.write_text("# and a third\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="champion_3.py already exists"):
        gate.promote(paths=paths, program=program, result=_result())
    assert stale.read_text() == "# an older champion\n"
    assert not (paths.floor / "main.py").exists()


def test_the_pool_registers_each_champion_own_file_not_the_shared_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two promotions leave two different programs in the pool, not two aliases.

    The floor is one path that every promotion overwrites. Registering it
    would make `champion_1` and `champion_2` both resolve to whatever was
    promoted last, so the pool would hold N copies of the newest agent and
    every earlier champion would be gone.
    """
    paths = _paths(tmp_path)
    first = _program(tmp_path, monkeypatch, body="PASS")
    p = pool.Pool(opponents={"a": "/x/a.py", "b": "/x/b.py"})
    gate.enroll(gate.promote(paths=paths, program=first, result=_result()), p, paths)

    second = _program(tmp_path, monkeypatch, body="WATER")
    reloaded = pool.Pool.load(paths.pool)
    gate.enroll(
        gate.promote(
            paths=paths, program=second, result=result("p9", 0.8, {"a": 0.9, "b": 0.6})
        ),
        reloaded,
        paths,
    )

    saved = pool.Pool.load(paths.pool)
    # One key, so the second promotion replaces the first in the pool. Both files
    # are still on disk under their own numbers, which is where the history is.
    held = saved.opponents[config.POOL_CHAMPION]
    assert Path(held).read_text() == Path(second.source_path).read_text()
    assert Path(held).name == "champion_2.py"
    assert (paths.champions / "champion_1.py").read_text() == Path(
        first.source_path
    ).read_text()
    floor = paths.floor / "main.py"
    assert floor.read_text() == Path(second.source_path).read_text()


def test_promote_writes_the_champion_record_before_it_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`champion.json` is what a restart trusts, so `record` writes it whole."""
    paths = _paths(tmp_path)
    program = _program(tmp_path, monkeypatch)
    deep = _result()

    champion = gate.record(
        paths=paths, champion=gate.promote(paths=paths, program=program, result=deep)
    )

    assert gate.load_champion(paths) == champion
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
    paths = _paths(tmp_path)
    program = _program(tmp_path, monkeypatch)
    paths.champions.mkdir(parents=True)
    for number in (1, 2, 3):
        (paths.champions / f"champion_{number}.py").write_text("# past\n")
    # What a retirement leaves behind: three champions promoted, one of them
    # still in the pool.
    retired = pool.Pool(opponents={"a": "/x/a.py", "champion_3": "/x/c3.py"})
    retired.save(paths.pool)

    champion = gate.promote(
        paths=paths,
        program=program,
        result=result("p9", 0.9, {"a": 0.99, "champion_3": 0.99}),
    )
    gate.enroll(gate.record(champion, paths), retired, paths)

    assert Path(champion.path).name == "champion_4.py"
    assert sorted(p.name for p in paths.champions.glob("champion_*.py")) == [
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
    paths = _paths(tmp_path)
    program = _program(tmp_path, monkeypatch)
    p = pool.Pool(opponents={"a": "/x/a.py", "b": "/x/b.py"})

    gate.enroll(
        gate.record(
            paths=paths,
            champion=gate.promote(paths=paths, program=program, result=_result()),
        ),
        p,
        paths,
    )

    assert not config.SERVED.exists()
    assert not (tmp_path / "src").exists()
    assert "SERVED" not in gate.__dict__


def test_a_second_promotion_on_the_saved_pool_yields_champion_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The champion counter reads the pool that was just saved, not a stale one."""
    paths = _paths(tmp_path)
    program = _program(tmp_path, monkeypatch)
    p = pool.Pool(opponents={"a": "/x/a.py", "b": "/x/b.py"})
    gate.enroll(
        gate.record(
            paths=paths,
            champion=gate.promote(paths=paths, program=program, result=_result()),
        ),
        p,
        paths,
    )
    reloaded = pool.Pool.load(paths.pool)
    champion = gate.promote(
        paths=paths, program=program, result=result("p9", 0.8, {"a": 0.9, "b": 0.6})
    )
    gate.enroll(gate.record(champion, paths), reloaded, paths)
    assert Path(champion.path).name == "champion_2.py"


def test_the_decisive_bar_counts_games_not_the_rate() -> None:
    """The denominator is games that ended with a winner, not games played.

    champion_55's numbers, measured 2026-09-08: promoted over champion_54 on
    thirty-two games that were two wins by five units and thirty exact draws. A
    rate of 1.000 and a rate of 0.531 describe that same record depending on
    whether the draws are counted, and only one of them says "the same program".

    So a perfect record over two decided games is refused, and a record over
    thirty-two is read on its merits. The bar is about how much was decided.
    """
    perfect, why = gate.promotion(beat(1.0, 2), "floor")
    assert not perfect and "only 2 of its games were decided" in why

    read, why = gate.promotion(beat(24 / 32, 32), "floor")
    assert read, why
