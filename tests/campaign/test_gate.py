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


def test_a_candidate_must_out_rate_the_floor_by_the_margin() -> None:
    """Beating the champion has to be shown, not merely scored higher.

    The bar was a rating gap of a fixed 0.15, there to stop the winner's
    curse: selecting the maximum of a noisy estimator is biased upward by
    construction. It could not do that. One unchanged agent's fitted rating
    moves with a standard deviation of 0.745 across draws, measured
    2026-09-11, so a 0.15 bar sat a fifth of a standard deviation out and
    filtered almost nothing -- while reliably blocking real improvements too
    small to clear a constant nobody had checked against the noise.

    A fixed size is the wrong shape of answer to a quantity that varies. The
    gate already plays the candidate against the champion over every gate seed
    in both seats, which is the same seasons twice and so a paired comparison,
    and twice its standard error is a bar that tightens when the measurement
    is good and refuses when it is not.
    """
    standings = {"mine": 1.0, "floor": 0.999, "other": 0.0}

    # Barely ahead on rating, but the margin over the floor is eight times its
    # own error: shown, and promoted. The old rule refused this outright.
    clear, why = gate.promotion(
        standings,
        "mine",
        "floor",
        decisive=config.DECISIVE_GAMES,
        over_champion=harness.Margin(
            mean=5968.0, worst=-400.0, best=9000.0, error=722.0
        ),
    )
    assert clear and "over floor" in why

    # A bigger rating gap, and a margin inside its own error: not shown, so
    # not promoted however far ahead the fit puts it.
    close, why = gate.promotion(
        {"mine": 1.0, "floor": 0.0, "other": -1.0},
        "mine",
        "floor",
        decisive=config.DECISIVE_GAMES,
        over_champion=harness.Margin(
            mean=700.0, worst=-9000.0, best=9000.0, error=722.0
        ),
    )
    assert not close and "inside twice its error" in why

    # And a margin nobody measured is not a promotion either.
    unmeasured, why = gate.promotion(
        standings, "mine", "floor", decisive=config.DECISIVE_GAMES
    )
    assert not unmeasured and "not measured" in why


def test_leading_the_field_is_not_enough_to_replace_the_floor() -> None:
    """Top of the table and level with the champion promotes nothing.

    Rank alone has no margin in it, and over a sampled draw a rank is
    estimated through the fit rather than measured. It is exactly the case the
    margin exists for.
    """
    ranked = {"mine": 2.0, "floor": 1.95, "third": 0.0}

    clear, _ = gate.promotion(
        ranked,
        "mine",
        "floor",
        decisive=config.DECISIVE_GAMES,
        over_champion=harness.Margin(
            mean=700.0, worst=-9000.0, best=9000.0, error=722.0
        ),
    )

    assert sorted(ranked, key=lambda n: -ranked[n])[0] == "mine"
    assert not clear


def test_clearing_the_floor_is_not_enough_without_topping_the_field() -> None:
    """The two questions came apart within an hour of the pool growing.

    Candidates promoted at third and fourth of sixty-four for clearing a floor
    that was no longer the best agent in the field. Beating the agent you
    replace says the ratchet turned; being the best says it is worth turning.
    """
    ranked = {
        "leader": 3.0,
        "runner_up": 2.5,
        "mine": 2.0,
        "floor": 1.0,
    }

    clear, why = gate.promotion(
        ranked,
        "mine",
        "floor",
        decisive=config.DECISIVE_GAMES,
        over_champion=harness.Margin(
            mean=5968.0, worst=-400.0, best=9000.0, error=722.0
        ),
    )

    # Comfortably past the floor and shown to be, and third of four.
    assert ranked["mine"] > ranked["floor"]
    assert not clear
    assert "3 of 4" in why and "below leader" in why


def test_before_there_is_a_floor_the_field_is_the_bar() -> None:
    """The first promotion of a run has nothing to be a margin above."""
    enough = config.DECISIVE_GAMES
    assert gate.promotion({"mine": 1.0, "a": 0.5}, "mine", None, decisive=enough)[0]
    assert not gate.promotion({"mine": 0.4, "a": 0.5}, "mine", None, decisive=enough)[0]
    # And a floor the fit has never heard of is no floor either.
    assert gate.promotion({"mine": 1.0, "a": 0.5}, "mine", "gone", decisive=enough)[0]
    # No floor means nothing to be indistinguishable from, so the decisive
    # bar has nothing to say and a fresh run is not blocked by it.
    assert gate.promotion({"mine": 1.0, "a": 0.5}, "mine", None, decisive=0)[0]


def test_second_place_is_not_promoted_however_close() -> None:
    """A floor that rose on anything but the best would not be the best."""
    ok, why = gate.promotion(
        {"a": 0.9001, "c": 0.9000, "b": -0.5}, "c", decisive=config.DECISIVE_GAMES
    )

    assert not ok
    assert "2 of 3" in why and "below a" in why


def test_the_reason_says_where_it_placed_not_who_it_failed_to_beat() -> None:
    """A place is something a next round can aim at; a list of losses is not.

    The absolute rule this replaced named every opponent a program did not
    beat, which for a program that beat none of them was the whole pool and
    told it nothing about which to attack first.
    """
    ok, why = gate.promotion(
        {"a": 2.0, "b": 1.0, "c": 0.0, "mine": -1.0},
        "mine",
        decisive=config.DECISIVE_GAMES,
    )

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
    assert gate.promotion(
        {"mine": 1.5, "a": 1.0, "b": 0.2}, "mine", decisive=config.DECISIVE_GAMES
    )[0]


def test_the_standings_are_over_the_pool_and_nothing_else() -> None:
    """The gate is a place in one tournament, not a score against a set.

    There is no held-out set any more and no second measurement: seeds
    are drawn fresh every evaluation, so every rate here was already
    measured on maps this program was never selected on.
    """
    candidate = result("c", 0.8, {"a": 0.9, "b": 0.8})

    assert gate.promotion(
        {"c": 1.0, "a": 0.1, "b": -0.4}, "c", decisive=config.DECISIVE_GAMES
    )[0]
    assert set(candidate.rates) == {"a", "b"}


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
    assert champion.name == "champion_1"
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
    assert "champion_1" in saved.names()


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
    one, two = saved.opponents["champion_1"], saved.opponents["champion_2"]
    assert one != two
    assert Path(one).read_text() == Path(first.source_path).read_text()
    assert Path(two).read_text() == Path(second.source_path).read_text()
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

    assert champion.name == "champion_4"
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
    assert champion.name == "champion_2"


def test_a_candidate_that_draws_the_floor_is_the_floor() -> None:
    """Out-rating the champion is not the same as beating it.

    The numbers are champion_55's, measured 2026-09-08. It was promoted over
    champion_54 on a rating gap the bar reads as "about a 54% head-to-head".
    Their thirty-two games were two wins by five units and thirty exact draws:
    the same program, plus a rounding error. A rating is fitted over every
    pairing on the record, so the gap came from beating *ancestors* -- which
    champion_54 also beat -- while the pairing that decides whether anything
    improved was drawn.
    """
    ranked = {"mine": 2.0, "floor": 1.0, "old": 0.0}

    stuck, why = gate.promotion(
        ranked,
        "mine",
        "floor",
        decisive=2,
        over_champion=harness.Margin(
            mean=5968.0, worst=-400.0, best=9000.0, error=722.0
        ),
    )

    # Top of the field, and its margin over the floor well past its own error:
    # everything else the bar asks for, cleared.
    assert sorted(ranked, key=lambda n: -ranked[n])[0] == "mine"
    assert ranked["mine"] > ranked["floor"]
    assert not stuck
    assert "the two play the same game" in why


def test_the_decisive_bar_is_read_on_games_not_on_the_rate() -> None:
    """Only one of the two shapes behind a rate of 0.53125 is evidence.

    Thirty draws and two wins score exactly as seventeen wins and fifteen
    losses do. So the bar counts games that ended with a winner. The same candidate, the
    same rating gap, and the only thing that differs is whether the pairing
    was ever decided.
    """
    ranked = {"mine": 2.0, "floor": 1.0, "old": 0.0}
    shown = harness.Margin(mean=5968.0, worst=-400.0, best=9000.0, error=722.0)

    assert not gate.promotion(ranked, "mine", "floor", decisive=2, over_champion=shown)[
        0
    ]
    assert gate.promotion(ranked, "mine", "floor", decisive=32, over_champion=shown)[0]
