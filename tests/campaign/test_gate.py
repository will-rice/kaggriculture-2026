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


def beat(
    rate: float,
    decided: int,
    field: float = 0.30,
    champion_field: float = 0.20,
    champion: str = "floor",
    shared: int = 20,
) -> tuple[evaluator.Result, gate.Champion]:
    """A candidate and the champion it is judged against.

    ``field`` and ``champion_field`` are each side's win rate over the opponents
    they share, so the two conditions can be moved independently: ``rate`` and
    ``decided`` drive the head-to-head, and the two field figures drive the
    win-rate comparison.
    """
    rates = {champion: rate, **{f"public_{i}": field for i in range(shared)}}
    theirs = {f"public_{i}": champion_field for i in range(shared)}
    result = evaluator.Result(
        program_id="mine",
        fitness=rate,
        field=field,
        rates=rates,
        margins={name: harness.Margin(mean=0.0, worst=0.0, best=0.0) for name in rates},
        decisive={champion: decided, **{f"public_{i}": 32 for i in range(shared)}},
        games=32,
        seeds=[1, 2, 3, 4],
        hardest=champion,
        states={},
    )
    standing = gate.Champion(
        name=champion,
        path=f"/champions/{champion}.py",
        tarball="",
        result=result.model_copy(update={"program_id": champion, "rates": theirs}),
    )
    return result, standing


def test_a_better_rate_and_a_decisive_win_promotes() -> None:
    """Both conditions, and this is the case that clears them.

    0.30 against the shared opponents where the champion has 0.20 is well past
    twice the error of the difference, and 22 of 32 against the champion is where
    the Wilson lower bound crosses 0.5.
    """
    clear, why = gate.promotion(*beat(22 / 32, 32))

    assert clear, why
    assert "against the field" in why and "lower bound" in why


def test_a_better_rate_without_the_pairing_does_not_promote() -> None:
    """Beating the field on average is not beating the program it replaces.

    An agent can win more of the field and still lose to the specific one it is
    taking the slot from, which is not a ratchet.
    """
    close, why = gate.promotion(*beat(21 / 32, 32))

    assert not close
    assert "not shown to beat it" in why


def test_winning_the_pairing_while_losing_the_field_does_not_promote() -> None:
    """The condition the campaign learned it needed, twice, on 2026-09-13.

    `champion_3` was promoted at a field rate of 0.114 over a champion at 0.155,
    and `champion_8` at 0.221 over one at 0.244 -- each having beaten the program
    it replaced decisively. Two of nine promotions handed back field ground,
    which is a ratchet turning the wrong way.
    """
    worse, why = gate.promotion(*beat(28 / 32, 32, field=0.114, champion_field=0.155))

    assert not worse
    assert "behind by more than twice its error" in why


def test_a_rate_level_with_the_champion_promotes_on_the_pairing() -> None:
    """Being level on the field is not a reason to refuse. Losing the pairing is.

    The field rate demanded an improvement until 2026-09-16, on the winner's
    curse: 78 of 471 programs once topped a noisy ranking and none survived a
    deeper look. That reasoning was about selecting the maximum of a noisy
    estimator, and the pairing is not one -- the Wilson lower bound over
    `DECISIVE_GAMES` decided games is a calibrated test, which is why it is the
    condition that does the deciding here.

    What ended the old rule is that it stopped being answerable. champion_19
    beat every agent in the pool, so its field rate was 1.000 and no program
    that could exist scores higher; fifty candidates were refused in six hours
    without the pairing ever being read, two of them beating champion_19 31-1
    and 30-2.
    """
    level, why = gate.promotion(*beat(28 / 32, 32, field=0.2253, champion_field=0.2209))

    assert level, why

    # And the pairing still has to be won, level field or not.
    drawn, why = gate.promotion(*beat(17 / 32, 32, field=0.2253, champion_field=0.2209))

    assert not drawn
    assert "not shown to beat it" in why


def test_a_candidate_that_mostly_draws_is_not_promoted() -> None:
    """Two programs that draw are the same program, whatever the rate says.

    champion_55 was promoted over champion_54 on two wins and thirty exact draws
    in 32 games; 2/2 reads as 1.000. `DECISIVE_GAMES` covers the band the
    interval does not -- 4 to 7 decided out of 32 is a candidate drawing 78% to
    88% of its games with the champion -- and below 4 the interval refuses on its
    own, since 3/3 is 0.438.
    """
    drawn, why = gate.promotion(*beat(1.0, 2))

    assert not drawn
    assert "only 2 of its games" in why and "the same game" in why

    thin, why = gate.promotion(*beat(1.0, 7))
    assert not thin and "the bar is 8" in why


def test_an_unplayed_champion_is_not_a_measurement() -> None:
    """A champion that moved under a running evaluation was never played."""
    result, standing = beat(1.0, 32)
    missing, why = gate.promotion(
        result, standing.model_copy(update={"name": "someone_else"})
    )

    assert not missing
    assert "did not play someone_else" in why


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


def banked(
    rate: float,
    decided: int,
    margin: float,
    champion_margin: float,
    error: float = 300.0,
    shared: int = 20,
) -> tuple[evaluator.Result, gate.Champion]:
    """A candidate level on win rate, differing only in coins a game.

    Both sides beat the field at the same rate, so the first condition turns
    entirely on the margin -- which is the case the guard exists for, and the
    one the win rate stopped being able to see once it reached 1.000.
    """
    names = [f"public_{i}" for i in range(shared)]
    rates = {"floor": rate, **dict.fromkeys(names, 0.60)}

    def result_for(pid: str, coins: float, keys: list[str]) -> evaluator.Result:
        return evaluator.Result(
            program_id=pid,
            fitness=rate,
            field=0.60,
            rates={k: rates[k] for k in keys},
            margins={
                k: harness.Margin(mean=coins, worst=coins, best=coins, error=error)
                for k in keys
            },
            games=decided,
            seeds=[1, 2],
            hardest=names[0],
            decisive={"floor": decided},
            states={},
        )

    champion = gate.Champion(
        name="floor",
        path="/x/floor.py",
        tarball="/x/floor.tar.gz",
        result=result_for("floor", champion_margin, names),
    )
    return result_for("mine", margin, ["floor", *names]), champion


def test_banking_less_than_the_champion_is_a_regression(tmp_path: Path) -> None:
    """Level on wins and behind on coins does not promote.

    A round-robin of fourteen public implementations over 96 fresh seeds, run by
    another team and published in the competition's discussions, found no
    intransitive triple, the newer implementation winning 86 of 91 chronological
    pairs, and the ordering tracking average final money closely. A stronger
    economy is a stronger agent here, so a candidate that wins its pairing while
    banking less against the same field has taken something out of the economy
    that the pairing did not charge it for.
    """
    poorer, why = gate.promotion(
        *banked(28 / 32, 32, margin=4_000, champion_margin=9_000)
    )

    assert not poorer
    assert "coins a game" in why, why


def test_banking_more_at_a_level_rate_still_promotes(tmp_path: Path) -> None:
    """The guard refuses a regression; it does not demand an improvement.

    The pairing remains what decides, so a candidate level on the field and
    ahead on coins passes the first condition and is judged on the head-to-head
    like anything else.
    """
    richer, why = gate.promotion(
        *banked(28 / 32, 32, margin=14_000, champion_margin=9_000)
    )

    assert richer, why


def test_a_margin_gap_inside_its_own_noise_is_not_a_regression() -> None:
    """Coins carry an error bar too, and a few hundred of them is not evidence."""
    close, why = gate.promotion(
        *banked(28 / 32, 32, margin=8_900, champion_margin=9_000)
    )

    assert close, why
