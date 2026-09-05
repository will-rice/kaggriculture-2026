"""The promotion rule on hand-built numbers, the tarball, and the floor write."""

import tarfile
from pathlib import Path

import pytest

from kaggriculture.campaign import archive, config, evaluator, gate, pool


def result(
    pid: str, score: float, low: float, rates: dict[str, float], width: float = 0.05
) -> evaluator.DeepResult:
    """A hand-built DeepResult with symmetric per-opponent intervals."""
    return evaluator.DeepResult(
        program_id=pid,
        score=score,
        low=low,
        high=min(1.0, score + (score - low)),
        rates=rates,
        intervals={
            n: (max(0.0, r - width), min(1.0, r + width)) for n, r in rates.items()
        },
        field=score,
        held_out={},
        games=128,
    )


def test_a_candidate_that_beats_every_opponent_is_promoted() -> None:
    """The whole rule: strictly more than half the exam games against each."""
    ok, why = gate.promotion(result("c", 0.7, 0.65, {"a": 0.51, "b": 0.9, "c": 0.6}))

    assert ok and why == "beat every opponent"


def test_one_opponent_it_does_not_beat_is_enough_to_refuse() -> None:
    """A champion is a gatekeeper and the program we would submit.

    Something that loses to an opponent in the pool is a weak gatekeeper and
    a weak submission, however well it does against the rest.
    """
    ok, why = gate.promotion(result("c", 0.8, 0.75, {"a": 0.99, "b": 0.99, "c": 0.4}))

    assert not ok and why == "did not beat c at 0.400"


def test_the_reason_names_every_opponent_it_failed_against() -> None:
    """This message is the campaign's answer to "why is nothing promoting?".

    Naming only the first would hide how far a lineage is from the bar, which
    is the difference between one opponent to fix and four.
    """
    ok, why = gate.promotion(
        result("c", 0.5, 0.45, {"a": 0.9, "b": 0.3, "c": 0.45, "d": 0.2})
    )

    assert not ok
    assert why == "did not beat b at 0.300, c at 0.450, d at 0.200"


def test_a_dead_heat_is_not_a_win() -> None:
    """Half the games is not more than half, and the pool is full of near-ties."""
    assert not gate.promotion(result("c", 0.75, 0.7, {"a": 0.9, "b": 0.5}))[0]


def test_held_out_opponents_are_not_part_of_the_bar() -> None:
    """They are the honest generalisation number, never the gate."""
    candidate = result("c", 0.8, 0.75, {"a": 0.9, "b": 0.8}).model_copy(
        update={"held_out": {"salemali7_2900": 0.1, "lynnsakurai_v5": 0.2}}
    )

    assert gate.promotion(candidate)[0]


def test_a_promotion_asks_nothing_of_the_champion() -> None:
    """The champion is in the pool, so beating it is part of beating them all."""
    assert "champion" not in gate.promotion.__code__.co_varnames


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
        fitness=0.5,
        rates={"a": 0.5},
        created=0.0,
    )


def _result() -> evaluator.DeepResult:
    """The deep result the promotion tests promote on."""
    return result("p9", 0.7, 0.65, {"a": 0.9, "b": 0.5})


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
    assert "main.py" in names and "kaggriculture_engine.so" in names


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
    gate.enroll(
        gate.promote(second, result("p9", 0.8, 0.75, {"a": 0.9, "b": 0.6})), reloaded
    )

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

    champion = gate.promote(
        program, result("p9", 0.9, 0.85, {"a": 0.99, "champion_3": 0.99})
    )
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
    champion = gate.promote(program, result("p9", 0.8, 0.75, {"a": 0.9, "b": 0.6}))
    gate.enroll(gate.record(champion), reloaded)
    assert champion.name == "champion_2"
