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


def test_no_champion_promotes_anything_with_a_score() -> None:
    """With no champion yet, any candidate is promoted."""
    ok, why = gate.promotion(result("c", 0.3, 0.25, {"a": 0.3}), None)
    assert ok


def test_lower_bound_must_beat_the_champion_point_estimate() -> None:
    """The candidate's lower Wilson bound must exceed the champion's score."""
    champ = result("k", 0.60, 0.55, {"a": 0.6})
    assert not gate.promotion(result("c", 0.62, 0.58, {"a": 0.62}), champ)[0]
    assert gate.promotion(result("c", 0.70, 0.65, {"a": 0.70}), champ)[0]


def test_field_may_not_drop_more_than_two_points() -> None:
    """A candidate that otherwise beats the bound still fails on field drop."""
    champ = result("k", 0.60, 0.55, {"a": 0.6})
    candidate = result("c", 0.70, 0.65, {"a": 0.70})
    candidate = candidate.model_copy(update={"field": 0.57})
    assert not gate.promotion(candidate, champ)[0]


def test_no_opponent_may_regress_beyond_noise() -> None:
    """A single opponent regressing past the wider interval blocks promotion."""
    champ = result("k", 0.60, 0.55, {"a": 0.9, "b": 0.3})
    candidate = result("c", 0.70, 0.65, {"a": 0.6, "b": 0.8})
    ok, why = gate.promotion(candidate, champ)
    assert not ok and "a" in why


def test_field_drop_of_exactly_the_tolerance_still_passes() -> None:
    """A field drop of exactly FIELD_TOLERANCE is not a drop beyond it."""
    champ = result("k", 0.60, 0.55, {"a": 0.6})
    candidate = result("c", 0.70, 0.65, {"a": 0.70})
    candidate = candidate.model_copy(
        update={"field": champ.field - gate.FIELD_TOLERANCE}
    )
    assert gate.promotion(candidate, champ)[0]


def test_opponent_regression_of_exactly_the_width_still_passes() -> None:
    """A rate exactly champion - width is not a regression beyond it."""
    champ = result("k", 0.60, 0.55, {"a": 0.9})
    width = champ.intervals["a"][1] - champ.intervals["a"][0]
    candidate = result("c", 0.70, 0.65, {"a": champ.rates["a"] - width})
    assert gate.promotion(candidate, champ)[0]


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
    monkeypatch.setattr(gate, "SERVED", tmp_path / "served" / "main.py")
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
    p = pool.Pool(opponents={"a": "/x/a.py"}, weights={"a": 1.0})

    champion = gate.promote(program, _result(), p)

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
    """promote() writes the floor read-only, mirrors served/, and saves the pool."""
    program = _program(tmp_path, monkeypatch)
    source = Path(program.source_path)
    p = pool.Pool(
        opponents={"a": "/x/a.py", "b": "/x/b.py"}, weights={"a": 0.5, "b": 0.5}
    )
    champion = gate.promote(program, _result(), p)
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
    assert gate.SERVED.read_text() == source.read_text()
    saved = pool.Pool.load(config.POOL)
    assert "champion_1" in saved.names() and saved.weights["b"] > saved.weights["a"]


def test_promote_refuses_a_champion_name_the_directory_already_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rebuilt pool must not overwrite -- or trip over -- an older champion file.

    No image test is stubbed here: the name check has to come before the
    gate spends ten minutes of docker on a tarball it could not keep.
    """
    program = _program(tmp_path, monkeypatch)
    p = pool.Pool(opponents={"a": "/x/a.py"}, weights={"a": 1.0})
    config.CHAMPIONS.mkdir(parents=True)
    stale = config.CHAMPIONS / "champion_1.py"
    stale.write_text("# an older champion\n", encoding="utf-8")
    stale.chmod(0o444)

    with pytest.raises(FileExistsError, match="champion_1.py already exists"):
        gate.promote(program, _result(), p)
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
    p = pool.Pool(
        opponents={"a": "/x/a.py", "b": "/x/b.py"}, weights={"a": 0.5, "b": 0.5}
    )
    gate.promote(first, _result(), p)

    second = _program(tmp_path, monkeypatch, body="WATER")
    reloaded = pool.Pool.load(config.POOL)
    gate.promote(second, result("p9", 0.8, 0.75, {"a": 0.9, "b": 0.6}), reloaded)

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
    """`champion.json` exists the moment promote returns; a restart trusts it."""
    program = _program(tmp_path, monkeypatch)
    p = pool.Pool(
        opponents={"a": "/x/a.py", "b": "/x/b.py"}, weights={"a": 0.5, "b": 0.5}
    )
    deep = _result()

    champion = gate.promote(program, deep, p)

    assert gate.load_champion() == champion
    assert champion.result == deep
    assert Path(champion.path).read_text() == Path(program.source_path).read_text()


def test_a_second_promotion_on_the_saved_pool_yields_champion_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The champion counter reads the pool that was just saved, not a stale one."""
    program = _program(tmp_path, monkeypatch)
    p = pool.Pool(
        opponents={"a": "/x/a.py", "b": "/x/b.py"}, weights={"a": 0.5, "b": 0.5}
    )
    gate.promote(program, _result(), p)
    reloaded = pool.Pool.load(config.POOL)
    champion = gate.promote(
        program,
        result("p9", 0.8, 0.75, {"a": 0.9, "b": 0.6}),
        reloaded,
    )
    assert champion.name == "champion_2"
