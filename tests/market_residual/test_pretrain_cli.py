"""The pretrain command: owned roots, bound parameters, resumable training.

Everything here runs without the engine: the loader is replaced with the same
synthetic sequences the loss tests use, so what is exercised is exactly what
the command adds — argument shape, root ownership, parameter binding, atomic
checkpoints, drift refusal, and the gate verdict — not the season replay it
would do on a real collection root.
"""

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from kaggriculture.learn.market_residual.model import MarketResidualNet, ModelConfig
from kaggriculture.learn.market_residual.offline import (
    EventSequence,
    PretrainConfig,
    PretrainDriftError,
    PretrainGateError,
    check_gates,
    load_checkpoint,
    pretrain,
    save_checkpoint,
)
from kaggriculture.scripts import market_pretrain
from tests.market_residual.test_offline import manual_sequence

CONFIG = ModelConfig.current()


def tiny_sequences(count: int = 3) -> tuple[EventSequence, ...]:
    """Return small unlearnable sequences: labels carry no feature signal."""
    return tuple(
        manual_sequence(
            {0: ((0.5, 0.1, False),), 2: ((-0.5, -0.1, False), (0.0, 0.0, False))},
            events=4,
            feature_fill=0.1 * (index + 1),
        )
        for index in range(count)
    )


def fake_collection_root(path: Path, bank: str) -> Path:
    """Write the minimum a collection root must hold to be bound against."""
    from kaggriculture.learn.market_residual.counterfactual import canonical_digest

    fields: dict[str, Any] = {"seed_bank": bank, "seeds": [0], "cells": []}
    path.mkdir(parents=True)
    (path / "shards").mkdir()
    (path / "collection-parameters.json").write_text(
        json.dumps({"fields": fields, "sha256": canonical_digest(fields)})
    )
    return path


@pytest.fixture()
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Return a pretrain root and two fake collection roots, loader patched."""
    monkeypatch.setattr(
        market_pretrain,
        "load_sequences",
        lambda root, workers: tiny_sequences(),
    )
    return {
        "root": tmp_path / "pretrain",
        "train": fake_collection_root(tmp_path / "train", "counterfactual_train"),
        "select": fake_collection_root(tmp_path / "select", "counterfactual_temporal"),
    }


def arguments(roots: dict[str, Path], *extra: str) -> list[str]:
    """Return the command line for one invocation against the fixture roots."""
    return [
        str(roots["root"]),
        str(roots["train"]),
        str(roots["select"]),
        "--device",
        "cpu",
        "--epochs",
        "1",
        *extra,
    ]


def test_cli_requires_exactly_one_mode(roots: dict[str, Path]) -> None:
    """No mode, or both modes, is refused at the parser."""
    with pytest.raises(SystemExit):
        market_pretrain.parser().parse_args(arguments(roots))
    with pytest.raises(SystemExit):
        market_pretrain.parser().parse_args(arguments(roots, "--create", "--resume"))


def test_required_arguments_are_positional(roots: dict[str, Path]) -> None:
    """The three required roots are positional; flags stay optional."""
    args = market_pretrain.parser().parse_args(arguments(roots, "--create"))
    assert args.root == roots["root"]
    assert args.train_root == roots["train"]
    assert args.select_root == roots["select"]
    assert args.device == "cpu"
    assert not args.wandb


def test_main_trains_reports_and_fails_its_own_gate(roots: dict[str, Path]) -> None:
    """An untrained head fails the gate after the report is published."""
    with pytest.raises(PretrainGateError):
        market_pretrain.main(arguments(roots, "--create"))
    report = json.loads((roots["root"] / "offline-report.json").read_text())
    assert report["epoch"] == 1
    assert set(report["gates"]) >= {
        "exploit_rate_argmax",
        "exploit_rate_any",
        "equal_retention",
        "expected_delta",
        "activation",
        "ranking_accuracy",
        "nonfinite",
    }
    assert (roots["root"] / "last.ckpt").exists()
    assert (roots["root"] / "best.ckpt").exists()
    assert not list(roots["root"].glob("*.tmp*"))


def test_create_refuses_a_claimed_root_and_resume_an_unclaimed_one(
    roots: dict[str, Path],
) -> None:
    """Root ownership is explicit in both directions."""
    with pytest.raises(PretrainGateError):
        market_pretrain.main(arguments(roots, "--create"))
    with pytest.raises(market_pretrain.PretrainParameterError):
        market_pretrain.main(arguments(roots, "--create"))
    fresh = {**roots, "root": roots["root"].with_name("unclaimed")}
    with pytest.raises(market_pretrain.PretrainParameterError):
        market_pretrain.main(arguments(fresh, "--resume"))


def test_resume_rejects_parameter_drift(roots: dict[str, Path]) -> None:
    """A resume under different parameters is refused by name."""
    with pytest.raises(PretrainGateError):
        market_pretrain.main(arguments(roots, "--create"))
    drifted = arguments(roots, "--resume")
    drifted[drifted.index("--epochs") + 1] = "2"
    with pytest.raises(market_pretrain.PretrainParameterError):
        market_pretrain.main(drifted)


def test_checkpoints_roundtrip_atomically(tmp_path: Path) -> None:
    """A checkpoint lands whole or not at all, and reloads."""
    path = tmp_path / "last.ckpt"
    save_checkpoint(path, {"epoch": 3, "weights": torch.arange(4)})
    save_checkpoint(path, {"epoch": 4, "weights": torch.arange(4)})
    assert load_checkpoint(path)["epoch"] == 4
    assert not list(tmp_path.glob("*.tmp*"))


def test_pretrain_resumes_where_the_interruption_left_it(tmp_path: Path) -> None:
    """An interrupted run continues from its last finished epoch."""
    sequences = tiny_sequences()
    config = PretrainConfig(epochs=3, batch_cells=2, device="cpu")

    def interrupt(epoch: int, metrics: dict[str, float]) -> None:
        if epoch == 2:
            raise KeyboardInterrupt

    torch.manual_seed(0)
    model = MarketResidualNet(CONFIG)
    with pytest.raises(KeyboardInterrupt):
        pretrain(model, sequences, sequences, config, tmp_path, log=interrupt)
    assert load_checkpoint(tmp_path / "last.ckpt")["epoch"] == 2
    report = pretrain(model, sequences, sequences, config, tmp_path)
    assert report["epoch"] == 3
    assert [row["epoch"] for row in report["history"]] == [1, 2, 3]


def test_pretrain_rejects_config_and_data_drift(tmp_path: Path) -> None:
    """Changed hyperparameters or changed data refuse to resume."""
    sequences = tiny_sequences()
    config = PretrainConfig(epochs=1, batch_cells=2, device="cpu")
    torch.manual_seed(0)
    model = MarketResidualNet(CONFIG)
    pretrain(model, sequences, sequences, config, tmp_path)
    with pytest.raises(PretrainDriftError):
        pretrain(
            model,
            sequences,
            sequences,
            PretrainConfig(epochs=1, batch_cells=2, device="cpu", learning_rate=1e-4),
            tmp_path,
        )
    poisoned = tiny_sequences()
    poisoned[0].features[0, 0] += 1.0
    with pytest.raises(PretrainDriftError):
        pretrain(model, poisoned, sequences, config, tmp_path)


def test_the_gate_verdict_is_the_declared_bands() -> None:
    """Every declared band fails the verdict exactly when broken."""
    passing = {
        "exploit_rate_argmax": 0.8,
        "exploit_rate_any": 0.9,
        "equal_retention": 0.9,
        "expected_delta": 0.01,
        "activation": 0.2,
        "ranking_accuracy": 0.9,
        "nonfinite": 0,
    }
    check_gates(passing)
    for key, value in (
        ("exploit_rate_argmax", 0.5),
        ("equal_retention", 0.4),
        ("expected_delta", -0.01),
        ("activation", 0.6),
        ("activation", 0.0),
        ("nonfinite", 1),
    ):
        with pytest.raises(PretrainGateError):
            check_gates({**passing, key: value})
