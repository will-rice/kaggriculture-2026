"""Pretrain the market residual from two owned counterfactual roots.

``--create`` claims a fresh pretraining root and writes down everything the
run is parameterised by: the training config, the model and runtime digests,
the digest of this trainer's own source, and a manifest of both data roots —
their stored collection parameters and every published shard's digest.
``--resume`` reopens a claimed root and refuses if any of that has changed;
what it may never do is quietly continue a run whose data, code or
hyperparameters are no longer the ones the checkpoints were trained under.

The temporal split is structural: the training root must be bound to the
``counterfactual_train`` bank and the selection root to the temporally later
``counterfactual_temporal`` bank, so best-checkpoint selection and every gate
are measured on seeds no update ever saw.

Rebuilding a cell means replaying its recorded season through the reference
engine, which is minutes of CPU across a few hundred cells; the sequences are
therefore cached in the root, keyed by the data manifest digest and named for
the families ``--exclude-families`` drops, and rebuilt only when the data
itself changed — which is a drift refusal anyway.

Gates are enforced, not just reported: after training, the best-selection
checkpoint must hold the known-exploit rate, retain the controller on
equal-outcome cells, keep the unseen paired win-point delta nonnegative, stay
finite everywhere, and land its activation inside the export band. The report
is published before the verdict, so a failed gate leaves the evidence.
"""

import argparse
import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from lightning import seed_everything

from kaggriculture.learn.market_residual import offline
from kaggriculture.learn.market_residual.counterfactual import canonical_digest
from kaggriculture.learn.market_residual.model import MarketResidualNet, ModelConfig
from kaggriculture.learn.market_residual.offline import (
    LAST_CHECKPOINT,
    EventSequence,
    PretrainConfig,
    check_gates,
    load_checkpoint,
    load_sequences,
    pretrain,
    write_json_atomic,
)
from kaggriculture.market_residual.numpy_policy import model_digest, runtime_digest
from kaggriculture.scripts.market_counterfactuals import root_lock

LOGGER = logging.getLogger(__name__)

PRETRAIN_VERSION = 1

PARAMETERS_NAME = "pretrain-parameters.json"

TRAIN_BANK = "counterfactual_train"
SELECT_BANK = "counterfactual_temporal"

WANDB_ENTITY = "will-rice"
WANDB_PROJECT = "kaggriculture-2026"


class PretrainParameterError(RuntimeError):
    """Raised when a root's stored parameters and this run's disagree."""


def main(argv: Sequence[str] | None = None) -> None:
    """Own one pretraining root, train the head, and hold it to the gates.

    Args:
        argv: The argument vector; the process's own when omitted.

    Raises:
        PretrainGateError: If the finished run is outside any declared band.
    """
    args = parser().parse_args(None if argv is None else list(argv))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = PretrainConfig(
        epochs=args.epochs,
        device=args.device,
        excluded_families=args.exclude_families,
    )
    with root_lock(args.root):
        train_manifest = collection_manifest(args.train_root, TRAIN_BANK)
        select_manifest = collection_manifest(args.select_root, SELECT_BANK)
        reconcile_parameters(
            args.root,
            pretrain_parameters(config, train_manifest, select_manifest),
            create=args.create,
        )
        seed_everything(config.seed, workers=True)
        train_sequences = cached_sequences(
            args.root, args.train_root, train_manifest, args.workers, config
        )
        select_sequences = cached_sequences(
            args.root, args.select_root, select_manifest, args.workers, config
        )
        model = MarketResidualNet(ModelConfig.current())
        run = wandb_run(args) if args.wandb else None
        report = pretrain(
            model,
            train_sequences,
            select_sequences,
            config,
            args.root,
            log=(lambda epoch, row: run.log(row, step=epoch)) if run else None,
            run_id=run.id if run else None,
        )
        if run is not None:
            run.summary.update(report["gates"])
            run.finish()
    LOGGER.info("gates: %s", report["gates"])
    check_gates(report["gates"])


def parser() -> argparse.ArgumentParser:
    """Build the command line without touching a file.

    Returns:
        The parser; exactly one of ``--create`` and ``--resume`` is required.
    """
    arguments = argparse.ArgumentParser(prog="market_pretrain", description=__doc__)
    arguments.add_argument("root", type=Path, help="pretraining root to own")
    arguments.add_argument(
        "train_root", type=Path, help="counterfactual-train collection root"
    )
    arguments.add_argument(
        "select_root", type=Path, help="temporal selection collection root"
    )
    mode = arguments.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create", action="store_true", help="claim a fresh root")
    mode.add_argument("--resume", action="store_true", help="reopen a claimed root")
    arguments.add_argument("--device", default="cuda")
    arguments.add_argument("--epochs", type=int, default=20)
    arguments.add_argument(
        "--workers", type=int, default=8, help="season replay processes"
    )
    arguments.add_argument(
        "--exclude-families",
        default="",
        help="comma-separated replacement families to drop from training",
    )
    arguments.add_argument("--wandb", action="store_true", help="log to W&B")
    return arguments


def collection_manifest(root: Path, bank: str) -> dict[str, Any]:
    """Describe one collection root exactly enough to notice it changing.

    Args:
        root: The collection root.
        bank: The seed bank this root must have been collected from.

    Returns:
        The root's stored parameter digest and every published shard's digest.

    Raises:
        PretrainParameterError: If the root is missing, fails its own digest,
            or was collected from the wrong bank.
    """
    path = root / "collection-parameters.json"
    if not path.exists():
        raise PretrainParameterError(f"{root} is not a collection root")
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored["sha256"] != canonical_digest(stored["fields"]):
        raise PretrainParameterError(f"{path} does not match its own digest")
    if stored["fields"]["seed_bank"] != bank:
        raise PretrainParameterError(
            f"{root} was collected from bank {stored['fields']['seed_bank']!r}, "
            f"this role requires {bank!r}"
        )
    shards = {
        manifest.name: manifest.read_text(encoding="utf-8").strip()
        for manifest in sorted((root / "shards").glob("*.sha256"))
    }
    return {
        "root": str(root),
        "seed_bank": bank,
        "parameters_sha256": stored["sha256"],
        "shards": shards,
    }


def pretrain_parameters(
    config: PretrainConfig,
    train_manifest: dict[str, Any],
    select_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Return the canonical parameterisation a pretraining root is bound to.

    Args:
        config: The training configuration.
        train_manifest: The training root's manifest.
        select_manifest: The selection root's manifest.

    Returns:
        The fields and their digest.
    """
    fields: dict[str, Any] = {
        "version": PRETRAIN_VERSION,
        "config": asdict(config),
        "model_sha256": model_digest(ModelConfig.current()),
        "runtime_sha256": runtime_digest(),
        "trainer_sha256": hashlib.sha256(
            Path(offline.__file__).read_bytes()
        ).hexdigest(),
        "train_manifest": train_manifest,
        "select_manifest": select_manifest,
    }
    return {"fields": fields, "sha256": canonical_digest(fields)}


def reconcile_parameters(
    root: Path, wanted: dict[str, Any], create: bool
) -> dict[str, Any]:
    """Bind a pretraining root to one parameterisation, once.

    Args:
        root: The pretraining root.
        wanted: This run's canonical parameters.
        create: Whether the root is being claimed rather than reopened.

    Returns:
        The stored parameters.

    Raises:
        PretrainParameterError: If a create meets a live root, a resume meets
            an unclaimed one, the stored file fails its own digest, or the
            stored parameters are not this run's.
    """
    path = root / PARAMETERS_NAME
    if create:
        if path.exists():
            raise PretrainParameterError(f"{root} is already claimed; use --resume")
        write_json_atomic(path, wanted)
        return wanted
    if not path.exists():
        raise PretrainParameterError(f"{root} was never created; use --create")
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored["sha256"] != canonical_digest(stored["fields"]):
        raise PretrainParameterError(f"{path} does not match its own digest")
    if stored["fields"] != wanted["fields"]:
        differing = sorted(
            key
            for key in set(stored["fields"]) | set(wanted["fields"])
            if stored["fields"].get(key) != wanted["fields"].get(key)
        )
        raise PretrainParameterError(
            f"{root} was created with different parameters: {differing}"
        )
    return stored


def cached_sequences(
    root: Path,
    collection_root: Path,
    manifest: dict[str, Any],
    workers: int,
    config: PretrainConfig,
) -> tuple[EventSequence, ...]:
    """Rebuild one root's sequences, or reuse the cache this root already holds.

    The cache file is named for the bank *and* the excluded families, because
    those two together decide which arms a rebuild produces: a stale cache from
    an unfiltered run holds arms this run must never see, and a manifest digest
    alone cannot tell the two apart.

    Args:
        root: The pretraining root the cache lives in.
        collection_root: The collection root to rebuild from.
        manifest: That root's manifest; its digest keys the cache.
        workers: How many replay processes a rebuild may use.
        config: The run's parameterisation, for the families it excludes.

    Returns:
        The sequences.
    """
    key = canonical_digest(manifest)
    excluded = config.excluded_families.replace(",", "-")
    name = f"sequences-{manifest['seed_bank']}"
    path = root / (f"{name}-without-{excluded}.pt" if excluded else f"{name}.pt")
    if path.exists():
        cached = load_checkpoint(path)
        if cached["manifest_sha256"] == key:
            return tuple(cached["sequences"])
    sequences = load_sequences(collection_root, workers, config.excluded())
    offline.save_checkpoint(
        path, {"manifest_sha256": key, "sequences": list(sequences)}
    )
    return sequences


def wandb_run(args: argparse.Namespace) -> Any:  # noqa: ANN401 - a wandb.Run
    """Start or resume this root's W&B run.

    The run identity lives in ``last.ckpt``; a resumed root continues the same
    run rather than scattering one training across several.

    Args:
        args: The parsed command line.

    Returns:
        The live run.
    """
    import wandb

    run_id = None
    last = args.root / LAST_CHECKPOINT
    if last.exists():
        run_id = load_checkpoint(last)["wandb_run_id"]
    return wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"market-pretrain-{args.root.name}",
        id=run_id,
        resume="must" if run_id else None,
        config={"root": str(args.root), "epochs": args.epochs},
    )


if __name__ == "__main__":
    main()
