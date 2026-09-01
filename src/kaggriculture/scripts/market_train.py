"""Fine-tune the market residual on its own seasons, export it, and gate it.

One command, three phases, in the only order that makes the last one mean
anything: train on the policy's own state distribution, export the trained head
into the artifact the submission actually loads, and play that artifact against
the bare frozen controller over the held-out exam seeds. Re-running the command
on a finished root retrains nothing and re-gates, because the root's own update
counter says the training is done.

**The gate is the whole verdict.** ``market_residual_main.py`` -- the frozen
controller plus this head -- against ``kaito_v56_policy.py`` -- the frozen
controller alone -- over ``holdout.GATE_SEEDS``, both seats, 128 games. Wilson
low above 0.5 is a real improvement; an interval spanning 0.5 is no measured
effect, which is the honest shape of "the policy converged to never deviate";
Wilson high below 0.5 means the head is harming and must not be served. Nothing
else here is a pass or a fail. Activation, event counts and fallback rates are
reported because they explain the verdict, never because they are one.

**The opponents are the ones that can tell two agents apart.** Our own lineage
saturates -- v56 beats everything else we hold at 1.000 -- so the training
league is v56 itself plus the three harvested public kernels that score between
0.27 and 0.50 against it. An opponent that loses every game teaches nothing,
because every deviation looks free against it.

**The learner owns exactly one GPU and the actors own the CPUs.** A season is
seven seconds of reference engine and microseconds of head, so the collection
is CPU-bound by construction and the production command refuses anything but
``--device cuda --devices 1``: a run silently demoted to CPU would take days and
a run spread over two GPUs would publish actor snapshots nobody agreed on.

The export goes through the interface's own tolerance rather than the default
``PARITY_ATOL``, which is a scale-blind absolute bound calibrated on an
untrained head. A trained head emits quantity logits above unit scale, where
one float32 ulp is already 9.5e-07, so ``1e-06`` sits below the rounding floor
and refuses every adequately trained export. ``EXPORT_ATOL`` is derived instead:
3.5x above the worst Torch/NumPy deviation measured over a real season, and 4.4x
below half the smallest margin by which that head decided anything.
"""

import argparse
import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch
from lightning import seed_everything

from kaggriculture.learn.market_residual import online
from kaggriculture.learn.market_residual.counterfactual import canonical_digest
from kaggriculture.learn.market_residual.export import export_numpy_policy
from kaggriculture.learn.market_residual.model import MarketResidualNet, ModelConfig
from kaggriculture.learn.market_residual.offline import (
    load_checkpoint,
    write_json_atomic,
)
from kaggriculture.learn.market_residual.online import (
    BEST_CHECKPOINT,
    EXPORT_ROWS,
    LAST_CHECKPOINT,
    Cell,
    EngineCollector,
    OnlineConfig,
    initialize_deferring,
    train_online,
)
from kaggriculture.market_residual.features import MarketFeatureVector
from kaggriculture.market_residual.numpy_policy import model_digest, runtime_digest
from kaggriculture.market_residual.schema import SEED_BANKS, MarketFeatureSchema
from kaggriculture.report import wilson_interval
from kaggriculture.scripts.market_counterfactuals import root_lock
from kaggriculture.search import arena
from kaggriculture.search.scripts import holdout

LOGGER = logging.getLogger(__name__)

TRAIN_VERSION = 1

PARAMETERS_NAME = "online-parameters.json"
GATE_NAME = "online-gate.json"

# The candidate the gate plays: the frozen controller with this head wrapped
# around it, and nothing else changed.
CANDIDATE = "market_residual_main.py"

# Where the submission's runtime loads the head from. The trained artifact is
# written here so the gate measures the file that would be served, not a copy
# of it sitting somewhere else.
ARTIFACT = Path("src/kaggriculture/market_residual/generated/export.json")

# Derived in `decisive-gate-report.md` from 516 real market events: 3.5x above
# the worst Torch/NumPy float32 deviation, and 4.4x below half the smallest
# margin by which that head decided anything. `export.py` is not modified.
EXPORT_ATOL = 2e-5

# Harvested public kernels, kept only as opponents. Weighted so that half of
# every batch is played against the controller the gate is decided by, and the
# rest against agents that fail differently -- `tetsutani_shopforge` is a coin
# flip against v56, and the other two are structurally unlike our lineage.
OPPONENT_ROOT = "/data/kaggriculture/opponents"

OPPONENTS: tuple[tuple[str, str, float], ...] = (
    ("v56", online.SERVED_PATH, 0.4),
    ("tetsutani_shopforge", f"{OPPONENT_ROOT}/tetsutani_shopforge/main.py", 0.2),
    ("indarkarhana_top10", f"{OPPONENT_ROOT}/indarkarhana_top10/main.py", 0.2),
    ("yhay81_router2929", f"{OPPONENT_ROOT}/yhay81_router2929/main.py", 0.2),
)

WANDB_ENTITY = "will-rice"
WANDB_PROJECT = "kaggriculture-2026"


class OnlineParameterError(RuntimeError):
    """Raised when a root's stored parameters and this run's disagree."""


def main(argv: Sequence[str] | None = None) -> None:
    """Own one online root, train the head, export it, and gate the export.

    Args:
        argv: The argument vector; the process's own when omitted.
    """
    args = parser().parse_args(None if argv is None else list(argv))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    require_one_gpu(args)
    config = OnlineConfig(
        updates=args.updates,
        episodes_per_update=args.episodes,
        seed_bank=args.seed_bank,
        learning_rate=args.learning_rate,
        entropy_start=args.entropy,
        passes=args.passes,
        seed=args.seed,
        device=args.device,
        workers=args.workers,
    )
    cells = build_cells(config, args.seeds)
    with root_lock(args.root):
        reconcile_parameters(
            args.root, online_parameters(config, cells), create=args.create
        )
        seed_everything(config.seed, workers=True)
        model = MarketResidualNet(ModelConfig.current())
        initialize_deferring(model, config)
        run = wandb_run(args) if args.wandb else None
        with EngineCollector(args.root, config) as collect:
            report = train_online(
                model,
                config,
                args.root,
                cells,
                collect,
                log=(lambda update, row: run.log(row, step=update)) if run else None,
                run_id=run.id if run else None,
            )
        LOGGER.info("training finished at update %s", report["updates"])
        verdict = gate(args.root, args.checkpoint, args.gate_workers)
        if run is not None:
            run.summary.update(verdict["gate"])
            run.finish()
    LOGGER.info("gate: %s", json.dumps(verdict["gate"], indent=2))


def parser() -> argparse.ArgumentParser:
    """Build the command line without touching a file.

    Returns:
        The parser; exactly one of ``--create`` and ``--resume`` is required.
    """
    arguments = argparse.ArgumentParser(prog="market_train", description=__doc__)
    arguments.add_argument("root", type=Path, help="online root to own")
    mode = arguments.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create", action="store_true", help="claim a fresh root")
    mode.add_argument("--resume", action="store_true", help="reopen a claimed root")
    arguments.add_argument("--device", default="cuda", help="must be cuda")
    arguments.add_argument("--devices", type=int, default=1, help="must be 1")
    arguments.add_argument("--updates", type=int, default=200)
    arguments.add_argument("--episodes", type=int, default=64)
    arguments.add_argument("--seeds", type=int, default=128, help="cells per opponent")
    arguments.add_argument("--seed-bank", default="online_train")
    arguments.add_argument("--learning-rate", type=float, default=3e-4)
    arguments.add_argument("--entropy", type=float, default=1e-3)
    arguments.add_argument(
        "--passes", type=int, default=2, help="gradient steps per collected batch"
    )
    arguments.add_argument("--seed", type=int, default=0)
    arguments.add_argument("--workers", type=int, default=48, help="actor processes")
    arguments.add_argument("--gate-workers", type=int, default=24)
    arguments.add_argument(
        "--checkpoint",
        default=BEST_CHECKPOINT,
        choices=(BEST_CHECKPOINT, LAST_CHECKPOINT),
        help="which trained head to export and gate",
    )
    arguments.add_argument("--wandb", action="store_true", help="log to W&B")
    return arguments


def require_one_gpu(args: argparse.Namespace) -> None:
    """Refuse a production command that is not one owned CUDA device.

    Args:
        args: The parsed arguments.

    Raises:
        OnlineParameterError: If the device is not an available CUDA device, or
            more than one was asked for.
    """
    if args.device != "cuda":
        raise OnlineParameterError(
            f"--device must be cuda, got {args.device!r}; a run demoted to the "
            "CPU takes days and is not the run this root is bound to"
        )
    if args.devices != 1:
        raise OnlineParameterError(f"--devices must be 1, got {args.devices}")
    if not torch.cuda.is_available():
        raise OnlineParameterError("no CUDA device is available")


def build_cells(config: OnlineConfig, seeds: int) -> tuple[Cell, ...]:
    """Return every paired game the run may play, in a fixed order.

    Args:
        config: The run's parameterisation, for the seed bank.
        seeds: How many bank seeds each opponent is played on.

    Returns:
        One cell per opponent, seed and seat.
    """
    bank = SEED_BANKS[config.seed_bank][:seeds]
    return tuple(
        Cell(opponent=name, path=path, seed=int(seed), seat=seat, weight=weight)
        for name, path, weight in OPPONENTS
        for seed in bank
        for seat in (0, 1)
    )


def online_parameters(config: OnlineConfig, cells: Sequence[Cell]) -> dict[str, Any]:
    """Return the canonical parameterisation an online root is bound to.

    Args:
        config: The run's parameterisation.
        cells: Every cell the run may play.

    Returns:
        The fields and their digest.
    """
    fields: dict[str, Any] = {
        "version": TRAIN_VERSION,
        "config": online.bound_parameters(config),
        "updates": config.updates,
        "model_sha256": model_digest(ModelConfig.current()),
        "runtime_sha256": runtime_digest(),
        "trainer_sha256": hashlib.sha256(
            Path(online.__file__).read_bytes()
        ).hexdigest(),
        "opponents": [
            {"name": name, "path": path, "weight": weight}
            for name, path, weight in OPPONENTS
        ],
        "cells_sha256": canonical_digest([asdict(cell) for cell in cells]),
    }
    return {"fields": fields, "sha256": canonical_digest(fields)}


def reconcile_parameters(
    root: Path, wanted: dict[str, Any], create: bool
) -> dict[str, Any]:
    """Bind an online root to one parameterisation, once.

    Args:
        root: The online root.
        wanted: This run's canonical parameters.
        create: Whether the root is being claimed rather than reopened.

    Returns:
        The stored parameters.

    Raises:
        OnlineParameterError: If a create meets a live root, a resume meets an
            unclaimed one, the stored file fails its own digest, or the stored
            parameters are not this run's.
    """
    path = root / PARAMETERS_NAME
    if create:
        if path.exists():
            raise OnlineParameterError(f"{root} is already claimed; use --resume")
        write_json_atomic(path, wanted)
        return wanted
    if not path.exists():
        raise OnlineParameterError(f"{root} was never created; use --create")
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored["sha256"] != canonical_digest(stored["fields"]):
        raise OnlineParameterError(f"{path} does not match its own digest")
    if stored["fields"] != wanted["fields"]:
        differing = sorted(
            key
            for key in set(stored["fields"]) | set(wanted["fields"])
            if stored["fields"].get(key) != wanted["fields"].get(key)
        )
        raise OnlineParameterError(
            f"{root} was created with different parameters: {differing}"
        )
    return stored


def gate(root: Path, checkpoint: str, workers: int) -> dict[str, Any]:
    """Export the trained head and play it against the bare frozen controller.

    The candidate is the served entrypoint reading the served artifact, so the
    thing measured here is the thing a submission would run. The result is
    published to the root before it is returned, so a gate that is read later
    is read from the run that produced it.

    Args:
        root: The owned online root.
        checkpoint: Which trained head to export.
        workers: Processes to spread the 128 games over.

    Returns:
        The gate document that was written to the root.
    """
    model = MarketResidualNet(ModelConfig.current())
    model.load_state_dict(load_checkpoint(root / checkpoint)["model"])
    model.eval()
    rows = export_rows(root)
    export_numpy_policy(model, ARTIFACT, rows, atol=EXPORT_ATOL)
    LOGGER.info("exported %s from %s over %d rows", ARTIFACT, checkpoint, len(rows))

    scores = arena.outcomes(
        CANDIDATE, {"v56": holdout.SERVED}, holdout.GATE_SEEDS, workers
    )
    rate = sum(scores) / len(scores)
    low, high = wilson_interval(sum(scores), len(scores))
    document = {
        "checkpoint": checkpoint,
        "candidate": CANDIDATE,
        "against": holdout.SERVED,
        "seeds": len(holdout.GATE_SEEDS),
        "gate": {
            "rate": rate,
            "wilson_low": low,
            "wilson_high": high,
            "games": len(scores),
            "wins": sum(1 for score in scores if score == 1.0),
            "ties": sum(1 for score in scores if score == 0.5),
            "mean_margin": sum(scores.margins) / len(scores.margins),
            "median_margin": sorted(scores.margins)[len(scores.margins) // 2],
            "runtime_seconds": scores.runtime_seconds,
        },
        "verdict": verdict(low, high),
    }
    write_json_atomic(root / GATE_NAME, document)
    return document


def verdict(low: float, high: float) -> str:
    """Return what a Wilson interval says about the head, and only that.

    Three outcomes, fixed before any of them existed. An interval that spans
    0.5 is *no measured effect*, not a failure: a policy that converged to
    never deviating plays the frozen controller exactly and lands there by
    construction, and that is a legitimate result about the controller's market
    play rather than a number to be improved.

    Args:
        low: The Wilson lower bound at 95%.
        high: The Wilson upper bound at 95%.

    Returns:
        ``"improves"``, ``"harms"``, or ``"no measured effect"``.
    """
    if low > 0.5:
        return "improves"
    if high < 0.5:
        return "harms"
    return "no measured effect"


def export_rows(root: Path) -> list[MarketFeatureVector]:
    """Return real market feature rows for the export's parity proof.

    Taken from a season this run actually collected rather than replayed for
    the occasion: the rows are already in the root, they are the geometry this
    head will be asked about, and a random tensor would exercise a cancellation
    a mostly-one-hot row never produces.

    Args:
        root: The owned online root.

    Returns:
        One season's rows, in the order one seat produced them.

    Raises:
        OnlineParameterError: If the root holds no collected season.
    """
    path = root / EXPORT_ROWS
    if not path.exists():
        raise OnlineParameterError(f"{root} holds no collected feature rows")
    features = load_checkpoint(path)["features"]
    schema = MarketFeatureSchema.current()
    return [
        MarketFeatureVector(schema=schema, values=tuple(row.tolist()))
        for row in features
    ]


def wandb_run(args: argparse.Namespace) -> Any:  # noqa: ANN401 - a wandb.Run
    """Open a W&B run for this training root.

    Args:
        args: The parsed arguments.

    Returns:
        The run handle.
    """
    import wandb

    return wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"market-online-{args.root.name}",
        config=vars(args),
        resume="allow",
        id=hashlib.sha256(str(args.root.resolve()).encode()).hexdigest()[:16],
    )


if __name__ == "__main__":
    main()
