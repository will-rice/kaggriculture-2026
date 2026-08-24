"""Toad Brigade's phase 1: shaped self-play from random initialisation.

Phase 1 of their five, and the only one that stands alone: it is the sole phase
with ``teacher_kl_cost: 0.``, so it is genuinely teacher-free self-play from
scratch and none of monobeast's teacher machinery is needed to run it. That makes
it the pure baseline the user asked for -- one faithful run, measured, before
anything is improved.

Every hyperparameter comes from ``toad/conf/conv_phase1_shaped_reward.yaml`` and
is named in ``toad_loss``. The loss is ``toad_loss.losses``, which is their
V-trace + UPGO + TD(lambda) composition over the vendored return functions. The
reward is ``toad_reward.shaped``, their five weighted deltas with the 10x
terminal result riding inside from turn one, through their /500 normaliser.

Deviations from monobeast, all deliberate and all recorded in the task report:

* **Synchronous actor/learner** instead of their multiprocessing queues (D7).
  The loss is unchanged, but V-trace only does work when the actor's weights lag
  the learner's, so the actor is an explicit frozen copy resynced every
  ``SYNC_EVERY`` updates rather than whatever lag the queues happened to give
  them. With no lag the importance ratios would be identically 1, V-trace would
  degenerate to TD silently, and the UPGO clipping would never bind.
* **The market head is ours** (D8). Their actor is per-tile and has no analogue
  for a non-spatial trading decision; ours pools the trunk. The unit head keeps
  their structure -- a per-tile readout gathered at each unit's own square.
* **719 decisions against their 360** (D2), which is why the discount is
  0.9995 rather than their 0.999 -- the same retention of a terminal payoff at
  turn 0 over twice the episode; see ``toad_loss.DISCOUNTING``.
* **Nothing rewards selling well** (D1), because Toad had no component that
  priced their transactions either. This is the reproduction's central open
  question and is left standing so the run measures it. What is *not* left out
  is the score itself: their `city` weight is Lux's win condition paid
  incrementally, so ours pays for coins banked -- see `toad_reward.MONEY_WEIGHT`
  and `--no-money`, the ablation that deletes it.
"""

import argparse
import logging
import os
import re
import subprocess
import sys
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import lightning
import torch
from lightning import seed_everything
from lightning.pytorch.loggers import WandbLogger

from kaggriculture.learn import CHECKPOINT
from kaggriculture.learn.critic import critic_scores
from kaggriculture.learn.model import Policy
from kaggriculture.learn.rollout import Trajectory, rollout_many
from kaggriculture.learn.toad.callbacks import (
    ActorSyncCallback,
    BoundaryCheckpoint,
    EnvironmentStepStop,
    PopulationSnapshotCallback,
)
from kaggriculture.learn.toad.compile import maybe_compile
from kaggriculture.learn.toad.config import (
    ModelConfig,
    RuntimeConfig,
    RuntimeMetadata,
    ToadConfig,
    load_config,
)
from kaggriculture.learn.toad.data import (
    ACTED_FIELDS as _ACTED_FIELDS,
)
from kaggriculture.learn.toad.data import (
    OBSERVED_FIELDS as _OBSERVED_FIELDS,
)
from kaggriculture.learn.toad.data import (
    NativeRoundSource,
    ReferenceRoundSource,
    ReferenceWorkerInput,
    ToadDataModule,
    segments,
)
from kaggriculture.learn.toad.lightning import ToadLightningModule
from kaggriculture.learn.toad.model import StatefulPolicy, uses_stateful_policy
from kaggriculture.learn.toad_loss import (
    MIN_LR_MOD,
    TOTAL_STEPS,
    UNROLL_LENGTH,
)
from kaggriculture.learn.toad_reward import (
    MONEY_WEIGHT_ENV,
)
from kaggriculture.learn.toad_reward import (
    money_weight as money_weight,
)

LOGGER = logging.getLogger(__name__)


class RuntimePreflightError(RuntimeError):
    """A requested runtime cannot be honored without changing the experiment."""


def _cuda_available() -> bool:
    """Inspect CUDA availability without selecting a device or creating a context."""
    return torch.cuda.is_available()


def _cuda_device_count() -> int:
    """Inspect CUDA topology without selecting a device or creating a context."""
    return torch.cuda.device_count()


def _cuda_bf16_supported() -> bool:
    """Probe BF16 in a disposable process so the trainer parent stays context-free."""
    try:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch; print(int(torch.cuda.is_bf16_supported()))",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0 and probe.stdout.strip() == "1"


def _effective_accelerator(runtime: RuntimeConfig) -> Literal["cpu", "gpu"]:
    """Resolve Lightning's ``auto`` selection without constructing a Trainer."""
    if runtime.accelerator == "auto":
        return "gpu" if _cuda_available() else "cpu"
    return runtime.accelerator


def _requested_gpu_count(runtime: RuntimeConfig) -> int:
    """Return the requested CUDA device count without selecting a CUDA device."""
    if runtime.devices == "auto":
        return _cuda_device_count()
    if isinstance(runtime.devices, tuple):
        return len(runtime.devices)
    return runtime.devices


def _requested_device_count(runtime: RuntimeConfig, accelerator: str) -> int:
    """Resolve the single-process topology without constructing a strategy."""
    if runtime.devices == "auto":
        return _cuda_device_count() if accelerator == "gpu" else 1
    if isinstance(runtime.devices, tuple):
        return len(runtime.devices)
    return runtime.devices


def _cpu_bf16_supported() -> bool:
    """Ask the installed PyTorch/Lightning stack whether CPU autocast exists."""
    from lightning.pytorch.plugins.precision import MixedPrecision

    try:
        MixedPrecision("bf16-mixed", "cpu")
        return torch.amp.autocast_mode.is_autocast_available("cpu")
    except (RuntimeError, ValueError):
        return False


def _rollout_preflight(config: ToadConfig) -> None:
    """Reject rollout modes that would otherwise fall back or be ignored."""
    runtime = config.runtime
    if (
        runtime.rollout_backend == "native"
        and config.population.scripted > 0
        and config.population.scripted_opponent != "economic"
    ):
        raise RuntimePreflightError(
            "native scripted rollout supports only the verified 'economic' opponent"
        )
    if runtime.rollout_backend == "native" and runtime.compile.enabled:
        raise RuntimePreflightError(
            "native rollout with torch.compile is not proved; refusing eager fallback"
        )
    if runtime.rollout_cuda_graph:
        if runtime.rollout_backend != "native":
            raise RuntimePreflightError(
                "CUDA-graph rollout requires rollout_backend='native'"
            )
        if config.population.scripted > 0:
            raise RuntimePreflightError(
                "scripted CUDA-graph rollout is unsupported because the scripted "
                "opponent reads simulator rows on the host"
            )
        raise RuntimePreflightError(
            "CUDA-graph round collection is not yet selectable; refusing eager fallback"
        )
    if runtime.rollout_backend == "reference" and runtime.rollout_device != "cpu":
        raise RuntimePreflightError(
            "reference rollout does not consume rollout_device; refusing an ignored "
            f"{runtime.rollout_device!r} request"
        )
    if (
        runtime.rollout_backend == "native"
        and runtime.rollout_device == "cuda"
        and not _cuda_available()
    ):
        raise RuntimePreflightError(
            "native CUDA rollout requested but CUDA is unavailable"
        )


def _gpu_runtime_preflight(runtime: RuntimeConfig) -> None:
    """Reject an unavailable explicit GPU or BF16 placement."""
    if not _cuda_available():
        if runtime.precision == "bf16-mixed":
            raise RuntimePreflightError(
                "bf16-mixed requested but CUDA BF16 is unavailable"
            )
        raise RuntimePreflightError("GPU accelerator requested but CUDA is unavailable")
    requested = _requested_gpu_count(runtime)
    available_devices = _cuda_device_count()
    if isinstance(runtime.devices, tuple) and len(set(runtime.devices)) != len(
        runtime.devices
    ):
        raise RuntimePreflightError(
            f"requested GPU device indexes must be unique: devices={runtime.devices}"
        )
    if requested < 1 or requested > available_devices:
        raise RuntimePreflightError(
            "requested GPU devices are unavailable: "
            f"requested={requested}, available={available_devices}"
        )
    if isinstance(runtime.devices, tuple) and any(
        device >= available_devices for device in runtime.devices
    ):
        raise RuntimePreflightError(
            "requested GPU device index is unavailable: "
            f"available={available_devices}, devices={runtime.devices}"
        )
    if runtime.precision == "bf16-mixed" and not _cuda_bf16_supported():
        raise RuntimePreflightError("bf16-mixed requested but CUDA BF16 is unavailable")


def runtime_preflight(config: ToadConfig) -> None:
    """Reject unavailable runtime requests before creating trainer side effects.

    This deliberately performs only hardware capability inspection.  It does
    not construct a Trainer, logger, worker, file, or policy, and it never
    changes an explicit request into a different precision or accelerator.
    """
    runtime = config.runtime
    _rollout_preflight(config)
    accelerator = _effective_accelerator(runtime)
    if accelerator == "cpu" and isinstance(runtime.devices, tuple):
        raise RuntimePreflightError(
            "CPU accelerator does not accept explicit device indexes"
        )
    world_size = runtime.num_nodes * _requested_device_count(runtime, accelerator)
    if world_size > 1 and runtime.strategy != "ddp":
        raise RuntimePreflightError(
            "resolved world size above one requires DDP strategy='ddp': "
            f"world_size={world_size}, strategy={runtime.strategy!r}"
        )
    if accelerator == "gpu":
        _gpu_runtime_preflight(runtime)
        return
    if runtime.precision == "bf16-mixed" and not _cpu_bf16_supported():
        raise RuntimePreflightError(
            "bf16-mixed requested but CPU BF16 autocast is unavailable"
        )


def runtime_metadata(config: ToadConfig) -> RuntimeMetadata:
    """Return immutable resolved identity for logging this trainer run."""
    accelerator = _effective_accelerator(config.runtime)
    return RuntimeMetadata(
        precision=config.runtime.precision,
        compile=config.runtime.compile,
        world_size=(
            config.runtime.num_nodes
            * _requested_device_count(config.runtime, accelerator)
        ),
        rollout_backend=config.runtime.rollout_backend,
    )


# Temporary aliases preserve the runner's original segmentation surface while
# callers move to ``kaggriculture.learn.toad.data.segments``.
ACTED_FIELDS = _ACTED_FIELDS
OBSERVED_FIELDS = _OBSERVED_FIELDS


@dataclass(frozen=True)
class Teacher:
    """The frozen checkpoint the KL pulls toward, and what it carried.

    The second field is the whole reason this is a pair rather than a bare
    ``Policy``. Every ``Policy`` object has a ``quantity_head``, freshly
    initialised if nothing loaded into it, so the network cannot be asked
    whether it was ever taught one -- only the state dict it came from can
    answer, and that answer is known exactly once, at load time, where
    ``load_policy_weights`` reports the keys it did not find. Carrying the two
    together makes the pairing impossible to get wrong downstream: there is no
    way to hand ``compute_loss`` a teacher without also telling it which heads
    that teacher is entitled to constrain.

    Attributes:
        policy: The frozen network, in eval mode with gradients off.
        quantity: Whether the checkpoint it was loaded from carried the
            quantity head. False means the head is a random initialisation and
            a KL toward it would be a pull toward noise, so the divergence
            covers the op and market heads alone.
    """

    policy: Policy
    quantity: bool


SEED = 0
# conv_phase1_shaped_reward.yaml:36. Four unrolls per learner batch, so one
# optimizer step sees 16 x 4 = 64 transitions. Collecting a whole round and
# stepping once on all of it would be a different algorithm -- and, measured, it
# also OOMs: 2,112 segments is a 34k-row forward.
BATCH_SEGMENTS = 4
# conv_phase1_shaped_reward.yaml: n_blocks 8, hidden_dim 128. Our current PPO
# network is 8 blocks at 256 channels; the width is theirs, not ours.
BLOCKS = 8
CHANNELS = 128
# toad/nns/models.py:157-160 and :181. StatefulMultiReward's spec is
# +/-1/MAX_DAYS and its `only_once` is False, so BaselineLayer expands the range
# by MAX_DAYS: the value head is confined to [-1, +1]. Ours was unbounded, and
# that alone diverged the first run.
VALUE_BOUND = 1.0
# Which reward the learner trains on. "shaped_money" is the faithful component
# set: their five, with the score constituent in the slot their `city` weight
# occupies -- in Lux `city` IS the score, and here the score is coins banked.
# "shaped" is the same set with that component deleted, which is the ablation
# (--no-money) rather than the baseline. Both are recorded on every trajectory,
# so switching this constant is the entire diff between the two arms and they
# share seeds, episodes and actions.
REWARD_FIELD = "shaped_money"
# Episodes per collection round. Their n_actor_envs is 16 across 2 actors; ours
# is one synchronous group, and both seats of a self-play episode are recorded.
ENVIRONMENTS = 24
# Decisions in a season. The engine runs 720 turns and the last one takes no
# action, so a recorded seat is 719 rows.
TURNS = 719
# The engine is single-threaded Python and is the bottleneck -- measured, the
# rollout is env-bound rather than network-bound, so the GPU does not fix it.
# Collection is spread over processes the way `selfplay.collect` does it.
#
# One per environment. ``ReferenceRoundSource`` fans the typed assignments out
# while preserving the round quota. ENVIRONMENTS is deliberately not changed
# with it: that would move the effective batch and therefore the recipe.
WORKERS = ENVIRONMENTS
# Torch threads per worker. One, so WORKERS processes do not each claim the
# whole machine; see ``_play_reference``.
THREADS = 1
# How many updates the actor's weights lag the learner's. See D7 above.
SYNC_EVERY = 4
OPPONENT = "src/kaggriculture/economic_policy.py"
# What a segment carries, split by how many rows of it there are. The acted
# fields have one row per decision. The observed fields have one more, because
# the value target bootstraps from the state *after* the segment's last action
# and not from that action's own state; see `_segments` for why the difference
# is the whole ballgame.
RUNS = Path("/data/kaggriculture/toad")
# Every 25 updates is ~13 minutes of work at the measured 3.97M steps/hour.
# Attempt one had none, and an external kill at update 253 cost 2.3 hours.
CHECKPOINT_EVERY = 25
WANDB_ENTITY = "will-rice"
WANDB_PROJECT = "kaggriculture-2026"
# wandb groups panels into dashboard sections by the text before the first
# `/`, so every key `_record` returns must start with one of these. That is
# the entire mechanism THE INCIDENT asks for: a reader sees which role a
# number plays before they read its name. `objective/` is what we actually
# want (margin and win rate against the scripted opponent); `proxy/` is what
# the gradient reads (the shaped reward and its components); `critic/` is
# value-head diagnostics; `diag/` is everything else -- population sizes,
# loss terms, bank, sale mechanics -- none of which can indicate capability
# on its own. See test_every_logged_metric_carries_a_role_prefix.
OBJECTIVE = "objective/"
PROXY = "proxy/"
CRITIC = "critic/"
DIAG = "diag/"
METRIC_PREFIXES = (OBJECTIVE, PROXY, CRITIC, DIAG)
# One line per logged key, shipped in the logger config so the meaning of a
# number sits beside the number instead of requiring someone to open this
# file. Derived from the comments at each metric's definition site in
# `_record`, `_population` and `_critic`; kept in sync with them by
# `test_every_logged_metric_carries_a_role_prefix`, which asserts this dict's
# keys are exactly the keys `_record` returns.
METRIC_DEFINITIONS: dict[str, str] = {
    "diag/update": "Optimizer rounds completed so far.",
    "diag/steps": "Environment decisions collected so far.",
    "diag/hours": "Wall-clock hours since the run started.",
    "diag/bank_mean": (
        "Mean terminal bank, both populations pooled. Corpus mining over "
        "1,350 seats put bank against ladder rating at Pearson -0.043, so "
        "this is a diagnostic, not a capability read."
    ),
    "diag/bank_max": (
        "Max terminal bank, both populations pooled; same caveat as diag/bank_mean."
    ),
    "diag/bank_mirror": (
        "Mean terminal bank, self-play seats only. Kept separate from "
        "diag/bank_vs_econ on purpose -- conflating the two read as "
        "capability for 209 updates on arm C''."
    ),
    "diag/bank_vs_econ": "Mean terminal bank, scripted-opponent seats only.",
    "diag/n_econ_envs": "Scripted-opponent trajectories collected this round.",
    "objective/win_rate_vs_econ": (
        "Fraction of scripted-opponent games with final_margin > 0 -- the "
        "competition's own statement of the win condition, undecomposed. "
        "Falsified by staying 0.000 while proxy/ and diag/ metrics rise."
    ),
    "diag/mirror_decisive_rate": (
        "2x the self-play win rate (final_margin > 0), rescaled from its "
        "raw 0-0.5 range: both self-play seats are recorded and exactly one "
        "wins, so the raw rate is 0.5 by construction and only rises when "
        "the policy stops tying with itself. Reaches 1.0 regardless of "
        "whether play is any good."
    ),
    "objective/margin_vs_econ": (
        "Mean final_margin (our bank minus theirs) against the scripted "
        "opponent -- the objective itself."
    ),
    "diag/margin_mean_mirror": (
        "Mean final_margin in self-play; near-deterministic and not "
        "comparable to objective/margin_vs_econ."
    ),
    "diag/mean_sale_price_vs_econ": (
        "Coins per unit realised on sale, scripted-opponent seats. Corpus "
        "mining says winners take +3.3% here at flat volume."
    ),
    "diag/realisation_vs_econ": (
        "Realised price over the market book (1.0 = par), scripted-opponent seats."
    ),
    "diag/sales_vs_econ": (
        "Completed clears per episode, scripted-opponent seats -- the "
        "volume control on diag/mean_sale_price_vs_econ."
    ),
    "diag/units_sold_vs_econ": "Units moved by those clears, scripted-opponent seats.",
    "diag/bought_vs_econ": "Units purchased, scripted-opponent seats.",
    "diag/final_capital_vs_econ": (
        "Producing animals owned at the horizon, scripted-opponent seats. "
        "If margin improves while this and bank both collapse, the "
        "absolute reward term is too weak."
    ),
    "diag/mean_sale_price_mirror": "Coins per unit realised on sale, self-play seats.",
    "diag/realisation_mirror": (
        "Realised price over the market book (1.0 = par), self-play seats."
    ),
    "diag/sales_mirror": "Completed clears per episode, self-play seats.",
    "diag/units_sold_mirror": "Units moved by those clears, self-play seats.",
    "diag/bought_mirror": "Units purchased, self-play seats.",
    "diag/final_capital_mirror": (
        "Producing animals owned at the horizon, self-play seats."
    ),
    "critic/ev_econ_games": (
        "Pooled explained variance of the actor's value estimates against "
        "the real discounted return, scripted-opponent seats. A critic "
        "that has learned only the turn clock scores ~0.98 here."
    ),
    "critic/ev_within_turn_econ_games": (
        "Explained variance after centring both series at each turn index, "
        "scripted-opponent seats -- whether the critic tells two episodes "
        "apart at the same point in the season, not just the calendar."
    ),
    "critic/ev_mirror_games": (
        "Pooled explained variance against the real return, self-play "
        "seats; a mirror seat's return is very nearly deterministic, so "
        "this is not comparable to critic/ev_econ_games."
    ),
    "critic/ev_within_turn_mirror_games": (
        "Within-turn explained variance, self-play seats; same "
        "non-comparability to critic/ev_within_turn_econ_games."
    ),
    "proxy/shaped_mean": (
        "Episode total of Toad's shaped reward, logged unconditionally as "
        "the counterfactual -- not the objective, and not necessarily what "
        "this arm trains on."
    ),
    "proxy/shaped_reward_mean": (
        "Episode total of whichever reward field this arm actually trains "
        "on (REWARD_FIELD, or --no-money / --margin). Rising while "
        "objective/win_rate_vs_econ stays flat is shaped credit, not "
        "progress -- about three-quarters of a typical value here is "
        "shaping, not coins."
    ),
    "diag/illegal": (
        "Stored action indices the stored mask forbade. Nonzero means "
        "sampling or storage is broken."
    ),
    "diag/gross_purchases": (
        "Gross coins spent buying, mean per episode. Rising alongside "
        "proxy/money_term without diag/bank_mean moving is the money-pump "
        "signature. The delta is signed, so a losing round trip cannot earn "
        "shaped reward; this watches for the clamp coming back."
    ),
    "proxy/money_term": (
        "The money component's own realised contribution to the shaped "
        "reward (shaped_money - shaped). Logged on both arms: the thing "
        "being paid for on the faithful arm, the counterfactual on "
        "--no-money."
    ),
    "diag/lr": "Current learning rate off the LambdaLR schedule.",
    "diag/warming": (
        "True while the value head is training alone (--value-warmup), "
        "before the policy gradient fires."
    ),
    "diag/warmup_left": "Batches still owed to that value-only warmup.",
    "diag/vtrace_pg": (
        "V-trace policy-gradient loss term. A loss term, not a capability signal."
    ),
    "diag/upgo_pg": "UPGO policy-gradient loss term.",
    "critic/baseline_self_consistency": (
        "Value loss against the round's own bootstrapped TD(lambda) "
        "target -- distance from its OWN target, not from the real "
        "return. A critic can sit at 0.998 explained variance against its "
        "own target while explaining the real return twelvefold worse "
        "than a constant; ten earlier arms read this falling as the "
        "critic learning. See critic/ev_econ_games for the real accuracy."
    ),
    "diag/entropy": (
        "Negative-entropy loss term; a falling value means the policy is "
        "collapsing onto fewer actions, not necessarily better ones."
    ),
    "diag/teacher_kl": "Teacher KL loss term, zero unless --teacher is set.",
    "diag/total_loss": "Sum of every loss term actually optimized this round.",
    "critic/baseline_passes_self_consistency": (
        "The same self-consistency loss as critic/baseline_self_consistency, "
        "averaged over the extra --value-passes replays, or equal to it "
        "when there are none."
    ),
}


def _parser() -> argparse.ArgumentParser:
    """Return the one-release legacy command line translator's parser.

    Production execution uses :func:`parse_config`. This parser remains for one
    release so existing invocations and helper-level tests can resolve their
    old flags into the same immutable :class:`ToadConfig` passed to
    :func:`run`. Defaults come from the control config or the current constants
    rather than a separately authoritative table.

    Returns:
        The argument parser ``main`` parses ``sys.argv`` with.
    """
    control = ToadConfig.control()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-money",
        action="store_true",
        help="delete the score constituent from the shaped reward, leaving "
        "Toad's Lux-specific five. The control this reproduction is measured "
        "against; the two arms must differ by this flag and nothing else.",
    )
    parser.add_argument(
        "--margin",
        action="store_true",
        help="train on the margin reward -- the competition's actual win "
        "condition -- instead of Toad's shaped components. Mutually exclusive "
        "with --no-money, --own and --sparse.",
    )
    parser.add_argument(
        "--own",
        action="store_true",
        help="train on this seat's own bank alone -- absolute rather than "
        "relative, and the thinnest possible shaping. Mutually exclusive with "
        "--no-money, --margin and --sparse.",
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="train on GameResultReward -- +1/-1/0 on the terminal turn alone, "
        "nothing else. Phases 2-5 of the curriculum. Mutually exclusive with "
        "--no-money, --margin and --own.",
    )
    parser.add_argument(
        "--money-weight",
        type=float,
        default=None,
        help="override the money component's weight (arm W uses 0.01). Set into "
        "the environment so the rollout workers see it too.",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=control.model.blocks,
        help="residual blocks in the learner's (and rollout actor's) trunk. "
        "Phases 3-5 of the curriculum need 16 and 24 against phase 1-2's 8.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=control.model.channels,
        help="trunk width. Arm C needs 256 to match the BC clone, against "
        "Toad's hidden_dim of 128 -- a declared deviation, not a tuning knob.",
    )
    parser.add_argument(
        "--teacher",
        type=Path,
        default=None,
        help="checkpoint to hold the policy near with monobeast's teacher KL "
        "(--teacher-kl-cost). Unset by default: phase 1 is teacher-free "
        "self-play from scratch. The recipe's teachers are always an earlier, "
        "smaller checkpoint from this same pipeline, never a behaviour clone.",
    )
    parser.add_argument(
        "--teacher-blocks",
        type=int,
        default=control.model.blocks,
        help="residual blocks in the frozen teacher's trunk -- the depth its "
        "own checkpoint was written at, not this arm's --blocks. The recipe's "
        "teachers are always smaller nets than the phase they teach (phase 5's "
        "is the 16-block checkpoint against its own 24), so this cannot be "
        "assumed equal to --blocks.",
    )
    parser.add_argument(
        "--teacher-quantity",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="declare whether the named legacy teacher has a complete quantity "
        "head; required with --teacher and translated into TeacherSpec",
    )
    parser.add_argument(
        "--econ-fraction",
        type=float,
        default=control.population.scripted,
        help="fraction of each round's environments played against "
        "economic_policy instead of the mirror. Only our seat trains from "
        "those: the scripted agent is market pressure, not a tape to clone.",
    )
    parser.add_argument(
        "--teacher-kl-cost",
        type=float,
        default=control.optimizer.teacher_kl_cost,
        help="teacher KL weight. Their cascade drops it to 0.001 at phase 3, "
        "where the policy should start out-earning its teacher.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=control.optimizer.lr,
        help="Adam learning rate the schedule decays from. Sweep knob; "
        "the default reproduces every earlier arm exactly.",
    )
    parser.add_argument(
        "--entropy-cost",
        type=float,
        default=control.optimizer.entropy_cost,
        help="coefficient on the entropy loss term. Sweep knob; the default "
        "reproduces every earlier arm exactly.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=control.optimizer.gamma,
        help="discount the return and advantage targets are built at "
        "(``losses``' ``discounting``). Sweep knob; the default reproduces "
        "every earlier arm exactly.",
    )
    parser.add_argument(
        "--lmb",
        type=float,
        default=control.optimizer.lmb,
        help="lambda for both TD(lambda) and UPGO, named to match "
        "toad_loss.losses' own parameter rather than shadowing the Python "
        "keyword. Their 0.8 for phases 1-4 and 0.9 for phase 5.",
    )
    parser.add_argument(
        "--value-warmup-batches",
        type=int,
        default=control.optimizer.value_warmup_batches,
        help="train the value head alone for this many batches before the "
        "policy gradient fires (arm C'). Their phase-2 mechanism, for a "
        "warm-started policy whose critic is untrained. 0 means no warmup.",
    )
    parser.add_argument(
        "--value-passes",
        type=int,
        default=control.optimizer.value_passes,
        help="extra value-only passes over each round after the policy's one "
        "(arm S uses 4). A round is otherwise seen once and discarded, and the "
        "ceiling experiment put a critic trained 150 times over the same data at "
        "+0.276 within-step where the live arm that produced it sat at -0.103. "
        "Zero reproduces every earlier arm exactly.",
    )
    parser.add_argument(
        "--clone-init",
        action="store_true",
        help="warm-start trunk and both heads from the BC clone instead of "
        "random init (arm C). The bounded value head is always fresh.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="wandb run name and file prefix, so arms cannot collide",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="checkpoint to continue from, restoring weights, optimizer, "
        "schedule and counters",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=control.runtime.total_environment_steps,
        help="environment steps this arm trains for. The curriculum's five "
        "phases each need a different budget (2e7-1e7) against this "
        "constant's declared-deviation 1e8.",
    )
    return parser


def _config_parser() -> argparse.ArgumentParser:
    """Return the primary typed-config command line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=JSON_VALUE",
    )
    return parser


def parse_config(argv: Sequence[str] | None = None) -> ToadConfig:
    """Resolve and validate the primary config CLI without runtime side effects."""
    arguments = _config_parser().parse_args(argv)
    return load_config(arguments.config, arguments.overrides)


def _legacy_config(argv: Sequence[str]) -> ToadConfig:
    """Translate the former flags into the native immutable config contract."""
    arguments = _parser().parse_args(argv)
    reward_field = _field(arguments)
    if arguments.name is not None and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", arguments.name
    ):
        raise ValueError("legacy --name must be a safe run name")
    if arguments.teacher is not None and arguments.teacher_quantity is None:
        raise ValueError(
            "legacy --teacher requires an explicit teacher quantity compatibility "
            "flag (--teacher-quantity or --no-teacher-quantity)"
        )
    if arguments.teacher is None and arguments.teacher_quantity is not None:
        raise ValueError("teacher quantity compatibility requires --teacher")

    control = ToadConfig.control()
    payload = control.model_dump(mode="python")
    payload["model"].update(blocks=arguments.blocks, channels=arguments.channels)
    payload["population"].update(
        selfplay=1.0 - arguments.econ_fraction,
        scripted=arguments.econ_fraction,
        teacher=(
            {
                "checkpoint": arguments.teacher,
                "blocks": arguments.teacher_blocks,
                "quantity": arguments.teacher_quantity,
            }
            if arguments.teacher is not None
            else None
        ),
    )
    payload["optimizer"].update(
        lr=arguments.lr,
        gamma=arguments.gamma,
        lmb=arguments.lmb,
        entropy_cost=arguments.entropy_cost,
        teacher_kl_cost=arguments.teacher_kl_cost,
        value_warmup_batches=arguments.value_warmup_batches,
        value_passes=arguments.value_passes,
    )
    payload["runtime"].update(
        total_environment_steps=arguments.total_steps,
        resume=arguments.resume,
        output_dir=(
            control.runtime.output_dir / arguments.name
            if arguments.name is not None
            else control.runtime.output_dir
        ),
    )
    payload["curriculum"].update(
        phase=arguments.name or control.curriculum.phase,
        reward_field=reward_field,
        warm_start_checkpoint=CHECKPOINT if arguments.clone_init else None,
        money_weight=arguments.money_weight
        if arguments.money_weight is not None
        else control.curriculum.money_weight,
    )
    return ToadConfig.model_validate(payload)


def _uses_legacy_cli(argv: Sequence[str]) -> bool:
    """Return whether ``argv`` contains a flag outside the primary surface."""
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument.startswith("--config=") or argument.startswith("--set="):
            index += 1
            continue
        if argument in {"-h", "--help"}:
            index += 1
            continue
        if argument in {"--config", "--set"}:
            index += 2
            continue
        return True
    return False


def build_wandb_logger(config: ToadConfig) -> WandbLogger:
    """Build the sole tracking surface from the complete resolved config."""
    logged_config = config.model_dump(mode="json")
    logged_config["runtime_metadata"] = runtime_metadata(config).model_dump(mode="json")
    logged_config["metric_definitions"] = METRIC_DEFINITIONS
    return WandbLogger(
        project=WANDB_PROJECT,
        name=config.curriculum.phase,
        save_dir=str(config.runtime.output_dir),
        config=logged_config,
    )


def build_reference_data_module(config: ToadConfig) -> ToadDataModule:
    """Build the requested synchronous collector behind the common bridge."""
    source: ReferenceRoundSource
    if config.runtime.rollout_backend == "native":
        source = NativeRoundSource(config)
    else:
        source = ReferenceRoundSource(config)
    return ToadDataModule(config, source)


def build_trainer(config: ToadConfig) -> lightning.Trainer:
    """Build the requested Lightning Trainer with round-boundary ownership."""
    devices: int | str | list[int] = (
        list(config.runtime.devices)
        if isinstance(config.runtime.devices, tuple)
        else config.runtime.devices
    )
    return lightning.Trainer(
        accelerator=config.runtime.accelerator,
        devices=devices,
        num_nodes=config.runtime.num_nodes,
        strategy=config.runtime.strategy,
        precision=config.runtime.precision,
        deterministic=config.runtime.deterministic,
        benchmark=config.runtime.benchmark,
        profiler=config.runtime.profiler,
        log_every_n_steps=config.runtime.log_every_n_steps,
        gradient_clip_val=config.optimizer.clip_grad_norm,
        gradient_clip_algorithm="norm",
        max_steps=-1,
        max_epochs=-1,
        use_distributed_sampler=False,
        enable_checkpointing=False,
        callbacks=[
            ActorSyncCallback(config.population.actor_sync_every_rounds),
            EnvironmentStepStop(config.runtime.total_environment_steps),
            PopulationSnapshotCallback(),
            BoundaryCheckpoint(config.runtime.output_dir),
        ],
        logger=build_wandb_logger(config),
    )


def run(config: ToadConfig) -> None:
    """Seed once and hand the complete native control path to Lightning."""
    runtime_preflight(config)
    seed_everything(config.runtime.seed, workers=True)
    module = ToadLightningModule(config)
    effective = module.config
    if effective.runtime.compile.enabled:
        module.policy = maybe_compile(module.policy, effective)
    data = build_reference_data_module(effective)
    build_trainer(effective).fit(
        module,
        datamodule=data,
        ckpt_path=effective.runtime.resume,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Validate one config, then run it exclusively through Lightning.

    Args:
        argv: Primary ``--config``/``--set`` arguments or one-release legacy
            flags. ``None`` reads ``sys.argv[1:]``.
    """
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if _uses_legacy_cli(arguments):
        config = _legacy_config(arguments)
        message = (
            "legacy Toad flags are deprecated for one release; use --config "
            "PATH plus repeatable --set PATH=JSON_VALUE"
        )
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        LOGGER.warning(message)
    else:
        config = parse_config(arguments)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(config)


def _record(
    mirror_batch: list[Trajectory],
    econ_batch: list[Trajectory],
    field: str,
    *,
    update: int,
    steps: int,
    hours: float,
    lr: float,
    warming: bool,
    warmup_left: int,
    terms: dict[str, float],
) -> dict[str, object]:
    """Return one update's metrics, with the two populations kept apart.

    THE SEPARATION IS THE POINT. C'' self-played at a mean bank of ~50 while
    banking 8 against economic_policy, and the conflated number read as
    capability for 209 updates. So every population-dependent metric is
    suffixed, and the win rate that decides this arm is the scripted one alone.

    Every key returned here starts with one of ``METRIC_PREFIXES`` --
    ``objective/`` for margin and win rate against the scripted opponent,
    ``proxy/`` for the shaped reward the gradient actually reads, ``critic/``
    for value-head diagnostics, and ``diag/`` for everything else, which
    cannot indicate capability on its own. That grouping is what lets wandb's
    panel sections answer "what role does this number play" before its name is
    read, which is the whole fix for THE INCIDENT this function's tests are
    named after.

    Args:
        mirror_batch: This round's self-play trajectories, both seats.
        econ_batch: This round's scripted-opponent trajectories, our seat only.
        field: The reward series the learner read.
        update: Optimizer rounds so far.
        steps: Environment steps so far.
        hours: Wall clock since the run started.
        lr: The schedule's current learning rate.
        warming: Whether the value head is still training alone.
        warmup_left: Batches still owed to that warmup.
        terms: The loss terms from this round.

    Returns:
        The record written to the jsonl and to wandb.
    """
    batch = mirror_batch + econ_batch
    banks = [t.final_bank for t in batch]
    # THE NUMBER THAT DECIDES THIS ARM. "Winning the match (having the most
    # coins in the bank at the end of 720 turns)" is the competition's own
    # statement of the objective, and `final_margin` is exactly that
    # quantity, so a win is `final_margin > 0` and nothing needs deriving.
    #
    # Read off the SCRIPTED population alone. The mirror rate is 0.5 by
    # construction -- both seats of a self-play episode are recorded and one
    # of them wins -- so averaging the two populations would produce a
    # number that starts at 0.25 and looks like progress it has not made.
    # This is the same conflation that made the mirror bank read as
    # capability for 209 updates.
    #
    # Bank is now a diagnostic. Corpus mining over 1,350 seats put bank
    # against ladder rating at Pearson -0.043; seats banking over 160k
    # averaged rating 2,828 against 3,030 for seats banking under 60k.
    mirror_banks = [t.final_bank for t in mirror_batch] or [float("nan")]
    econ_banks = [t.final_bank for t in econ_batch] or [float("nan")]
    record = {
        "diag/update": update,
        "diag/steps": steps,
        "diag/hours": hours,
        "diag/bank_mean": sum(banks) / len(banks),
        "diag/bank_max": max(banks),
        "diag/bank_mirror": sum(mirror_banks) / len(mirror_banks),
        "diag/bank_vs_econ": sum(econ_banks) / len(econ_banks),
        "diag/n_econ_envs": len(econ_batch),
        "objective/win_rate_vs_econ": _mean(
            [float(t.final_margin > 0.0) for t in econ_batch]
        ),
        # win_rate_mirror was 0.5 by construction -- both seats of a
        # self-play episode are recorded and one of them wins -- so it rose
        # 0.25 -> 0.50 purely from the policy no longer tying with itself at
        # zero, then pinned at its ceiling for 300+ updates, and read as
        # progress the whole time. Rescaled x2 here: 1.0 now means every
        # mirror game was decisive, a real 0-1 scale instead of a 0-0.5 one.
        # Still not capability -- see METRIC_DEFINITIONS.
        "diag/mirror_decisive_rate": 2.0
        * _mean([float(t.final_margin > 0.0) for t in mirror_batch]),
        "objective/margin_vs_econ": _mean([t.final_margin for t in econ_batch]),
        "diag/margin_mean_mirror": _mean([t.final_margin for t in mirror_batch]),
        # Where the corpus says the edge actually lives: winners in paired
        # same-episode comparisons take +3.3% on mean sale price at FLAT
        # volume. Volume is logged beside price so a price rise bought by
        # simply selling less is visible rather than inferred.
        **_population(econ_batch, "vs_econ"),
        **_population(mirror_batch, "mirror"),
        # WHAT THE `critic/baseline_self_consistency` TERM DOES NOT SAY. That
        # term is the value head's distance from its OWN bootstrapped target,
        # so it measures self-consistency; a critic can sit at 0.998 explained
        # variance against its own target while explaining the real return
        # twelvefold WORSE than a constant, and ten arms read the falling term
        # as the critic learning. The `critic/ev_*` keys are the accuracy,
        # against the actual discounted return-to-go of the episodes just
        # played, and a run whose baseline falls while these stay negative has
        # not trained a critic. Split by population for the reason everything
        # else here is: a mirror seat's return is very nearly deterministic,
        # so the two are not comparable numbers.
        **_critic(econ_batch, field, "vs_econ"),
        **_critic(mirror_batch, field, "mirror"),
        "proxy/shaped_mean": float(torch.stack([t.shaped.sum() for t in batch]).mean()),
        # The episode total of the series the learner is actually reading.
        # `proxy/shaped_mean` is logged unconditionally and is the
        # counterfactual on this arm, not the objective -- reading either of
        # these as progress would be watching the wrong curve entirely.
        "proxy/shaped_reward_mean": float(
            torch.stack([getattr(t, field).sum() for t in batch]).mean()
        ),
        "diag/illegal": sum(t.illegal for t in batch),
        # The money component's own realised contribution, separable
        # because both rewards ride on every trajectory. Logged for BOTH
        # arms: on the baseline it is the counterfactual, on phase-1b it is
        # the thing being paid for. Rising here while bank_mean stays flat
        # is the money-pump signature -- a clamped delta means a buy-then-
        # sell round trip that loses coins still earns shaped reward -- and
        # this project has hit that failure three times. It must be visible
        # in the charts from update 1, not reconstructed afterwards.
        # Gross coins spent per episode. If the pump fires, buy volume
        # rises alongside proxy/money_term -- the two together separate
        # "learned to trade" from "learned to churn".
        "diag/gross_purchases": float(
            torch.stack([(-t.own.clamp(max=0.0)).sum() for t in batch]).mean()
        ),
        "proxy/money_term": float(
            torch.stack([(t.shaped_money - t.shaped).sum() for t in batch]).mean()
        ),
        "diag/lr": lr,
        # True while the value head is training alone, so the warmup
        # window is readable off the data rather than inferred from a
        # batch count.
        "diag/warming": warming,
        "diag/warmup_left": warmup_left,
        **_prefixed_terms(terms),
    }
    return record


def _collection_metrics(
    mirror_batch: list[Trajectory], econ_batch: list[Trajectory], field: str
) -> dict[str, float | int]:
    """Return the legacy stable metrics available immediately after collection."""
    terms = {
        "vtrace_pg": 0.0,
        "upgo_pg": 0.0,
        "baseline": 0.0,
        "baseline_passes": 0.0,
        "entropy": 0.0,
        "teacher": 0.0,
        "total": 0.0,
    }
    record = _record(
        mirror_batch,
        econ_batch,
        field,
        update=0,
        steps=0,
        hours=0.0,
        lr=0.0,
        warming=False,
        warmup_left=0,
        terms=terms,
    )
    dynamic = {
        "diag/update",
        "diag/steps",
        "diag/hours",
        "diag/lr",
        "diag/warming",
        "diag/warmup_left",
        *(_TERM_PREFIX.values()),
    }
    return {
        name: cast(float | int, value)
        for name, value in record.items()
        if name not in dynamic
    }


def _mean(values: list[float]) -> float:
    """Return the mean, or NaN for an empty population.

    An arm with ``--econ-fraction 0`` has no scripted episodes at all, and a
    zero would be a *reading* -- "we win none of them" -- rather than the
    absence of one. NaN is the honest value and wandb draws no point for it.
    """
    return sum(values) / len(values) if values else float("nan")


def _population(batch: list[Trajectory], population: str) -> dict[str, float]:
    """Return one population's per-episode outcomes, suffixed by which it is.

    Computed every update rather than every Nth, because it is free: measured
    against a full 17.5-second episode, building the 719 snapshots the sale
    scan reads costs 0.6 ms and scanning them costs 0.3 ms, which together is
    0.01% of collection.

    ``final_capital`` is here because it is half of a pre-registered read: if
    the margin improves while capital and bank both collapse, the absolute term
    is too weak and that is to be reported rather than tuned around. A read
    nobody can take is not pre-registered, so the number it needs is logged
    from update one rather than reconstructed afterwards.

    None of these can indicate capability on their own -- price, volume and
    capital are mechanics, not the win condition -- so every key returned here
    carries ``DIAG``.

    Args:
        batch: That population's trajectories, possibly empty.
        population: ``"vs_econ"`` or ``"mirror"``, appended to every key.

    Returns:
        Sale price, realisation, volume, purchases and end-of-season capital
        for the population, keyed so the two never merge.
    """
    return {
        f"{DIAG}{name}_{population}": _mean([getattr(t, name) for t in batch])
        for name in (
            "mean_sale_price",
            "realisation",
            "sales",
            "units_sold",
            "bought",
            "final_capital",
        )
    }


# critic_scores' own key -> the short name it takes in the logged, prefixed
# key. Kept distinct from `critic_scores`' names because `_POPULATION_GAMES`
# below reads more naturally as "games" than as the "vs_econ"/"mirror" suffix
# `_population` uses -- the two functions suffix independently on purpose.
_CRITIC_SHORT_NAMES = {
    "ev_vs_return": "ev",
    "ev_vs_return_within_turn": "ev_within_turn",
}
_POPULATION_GAMES = {"vs_econ": "econ_games", "mirror": "mirror_games"}


def _critic(batch: list[Trajectory], field: str, population: str) -> dict[str, float]:
    """Return one population's critic accuracy, suffixed by which it is.

    The values are the ones the ACTOR emitted while playing, so they lag the
    learner by up to ``SYNC_EVERY`` updates. That is the honest reading -- it is
    the critic that actually shaped this round's advantages -- and it costs
    nothing, since a rollout already recorded them and the return is one
    backward scan over rewards the trajectory is already carrying.

    Args:
        batch: That population's trajectories, possibly empty.
        field: The reward series the learner reads, so the critic is scored
            against the return it was trained on and not another one.
        population: ``"vs_econ"`` or ``"mirror"``, appended to every key.

    Returns:
        Explained variance against the real return, pooled and within-turn,
        keyed under ``CRITIC``.
    """
    scores = critic_scores(
        [t.values for t in batch],
        [getattr(t, field) for t in batch],
        [t.dones for t in batch],
    )
    games = _POPULATION_GAMES[population]
    return {
        f"{CRITIC}{_CRITIC_SHORT_NAMES[name]}_{games}": value
        for name, value in scores.items()
    }


# The internal ``LossReport`` term names -> their logged, prefixed key. This
# table is the one place those names become the public dashboard-grouped ones.
_TERM_PREFIX = {
    "vtrace_pg": f"{DIAG}vtrace_pg",
    "upgo_pg": f"{DIAG}upgo_pg",
    "entropy": f"{DIAG}entropy",
    "teacher": f"{DIAG}teacher_kl",
    "total": f"{DIAG}total_loss",
    "baseline": f"{CRITIC}baseline_self_consistency",
    "baseline_passes": f"{CRITIC}baseline_passes_self_consistency",
}


def _prefixed_terms(terms: dict[str, float]) -> dict[str, float]:
    """Return loss-report terms under their public, prefixed keys.

    Args:
        terms: Loss terms by their internal names.

    Returns:
        The same values keyed by ``_TERM_PREFIX``.
    """
    return {_TERM_PREFIX[key]: value for key, value in terms.items()}


def _field(arguments: argparse.Namespace) -> str:
    """Return which recorded reward series this arm trains on.

    Every trajectory carries all of them, so the arm is a choice of field and
    nothing else -- which is what lets two arms share seeds, episodes and
    actions and differ by one string.

    Args:
        arguments: The parsed command line.

    Returns:
        The ``Trajectory`` attribute name the learner reads.

    Raises:
        ValueError: If more than one reward is asked for at once.
    """
    chosen = [
        field
        for field, flag in (
            ("shaped", arguments.no_money),
            ("margin", arguments.margin),
            ("own", arguments.own),
            ("sparse", arguments.sparse),
        )
        if flag
    ]
    if len(chosen) > 1:
        raise ValueError(
            f"--no-money, --margin, --own and --sparse name different rewards; "
            f"an arm trains on one of them, not {chosen}"
        )
    return chosen[0] if chosen else REWARD_FIELD


def _decay(
    econ_fraction: float,
    total_steps: int = TOTAL_STEPS,
    environments: int = ENVIRONMENTS,
    min_lr_multiplier: float = MIN_LR_MOD,
) -> Callable[[int], float]:
    """Return the LR multiplier function, floored at their ``min_lr_mod``.

    Toad's schedule reaches the floor exactly at ``total_steps``, so ours has to
    be keyed on what this arm actually collects per round rather than on the
    environment count. Those differ whenever an arm mixes opponents: a mirror
    episode records both seats and a scripted one records ours alone, so
    ``--econ-fraction 0.5`` collects 36 seats a round against the 48 the naive
    count assumes. Keyed on 48 the learning rate reaches the floor at three
    quarters of the budget and the rest of the run trains at 1e-6 -- which is
    tolerable in a 2e7 arm and is a quarter of this one.

    A plain function rather than a lambda so a resume restores the same
    schedule: ``LambdaLR.state_dict`` stores ``None`` for a function and the
    multiplier is rebuilt here from ``total_steps``, where a schedule silently
    restarted from step zero would restore the initial rate and quietly change
    the recipe partway through a run.

    Args:
        econ_fraction: Share of each round played against ``OPPONENT``.
        total_steps: Environment steps this arm trains for, from ``--total-steps``
            (default ``TOTAL_STEPS``). The curriculum's five phases each need a
            different budget (2e7-1e7 against the constant's 1e8).
        environments: Games collected in each logical round.
        min_lr_multiplier: Floor applied after linear learning-rate decay.

    Returns:
        The multiplier at a given schedule step.
    """
    updates = max(
        total_steps // (_seats_per_update(econ_fraction, environments) * TURNS), 1
    )

    def decay(step: int) -> float:
        return max(1.0 - step / updates, min_lr_multiplier)

    return decay


def _reference_policy(
    model: ModelConfig,
    state: dict[str, torch.Tensor],
    *,
    frozen: bool,
) -> Policy | StatefulPolicy:
    """Construct and strictly load one resolved worker-side policy."""
    if uses_stateful_policy(model):
        policy: Policy | StatefulPolicy = StatefulPolicy(model)
    else:
        policy = Policy(
            blocks=model.blocks,
            channels=model.channels,
            value_bound=model.value_bound,
            kernel_size=model.kernel_size,
            activation=model.activation,
        )
    policy.load_state_dict(state, strict=True)
    policy.eval()
    if frozen:
        policy.requires_grad_(False)
    return policy


def _play_reference(work: ReferenceWorkerInput) -> list[Trajectory]:
    """Play one typed native-worker request with its resolved architecture."""
    torch.set_num_threads(THREADS)
    actor = _reference_policy(work.model, work.actor_state, frozen=False)
    opponent: Policy | StatefulPolicy | str
    if work.opponent_state is not None:
        if work.opponent_model is None:
            raise ValueError("neural opponent weights require a resolved model config")
        opponent = _reference_policy(
            work.opponent_model,
            work.opponent_state,
            frozen=True,
        )
    else:
        opponent = work.versus if work.versus is not None else actor
    previous_money_weight = os.environ.get(MONEY_WEIGHT_ENV)
    os.environ[MONEY_WEIGHT_ENV] = repr(work.money_weight)
    try:
        with torch.no_grad():
            return rollout_many(
                actor,
                opponent,
                work.seeds,
                state_unroll_length=work.unroll_length,
            )
    finally:
        if previous_money_weight is None:
            os.environ.pop(MONEY_WEIGHT_ENV, None)
        else:
            os.environ[MONEY_WEIGHT_ENV] = previous_money_weight


def _seats_per_update(econ_fraction: float, environments: int = ENVIRONMENTS) -> int:
    """Return how many recorded seats one collection round yields.

    ``rollout_many`` records both seats of a mirror episode and ours alone
    against a scripted opponent, so the round's size is not ``ENVIRONMENTS``
    doubled unless the arm is a pure mirror.

    Args:
        econ_fraction: Share of each round played against ``OPPONENT``.
        environments: Games collected in each logical round.

    Returns:
        Trajectories, and so ``TURNS`` times this many environment steps.
    """
    econ = int(environments * econ_fraction)
    return 2 * (environments - econ) + econ


def _batches_per_update(econ_fraction: float) -> int:
    """Return how many learner batches one collection round yields."""
    segments = _seats_per_update(econ_fraction) * (TURNS // UNROLL_LENGTH)
    return segments // BATCH_SEGMENTS


def _acted(logits: torch.Tensor, turns: int, width: int) -> torch.Tensor:
    """Drop the trailing bootstrap state's row from a flattened head output.

    Args:
        logits: ``((turns + 1) * width, ...)`` one head's output over every
            forwarded state.
        turns: Acted rows per segment.
        width: Segments in the batch.

    Returns:
        ``(turns * width, ...)``, the acted states alone.
    """
    return logits.view(turns + 1, width, *logits.shape[1:])[:-1].flatten(0, 1)


def _kl(
    learner_log_probs: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return per-row KL(teacher || learner), summed over slots.

    monobeast.py:129-140. Theirs is ``F.kl_div(learner_log_probs, teacher_probs)``
    -- the forward direction, penalising the learner for putting low mass where
    the teacher puts high. Our ``ppo.divergence_of`` is the reverse direction and
    is deliberately not reused. Masked positions are zeroed before summing so the
    ``-inf`` they carry never reaches the arithmetic.

    Args:
        learner_log_probs: ``(rows, slots, options)`` masked log-probabilities.
        teacher_logits: ``(rows, slots, options)`` raw teacher logits.
        mask: ``(rows, slots, options)`` legality mask.

    Returns:
        ``(rows,)`` summed divergences.
    """
    teacher_log_probs = torch.log_softmax(
        teacher_logits.masked_fill(~mask, -torch.inf), dim=-1
    )
    teacher_probs = teacher_log_probs.exp()
    per_option = teacher_probs * (teacher_log_probs - learner_log_probs)
    return per_option.masked_fill(~mask, 0.0).sum(dim=-1).sum(dim=-1)


def _segments(trajectory: Trajectory) -> list[dict[str, torch.Tensor]]:
    """Chop one episode into ``UNROLL_LENGTH`` segments, anchored to its end.

    Their unroll_length is 16 and their reduction is ``sum``, so the loss scales
    with the segment shape; keeping both is part of keeping their coefficients
    meaningful. Two things beyond the length are load-bearing, and both were
    wrong until the critic was measured against the return it is supposed to
    predict rather than against its own target.

    **The segments are anchored to the last turn, not the first.** Tiling
    forward from turn 0 and dropping whatever does not fill a segment discards
    the final 15 turns of 719 -- and the terminal turn with them, which is the
    only row in the whole episode where ``dones`` is True. The learner therefore
    never saw a done, no discount was ever zeroed, and no target it ever built
    knew the season ends at all. Anchored to the end instead, the same 44
    segments cover turns 15-718 and the terminal is always the last acted row of
    the last segment. What gets dropped is the opening, whose encoded state is
    byte-identical across seeds anyway.

    **A segment carries one state more than it has decisions.** monobeast's
    buffers are ``unroll_length + 1`` long and it bootstraps from
    ``learner_outputs["baseline"][-1]`` (monobeast.py:292) after slicing the
    acted rows off with ``learner_outputs[:-1]`` (:296) -- i.e. from the value
    of the state *after* the segment's last action. Ours bootstrapped from the
    last acted row's own value, which seals each 16-turn window off from every
    state outside it: nothing beyond the window can enter the target, adjacent
    segments are never coupled, and the horizon can never propagate backwards
    however long the run goes on. The fixed point of that sealed target is
    ``r / (1 - gamma)``, the local reward rate extrapolated to an *infinite*
    horizon -- 2000x a turn's reward at our gamma of 0.9995, regardless of how
    many turns the season actually has left. That is precisely the critic the
    2026-08-09 measurement found: running to its -1 rail while the true
    return-to-go rose toward zero, at a predicted-over-actual ratio of 1.95
    rising to 8.91 across the season, and explaining the return worse than a
    constant would.

    Args:
        trajectory: One recorded seat's episode.

    Returns:
        One dict per segment, ``ACTED_FIELDS`` carrying ``UNROLL_LENGTH`` rows
        and ``OBSERVED_FIELDS`` carrying one more.
    """
    return segments(trajectory, UNROLL_LENGTH)


def _observed(turns: int, start: int) -> torch.Tensor:
    """Return the state rows one segment reads: its acted rows plus its bootstrap.

    The last segment of an episode ends the season, so the state after it does
    not exist. Its last acted row carries ``dones``, which zeroes that step's
    discount and so deletes the bootstrap from every return ``toad_loss``
    computes -- the value read there cannot reach the target whatever it is, and
    the final state is repeated only to keep every segment one shape. Clamping
    rather than special-casing is what stops the ragged tail coming back: there
    is no index this can return that is out of range, so a segment can never be
    short and no episode remainder can ever be silently dropped again.

    Args:
        turns: Acted rows in the episode.
        start: The segment's first acted row.

    Returns:
        ``(UNROLL_LENGTH + 1,)`` int64 row indices.
    """
    return torch.arange(start, start + UNROLL_LENGTH + 1).clamp(max=turns - 1)


if __name__ == "__main__":
    main()
