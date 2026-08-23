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
import copy
import json
import logging
import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import torch
import wandb
from lightning import seed_everything

from kaggriculture.learn import CHECKPOINT
from kaggriculture.learn.critic import critic_scores
from kaggriculture.learn.model import Policy, load_policy_weights
from kaggriculture.learn.rollout import Trajectory, rollout_many
from kaggriculture.learn.toad.data import (
    ACTED_FIELDS as _ACTED_FIELDS,
)
from kaggriculture.learn.toad.data import (
    OBSERVED_FIELDS as _OBSERVED_FIELDS,
)
from kaggriculture.learn.toad.data import (
    segments,
)
from kaggriculture.learn.toad_loss import (
    ADAM_EPS,
    CLIP_GRADS,
    DISCOUNTING,
    ENTROPY_COST,
    LEARNING_RATE,
    LMB,
    MIN_LR_MOD,
    TEACHER_KL_COST,
    TOTAL_STEPS,
    UNROLL_LENGTH,
    VALUE_WARMUP_BATCHES,
    losses,
)
from kaggriculture.learn.toad_reward import (
    ABSOLUTE_WEIGHT,
    CAPITAL_WEIGHT,
    MARGIN_WEIGHT,
    MONEY_WEIGHT_ENV,
    money_weight,
)

LOGGER = logging.getLogger(__name__)

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
    way to hand ``_step`` a teacher without also telling it which heads that
    teacher is entitled to constrain.

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
# One per environment. `_collect` splits the pool in half when an arm mixes
# opponents, so 12 gave each worker two episodes to play in series on a 64-core
# box that was running about ten of them. ENVIRONMENTS is deliberately NOT
# changed with it: that would move the effective batch and therefore the recipe.
WORKERS = ENVIRONMENTS
# Torch threads per worker. One, so WORKERS processes do not each claim the
# whole machine; see _play.
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
# One line per logged key, shipped into every run (see `_log_definitions`) so
# the meaning of a number sits beside the number instead of requiring someone
# to open this file. Derived from the comments at each metric's definition
# site in `_record`, `_population` and `_critic`; kept in sync with them by
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
    """Return this arm's command line, pulled out of ``main`` so a test can parse it.

    Every flag's default is the constant it overrides, never a literal, so an
    invocation that passes none of them is byte-identical to one that predates
    the flag existing.

    Returns:
        The argument parser ``main`` parses ``sys.argv`` with.
    """
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
        default=BLOCKS,
        help="residual blocks in the learner's (and rollout actor's) trunk. "
        "Phases 3-5 of the curriculum need 16 and 24 against phase 1-2's 8.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=CHANNELS,
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
        default=BLOCKS,
        help="residual blocks in the frozen teacher's trunk -- the depth its "
        "own checkpoint was written at, not this arm's --blocks. The recipe's "
        "teachers are always smaller nets than the phase they teach (phase 5's "
        "is the 16-block checkpoint against its own 24), so this cannot be "
        "assumed equal to --blocks.",
    )
    parser.add_argument(
        "--econ-fraction",
        type=float,
        default=0.0,
        help="fraction of each round's environments played against "
        "economic_policy instead of the mirror. Only our seat trains from "
        "those: the scripted agent is market pressure, not a tape to clone.",
    )
    parser.add_argument(
        "--teacher-kl-cost",
        type=float,
        default=TEACHER_KL_COST,
        help="teacher KL weight. Their cascade drops it to 0.001 at phase 3, "
        "where the policy should start out-earning its teacher.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help="Adam learning rate the schedule decays from. Sweep knob; "
        "the default reproduces every earlier arm exactly.",
    )
    parser.add_argument(
        "--entropy-cost",
        type=float,
        default=ENTROPY_COST,
        help="coefficient on the entropy loss term. Sweep knob; the default "
        "reproduces every earlier arm exactly.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=DISCOUNTING,
        help="discount the return and advantage targets are built at "
        "(``losses``' ``discounting``). Sweep knob; the default reproduces "
        "every earlier arm exactly.",
    )
    parser.add_argument(
        "--lmb",
        type=float,
        default=LMB,
        help="lambda for both TD(lambda) and UPGO, named to match "
        "toad_loss.losses' own parameter rather than shadowing the Python "
        "keyword. Their 0.8 for phases 1-4 and 0.9 for phase 5.",
    )
    parser.add_argument(
        "--value-warmup-batches",
        type=int,
        default=VALUE_WARMUP_BATCHES,
        help="train the value head alone for this many batches before the "
        "policy gradient fires (arm C'). Their phase-2 mechanism, for a "
        "warm-started policy whose critic is untrained. 0 means no warmup.",
    )
    parser.add_argument(
        "--value-passes",
        type=int,
        default=0,
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
        default=TOTAL_STEPS,
        help="environment steps this arm trains for. The curriculum's five "
        "phases each need a different budget (2e7-1e7) against this "
        "constant's declared-deviation 1e8.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run one arm to ``--total-steps``, checkpointing and logging as it goes.

    Args:
        argv: Flags to parse, or None for ``sys.argv[1:]`` (the CLI's own
            default). ``curriculum.py`` passes an explicit list here so it can
            invoke this entry point directly rather than shelling out or
            duplicating the training loop.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    arguments = _parser().parse_args(argv)

    seed_everything(SEED, workers=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    # The single constant the ablation turns on. Everything downstream -- file
    # names, wandb run name, which reward the learner reads -- follows from it,
    # so the two arms cannot drift apart in any other respect.
    if arguments.money_weight is not None:
        # Into the environment before the worker pool forks, so every rollout
        # process computes `shaped_money` at this arm's weight.
        os.environ[MONEY_WEIGHT_ENV] = repr(arguments.money_weight)
    field = _field(arguments)
    prefix = _prefix(arguments)

    device = _device()
    learner = _learner(arguments, device)
    if arguments.clone_init:
        _warm_start(learner, device)
    optimizer = _optimizer(learner, arguments.lr)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _decay(arguments.econ_fraction, arguments.total_steps)
    )

    steps, update = 0, 0
    if arguments.resume is not None:
        steps, update = _restore(arguments.resume, learner, optimizer, schedule, device)
        LOGGER.info(
            "resumed from %s at update %d, step %d", arguments.resume, update, steps
        )

    log = RUNS / f"{prefix}_{int(time.time())}.jsonl"
    run = _start_run(arguments, field)
    parameters = sum(p.numel() for p in learner.parameters())
    LOGGER.info(
        "phase 1: %d params, device %s, log %s, wandb %s",
        parameters,
        device,
        log,
        run.url,
    )

    teacher = _teacher(arguments, device)
    actor = copy.deepcopy(learner).eval()
    warmup_left = arguments.value_warmup_batches
    if warmup_left:
        LOGGER.info(
            "value warmup: %d batches (~%.1f updates at %d batches/update)",
            warmup_left,
            warmup_left / _batches_per_update(arguments.econ_fraction),
            _batches_per_update(arguments.econ_fraction),
        )
    started = time.monotonic()
    pool = ProcessPoolExecutor(max_workers=WORKERS)
    while steps < arguments.total_steps:
        seeds = tuple(range(update * ENVIRONMENTS, (update + 1) * ENVIRONMENTS))
        weights = {key: value.cpu() for key, value in actor.state_dict().items()}
        mirror_batch, econ_batch = _collect(
            pool,
            weights,
            seeds,
            arguments.blocks,
            arguments.channels,
            arguments.econ_fraction,
        )
        batch = mirror_batch + econ_batch
        steps += sum(int(t.shaped.shape[0]) for t in batch)
        terms, consumed = _update(
            learner,
            optimizer,
            batch,
            device,
            field,
            warmup_left,
            teacher,
            arguments.teacher_kl_cost,
            arguments.value_passes,
            entropy_cost=arguments.entropy_cost,
            discounting=arguments.gamma,
            lmb=arguments.lmb,
        )
        warming = warmup_left > 0
        warmup_left = max(0, warmup_left - consumed)
        schedule.step()
        update += 1
        # No sync while the value head warms up: the actor must keep rolling
        # out the warm-started policy so the critic learns on the distribution
        # it will actually have to evaluate. Syncing here is precisely what
        # destroyed arm C at update 5.
        if not warming and update % SYNC_EVERY == 0:
            actor.load_state_dict(learner.state_dict())
        if update % CHECKPOINT_EVERY == 0:
            _checkpoint(learner, optimizer, schedule, steps, update, prefix)

        record = _record(
            mirror_batch,
            econ_batch,
            field,
            update=update,
            steps=steps,
            hours=round((time.monotonic() - started) / 3600.0, 4),
            lr=float(schedule.get_last_lr()[0]),
            warming=warming,
            warmup_left=warmup_left,
            terms=terms,
        )
        with log.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        wandb.log(record, step=steps)
        LOGGER.info(
            "update %d steps %d win_vs_econ %.3f margin %.1f bank %.1f "
            "sale_price %.1f reward %.4f total_loss %.3f",
            update,
            steps,
            record["objective/win_rate_vs_econ"],
            record["objective/margin_vs_econ"],
            record["diag/bank_mean"],
            record["diag/mean_sale_price_vs_econ"],
            record["proxy/shaped_reward_mean"],
            record["diag/total_loss"],
        )
    wandb.finish()


def _learner(arguments: argparse.Namespace, device: str) -> Policy:
    """Return the network being trained, built from the parsed command line.

    Pulled out of ``main`` so a test can construct one from a parsed
    ``--blocks`` and count its own residual blocks -- the only way to tell
    "parsed the flag" from "parsed the flag and never used it".

    Args:
        arguments: The parsed command line.
        device: Where to place the network.

    Returns:
        The learner, not yet warm-started or optimized.
    """
    return Policy(
        blocks=arguments.blocks, channels=arguments.channels, value_bound=VALUE_BOUND
    ).to(device)


def _optimizer(learner: Policy, lr: float) -> torch.optim.Optimizer:
    """Return the Adam optimizer this arm trains with.

    Pulled out of ``main`` so a test can construct one from a parsed ``--lr``
    and read the rate back off its own ``param_groups`` -- the only way to
    tell "parsed the flag" from "parsed the flag and never used it".

    Args:
        learner: The network whose parameters the optimizer will update.
        lr: The learning rate, threaded from ``--lr`` (default ``LEARNING_RATE``).

    Returns:
        The constructed optimizer, before any schedule wraps it.
    """
    return torch.optim.Adam(learner.parameters(), lr=lr, eps=ADAM_EPS)


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


# The internal loss-term names `_step` returns -> their logged, prefixed key.
# `_step` and `_update` keep the short names (tested directly by
# test_toad_runner.py); this table is the one place those names become the
# public, dashboard-grouped ones.
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
    """Return ``_update``'s loss terms under their public, prefixed keys.

    Args:
        terms: The loss terms ``_update`` returned, by their internal names.

    Returns:
        The same values keyed by ``_TERM_PREFIX``.
    """
    return {_TERM_PREFIX[key]: value for key, value in terms.items()}


def _start_run(arguments: argparse.Namespace, field: str) -> "wandb.sdk.wandb_run.Run":
    """Open the tracked wandb run for this arm and ship every metric's meaning into it.

    Every knob that distinguishes one arm from another is recorded here, so a
    run's identity can be read off the dashboard rather than reconstructed from
    a shell command nobody kept. That includes the weights of whichever reward
    is live and zeroes for the ones that are not.

    THE INCIDENT this exists to prevent read a dashboard and trusted two
    numbers whose correct interpretation was sitting in this file's comments
    the whole time, unreachable from wandb. ``METRIC_DEFINITIONS`` is those
    comments, one line each; logging it as a table puts the run's own page one
    click from the explanation instead of a source read.

    Args:
        arguments: The parsed command line.
        field: The reward series the learner will read.

    Returns:
        The started run, with its metric glossary already logged.
    """
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=arguments.name or _default_name(arguments),
        config={
            "blocks": arguments.blocks,
            "teacher_blocks": arguments.teacher_blocks if arguments.teacher else None,
            "channels": arguments.channels,
            "value_bound": VALUE_BOUND,
            "reward_field": field,
            "money_weight": 0.0 if arguments.no_money else money_weight(),
            "margin_weight": MARGIN_WEIGHT if arguments.margin else 0.0,
            "absolute_weight": ABSOLUTE_WEIGHT if arguments.margin else 0.0,
            # Live on every arm that reads a `shaped` field and zero on the
            # margin arm, which prices capital through ABSOLUTE_WEIGHT instead.
            # Recorded because it is the one weight in the shaped set that is
            # not Toad's published number, so a run's reward is not identifiable
            # without it.
            "capital_weight": 0.0 if arguments.margin else CAPITAL_WEIGHT,
            "econ_fraction": arguments.econ_fraction,
            "value_warmup_batches": arguments.value_warmup_batches,
            "value_passes": arguments.value_passes,
            "teacher": str(arguments.teacher)
            if arguments.teacher is not None
            else None,
            "teacher_kl_cost": arguments.teacher_kl_cost,
            "clone_init": arguments.clone_init,
            "environments": ENVIRONMENTS,
            "batch_segments": BATCH_SEGMENTS,
            "unroll_length": UNROLL_LENGTH,
            "sync_every": SYNC_EVERY,
            "lr": arguments.lr,
            "entropy_cost": arguments.entropy_cost,
            "gamma": arguments.gamma,
            "lmb": arguments.lmb,
            "adam_eps": ADAM_EPS,
            "clip_grads": CLIP_GRADS,
            "total_steps": arguments.total_steps,
            "engine": metadata.version("kaggle-environments"),
        },
    )
    _log_definitions(run)
    return run


def _log_definitions(run: "wandb.sdk.wandb_run.Run") -> None:
    """Log ``METRIC_DEFINITIONS`` once, as a table, so it lives beside the run.

    A ``wandb.Table`` logged once (the default ``log_mode="IMMUTABLE"``) is
    stored as a single snapshot on the run and surfaces in its Tables tab and
    summary -- the documented pattern for "one value that does not change over
    the run" (see the Tables logging guide), as opposed to a metric logged
    every step. A metric definition does not change during a run either, so
    this is logged once here rather than folded into ``record`` and repeated
    719 times.

    Args:
        run: The run just opened by ``wandb.init``.
    """
    run.log(
        {
            "metric_definitions": wandb.Table(
                columns=["metric", "definition"],
                data=[list(row) for row in sorted(METRIC_DEFINITIONS.items())],
            )
        }
    )


def _teacher(arguments: argparse.Namespace, device: str) -> Teacher | None:
    """Return the frozen checkpoint the learner is held near, or None.

    The recipe's teachers are "always the pipeline's own earlier, smaller
    checkpoints" -- never a behaviour clone and never a replay. ``--teacher``
    names one such checkpoint per phase; phase 1 passes none, because its
    ``teacher_kl_cost`` is 0 and it is genuinely teacher-free self-play from
    random initialisation. This project has measured what anchoring to the
    wrong thing costs: removing the penalty took the bank from 17,675 to 9 in
    five updates, and dropping the cost to 0.001 cliffed within a single sync
    cycle -- both true of *a* teacher, not of any one checkpoint being right.

    Which heads the KL may cover is decided here and nowhere else, from the
    keys the checkpoint was missing. A checkpoint written before the quantity
    head existed leaves that head at its random initialisation, and anchoring
    the learner to random weights is worse than not anchoring it at all -- so
    the answer travels with the network, in ``Teacher``, rather than being
    re-guessed at the loss.

    The teacher's trunk is built at ``--teacher-blocks``, not ``--blocks``:
    the recipe's teachers are smaller nets than the phase they teach (phase
    5's is the 16-block checkpoint, taught against its own 24), so the two
    cannot be assumed equal, and ``load_policy_weights`` raises rather than
    silently reconciling a shape mismatch if they are.

    Args:
        arguments: The parsed command line.
        device: Where to place the teacher.

    Returns:
        The frozen policy loaded from ``arguments.teacher``, paired with
        whether its checkpoint taught the quantity head, or None when the
        arm runs teacher-free (``arguments.teacher is None``).
    """
    if arguments.teacher is None:
        return None
    policy = Policy(
        blocks=arguments.teacher_blocks,
        channels=arguments.channels,
        value_bound=VALUE_BOUND,
    ).to(device)
    state = torch.load(arguments.teacher, map_location=device, weights_only=True)
    missing = load_policy_weights(policy, state)
    policy.eval()
    policy.requires_grad_(False)
    quantity = not any(key.startswith("quantity_head.") for key in missing)
    LOGGER.info(
        "teacher: %s, kl_cost %.4f, quantity head %s",
        arguments.teacher,
        arguments.teacher_kl_cost,
        "taught" if quantity else "absent from the checkpoint, excluded from the KL",
    )
    return Teacher(policy=policy, quantity=quantity)


def _prefix(arguments: argparse.Namespace) -> str:
    """Return the stem for this arm's checkpoints and jsonl.

    Distinct per arm, because two arms sharing a prefix overwrite each other's
    checkpoints silently and the evaluator globs on it.
    """
    if arguments.name:
        return arguments.name
    if arguments.margin:
        return "phase1m"
    if arguments.no_money:
        return "phase1-no-money"
    return "phase1"


def _default_name(arguments: argparse.Namespace) -> str:
    """Return the run name for an arm that did not pass ``--name``."""
    if arguments.margin:
        return "toad-phase1m-margin"
    if arguments.no_money:
        return "toad-phase1-no-money"
    return "toad-phase1-baseline"


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


def _warm_start(learner: Policy, device: str) -> list[str]:
    """Load the BC clone into the trunk and every trained head.

    ``load_policy_weights`` does the loading; see it for what non-strict means
    here. It is not a promise that ``CHECKPOINT`` loads -- the quantity-lane
    widening resized ``trade_head``, an existing key, and ``CHECKPOINT`` on
    disk predates that widening, so this currently raises out of
    ``load_state_dict`` before ``load_policy_weights``'s own check ever runs.
    Retraining the clone is Phase 2's job, not this arm's.

    Args:
        learner: The network to warm-start, modified in place.
        device: Where to map the checkpoint.

    Returns:
        The keys the checkpoint did not carry, which ``load_policy_weights``
        confines to the quantity head.

    Raises:
        ValueError: If the checkpoint is missing or renaming anything beyond
            the quantity head.
        RuntimeError: If a key present in both the checkpoint and the module
            has a shape ``load_state_dict`` cannot reconcile -- the case
            ``CHECKPOINT`` currently hits.
    """
    state = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    missing = load_policy_weights(learner, state)
    LOGGER.info(
        "warm started from %s; fresh heads (%s)", CHECKPOINT, ", ".join(missing)
    )
    return missing


def _decay(
    econ_fraction: float, total_steps: int = TOTAL_STEPS
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

    Returns:
        The multiplier at a given schedule step.
    """
    updates = max(total_steps // (_seats_per_update(econ_fraction) * TURNS), 1)

    def decay(step: int) -> float:
        return max(1.0 - step / updates, MIN_LR_MOD)

    return decay


def _checkpoint(
    learner: Policy,
    optimizer: torch.optim.Optimizer,
    schedule: torch.optim.lr_scheduler.LRScheduler,
    steps: int,
    update: int,
    prefix: str,
) -> Path:
    """Write everything needed to continue the run, atomically.

    Toad's monobeast checkpoints continuously; ours did not, and an external
    kill at update 253 cost 2.3 hours of training because the weights lived only
    in the process. Written to a temporary name and renamed, so a kill during
    the write leaves the previous checkpoint intact rather than a truncated one.

    Args:
        learner: The network being trained.
        optimizer: Its optimizer, whose Adam moments matter as much as the weights.
        schedule: The LR schedule, so the recipe continues rather than restarts.
        steps: Environment steps consumed so far.
        update: Optimizer rounds so far.
        prefix: Run-distinguishing stem, so the two arms cannot clobber each
            other's checkpoints.

    Returns:
        The path written.
    """
    path = RUNS / f"{prefix}_{update:06d}.pt"
    temporary = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "learner": learner.state_dict(),
            "optimizer": optimizer.state_dict(),
            "schedule": schedule.state_dict(),
            "steps": steps,
            "update": update,
        },
        temporary,
    )
    temporary.rename(path)
    LOGGER.info("checkpoint %s", path)
    return path


def _restore(
    path: Path,
    learner: Policy,
    optimizer: torch.optim.Optimizer,
    schedule: torch.optim.lr_scheduler.LRScheduler,
    device: str,
) -> tuple[int, int]:
    """Load a checkpoint in place and return its ``(steps, update)``.

    Restores the schedule's own state rather than fast-forwarding it, because
    recomputing the position by stepping it ``update`` times is exactly where an
    off-by-one would hide, and a schedule one step out changes the learning rate
    for the rest of the run.

    ``load_policy_weights`` tolerates a checkpoint missing ``quantity_head.*``
    because that gap is expected of a warm start from an older, pre-widening
    clone. A resume checkpoint is not that -- it is this script's own prior
    output, already written by a ``Policy`` that has every current head -- so
    a gap here means the wrong file was pointed at, and loading it anyway
    would silently continue training with that head randomly initialised.

    Args:
        path: The checkpoint.
        learner: Network to load into.
        optimizer: Optimizer to load into.
        schedule: Schedule to load into.
        device: Where to map the tensors.

    Returns:
        The ``(steps, update)`` the checkpoint was written at.

    Raises:
        ValueError: If ``load_policy_weights`` reports any missing key, since
            a resume checkpoint should never have one.
    """
    state = torch.load(path, map_location=device, weights_only=False)
    missing = load_policy_weights(learner, state["learner"])
    if missing:
        raise ValueError(
            f"{path} is missing {missing}: a resume checkpoint is this "
            "script's own prior output, not a warm start from an older "
            "clone, so it should already carry every current head, and "
            "continuing would silently leave the missing head randomly "
            "initialised for the rest of the run"
        )
    optimizer.load_state_dict(state["optimizer"])
    schedule.load_state_dict(state["schedule"])
    return int(state["steps"]), int(state["update"])


def _device() -> str:
    """Return the CUDA device with the most free memory, or CPU.

    The machine's GPUs are shared with other runs, so a hardcoded ``cuda:0``
    lands on whichever card someone else already filled -- measured, that is how
    the first attempt died.
    """
    if not torch.cuda.is_available():
        return "cpu"
    free = [
        torch.cuda.mem_get_info(index)[0] for index in range(torch.cuda.device_count())
    ]
    return f"cuda:{free.index(max(free))}"


def _collect(
    pool: ProcessPoolExecutor,
    state: dict[str, torch.Tensor],
    seeds: Sequence[int],
    blocks: int = BLOCKS,
    channels: int = CHANNELS,
    econ_fraction: float = 0.0,
) -> tuple[list[Trajectory], list[Trajectory]]:
    """Play ``seeds`` across worker processes and return every trajectory.

    Both seats of each episode are recorded, because the actor plays itself and
    the two seats therefore carry identical weights -- the condition
    ``rollout_many`` requires before it will record seat 1.

    Args:
        pool: The process pool to spread episodes over.
        state: The actor's weights, on CPU so they pickle to the workers.
        seeds: One seed per episode.
        blocks: Residual blocks in the trunk, so the workers rebuild the actor
            at this arm's depth rather than the module default.
        channels: Trunk width, so the workers rebuild the actor at this arm's
            size rather than the module default.
        econ_fraction: Share of the round played against ``OPPONENT`` instead of
            the mirror. Those episodes record our seat only.

    Returns:
        Every recorded trajectory: two per mirror seed and one per scripted one.
    """
    split = int(len(seeds) * econ_fraction)
    econ_seeds, mirror_seeds = list(seeds[:split]), list(seeds[split:])
    work = []
    for group, versus in ((mirror_seeds, None), (econ_seeds, OPPONENT)):
        share = max(1, WORKERS // 2) if econ_seeds else WORKERS
        chunks = [group[index::share] for index in range(share)]
        work.extend(
            (state, chunk, blocks, channels, versus) for chunk in chunks if chunk
        )
    played = list(pool.map(_play, work))
    mirror: list[Trajectory] = []
    econ: list[Trajectory] = []
    for (_, _, _, _, versus), batch in zip(work, played, strict=True):
        (econ if versus else mirror).extend(batch)
    return mirror, econ


def _play(
    work: tuple[dict[str, torch.Tensor], list[int], int, int, str | None],
) -> list[Trajectory]:
    """Play one worker's share of a round. Runs in a subprocess.

    Torch is pinned to one thread here for the same reason ``selfplay`` pins it
    (:509): every worker otherwise defaults to a pool the width of the machine,
    and WORKERS of those oversubscribe it badly enough to be slower than a
    single process. Measured at 64 cores, leaving this out drove load average
    past 230 and the round did not finish.
    """
    torch.set_num_threads(THREADS)
    state, seeds, blocks, channels, versus = work
    actor = Policy(blocks=blocks, channels=channels, value_bound=VALUE_BOUND)
    actor.load_state_dict(state)
    actor.eval()
    with torch.no_grad():
        # A mirror records both seats; against a named agent `rollout_many`
        # records seat 0 alone, which is exactly what we want -- we never train
        # on the scripted agent's actions.
        return rollout_many(actor, versus if versus else actor, seeds)


def _seats_per_update(econ_fraction: float) -> int:
    """Return how many recorded seats one collection round yields.

    ``rollout_many`` records both seats of a mirror episode and ours alone
    against a scripted opponent, so the round's size is not ``ENVIRONMENTS``
    doubled unless the arm is a pure mirror.

    Args:
        econ_fraction: Share of each round played against ``OPPONENT``.

    Returns:
        Trajectories, and so ``TURNS`` times this many environment steps.
    """
    econ = int(ENVIRONMENTS * econ_fraction)
    return 2 * (ENVIRONMENTS - econ) + econ


def _batches_per_update(econ_fraction: float) -> int:
    """Return how many learner batches one collection round yields."""
    segments = _seats_per_update(econ_fraction) * (TURNS // UNROLL_LENGTH)
    return segments // BATCH_SEGMENTS


def _update(
    learner: Policy,
    optimizer: torch.optim.Optimizer,
    batch: list[Trajectory],
    device: str,
    field: str,
    warmup_left: int = 0,
    teacher: Teacher | None = None,
    teacher_kl_cost: float = TEACHER_KL_COST,
    value_passes: int = 0,
    entropy_cost: float = ENTROPY_COST,
    discounting: float = DISCOUNTING,
    lmb: float = LMB,
) -> tuple[dict[str, float], int]:
    """Take one optimizer step per ``BATCH_SEGMENTS`` unrolls and return the means.

    Their learner consumes batches of four 16-step unrolls and steps once per
    batch, continuously fed by the actors. Collecting a whole round and stepping
    once on all of it would be a different algorithm with a different effective
    learning rate, so the round is chopped into their batch shape instead.

    **A round's data is otherwise seen once and thrown away**, and that is what
    ``value_passes`` exists to change. Measured on 2026-08-15: a fresh value head
    trained on 32 episodes from a live arm's own checkpoint reaches held-out
    explained variance of 0.348 under this exact target -- ``_segments`` feeding
    ``_step(baseline_only=True)`` -- against a clock-only critic's 0.121, while
    the critic inside the arm that produced those episodes sat at -0.103
    within-step. The target was not the difference; the number of looks was, 150
    against one. Each extra pass reshuffles and replays the same round through the
    value head alone, which is the same call ``--value-warmup`` already makes, so
    the deviation is in how often it is made rather than in what it does. Their
    warmup backpropagates through the shared trunk too, so this is not
    policy-neutral -- monobeast.py:416-418 and ``toad_loss.losses``.

    Args:
        learner: The network being trained, updated in place.
        optimizer: Its optimizer.
        batch: The trajectories collected this round.
        device: Where to run the learner.
        field: Which recorded reward series the learner reads.
        warmup_left: Batches still owed to the value head alone.
        teacher: The frozen checkpoint to stay near and what it was taught,
            or None.
        teacher_kl_cost: Coefficient on that KL.
        value_passes: Extra value-only passes over the same round, after the
            policy has taken its one. Zero reproduces every earlier arm exactly.
        entropy_cost: Coefficient on the entropy loss term.
        discounting: Gamma the return and advantage targets are built at,
            passed through to every ``_step`` and ``_value_passes`` call this
            round makes.
        lmb: Lambda for both TD(lambda) and UPGO. Their 0.8 for phases 1-4,
            0.9 for phase 5.

    Returns:
        The loss terms averaged over the round's policy steps, plus
        ``baseline_passes`` -- the value loss averaged over the extra passes, or
        the round's own baseline when there are none -- and the number of batches
        the policy pass consumed. The extra passes are deliberately absent from
        that count: it is the warmup budget's clock, and warmup is measured in
        batches the *policy* did not learn from.
    """
    segments = [s for trajectory in batch for s in _segments(trajectory)]
    totals: dict[str, float] = {}
    steps = 0
    for start in range(0, len(segments) - BATCH_SEGMENTS + 1, BATCH_SEGMENTS):
        terms = _step(
            learner,
            optimizer,
            segments[start : start + BATCH_SEGMENTS],
            device,
            field,
            baseline_only=steps < warmup_left,
            teacher=teacher,
            teacher_kl_cost=teacher_kl_cost,
            entropy_cost=entropy_cost,
            discounting=discounting,
            lmb=lmb,
        )
        for key, value in terms.items():
            totals[key] = totals.get(key, 0.0) + value
        steps += 1
    means = {key: value / max(steps, 1) for key, value in totals.items()}
    means["baseline_passes"] = _value_passes(
        learner,
        optimizer,
        segments,
        device,
        field,
        value_passes,
        discounting=discounting,
        lmb=lmb,
    ) or means.get("baseline", 0.0)
    return means, steps


def _value_passes(
    learner: Policy,
    optimizer: torch.optim.Optimizer,
    segments: list[dict[str, torch.Tensor]],
    device: str,
    field: str,
    passes: int,
    discounting: float = DISCOUNTING,
    lmb: float = LMB,
) -> float:
    """Replay a round through the value head alone and return the mean loss.

    Reshuffled each pass, because the segments arrive ordered by episode and then
    by turn: walking that order repeatedly would hand the optimizer a sequence of
    batches drawn from one episode at a time, which is the correlation the
    ceiling experiment's ``randperm`` did not have.

    Args:
        learner: The network being trained, updated in place.
        optimizer: Its optimizer.
        segments: The round's unrolls.
        device: Where to run the learner.
        field: Which recorded reward series the learner reads.
        passes: How many times to replay. Zero returns 0.0 and touches nothing.
        discounting: Gamma the value target is built at, same one the policy
            pass used.
        lmb: Lambda for the TD(lambda) value target.

    Returns:
        The mean ``baseline`` loss across every extra batch, or 0.0 if there were
        none.
    """
    total, steps = 0.0, 0
    for _ in range(passes):
        order = torch.randperm(len(segments))
        for start in range(0, len(segments) - BATCH_SEGMENTS + 1, BATCH_SEGMENTS):
            terms = _step(
                learner,
                optimizer,
                [segments[index] for index in order[start : start + BATCH_SEGMENTS]],
                device,
                field,
                baseline_only=True,
                discounting=discounting,
                lmb=lmb,
            )
            total += terms["baseline"]
            steps += 1
    return total / steps if steps else 0.0


def _step(
    learner: Policy,
    optimizer: torch.optim.Optimizer,
    segments: list[dict[str, torch.Tensor]],
    device: str,
    field: str,
    baseline_only: bool = False,
    teacher: Teacher | None = None,
    teacher_kl_cost: float = TEACHER_KL_COST,
    entropy_cost: float = ENTROPY_COST,
    discounting: float = DISCOUNTING,
    lmb: float = LMB,
) -> dict[str, float]:
    """Take one gradient step on one batch of unrolls.

    Each unroll is re-scored under the learner's current weights, which is what
    makes the importance ratios meaningful: the batch was played by the actor,
    and the actor lags.

    Args:
        learner: The network being trained, updated in place.
        optimizer: Its optimizer.
        segments: ``BATCH_SEGMENTS`` unrolls of ``UNROLL_LENGTH`` turns.
        device: Where to run the learner.
        field: Which recorded reward series the learner reads.
        baseline_only: Train the value head alone, excluding the policy gradient
            and entropy terms from the total.
        teacher: The frozen checkpoint to stay near and what it was taught,
            or None.
        teacher_kl_cost: Coefficient on that KL.
        entropy_cost: Coefficient on the entropy loss term.
        discounting: Gamma the return and advantage targets are built at.
            ``losses``' own default, ``toad_loss.DISCOUNTING``.
        lmb: Lambda for both TD(lambda) and UPGO, passed to ``toad_loss.losses``.

    Returns:
        The four loss terms and their total, as floats.
    """
    from kaggriculture.learn.toad.config import ToadConfig
    from kaggriculture.learn.toad.data import BatchKind, LearnerBatch
    from kaggriculture.learn.toad.lightning import compute_loss

    control = ToadConfig.control()
    config = control.model_copy(
        update={
            "optimizer": control.optimizer.model_copy(
                update={
                    "teacher_kl_cost": teacher_kl_cost,
                    "entropy_cost": entropy_cost,
                    "gamma": discounting,
                    "lmb": lmb,
                }
            ),
            "curriculum": control.curriculum.model_copy(
                update={"reward_field": field}
            ),
        }
    )
    batch = LearnerBatch(
        segments=tuple(
            {name: tensor.to(device) for name, tensor in segment.items()}
            for segment in segments
        ),
        kind=BatchKind.SELFPLAY,
        baseline_only=baseline_only,
        first_of_round=True,
        end_of_round=True,
        collected_steps=0,
        round_id=0,
        actor_version=0,
        game_ids=(),
        opponent_ids=(),
    )
    report = compute_loss(
        learner,
        batch,
        config,
        teacher,
        baseline_only=baseline_only,
        _losses=losses,
    )
    optimizer.zero_grad(set_to_none=True)
    report.total.backward()
    # monobeast.py:502-505, with their saved runs' clip_grads of 10.0. Without
    # it this diverges to a loss of 6e20 inside two updates.
    torch.nn.utils.clip_grad_norm_(learner.parameters(), CLIP_GRADS)
    optimizer.step()
    return {name: value.item() for name, value in report.terms.items()}


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
