"""Toad Brigade's five-phase curriculum, as a table instead of a memory.

Every arm this project has run executed phase 1 with phase 2's teacher cost,
against a behaviour clone the recipe uses nowhere, because those numbers lived
in a command line nobody could check. This module is the fix: the five
phases, transcribed verbatim from the recipe into ``PHASES``, and a runner
that translates one row into the validated ``ToadConfig`` consumed by the
native Lightning entry point rather than a remembered invocation.

``PHASES`` is deliberately the only place these numbers are typed. A phase
boundary is now a dataclass field a test can assert against, not a line in a
shell history.

Note the two columns that are easy to get backwards: ``teacher_from`` names
the phase whose *checkpoint* teaches this one, and the recipe's teachers are
always smaller, earlier nets -- phases 3 and 4 are both taught by phase 1's
8-block checkpoint, and phase 5 by phase 3's 16-block one, never by the phase
immediately before it. Phase 4 is the one row that continues its
predecessor's own weights (only ``teacher_kl_cost`` moves between phase 3 and
phase 4); every other phase starts from its own fresh initialisation, guided
toward its teacher by the KL term alone. This runner resolves each phase's
teacher checkpoint and refuses to start without it; it does not attempt to
resume phase 4's weights from phase 3 automatically -- see the module's task
report for why, and pass an explicit ``--set runtime.resume="..."`` if that
is wanted.
"""

import argparse
import dataclasses
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from kaggriculture.learn.scripts import toad
from kaggriculture.learn.toad.config import (
    CurriculumConfig,
    ModelConfig,
    OptimizerConfig,
    PopulationConfig,
    RuntimeConfig,
    ToadConfig,
    apply_overrides,
)

LOGGER = logging.getLogger(__name__)

RewardField = Literal["shaped_money", "shaped", "sparse", "own"]


@dataclasses.dataclass(frozen=True)
class Phase:
    """One phase of the curriculum, transcribed verbatim from the recipe.

    Attributes:
        name: This phase's identifier -- also its wandb run name and
            checkpoint-file prefix, via ``--name``.
        blocks: Residual blocks in this phase's own trunk.
        steps: Environment steps this phase trains for.
        reward: Which ``Trajectory`` field the arm trains on: ``"shaped_money"``
            (phase 1's dense per-turn reward, Toad's five components with the
            score constituent in the slot their ``city`` weight occupies) or
            ``"sparse"`` (phases 2-5's terminal-only ``GameResultReward``).
            ``"shaped"`` -- that same set with the score constituent deleted --
            is the ablation, reachable here but not on the recipe's path.
        teacher_kl_cost: Coefficient on the KL toward this phase's teacher.
            Zero for phase 1, which has none.
        lr: Adam learning rate the schedule decays from.
        entropy_cost: Coefficient on the entropy loss term.
        lmb: Lambda for both TD(lambda) and UPGO.
        teacher_from: The phase whose checkpoint teaches this one, or None
            for phase 1, which is teacher-free self-play from scratch.
    """

    name: str
    blocks: int
    steps: int
    reward: RewardField
    teacher_kl_cost: float
    lr: float
    entropy_cost: float
    lmb: float
    teacher_from: str | None


# The five phases, verbatim from the recipe. Transcribed once, here, so that
# every arm reads the same table instead of a remembered command line --
# see the module docstring and test_the_phase_table_matches_the_recipe.
PHASES: tuple[Phase, ...] = (
    Phase(
        name="phase1",
        blocks=8,
        steps=int(2e7),
        reward="shaped_money",
        teacher_kl_cost=0.0,
        lr=1e-4,
        entropy_cost=1e-3,
        lmb=0.8,
        teacher_from=None,
    ),
    Phase(
        name="phase2",
        blocks=8,
        steps=int(1e7),
        reward="sparse",
        teacher_kl_cost=0.005,
        lr=1e-4,
        entropy_cost=1e-3,
        lmb=0.8,
        teacher_from="phase1",
    ),
    Phase(
        name="phase3",
        blocks=16,
        steps=int(2e7),
        reward="sparse",
        teacher_kl_cost=0.01,
        lr=5e-5,
        entropy_cost=2e-4,
        lmb=0.8,
        teacher_from="phase1",
    ),
    Phase(
        name="phase4",
        blocks=16,
        steps=int(2e7),
        reward="sparse",
        teacher_kl_cost=0.001,
        lr=5e-5,
        entropy_cost=2e-4,
        lmb=0.8,
        teacher_from="phase1",
    ),
    Phase(
        name="phase5",
        blocks=24,
        steps=int(2e7),
        reward="sparse",
        teacher_kl_cost=0.005,
        lr=5e-5,
        entropy_cost=2e-4,
        lmb=0.9,
        teacher_from="phase3",
    ),
)

_BY_NAME = {phase.name: phase for phase in PHASES}

# Every reward the table may name, and the flags that select it. Total rather
# than a test against ``"sparse"`` alone, because an unlisted reward must raise
# instead of falling through to whatever ``toad.REWARD_FIELD`` defaults to.
# ``shaped_money`` *is* that default and so has no flag of its own; the empty
# tuple says so out loud, and
# ``test_every_phase_reward_survives_toad_s_own_parser`` is what pins the two
# together -- the default has moved once already.
REWARD_FLAGS: dict[str, tuple[str, ...]] = {
    "shaped_money": (),
    "shaped": ("--no-money",),
    "sparse": ("--sparse",),
}


def _phase(name: str) -> Phase:
    """Return the named phase.

    Args:
        name: A phase's ``name`` field.

    Returns:
        That ``Phase``.

    Raises:
        ValueError: If ``name`` names no phase in ``PHASES``.
    """
    if name not in _BY_NAME:
        raise ValueError(f"no such phase {name!r}; choose one of {tuple(_BY_NAME)}")
    return _BY_NAME[name]


def _teacher_blocks(phase: Phase) -> int:
    """Return the block count of the network that taught ``phase``.

    The recipe's teachers are the *pipeline's own* earlier, smaller
    checkpoints -- not the phase immediately before this one -- so this reads
    ``teacher_from``'s own ``blocks``, not ``phase``'s.

    Args:
        phase: A phase with a teacher.

    Returns:
        ``teacher_from``'s ``blocks``.

    Raises:
        ValueError: If ``phase`` has no teacher.
    """
    if phase.teacher_from is None:
        raise ValueError(f"{phase.name} has no teacher_from, so it has no teacher")
    return _phase(phase.teacher_from).blocks


def _checkpoint(name: str) -> Path:
    """Return the newest checkpoint a phase has written, or raise.

    ``toad`` writes ``RUNS/{name}_{update:06d}.pt`` every
    ``CHECKPOINT_EVERY`` updates and this runner always names a phase's run
    after the phase itself (``--name``), so ``{name}_*.pt`` is exactly that
    phase's checkpoint family. The zero-padded update number sorts
    lexicographically the same as numerically, so the last glob match is the
    latest one -- no separate "final" marker is needed.

    Args:
        name: A phase's ``name`` field.

    Returns:
        The newest matching checkpoint.

    Raises:
        FileNotFoundError: If ``name`` has no checkpoint on disk. A missing
            teacher must fail loudly, not train unanchored.
    """
    checkpoints = sorted(toad.RUNS.glob(f"{name}_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"{name} has no checkpoint matching {toad.RUNS}/{name}_*.pt "
            f"-- run {name} to completion first"
        )
    return checkpoints[-1]


def _flags(phase: Phase) -> list[str]:
    """Translate one ``Phase`` into the deprecated flags ``toad`` exposed.

    Kept for one release so helper imports and compatibility tests can inspect
    the former CLI mapping. Production curriculum execution uses
    :func:`phase_config` and :func:`toad.run` directly.

    Args:
        phase: The phase to run.

    Returns:
        The argument list ``toad.main`` should parse.

    Raises:
        FileNotFoundError: If ``phase`` names a teacher whose checkpoint is
            not on disk.
        KeyError: If ``phase.reward`` names no reward ``toad`` can select.
    """
    flags = [
        "--name",
        phase.name,
        "--blocks",
        str(phase.blocks),
        "--total-steps",
        str(phase.steps),
        "--teacher-kl-cost",
        str(phase.teacher_kl_cost),
        "--lr",
        str(phase.lr),
        "--entropy-cost",
        str(phase.entropy_cost),
        "--lmb",
        str(phase.lmb),
    ]
    flags += REWARD_FLAGS[phase.reward]
    if phase.teacher_from is not None:
        flags += [
            "--teacher",
            str(_checkpoint(phase.teacher_from)),
            "--teacher-blocks",
            str(_teacher_blocks(phase)),
        ]
    return flags


def phase_config(phase: Phase) -> ToadConfig:
    """Resolve one declared curriculum phase into nested Pydantic models."""
    teacher = _checkpoint(phase.teacher_from) if phase.teacher_from else None
    return ToadConfig(
        model=ModelConfig(blocks=phase.blocks, channels=toad.CHANNELS),
        optimizer=OptimizerConfig(
            lr=phase.lr,
            lmb=phase.lmb,
            entropy_cost=phase.entropy_cost,
            teacher_kl_cost=phase.teacher_kl_cost,
        ),
        population=PopulationConfig(
            teacher_checkpoint=teacher,
            teacher_blocks=_teacher_blocks(phase) if teacher is not None else None,
        ),
        runtime=RuntimeConfig(total_environment_steps=phase.steps),
        curriculum=CurriculumConfig(
            phase=phase.name,
            reward_field=phase.reward,
        ),
    )


def _parser() -> argparse.ArgumentParser:
    """Return this runner's command line.

    Returns:
        The argument parser ``main`` parses ``sys.argv`` with.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=tuple(_BY_NAME),
        help="which phase of the curriculum to run",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=JSON_VALUE",
        help="validated dotted override applied after the declared phase",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    r"""Resolve one phase config and run the native Lightning entry point.

    No phase resumes another implicitly. Operational resume remains an explicit
    ``--set runtime.resume=\"...\"`` override.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    arguments = _parser().parse_args(argv)
    phase = _phase(arguments.phase)
    config = apply_overrides(phase_config(phase), arguments.overrides)
    LOGGER.info("curriculum: running %s", phase.name)
    toad.run(config)


if __name__ == "__main__":
    main()
