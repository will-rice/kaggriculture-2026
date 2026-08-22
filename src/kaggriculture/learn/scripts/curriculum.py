"""Toad Brigade's five-phase curriculum, as a table instead of a memory.

Every arm this project has run executed phase 1 with phase 2's teacher cost,
against a behaviour clone the recipe uses nowhere, because those numbers lived
in a command line nobody could check. This module is the fix: the five
phases, transcribed verbatim from the recipe into ``PHASES``, and a runner
that translates one row of that table into the flags ``toad`` already
exposes rather than a remembered invocation.

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
report for why, and pass ``--resume`` to ``toad`` by hand if that is
wanted.
"""

import argparse
import dataclasses
import logging
from pathlib import Path

from kaggriculture.learn.scripts import toad

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Phase:
    """One phase of the curriculum, transcribed verbatim from the recipe.

    Attributes:
        name: This phase's identifier -- also its wandb run name and
            checkpoint-file prefix, via ``--name``.
        blocks: Residual blocks in this phase's own trunk.
        steps: Environment steps this phase trains for.
        reward: Which ``Trajectory`` field the arm trains on: ``"shaped"``
            (phase 1's dense per-turn reward) or ``"sparse"`` (phases 2-5's
            terminal-only ``GameResultReward``).
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
    reward: str
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
        reward="shaped",
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
    """Translate one ``Phase`` into the flags ``toad`` exposes.

    Args:
        phase: The phase to run.

    Returns:
        The argument list ``toad.main`` should parse.

    Raises:
        FileNotFoundError: If ``phase`` names a teacher whose checkpoint is
            not on disk.
        ValueError: If ``phase.reward`` is neither ``"shaped"`` nor ``"sparse"``.
    """
    if phase.reward not in ("shaped", "sparse"):
        raise ValueError(
            f"{phase.name}: reward must be 'shaped' or 'sparse', got {phase.reward!r}"
        )
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
    if phase.reward == "sparse":
        flags.append("--sparse")
    if phase.teacher_from is not None:
        flags += [
            "--teacher",
            str(_checkpoint(phase.teacher_from)),
            "--teacher-blocks",
            str(_teacher_blocks(phase)),
        ]
    return flags


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
    return parser


def main() -> None:
    """Resolve one phase's flags and hand them to ``toad``'s own entry point.

    Does not duplicate the training loop: every phase runs through
    ``toad.main``, which is the same code path Tasks 1-3 tested.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    arguments = _parser().parse_args()
    phase = _phase(arguments.phase)
    flags = _flags(phase)
    LOGGER.info("curriculum: running %s as %s", phase.name, " ".join(flags))
    toad.main(flags)


if __name__ == "__main__":
    main()
