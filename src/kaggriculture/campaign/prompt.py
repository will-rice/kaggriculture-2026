"""Build the sandbox one codex call works in.

``AGENTS.md`` is codex's standing context: the task prompt phase 1 wrote,
how to use the harness, and the doctrine. ``PROMPT.md`` is FAMOU's mutation
instruction. ``feedback.md`` is what the evaluator said about the parent.
Nothing in the sandbox names where an opponent lives.

The task prompt phase 1 wrote already contains a harness section, but its
commands are hard-coded to the worktree that wrote them. ``_strip_stale_harness``
removes that block so ``AGENTS.md`` can carry a version built fresh from
``config.ROOT`` at build time, correct wherever the repository is checked out.
"""

import logging
import re
import shutil
from pathlib import Path
from typing import Literal

from kaggle_environments.envs.kaggriculture import kaggriculture as engine_module

from kaggriculture.campaign import config, harness

LOGGER = logging.getLogger(__name__)

TASK_PROMPT = Path(__file__).with_name("task_prompt.md")

# A program id becomes a directory name under config.SANDBOXES; it must be a
# single path segment so a crafted id (e.g. "..") can never resolve outside
# the sandboxes directory before it is rmtree'd.
_PROGRAM_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

# Matches the fenced ```bash block containing `uv run --project` and the
# "Valid measured opponents" sentence that follows it, however it is spaced.
_STALE_HARNESS_RE = re.compile(
    r"```bash\n.*?uv run --project.*?\n```\n\n"
    r"Valid measured opponents[^\n]*\n(?:[^\n]*\n)*?\n",
    re.DOTALL,
)

# {project} is filled with ``config.ROOT`` at module load, so the commands are
# right wherever this repository is checked out; {workers} is the harness's own
# cap, so the sandbox is never promised more cores than the command allows.
HARNESS_SECTION = """
## Testing what you write

```bash
uv run --project {project} campaign check child.py
uv run --project {project} campaign play child.py --vs NAME... --seeds A-B --workers N
```

- `campaign check child.py` — loads the file as Kaggle does and plays one
  episode against itself; reports the worst per-step latency (budget 0.5 s).
- `campaign play child.py --vs NAME... --seeds A-B --workers N` — plays the
  named opponents on the engine, both seats. Opponents x seeds x 2 seats may
  not exceed 16 games and `--workers` may not exceed {workers}; the command refuses
  anything larger, because the rest of the box is running the loop. Opponent
  names are those in feedback.md. Seeds are your choice; the evaluator uses
  others.

The evaluator measures for real after you finish; use these only to make
sure the file runs and does what you intended.
""".format(project=config.ROOT, workers=harness.SANDBOX_WORKER_CAP)

DOCTRINE = """
## Doctrine

Measure opponents through the harness. Never read their source, never ask
for it, never reconstruct it: the gate rejects code that resembles any
opponent's. Your agent is ours.
"""

FULL = """Read AGENTS.md, then parent.py and feedback.md.

Write child.py: a complete agent file implementing the interface in
AGENTS.md. Start from parent.py. Keep what the feedback says is winning;
change what is losing, and say in a docstring at the top what you changed
and why. The weakest opponent is where the score moves most.

Do not read, request, or reconstruct any opponent's source; the gate rejects
code resembling any opponent's.

Run `campaign check child.py` before you finish. Do not modify parent.py.

This session is stopped after {minutes} minutes. Whatever child.py holds at
that moment is what gets evaluated, so write a complete child first and
refine it in place; do not leave it half-edited while you run experiments.
""".format(minutes=config.MUTATION_TIMEOUT_SECONDS // 60)

CROSS = """Read AGENTS.md, then parent.py, inspiration.py and feedback.md.

Write child.py: a complete agent file implementing the interface in
AGENTS.md, combining the strongest ideas of parent.py and inspiration.py.
Say in a docstring at the top which idea came from which and why the
combination should beat both.

Do not read, request, or reconstruct any opponent's source; the gate rejects
code resembling any opponent's.

Run `campaign check child.py` before you finish. Do not modify parent.py or
inspiration.py.

This session is stopped after {minutes} minutes. Whatever child.py holds at
that moment is what gets evaluated, so write a complete child first and
refine it in place; do not leave it half-edited while you run experiments.
""".format(minutes=config.MUTATION_TIMEOUT_SECONDS // 60)


def _strip_stale_harness(text: str) -> str:
    """Remove phase 1's hard-coded harness block from the task prompt text.

    Args:
        text: The raw contents of ``task_prompt.md``.

    Returns:
        ``text`` with the stale ```bash harness block and the "Valid measured
        opponents" sentence that follows it removed.
    """
    stripped, count = _STALE_HARNESS_RE.subn("", text)
    if count != 1:
        raise ValueError(
            f"expected exactly one stale harness block in task_prompt.md, found {count}"
        )
    return stripped


def _feedback_lines(
    feedback: dict[str, float],
    weakest: str,
    weights: dict[str, float],
    failures: list[str],
) -> list[str]:
    """Render the parent's per-opponent rates, pool weights, and lineage failures.

    Args:
        feedback: The parent's fast-eval win rate per opponent name.
        weakest: The name of the pool's weakest opponent, or "" if unknown.
        weights: The pool weight per opponent name.
        failures: Recent failure descriptions from this lineage.

    Returns:
        Lines of a markdown document naming opponents only, never paths.
    """
    lines = [
        "# Feedback on parent.py",
        "",
        "| opponent | parent win rate | weight |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| {name} | {feedback.get(name, float('nan')):.3f} "
        f"| {weights.get(name, 0.0):.2f} |"
        for name in weights
    ]
    if weakest:
        lines += ["", f"Weakest opponent: **{weakest}**."]
    if failures:
        lines += ["", "Recent failures in this lineage (do not repeat):"]
        lines += [f"- {failure}" for failure in failures]
    return lines


def _sandbox_path(program_id: str) -> Path:
    """Resolve ``program_id`` to a directory directly under ``config.SANDBOXES``.

    Args:
        program_id: The child program id.

    Returns:
        ``config.SANDBOXES / program_id``.

    Raises:
        ValueError: If ``program_id`` is not a single safe path segment, or
            resolves outside ``config.SANDBOXES`` (e.g. ``".."``).
    """
    if not _PROGRAM_ID_RE.fullmatch(program_id):
        raise ValueError(f"unsafe program_id: {program_id!r}")
    box = config.SANDBOXES / program_id
    if box.resolve().parent != config.SANDBOXES.resolve():
        raise ValueError(f"unsafe program_id: {program_id!r}")
    return box


def build_sandbox(
    program_id: str,
    kind: Literal["full", "cross"],
    parent: Path,
    inspiration: Path | None,
    feedback: dict[str, float],
    weakest: str,
    weights: dict[str, float],
    failures: list[str],
) -> Path:
    """Build the sandbox directory one codex mutation call works in.

    Args:
        program_id: The child program id; also the sandbox's directory name.
        kind: "full" for a from-parent mutation, "cross" to combine parent
            and inspiration.
        parent: Path to the parent agent's program file.
        inspiration: Path to the crossover partner's program file, required
            when ``kind`` is "cross".
        feedback: The parent's fast-eval win rate per pool opponent name.
        weakest: The pool's weakest opponent name, or "" if unknown.
        weights: The pool weight per opponent name.
        failures: Recent failure descriptions from this lineage.

    Returns:
        The path to the built sandbox directory, rebuilt clean each call.

    Raises:
        ValueError: If ``program_id`` is not a safe single path segment
            directly under ``config.SANDBOXES``.
    """
    box = _sandbox_path(program_id)
    if box.exists():
        shutil.rmtree(box)
    (box / "engine").mkdir(parents=True)

    task_prompt = _strip_stale_harness(TASK_PROMPT.read_text(encoding="utf-8"))
    (box / "AGENTS.md").write_text(
        task_prompt + HARNESS_SECTION + DOCTRINE, encoding="utf-8"
    )

    shutil.copy(parent, box / "parent.py")
    if inspiration is not None:
        shutil.copy(inspiration, box / "inspiration.py")
    shutil.copy(engine_module.__file__, box / "engine" / "kaggriculture.py")

    lines = _feedback_lines(feedback, weakest, weights, failures)
    (box / "feedback.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (box / "PROMPT.md").write_text(CROSS if kind == "cross" else FULL, encoding="utf-8")

    LOGGER.info("built sandbox %s (%s)", box, kind)
    return box
