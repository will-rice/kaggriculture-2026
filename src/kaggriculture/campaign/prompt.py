"""Build the sandbox one codex session works in.

``AGENTS.md`` is codex's standing context: the task prompt phase 1 wrote,
how to use the harness, and the doctrine. ``child.py`` is the champion,
copied in; it is the file the session edits in place until it clears the
bar. ``PROMPT.md`` carries the instruction drawn for this session and the
budget. ``feedback.md`` states the bar and what the evaluator measured
about the champion. Nothing in the sandbox names where an opponent lives.

The task prompt phase 1 wrote already contains a harness section, but its
commands are hard-coded to the worktree that wrote them. ``_strip_stale_harness``
removes that block so ``AGENTS.md`` can carry a version built fresh from
``config.ROOT`` at build time, correct wherever the repository is checked out.
"""

import logging
import re
import shutil
from pathlib import Path

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

# FAMOU appendix C.2's five rewrite instructions. One is drawn per session,
# so eight sessions starting from the same champion are pushed eight
# different ways. The caller draws the pair, records the name on the program
# it produces, and hands ``build_sandbox`` the text; a crossover session is
# handed CROSS the same way.
INSTRUCTIONS: tuple[tuple[str, str], ...] = (
    ("improve", "Improve child.py's performance against the pool."),
    (
        "different",
        "Replace child.py with a completely different algorithm for the same game.",
    ),
    (
        "inspired",
        "Create a novel approach inspired by child.py that works fundamentally "
        "differently.",
    ),
    (
        "restructure",
        "Redesign child.py's core components, keeping what the feedback says wins.",
    ),
    (
        "tune",
        "Tune child.py's constants and thresholds only; keep its structure.",
    ),
)

CROSS = (
    "Fold the strongest ideas of inspiration.py into child.py. Say in the "
    "docstring which idea came from which and why the combination should beat "
    "both. Do not modify inspiration.py."
)

# Every instruction is wrapped in these at build time, so the bar, the
# doctrine and the budget are stated once and cannot drift between variants.
PREAMBLE = """Read AGENTS.md and feedback.md, then child.py.

child.py is the champion, and it is the file you edit. Keep editing it in
place until it clears the bar feedback.md states, testing it with the
harness as you go. Stop as soon as it clears both parts of the bar.

Your instruction for this session:

"""

POSTAMBLE = """

Keep what the feedback says is winning; change what is losing, and say in a
docstring at the top of child.py what you changed and why. The weakest
opponent named in feedback.md is where the score moves most.

Do not read, request, or reconstruct any opponent's source; the gate rejects
code resembling any opponent's.

Run `campaign check child.py` before you finish.

Your budget is {minutes} minutes; the session is stopped then, and whatever
child.py holds at that moment is what is evaluated. Keep it complete and
runnable throughout; do not leave it half-edited while you run experiments.
""".format(minutes=config.SESSION_LIMIT_SECONDS // 60)


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
    bar: dict[str, float],
    rates: dict[str, float],
    weights: dict[str, float],
    weakest: str,
    failures: list[str],
    started_from: str,
) -> list[str]:
    """Render the bar, the champion's rates, the pool weights, and past failures.

    Args:
        bar: What the edit has to beat; ``bar["fitness"]`` is the champion's
            weighted pool fitness.
        rates: The champion's fast-eval win rate per opponent name.
        weights: The pool weight per opponent name.
        weakest: The name of the pool's weakest opponent, or "" if unknown.
        failures: Recent failure descriptions from this lineage.
        started_from: The champion's id, which is also its pool opponent name.

    Returns:
        Lines of a markdown document naming opponents only, never paths.
    """
    lines = [
        "# Feedback on child.py",
        "",
        "## The bar",
        "",
        f"child.py is the champion `{started_from}`. Keep editing it until it "
        "clears both of these, and stop as soon as it does:",
        "",
        f"1. A weighted pool fitness above **{bar['fitness']:.4f}**, the "
        "champion's, measured over every opponent in the table below.",
        "2. A winning head-to-head record against the champion itself, which "
        f"is the pool opponent named `{started_from}`.",
        "",
        "Both, because the game is non-transitive: an edit told only to beat "
        "the champion breeds a champion-counter. An edit that clears the first "
        "alone is still worth writing.",
        "",
        "## The champion against the pool",
        "",
        "| opponent | champion win rate | weight |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| {name} | {rates.get(name, float('nan')):.3f} "
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
    champion: Path,
    instruction: str,
    inspiration: Path | None,
    bar: dict[str, float],
    rates: dict[str, float],
    weights: dict[str, float],
    weakest: str,
    failures: list[str],
    started_from: str,
) -> Path:
    """Build the sandbox directory one codex session works in.

    The champion is copied in as ``child.py``: the session edits that file in
    place until it clears the bar, and whatever it holds when the budget runs
    out is what is evaluated.

    Args:
        program_id: The child program id; also the sandbox's directory name.
        champion: Path to the champion's program file, copied in as child.py.
        instruction: The drawn instruction's text — one of ``INSTRUCTIONS``'
            second elements, or ``CROSS``. It is recorded in PROMPT.md.
        inspiration: Path to the crossover partner's program file, or None.
        bar: What the edit has to beat; ``bar["fitness"]`` is the champion's
            weighted pool fitness.
        rates: The champion's fast-eval win rate per pool opponent name.
        weights: The pool weight per opponent name.
        weakest: The pool's weakest opponent name, or "" if unknown.
        failures: Recent failure descriptions from this lineage.
        started_from: The champion's id, which is also its pool opponent name.

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

    child = box / "child.py"
    shutil.copy(champion, child)
    # The gate writes a champion read-only so nothing can edit the file the
    # pool plays, and `shutil.copy` carries that mode across. This copy is
    # the one file the session must be able to write.
    child.chmod(0o644)
    if inspiration is not None:
        shutil.copy(inspiration, box / "inspiration.py")
    shutil.copy(engine_module.__file__, box / "engine" / "kaggriculture.py")

    lines = _feedback_lines(bar, rates, weights, weakest, failures, started_from)
    (box / "feedback.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (box / "PROMPT.md").write_text(PREAMBLE + instruction + POSTAMBLE, encoding="utf-8")

    LOGGER.info("built sandbox %s from %s", box, started_from)
    return box
