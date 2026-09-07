"""Take newly published kernels into the opponent pool.

The pool is what a candidate must finish above, so it decides what "better"
means. Left alone it becomes a closed set: champions join on every promotion
and the weakest opponent makes way, so within hours it is the campaign's own
lineage playing itself. That is the ratchet working, and it is also how a
search stops meeting anything it did not produce.

A harvest is the other half. The competition publishes kernels daily, and a
newly published one is an agent this lineage has never been selected against
-- which is the whole of what a held-out opponent was for, arriving
continuously and at the strength of the current field rather than frozen at
the strength of whenever it was pinned.

Discovery, extraction and the compiled-kernel build are `kernel_watch`'s, and
they are careful in ways worth keeping: a notebook is never executed, the
payload's own asserted digest is verified, and the entrypoint is resolved the
way the engine's loader resolves it. What is added here is the other end --
checking a kernel can actually play, vendoring it, and putting it in the pool.

Nothing decides whether it stays. The next gate measures its pairings and
`Pool.trim` keeps the highest-rated `POOL_SIZE`, so a weak harvest leaves on
the first promotion and a strong one displaces a champion. That is the same
rule everything else in the pool lives by.
"""

import logging
import re
import shutil
from pathlib import Path

from kaggriculture.campaign import config, harness, kernel_watch
from kaggriculture.campaign.pool import Pool

LOGGER = logging.getLogger(__name__)

# Turns a kernel ref into an opponent name. The vendored opponents are named
# for their author and kernel -- `boatlee_v29`, `pilkwang_economic` -- and the
# copy check keys its corpus on the directory, so the name is what a copy
# verdict says out loud. It must never be a path.
UNSAFE = re.compile(r"[^a-z0-9]+")
NAME_LIMIT = 40


def opponent_name(ref: str) -> str:
    """``author/some-kernel-slug`` as a directory name, never a path.

    Args:
        ref: The kernel ref.

    Returns:
        A lowercase, underscore-separated name.
    """
    user, _, slug = ref.partition("/")
    name = UNSAFE.sub("_", f"{user}_{slug}".lower()).strip("_")
    return name[:NAME_LIMIT].rstrip("_")


def playable(entry: Path, steps: int = 720) -> str:
    """Why this kernel cannot be an opponent, or "" if it can.

    An opponent is played thousands of times by every later candidate, so one
    that crashes or runs long is not a hard opponent, it is a broken pool: the
    loop halts on an opponent crash rather than scoring around it, and a slow
    one multiplies into every evaluation the campaign ever runs.

    Args:
        entry: The kernel's ``main.py``.
        steps: Turns to play. A whole season by default, because a kernel
            that dies on turn six hundred is one the pool would die on
            too; a test asking only whether the check reads latency at
            all passes a handful, since a slow agent is slow on turn one.

    Returns:
        A sentence naming the problem, or "" when it plays.
    """
    report = harness.check(entry, steps=steps)
    if not report.loaded or report.error is not None:
        return report.error or "did not load"
    if report.worst_step_seconds > harness.LATENCY_BUDGET:
        return (
            f"worst step {report.worst_step_seconds:.3f}s, over the "
            f"{harness.LATENCY_BUDGET}s an opponent may take"
        )
    return ""


def vendor(entry: Path, name: str) -> Path:
    """Copy a kernel's directory under `config.OPPONENTS` and return its entry.

    The whole directory, because a kernel is not always one file: the compiled
    ones carry headers, a tape and the ``agent.so`` built beside their
    ``main.py``, and an agent without them loads and then fails on turn one.

    Args:
        entry: The kernel's ``main.py``, wherever it was built or extracted.
        name: The opponent name to vendor it under.

    Returns:
        The vendored ``main.py``.
    """
    target = config.OPPONENTS / name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(entry.parent, target)
    return target / entry.name


def harvest(limit: int, pool_file: Path, author: str | None = None) -> list[str]:
    """Discover, check and enroll newly published kernels as opponents.

    Args:
        limit: How many new refs to take.
        pool_file: The pool to add them to.
        author: Restrict to one author, or None for the whole competition.

    Returns:
        The opponent names added.
    """
    refs = kernel_watch.discover(author, limit)
    if not refs:
        LOGGER.info("nothing new published")
        return []
    pool = Pool.load(pool_file) if pool_file.exists() else Pool.initial()
    added: list[str] = []
    for ref in refs:
        name = opponent_name(ref)
        if name in pool.opponents:
            LOGGER.info("%s: already in the pool as %s", ref, name)
            continue
        entry = _entrypoint(ref)
        if entry is None:
            LOGGER.info("%s: no agent this can build or extract; skipped", ref)
            continue
        problem = playable(entry)
        if problem:
            LOGGER.info("%s: %s; skipped", ref, problem)
            continue
        vendored = vendor(entry, name)
        pool.opponents[name] = str(vendored)
        added.append(name)
        LOGGER.info("%s: enrolled as %s", ref, name)
    # Remembered whatever happened to them: a kernel that could not be built
    # will not build tomorrow either, and re-checking every one of them every
    # day is how a daily job turns into an hourly one.
    kernel_watch.remember(refs)
    if added:
        pool.save(pool_file)
        LOGGER.info("pool: %d added, now %d opponents", len(added), len(pool.opponents))
    return added


def _entrypoint(ref: str) -> Path | None:
    """A runnable ``main.py`` for one kernel, built or extracted.

    Compiled first, because such a kernel's ``main.py`` extracts perfectly
    well as source and is useless on its own: it loads an ``agent.so`` that
    exists only once the notebook's build cell has run.
    """
    built = kernel_watch.build_compiled(ref)
    if built is not None:
        return built
    source = kernel_watch.extract(ref)
    if source is None:
        return None
    directory = kernel_watch.WORK / ref.replace("/", "__")
    directory.mkdir(parents=True, exist_ok=True)
    entry = directory / "main.py"
    entry.write_text(source, encoding="utf-8")
    return entry
