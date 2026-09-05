"""Find a newly published kernel, tune it, and gate it -- the whole upgrade path.

Three ladder upgrades in a week came from one competitor publishing v54, v56
and v58, and each was worth more than any learning we attempted. The manual
version of this took two hours per kernel and one mistake nearly cost a
submission slot, so it is written down here.

What it does, in order:

1. Lists public kernels for the competition and reports any whose ref we have
   not already recorded in ``run/kernel-watch/seen.json``.
2. Extracts each new one statically -- ``ast.literal_eval`` of the payload
   assignment, then base85 and zlib -- and verifies the digest the notebook
   itself asserts. It never executes a notebook cell.
3. Resolves the entrypoint the way the engine's loader does. These artifacts
   shadow ``agent``, sometimes twice, and the surviving definition is not the
   one that plays; getting this wrong measures a different agent than the one
   a submission would run.
4. Applies the two thresholds this author leaves conservative across the whole
   lineage -- ``preempt_horizon`` 2 -> 7 and ``preempt_minimum_price_ratio``
   0.45 -> 0.225 -- to any kernel that carries a ``ResidualConfig`` literal.
   Measured on v56 that is worth 0.9922 of 128 mirror games, and it applies
   unchanged to v58.
5. Gates the tuned artifact against the FIELD lineages over every exam seed,
   both seat orderings.

It stops there. Submitting is a slot-spending, externally visible action and
stays a human decision; the report ends with the command to run.
"""

import argparse
import ast
import base64
import hashlib
import json
import logging
import re
import zlib
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

COMPETITION = "kaggriculture"
SEEN = Path("run/kernel-watch/seen.json")
WORK = Path("run/kernel-watch/kernels")

# What marks a cell as holding the agent: a def the engine can call, or a
# module-level binding of one, which is how a factory-built policy is
# published.
ENTRYPOINT = r"^(?:def (?:agent|kaggle_agent)\(|(?:agent|main|policy)\s*=)"

# The two fields, and the values 128-game gates picked out of a 44-parameter
# sweep. The horizon curve is unimodal (2 -> 0.500, 5 -> 0.914, 7 -> 0.930,
# 9 -> 0.844, 12 -> 0.656) and the pair sits on a plateau, not a spike.
TUNED_HORIZON = 7
TUNED_PRICE_RATIO = 0.225


def main() -> None:
    """Scan, extract, tune and gate; print what a human should do next."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author", default=None, help="only this kernel author")
    parser.add_argument("--limit", type=int, default=5, help="new kernels to process")
    parser.add_argument("--gate-seeds", type=int, default=64, help="exam seeds to play")
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    fresh = discover(args.author, args.limit)
    if not fresh:
        LOGGER.info("no new kernels")
        return
    for ref in fresh:
        try:
            report(ref, args.gate_seeds, args.workers)
        except Exception:
            LOGGER.exception("%s failed; continuing", ref)
    remember(fresh)


def discover(author: str | None, limit: int) -> list[str]:
    """Return refs published since the last scan, newest first.

    Args:
        author: Restrict to one author, or None for the whole competition.
        limit: How many new refs to return.

    Returns:
        The refs not already in the seen file.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    listed: dict[str, str] = {}
    for order in ("dateCreated", "voteCount", "hotness"):
        found = api.kernels_list(
            competition=COMPETITION, sort_by=order, user=author, page_size=50
        )
        for kernel in found or ():
            if kernel is not None:
                listed[kernel.ref] = kernel.title
    seen = set(json.loads(SEEN.read_text())) if SEEN.exists() else set()
    fresh = [ref for ref in listed if ref not in seen][:limit]
    LOGGER.info("%d listed, %d new", len(listed), len(fresh))
    for ref in fresh:
        LOGGER.info("   new: %s | %s", ref, listed[ref])
    return fresh


def remember(refs: list[str]) -> None:
    """Record refs so the next scan reports only what is newer.

    Args:
        refs: The refs just processed.
    """
    seen = set(json.loads(SEEN.read_text())) if SEEN.exists() else set()
    seen.update(refs)
    SEEN.parent.mkdir(parents=True, exist_ok=True)
    SEEN.write_text(json.dumps(sorted(seen), indent=1))


def report(ref: str, gate_seeds: int, workers: int) -> None:
    """Extract, tune and gate one kernel, logging every step's evidence.

    Args:
        ref: The kernel ref to process.
        gate_seeds: How many exam seeds to play, both seat orderings.
        workers: Arena processes.
    """
    # A compiled kernel is tried first, because its main.py extracts perfectly
    # well as source and is useless on its own: it loads an agent.so that only
    # exists once the notebook's own build cell has run. Left to the source
    # path it resolves, then fails on the first turn.
    built = build_compiled(ref)
    if built is not None:
        rate, low, high, games = gate(built, gate_seeds, workers)
        LOGGER.info(
            "%s [compiled]: FIELD %.4f Wilson [%.4f, %.4f] over %d games",
            ref,
            rate,
            low,
            high,
            games,
        )
        if low > 0.80:
            LOGGER.info("   CANDIDATE. Archive at %s", built.parent)
        return
    source = extract(ref)
    if source is None:
        LOGGER.info("%s: no verifiable agent payload; skipped", ref)
        return
    tuned = retune(source)
    directory = WORK / ref.replace("/", "__")
    directory.mkdir(parents=True, exist_ok=True)
    plain = directory / "main.py"
    plain.write_text(source, encoding="utf-8")
    entry = resolved_entrypoint(plain)
    LOGGER.info("%s: entrypoint %s", ref, entry)
    candidates = {"as_published": plain}
    if tuned is not None:
        patched = directory / "main_tuned.py"
        patched.write_text(tuned, encoding="utf-8")
        candidates["tuned"] = patched
    for label, path in candidates.items():
        rate, low, high, games = gate(path, gate_seeds, workers)
        LOGGER.info(
            "%s [%s]: FIELD %.4f Wilson [%.4f, %.4f] over %d games",
            ref,
            label,
            rate,
            low,
            high,
            games,
        )
        # The bar is what it takes to be worth standing over the agent we would
        # otherwise stand: our tuned_v58 gates at 0.700 here and shopforge at
        # 0.906. Those two are not a like-for-like comparison -- every
        # candidate is dropped from its own field, so shopforge is scored on
        # four lineages and ours on five including shopforge. The per-lineage
        # rates above are the ones to read when ranking two candidates; this
        # number only has to answer "worth a closer look".
        #
        # A kernel that republishes a lineage we already hold ties itself at
        # 0.5 in that row and reads lower than it plays -- scanning shopforge
        # itself gives 0.85, not 0.906. That row is the tell, and it is worth
        # more than the mean: an opponent it cannot beat is one it already is.
        if low > 0.80:
            LOGGER.info(
                "   CANDIDATE. To submit: uv run --with kaggle kaggle competitions "
                "submit %s -f <archive> -m '%s %s'",
                COMPETITION,
                ref,
                label,
            )


def build_compiled(ref: str) -> Path | None:
    """Build a kernel that ships a compiled agent, and return its entrypoint.

    The strongest agent in this competition is C++: its notebook writes
    policy.cpp, several headers and a pinned tape, then compiles them into an
    agent.so that main.py loads through ctypes. Extraction sees a main.py that
    imports a shared library which does not exist, so it read as a kernel with
    no agent, and the gate then reported the ctypes failure. Since the top of
    the field publishes this way, the scan builds it.

    Everything is taken from the notebook: the ``%%writefile`` targets are its
    own, and the compiler line is the one its own build cell runs. The build
    happens in the kernel's work directory because the agent needs its .so
    beside it, and a compiled artifact cannot be handed around as source.

    Args:
        ref: The kernel ref, already pulled.

    Returns:
        The path to a runnable main.py, or None if this kernel is not one.
    """
    import subprocess

    directory = WORK / ref.replace("/", "__")
    cells = published_cells(directory)
    if cells is None:
        return None
    build: list[str] | None = None
    wrote = False
    for cell in cells:
        text = "".join(cell["source"])
        first = text.split("\n")[0]
        if text.lstrip().startswith("%%") and "writefile" in first:
            target = directory / first.split()[-1]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("\n".join(text.split("\n")[1:]), encoding="utf-8")
            wrote = True
            continue
        found = compiler_command(text)
        if found is not None:
            build = found
    if not wrote or build is None:
        return None
    LOGGER.info("%s: compiled agent, building with %s", ref, build[0])
    result = subprocess.run(build, cwd=directory, capture_output=True, text=True)
    if result.returncode != 0:
        LOGGER.info("%s: build failed: %s", ref, result.stderr.strip()[:200])
        return None
    entry = directory / "main.py"
    return entry if entry.exists() else None


def compiler_command(text: str) -> list[str] | None:
    """Return the compiler invocation a build cell runs, read as a literal.

    Only a list of string constants is accepted, so the command is read rather
    than evaluated and a cell that computes its arguments is declined instead
    of executed.

    Args:
        text: One notebook code cell's source.

    Returns:
        The argument list, or None if the cell runs no compiler.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.List):
            continue
        parts = [
            element.value
            for element in node.value.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        ]
        if len(parts) == len(node.value.elts) and parts and parts[0] in ("g++", "gcc"):
            return parts
    return None


def extract(ref: str) -> str | None:
    """Return one kernel's agent source, digest-checked, without executing it.

    Args:
        ref: The kernel ref to pull.

    Returns:
        The decoded agent source, or None if the notebook ships no payload
        whose digest it asserts.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    directory = WORK / ref.replace("/", "__")
    directory.mkdir(parents=True, exist_ok=True)
    api.kernels_pull(ref, path=str(directory), metadata=True)
    cells = published_cells(directory)
    if cells is None:
        return None
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        payload = payload_literal(text)
        if payload is None:
            continue
        try:
            raw = zlib.decompress(base64.b85decode(payload))
        except (ValueError, zlib.error):
            # The assignment is named like a payload but is not b85+zlib.
            LOGGER.info("%s: payload is not b85+zlib; trying other cells", ref)
            continue
        asserted = re.search(r"digest == '([0-9a-f]{64})'", text)
        digest = hashlib.sha256(raw).hexdigest()
        if asserted and digest != asserted.group(1):
            raise ValueError(f"{ref}: payload digest {digest} != asserted")
        return raw.decode("utf-8")
    embedded = embedded_source(cells)
    if embedded is not None and loadable(embedded):
        LOGGER.info("%s: agent embedded as a string, UNVERIFIED -- opponent only", ref)
        return embedded
    return written_source(cells, ref)


def published_cells(directory: Path) -> list[dict[str, Any]] | None:
    """Return a kernel's code cells whether it was published as a book or not.

    Kaggle publishes a kernel as either a notebook or a plain ``.py`` script,
    and a script is not a lesser artifact: the ones found this way carry the
    same b85+zlib payload and the same shadowed ``agent`` definitions as any
    notebook. Globbing only for ``*.ipynb`` silently skipped every one of them,
    including a whole competitor's v28/v29/v30/v31 series.

    A script becomes a single code cell, which is what it is, and every step
    downstream -- payload digest, written source, entrypoint -- then works on
    it unchanged.

    Args:
        directory: Where the kernel was pulled.

    Returns:
        The code cells, or None if the pull produced neither form.
    """
    books = list(directory.glob("*.ipynb"))
    if books:
        cells = json.loads(books[0].read_text(encoding="utf-8"))["cells"]
        return [cell for cell in cells if cell["cell_type"] == "code"]
    scripts = list(directory.glob("*.py"))
    if not scripts:
        return None
    return [{"cell_type": "code", "source": scripts[0].read_text(encoding="utf-8")}]


def written_source(cells: list[dict[str, Any]], ref: str) -> str | None:
    """Return an agent published as plain source rather than a payload.

    Several kernels -- including the one holding the largest share of the
    ladder -- ship their agent as ordinary code, either written to ``main.py``
    by a cell magic or simply defined in a long cell. Those carry no digest to
    check, so the caller must treat them as unverifiable: usable as opponents,
    never as something we submit.

    Args:
        cells: The notebook's cells.
        ref: The kernel ref, for logging.

    Returns:
        The agent source, or None if no cell defines a loadable agent.
    """
    bodies = []
    written: dict[str, str] = {}
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        first = text.split("\n")[0]
        if text.lstrip().startswith("%%") and "writefile" in first:
            text = "\n".join(text.split("\n")[1:])
            name = first.split()[-1]
            if name.endswith(".py"):
                written[name[: -len(".py")]] = text
        elif text.lstrip().startswith("%%"):
            continue
        bodies.append(text)
    # The engine's loader takes the last callable a file leaves behind, so an
    # entrypoint need not be a def. The strongest agent we have measured ends
    # its module with ``agent = make_agent()`` and ``main = agent``, and a
    # ``def agent(`` test alone does not see it.
    defining = [t for t in bodies if re.search(ENTRYPOINT, t, re.M)]
    if not defining:
        return None
    LOGGER.info("%s: agent published as source, UNVERIFIED -- opponent only", ref)
    longest = inline_written(max(defining, key=len), written)
    if loadable(longest):
        return longest
    # The agent's definitions can span cells, so the longest one alone may
    # reference names defined elsewhere. Fall back to every code cell up to
    # and including the last one that defines an agent, in notebook order.
    last = len(bodies) - 1 - bodies[::-1].index(defining[-1])
    joined = inline_written("\n\n".join(bodies[: last + 1]), written)
    return joined if loadable(joined) else None


def inline_written(source: str, written: dict[str, str]) -> str:
    """Replace imports of the notebook's own written modules with their text.

    A kernel can split its agent across files it writes itself -- the strongest
    agent we have measured ships a readable controller that does
    ``from fieldbook_tapes import PLAN_SCRIPTS`` and a second cell that writes
    ``fieldbook_tapes.py``. Concatenating the cells does not help, because the
    import still names a module that exists only inside the notebook's own
    working directory. Its assembly cell substitutes the tape at exactly that
    import, so doing the same here reproduces what the kernel would run.

    Args:
        source: Candidate agent source.
        written: Module name to body, for every ``%%writefile`` cell.

    Returns:
        The source with those imports replaced by the module bodies.
    """
    if not written:
        return source
    lines = []
    for line in source.split("\n"):
        module = re.match(r"\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_]*)\b", line)
        if module and module.group(1) in written:
            lines.append(written[module.group(1)])
        else:
            lines.append(line)
    return "\n".join(lines)


def resolve_in_sandbox(source: str) -> object:
    """Run the engine's loader over untrusted source from a scratch directory.

    Resolving means EXECUTING it, so a stranger's code gets two things it must
    not have.

    It must not have our working directory. Published notebooks write files
    when they run: one dropped a `submission_agent.py` into the repository
    root, an earlier harvest overwrote `main.py`, and a scan wrote a
    `submission.csv` there through the one call site left unwrapped. Every
    caller resolves through here so the protection cannot be forgotten twice.

    It must not have the power to end the scan. A published kernel that cannot
    find its dataset prints "Reference-agent dataset not attached" and calls
    ``sys.exit``, and ``SystemExit`` derives from ``BaseException``, so the
    scan's ``except Exception`` never saw it -- measured, one such kernel
    killed a sweep at 80 of 343 with no traceback and no failed run to show
    for it. A kernel declining to load is an ordinary failure to resolve, so
    it is reported as one. ``KeyboardInterrupt`` still belongs to the operator
    and passes through untouched.

    Args:
        source: Candidate agent source.

    Returns:
        The callable the loader selects.
    """
    import os
    import tempfile

    from kaggle_environments.agent import get_last_callable

    origin = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="kernel-watch-") as sandbox:
        os.chdir(sandbox)
        try:
            return get_last_callable(source, path="main.py")
        except SystemExit as exit_call:
            raise RuntimeError(f"source called sys.exit({exit_call.code})") from None
        finally:
            os.chdir(origin)


def loadable(source: str) -> bool:
    """Return whether the engine's own loader can resolve this source.

    Args:
        source: Candidate agent source.

    Returns:
        True if a final callable resolves without raising.
    """
    try:
        resolve_in_sandbox(source)
    except Exception:
        return False
    return True


def embedded_source(cells: list[dict[str, Any]]) -> str | None:
    """Return an agent a cell carries as a plain string it later writes out.

    Some kernels neither compress their agent nor define it inline: they hold
    it in a raw string, print its size, and write it to disk. Extracting the
    surrounding cell gets the wrapper, whose last callable is not a policy, so
    the candidate resolves and then fails on the first turn. The agent itself
    is a string constant sitting in plain sight.

    Only a constant is read, never a call, and nothing is executed to find it.

    Args:
        cells: The notebook's code cells.

    Returns:
        The largest embedded source that defines an agent, or None.
    """
    best: str | None = None
    for cell in cells:
        try:
            tree = ast.parse("".join(cell["source"]))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            value = node.value
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                continue
            candidate = value.value
            if not re.search(ENTRYPOINT, candidate, re.M):
                continue
            if best is None or len(candidate) > len(best):
                best = candidate
    return best


def payload_literal(text: str) -> str | None:
    """Return the payload string a code cell assigns, without running it.

    Args:
        text: One notebook code cell's source.

    Returns:
        The assigned string, or None if the cell assigns no payload.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        target = getattr(node.targets[0], "id", "")
        if target not in {"payload", "PAYLOAD", "blob"}:
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            # Some notebooks build the payload with a call rather than
            # assigning a literal. There is no safe way to evaluate that
            # statically, and executing the cell to find out is the one thing
            # this scanner will not do, so the kernel is reported as
            # unverifiable and skipped.
            LOGGER.info("payload for %r is computed, not literal; skipping", target)
            return None
        return value if isinstance(value, str) else None
    return None


def retune(source: str) -> str | None:
    """Return the source with this author's two conservative thresholds moved.

    Args:
        source: A kernel's agent source.

    Returns:
        The patched source, or None if it carries no configuration literal to
        patch -- in which case the kernel is gated as published only.
    """
    if "'preempt_horizon': 2" not in source:
        return None
    patched = source.replace(
        "'preempt_horizon': 2", f"'preempt_horizon': {TUNED_HORIZON}"
    )
    ratio = f"'preempt_minimum_price_ratio': {TUNED_PRICE_RATIO}"
    return patched.replace(
        "'preempt_maximum_batch': 8", f"'preempt_maximum_batch': 8, {ratio}"
    )


def resolved_entrypoint(path: Path) -> str:
    """Return the callable the engine's own loader selects from a file.

    Args:
        path: The agent file.

    Returns:
        The selected callable's name. These artifacts shadow ``agent``, so the
        surviving definition is often not the one that plays.
    """
    selected = resolve_in_sandbox(path.read_text(encoding="utf-8"))
    return getattr(selected, "__name__", repr(selected))


def gate(path: Path, gate_seeds: int, workers: int) -> tuple[float, float, float, int]:
    """Play one candidate against the field lineages over the exam seeds.

    This deliberately does not gate against our own served agent. Doing so asks
    "does this beat us", and the agent worth adopting is the one that beats the
    FIELD -- which is not the same question, because the field is
    non-transitive. Measured: shopforge scores 0.6875 against v56 and would
    have missed the old 0.75 bar, while scoring 0.979 against the field against
    our own 0.779. The best agent we have found would have been scanned,
    logged and discarded by its own daily scan.

    Args:
        path: The candidate agent file.
        gate_seeds: How many exam seeds, both seat orderings.
        workers: Arena processes.

    Returns:
        The equal-weighted field rate, its Wilson bounds, and the games played.
    """
    import os
    import tempfile

    from kaggriculture.campaign import config, roster
    from kaggriculture.campaign.field_gate import score_field
    from kaggriculture.report import wilson_interval

    # A gated candidate is usually a stranger's kernel, and playing it executes
    # it: one wrote its own `main.py` over ours and left four agent files and a
    # tarball behind. `score_field` plays on the caller's own working
    # directory, so this is the one place a stranger's writes must be diverted.
    # The path is resolved to absolute before the sandbox is entered, because
    # every relative path a caller might pass -- the daily scan's own kernel
    # directories included -- stops resolving the moment the sandbox does.
    absolute = path.resolve()
    seeds = config.EXAM_SEEDS[:gate_seeds]
    origin = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="kernel-watch-gate-") as sandbox:
        os.chdir(sandbox)
        try:
            rates, _ = score_field(absolute, seeds, workers, list(roster.TRAINING))
        finally:
            os.chdir(origin)
    games = len(rates) * len(seeds) * 2
    equal = sum(rates.values()) / len(rates)
    low, high = wilson_interval(equal * games, games)
    return equal, low, high, games


if __name__ == "__main__":
    main()
