"""Read a newly published kernel's agent out of its notebook.

Three ladder upgrades in a week came from one competitor publishing v54, v56
and v58, and each was worth more than any learning we attempted. `harvest`
decides which published kernels join the pool and when; this is how one is
turned into a file the harness can play.

What it does:

1. Lists public kernels for the competition and reports any whose ref is not
   already in ``run/kernel-watch/seen.json``.
2. Extracts each statically -- ``ast.literal_eval`` of the payload assignment,
   then base85 and zlib -- and verifies the digest the notebook itself asserts.
   It never executes a notebook cell.
3. Resolves the entrypoint the way the engine's loader does, in a sandbox.
   These artifacts shadow ``agent``, sometimes twice, and the surviving
   definition is not the one that plays; getting this wrong measures a
   different agent than the one a submission would run.
4. Builds a compiled kernel from the compiler line its own notebook runs, when
   that line is a literal.

It used to tune and gate as well -- a second gate beside `gate.py`, scoring
against a hand-kept roster through its own field function -- and stop short of
submitting. The pool is `harvest`'s to decide and the gate is `gate.py`'s.
"""

import ast
import base64
import hashlib
import json
import logging
import re
import zlib
from pathlib import Path
from typing import Any

from kaggriculture.campaign import config, pools

LOGGER = logging.getLogger(__name__)

COMPETITION = "kaggriculture"
# Absolute, anchored to the repository rather than to wherever the process
# happens to stand. These were `Path("run/kernel-watch/...")`, which is right
# for as long as nothing ever moves the working directory -- and things do:
# every worker that runs a program is given a directory of its own, and the
# harvest runs in the same process as the loop that starts them. A relative
# constant then resolves somewhere else, silently, and the harvest writes its
# memory of which kernels it has seen into a scratch tree about to be deleted.
# Nothing raises; it simply forgets, and re-checks every kernel forever.
SEEN = config.ROOT / "run" / "kernel-watch" / "seen.json"
WORK = config.ROOT / "run" / "kernel-watch" / "kernels"

# What marks a cell as holding the agent: a def the engine can call, or a
# module-level binding of one, which is how a factory-built policy is
# published.
ENTRYPOINT = r"^(?:def (?:agent|kaggle_agent)\(|(?:agent|main|policy)\s*=)"
# Every digest a cell states, whether written into the comparison or held in a
# constant beside it. The check used to be `digest == '<hex>'`, which is one
# author's house style: the notebooks that ship the field's strongest agents
# write `assert hashlib.sha256(SOURCE_BYTES).hexdigest() == EXPECTED_MAIN_SHA256`
# and that regex sees nothing to check.
DIGEST = re.compile(r"\b[0-9a-f]{64}\b")


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
    """Return one kernel's agent source, without executing the notebook.

    Args:
        ref: The kernel ref to pull.

    Returns:
        The decoded agent source, or None if the notebook ships no agent this
        can read.
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
        source = payload_source(text)
        if source is None:
            continue
        # Verified when the cell states this payload's digest, and it does not
        # always: of the thirty-six payloads on disk on 2026-09-27, twenty-seven
        # stated a digest and every one of those matched, while nine stated
        # none. An unstated digest is not a reason to refuse an opponent -- a
        # source-published agent carries no digest either and the pool is full
        # of them -- and a mismatch is no longer raised, because a raise here
        # is a stranger's notebook taking down the whole hourly harvest.
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if digest not in DIGEST.findall(text):
            LOGGER.info("%s: payload digest unstated, UNVERIFIED -- opponent only", ref)
        return source
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


def resolve_in_sandbox(source: str) -> str:
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
        The name of the callable the loader selects.
    """
    return pools.isolated(entrypoint_name, source)


def entrypoint_name(source: str) -> str:
    """The name of the callable the loader selects. Runs in `isolated`'s child.

    A name rather than the callable, because this crosses a process boundary
    and a freshly-imported function does not pickle. Both callers only ever
    wanted the name or the fact that one resolved at all.

    Args:
        source: Candidate agent source.

    Returns:
        The selected callable's name.
    """
    from kaggle_environments.agent import get_last_callable

    try:
        selected = get_last_callable(source, path="main.py")
    except SystemExit as exit_call:
        raise RuntimeError(f"source called sys.exit({exit_call.code})") from None
    return getattr(selected, "__name__", repr(selected))


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


def concatenated(node: ast.expr) -> str | None:
    """Return a string a cell builds out of constants, without running it.

    `ast.literal_eval` reads a lone constant and implicit concatenation and
    stops there. A megabyte of base85 is not published on one line: the
    notebooks shipping the field's strongest agents write
    ``''.join(('...', '...', ...))``, which `literal_eval` sees as a call and
    refuses -- so the payload was reported as computed and the kernel as
    having no agent.

    A join of constants is decided by constants. Reading it executes nothing,
    which is the property that matters: the separator and every element are
    literals and the result follows from them.

    Args:
        node: The expression a cell assigns.

    Returns:
        The string, or None when constants do not decide the expression.
    """
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = concatenated(node.left), concatenated(node.right)
        return None if left is None or right is None else left + right
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and len(node.args) == 1
        and isinstance(node.args[0], (ast.Tuple, ast.List))
    ):
        separator = concatenated(node.func.value)
        if separator is None:
            return None
        parts = []
        for element in node.args[0].elts:
            part = concatenated(element)
            if part is None:
                return None
            parts.append(part)
        return separator.join(parts)
    return None


def payload_source(text: str) -> str | None:
    """Return the agent source a code cell carries as an encoded blob.

    A payload is identified by what it decodes to, not by what it is called.
    It used to be identified by name -- ``payload``, ``PAYLOAD``, ``blob`` --
    and by being a single literal, which is what the competition published in
    early September and has not published since.

    Measured 2026-09-27 over the 327 kernels the scan had pulled: twelve
    matched that shape, and another twenty-four carried a payload under nine
    other names -- ``SOURCE_BLOB``, ``_BLOB``, ``_PAYLOAD``, ``SOURCE_B85``,
    ``PACKED_AGENT``, ``agent_b85``, ``AGENT_B85``, ``SUBMISSION_B85``,
    ``_PACKED_SOURCE`` -- nearly all of them joined from pieces, one in base64
    rather than base85. Those twenty-four are where the field is: v52 through
    v57 of an author publishing a version every day, the top public agent's
    submission v13, three of another author's stack. Every one of them was
    logged "no agent this can build or extract" and skipped, and the pool held
    six of the fifty most recently published kernels.

    So the test is the plaintext: a string built out of constants, base85 or
    base64, zlib-compressed, that parses as Python and leaves an entrypoint
    behind. That is also what tells the agent from the other blobs a cell
    carries -- a licence, a table of routes -- which decode perfectly well and
    are not agents.

    Nothing is executed. The blob is read as a literal and decoded.

    Args:
        text: One notebook code cell's source.

    Returns:
        The agent source, or None when the cell carries no payload.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        blob = concatenated(node.value)
        if blob is None:
            continue
        # base85 first: `b64decode` ignores characters outside its alphabet, so
        # handed base85 it returns bytes rather than refusing, and those bytes
        # are not what the notebook published.
        for decode in (base64.b85decode, base64.b64decode):
            try:
                plain = zlib.decompress(decode(blob)).decode("utf-8")
            except (ValueError, zlib.error, UnicodeDecodeError):
                continue
            if not re.search(ENTRYPOINT, plain, re.M):
                continue
            try:
                ast.parse(plain)
            except SyntaxError:
                continue
            return plain
    return None
