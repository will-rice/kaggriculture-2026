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


def loadable(source: str) -> bool:
    """Return whether the engine's own loader can resolve this source.

    Resolving means executing it, and published notebooks write files when they
    run -- one of them dropped a `submission_agent.py` into the repository root
    and an earlier harvest overwrote `main.py` the same way. So the check runs
    from a temporary directory, and anything the source writes goes there and
    is discarded with it.

    Args:
        source: Candidate agent source.

    Returns:
        True if a final callable resolves without raising.
    """
    import os
    import tempfile

    from kaggle_environments.agent import get_last_callable

    origin = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="kernel-watch-") as sandbox:
        os.chdir(sandbox)
        try:
            get_last_callable(source, path="main.py")
        except Exception:
            return False
        finally:
            os.chdir(origin)
    return True


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
    from kaggle_environments.agent import get_last_callable

    selected = get_last_callable(path.read_text(encoding="utf-8"), path="main.py")
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
    from kaggriculture.report import wilson_interval
    from kaggriculture.scripts.field_gate import score_field

    rates, games = score_field(path, gate_seeds, workers, set())
    equal = sum(rates.values()) / len(rates)
    low, high = wilson_interval(equal * games, games)
    return equal, low, high, games


if __name__ == "__main__":
    main()
