"""Token-shingle similarity between a candidate and every opponent source.

Independent programs that solve the same game share vocabulary (the op
names, the item names) but not eight consecutive tokens; a lifted block
does. Measured on the roster corpus (39 files, ~42k shingles): a realistic
~60-line independent agent, written fresh against the shared op/item
vocabulary, scores 0.0 against every corpus file — no 8-token window it
produces happens to recur anywhere. A 200-line block lifted verbatim from
the middle of `shopforge`'s 3,778 lines and pasted into the PASS skeleton
scores 0.048 against `shopforge:main.py` (the Jaccard denominator carries
the other ~3,700 unshared lines of that file, so even an exact lift reads
well under 1.0). THRESHOLD is set at 0.02: below the measured lift with
room to spare, above the independent agent's exact-zero floor. A verbatim
opponent file scores 1.0, as expected.
"""

import functools
import logging
from pathlib import Path

from kaggriculture.campaign import roster

logger = logging.getLogger(__name__)

K = 8
THRESHOLD = 0.02
CORPUS_SUFFIXES = (".py", ".cpp", ".inc", ".hpp")
ENGINE_DIR = Path(__file__).parent / "engine"


def tokens(source: str) -> list[str]:
    """Whitespace-delimited tokens.

    Python's `tokenize` was tried first, but it drops comments and string
    literals as a single opaque unit each — and several opponents store
    their strategy as a literal data table (a big string constant). A
    fragment lifted from the middle of such a table tokenizes as ordinary
    code once it loses its enclosing quotes, while the source file that
    still has the quotes tokenizes it as one dropped STRING token: the two
    sides never share a single token, let alone a shingle. Splitting on
    whitespace has no such asymmetry — it is the same one scheme for every
    file, Python or C++, syntactically valid or not — so a lifted block
    always shingle-matches the file it was lifted from.
    """
    return source.split()


def shingles(source: str, k: int = K) -> set[tuple[str, ...]]:
    """The set of contiguous k-token windows in `source`."""
    stream = tokens(source)
    return {tuple(stream[i : i + k]) for i in range(max(0, len(stream) - k + 1))}


def _jaccard(a: frozenset[tuple[str, ...]], b: frozenset[tuple[str, ...]]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def similarity(a: str, b: str, k: int = K) -> float:
    """Jaccard similarity of the two sources' k-shingle sets."""
    return _jaccard(frozenset(shingles(a, k)), frozenset(shingles(b, k)))


def _engine_header_bytes() -> set[bytes]:
    """Bytes of every header we adopted into our own engine port.

    A candidate that includes engine constants (our own `sim.hpp`,
    `pyrandom.hpp`) must not trip the gate just because an opponent's build
    also vendors a byte-identical copy of the same header.
    """
    return {path.read_bytes() for path in ENGINE_DIR.glob("*.hpp")}


@functools.lru_cache(maxsize=1)
def _corpus() -> dict[str, frozenset[tuple[str, ...]]]:
    """Every opponent's shingle sets, keyed `"<name>:<relative path>"`.

    Cached for the process: `against_opponents` runs hundreds of times a
    day and the corpus does not change underneath it.
    """
    engine_bytes = _engine_header_bytes()
    corpus: dict[str, frozenset[tuple[str, ...]]] = {}
    for name in roster.names():
        root = roster.path(name).parent
        for file in sorted(root.rglob("*")):
            if not file.is_file() or file.suffix not in CORPUS_SUFFIXES:
                continue
            raw = file.read_bytes()
            if raw in engine_bytes:
                continue
            text = raw.decode("utf-8", errors="replace")
            key = f"{name}:{file.relative_to(root)}"
            corpus[key] = frozenset(shingles(text))
    total = sum(len(shingle_set) for shingle_set in corpus.values())
    logger.info("copycheck corpus: %d files, %d shingles", len(corpus), total)
    return corpus


def against_opponents(source: str) -> tuple[str, float]:
    """The most similar opponent file and its score."""
    candidate = frozenset(shingles(source))
    worst_name, worst = "", 0.0
    for name, shingle_set in _corpus().items():
        score = _jaccard(candidate, shingle_set)
        if score > worst:
            worst_name, worst = name, score
    return worst_name, worst
