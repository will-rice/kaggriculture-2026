"""Token-shingle similarity between a candidate and every opponent source.

Independent programs that solve the same game share vocabulary (the op
names, the item names) but not eight consecutive tokens; a lifted block
does — and stays sharing them under non-semantic reformatting, because the
tokenizer sees words, numbers, and lone punctuation characters, never
whitespace or quote style. Measured on the roster corpus (39 files, ~42k
shingles):

- a verbatim opponent file (`v54:main.py` fed back in) scores 1.0.
- the PASS skeleton (~10 lines) scores 0.0062 against its best match.
- a realistic 70-line independent agent (53 non-blank lines: watering,
  harvesting, planting, hiring, and selling branches — the size an evolved
  candidate will actually be) scores 0.0066 against its best match.
- a 200-line block lifted verbatim from the middle of `shopforge`'s
  3,778 lines, pasted into the skeleton, scores 0.152 against
  `shopforge:main.py`.
- a 110-line block lifted verbatim from `indarkarhana`'s
  `e749a_niklita_consensus_network.py` (lines 91-200) scores 0.231.
- that same 110-line block after collapsing whitespace around `:`, `,`,
  `=`, `->` and swapping `"` for `'` — what an LLM asked to "rewrite this"
  would produce — still scores 0.231: byte-identical tokens either way.

THRESHOLD is the log-midpoint between the independent score (0.0066) and
the smaller of the two lifted-block scores (0.152), rounded down for
margin: sqrt(0.0066 * 0.152) ~= 0.032, so THRESHOLD = 0.03 sits about
4.5x above the independent floor and 5x below the weakest lift.

One opponent is exempt: the published agent `config.SEED` starts the
campaign from, which every program then descends from. See `_corpus`.
"""

import functools
import logging
import re
from pathlib import Path

from kaggriculture.campaign import config, roster

logger = logging.getLogger(__name__)

K = 8
THRESHOLD = 0.03
CORPUS_SUFFIXES = (".py", ".cpp", ".inc", ".hpp")
ENGINE_DIR = Path(__file__).parent / "engine"

# Identifiers/keywords, numbers, or a single punctuation character. Quote
# characters are excluded from the punctuation class so `"x"` and `'x'`
# tokenize identically; whitespace never matches, so it never produces a
# token and reformatting (spacing, line breaks) cannot change the stream.
_TOKEN_RE = re.compile(r"[A-Za-z_]\w*|\d+(?:\.\d+)?|[^\w\s'\"`]")


def tokens(source: str) -> list[str]:
    """Formatting-invariant tokens: words, numbers, and lone punctuation.

    Whitespace-splitting (tried first) is defeated by non-semantic
    reformatting: collapsing the spaces around `:`, `,`, `=`, `->` glues
    tokens together, and swapping `"` for `'` changes every string token —
    exactly what an LLM asked to "rewrite this" will do to a lifted block.
    A regex tokenizer that treats punctuation characters individually and
    drops quote marks is invariant to both: `x: int = 1` and `x:int=1`
    yield the same token stream, and `"WATER"` and `'WATER'` yield the
    same token. It also treats every file — Python, C++, and any embedded
    data table — the same way, since it never depends on the text being
    syntactically valid in any particular language.
    """
    return _TOKEN_RE.findall(source)


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
def _lineage() -> frozenset[tuple[str, ...]]:
    """Shingles of the published agent a cold start seeds from.

    The gate asks whether a candidate lifted from an opponent it was never
    given, and the seed is the one opponent it *was* given: `config.SEED` is
    what a cold start copies into the database, and every program in the
    campaign descends from it. A child that still resembles its own ancestor
    has copied nothing.

    Empty where the opponents are not installed, because `_corpus` is empty
    there too -- `rglob` over a directory that is not on this machine yields
    nothing, so there is nothing for a lineage to be exempt from. The two
    have to agree: CI carries no `/data`, and a lineage that raised where the
    corpus quietly returns nothing would take the suite down on every machine
    that is not the box.
    """
    if not config.SEED.exists():
        return frozenset()
    return frozenset(shingles(config.SEED.read_text(encoding="utf-8")))


@functools.lru_cache(maxsize=1)
def _corpus() -> dict[str, frozenset[tuple[str, ...]]]:
    """Every opponent's shingle sets, keyed `"<name>:<relative path>"`.

    The seed's own lineage is left out, by the same threshold the gate judges
    on. A corpus file that close to `config.SEED` cannot tell a lifted block
    from an inherited one, because every candidate inherits that block
    legitimately -- keeping it would reject the whole campaign rather than a
    copy. Measured on 2026-09-07 that excludes exactly one file of 53, the
    seed's own roster copy at 1.000; the next highest scores 0.003, so the
    other eleven opponents are gated exactly as they were.

    The exemption is by similarity rather than by name because this field
    republishes itself: `pilkwang_economic` and `v58` are 0.995 apart under
    two different authors, and a seed with a twin like that must exempt the
    twin or reject every child it will ever have.

    Cached for the process: `against_opponents` runs hundreds of times a
    day and the corpus does not change underneath it.
    """
    engine_bytes = _engine_header_bytes()
    lineage = _lineage()
    corpus: dict[str, frozenset[tuple[str, ...]]] = {}
    exempt: list[str] = []
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
            shingle_set = frozenset(shingles(text))
            if _jaccard(shingle_set, lineage) >= THRESHOLD:
                exempt.append(key)
                continue
            corpus[key] = shingle_set
    total = sum(len(shingle_set) for shingle_set in corpus.values())
    logger.info("copycheck corpus: %d files, %d shingles", len(corpus), total)
    # Logged every time, because this line is the gate standing down. An
    # exemption that quietly grew a second entry is how a real copy gets in.
    logger.info("copycheck exempts the seed lineage: %s", ", ".join(exempt) or "none")
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
