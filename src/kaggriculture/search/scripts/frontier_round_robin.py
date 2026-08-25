"""Measure a verified public-agent frontier at low CPU priority."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from kaggriculture.search.frontier import rank_frontier, verify_frontier

FRONTIER_SEEDS = tuple(range(810_000, 810_064))
DEFAULT_MANIFEST = Path("src/kaggriculture/search/frontier_manifest.json")
DEFAULT_ARTIFACT_ROOT = Path("/data/kaggriculture/search/public-frontier")


def parser() -> argparse.ArgumentParser:
    """Build the command-line interface without beginning any games."""
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    arguments.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    arguments.add_argument("--workers", type=int, default=16)
    arguments.add_argument("--output", type=Path, required=True)
    return arguments


def main() -> None:
    """Verify byte identity, play the fixed field, and serialize every matchup."""
    args = parser().parse_args()
    if not 1 <= args.workers <= 16:
        raise SystemExit("--workers must be between 1 and 16 while Toad is running")
    verified = verify_frontier(args.manifest, args.artifact_root)
    report = rank_frontier(verified, FRONTIER_SEEDS, args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
