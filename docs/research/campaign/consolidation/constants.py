"""Which config constants are read, and by how many modules; and is roster.TRAINING still the pool?"""

import ast
import json
import re
from pathlib import Path

ROOT = Path(
    "/home/will/projects/kaggriculture-2026/.claude/worktrees/campaign/src/kaggriculture"
)


def main() -> None:
    """Count readers per constant, then compare the hand-kept roster to the live pool."""
    config = ROOT / "campaign" / "config.py"
    tree = ast.parse(config.read_text(encoding="utf-8"))
    names = sorted(
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id.isupper()
    )
    sources = {
        path: path.read_text(encoding="utf-8")
        for path in ROOT.rglob("*.py")
        if path != config and "served" not in path.parts
    }
    tests = {
        path: path.read_text(encoding="utf-8")
        for path in (ROOT.parent.parent / "tests").rglob("*.py")
    }
    unread = []
    once = []
    for name in names:
        pattern = re.compile(rf"\bconfig\.{name}\b")
        readers = [p.name for p, text in sources.items() if pattern.search(text)]
        tested = any(pattern.search(text) for text in tests.values())
        if not readers:
            unread.append((name, tested))
        elif len(readers) == 1:
            once.append((name, readers[0]))
    print(f"{len(names)} upper-case constants in config.py")
    print(f"{len(unread)} read by no module in the package:")
    for name, tested in unread:
        print(f"   {name}{'  (a test reads it)' if tested else ''}")
    print(f"{len(once)} read by exactly one module:")
    for name, reader in once:
        print(f"   {name:<32} {reader}")

    roster = ROOT / "campaign" / "roster.py"
    training = re.findall(
        r'^\s+"([^"]+)":\s*config\.', roster.read_text(encoding="utf-8"), re.M
    )
    pool = json.loads(
        Path("/data/kaggriculture/campaign/pool.json").read_text(encoding="utf-8")
    )
    held = set(pool.get("opponents", pool) if isinstance(pool, dict) else [])
    print(
        f"\nroster.TRAINING names {len(training)} opponents; the pool holds {len(held)}"
    )
    print(f"   TRAINING not in pool: {sorted(set(training) - held)}")


if __name__ == "__main__":
    main()
