"""Dependency-free fixed schemas shared by hybrid authoring and search."""

PHASE_START_DAYS: tuple[tuple[int, ...], ...] = (
    tuple(range(0, 10)),
    tuple(range(10, 20)),
    tuple(range(20, 30)),
)
