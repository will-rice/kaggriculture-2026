"""Dependency-free fixed schemas shared by hybrid authoring and search."""

PHASE_START_DAYS: tuple[tuple[int, ...], ...] = (
    (0,),
    tuple(range(10, 20)),
    tuple(range(20, 30)),
)
