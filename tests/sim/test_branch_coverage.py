"""Executable branch-counter tests for the reference campaign."""
# ruff: noqa: D103

from kaggle_environments import make

from kaggriculture.sim.fidelity import reference_branches, track_reference_branches


def test_reference_branch_counter_manifest_and_smoke_hits() -> None:
    with track_reference_branches() as counts:
        environment = make("kaggriculture", configuration={"seed": 193}, debug=True)
        environment.reset(2)
        passes = [{"farmer": ["PASS"], "hands": [], "market": []}] * 2
        environment.step(passes)

    assert tuple(sorted(counts)) == reference_branches()
    assert sum(value["true"] + value["false"] for value in counts.values()) > 0
