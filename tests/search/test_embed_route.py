"""The generated route policy must play exactly like the route it embeds."""

from pathlib import Path

import pytest
from kaggle_environments import make

import kaggriculture.searched_route_policy as generated
from kaggriculture.constants import ENVIRONMENT, EPISODE_STEPS
from kaggriculture.search import arena
from kaggriculture.search.route import load

GENERATED = Path("src/kaggriculture/searched_route_policy.py")
SERVED = "src/kaggriculture/boatlee_v14_policy.py"
SOURCE_ROUTE = Path("/data/kaggriculture/search/accepted-20260819/accepted-0149.json")
GATE_SEED = 700_000

_needs_source_route = pytest.mark.skipif(
    not SOURCE_ROUTE.exists(), reason="searched route not on this machine"
)


@_needs_source_route
def test_the_generated_module_banks_identically_to_the_route_it_embeds() -> None:
    """The one assertion that proves the shipped artifact is the gated route.

    ``searched_route_policy.py`` carries its own copy of ``arena._replay``'s
    ``+1`` indexing, decoded from a self-contained blob rather than imported
    from ``arena``. Nothing keeps that copy honest except playing it out: this
    runs the generated module as a real file agent through the Kaggle loader,
    and separately plays the route it was generated from through
    ``arena.play``, on the same held-out gate seed against the same opponent,
    and requires the final banks to match exactly. A mismatch here would mean
    the file about to ship is not the route that passed the gate -- whether
    from a broken offset, a lossy encode, or a stale regeneration.
    """
    route = load(SOURCE_ROUTE)

    environment = make(
        ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS, "seed": GATE_SEED}
    )
    environment.run([str(GENERATED), SERVED])
    final = environment.steps[-1]
    generated_banks = (int(final[0].reward), int(final[1].reward))

    route_banks = arena.play(route, SERVED, [GATE_SEED])[0]

    assert generated_banks == route_banks


def test_the_generated_module_imports_nothing_but_the_standard_library() -> None:
    """Kaggle execs this file standalone; a non-stdlib import cannot survive it.

    A ``torch`` import would blow the sandbox's overage budget the way
    ``main.py``'s own docstring warns against; a ``kaggriculture`` import
    would fail outright, since the archive that ships this file excludes the
    package's ``search`` tree entirely -- see ``EXCLUDED`` in
    ``scripts/package.py``.
    """
    source = GENERATED.read_text()

    assert "import torch" not in source
    assert "import kaggriculture" not in source
    assert "from kaggriculture" not in source


@_needs_source_route
def test_the_embedded_route_round_trips_to_the_source_route_exactly() -> None:
    """The decoded blob must equal ``route.load`` of its source route, exactly.

    This is the structural half of the fidelity check above: it pins the
    encode/decode round trip on its own, with no engine and no opponent
    involved, so a compression or serialisation bug is caught here rather than
    surfacing only as a bank mismatch.
    """
    assert generated._ROUTE == load(SOURCE_ROUTE)
