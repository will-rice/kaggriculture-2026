"""Fixtures pytest hands to any test in this directory that names one."""

import pytest

from kaggriculture.campaign import games


@pytest.fixture
def scratch(request: pytest.FixtureRequest) -> str:
    """A database of this test's own, with the real schema, dropped after.

    Every live test takes this rather than writing into the campaign's own.
    They used not to, and tidied up afterwards instead -- which means tidying
    the tables somebody remembered: a write goes to five of them, the teardown
    cleared one, and ten probe episodes were sitting in the real store before
    anything noticed.

    In a `conftest` rather than in the module that first wanted it, now that a
    second module does. Imported instead, it would be a name the linter sees
    shadowed by every test that asks for it.
    """
    name = f"test_{abs(hash(request.node.name)):x}"[:40]
    games.query(f"DROP DATABASE IF EXISTS {name}")
    request.addfinalizer(lambda: games.query(f"DROP DATABASE IF EXISTS {name}"))
    games.create(name)
    return name
