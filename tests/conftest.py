"""What needs the local data box is slow, and skipped where the box is not there.

The opponent corpus, the episode archive and the Kaggle image live under
``/data/kaggriculture`` and on the local docker daemon; they are vendored
third-party material and multi-gigabyte images that never enter the
repository. Tests that need them carry ``@pytest.mark.local_data`` and skip
on a runner without them, so CI proves everything else.

They are also slow -- real games on the engine, real images -- and the
commit hook runs the suite on every commit. So ``local_data`` implies
``slow`` and is added here rather than on every test: the default run
(``-m 'not slow'``) leaves them out, and ``uv run pytest -m slow`` is what
proves them before a push.
"""

import pytest

from kaggriculture.campaign import config

LOCAL_DATA_SKIP = pytest.mark.skip(
    reason=f"needs the local data box ({config.OPPONENTS.parent} is absent)"
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark every ``local_data`` test slow, and skip them without the box.

    Args:
        items: The collected tests, marked in place.
    """
    local = [item for item in items if "local_data" in item.keywords]
    for item in local:
        item.add_marker(pytest.mark.slow)
    if config.OPPONENTS.parent.exists():
        return
    for item in local:
        item.add_marker(LOCAL_DATA_SKIP)
