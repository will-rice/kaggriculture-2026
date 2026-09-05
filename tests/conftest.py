"""Skip what needs the local data box when it is not there.

The opponent corpus, the episode archive and the Kaggle image live under
``/data/kaggriculture`` and on the local docker daemon; they are vendored
third-party material and multi-gigabyte images that never enter the
repository. Tests that need them carry ``@pytest.mark.local_data`` and skip
on a runner without them, so CI proves everything else.
"""

import pytest

from kaggriculture.campaign import config

LOCAL_DATA_SKIP = pytest.mark.skip(
    reason=f"needs the local data box ({config.OPPONENTS.parent} is absent)"
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip ``local_data`` tests when the data directory is missing."""
    if config.OPPONENTS.parent.exists():
        return
    for item in items:
        if "local_data" in item.keywords:
            item.add_marker(LOCAL_DATA_SKIP)
