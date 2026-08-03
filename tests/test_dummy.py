"""Smoke tests for package metadata."""


from kaggriculture.config import HarnessConfig


def test_package_imports() -> None:
    """The renamed package remains importable."""
    config = HarnessConfig()

    assert config.seed == 42
