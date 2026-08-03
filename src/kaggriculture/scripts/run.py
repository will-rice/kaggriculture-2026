"""Main run script for the agent harness."""

from kaggriculture.config import HarnessConfig
from kaggriculture.harness import Harness


def main() -> None:
    """Main entry point for the agent harness."""
    config = HarnessConfig()
    harness = Harness(config=config)
    print(f"Agent harness initialized: {harness.config}")


if __name__ == "__main__":
    main()
