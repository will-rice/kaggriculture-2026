# kaggriculture-2026

The kaggriculture project package and supporting tooling.

## Features

- **Pydantic Configuration**: Type-safe configuration management
- **Protocol-based Design**: Clean `Agent` and `Task` protocols for easy extensibility
- **Result Tracking**: Structured result collection with `pydantic` models
- **Modern Tooling**: Built with `uv` for fast dependency management
- **Code Quality**: Pre-configured with `ruff`, `ty`, `pytest`, and `pre-commit` hooks

## Project Structure

```
.
├── src/kaggriculture/
│   ├── agent.py               # Agent protocol
│   ├── config.py              # Pydantic configuration
│   ├── harness.py             # Main Harness class
│   ├── result.py              # Result data models
│   ├── task.py                # Task protocol
│   └── scripts/
│       └── run.py             # Entry point script
├── tests/                     # Test files
├── pyproject.toml             # Project metadata and dependencies
├── .pre-commit-config.yaml    # Pre-commit hooks configuration
└── .env.example               # Example environment variables
```

## Quick Start

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Set up environment variables

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 4. Install pre-commit hooks

```bash
uv run pre-commit install
```

## Usage

### Running the Harness

Run the entry point script:

```bash
uv run run
```

### Implementing Your Agent

Create a class that conforms to the `Agent` protocol:

```python
from kaggriculture.agent import Agent


class MyAgent:
    """A custom agent implementation."""

    def run(self, task):
        """Run the agent on a task and return the output."""
        # Your agent logic here
        return "agent output"
```

### Implementing Your Task

Create a class that conforms to the `Task` protocol:

```python
from kaggriculture.task import Task


class MyTask:
    """A custom task implementation."""

    @property
    def id(self) -> str:
        """Return the task identifier."""
        return "my_task_001"

    def evaluate(self, output) -> float:
        """Evaluate agent output and return a score between 0 and 1."""
        return 1.0 if output == "expected output" else 0.0
```

### Running an Evaluation

```python
from kaggriculture.config import HarnessConfig
from kaggriculture.harness import Harness

config = HarnessConfig(max_workers=4, timeout=30.0)
harness = Harness(config=config)

agent = MyAgent()
tasks = [MyTask()]

results = harness.run(agent, tasks)
for result in results:
    print(f"Task {result.task_id}: score={result.score}")
```

## Development

### Running Tests

```bash
uv run pytest
```

### Type Checking

```bash
uv run ty check src/
```

### Linting and Formatting

```bash
uv run ruff check src/
uv run ruff format src/
```

### Pre-commit Hooks

Pre-commit hooks will automatically run on every commit to ensure code quality. To run manually:

```bash
uv run pre-commit run --all-files
```

## Configuration

Edit `src/kaggriculture/config.py` to customize harness settings:

```python
from pydantic import BaseModel


class HarnessConfig(BaseModel):
    max_workers: int = 1
    timeout: float = 60.0
    seed: int = 42
    debug: bool = False
```

## Dependencies

Core dependencies:

- **Pydantic**: Data validation and configuration
- **python-dotenv**: Environment variable management

Development tools:

- **ruff**: Fast Python linter and formatter
- **ty**: Static type checker
- **pytest**: Testing framework
- **pre-commit**: Git hooks for code quality

## Build System

This project uses `uv_build` as the build backend. To build the project:

```bash
uv build
```

## License

See [LICENSE](LICENSE) file for details.

## Contributing

1. Create a new branch for your feature
2. Make your changes
3. Ensure all tests pass and pre-commit hooks succeed
4. Submit a pull request
