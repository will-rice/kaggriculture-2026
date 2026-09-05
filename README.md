# kaggriculture-2026

Agent for the [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture)
simulation competition: two players farm for 30 in-game days and the one with
the most coins banked wins.

See [docs/competition.md](docs/competition.md) for the format, the rules the code
relies on, and the places where the engine disagrees with the documentation.

## Features

- **Turn Policy**: Job-list planner that gives each farmer and hired hand the nearest useful action every turn
- **Pydantic Configuration**: Type-safe configuration management
- **Protocol-based Design**: Clean `Agent` and `Task` protocols for easy extensibility
- **Result Tracking**: Structured result collection with `pydantic` models
- **Parallel Evaluation**: Seeded head-to-head matches against the built-in agents, fanned out across processes
- **Submission Tooling**: One command to build the archive, one to upload it
- **Modern Tooling**: Built with `uv` for fast dependency management
- **Code Quality**: Pre-configured with `ruff`, `ty`, `pytest`, and `pre-commit` hooks

## Project Structure

```
.
├── main.py                    # Competition entrypoint; Kaggle imports `agent` from here
├── src/kaggriculture/
│   ├── policy.py              # The agent: one turn of farm and market decisions
│   ├── observation.py         # Typed view over the raw observation dict
│   ├── actions.py             # Turn/action construction and movement
│   ├── constants.py           # Rules tables, imported from kaggle-environments
│   ├── agent.py               # Agent protocol and the episode-playing agent
│   ├── config.py              # Pydantic configuration
│   ├── harness.py             # Main Harness class
│   ├── result.py              # Result data models
│   ├── task.py                # Task protocol and seeded match task
│   └── scripts/
│       ├── run.py             # Evaluate against the built-in agents
│       ├── package.py         # Build submission.tar.gz
│       ├── submit.py          # Package and upload to Kaggle
│       └── replay_corpus.py   # Replay Kaggle episode archives through the Rust engine
├── rust/                      # Rust port of the reference engine + Python bindings (see rust/README.md)
├── docs/competition.md        # Competition notes
├── tests/                     # Test files
│   └── rust/                  # Drives the Rust engine and the reference through one tape
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

The project pins Python 3.11 because agents run on Kaggle's image; developing
below the runner's version keeps anything that works locally working there too.

### 3. Set up environment variables

Copy the example environment file and add your Kaggle API token:

```bash
cp .env.example .env
# Edit .env, or save the token to ~/.kaggle/access_token
```

### 4. Install pre-commit hooks

```bash
uv run pre-commit install
```

## Usage

### Evaluating the Agent

Play the submission entrypoint against the environment's built-in agents:

```bash
uv run run --games 8
```

```
vs starter    win_rate 1.00  (8W 0L 0T)  bank    48221 vs     3506
vs random     win_rate 1.00  (8W 0L 0T)  bank    48221 vs       27
vs pass       win_rate 1.00  (8W 0L 0T)  bank    48221 vs     3000
```

Play a specific matchup, or save replays for the visualizer:

```bash
uv run run --agent main.py --opponents starter --games 4 --replays replays/
```

### Submitting

Joining the competition on the website is required before the first submission.

```bash
uv run package                       # writes submission.tar.gz
uv run submit "melon loop v1"        # packages, confirms, then uploads
kaggle competitions submissions kaggriculture
```

Submitting spends one of the day's five slots, so `submit` asks before
uploading; pass `--yes` to skip the prompt.

### Changing the Strategy

`Strategy` in `src/kaggriculture/policy.py` holds the tunable knobs — crop
choice, crew size, how much land one unit can service, sell throttling. The
defaults come from sweeping each knob over seeded matches against `starter`.

```python
from kaggriculture import policy
from kaggriculture.policy import Strategy

policy.STRATEGY = Strategy(crop="WHEAT", max_hands=10)
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
        return [0.0, 0.0]
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
        return 1.0 if output[0] > output[1] else 0.0
```

### Running an Evaluation

```python
from kaggriculture.agent import EpisodeAgent
from kaggriculture.config import HarnessConfig
from kaggriculture.harness import Harness

config = HarnessConfig(games=4, opponents=("starter",))
harness = Harness(config=config)

results = harness.run(EpisodeAgent(spec="main.py"), harness.matches())
for result in results:
    print(f"Task {result.task_id}: score={result.score} banks={result.scores}")
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
    max_workers: int | None = None
    seed: int = 42
    games: int = 8
    opponents: tuple[str, ...] = ("starter", "random", "pass")
    episode_steps: int = 720
    debug: bool = False
```

## Dependencies

Core dependencies:

- **kaggle-environments**: The Kaggriculture simulator and its rules tables
- **kaggle**: Competition CLI used for submission
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
