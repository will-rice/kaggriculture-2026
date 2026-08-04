# Evaluation and League Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace "mean bank over 16 seeds against one opponent" with win rate and a confidence interval against a frozen, diverse league, so that every later change can be accepted or rejected on the metric the ladder actually scores.

**Architecture:** Three additions to the existing harness. A `report` module turns `Result` lists into per-opponent win rates with Wilson score intervals. A `baselines/` directory holds frozen named opponents, each a thin module that runs the existing policy under a pinned `Strategy` — so a baseline is a configuration, not a copy of the code, and cannot rot. A `replay` module turns the ad-hoc replay inspection used during development into a real API, because every later phase reads it.

**Tech Stack:** Python 3.11, `pydantic` models, `pytest` functional style, `kaggle-environments` for episodes, `ProcessPoolExecutor` for fan-out, `wandb` for the run record.

## Global Constraints

- Run everything with `uv run`. Add dependencies with `uv add`. Never invoke `pip` or a bare `python`.
- `uv run pre-commit run -a` must pass before every commit. Fix type errors; never add ignore comments.
- Google-style docstrings on every public function. Absolute imports. `main()` at the top of scripts, then functions in call order.
- No `from __future__ import annotations`.
- Use `logging.info`, never `print()`. No global loggers beyond the module-level `LOGGER = logging.getLogger(__name__)` the codebase already uses.
- Required argparse arguments take no `--` prefix.
- Tests are pytest functional style and exercise the real environment. Minimise mocks.
- Never name personal hardware in commits or docs; say "local workstation".
- Do not mention Claude in commit messages.

## File Structure

| File                                    | Responsibility                                                                                                                                        |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/kaggriculture/report.py`           | **Create.** Wilson interval, per-opponent aggregation, one formatted line per opponent. Pure functions over `Result` lists — no I/O, no environment.  |
| `src/kaggriculture/replay.py`           | **Create.** Load an episode JSON and answer questions about it: crop mix per day, bank trajectory, animals lost, realised prices, end-of-season shed. |
| `baselines/heuristic_v1.py`             | **Create.** The melon monoculture that first went on the ladder, pinned as a `Strategy`.                                                              |
| `baselines/meta_build.py`               | **Create.** The reconstructed meta build — a _reactive_ agent playing what 75% of the ladder plays, unlike the open-loop tape.                        |
| `src/kaggriculture/config.py`           | **Modify.** Add `LEAGUE` naming the frozen opponents, and default `opponents` to it.                                                                  |
| `src/kaggriculture/scripts/tracking.py` | **Create.** One wandb run per evaluation. Under `scripts/` so it can never reach the submission archive.                                              |
| `src/kaggriculture/scripts/run.py`      | **Modify.** Report win rate with CI instead of mean bank; add `--track`.                                                                              |
| `tests/test_report.py`                  | **Create.** Interval maths and aggregation.                                                                                                           |
| `tests/test_replay.py`                  | **Create.** Replay analysis against a real short episode.                                                                                             |
| `tests/test_tracking.py`                | **Create.** The shape of the wandb payload, without touching the network.                                                                             |
| `tests/test_baselines.py`               | **Create.** Every league member plays a legal season.                                                                                                 |

---

### Task 1: Wilson score interval and per-opponent aggregation

The current runner reports mean bank, which is not what the ladder scores. It also cannot express uncertainty, so a 16-seed result and a 200-seed result look identical. Wilson is used rather than the normal approximation because we are frequently at 0 wins, where the normal approximation collapses to a zero-width interval at zero — precisely the case we most need honest error bars for.

**Files:**

- Create: `src/kaggriculture/report.py`
- Test: `tests/test_report.py`

**Interfaces:**

- Consumes: `kaggriculture.result.Result` (fields `task_id`, `score`, `error`, `scores`; property `opponent`).
- Produces:
  - `wilson_interval(wins: float, games: int, z: float = 1.96) -> tuple[float, float]`
  - `class Standing(BaseModel)` with fields `opponent: str`, `games: int`, `wins: int`, `losses: int`, `ties: int`, `errors: int`, `win_rate: float`, `low: float`, `high: float`, `bank: float`, `opponent_bank: float`, and property `half_width: float`
  - `standings(results: list[Result]) -> list[Standing]`
  - `format_standing(standing: Standing) -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/test_report.py`:

```python
"""Tests for win-rate reporting and its confidence intervals."""

from kaggriculture.report import Standing, standings, wilson_interval
from kaggriculture.result import Result


def test_wilson_interval_is_not_degenerate_at_zero_wins() -> None:
    """Losing every game is the case we most need honest error bars for."""
    low, high = wilson_interval(wins=0, games=16)

    assert low == 0.0
    assert 0.0 < high < 0.3


def test_wilson_interval_narrows_as_games_accumulate() -> None:
    """More seeded games must buy a tighter interval, or the gate is meaningless."""
    _, few = wilson_interval(wins=5, games=10)
    _, many = wilson_interval(wins=50, games=100)

    assert many < few


def test_a_hundred_even_games_meet_the_phase_one_gate() -> None:
    """The gate asks for a half-width under 0.1, which sizes the sweep at ~100."""
    low, high = wilson_interval(wins=50, games=100)

    assert (high - low) / 2 < 0.1


def test_standings_separate_wins_losses_ties_and_errors() -> None:
    """An errored episode is not a loss; counting it as one hides broken agents."""
    results = [
        Result(task_id="meta-build:1", agent_id="a", score=1.0),
        Result(task_id="meta-build:2", agent_id="a", score=0.0),
        Result(task_id="meta-build:3", agent_id="a", score=0.5),
        Result(task_id="meta-build:4", agent_id="a", score=0.0, error="boom"),
    ]

    (standing,) = standings(results)

    assert isinstance(standing, Standing)
    assert standing.opponent == "meta-build"
    assert (standing.wins, standing.losses, standing.ties) == (1, 1, 1)
    assert standing.errors == 1
    assert standing.games == 3


def test_standings_average_the_banks_of_scored_games() -> None:
    """Bank is no longer the acceptance metric, but it is still the diagnostic."""
    results = [
        Result(task_id="tape:1", agent_id="a", score=0.0, scores=[40000.0, 170000.0]),
        Result(task_id="tape:2", agent_id="a", score=0.0, scores=[60000.0, 150000.0]),
        Result(task_id="tape:3", agent_id="a", score=0.0, error="boom"),
    ]

    (standing,) = standings(results)

    assert standing.bank == 50000.0
    assert standing.opponent_bank == 160000.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kaggriculture.report'`

- [ ] **Step 3: Write the implementation**

Create `src/kaggriculture/report.py`:

```python
"""Turning episode results into the number the ladder actually scores.

The competition ranks by wins, not by coins banked, so every acceptance
decision reads a win rate here. Rates carry a confidence interval because a
sixteen-seed result and a two-hundred-seed result are otherwise indistinguishable
on the page, and the difference decides whether a change ships.
"""

import math
from typing import Iterable

from pydantic import BaseModel

from kaggriculture.result import Result


def wilson_interval(wins: float, games: int, z: float = 1.96) -> tuple[float, float]:
    """Return a Wilson score interval for a win rate.

    The normal approximation collapses to zero width at zero wins, which is
    exactly where this agent has spent most of its life, so it would report
    certainty precisely when there is none. Wilson stays finite at both ends.

    Args:
        wins: Wins, counting a tie as half.
        games: Games played, excluding episodes that errored.
        z: Standard-normal quantile; 1.96 gives 95%.

    Returns:
        Lower and upper bounds, each within [0, 1].
    """
    if games <= 0:
        return 0.0, 1.0
    rate = wins / games
    denominator = 1 + z**2 / games
    centre = (rate + z**2 / (2 * games)) / denominator
    spread = z * math.sqrt(rate * (1 - rate) / games + z**2 / (4 * games**2))
    spread /= denominator
    return max(0.0, centre - spread), min(1.0, centre + spread)


class Standing(BaseModel):
    """One agent's record against one opponent."""

    opponent: str
    games: int
    wins: int
    losses: int
    ties: int
    errors: int
    win_rate: float
    low: float
    high: float
    bank: float
    opponent_bank: float

    @property
    def half_width(self) -> float:
        """Return half the confidence interval, which is what gates read."""
        return (self.high - self.low) / 2


def standings(results: list[Result]) -> list[Standing]:
    """Return one ``Standing`` per opponent, in first-seen order.

    Episodes that errored are counted separately rather than as losses: a
    crashed agent and a beaten one need different responses, and folding them
    together hides the crash.
    """
    by_opponent: dict[str, list[Result]] = {}
    for result in results:
        by_opponent.setdefault(result.opponent, []).append(result)

    table: list[Standing] = []
    for opponent, played in by_opponent.items():
        scored = [result for result in played if result.error is None]
        wins = sum(result.score == 1.0 for result in scored)
        ties = sum(result.score == 0.5 for result in scored)
        losses = len(scored) - wins - ties
        points = wins + ties / 2
        low, high = wilson_interval(points, len(scored))
        banked = [result.scores for result in scored if result.scores]
        table.append(
            Standing(
                opponent=opponent,
                games=len(scored),
                wins=wins,
                losses=losses,
                ties=ties,
                errors=len(played) - len(scored),
                win_rate=points / len(scored) if scored else 0.0,
                low=low,
                high=high,
                bank=mean(pair[0] for pair in banked) if banked else 0.0,
                opponent_bank=mean(pair[1] for pair in banked) if banked else 0.0,
            )
        )
    return table


def mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean, or zero for an empty sequence."""
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def format_standing(standing: Standing) -> str:
    """Return one aligned line describing a standing."""
    return (
        f"vs {standing.opponent:<14s} "
        f"{standing.win_rate:.3f} [{standing.low:.3f}, {standing.high:.3f}]  "
        f"({standing.wins}W {standing.losses}L {standing.ties}T"
        f"{f' {standing.errors}E' if standing.errors else ''})"
        f"  bank {standing.bank:8.0f} vs {standing.opponent_bank:8.0f}"
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_report.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
uv run pre-commit run -a
git add src/kaggriculture/report.py tests/test_report.py
git commit -m "feat: score agents by win rate with a confidence interval

The ladder ranks by wins, and we have been accepting changes on mean bank
against a single opponent. Wilson rather than the normal approximation
because we are usually at zero wins, where the normal approximation reports
a zero-width interval and therefore certainty exactly where there is none."
```

---

### Task 2: Frozen league opponents

The heuristic is currently measured against one open-loop replay. Beating a recording can be achieved by exploiting its blindness, which is why STRATEGY.md warns against using it as a sole adversary. These two baselines are _reactive_ — they respond to the board — and they are pinned configurations of the existing policy rather than copies of it, so they exercise the shipped code path and cannot drift out of date.

**Files:**

- Create: `baselines/heuristic_v1.py`
- Create: `baselines/meta_build.py`
- Modify: `src/kaggriculture/config.py` (add `LEAGUE`, default `opponents` to it)
- Test: `tests/test_baselines.py`

**Interfaces:**

- Consumes: `kaggriculture.policy.Strategy`, `kaggriculture.policy.plan`, `kaggriculture.observation.Observation`.
- Produces:
  - `baselines/heuristic_v1.py::agent(raw_obs: Mapping[str, Any]) -> dict[str, Any]`
  - `baselines/meta_build.py::agent(raw_obs: Mapping[str, Any]) -> dict[str, Any]`
  - `kaggriculture.config.LEAGUE: tuple[str, ...]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_baselines.py`:

```python
"""Every league member must play a legal season, or the league lies."""

import pytest
from kaggle_environments import make

from kaggriculture.config import LEAGUE
from kaggriculture.constants import ENVIRONMENT


@pytest.mark.parametrize("baseline", LEAGUE)
def test_league_member_plays_a_legal_season(baseline: str) -> None:
    """A baseline that errors would be scored as a free win for everyone."""
    env = make(ENVIRONMENT, configuration={"episodeSteps": 96, "seed": 3})

    env.run([baseline, "pass"])

    assert [state.status for state in env.steps[-1]] == ["DONE", "DONE"]


def test_heuristic_v1_keeps_no_livestock() -> None:
    """v1 predates the herd; if it grew animals it would not be the v1 we shipped."""
    from baselines.heuristic_v1 import STRATEGY

    assert sum(STRATEGY.herd.values()) == 0


def test_meta_build_matches_the_measured_ladder_build() -> None:
    """The reconstruction is only useful if it is the build 75% of the field plays."""
    from baselines.meta_build import STRATEGY

    assert dict(STRATEGY.crops) == {"WHEAT": 11, "MELON": 11, "STRAWBERRY": 40}
    assert dict(STRATEGY.herd) == {"COW": 8, "SHEEP": 6}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_baselines.py -v`
Expected: FAIL with `ImportError: cannot import name 'LEAGUE'`

- [ ] **Step 3: Write the baselines**

Create `baselines/heuristic_v1.py`:

```python
"""The melon monoculture that first went on the ladder, frozen as an opponent.

Pinned as a configuration rather than a copy of the policy, so it keeps
exercising the shipped code path. It is kept in the league because it is a
genuinely different shape of opponent — no livestock, one crop, a small crew —
and a change that only beats livestock builds has not been shown to generalise.
"""

from typing import Any, Mapping

from kaggriculture.observation import Observation
from kaggriculture.policy import Strategy, plan

STRATEGY = Strategy(
    crops={"MELON": 36},
    herd={},
    max_hands=8,
    tiles_per_unit=4,
    land_reserve=800,
)


def agent(raw_obs: Mapping[str, Any]) -> dict[str, Any]:
    """Return this turn's action dict for the environment."""
    return plan(Observation.parse(raw_obs), STRATEGY).to_action()
```

Create `baselines/meta_build.py`:

```python
"""A reactive reconstruction of the build 75% of the ladder plays.

Measured from a replay: wheat and melon from day three, strawberry ramped to
forty, all three quadrants by day twelve, eight cows and six sheep in place by
day twelve. The recorded tape at
``/data/kaggriculture/baselines/meta_tape.py`` plays the same build but
open-loop, so it cannot respond to anything we do. Beating the recording can be
achieved by exploiting its blindness; beating this one cannot, which is why both
belong in the league.
"""

from typing import Any, Mapping

from kaggriculture.observation import Observation
from kaggriculture.policy import Strategy, plan

STRATEGY = Strategy(
    crops={"WHEAT": 11, "MELON": 11, "STRAWBERRY": 40},
    herd={"COW": 8, "SHEEP": 6},
    max_hands=12,
    tiles_per_unit=5,
)


def agent(raw_obs: Mapping[str, Any]) -> dict[str, Any]:
    """Return this turn's action dict for the environment."""
    return plan(Observation.parse(raw_obs), STRATEGY).to_action()
```

- [ ] **Step 4: Register the league**

`LEAGUE` goes in `src/kaggriculture/config.py`, **not** `constants.py`. `constants.py` is imported by
`policy.py` and therefore ships inside the submission archive, where an absolute path to a local
directory is machine-specific dead weight. `config.py` is imported only by the harness and the runner,
neither of which ships.

Add to `src/kaggriculture/config.py`, above the `HarnessConfig` class:

```python
# Frozen named opponents, strongest first. Paths rather than names because the
# environment loads them as files; the recorded tape is kept alongside the
# reactive reconstruction because the two fail differently — a recording can be
# beaten by exploiting its blindness, and a reactive opponent cannot.
LEAGUE = (
    "baselines/meta_build.py",
    "/data/kaggriculture/baselines/meta_tape.py",
    "baselines/heuristic_v1.py",
    "starter",
)
```

and change the field to:

```python
    opponents: tuple[str, ...] = LEAGUE
```

`BASELINE_AGENTS` stays in `constants.py` untouched — it names built-in environment agents, which are
not paths.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_baselines.py -v`
Expected: PASS, 6 tests (4 parametrised plus 2).

If `test_league_member_plays_a_legal_season` fails for `heuristic_v1` with a `KeyError` on an animal lookup, the empty-herd path is broken — fix `pasture_sites` and `animal_orders` to handle `herd={}` rather than changing the baseline.

- [ ] **Step 6: Commit**

```bash
uv run pre-commit run -a
git add baselines tests/test_baselines.py src/kaggriculture/config.py
git commit -m "feat: freeze a league of named opponents

Tuning against one open-loop recording rewards exploiting its blindness,
which the strategy document warned about and we then spent a day doing. The
reconstruction plays the same measured build but reacts, so it cannot be
beaten that way. Both stay in the league because they fail differently.

Baselines are pinned Strategy configurations rather than copies of the
policy, so they keep exercising the shipped code path."
```

---

### Task 3: Replay analysis module

Every diagnosis so far — animals starving, land bought six days late, the shed ending full — came from inspecting replays with throwaway code. Each later phase reads the same signals, so they belong in a module with tests.

**Files:**

- Create: `src/kaggriculture/replay.py`
- Test: `tests/test_replay.py`

**Interfaces:**

- Consumes: episode JSON as written by `EpisodeAgent.run` (`{"steps": [[{"observation": ..., "reward": ...}, ...], ...]}`).
- Produces:
  - `load(path: Path) -> list[list[dict[str, Any]]]`
  - `class Season(BaseModel)` with fields `player: int`, `bank: list[float]`, `crops: dict[int, dict[str, int]]`, `animals: list[int]`, `animals_lost: int`, `final_shed: dict[str, int]`, `final_prices: dict[str, int]`
  - `summarise(steps: list[list[dict[str, Any]]], player: int) -> Season`

- [ ] **Step 1: Write the failing test**

Create `tests/test_replay.py`:

```python
"""Tests for replay analysis, run against a real short episode."""

import json
from pathlib import Path

from kaggle_environments import make

from kaggriculture.constants import ENVIRONMENT
from kaggriculture.replay import Season, load, summarise


def test_summarise_reads_a_real_episode(tmp_path: Path) -> None:
    """The module has to read what the harness actually writes."""
    env = make(ENVIRONMENT, configuration={"episodeSteps": 96, "seed": 5})
    env.run(["baselines/heuristic_v1.py", "pass"])
    path = tmp_path / "episode.json"
    path.write_text(json.dumps(env.toJSON()))

    season = summarise(load(path), player=0)

    assert isinstance(season, Season)
    assert len(season.bank) == 96
    assert season.bank[0] == 3000
    assert season.animals_lost >= 0
    assert set(season.final_prices) >= {"WHEAT", "MILK", "STRAWBERRY"}


def test_animals_lost_counts_every_disappearance(tmp_path: Path) -> None:
    """A starved animal is a bought asset walking off the farm, and must be visible."""
    steps = [
        [{"observation": {"day": 0, "farms": [{"tiles": [[{"animal": "COW"}, {"animal": "COW"}]], "money": 3000}], "private": {"shed": {}}, "market": {"prices": {}}}, "reward": 3000}],
        [{"observation": {"day": 1, "farms": [{"tiles": [[{"animal": "COW"}, None]], "money": 3000}], "private": {"shed": {}}, "market": {"prices": {}}}, "reward": 3000}],
    ]

    season = summarise(steps, player=0)

    assert season.animals_lost == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_replay.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kaggriculture.replay'`

- [ ] **Step 3: Write the implementation**

Create `src/kaggriculture/replay.py`:

```python
"""Reading a finished episode back for diagnosis.

Every finding worth acting on so far came from a replay rather than from a
score: livestock starving because one carrier cannot walk fourteen pastures,
land bought six days later than the meta build buys it, a season ending with
fifty-two melons still in the shed. A score says a change was worse; a replay
says why.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

Step = list[dict[str, Any]]


class Season(BaseModel):
    """One player's season, reduced to the signals worth looking at."""

    player: int
    bank: list[float]
    crops: dict[int, dict[str, int]]
    animals: list[int]
    animals_lost: int
    final_shed: dict[str, int]
    final_prices: dict[str, int]


def load(path: Path) -> list[Step]:
    """Return the steps of a replay written by ``EpisodeAgent``."""
    return json.loads(path.read_text())["steps"]


def summarise(steps: list[Step], player: int) -> Season:
    """Return one player's season summary.

    Args:
        steps: Replay steps, outermost index being the turn.
        player: Seat to report on.

    Returns:
        A ``Season`` holding the bank trajectory, the crop mix on each day, the
        herd size per turn, how many animals vanished, and the closing shed and
        prices.
    """
    bank: list[float] = []
    crops: dict[int, dict[str, int]] = {}
    animals: list[int] = []
    lost = 0
    for step in steps:
        observation = step[0]["observation"]
        farm = observation["farms"][player]
        bank.append(farm["money"])
        crops[observation["day"]] = crop_mix(farm)
        herd = sum(
            1
            for row in farm["tiles"]
            for tile in row
            if isinstance(tile, dict) and tile.get("animal")
        )
        if animals and herd < animals[-1]:
            lost += animals[-1] - herd
        animals.append(herd)

    closing = steps[-1][player]["observation"]
    return Season(
        player=player,
        bank=bank,
        crops=crops,
        animals=animals,
        animals_lost=lost,
        final_shed=dict(closing.get("private", {}).get("shed", {})),
        final_prices=dict(steps[-1][0]["observation"]["market"]["prices"]),
    )


def crop_mix(farm: dict[str, Any]) -> dict[str, int]:
    """Return how many tiles of the farm hold each crop."""
    mix: dict[str, int] = {}
    for row in farm["tiles"]:
        for tile in row:
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                mix[tile["crop"]] = mix.get(tile["crop"], 0) + 1
    return mix
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_replay.py -v`
Expected: PASS, 2 tests.

- [ ] **Step 5: Commit**

```bash
uv run pre-commit run -a
git add src/kaggriculture/replay.py tests/test_replay.py
git commit -m "feat: read replays through a module rather than throwaway code

A score says a change was worse. A replay says why, and every finding worth
acting on so far came from one — starving livestock, land bought six days
late, a shed still full at the final bell. Every later phase reads these
same signals, so they belong somewhere tested."
```

---

### Task 4: Report win rate from the runner

**Files:**

- Modify: `src/kaggriculture/scripts/run.py:52-70` (replace `report`, drop its unused parameter)
- Test: covered by Task 1's unit tests plus a manual gate run in Task 5.

**Interfaces:**

- Consumes: `kaggriculture.report.standings`, `kaggriculture.report.format_standing`.
- Produces: `report(results: list[Result]) -> None`. The old signature took a `HarnessConfig` that the new body never reads, and unused parameters are an unused code path.

- [ ] **Step 1: Replace the reporting function**

In `src/kaggriculture/scripts/run.py`, replace the `report` function and drop the now-unused `from statistics import mean` import:

```python
def report(results: list[Result]) -> None:
    """Log one line per opponent, plus any episode that failed to run.

    Win rate rather than bank: the ladder scores wins, and a change that banks
    more while winning less has not improved anything. The interval is what
    decides whether a difference is real or is sixteen seeds of noise.
    """
    for standing in standings(results):
        LOGGER.info("%s", format_standing(standing))
    for result in results:
        if result.error:
            LOGGER.error("%s failed: %s", result.task_id, result.error)
```

Add the import `from kaggriculture.report import format_standing, standings` and remove `from statistics import mean`. Update the call site in `main` from `report(results, config)` to `report(results)`. Drop the now-unused `HarnessConfig` import from the module only if nothing else in the file uses it — `main` constructs one, so it stays.

- [ ] **Step 2: Verify the runner still works end to end**

Run: `uv run python -m kaggriculture.scripts.run --games 2 --workers 2 --opponents pass`
Expected: one line of the form `vs pass           1.000 [0.342, 1.000]  (2W 0L 0T)`

- [ ] **Step 3: Commit**

```bash
uv run pre-commit run -a
git add src/kaggriculture/scripts/run.py
git commit -m "feat: report win rate and interval instead of mean bank

A change that banks more while winning less has not improved anything, and
the ladder only counts wins."
```

---

### Task 5: Log every evaluation run to Weights & Biases

Roughly a dozen sweeps were run in one day — herd size, crop, sell rate, tiles per unit, land reserve — and every result exists only in terminal scrollback and prose in commit messages. That is not a record anyone can query, and it is why re-checking whether an old constant still holds means re-running it. One wandb run per evaluation, with the whole `Strategy` as config, makes the sweep table the thing it should always have been: a queryable record of which configuration beat which opponent, comparable across days.

**Critical:** this module lives under `src/kaggriculture/scripts/`, which `package.py` excludes from the submission archive (`EXCLUDED = shutil.ignore_patterns("__pycache__", "scripts")`). `wandb` must never reach the competition sandbox — there is no network there and no API key, and an import error on turn zero forfeits the game.

**Files:**

- Create: `src/kaggriculture/scripts/tracking.py`
- Modify: `src/kaggriculture/scripts/run.py` (add `--track` flag, call the logger)
- Modify: `pyproject.toml` (via `uv add`)
- Test: `tests/test_tracking.py`

**Interfaces:**

- Consumes: `kaggriculture.report.Standing`, `kaggriculture.config.HarnessConfig`, `kaggriculture.policy.Strategy`.
- Produces:
  - `run_config(config: HarnessConfig, strategy: Strategy, agent: str) -> dict[str, Any]`
  - `run_metrics(standings: list[Standing]) -> dict[str, float]`
  - `log_evaluation(standings: list[Standing], config: HarnessConfig, strategy: Strategy, agent: str) -> None`
  - Module constants `PROJECT = "kaggriculture-2026"`, `ENTITY = "will-rice"`

- [ ] **Step 1: Add the dependency**

```bash
uv add wandb
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_tracking.py`. The two payload builders are pure functions, so they are tested directly and no network is touched:

```python
"""Tests for the shape of what we send to Weights & Biases."""

from kaggriculture.config import HarnessConfig
from kaggriculture.policy import Strategy
from kaggriculture.report import Standing
from kaggriculture.scripts.tracking import run_config, run_metrics


def standing(opponent: str, win_rate: float) -> Standing:
    """Return a standing with the fields the metric builder reads."""
    return Standing(
        opponent=opponent,
        games=100,
        wins=int(win_rate * 100),
        losses=100 - int(win_rate * 100),
        ties=0,
        errors=0,
        win_rate=win_rate,
        low=win_rate - 0.05,
        high=win_rate + 0.05,
        bank=50000.0,
        opponent_bank=170000.0,
    )


def test_run_config_carries_every_strategy_knob() -> None:
    """A sweep is only comparable if the configuration that produced it is recorded."""
    config = run_config(HarnessConfig(), Strategy(), agent="main.py")

    assert config["crops"] == {"WHEAT": 11, "MELON": 11, "STRAWBERRY": 40}
    assert config["sell_rate"] == 2
    assert config["tiles_per_unit"] == 5
    assert config["agent"] == "main.py"
    assert "games" in config and "seed" in config


def test_metric_keys_survive_opponents_named_by_path() -> None:
    """League members are file paths; slashes would nest them into separate charts."""
    metrics = run_metrics([standing("baselines/heuristic_v1.py", 0.4)])

    assert metrics["win_rate/heuristic_v1"] == 0.4
    assert metrics["bank/heuristic_v1"] == 50000.0
    assert not any("/" in key.split("/", 1)[1] for key in metrics)


def test_league_win_rate_is_the_headline_number() -> None:
    """One number decides whether a change ships, and it spans the whole league."""
    metrics = run_metrics([standing("a.py", 0.4), standing("b.py", 0.6)])

    assert metrics["win_rate/league"] == 0.5
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_tracking.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kaggriculture.scripts.tracking'`

- [ ] **Step 4: Write the implementation**

Create `src/kaggriculture/scripts/tracking.py`:

```python
"""Recording evaluation runs to Weights & Biases.

A dozen sweeps in a day left their results in terminal scrollback and in prose,
which means the only way to recheck an old constant is to run it again. One run
per evaluation, carrying the whole ``Strategy`` as config, turns the sweep into
something queryable: which configuration, against which opponent, at what win
rate, at which commit.

This module deliberately lives under ``scripts``, which ``package.py`` excludes
from the submission archive. The competition sandbox has no network and no API
key, so an import of ``wandb`` reaching it would forfeit the episode on turn
zero.
"""

import dataclasses
import logging
import subprocess
from pathlib import Path
from typing import Any

import wandb

from kaggriculture.config import HarnessConfig
from kaggriculture.policy import Strategy
from kaggriculture.report import Standing

LOGGER = logging.getLogger(__name__)

PROJECT = "kaggriculture-2026"
ENTITY = "will-rice"


def log_evaluation(
    standings: list[Standing],
    config: HarnessConfig,
    strategy: Strategy,
    agent: str,
) -> None:
    """Send one evaluation run to Weights & Biases."""
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        job_type="evaluation",
        config=run_config(config, strategy, agent),
    )
    run.log(run_metrics(standings))
    run.finish()
    LOGGER.info("logged to %s", run.url)


def run_config(
    config: HarnessConfig, strategy: Strategy, agent: str
) -> dict[str, Any]:
    """Return the configuration that produced a run, flat enough to group by.

    Every ``Strategy`` field is included because any of them may turn out to be
    the one that mattered — the herd size and the crop both did, and neither was
    predictable in advance.
    """
    return {
        **dataclasses.asdict(strategy),
        "agent": agent,
        "games": config.games,
        "seed": config.seed,
        "opponents": list(config.opponents),
        "episode_steps": config.episode_steps,
        "commit": commit(),
    }


def run_metrics(standings: list[Standing]) -> dict[str, float]:
    """Return the metrics for one evaluation, keyed by opponent.

    Opponents are named by file path, so the stem is used: a slash in a metric
    key nests it into a separate chart group and the league stops being
    comparable at a glance.
    """
    metrics: dict[str, float] = {}
    for standing in standings:
        name = Path(standing.opponent).stem
        metrics[f"win_rate/{name}"] = standing.win_rate
        metrics[f"low/{name}"] = standing.low
        metrics[f"high/{name}"] = standing.high
        metrics[f"bank/{name}"] = standing.bank
        metrics[f"opponent_bank/{name}"] = standing.opponent_bank
    if standings:
        metrics["win_rate/league"] = sum(s.win_rate for s in standings) / len(standings)
    return metrics


def commit() -> str:
    """Return the short commit the evaluation ran at, or ``unknown``."""
    finished = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return finished.stdout.strip() or "unknown"
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_tracking.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 6: Wire it into the runner**

In `src/kaggriculture/scripts/run.py`, add the flag inside `main`'s parser:

```python
    parser.add_argument(
        "--track", action="store_true", help="log this evaluation to wandb"
    )
```

and after `report(results, config)`:

```python
    if args.track:
        from kaggriculture.scripts.tracking import log_evaluation

        log_evaluation(standings(results), config, STRATEGY, args.agent)
```

Add `from kaggriculture.policy import STRATEGY` to the imports. The `tracking` import stays inside the branch so an evaluation without `--track` never imports `wandb`, which keeps the common path fast and keeps a missing API key from breaking a local sweep.

- [ ] **Step 7: Verify against the real service**

```bash
uv run python -m kaggriculture.scripts.run --games 4 --workers 4 --opponents pass --track
```

Expected: the usual standings line, then a `logged to https://wandb.ai/will-rice/kaggriculture-2026/runs/...` line. Open it and confirm the config panel shows all sixteen `Strategy` fields and the commit.

- [ ] **Step 8: Commit**

```bash
uv run pre-commit run -a
git add src/kaggriculture/scripts/tracking.py tests/test_tracking.py src/kaggriculture/scripts/run.py pyproject.toml uv.lock
git commit -m "feat: log evaluations to wandb

A dozen sweeps in one day left their results in terminal scrollback and in
commit prose, so rechecking an old constant means running it again. One run
per evaluation carrying the whole Strategy as config makes the sweep
queryable instead.

Kept under scripts/, which the submission archive excludes: the sandbox has
no network and no API key, and an import of wandb reaching it would forfeit
the episode on turn zero."
```

---

### Task 6: Pass the Phase 1 gate

STRATEGY.md's gate: `heuristic-v1` vs the strongest league opponent over 100 seeded games returns a win rate
with a CI narrower than ±0.1, in under 10 minutes on 64 cores.

> **Substitution.** The gate was written as `heuristic-v1` vs `meta-build`. `meta_build` was ruled out of
> the league during Task 2 — it was byte-for-byte the shipped agent, so the matchup would have been
> self-play. The recorded tape takes its place as the strongest available opponent. The gate tests
> whether the infrastructure returns a tight interval quickly, and any real opponent serves that.

**Files:**

- Modify: `STRATEGY.md` (record the result under Phase 1)

- [ ] **Step 1: Run the gate**

```bash
time uv run python -m kaggriculture.scripts.run \
  --agent baselines/heuristic_v1.py \
  --opponents /data/kaggriculture/baselines/meta_tape.py \
  --games 100 --workers 64 --seed 1000 --track
```

Expected: a `vs baselines/meta_build.py` line whose interval half-width is below 0.1, and a wall-clock under 10 minutes. At 100 games an even split gives a half-width of about 0.098, so the gate is satisfied by any result at this sample size; a lopsided result gives a narrower one.

- [ ] **Step 2: Run the full league for a reference standing**

```bash
uv run python -m kaggriculture.scripts.run --games 100 --workers 64 --seed 1000 --track
```

Record the four resulting lines. This is the baseline every later change is compared against.

- [ ] **Step 3: Record the result and the new acceptance rule**

Add under Phase 1 in `STRATEGY.md`, filling in the measured numbers:

```markdown
#### Result — gate passed YYYY-MM-DD

`heuristic-v1` vs the recorded tape over 100 seeded games: win rate X.XXX
[low, high], half-width 0.0XX, in Xm Xs on 64 cores.

Reference league standing for the current submission, 100 games each:

| opponent     | win rate | 95% interval |
| ------------ | -------- | ------------ |
| meta-build   |          |              |
| meta-tape    |          |              |
| heuristic-v1 |          |              |
| starter      |          |              |

**From here, a change ships only if it beats the current submission on win
rate against this league, with non-overlapping intervals.** Mean bank against a
single opponent is retired as an acceptance metric; it was what let a herd
tuned to one open-loop recording look like progress.
```

- [ ] **Step 4: Commit**

```bash
uv run pre-commit run -a
git add STRATEGY.md
git commit -m "docs: Phase 1 gate passed, and bank is retired as an acceptance metric

Every heuristic decision so far was accepted on mean bank against one
open-loop recording. That is neither what the ladder scores nor a defensible
sample, and the strategy document said so before we did it anyway."
```

---

## Self-Review

**Spec coverage.** STRATEGY.md Phase 1 asks for three things. Frozen named opponents including a reconstructed meta build — Task 2, with the tape retained alongside the reconstruction because they fail differently. Head-to-head evaluation with paired seeds reporting win rate with a confidence interval — Task 1 and Task 4; seeds are already paired across matchups by `Harness.matches`, which iterates opponents over one shared `range`. A replay-analysis module reporting crop mix, bank trajectory, prices and inventory — Task 3, which also reports animals lost, since that is a live defect. The gate is Task 5.

Run tracking is Task 5, which is not in STRATEGY.md's Phase 1 list but belongs to the same subsystem: the plan retires bank-against-one-opponent as the acceptance metric, and a queryable record of what was measured is what stops the replacement decaying the same way.

**Deliberately out of scope**, each needing its own plan: forward search over market decisions (the 99.9% of the turn budget currently unused); automated constant optimisation over the 16 `Strategy` knobs; the RL track's Phases 2–4. All three depend on this plan's win-rate metric to be measurable, which is why this one comes first.

**Placeholders.** None. The one templated block is the results table in Task 5, whose numbers cannot exist before the run that produces them.

**Type consistency.** `run_config` and `run_metrics` read only `Standing` fields defined in Task 1, including `bank` and `opponent_bank` which Task 1 adds for this purpose. `Standing` and `Season` are pydantic models as the codebase uses for `Result` and `HarnessConfig`. `standings` and `format_standing` are the only names Task 4 imports and both are defined in Task 1. `LEAGUE` is defined in Task 2 and consumed by `config.py` in the same task. `load` and `summarise` are used only inside Task 3's tests.

**One risk worth stating.** Task 2 assumes `Strategy(herd={})` runs cleanly, which the current code appears to support but has never been exercised. Step 5 of that task names the symptom and says to fix the policy rather than the baseline if it does not.
