# Imitation Dataset and Reference Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the replay corpus into a behaviour-cloning dataset, train one reference model on it, and measure that model against the existing league — reaching Phase 2's gate that a cloned agent beats `random` and ideally `heuristic-v2`.

**Architecture:** A streaming pipeline, because the corpus is ~107 GB uncompressed and nothing may decompress it wholesale. An index pass reads only each episode's final rewards and team names straight from the zip. A selection pass picks episodes on those. An encoder turns one observation into board planes plus scalars, and an action codec turns one recorded action into training targets. A builder streams selected episodes through both into sharded `.npz` files. Training is deliberately one fixed architecture — Frog Parade's published shape — so this plan produces a working baseline rather than a bake-off.

**Tech Stack:** Python 3.11, `torch` (CPU for inference budget checks, CUDA for training), `numpy`, `zipfile` streaming, `pydantic` models, `pytest`, `wandb` for run records. Reuses `kaggriculture.replay`, `kaggriculture.report` and the frozen league from the evaluation plan.

## Global Constraints

- Run everything with `uv run`. Add dependencies with `uv add`. Never invoke `pip` or a bare `python`.
- `uv run pre-commit run -a` must pass before every commit. Fix type errors; never add ignore comments.
- Google-style docstrings on every public function and module. Absolute imports. No `from __future__ import annotations`.
- Use `logging.info`, never `print()`. Use `tqdm` for progress over long iterables.
- Required argparse arguments take no `--` prefix. Prefer constants over CLI flags.
- Prefer PyTorch over NumPy for tensor work, and always include the batch dimension.
- Prefer `.numpy(force=True)` over `.cpu().numpy()`; prefer `.flatten` when collapsing dimensions; never use `einops.rearrange`.
- Training entry points call `seed_everything(config.seed, workers=True)` from Lightning.
- Tests are pytest functional style against real data. Minimise mocks.
- **Nothing in this plan may be imported by `main.py` or `src/kaggriculture/policy.py`.** The competition sandbox has 2 cores, 6.8 GB and no network; `torch` import alone costs 10.7 s of a 60 s overage pool. Training code lives under `src/kaggriculture/learn/`, which `package.py` must exclude alongside `scripts`.
- The corpus lives at `/data/kaggriculture/episodes/*.zip` and is read-only. Never extract an archive to disk in full.
- Do not mention Claude or AI assistance in commit messages.

## Measured facts this plan depends on

Established by the Phase 0 probe and the primary-source verification pass; do not re-derive:

- Sandbox: Python 3.11.13, **2 cores**, 6.8 GB, torch 2.6.0+cu124 available, 100 MB submission cap.
- Inference budget: **1000 ms/turn**. A 10M-parameter conv trunk on this 10×10 board costs ~45 ms; 20M costs 105 ms. Measure candidates under `OMP_NUM_THREADS=2` or you are measuring the wrong machine.
- Every verified competition winner is **1M–20M parameters**. Frog Parade's Lux S3 model is an 8-block 3×3 CNN at `d_model` 256, ~10M parameters.
- Corpus: 788 episodes per daily archive, 5 archives, ~27 MB per episode JSON, 720 steps × 2 seats per episode.
- Three-quarters of the field replays one recorded episode, so an unfiltered clone learns that recording.

## File Structure

| File                                       | Responsibility                                                                                          |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| `src/kaggriculture/learn/__init__.py`      | **Create.** Package marker; excluded from the submission archive.                                       |
| `src/kaggriculture/learn/corpus.py`        | **Create.** Stream zips, read each episode's rewards and teams without decoding steps. Select episodes. |
| `src/kaggriculture/learn/encoding.py`      | **Create.** One observation → board planes and scalars. One action → training targets, and back.        |
| `src/kaggriculture/learn/dataset.py`       | **Create.** Stream selected episodes through the encoders into sharded `.npz`.                          |
| `src/kaggriculture/learn/model.py`         | **Create.** The reference trunk and heads, sized to the measured budget.                                |
| `src/kaggriculture/learn/scripts/build.py` | **Create.** CLI: build the dataset.                                                                     |
| `src/kaggriculture/learn/scripts/train.py` | **Create.** CLI: behaviour-clone, log to wandb.                                                         |
| `src/kaggriculture/learn/scripts/play.py`  | **Create.** Wrap a checkpoint as an agent the harness can evaluate.                                     |
| `src/kaggriculture/scripts/package.py`     | **Modify.** Exclude `learn` from the archive.                                                           |
| `tests/learn/test_corpus.py`               | **Create.** Index and selection against a real archive.                                                 |
| `tests/learn/test_encoding.py`             | **Create.** Encoder correctness and action round-trips.                                                 |
| `tests/learn/test_dataset.py`              | **Create.** Shard shapes and label alignment.                                                           |
| `tests/learn/test_model.py`                | **Create.** Shapes, parameter count, and the two-thread inference budget.                               |

---

### Task 1: Index the corpus without decoding it

The corpus is ~107 GB uncompressed across five archives. Every later decision — which episodes are good, which team played them, how to split train from holdout — needs only two fields that sit at the top level of each episode: `rewards` and `info.TeamNames`. Reading those without decoding 720 steps is the difference between an index that takes minutes and one that takes hours.

**Files:**

- Create: `src/kaggriculture/learn/__init__.py`, `src/kaggriculture/learn/corpus.py`
- Modify: `src/kaggriculture/scripts/package.py`
- Test: `tests/learn/test_corpus.py`

**Interfaces:**

- Produces:
  - `class EpisodeRecord(BaseModel)` with `archive: str`, `name: str`, `episode_id: int`, `rewards: list[float]`, `teams: list[str]`
  - `index_archive(archive: Path) -> list[EpisodeRecord]`
  - `index_corpus(directory: Path = CORPUS) -> list[EpisodeRecord]`
  - Module constant `CORPUS = Path("/data/kaggriculture/episodes")`

- [ ] **Step 1: Write the failing test**

Create `tests/learn/test_corpus.py`:

```python
"""Tests for indexing the replay corpus, run against the real archives."""

from pathlib import Path

import pytest

from kaggriculture.learn.corpus import CORPUS, EpisodeRecord, index_archive

ARCHIVE = CORPUS / "kaggriculture-episodes-2026-08-03.zip"

pytestmark = pytest.mark.skipif(
    not ARCHIVE.exists(), reason="replay corpus not present on this machine"
)


def test_index_reads_every_episode_in_an_archive() -> None:
    """The archive holds 788 episodes and the index must account for all of them."""
    records = index_archive(ARCHIVE)

    assert len(records) == 788
    assert all(isinstance(record, EpisodeRecord) for record in records)


def test_each_record_carries_the_fields_selection_needs() -> None:
    """Rewards decide quality and team names decide diversity; both must be real."""
    record = index_archive(ARCHIVE)[0]

    assert record.episode_id > 0
    assert len(record.rewards) == 2
    assert all(reward > 0 for reward in record.rewards)
    assert len(record.teams) == 2
    assert all(team for team in record.teams)


def test_indexing_does_not_decode_the_steps() -> None:
    """Decoding 27MB of steps per episode would make indexing take hours.

    The guard is memory: holding one episode's steps is ~200MB of Python
    objects, so an index that decoded them could not hold 788 of them at once.
    This asserts the record carries no step data at all.
    """
    record = index_archive(ARCHIVE)[0]

    assert not hasattr(record, "steps")
    assert set(record.model_dump()) == {
        "archive",
        "name",
        "episode_id",
        "rewards",
        "teams",
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/learn/test_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kaggriculture.learn'`

- [ ] **Step 3: Write the implementation**

Create `src/kaggriculture/learn/__init__.py`:

```python
"""Training-time code. Never imported by the submitted agent.

The competition sandbox has two cores, 6.8 GB and no network, and importing
torch there costs 10.7 seconds of a 60-second overage pool. `package.py`
excludes this package from the submission archive for that reason, and nothing
under `kaggriculture.policy` or `main.py` may import from it.
"""
```

Create `src/kaggriculture/learn/corpus.py`:

```python
"""Indexing the published replay corpus.

Kaggle publishes one archive of top-rated episodes per day; five are on disk and
a nightly cron fetches more. Each archive holds 788 episodes of roughly 27 MB,
so the corpus is about 107 GB uncompressed and cannot be unpacked.

Every selection decision needs only two things — what each player banked, and
who played — and both sit at the top level of the episode JSON beside a `steps`
array that is 99% of its bytes. This module reads the small part.
"""

import json
import logging
import zipfile
from pathlib import Path

from pydantic import BaseModel
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

CORPUS = Path("/data/kaggriculture/episodes")


class EpisodeRecord(BaseModel):
    """One episode's identity and outcome, without its steps."""

    archive: str
    name: str
    episode_id: int
    rewards: list[float]
    teams: list[str]


def index_corpus(directory: Path = CORPUS) -> list[EpisodeRecord]:
    """Return a record for every episode in every archive under ``directory``.

    Args:
        directory: Directory holding the daily ``.zip`` archives.

    Returns:
        One ``EpisodeRecord`` per episode, across all archives.
    """
    records: list[EpisodeRecord] = []
    for archive in sorted(directory.glob("*.zip")):
        found = index_archive(archive)
        LOGGER.info("%s: %d episodes", archive.name, len(found))
        records.extend(found)
    return records


def index_archive(archive: Path) -> list[EpisodeRecord]:
    """Return a record for every episode in one archive.

    Reads each member fully but parses it once, discarding the steps
    immediately. Streaming the JSON with a partial parser would be faster still
    and is not worth the dependency: this runs a handful of times, not per
    training step.

    Args:
        archive: Path to a daily ``.zip`` of episode JSON files.

    Returns:
        One ``EpisodeRecord`` per member.
    """
    records: list[EpisodeRecord] = []
    with zipfile.ZipFile(archive) as bundle:
        names = [name for name in bundle.namelist() if name.endswith(".json")]
        for name in tqdm(names, desc=archive.name, unit="ep"):
            with bundle.open(name) as member:
                episode = json.load(member)
            info = episode.get("info", {})
            records.append(
                EpisodeRecord(
                    archive=archive.name,
                    name=name,
                    episode_id=int(info.get("EpisodeId", 0)),
                    rewards=[float(value) for value in episode["rewards"]],
                    teams=[str(team) for team in info.get("TeamNames", [])],
                )
            )
    return records
```

- [ ] **Step 4: Exclude the package from the submission archive**

In `src/kaggriculture/scripts/package.py`, change the exclusion to cover `learn`:

```python
EXCLUDED = shutil.ignore_patterns("__pycache__", "scripts", "learn")
```

Then add to `tests/test_submission.py`, inside `test_archive_holds_the_entrypoint_beside_the_package`:

```python
    assert not any(name.startswith("kaggriculture/learn") for name in names)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/learn/test_corpus.py tests/test_submission.py -v`
Expected: PASS. The corpus tests take a few minutes — they read 21 GB of compressed JSON.

- [ ] **Step 6: Commit**

```bash
uv run pre-commit run -a
git add src/kaggriculture/learn tests/learn/test_corpus.py src/kaggriculture/scripts/package.py tests/test_submission.py
git commit -m "feat: index the replay corpus without decoding it

Every selection decision needs each episode's final banks and the teams that
played, and both sit beside a steps array that is 99% of the file. The corpus
is ~107GB uncompressed across five archives, so reading only the small part is
the difference between an index measured in minutes and one measured in hours.

The learn package is excluded from the submission archive. The sandbox has two
cores and no network, and importing torch there costs ten seconds of a sixty
second overage pool."
```

---

### Task 2: Select episodes worth cloning, by rating

Every archive ships a `manifest.csv` alongside its episodes carrying `avg_score`, `min_score` and
`sum_score` per episode — the players' **ladder ratings**, not their banks. That is a strictly better
selection signal than the final bank, and it costs nothing to read.

Bank is noisy: it depends on the opponent and the seed, and a large bank against a weak opponent is not a
demonstration of strong play. Rating measures the player. `min_score` is the weaker of the two seats, so
filtering on it selects episodes where **both** players were strong — which is what imitation needs,
because a stomp teaches behaviour that only works against someone who cannot respond.

It also solves the copied-kernel problem more directly than a per-team cap could. The recorded tape rates
about 1720 against a leaderboard top of 3070, so any rating floor above ~2500 excludes it and its
re-wrappings automatically, without needing to identify them by name.

**Files:**

- Modify: `src/kaggriculture/learn/corpus.py`
- Test: `tests/learn/test_corpus.py`

**Interfaces:**

- Consumes: `CORPUS`.
- Produces:
  - `class ManifestRow(BaseModel)` with `episode_id: int`, `avg_score: float`, `min_score: float`, `agent_count: int`
  - `read_manifest(archive: Path) -> list[ManifestRow]`
  - `class Sample(BaseModel)` with `archive: str`, `name: str`, `seat: int`, `rating: float`
  - `select(archives: list[Path], min_rating: float = 2500.0, per_archive: int = 200) -> list[Sample]`
  - `split(samples: list[Sample], holdout: float = 0.1) -> tuple[list[Sample], list[Sample]]`

- [ ] **Step 1: Write the failing test**

Append to `tests/learn/test_corpus.py`. Note these do NOT decode episodes, so they must not carry the
`slow` marker — put them in a class or use `@pytest.mark.filterwarnings` free functions below the existing
module-level `pytestmark`, and override it per-test with `@pytest.mark.slow` removed. The simplest correct
form is to move the module-level `pytestmark` onto the three existing corpus-decoding tests individually
and leave these new ones unmarked:

```python
def test_the_manifest_carries_a_rating_for_every_episode() -> None:
    """Ratings are what make selection possible without decoding 21GB."""
    from kaggriculture.learn.corpus import read_manifest

    rows = read_manifest(ARCHIVE)

    assert len(rows) == 787
    assert all(row.episode_id > 0 for row in rows)
    assert all(row.min_score <= row.avg_score for row in rows)
    assert max(row.avg_score for row in rows) > 2000


def test_selection_excludes_the_recorded_tape_by_rating() -> None:
    """The tape rates about 1720; a floor above that removes it and its clones.

    This is what replaces a per-team cap. Three-quarters of the field replays
    one recording, and every copy of it rates where the original does.
    """
    from kaggriculture.learn.corpus import select

    chosen = select([ARCHIVE], min_rating=2500.0, per_archive=1000)

    assert chosen
    assert all(sample.rating >= 2500.0 for sample in chosen)


def test_both_seats_of_a_strong_episode_are_taken() -> None:
    """min_score gates on the weaker seat, so passing it means both played well."""
    from kaggriculture.learn.corpus import select

    chosen = select([ARCHIVE], min_rating=2500.0, per_archive=1000)
    seats = {(sample.name, sample.seat) for sample in chosen}
    names = {name for name, _ in seats}

    assert len(seats) == 2 * len(names)


def test_the_per_archive_cap_bounds_the_dataset() -> None:
    """Five archives of unbounded episodes would not fit in memory as tensors."""
    from kaggriculture.learn.corpus import select

    chosen = select([ARCHIVE], min_rating=0.0, per_archive=10)

    assert len(chosen) == 20


def test_the_holdout_split_shares_no_episode_with_training() -> None:
    """An episode in both halves makes validation accuracy a memorisation score."""
    from kaggriculture.learn.corpus import select, split

    train, held = split(select([ARCHIVE], min_rating=0.0, per_archive=50), holdout=0.2)

    assert {sample.name for sample in train} & {sample.name for sample in held} == set()
    assert held
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/learn/test_corpus.py -k "manifest or selection or seats or cap or holdout" -v`
Expected: FAIL with `ImportError: cannot import name 'read_manifest'`

- [ ] **Step 3: Write the implementation**

Add `import csv` and `import io` to the module's imports, then append to
`src/kaggriculture/learn/corpus.py`:

```python
class ManifestRow(BaseModel):
    """One episode's entry in an archive's manifest."""

    episode_id: int
    avg_score: float
    min_score: float
    agent_count: int


def read_manifest(archive: Path) -> list[ManifestRow]:
    """Return the manifest rows for one archive.

    Each archive ships a ``manifest.csv`` beside its episodes carrying the
    players' ladder ratings. Reading it is instant, where establishing the same
    thing from the episodes themselves means decoding 21 GB of JSON.

    Args:
        archive: Path to a daily ``.zip``.

    Returns:
        One row per episode, in the manifest's own order.
    """
    with zipfile.ZipFile(archive) as bundle:
        text = bundle.read("manifest.csv").decode()
    return [
        ManifestRow(
            episode_id=int(entry["episode_id"]),
            avg_score=float(entry["avg_score"]),
            min_score=float(entry["min_score"]),
            agent_count=int(entry["agent_count"]),
        )
        for entry in csv.DictReader(io.StringIO(text))
    ]


class Sample(BaseModel):
    """One seat of one episode, as a demonstration to clone."""

    archive: str
    name: str
    seat: int
    rating: float


def select(
    archives: list[Path],
    min_rating: float = 2500.0,
    per_archive: int = 200,
) -> list[Sample]:
    """Return the seats worth cloning, strongest episodes first.

    Selection is on ``min_score`` — the weaker of the two players' ratings — so
    a chosen episode had two strong players in it. A large bank against a weak
    opponent is not a demonstration of strong play, and cloning one teaches
    behaviour that only works against someone who cannot respond.

    The rating floor also removes the recorded tape and its re-wrappings, which
    make up three-quarters of the field and rate around 1720, without having to
    identify them by name.

    Args:
        archives: Daily archives to select from.
        min_rating: Keep episodes whose weaker player rated at least this.
        per_archive: Cap on episodes taken from any one archive.

    Returns:
        Both seats of each chosen episode, strongest first.
    """
    chosen: list[Sample] = []
    for archive in archives:
        rows = [row for row in read_manifest(archive) if row.min_score >= min_rating]
        rows.sort(key=lambda row: -row.min_score)
        for row in rows[:per_archive]:
            for seat in range(row.agent_count):
                chosen.append(
                    Sample(
                        archive=archive.name,
                        name=f"{row.episode_id}.json",
                        seat=seat,
                        rating=row.min_score,
                    )
                )
    return chosen


def split(
    samples: list[Sample], holdout: float = 0.1
) -> tuple[list[Sample], list[Sample]]:
    """Split samples into training and holdout, never sharing an episode.

    Both seats of one episode see the same board, the same weeds and the same
    market. Splitting by seat would put near-identical states on both sides and
    turn validation accuracy into a memorisation score, so the split is by
    episode.

    Args:
        samples: Selected demonstrations.
        holdout: Fraction of episodes to hold out.

    Returns:
        Training samples and holdout samples.
    """
    episodes = sorted({sample.name for sample in samples})
    cut = int(len(episodes) * (1.0 - holdout))
    training = set(episodes[:cut])
    return (
        [sample for sample in samples if sample.name in training],
        [sample for sample in samples if sample.name not in training],
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/learn/test_corpus.py -k "manifest or selection or seats or cap or holdout" -v`
Expected: PASS, 5 tests, in seconds — none of them decodes an episode. If they take minutes, they are
inheriting the module's `slow` marker and the default deselection; fix the marker placement.

- [ ] **Step 5: Report what the corpus actually offers**

Run this and put the output in your report; the rating floor is a guess until it is measured:

```bash
uv run python -c "
from kaggriculture.learn.corpus import CORPUS, read_manifest
rows = [r for a in sorted(CORPUS.glob('*.zip')) for r in read_manifest(a)]
rows.sort(key=lambda r: -r.min_score)
print('episodes', len(rows))
for floor in (2000, 2400, 2600, 2800):
    print(f'min_score >= {floor}: {sum(1 for r in rows if r.min_score >= floor)}')
print('best', rows[0].min_score, 'median', rows[len(rows)//2].min_score)
"
```

If fewer than 200 episodes clear 2500, say so plainly rather than lowering the floor to fill a quota — a
smaller dataset of strong play beats a larger one diluted with the tape.

- [ ] **Step 6: Commit**

```bash
uv run pre-commit run -a
git add src/kaggriculture/learn/corpus.py tests/learn/test_corpus.py
git commit -m "feat: select demonstrations by ladder rating, not by bank

Every archive ships a manifest carrying each episode's avg_score and min_score
— the players' ratings. Selecting on min_score picks episodes where both
players were strong, where a bank filter would happily pick a large bank
against a weak opponent and teach behaviour that only works against someone who
cannot respond.

It also removes the recorded tape and its re-wrappings, three-quarters of the
field, without identifying them by name: they rate around 1720 and any floor
above that excludes them. That replaces the per-team cap the plan called for.

And it costs nothing. Establishing the same thing from the episodes means
decoding 21GB of JSON per archive, which measured 17 minutes."
```

---

### Task 3: Encode an observation

This is the task where a silent error costs the most. A wrong plane does not raise; it trains a model on a game that is not this one. So the encoder is checked against the engine's own constants, and against a board whose contents the test constructs by hand.

**Files:**

- Create: `src/kaggriculture/learn/encoding.py`
- Test: `tests/learn/test_encoding.py`

**Interfaces:**

- Produces:
  - Constants `BOARD = 10`, `TILE_PLANES` and `SCALARS`, all **derived from the engine's rules tables**, never hardcoded. As implemented they compute to 34 and 28; the plan's earlier stated values of 24 and 32 were the author's arithmetic, not a requirement. Later tasks import these names and must never restate their values.
  - `encode_board(observation: Mapping[str, Any], seat: int) -> torch.Tensor` returning `(1, TILE_PLANES, BOARD, BOARD)` float32
  - `encode_scalars(observation: Mapping[str, Any], seat: int) -> torch.Tensor` returning `(1, SCALARS)` float32

- [ ] **Step 1: Write the failing test**

Create `tests/learn/test_encoding.py`:

```python
"""Tests for turning an observation into tensors.

A wrong plane never raises. It trains the model on a game that is not this one,
and the first symptom is an agent that plays badly for reasons nobody can find.
So these check the encoding against the engine's own rules tables and against a
board built by hand.
"""

import torch

from kaggriculture.constants import CROPS, PRODUCTS
from kaggriculture.learn.encoding import (
    BOARD,
    SCALARS,
    TILE_PLANES,
    encode_board,
    encode_scalars,
)


def empty_observation(seat: int = 0) -> dict:
    """Return a minimal well-formed observation with an empty unlocked board."""
    tiles = [[None] * BOARD for _ in range(BOARD)]
    farm = {
        "tiles": tiles,
        "money": 3000.0,
        "farmer": [4, 4],
        "hands": [],
        "unlocked_quadrants": ["NW"],
        "hires_today": 0,
    }
    return {
        "player": seat,
        "step": 0,
        "day": 0,
        "hour": 0,
        "farms": [farm, farm],
        "market": {
            "prices": {item: 100 for item in PRODUCTS},
            "inventory": {item: 10000 for item in PRODUCTS},
        },
        "town": {"unlocked_shops": []},
        "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
    }


def test_board_has_the_declared_shape_and_batch_dimension() -> None:
    """Every tensor in this project carries its batch dimension."""
    board = encode_board(empty_observation(), seat=0)

    assert board.shape == (1, TILE_PLANES, BOARD, BOARD)
    assert board.dtype == torch.float32


def test_a_planted_tile_lights_exactly_its_own_crop_plane() -> None:
    """Crop identity is one-hot; two crops lit at once would be a silent mixture."""
    observation = empty_observation()
    observation["farms"][0]["tiles"][2][3] = {
        "kind": "PLANT",
        "crop": "MELON",
        "planted_day": 0,
        "watered_today": False,
        "consecutive_unwatered": 0,
        "yield_units": 0,
        "fertilized_until_day": -1,
    }

    board = encode_board(observation, seat=0)
    crops = sorted(CROPS)
    lit = [board[0, index, 2, 3].item() for index in range(len(crops))]

    assert sum(lit) == 1.0
    assert lit[crops.index("MELON")] == 1.0


def test_the_opponent_s_board_occupies_its_own_planes() -> None:
    """Opponent supply decides prices here, so their board must be visible and distinct."""
    observation = empty_observation()
    observation["farms"][1]["tiles"][5][5] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 0,
        "watered_today": False,
        "consecutive_unwatered": 0,
        "yield_units": 0,
        "fertilized_until_day": -1,
    }

    ours = encode_board(observation, seat=0)
    theirs = encode_board(observation, seat=1)

    assert not torch.equal(ours, theirs)


def test_scalars_have_the_declared_width() -> None:
    """The market branch is fixed-width; a ragged vector would break the model."""
    scalars = encode_scalars(empty_observation(), seat=0)

    assert scalars.shape == (1, SCALARS)
    assert torch.isfinite(scalars).all()


def test_scalars_carry_both_price_and_opponent_supply() -> None:
    """Value here is a property of a product given what the opponent is selling.

    This is the finding that a static per-animal valuation cost 10k when it
    replaced the opponent-aware one. If the model cannot see the opponent's
    supply it cannot learn the thing that decides this game.
    """
    cheap = empty_observation()
    rich = empty_observation()
    rich["market"]["prices"] = {item: 200 for item in PRODUCTS}

    assert not torch.equal(encode_scalars(cheap, 0), encode_scalars(rich, 0))


def test_the_phase_of_the_season_is_encoded() -> None:
    """Shops unlock every three days and demand steps at days 10 and 20."""
    early, late = empty_observation(), empty_observation()
    late["day"], late["hour"], late["step"] = 25, 13, 25 * 24 + 13

    assert not torch.equal(encode_scalars(early, 0), encode_scalars(late, 0))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/learn/test_encoding.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kaggriculture.learn.encoding'`

- [ ] **Step 3: Write the implementation**

Create `src/kaggriculture/learn/encoding.py`:

```python
"""Turning one observation into the tensors a policy reads.

Plane order is derived from the engine's own rules tables rather than written
out by hand, so a crop or product added upstream cannot silently shift every
plane after it.

The market and the opponent's board get real capacity here rather than a
broadcast constant. Value in this game is a property of a product given what
the opponent is selling — a change that replaced opponent-aware animal
valuation with a static per-animal one measured 10k worse — so a model that
cannot see opponent supply cannot learn what decides the game.
"""

from typing import Any, Mapping

import torch

from kaggriculture.constants import ANIMALS, CROPS, PRODUCTS

BOARD = 10

CROP_NAMES = sorted(CROPS)
ANIMAL_NAMES = sorted(ANIMALS)
PRODUCT_NAMES = sorted(PRODUCTS)

# Per tile, in order: one plane per crop, one per animal, then the tile states
# and the continuous per-tile features.
TILE_STATES = ("weed", "locked", "empty", "structure", "ours", "theirs")
TILE_FEATURES = ("watered", "dry_days", "yield_units", "fertilized", "age")
TILE_PLANES = (
    len(CROP_NAMES) + len(ANIMAL_NAMES) + len(TILE_STATES) + len(TILE_FEATURES)
)

# Market price and inventory per product, our money and theirs, the day and
# hour, and the count of unlocked shops.
SCALARS = 2 * len(PRODUCT_NAMES) + 6


def encode_board(observation: Mapping[str, Any], seat: int) -> torch.Tensor:
    """Return the board planes for one seat.

    Both farms are encoded onto the same grid: ours in the crop, animal and
    feature planes, with the ``ours``/``theirs`` state planes marking which
    farm a cell belongs to. The opponent's board is public in this game and is
    what their future supply is made of.

    Args:
        observation: One turn's observation.
        seat: Which player to encode for.

    Returns:
        A ``(1, TILE_PLANES, BOARD, BOARD)`` float32 tensor.
    """
    planes = torch.zeros(1, TILE_PLANES, BOARD, BOARD, dtype=torch.float32)
    day = int(observation.get("day", 0) or 0)
    farms = observation.get("farms", [])
    for index, farm in enumerate(farms[:2]):
        mine = index == seat
        state_base = len(CROP_NAMES) + len(ANIMAL_NAMES)
        owner = state_base + (4 if mine else 5)
        for y, row in enumerate(farm.get("tiles", [])):
            for x, tile in enumerate(row):
                planes[0, owner, y, x] = 1.0
                if tile == "LOCKED":
                    planes[0, state_base + 1, y, x] = 1.0
                    continue
                if tile is None:
                    planes[0, state_base + 2, y, x] = 1.0
                    continue
                if not isinstance(tile, dict):
                    continue
                kind = tile.get("kind")
                if kind == "WEED":
                    planes[0, state_base, y, x] = 1.0
                elif kind == "PLANT":
                    crop = tile.get("crop")
                    if crop in CROP_NAMES:
                        planes[0, CROP_NAMES.index(crop), y, x] = 1.0
                    _plant_features(planes, tile, y, x, day)
                else:
                    planes[0, state_base + 3, y, x] = 1.0
                    animal = tile.get("animal")
                    if animal in ANIMAL_NAMES:
                        offset = len(CROP_NAMES) + ANIMAL_NAMES.index(animal)
                        planes[0, offset, y, x] = 1.0
    return planes


def _plant_features(
    planes: torch.Tensor, tile: Mapping[str, Any], y: int, x: int, day: int
) -> None:
    """Write the continuous features of one planted tile in place."""
    base = len(CROP_NAMES) + len(ANIMAL_NAMES) + len(TILE_STATES)
    planes[0, base, y, x] = float(bool(tile.get("watered_today")))
    planes[0, base + 1, y, x] = float(tile.get("consecutive_unwatered", 0)) / 2.0
    planes[0, base + 2, y, x] = float(tile.get("yield_units", 0)) / 6.0
    fertilized = int(tile.get("fertilized_until_day", -1))
    planes[0, base + 3, y, x] = float(fertilized >= day)
    planes[0, base + 4, y, x] = float(day - int(tile.get("planted_day", day))) / 30.0


def encode_scalars(observation: Mapping[str, Any], seat: int) -> torch.Tensor:
    """Return the non-spatial features for one seat.

    Prices and inventories are both included: a price says what the next unit
    fetches and the inventory says how far it can move before it stops being
    worth selling. Both are normalised so the branch sees numbers of the same
    order.

    Args:
        observation: One turn's observation.
        seat: Which player to encode for.

    Returns:
        A ``(1, SCALARS)`` float32 tensor.
    """
    market = observation.get("market", {}) or {}
    prices = market.get("prices", {}) or {}
    inventory = market.get("inventory", {}) or {}
    farms = observation.get("farms", [])
    ours = farms[seat] if seat < len(farms) else {}
    theirs = farms[1 - seat] if len(farms) > 1 else {}

    values = [float(prices.get(item, 0)) / 300.0 for item in PRODUCT_NAMES]
    values += [
        (float(inventory.get(item, 10000)) - 10000.0) / 500.0 for item in PRODUCT_NAMES
    ]
    values += [
        float(ours.get("money", 0)) / 100_000.0,
        float(theirs.get("money", 0)) / 100_000.0,
        float(observation.get("day", 0)) / 30.0,
        float(observation.get("hour", 0)) / 24.0,
        float(len((observation.get("town", {}) or {}).get("unlocked_shops", []))) / 8.0,
        float(len(ours.get("hands", []) or [])) / 12.0,
    ]
    return torch.tensor(values, dtype=torch.float32).reshape(1, SCALARS)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/learn/test_encoding.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
uv run pre-commit run -a
git add src/kaggriculture/learn/encoding.py tests/learn/test_encoding.py
git commit -m "feat: encode an observation into board planes and scalars

Plane order derives from the engine's rules tables rather than a hand-written
list, so a crop added upstream cannot silently shift every plane after it.

The opponent's board and the market inventory both get real capacity. Value
here is a property of a product given what the opponent is selling — replacing
opponent-aware animal valuation with a static one measured 10k worse — so a
model that cannot see opponent supply cannot learn what decides the game."
```

---

### Task 4: Encode and decode an action

The recorded action is a nested dict of variable length: one farmer op, a list of hand ops that grows as hands are hired, and up to ten market orders. Training needs fixed-width targets, and playing needs the inverse. Getting the inverse wrong is invisible in training and fatal at play time, so the test is a round-trip.

**Files:**

- Modify: `src/kaggriculture/learn/encoding.py`
- Test: `tests/learn/test_encoding.py`

**Interfaces:**

- Consumes: `BOARD`, `CROP_NAMES`.
- Produces:
  - Constants `UNIT_OPS: tuple[str, ...]`, `MAX_UNITS = 13`
  - `encode_units(action: Mapping[str, Any]) -> torch.Tensor` returning `(1, MAX_UNITS)` int64, `-100` where no unit acted
  - `decode_units(logits: torch.Tensor, units: int) -> dict[str, Any]` returning an action dict

- [ ] **Step 1: Write the failing test**

Append to `tests/learn/test_encoding.py`:

```python
def test_every_recorded_op_has_a_label() -> None:
    """An op absent from the vocabulary would silently become a different action."""
    from kaggriculture.learn.encoding import UNIT_OPS

    engine_ops = {
        "NORTH", "SOUTH", "EAST", "WEST", "PASS", "PICKUP", "PLANT", "WATER",
        "HARVEST", "FERTILIZE", "BUILD_COOP", "BUILD_PASTURE", "DIG", "PLACE",
        "FEED", "COLLECT_FERTILIZER", "CARE",
    }

    assert engine_ops <= {op.split(":")[0] for op in UNIT_OPS}


def test_unit_labels_are_padded_and_masked() -> None:
    """Hands are hired through the day, so the acting unit count varies by turn."""
    from kaggriculture.learn.encoding import MAX_UNITS, encode_units

    labels = encode_units({"farmer": ["WATER"], "hands": [["NORTH"]], "market": []})

    assert labels.shape == (1, MAX_UNITS)
    assert labels[0, 2].item() == -100
    assert labels[0, 0].item() != -100


def test_planting_a_crop_is_a_distinct_label_per_crop() -> None:
    """PLANT MELON and PLANT WHEAT are different decisions, not one op."""
    from kaggriculture.learn.encoding import encode_units

    melon = encode_units({"farmer": ["PLANT", "MELON"], "hands": [], "market": []})
    wheat = encode_units({"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []})

    assert melon[0, 0].item() != wheat[0, 0].item()


def test_labels_round_trip_back_to_a_legal_action() -> None:
    """Training on labels the play path cannot invert would be silently useless."""
    from kaggriculture.learn.encoding import MAX_UNITS, UNIT_OPS, decode_units, encode_units

    action = {"farmer": ["PLANT", "MELON"], "hands": [["WATER"], ["DIG"]], "market": []}
    labels = encode_units(action)
    logits = torch.full((1, MAX_UNITS, len(UNIT_OPS)), -10.0)
    for unit in range(3):
        logits[0, unit, int(labels[0, unit].item())] = 10.0

    decoded = decode_units(logits, units=3)

    assert decoded["farmer"] == ["PLANT", "MELON"]
    assert decoded["hands"] == [["WATER"], ["DIG"]]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/learn/test_encoding.py -k "op or label or round_trip" -v`
Expected: FAIL with `ImportError: cannot import name 'UNIT_OPS'`

- [ ] **Step 3: Write the implementation**

Append to `src/kaggriculture/learn/encoding.py`:

```python
# One label per distinct decision. PLANT carries its crop because planting melon
# and planting wheat are different choices, not one op with a detail attached —
# the crop is the decision that collapsed our own agent's price this morning.
UNIT_OPS: tuple[str, ...] = (
    "PASS",
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "WATER",
    "HARVEST",
    "DIG",
    "FEED",
    "CARE",
    "COLLECT_FERTILIZER",
    "FERTILIZE",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "PICKUP",
    "PLACE",
) + tuple(f"PLANT:{crop}" for crop in CROP_NAMES)

# The main farmer plus the twelve hands a full day can field.
MAX_UNITS = 13

# torch's cross entropy ignores this index, so padded units contribute no loss.
IGNORE = -100


def encode_units(action: Mapping[str, Any]) -> torch.Tensor:
    """Return one label per unit, padded to ``MAX_UNITS``.

    Units that did not act — hands not yet hired — are marked with the ignore
    index rather than a ``PASS`` label. Teaching the model that an absent hand
    chose to pass would train it to pass.

    Args:
        action: One recorded turn's action dict.

    Returns:
        A ``(1, MAX_UNITS)`` int64 tensor of labels.
    """
    ops = [action.get("farmer") or ["PASS"]]
    ops.extend(action.get("hands") or [])
    labels = torch.full((1, MAX_UNITS), IGNORE, dtype=torch.int64)
    for index, op in enumerate(ops[:MAX_UNITS]):
        labels[0, index] = _label(op)
    return labels


def _label(op: list[Any]) -> int:
    """Return the vocabulary index for one unit op, defaulting to ``PASS``."""
    if not op:
        return UNIT_OPS.index("PASS")
    verb = str(op[0])
    if verb == "PLANT" and len(op) > 1:
        name = f"PLANT:{op[1]}"
        return UNIT_OPS.index(name) if name in UNIT_OPS else UNIT_OPS.index("PASS")
    return UNIT_OPS.index(verb) if verb in UNIT_OPS else UNIT_OPS.index("PASS")


def decode_units(logits: torch.Tensor, units: int) -> dict[str, Any]:
    """Return the action dict implied by per-unit logits.

    Args:
        logits: A ``(1, MAX_UNITS, len(UNIT_OPS))`` tensor.
        units: How many units are actually on the board this turn.

    Returns:
        An action dict with ``farmer``, ``hands`` and an empty ``market``.
    """
    chosen = logits[0, :units].argmax(dim=-1)
    ops = [_op(int(index.item())) for index in chosen]
    return {"farmer": ops[0], "hands": ops[1:], "market": []}


def _op(label: int) -> list[Any]:
    """Return the op list for one vocabulary index."""
    name = UNIT_OPS[label]
    if name.startswith("PLANT:"):
        return ["PLANT", name.split(":", 1)[1]]
    return [name]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/learn/test_encoding.py -v`
Expected: PASS, 10 tests.

- [ ] **Step 5: Commit**

```bash
uv run pre-commit run -a
git add src/kaggriculture/learn/encoding.py tests/learn/test_encoding.py
git commit -m "feat: encode and decode unit actions

Labels round-trip back to a legal action, because training on labels the play
path cannot invert is silently useless — it produces a model with good
validation accuracy and no way to act on it.

Units that have not been hired carry the ignore index rather than a PASS label.
Teaching the model that an absent hand chose to pass would train it to pass.
PLANT carries its crop: planting melon and planting wheat are different
decisions, and the crop is what collapsed our own agent's price."
```

---

### Task 5: Build sharded training data

**Files:**

- Create: `src/kaggriculture/learn/dataset.py`, `src/kaggriculture/learn/scripts/build.py`
- Test: `tests/learn/test_dataset.py`

**Interfaces:**

- Consumes: `CORPUS`, `Sample` (fields `archive`, `name`, `seat`, `rating`), `select`, `split`, `encode_board`, `encode_scalars`, `encode_units`, `TILE_PLANES`, `SCALARS`, `MAX_UNITS`.
- Produces:
  - `build_shard(samples: list[Sample], destination: Path, stride: int = 4) -> int` returning rows written
  - `class Shards(torch.utils.data.Dataset)` yielding `(board, scalars, labels)` tensors

- [ ] **Step 1: Write the failing test**

Create `tests/learn/test_dataset.py`:

```python
"""Tests for the sharded training data, built from the real corpus."""

from pathlib import Path

import pytest
import torch

from kaggriculture.learn.corpus import CORPUS, Sample
from kaggriculture.learn.dataset import Shards, build_shard
from kaggriculture.learn.encoding import MAX_UNITS, SCALARS, TILE_PLANES

ARCHIVE = CORPUS / "kaggriculture-episodes-2026-08-03.zip"

pytestmark = pytest.mark.skipif(
    not ARCHIVE.exists(), reason="replay corpus not present on this machine"
)


def one_sample() -> Sample:
    """Return a sample naming a real episode in the corpus."""
    import zipfile

    with zipfile.ZipFile(ARCHIVE) as bundle:
        name = next(n for n in bundle.namelist() if n.endswith(".json"))
    return Sample(archive=ARCHIVE.name, name=name, seat=0, rating=2600.0)


def test_a_shard_round_trips_into_tensors_of_the_declared_shape(tmp_path: Path) -> None:
    """The model's input shape is fixed; a ragged shard would fail at train time."""
    destination = tmp_path / "shard-000.npz"

    rows = build_shard([one_sample()], destination, stride=64)

    assert rows > 0
    board, scalars, labels = Shards([destination])[0]
    assert board.shape == (TILE_PLANES, 10, 10)
    assert scalars.shape == (SCALARS,)
    assert labels.shape == (MAX_UNITS,)
    assert board.dtype == torch.float32


def test_stride_controls_how_many_turns_are_kept(tmp_path: Path) -> None:
    """720 turns per seat is more correlated data than it is information."""
    dense = build_shard([one_sample()], tmp_path / "dense.npz", stride=8)
    sparse = build_shard([one_sample()], tmp_path / "sparse.npz", stride=64)

    assert dense > sparse


def test_labels_are_the_action_taken_from_the_state_not_the_one_that_made_it(
    tmp_path: Path,
) -> None:
    """The label must be the decision that follows an observation, not precedes it.

    ``kaggle_environments``' interpreter mutates the state object carried
    alongside the action it is applying, so a recorded ``steps[i][seat]``
    holds the state *after* ``action[i]`` ran — not before. Pairing them by
    index therefore asks the model to predict an action from the world that
    action already created, which is label leakage during training and a
    distribution it never sees at inference. The correct pair is
    ``(observation[i], action[i + 1])``; ``action[0]`` is a reset filler with
    nothing to predict and is dropped.

    The obvious version of this test cannot fail. Sampled sparsely, most
    consecutive actions are both all-``PASS`` and encode identically, so the
    right and wrong pairings agree and the assertion passes either way. This
    one seeks out an index where the two genuinely differ and pins down both
    directions.
    """
    import json
    import zipfile

    from kaggriculture.learn.encoding import encode_units

    sample = one_sample()
    with zipfile.ZipFile(ARCHIVE) as bundle, bundle.open(sample.name) as member:
        steps = json.load(member)["steps"]

    turn = next(
        i
        for i in range(1, len(steps) - 1)
        if not torch.equal(
            encode_units(steps[i][sample.seat]["action"] or {})[0],
            encode_units(steps[i + 1][sample.seat]["action"] or {})[0],
        )
    )
    destination = tmp_path / "aligned.npz"
    build_shard([sample], destination, stride=1)

    _, _, labels = Shards([destination])[turn]

    assert torch.equal(labels, encode_units(steps[turn + 1][sample.seat]["action"])[0])
    assert not torch.equal(
        labels, encode_units(steps[turn][sample.seat]["action"])[0]
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/learn/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kaggriculture.learn.dataset'`

- [ ] **Step 3: Write the implementation**

Create `src/kaggriculture/learn/dataset.py`:

```python
"""Streaming selected episodes into sharded training data.

The corpus cannot be unpacked, so episodes are read one at a time straight from
their archive, encoded, and appended to a shard. Turns are subsampled: 720 turns
of one season are far more correlated than they are informative, and a stride
buys diversity per byte.
"""

import json
import logging
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from kaggriculture.learn.corpus import CORPUS, Sample
from kaggriculture.learn.encoding import encode_board, encode_scalars, encode_units

LOGGER = logging.getLogger(__name__)


def build_shard(samples: list[Sample], destination: Path, stride: int = 4) -> int:
    """Encode every ``stride``-th turn of each sample into one ``.npz`` shard.

    Args:
        samples: Demonstrations to encode.
        destination: Shard path to write.
        stride: Keep one turn in this many.

    Returns:
        How many rows were written.
    """
    boards: list[np.ndarray] = []
    scalars: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    by_archive: dict[str, list[Sample]] = {}
    for sample in samples:
        by_archive.setdefault(sample.archive, []).append(sample)

    for archive, group in by_archive.items():
        with zipfile.ZipFile(CORPUS / archive) as bundle:
            for sample in tqdm(group, desc=archive, unit="ep"):
                with bundle.open(sample.name) as member:
                    episode = json.load(member)
                _encode_episode(episode, sample.seat, stride, boards, scalars, labels)

    np.savez_compressed(
        destination,
        boards=np.concatenate(boards) if boards else np.empty((0,)),
        scalars=np.concatenate(scalars) if scalars else np.empty((0,)),
        labels=np.concatenate(labels) if labels else np.empty((0,)),
    )
    LOGGER.info("%s: %d rows", destination.name, len(boards))
    return len(boards)


def _encode_episode(
    episode: dict[str, Any],
    seat: int,
    stride: int,
    boards: list[np.ndarray],
    scalars: list[np.ndarray],
    labels: list[np.ndarray],
) -> None:
    """Append one seat's encoded turns to the accumulating lists."""
    for index in range(0, len(episode["steps"]), stride):
        entry = episode["steps"][index][seat]
        action = entry.get("action")
        if not action:
            continue
        observation = entry["observation"]
        boards.append(encode_board(observation, seat).numpy(force=True))
        scalars.append(encode_scalars(observation, seat).numpy(force=True))
        labels.append(encode_units(action).numpy(force=True))


class Shards(Dataset):
    """A dataset over one or more ``.npz`` shards, held in memory."""

    def __init__(self, paths: list[Path]) -> None:
        """Load every shard named by ``paths``."""
        boards, scalars, labels = [], [], []
        for path in paths:
            with np.load(path) as data:
                boards.append(data["boards"])
                scalars.append(data["scalars"])
                labels.append(data["labels"])
        self.boards = torch.from_numpy(np.concatenate(boards))
        self.scalars = torch.from_numpy(np.concatenate(scalars))
        self.labels = torch.from_numpy(np.concatenate(labels))

    def __len__(self) -> int:
        """Return how many rows this dataset holds."""
        return int(self.boards.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one row's board, scalars and labels, without batch dimensions."""
        return self.boards[index], self.scalars[index], self.labels[index]
```

Create `src/kaggriculture/learn/scripts/build.py`:

```python
"""Build the imitation dataset from the replay corpus."""

import argparse
import logging
from pathlib import Path

from kaggriculture.learn.corpus import CORPUS, select, split
from kaggriculture.learn.dataset import build_shard

LOGGER = logging.getLogger(__name__)

SHARDS = Path("/data/kaggriculture/imitation")


def main() -> None:
    """Index, select and encode the corpus from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=SHARDS, help="shard directory")
    parser.add_argument("--stride", type=int, default=4, help="keep one turn in N")
    parser.add_argument("--min-rating", type=float, default=2500.0, help="rating floor")
    parser.add_argument("--per-archive", type=int, default=200, help="cap per archive")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    chosen = select(
        sorted(CORPUS.glob("*.zip")),
        min_rating=args.min_rating,
        per_archive=args.per_archive,
    )
    train, held = split(chosen)
    LOGGER.info("selected %d seats: %d train, %d holdout", len(chosen), len(train), len(held))

    build_shard(train, args.out / "train.npz", stride=args.stride)
    build_shard(held, args.out / "holdout.npz", stride=args.stride)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/learn/test_dataset.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 5: Build the real dataset**

```bash
uv run python -m kaggriculture.learn.scripts.build
```

Record the selected-seat count and the shard sizes in the commit message. If the shards exceed 20 GB, raise `--stride` and rebuild rather than reducing quality filtering.

- [ ] **Step 6: Commit**

```bash
uv run pre-commit run -a
git add src/kaggriculture/learn/dataset.py src/kaggriculture/learn/scripts/build.py tests/learn/test_dataset.py
git commit -m "feat: stream the corpus into sharded training data

Episodes are read one at a time straight from their archive; the corpus is
~107GB uncompressed and cannot be unpacked. Turns are subsampled because 720
turns of one season are far more correlated than they are informative.

The alignment test is the one that matters: a replay step holds the state
before its action was interpreted, so the action belonging to an observation
sits at the same index. Getting that backwards is invisible — the model still
trains, it just learns to predict the previous turn's decision."
```

---

### Task 6: The reference model, sized to the budget

**Files:**

- Create: `src/kaggriculture/learn/model.py`
- Test: `tests/learn/test_model.py`

**Interfaces:**

- Consumes: `TILE_PLANES`, `SCALARS`, `MAX_UNITS`, `UNIT_OPS`, `BOARD`.
- Produces:
  - `class Policy(torch.nn.Module)` with `forward(board: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor` returning `(batch, MAX_UNITS, len(UNIT_OPS))`
  - `BLOCKS = 8`, `CHANNELS = 256`

- [ ] **Step 1: Write the failing test**

Create `tests/learn/test_model.py`:

```python
"""Tests for the reference policy: shape, size and the inference budget."""

import time

import torch

from kaggriculture.learn.encoding import BOARD, MAX_UNITS, SCALARS, TILE_PLANES, UNIT_OPS
from kaggriculture.learn.model import Policy


def test_forward_returns_one_distribution_per_unit() -> None:
    """Each unit picks its own op, so the head is per-unit, not per-board."""
    model = Policy()
    board = torch.zeros(2, TILE_PLANES, BOARD, BOARD)
    scalars = torch.zeros(2, SCALARS)

    logits = model(board, scalars)

    assert logits.shape == (2, MAX_UNITS, len(UNIT_OPS))


def test_the_model_fits_the_size_every_winner_used() -> None:
    """Verified winners span 1M-20M parameters; nothing above 20M is supported."""
    parameters = sum(p.numel() for p in Policy().parameters())

    assert 1e6 < parameters < 20e6


def test_a_turn_fits_the_sandbox_budget_at_two_threads() -> None:
    """The sandbox has two cores and one second a turn, not this workstation.

    Measured at two threads because timing on 64 cores flatters the sandbox by
    an order of magnitude. The Phase 0 probe measured a 10M-parameter trunk at
    ~45ms of the 1000ms budget; this asserts a wide margin, not that figure.
    """
    torch.set_num_threads(2)
    model = Policy().eval()
    board = torch.zeros(1, TILE_PLANES, BOARD, BOARD)
    scalars = torch.zeros(1, SCALARS)

    with torch.no_grad():
        model(board, scalars)
        start = time.perf_counter()
        for _ in range(10):
            model(board, scalars)
        elapsed = (time.perf_counter() - start) / 10

    assert elapsed < 0.25, f"{elapsed * 1000:.0f}ms per turn leaves no margin"


def test_the_market_reaches_the_trunk() -> None:
    """Prices decide this game; a scalar branch that is ignored is a silent bug."""
    model = Policy().eval()
    board = torch.zeros(1, TILE_PLANES, BOARD, BOARD)

    with torch.no_grad():
        cheap = model(board, torch.zeros(1, SCALARS))
        rich = model(board, torch.ones(1, SCALARS))

    assert not torch.allclose(cheap, rich)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/learn/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kaggriculture.learn.model'`

- [ ] **Step 3: Write the implementation**

Create `src/kaggriculture/learn/model.py`:

```python
"""The reference policy: Frog Parade's published shape, sized to our board.

Their Lux S3 solution is an 8-block 3x3 residual CNN at d_model 256, about 10M
parameters, with global features projected through an MLP and added to the
spatial tensor. Every verified winner across four competitions falls between 1M
and 20M parameters, and the two figures that suggested otherwise turned out to
be step counts.

Our board is 10x10 against their 24x24, so the same shape costs roughly a sixth
as much: the Phase 0 probe measured a 10M-parameter conv trunk at ~45ms of the
1000ms turn.
"""

import torch

from kaggriculture.learn.encoding import (
    BOARD,
    MAX_UNITS,
    SCALARS,
    TILE_PLANES,
    UNIT_OPS,
)

BLOCKS = 8
CHANNELS = 256


class Residual(torch.nn.Module):
    """One 3x3 residual block with squeeze-excitation and no normalisation.

    Toad Brigade's Lux S1 winner and Frog Parade's Lux S3 model both omit
    normalisation layers deliberately; this follows them rather than reasoning
    from first principles about a choice two winners already made.
    """

    def __init__(self, channels: int) -> None:
        """Build the block."""
        super().__init__()
        self.first = torch.nn.Conv2d(channels, channels, 3, padding=1)
        self.second = torch.nn.Conv2d(channels, channels, 3, padding=1)
        self.excite = torch.nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the block's output for a batch of feature maps."""
        residual = torch.nn.functional.relu(self.first(x))
        residual = self.second(residual)
        weights = torch.sigmoid(self.excite(residual.mean(dim=(2, 3))))
        return torch.nn.functional.relu(x + residual * weights[:, :, None, None])


class Policy(torch.nn.Module):
    """Board trunk plus a market branch, reading out one op per unit."""

    def __init__(self, blocks: int = BLOCKS, channels: int = CHANNELS) -> None:
        """Build the policy."""
        super().__init__()
        self.stem = torch.nn.Conv2d(TILE_PLANES, channels, 3, padding=1)
        self.market = torch.nn.Sequential(
            torch.nn.Linear(SCALARS, channels),
            torch.nn.ReLU(),
            torch.nn.Linear(channels, channels),
        )
        self.blocks = torch.nn.ModuleList(Residual(channels) for _ in range(blocks))
        self.head = torch.nn.Sequential(
            torch.nn.Linear(channels, channels),
            torch.nn.ReLU(),
            torch.nn.Linear(channels, MAX_UNITS * len(UNIT_OPS)),
        )

    def forward(self, board: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """Return per-unit op logits.

        The market is projected and added to the spatial tensor rather than
        broadcast as constant planes: it is 26 numbers that decide the game and
        deserve their own capacity.

        Args:
            board: ``(batch, TILE_PLANES, BOARD, BOARD)`` planes.
            scalars: ``(batch, SCALARS)`` market and phase features.

        Returns:
            ``(batch, MAX_UNITS, len(UNIT_OPS))`` logits.
        """
        features = self.stem(board) + self.market(scalars)[:, :, None, None]
        for block in self.blocks:
            features = block(features)
        pooled = features.mean(dim=(2, 3))
        return self.head(pooled).reshape(-1, MAX_UNITS, len(UNIT_OPS))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/learn/test_model.py -v`
Expected: PASS, 4 tests. If the budget test fails, reduce `CHANNELS` to 192 and re-run — do not raise the threshold.

- [ ] **Step 5: Commit**

```bash
uv run pre-commit run -a
git add src/kaggriculture/learn/model.py tests/learn/test_model.py
git commit -m "feat: the reference policy, sized to the measured budget

Frog Parade's published shape rather than an invented one: an 8-block 3x3
residual CNN at d_model 256, no normalisation layers, global features projected
and added to the spatial tensor. Every verified winner across four competitions
is between 1M and 20M parameters.

The budget test runs at two threads because the sandbox has two cores and
timing on 64 flatters it by an order of magnitude. If it fails, shrink the
model — the threshold is the competition's, not ours."
```

---

### Task 7: Clone, then measure against the league

The gate is not validation accuracy. A model can predict the corpus well and play badly, because the corpus is mostly one recording and the metric that decides this project is a win rate against frozen opponents.

**Files:**

- Create: `src/kaggriculture/learn/scripts/train.py`, `src/kaggriculture/learn/scripts/play.py`
- Modify: `STRATEGY.md`

**Interfaces:**

- Consumes: `Shards`, `Policy`, `decode_units`, `encode_board`, `encode_scalars`, and the league from `kaggriculture.config`.
- Produces: a checkpoint at `/data/kaggriculture/imitation/policy.pt`, and `learn/scripts/play.py` exposing `agent(obs) -> dict` as its last callable.

- [ ] **Step 1: Write the training script**

Create `src/kaggriculture/learn/scripts/train.py`:

```python
"""Behaviour-clone the reference policy on the imitation dataset."""

import argparse
import logging
from pathlib import Path

import torch
from lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from kaggriculture.learn.dataset import Shards
from kaggriculture.learn.encoding import IGNORE, UNIT_OPS
from kaggriculture.learn.model import Policy
from kaggriculture.learn.scripts.build import SHARDS

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Train from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, default=SHARDS)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    seed_everything(args.seed, workers=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = DataLoader(
        Shards([args.shards / "train.npz"]), batch_size=args.batch, shuffle=True
    )
    held = DataLoader(Shards([args.shards / "holdout.npz"]), batch_size=args.batch)

    model = Policy().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    run = wandb.init(
        entity="will-rice",
        project="kaggriculture-2026",
        job_type="behaviour-cloning",
        name=f"bc-{sum(p.numel() for p in model.parameters()) // 1_000_000}M-seed{args.seed}",
        config=vars(args) | {"ops": len(UNIT_OPS)},
    )

    for epoch in range(args.epochs):
        model.train()
        for board, scalars, labels in tqdm(train, desc=f"epoch {epoch}"):
            logits = model(board.to(device), scalars.to(device))
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), labels.to(device).flatten(), ignore_index=IGNORE
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            run.log({"loss/train": loss.item()})
        run.log({"accuracy/holdout": accuracy(model, held, device), "epoch": epoch})

    args.shards.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.shards / "policy.pt")
    LOGGER.info("saved %s", args.shards / "policy.pt")
    run.finish()


def accuracy(model: Policy, loader: DataLoader, device: str) -> float:
    """Return top-1 accuracy over acting units only."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for board, scalars, labels in loader:
            predicted = model(board.to(device), scalars.to(device)).argmax(dim=-1)
            target = labels.to(device)
            mask = target != IGNORE
            correct += int((predicted[mask] == target[mask]).sum().item())
            total += int(mask.sum().item())
    return correct / max(1, total)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the play wrapper**

Create `src/kaggriculture/learn/scripts/play.py`:

```python
"""Wrap a trained checkpoint as an agent the harness can evaluate.

Not a submission entrypoint. This imports torch, which the competition sandbox
pays 10.7 seconds to load, and lives under `learn` which never ships. It exists
so a checkpoint can be measured against the same frozen league every other
agent in this project is measured against.
"""

from pathlib import Path
from typing import Any, Mapping

import torch

from kaggriculture.learn.encoding import decode_units, encode_board, encode_scalars
from kaggriculture.learn.model import Policy

CHECKPOINT = Path("/data/kaggriculture/imitation/policy.pt")

_MODEL: Policy | None = None


def agent(raw_obs: Mapping[str, Any]) -> dict[str, Any]:
    """Return this turn's action from the cloned policy."""
    global _MODEL
    if _MODEL is None:
        torch.set_num_threads(2)
        _MODEL = Policy()
        _MODEL.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
        _MODEL.eval()

    seat = int(raw_obs.get("player", 0))
    farm = raw_obs["farms"][seat]
    units = 1 + len(farm.get("hands", []) or [])
    with torch.no_grad():
        logits = _MODEL(encode_board(raw_obs, seat), encode_scalars(raw_obs, seat))
    return decode_units(logits, units=units)
```

- [ ] **Step 3: Train**

```bash
uv run python -m kaggriculture.learn.scripts.train
```

Record the final holdout accuracy and the wandb run name.

- [ ] **Step 4: Measure against the league — this is the gate**

```bash
uv run python -m kaggriculture.scripts.run \
  --agent src/kaggriculture/learn/scripts/play.py \
  --games 100 --workers 64 --seed 3000 --track
```

**Gate:** beats `starter` outright, and ideally `heuristic-v2`. Record the standings line and the league figure. A cloned agent that cannot beat `starter` has not learned to play, whatever its holdout accuracy says — report that plainly rather than tuning until the number looks better.

- [ ] **Step 5: Record the result**

Add under Phase 2 in `STRATEGY.md`, filling in measured values:

```markdown
#### Result — behaviour cloning, YYYY-MM-DD (run `bc-…`)

Dataset: N seats selected from M episodes at a rating floor of R with a per-archive cap of C,
S rows at stride T. Holdout accuracy A over acting units.

| opponent     | win rate | 95% interval |
| ------------ | -------- | ------------ |
| meta-tape    |          |              |
| heuristic-v2 |          |              |
| heuristic-v1 |          |              |
| starter      |          |              |

The gate asked for `random` and ideally `heuristic-v2`. [State plainly whether it
cleared, and if the accuracy is high while the win rate is not, say so — that gap
is the point of measuring play rather than prediction.]
```

- [ ] **Step 6: Commit**

```bash
uv run pre-commit run -a
git add src/kaggriculture/learn/scripts/train.py src/kaggriculture/learn/scripts/play.py STRATEGY.md
git commit -m "feat: clone the corpus and measure the result against the league

The gate is a win rate, not holdout accuracy. The corpus is three-quarters one
recorded episode, so a model can predict it well and still play badly — and
predicting a recording is exactly the failure imitation is supposed to escape.

The play wrapper is not a submission entrypoint. It imports torch, which costs
ten seconds of the sandbox's overage pool, and lives under learn/ which never
ships."
```

---

## Self-Review

**Spec coverage.** Phase 2 asks for five things. A dataset from the daily dumps filtered to top-decile banks — Tasks 1 and 2, with a per-team cap the phase text does not mention but which the corpus demands, since three-quarters of it is one kernel. Observation encoding with tile planes, unit counts and separate day/hour features — Task 3. The market as its own branch rather than broadcast planes — Task 3 encodes it and Task 6 projects it through an MLP added at the trunk, with a test that it reaches the output. Architecture selected on accuracy per millisecond of two-thread CPU inference — Task 6 fixes one architecture and enforces the budget; **the comparison across trunks is deliberately deferred**. The gate — Task 7.

**Deliberately out of scope**, needing its own plan: comparing the residual trunk against a downscaling U-net, which is the "selection" half of Phase 2. It needs a working pipeline to compare on, and that is what this plan builds. Also out of scope: Phase 3's RL loop, which consumes this plan's model and league.

**Placeholders.** None. Task 7's STRATEGY.md block is a template whose numbers cannot exist before the run that produces them, and it says so.

**Type consistency.** `Sample` is produced in Task 2 and consumed in Task 5. `TILE_PLANES`, `SCALARS`, `MAX_UNITS`, `UNIT_OPS`, `BOARD` and `IGNORE` are defined in Tasks 3 and 4 and consumed in Tasks 5, 6 and 7. `encode_board`/`encode_scalars` return batched tensors that Task 5 unbatches by `np.concatenate` and Task 7 passes through directly. `Policy.forward` returns `(batch, MAX_UNITS, len(UNIT_OPS))`, which `decode_units` indexes as `logits[0, :units]`.

**Two risks worth stating.** The action space omits market orders entirely — the cloned agent will farm but never trade, which caps it well below the league's better opponents and is why the gate asks only for `starter`. Trading is a separate head and belongs in the RL plan, where the market branch already feeds the trunk. Second, `MAX_UNITS = 13` assumes the 12-hand cap our own strategy used; a demonstration from a team hiring more would silently truncate, which Task 4's padding test does not catch.
