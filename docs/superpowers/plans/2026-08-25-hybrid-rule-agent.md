# Hybrid Rule Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and search a deterministic hybrid opening-and-recovery agent that shares its feature/action core with RL and must beat both Boatlee v14 and the locally measured public frontier before it can be proposed for submission.

**Architecture:** A standard-library-only canonical encoder and action codec sit below both the hybrid runtime and thin Torch compatibility adapters. The runtime combines typed opening targets, reactive unit jobs, and live market scoring; Pydantic validates offline configurations, which are frozen into dependency-free runtime dataclasses. Existing reference-engine arena code is extended to score exact public versions and in-memory hybrid configurations, then a CPU-only evolutionary search and fixed paired holdout gate produce evidence without changing `main.py`.

**Tech Stack:** Python 3.11, Pydantic 2, PyTorch 2.13 compatibility adapters, `kaggle-environments` 1.32.7 reference engine, pytest, Ruff, ty, standard-library multiprocessing and statistics.

**Spec:** `docs/superpowers/specs/2026-08-25-hybrid-rule-agent-design.md`

## Global Constraints

- The active single-GPU Toad run must not be stopped, reconfigured, or assigned additional GPU work.
- Offline frontier evaluation and search are CPU-only, run with at most 16 workers and process niceness 10 on this 64-thread host.
- `main.py` remains on `kaggriculture.boatlee_v14_policy.agent`; changing the served agent or submitting to Kaggle requires separate explicit approval.
- The packaged turn path imports neither Torch nor Pydantic and performs no file IO, network IO, random exploration, or inference-time rollout search.
- The canonical feature/action core is the sole raw-observation interpreter for the reference RL path and hybrid runtime; the CUDA-native backend is the only permitted duplicate execution backend and stays under differential tests.
- Existing tensor shapes, feature order, normalization, masks, action vocabulary, and checkpoint compatibility remain bit-identical.
- Every categorical parameter and every bounded-integer parameter is represented by one logit group and decoded with `argmax`; only genuinely continuous parameters map from `[0, 1]`.
- Search uses 8 screening seeds, 32 development seeds, and 128 untouched promotion seeds, all disjoint and seat-swapped.
- Promotion uses paired 95% bootstrap intervals and cannot be waived by leaderboard ratings, bank totals, or a favorable aggregate that hides one failed matchup.
- Search code and public opponent artifacts remain outside the submission archive.

---

## File Structure

### Submission-safe runtime

- `src/kaggriculture/action_codec.py`: action vocabulary, quantity buckets, selected-index decoder, and safe PASS action; no Torch or Pydantic.
- `src/kaggriculture/features.py`: canonical immutable observation features, named accessors, and Python legality masks; no Torch or Pydantic.
- `src/kaggriculture/hybrid/runtime.py`: frozen standard-library runtime configuration and round-trip serialization.
- `src/kaggriculture/hybrid/opening.py`: phase selection and target deficits.
- `src/kaggriculture/hybrid/jobs.py`: per-unit guarded job scoring and deterministic assignment.
- `src/kaggriculture/hybrid/market.py`: live market scoring, reserves, and liquidation.
- `src/kaggriculture/hybrid/policy.py`: integrated controller and Kaggle `agent` callable.

### Training compatibility

- `src/kaggriculture/learn/encoding.py`: preserve public tensor interfaces as adapters over `action_codec` and `features`.
- `src/kaggriculture/learn/mask.py`: preserve tensor-mask interfaces as adapters over canonical Python masks.
- `src/kaggriculture/learn/play.py` and `src/kaggriculture/learn/rollout.py`: build the canonical bundle once per observation rather than reparsing it for every tensor.
- `src/kaggriculture/sim/observe.py` and `src/kaggriculture/sim/legality.py`: consume shared schema constants and retain differential parity.

### Offline configuration and search

- `src/kaggriculture/hybrid/config.py`: strict Pydantic authoring configuration and conversion to runtime dataclasses; not imported by `hybrid.policy`.
- `src/kaggriculture/search/genome.py`: discrete logit groups, continuous genes, and deterministic `HybridConfig` decoding.
- `src/kaggriculture/search/frontier.py`: exact-version manifest verification and current-engine ranking.
- `src/kaggriculture/search/frontier_manifest.json`: exact public versions, historical scores, source hashes, and extraction provenance.
- `src/kaggriculture/search/fitness.py`: strength-tier weights and the fixed `0.70 mean + 0.30 worst` objective.
- `src/kaggriculture/search/evolution.py`: deterministic CPU evolutionary loop with progressive evaluation.
- `src/kaggriculture/search/promotion.py`: paired bootstrap intervals and all promotion rulings.
- `src/kaggriculture/search/scripts/frontier_round_robin.py`: materialize the measured frontier report.
- `src/kaggriculture/search/scripts/hybrid_search.py`: run screening/development search and write resumable artifacts.
- `src/kaggriculture/search/scripts/hybrid_holdout.py`: run the untouched promotion gate.
- `src/kaggriculture/search/scripts/freeze_hybrid.py`: convert an approved Pydantic candidate to a dependency-free runtime module; tests write only to temporary paths.

---

### Task 1: Exact Public-Frontier Manifest and Local Ranking

**Files:**

- Create: `src/kaggriculture/search/frontier.py`
- Create: `src/kaggriculture/search/frontier_manifest.json`
- Create: `src/kaggriculture/search/scripts/frontier_round_robin.py`
- Modify: `src/kaggriculture/search/arena.py:29-205`
- Test: `tests/search/test_frontier.py`
- Test: `tests/search/test_arena.py`

**Interfaces:**

- Consumes: existing `arena.Opponent`, `arena.outcomes`, and exact public agent files under `/data/kaggriculture/search/public-frontier`.
- Produces: `FrontierArtifact`, `FrontierManifest`, `VerifiedFrontier`, `verify_frontier(manifest_path: Path, artifact_root: Path) -> VerifiedFrontier`, and `rank_frontier(frontier: VerifiedFrontier, seeds: Sequence[int], workers: int) -> FrontierReport`.

- [ ] **Step 1: Write manifest-verification and ranking tests**

```python
def test_verify_frontier_rejects_changed_source_bytes(tmp_path: Path) -> None:
    source = tmp_path / "kaito-v27.py"
    source.write_text("def agent(obs, config=None): return {}\n")
    manifest = FrontierManifest(
        engine="1.32.7",
        artifacts=(
            FrontierArtifact(
                name="kaito_v27",
                source_url="https://www.kaggle.com/code/kaitofukami/25-27-strict-future-v27-midgame-meta-reset",
                notebook_version=4,
                historical_score=3090.1,
                provenance="Kaggle notebook version 4, extracted last callable",
                relative_path=Path("kaito-v27.py"),
                sha256="0" * 64,
                archive_sha256="1" * 64,
            ),
        ),
    )
    path = tmp_path / "manifest.json"
    path.write_text(manifest.model_dump_json())

    with pytest.raises(FrontierIntegrityError, match="kaito_v27.*sha256"):
        verify_frontier(path, tmp_path)


def test_rank_frontier_uses_seat_swapped_field_win_points(
    monkeypatch: pytest.MonkeyPatch, verified_frontier: VerifiedFrontier
) -> None:
    monkeypatch.setattr(
        frontier.arena,
        "outcomes",
        lambda candidate, league, seeds, workers: {
            "a.py": [1.0, 1.0, 0.5, 0.5],
            "b.py": [0.0, 0.0, 0.5, 0.5],
        }[Path(str(candidate)).name],
    )

    report = rank_frontier(verified_frontier, seeds=(11, 12), workers=1)

    assert report.rows[0].name == "a"
    assert report.frontier_name == "a"
    assert report.rows[0].games == 4
```

- [ ] **Step 2: Run the RED tests**

Run: `.venv/bin/python -m pytest tests/search/test_frontier.py tests/search/test_arena.py -q`

Expected: FAIL during collection because `kaggriculture.search.frontier` and its public types do not exist.

- [ ] **Step 3: Generalize the arena candidate type and implement strict manifest loading**

```python
# arena.py
def outcomes(
    candidate: Opponent,
    league: Mapping[str, Opponent],
    seeds: Sequence[int],
    workers: int | None = None,
) -> list[float]:
    scores: list[float] = []
    for opponent in league.values():
        first = play(candidate, opponent, seeds, workers)
        second = play(opponent, candidate, seeds, workers)
        for (ours_first, theirs_first), (theirs_second, ours_second) in zip(
            first, second, strict=True
        ):
            scores.append(_win(ours_first, theirs_first))
            scores.append(_win(ours_second, theirs_second))
    return scores


# frontier.py
class FrontierArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str = Field(pattern=r"^[a-z0-9_]+$")
    source_url: HttpUrl | None = None
    notebook_version: PositiveInt | None = None
    historical_score: float | None
    provenance: str = Field(min_length=1)
    relative_path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    archive_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def complete_notebook_identity(self) -> Self:
        if (self.source_url is None) != (self.notebook_version is None):
            raise ValueError("source_url and notebook_version must be set together")
        if self.source_url is not None and self.archive_sha256 is None:
            raise ValueError("Kaggle artifacts require archive_sha256")
        return self


class FrontierManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    engine: str
    artifacts: tuple[FrontierArtifact, ...]


@dataclass(frozen=True)
class VerifiedFrontier:
    engine: str
    opponents: Mapping[str, str]
    artifacts: tuple[FrontierArtifact, ...]


def verify_frontier(manifest_path: Path, artifact_root: Path) -> VerifiedFrontier:
    manifest = FrontierManifest.model_validate_json(manifest_path.read_text())
    resolved: dict[str, str] = {}
    for artifact in manifest.artifacts:
        source = (artifact_root / artifact.relative_path).resolve()
        if artifact_root.resolve() not in source.parents:
            raise FrontierIntegrityError(f"{artifact.name}: path escapes artifact root")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != artifact.sha256:
            raise FrontierIntegrityError(
                f"{artifact.name}: sha256 {digest} != {artifact.sha256}"
            )
        resolved[artifact.name] = str(source)
    return VerifiedFrontier(manifest.engine, resolved, manifest.artifacts)
```

Implement `rank_frontier` as a true all-pairs, seat-swapped round robin using 64 seeds. Rank by average field win points, then worst-matchup win points, then artifact name. Record every pair rather than discarding the non-transitive matrix.

- [ ] **Step 4: Archive and validate the exact initial versions**

Use the authenticated Kaggle archive workflow, placing sources under `/data/kaggriculture/search/public-frontier`:

```bash
/home/will/projects/kaggriculture-2026/.venv/bin/python /home/will/.agents/skills/nvidia-kaggle-skill/scripts/kernel_archive.py kaitofukami/25-27-strict-future-v27-midgame-meta-reset /data/kaggriculture/search/public-frontier --version 4
/home/will/projects/kaggriculture-2026/.venv/bin/python /home/will/.agents/skills/nvidia-kaggle-skill/scripts/kernel_archive.py kaitofukami/40-40-early-floor-39-46-top-10-v48-fast-routes /data/kaggriculture/search/public-frontier --version 2
/home/will/projects/kaggriculture-2026/.venv/bin/python /home/will/.agents/skills/nvidia-kaggle-skill/scripts/kernel_archive.py kaitofukami/40-40-early-floor-39-46-top-10-v48-fast-routes /data/kaggriculture/search/public-frontier --version 8
/home/will/projects/kaggriculture-2026/.venv/bin/python /home/will/.agents/skills/nvidia-kaggle-skill/scripts/kernel_archive.py boatlee/84-84-base-public-holdout-v14-clone-preemption /data/kaggriculture/search/public-frontier --version 4
/home/will/projects/kaggriculture-2026/.venv/bin/python /home/will/.agents/skills/nvidia-kaggle-skill/scripts/kernel_archive.py boatlee/v16-rc5-high-score-8c-4s-premium-market-lead /data/kaggriculture/search/public-frontier --version 2
/home/will/projects/kaggriculture-2026/.venv/bin/python /home/will/.agents/skills/nvidia-kaggle-skill/scripts/kernel_archive.py boatlee/v20-adaptive-r1-multi-route-agent /data/kaggriculture/search/public-frontier --version 1
/home/will/projects/kaggriculture-2026/.venv/bin/python /home/will/.agents/skills/nvidia-kaggle-skill/scripts/kernel_archive.py boatlee/v21-r1-public-state-route-portfolio /data/kaggriculture/search/public-frontier --version 2
```

Extract the exact runnable agent file from each archived source without editing its logic, run `kaggle_environments.agent.get_last_callable` against it, calculate both the archived notebook SHA-256 and runnable-source SHA-256, and write the seven concrete entries to `frontier_manifest.json`. Add `economic_policy.py`, `searched_route_policy.py`, and current Boatlee v14 as named local entries with their runnable-source hashes, repository provenance, and null Kaggle/archive fields.

- [ ] **Step 5: Implement the report CLI and rerun GREEN**

```python
# frontier_round_robin.py
FRONTIER_SEEDS = tuple(range(810_000, 810_064))

def main() -> None:
    args = parser().parse_args()
    verified = verify_frontier(args.manifest, args.artifact_root)
    report = rank_frontier(verified, FRONTIER_SEEDS, args.workers)
    args.output.write_text(json.dumps(asdict(report), indent=2, sort_keys=True))
```

Run: `.venv/bin/python -m pytest tests/search/test_frontier.py tests/search/test_arena.py -q`

Expected: PASS.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/kaggriculture/search/frontier.py src/kaggriculture/search/frontier_manifest.json src/kaggriculture/search/scripts/frontier_round_robin.py src/kaggriculture/search/arena.py tests/search/test_frontier.py tests/search/test_arena.py
git commit -m "feat: measure the exact public frontier"
```

---

### Task 2: Submission-Safe Action Vocabulary and Decoder

**Files:**

- Create: `src/kaggriculture/action_codec.py`
- Modify: `src/kaggriculture/learn/encoding.py:620-712,873-988,1017-1250`
- Modify: `src/kaggriculture/learn/mask.py:83-100`
- Modify: `src/kaggriculture/sim/rollout.py`
- Test: `tests/test_action_codec.py`
- Test: `tests/learn/test_encoding.py`

**Interfaces:**

- Consumes: engine rule tables from `kaggriculture.constants`.
- Produces: `UNIT_OPS`, `TRANSFER_OPS`, `QUANTITIES`, `MARKET_SLOTS`, `HIRE_SLOT`, `LAND_SLOT`, `SelectedActions`, `decode_selected(selected: SelectedActions) -> dict[str, Any]`, and `safe_pass_action() -> dict[str, Any]`.

- [ ] **Step 1: Write parity and import-boundary tests**

```python
def test_selected_decoder_matches_the_existing_tensor_decoder() -> None:
    unit_indices = (UNIT_OPS.index("PICKUP:WHEAT"), UNIT_OPS.index("WATER"))
    quantity_indices = (QUANTITIES.index(3), QUANTITIES.index(1))
    market_indices = [0] * (len(MARKET_SLOTS) + 2)
    market_indices[HIRE_SLOT] = QUANTITIES.index(2)
    market_indices[LAND_SLOT] = QUANTITIES.index(1)
    selected = SelectedActions(unit_indices, quantity_indices, tuple(market_indices))

    actual = decode_selected(selected)

    assert actual["farmer"] == ["PICKUP", "WHEAT", 3]
    assert actual["hands"] == [["WATER"]]
    assert actual["market"][-3:] == [["HIRE"], ["HIRE"], ["BUY_LAND"]]


def test_action_codec_imports_neither_torch_nor_pydantic() -> None:
    imported = import_in_fresh_process("kaggriculture.action_codec")
    assert "torch" not in imported
    assert "pydantic" not in imported
```

Also parameterize every `UNIT_OPS` label and every market slot against the current `decode_units`/`decode_market` tensor wrappers, using one-hot logits and all-true masks.

- [ ] **Step 2: Run RED**

Run: `.venv/bin/python -m pytest tests/test_action_codec.py tests/learn/test_encoding.py -q`

Expected: FAIL because `kaggriculture.action_codec` does not exist.

- [ ] **Step 3: Move—not copy—the vocabulary and pure decoding rules**

```python
@dataclass(frozen=True)
class SelectedActions:
    unit_indices: tuple[int, ...]
    quantity_indices: tuple[int, ...]
    market_indices: tuple[int, ...]


def decode_selected(selected: SelectedActions) -> dict[str, Any]:
    if len(selected.unit_indices) != len(selected.quantity_indices):
        raise ValueError("one quantity index is required per unit index")
    market_rows = len(MARKET_SLOTS) + 2
    if len(selected.market_indices) != market_rows:
        raise ValueError(f"expected {market_rows} market selections")
    operations = tuple(
        operation_of(op_index, quantity_of(quantity_index))
        for op_index, quantity_index in zip(
            selected.unit_indices, selected.quantity_indices, strict=True
        )
    )
    market = market_orders_of(selected.market_indices)
    return {
        "farmer": operations[0] if operations else ["PASS"],
        "hands": list(operations[1:]),
        "market": market,
    }


def safe_pass_action() -> dict[str, Any]:
    return {"farmer": ["PASS"], "hands": [], "market": []}
```

Cut the existing `UNIT_OPS`, `ITEM_VERBS`, `TRANSFER_OPS`, `MAX_TRANSFER`, `QUANTITIES`, market-slot declarations, `quantity_of`, `_op`, and pure market-order construction out of `learn.encoding` and place them in `action_codec.py`. Re-export them from `learn.encoding` for one release so existing imports and checkpoint topology tests do not move.

- [ ] **Step 4: Make Torch decoders thin selected-index adapters**

```python
def decode_units(
    logits: torch.Tensor,
    quantity_logits: torch.Tensor,
    units: int,
    mask: torch.Tensor,
    quantity_mask: torch.Tensor,
) -> dict[str, Any]:
    legal = logits[0, :units].masked_fill(~mask[0, :units], -torch.inf)
    legal_quantity = quantity_logits[0, :units].masked_fill(
        ~quantity_mask[0, :units], -torch.inf
    )
    selected = SelectedActions(
        tuple(int(value) for value in legal.argmax(dim=-1)),
        tuple(int(value) for value in legal_quantity.argmax(dim=-1)),
        tuple(0 for _ in range(len(MARKET_SLOTS) + 2)),
    )
    decoded = decode_selected(selected)
    return {"farmer": decoded["farmer"], "hands": decoded["hands"]}
```

Keep `decode_market` responsible only for masked `argmax`, then delegate the selected indices to `market_orders_of`. Update simulator imports to read schema constants from `action_codec`, not `learn.encoding`.

- [ ] **Step 5: Run GREEN and the action-path regressions**

Run: `.venv/bin/python -m pytest tests/test_action_codec.py tests/learn/test_encoding.py tests/learn/test_mask.py tests/learn/test_play.py tests/sim/test_rollout.py -q`

Expected: PASS with exact existing action dictionaries.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/kaggriculture/action_codec.py src/kaggriculture/learn/encoding.py src/kaggriculture/learn/mask.py src/kaggriculture/sim/rollout.py tests/test_action_codec.py tests/learn/test_encoding.py
git commit -m "refactor: share the action codec with submissions"
```

---

### Task 3: Canonical Observation Features and Torch Adapters

**Files:**

- Create: `src/kaggriculture/features.py`
- Modify: `src/kaggriculture/learn/encoding.py:1-619,723-801`
- Modify: `src/kaggriculture/learn/mask.py:104-520`
- Modify: `src/kaggriculture/learn/play.py`
- Modify: `src/kaggriculture/learn/rollout.py`
- Modify: `src/kaggriculture/sim/observe.py`
- Modify: `src/kaggriculture/sim/legality.py`
- Test: `tests/test_features.py`
- Test: `tests/learn/test_encoding.py`
- Test: `tests/learn/test_mask.py`
- Test: `tests/sim/test_differential.py`

**Interfaces:**

- Consumes: `action_codec` vocabulary/schema and raw observation mappings.
- Produces: `EncodedObservation`, `encode_observation(observation: Mapping[str, Any], seat: int) -> EncodedObservation`, named accessors, and `to_torch(encoded: EncodedObservation) -> TorchObservation` in `learn.encoding`.

- [ ] **Step 1: Write exact compatibility tests before moving calculations**

```python
def test_canonical_features_round_trip_to_identical_tensors() -> None:
    observation = rich_observation_fixture()
    old = (
        legacy_encode_board(observation, 0),
        legacy_encode_scalars(observation, 0),
        legacy_encode_positions(observation, 0),
        legacy_unit_mask(observation, 0),
        legacy_unit_quantity_mask(observation, 0),
        legacy_market_mask(observation, 0),
    )

    encoded = encode_observation(observation, seat=0)
    actual = to_torch(encoded)

    assert torch.equal(actual.board, old[0])
    assert torch.equal(actual.scalars, old[1])
    assert torch.equal(actual.positions, old[2])
    assert torch.equal(actual.unit_mask, old[3])
    assert torch.equal(actual.quantity_mask, old[4])
    assert torch.equal(actual.market_mask, old[5])


@pytest.mark.slow
def test_canonical_features_are_bit_identical_across_the_replay_corpus() -> None:
    for observation, seat in sampled_corpus_observations(limit=2048):
        assert_torch_parity(observation, seat)
```

Add named-accessor tests for money, phase, live product price/inventory, seed count, shed count, carried count, unit position, crop/animal tile identity, distress, and opponent public supply. Add a fresh-process import test proving `features` imports neither Torch nor Pydantic.

- [ ] **Step 2: Run RED**

Run: `.venv/bin/python -m pytest tests/test_features.py -q`

Expected: FAIL because `kaggriculture.features` does not exist.

- [ ] **Step 3: Define immutable Python storage and stable names**

```python
@dataclass(frozen=True)
class EncodedObservation:
    board: tuple[tuple[tuple[float, ...], ...], ...]
    scalars: tuple[float, ...]
    positions: tuple[int, ...]
    unit_mask: tuple[tuple[bool, ...], ...]
    quantity_mask: tuple[tuple[bool, ...], ...]
    market_mask: tuple[tuple[bool, ...], ...]
    units: int

    def scalar(self, name: str) -> float:
        return self.scalars[SCALAR_INDEX[name]]

    def plane(self, name: str, x: int, y: int, *, opponent: bool = False) -> float:
        index = PLANE_INDEX[name] + (PER_FARM_PLANES if opponent else 0)
        return self.board[index][y][x]


def encode_observation(
    observation: Mapping[str, Any], seat: int
) -> EncodedObservation:
    board = _encode_board_values(observation, seat)
    scalars = _encode_scalar_values(observation, seat)
    positions = _encode_position_values(observation, seat)
    return EncodedObservation(
        board=board,
        scalars=scalars,
        positions=positions,
        unit_mask=_unit_mask_values(observation, seat),
        quantity_mask=_quantity_mask_values(observation, seat),
        market_mask=_market_mask_values(observation, seat),
        units=1 + len(observation["farms"][seat]["hands"]),
    )
```

Move the existing feature arithmetic and legality branches into these pure functions. Define every scalar and plane name from the same engine-derived lists that currently determine layout; reject duplicate names at import time.

- [ ] **Step 4: Replace legacy functions with adapters and migrate combined call sites**

```python
@dataclass(frozen=True)
class TorchObservation:
    board: torch.Tensor
    scalars: torch.Tensor
    positions: torch.Tensor
    unit_mask: torch.Tensor
    quantity_mask: torch.Tensor
    market_mask: torch.Tensor


def to_torch(encoded: EncodedObservation) -> TorchObservation:
    return TorchObservation(
        board=torch.tensor(encoded.board, dtype=torch.float32).unsqueeze(0),
        scalars=torch.tensor(encoded.scalars, dtype=torch.float32).unsqueeze(0),
        positions=torch.tensor(encoded.positions, dtype=torch.int64).unsqueeze(0),
        unit_mask=torch.tensor(encoded.unit_mask, dtype=torch.bool).unsqueeze(0),
        quantity_mask=torch.tensor(encoded.quantity_mask, dtype=torch.bool).unsqueeze(0),
        market_mask=torch.tensor(encoded.market_mask, dtype=torch.bool).unsqueeze(0),
    )
```

Make the old `encode_board`, `encode_scalars`, `encode_positions`, and mask functions return the relevant field from `to_torch(encode_observation(...))`. Change `learn.play` and the reference rollout to call `encode_observation` once and reuse one `TorchObservation`; do not leave six reparses in production.

- [ ] **Step 5: Bind the native schema and run differential parity**

Import channel names, indices, and normalization constants from `features` in `sim.observe`/`sim.legality`; do not route device tensors through Python tuples. Extend `tests/sim/test_differential.py` to compare all canonical fields and masks on the existing legal pickup/place tape and one crop/animal/market state.

Run: `.venv/bin/python -m pytest tests/test_features.py tests/learn/test_encoding.py tests/learn/test_mask.py tests/learn/test_play.py tests/learn/test_rollout.py tests/sim/test_observe.py tests/sim/test_differential.py -q`

Expected: PASS; corpus-only tests may be selected separately with `-m slow`.

- [ ] **Step 6: Run the slow corpus parity test**

Run: `.venv/bin/python -m pytest tests/test_features.py -m slow -q`

Expected: PASS when the corpus is mounted; otherwise one explicit corpus-absence skip and no failure.

- [ ] **Step 7: Commit Task 3**

```bash
git add src/kaggriculture/features.py src/kaggriculture/learn/encoding.py src/kaggriculture/learn/mask.py src/kaggriculture/learn/play.py src/kaggriculture/learn/rollout.py src/kaggriculture/sim/observe.py src/kaggriculture/sim/legality.py tests/test_features.py tests/learn/test_encoding.py tests/learn/test_mask.py tests/sim/test_differential.py
git commit -m "refactor: share canonical farm features"
```

---

### Task 4: Strict Hybrid Configuration and One-Hot Genome

**Files:**

- Create: `src/kaggriculture/hybrid/__init__.py`
- Create: `src/kaggriculture/hybrid/runtime.py`
- Create: `src/kaggriculture/hybrid/config.py`
- Create: `src/kaggriculture/search/genome.py`
- Test: `tests/hybrid/test_config.py`
- Test: `tests/search/test_genome.py`

**Interfaces:**

- Consumes: engine-derived crop, animal, and product names.
- Produces: `HybridConfig`, `OpeningPhaseConfig`, `JobWeightsConfig`, `MarketConfig`, `RuntimeConfig`, `to_runtime(config: HybridConfig) -> RuntimeConfig`, `GenomeCodec`, `encode(config: HybridConfig) -> tuple[float, ...]`, and `decode(vector: Sequence[float]) -> HybridConfig`.

- [ ] **Step 1: Write strict-config and discrete-group tests**

```python
def test_hybrid_config_is_frozen_finite_and_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        HybridConfig.model_validate({"cash_reserve": float("nan")})
    with pytest.raises(ValidationError):
        HybridConfig.model_validate({"unknown_knob": 3})


def test_every_integer_and_category_decodes_from_one_argmax_group() -> None:
    codec = GenomeCodec.default()
    vector = [0.0] * codec.width
    for group in codec.discrete_groups:
        vector[group.start + len(group.values) - 1] = 10.0

    decoded = codec.decode(vector)

    assert decoded.opening.phases[0].target_hands == HAND_TARGETS[-1]
    assert decoded.opening.phases[0].primary_crop == CROP_NAMES[-1]
    assert all(group.width == len(group.values) for group in codec.discrete_groups)


def test_pydantic_to_runtime_round_trip_is_exact_and_dependency_free() -> None:
    config = HybridConfig.default()
    runtime = to_runtime(config)
    assert HybridConfig.from_runtime(runtime) == config
    assert "pydantic" not in import_in_fresh_process("kaggriculture.hybrid.runtime")
```

- [ ] **Step 2: Run RED**

Run: `.venv/bin/python -m pytest tests/hybrid/test_config.py tests/search/test_genome.py -q`

Expected: FAIL because the hybrid package and genome module do not exist.

- [ ] **Step 3: Implement Pydantic authoring models and frozen runtime dataclasses**

```python
class OpeningPhaseConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    start_day: int = Field(ge=0, le=29)
    target_hands: int = Field(ge=0, le=19)
    target_quadrants: int = Field(ge=1, le=5)
    primary_crop: str
    crop_targets: tuple[int, ...]
    animal_targets: tuple[int, ...]
    structure_targets: tuple[int, int]
    cash_reserve: int = Field(ge=0, le=5_000)
    inventory_reserve: int = Field(ge=0, le=100)


class OpeningConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    phases: tuple[OpeningPhaseConfig, ...] = Field(min_length=1)


class JobWeightsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    recovery: float = Field(ge=0.0, le=10.0)
    urgency: float = Field(ge=0.0, le=10.0)
    distance_penalty: float = Field(ge=0.0, le=10.0)
    production: float = Field(ge=0.0, le=10.0)
    transport: float = Field(ge=0.0, le=10.0)
    structure: float = Field(ge=0.0, le=10.0)


class MarketConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    live_price: float = Field(ge=0.0, le=10.0)
    town_demand: float = Field(ge=0.0, le=10.0)
    opponent_supply: float = Field(ge=0.0, le=10.0)
    reserve_penalty: float = Field(ge=0.0, le=10.0)
    liquidation_urgency: float = Field(ge=0.0, le=10.0)


class HybridConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    opening: OpeningConfig
    jobs: JobWeightsConfig
    market: MarketConfig
    liquidation_start_day: int = Field(ge=20, le=29)

    @model_validator(mode="after")
    def ordered_phases_and_fixed_schema(self) -> Self:
        starts = tuple(phase.start_day for phase in self.opening.phases)
        if starts != tuple(sorted(set(starts))):
            raise ValueError("opening phases must have unique increasing start_day")
        for phase in self.opening.phases:
            if len(phase.crop_targets) != len(CROP_NAMES):
                raise ValueError("crop_targets must follow the fixed crop schema")
            if len(phase.animal_targets) != len(ANIMAL_NAMES):
                raise ValueError("animal_targets must follow the fixed animal schema")
        return self


# runtime.py: same field order, standard-library types only
@dataclass(frozen=True)
class RuntimeOpeningPhase:
    start_day: int
    target_hands: int
    target_quadrants: int
    primary_crop: str
    crop_targets: tuple[int, ...]
    animal_targets: tuple[int, ...]
    structure_targets: tuple[int, int]
    cash_reserve: int
    inventory_reserve: int


@dataclass(frozen=True)
class RuntimeConfig:
    phases: tuple[RuntimeOpeningPhase, ...]
    jobs: RuntimeJobWeights
    market: RuntimeMarketWeights
    liquidation_start_day: int

    def to_payload(self) -> dict[str, object]:
        return {
            "phases": [asdict(phase) for phase in self.phases],
            "jobs": asdict(self.jobs),
            "market": asdict(self.market),
            "liquidation_start_day": self.liquidation_start_day,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> Self:
        expected = {"phases", "jobs", "market", "liquidation_start_day"}
        if set(payload) != expected:
            raise ValueError(f"runtime payload keys must be {sorted(expected)}")
        phases = payload["phases"]
        jobs = payload["jobs"]
        market = payload["market"]
        if not isinstance(phases, list) or not all(
            isinstance(phase, dict) for phase in phases
        ):
            raise TypeError("phases must be a list of dictionaries")
        if not isinstance(jobs, dict) or not isinstance(market, dict):
            raise TypeError("jobs and market must be dictionaries")
        return cls(
            phases=tuple(RuntimeOpeningPhase(**phase) for phase in phases),
            jobs=RuntimeJobWeights(**jobs),
            market=RuntimeMarketWeights(**market),
            liquidation_start_day=int(payload["liquidation_start_day"]),
        )
```

Define `RuntimeJobWeights` and `RuntimeMarketWeights` as frozen dataclasses with the fields and order of `JobWeightsConfig` and `MarketConfig`. The Pydantic-to-runtime conversion is the validation boundary; `runtime.py` owns explicit recursive `to_payload`/`from_payload` conversion and exact key/container checks for the generated module. `config.py` owns `to_runtime` and `HybridConfig.from_runtime`; `runtime.py` must not import `config.py`.

- [ ] **Step 4: Implement explicit gene domains and deterministic decoding**

Declare bounded integer domains as explicit tuples—for example `HAND_TARGETS = tuple(range(0, 20))`, `CASH_RESERVES = tuple(range(0, 5_001, 100))`, and `LIQUIDATION_DAYS = tuple(range(20, 30))`. No integer is encoded as a scalar.

```python
@dataclass(frozen=True)
class DiscreteGroup:
    path: tuple[str | int, ...]
    values: tuple[object, ...]
    start: int

    @property
    def width(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class ContinuousGene:
    path: tuple[str | int, ...]
    low: float
    high: float
    index: int


class GenomeCodec:
    def decode(self, vector: Sequence[float]) -> HybridConfig:
        if len(vector) != self.width or not all(math.isfinite(v) for v in vector):
            raise ValueError(f"genome must hold {self.width} finite values")
        payload = copy.deepcopy(self.template)
        for group in self.discrete_groups:
            logits = vector[group.start : group.start + group.width]
            selected = max(range(group.width), key=lambda index: (logits[index], -index))
            assign_path(payload, group.path, group.values[selected])
        for gene in self.continuous_genes:
            unit = min(max(vector[gene.index], 0.0), 1.0)
            assign_path(payload, gene.path, gene.low + unit * (gene.high - gene.low))
        return HybridConfig.model_validate(payload)
```

Encoding writes `1.0` at the selected discrete value and `0.0` elsewhere; continuous values invert their linear mapping. Add a round-trip test over at least 100 deterministic random valid configs.

- [ ] **Step 5: Run GREEN and static checks**

Run: `.venv/bin/python -m pytest tests/hybrid/test_config.py tests/search/test_genome.py -q`

Run: `.venv/bin/ruff check src/kaggriculture/hybrid src/kaggriculture/search/genome.py tests/hybrid tests/search/test_genome.py`

Run: `.venv/bin/ty check src/kaggriculture/hybrid src/kaggriculture/search/genome.py`

Expected: all pass.

- [ ] **Step 6: Commit Task 4**

```bash
git add src/kaggriculture/hybrid src/kaggriculture/search/genome.py tests/hybrid/test_config.py tests/search/test_genome.py
git commit -m "feat: type the hybrid strategy genome"
```

---

### Task 5: Opening Phases and Named Feature Facts

**Files:**

- Create: `src/kaggriculture/hybrid/opening.py`
- Test: `tests/hybrid/test_opening.py`

**Interfaces:**

- Consumes: `EncodedObservation`, `RuntimeConfig`, and named feature accessors.
- Produces: `OpeningTargets`, `select_phase(encoded: EncodedObservation, config: RuntimeConfig) -> RuntimeOpeningPhase`, and `opening_targets(encoded: EncodedObservation, config: RuntimeConfig) -> OpeningTargets`.

- [ ] **Step 1: Write phase, deficit, reserve, and delay tests**

```python
def test_opening_targets_are_state_deficits_not_taped_actions() -> None:
    encoded = encoded_fixture(day=5, hands=2, crops={"WHEAT": 3})
    config = runtime_config(
        phases=(phase(start_day=0, target_hands=4, crops={"WHEAT": 5}),)
    )

    targets = opening_targets(encoded, config)

    assert targets.hands_needed == 2
    assert targets.crop_deficits["WHEAT"] == 2


def test_missed_phase_does_not_replay_already_satisfied_work() -> None:
    encoded = encoded_fixture(day=8, hands=5, crops={"WHEAT": 8})
    config = runtime_config(
        phases=(
            phase(start_day=0, target_hands=3, crops={"WHEAT": 4}),
            phase(start_day=7, target_hands=5, crops={"WHEAT": 8}),
        )
    )

    targets = opening_targets(encoded, config)

    assert targets.hands_needed == 0
    assert targets.crop_deficits["WHEAT"] == 0
```

Add cases for delayed land unlock, existing animals in field/shed/hands, reserve-protected unaffordability, and final liquidation overriding future production targets.

- [ ] **Step 2: Run RED**

Run: `.venv/bin/python -m pytest tests/hybrid/test_opening.py -q`

Expected: FAIL because `kaggriculture.hybrid.opening` does not exist.

- [ ] **Step 3: Implement typed targets strictly from named features**

```python
@dataclass(frozen=True)
class OpeningTargets:
    phase_start_day: int
    hands_needed: int
    crop_deficits: Mapping[str, int]
    animal_deficits: Mapping[str, int]
    protected_cash: int
    protected_inventory: Mapping[str, int]
    liquidating: bool


def opening_targets(
    encoded: EncodedObservation, config: RuntimeConfig
) -> OpeningTargets:
    phase = select_phase(encoded, config)
    hands = round(encoded.scalar("our_hands") * 8)
    crop_deficits = {
        crop: max(phase.crop_targets[index] - encoded.crop_count(crop), 0)
        for index, crop in enumerate(CROP_NAMES)
    }
    liquidating = encoded.scalar("day") * SEASON_DAYS >= config.liquidation_start_day
    if liquidating:
        crop_deficits = dict.fromkeys(CROP_NAMES, 0)
    animal_deficits = {
        animal: max(phase.animal_targets[index] - encoded.animal_count(animal), 0)
        for index, animal in enumerate(ANIMAL_NAMES)
    }
    if liquidating:
        animal_deficits = dict.fromkeys(ANIMAL_NAMES, 0)
    return OpeningTargets(
        phase_start_day=phase.start_day,
        hands_needed=max(phase.target_hands - hands, 0),
        crop_deficits=crop_deficits,
        animal_deficits=animal_deficits,
        protected_cash=phase.cash_reserve,
        protected_inventory=dict.fromkeys(
            PRODUCT_NAMES, phase.inventory_reserve
        ),
        liquidating=liquidating,
    )
```

If an accessor needed here does not exist, add it to `features.py` with an isolated accessor test; do not parse `raw_obs` inside `opening.py`.

- [ ] **Step 4: Run GREEN**

Run: `.venv/bin/python -m pytest tests/hybrid/test_opening.py tests/test_features.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 5**

```bash
git add src/kaggriculture/hybrid/opening.py src/kaggriculture/features.py tests/hybrid/test_opening.py tests/test_features.py
git commit -m "feat: derive guarded opening targets"
```

---

### Task 6: Reactive Unit Jobs and Deterministic Legal Assignment

**Files:**

- Create: `src/kaggriculture/hybrid/jobs.py`
- Test: `tests/hybrid/test_jobs.py`

**Interfaces:**

- Consumes: `EncodedObservation`, `OpeningTargets`, `RuntimeConfig`, `UNIT_OPS`, and quantity buckets.
- Produces: `Job`, `UnitSelection`, `collect_jobs(encoded: EncodedObservation, targets: OpeningTargets, config: RuntimeConfig) -> tuple[Job, ...]`, and `select_units(encoded: EncodedObservation, targets: OpeningTargets, config: RuntimeConfig) -> UnitSelection`.

- [ ] **Step 1: Write recovery, urgency, locality, and legality tests**

```python
def test_weeds_and_at_risk_animals_preempt_new_planting() -> None:
    encoded = encoded_fixture(
        units=((4, 4), (1, 1)),
        weeds=((4, 4),),
        unfed_animals=((1, 1),),
        seeds={"WHEAT": 5},
    )
    targets = targets_fixture(crop_deficits={"WHEAT": 3})

    selected = select_units(encoded, targets, runtime_config())

    assert UNIT_OPS[selected.operation_indices[0]] == "DIG"
    assert UNIT_OPS[selected.operation_indices[1]] == "FEED"


def test_illegal_top_choice_falls_through_to_next_ranked_legal_choice() -> None:
    encoded = encoded_fixture(unit_legal=("PASS", "EAST"), desired_tile=(4, 3))

    selected = select_units(encoded, targets_fixture(), runtime_config())

    assert UNIT_OPS[selected.operation_indices[0]] == "EAST"


def test_assignment_is_nearest_unit_then_stable_unit_index() -> None:
    encoded = encoded_fixture(units=((0, 0), (2, 0)), weeds=((1, 0),))
    assert select_units(encoded, targets_fixture(), runtime_config()) == select_units(
        encoded, targets_fixture(), runtime_config()
    )
```

Add bulk pickup/place quantity tests, delayed worker displacement, full shed, missing seed, animal placement, care/water/harvest urgency, and final-day harvest/haul tests.

- [ ] **Step 2: Run RED**

Run: `.venv/bin/python -m pytest tests/hybrid/test_jobs.py -q`

Expected: FAIL because `kaggriculture.hybrid.jobs` does not exist.

- [ ] **Step 3: Implement jobs as desired legal categories, not raw engine actions**

```python
@dataclass(frozen=True, order=True)
class Job:
    priority: float
    distance: int
    stable_order: int
    position: tuple[int, int]
    operation_index: int
    quantity_index: int
    bound_unit: int | None = None


@dataclass(frozen=True)
class UnitSelection:
    operation_indices: tuple[int, ...]
    quantity_indices: tuple[int, ...]


def select_units(
    encoded: EncodedObservation,
    targets: OpeningTargets,
    config: RuntimeConfig,
) -> UnitSelection:
    jobs = collect_jobs(encoded, targets, config)
    assignments = assign_nearest_legal(encoded, jobs)
    operations: list[int] = []
    quantities: list[int] = []
    for unit in range(encoded.units):
        ranked = ranked_choices_for(unit, encoded, assignments.get(unit), config)
        choice = next(
            candidate
            for candidate in ranked
            if encoded.unit_mask[unit][candidate.operation_index]
            and encoded.quantity_mask[unit][candidate.quantity_index]
        )
        operations.append(choice.operation_index)
        quantities.append(choice.quantity_index)
    return UnitSelection(tuple(operations), tuple(quantities))
```

Movement uses the existing deterministic `step_toward` geometry but maps the result back to a `UNIT_OPS` index. Stable tie order is urgency, distance, target `(y, x)`, job kind, then unit index.

- [ ] **Step 4: Run GREEN and existing heuristic regressions**

Run: `.venv/bin/python -m pytest tests/hybrid/test_jobs.py tests/test_policy.py tests/learn/test_mask.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 6**

```bash
git add src/kaggriculture/hybrid/jobs.py tests/hybrid/test_jobs.py
git commit -m "feat: assign guarded hybrid farm jobs"
```

---

### Task 7: Reactive Market, Integrated Hybrid Agent, and Safe Failure Boundary

**Files:**

- Create: `src/kaggriculture/hybrid/market.py`
- Create: `src/kaggriculture/hybrid/policy.py`
- Test: `tests/hybrid/test_market.py`
- Test: `tests/hybrid/test_policy.py`
- Test: `tests/test_submission.py`

**Interfaces:**

- Consumes: canonical features, unit selections, opening targets, runtime config, and `decode_selected`.
- Produces: `MarketSelection`, `select_market(encoded: EncodedObservation, targets: OpeningTargets, config: RuntimeConfig) -> MarketSelection`, `HybridPolicy.decide(encoded: EncodedObservation) -> dict[str, Any]`, `build_agent(runtime: RuntimeConfig) -> Callable`, and a conservative development `agent` callable that is not wired into `main.py`.

- [ ] **Step 1: Write market and integrated-policy RED tests**

```python
def test_market_preserves_cash_before_buying_and_hiring() -> None:
    encoded = encoded_fixture(money=350, prices={"WHEAT": 25})
    targets = targets_fixture(protected_cash=300, hands_needed=2)

    selected = select_market(encoded, targets, runtime_config())

    assert selected.total_cost <= 50


def test_live_price_and_opponent_supply_change_the_sale_ranking() -> None:
    scarce = encoded_fixture(shed={"TOMATO": 8}, inventory={"TOMATO": -200})
    flooded = encoded_fixture(shed={"TOMATO": 8}, inventory={"TOMATO": 800})

    assert select_market(scarce, targets_fixture(), runtime_config()).indices != (
        select_market(flooded, targets_fixture(), runtime_config()).indices
    )


def test_final_liquidation_sells_available_stock_despite_normal_reserves() -> None:
    encoded = encoded_fixture(day=29, shed={"WHEAT": 12})
    selected = select_market(encoded, targets_fixture(liquidating=True), runtime_config())
    wheat_slot = MARKET_SLOTS.index(("SELL", "WHEAT"))
    assert selected.indices[wheat_slot] != 0


def test_outer_failure_boundary_returns_exact_safe_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(policy, "encode_observation", Mock(side_effect=KeyError("new")))
    assert policy.agent({"unexpected": "schema"}) == safe_pass_action()
```

Add a property test over generated legal observations asserting every emitted category is enabled in the same `EncodedObservation` masks used to decide it. Add a fixed 20-seed episode test with zero ERROR/INVALID statuses.

- [ ] **Step 2: Run RED**

Run: `.venv/bin/python -m pytest tests/hybrid/test_market.py tests/hybrid/test_policy.py -q`

Expected: FAIL because market and policy modules do not exist.

- [ ] **Step 3: Implement slot-by-slot budgeted market selection**

```python
@dataclass(frozen=True)
class MarketSelection:
    indices: tuple[int, ...]
    total_cost: float


def select_market(
    encoded: EncodedObservation,
    targets: OpeningTargets,
    config: RuntimeConfig,
) -> MarketSelection:
    balance = encoded.scalar("our_money") * 10_000.0
    selected: list[int] = []
    total_cost = 0.0
    for slot, legal in enumerate(encoded.market_mask):
        ranked = rank_market_buckets(slot, encoded, targets, config, balance)
        bucket = next(index for index in ranked if legal[index])
        cost = quoted_cost(slot, bucket, encoded)
        if cost > max(balance - targets.protected_cash, 0.0):
            bucket, cost = 0, 0.0
        selected.append(bucket)
        balance -= cost
        total_cost += cost
    return MarketSelection(tuple(selected), total_cost)
```

Sales rank by live marginal value, demand, opponent supply, retained inventory, and liquidation urgency. Purchases rank only when opening deficits exist and reserve constraints remain satisfied. Market slot order is deterministic and uses the existing engine order.

- [ ] **Step 4: Integrate one decision path and no raw-observation fallback**

```python
class HybridPolicy:
    def __init__(self, runtime: RuntimeConfig) -> None:
        self.runtime = runtime

    def decide(self, encoded: EncodedObservation) -> dict[str, Any]:
        targets = opening_targets(encoded, self.runtime)
        units = select_units(encoded, targets, self.runtime)
        market = select_market(encoded, targets, self.runtime)
        return decode_selected(
            SelectedActions(
                units.operation_indices,
                units.quantity_indices,
                market.indices,
            )
        )


AgentCallable = Callable[
    [Mapping[str, Any], object | None],
    dict[str, Any],
]


def build_agent(runtime: RuntimeConfig) -> AgentCallable:
    controller = HybridPolicy(runtime)

    def agent(
        observation: Mapping[str, Any], configuration: object | None = None
    ) -> dict[str, Any]:
        try:
            seat = int(observation.get("player", 0))
            return controller.decide(encode_observation(observation, seat))
        except (KeyError, TypeError, ValueError, IndexError):
            return safe_pass_action()

    return agent
```

The development `agent` binds `build_agent(DEFAULT_RUNTIME_CONFIG)`. Do not import `hybrid.config`, `search`, Torch, or Pydantic from this module, and do not edit `main.py`.

- [ ] **Step 5: Run GREEN, episode smokes, and import audit**

Run: `.venv/bin/python -m pytest tests/hybrid/test_market.py tests/hybrid/test_policy.py tests/test_submission.py -q`

Run: `.venv/bin/python -c "import sys; import kaggriculture.hybrid.policy; assert 'torch' not in sys.modules and 'pydantic' not in sys.modules"`

Expected: all pass.

- [ ] **Step 6: Commit Task 7**

```bash
git add src/kaggriculture/hybrid/market.py src/kaggriculture/hybrid/policy.py tests/hybrid/test_market.py tests/hybrid/test_policy.py tests/test_submission.py
git commit -m "feat: play the reactive hybrid policy"
```

---

### Task 8: League Fitness and Progressive CPU Evolution

**Files:**

- Create: `src/kaggriculture/search/fitness.py`
- Create: `src/kaggriculture/search/evolution.py`
- Create: `src/kaggriculture/search/scripts/hybrid_search.py`
- Modify: `src/kaggriculture/search/arena.py`
- Test: `tests/search/test_fitness.py`
- Test: `tests/search/test_evolution.py`
- Test: `tests/search/test_arena.py`

**Interfaces:**

- Consumes: `GenomeCodec`, `RuntimeConfig`, verified frontier opponents, and arena outcomes.
- Produces: `HybridOpponent`, `StrengthWeights`, `Fitness`, `score_fitness`, `EvolutionConfig`, `SearchState`, `evolve(codec: GenomeCodec, initial_configs: Sequence[HybridConfig], league: Mapping[str, Opponent], weights: StrengthWeights, config: EvolutionConfig, output: Path, resume: SearchState | None = None) -> SearchState`, and resumable JSON artifacts.

- [ ] **Step 1: Write fitness and hybrid-opponent arena tests**

```python
def test_fitness_is_seventy_percent_weighted_mean_and_thirty_percent_worst() -> None:
    rates = {"frontier": 0.40, "strong": 0.60, "weak": 1.00}
    weights = StrengthWeights({"frontier": 4, "strong": 3, "weak": 1})

    result = score_fitness(rates, weights)

    mean = (4 * 0.40 + 3 * 0.60 + 1 * 1.00) / 8
    assert result.value == pytest.approx(0.70 * mean + 0.30 * 0.40)


def test_illegal_or_failed_candidate_is_ineligible() -> None:
    result = score_fitness({"frontier": 1.0}, StrengthWeights({"frontier": 4}), failures=1)
    assert result.eligible is False
    assert result.value == -math.inf


def test_arena_can_play_a_picklable_hybrid_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    opponent = HybridOpponent(runtime_config())
    side = arena._side(opponent)
    assert callable(side)
    assert side(empty_observation())["farmer"]
```

- [ ] **Step 2: Run RED**

Run: `.venv/bin/python -m pytest tests/search/test_fitness.py tests/search/test_evolution.py tests/search/test_arena.py -q`

Expected: FAIL because fitness, evolution, and `HybridOpponent` do not exist.

- [ ] **Step 3: Add the picklable hybrid opponent and fixed strength tiers**

```python
@dataclass(frozen=True)
class HybridOpponent:
    runtime: RuntimeConfig


Opponent = Route | str | HybridOpponent


def _side(opponent: Opponent) -> str | _Agent:
    if isinstance(opponent, HybridOpponent):
        return build_agent(opponent.runtime)
    return opponent if isinstance(opponent, str) else _replay(opponent)
```

Derive quartile weights 4/3/2/1 from `FrontierReport` before search. Force Boatlee v14 to at least 3. Serialize the exact mapping with every `SearchState`.

- [ ] **Step 4: Implement deterministic evolution over logits**

```python
@dataclass(frozen=True)
class EvolutionConfig:
    population: int = 32
    elites: int = 4
    generations: int = 40
    mutation_sigma: float = 0.35
    seed: int = 20260825
    workers: int = 16


def mutate_genome(
    parent: Sequence[float], sigma: float, rng: random.Random
) -> tuple[float, ...]:
    return tuple(value + rng.gauss(0.0, sigma) for value in parent)


def evolve(
    *,
    codec: GenomeCodec,
    initial_configs: Sequence[HybridConfig],
    league: Mapping[str, Opponent],
    weights: StrengthWeights,
    config: EvolutionConfig,
    output: Path,
    resume: SearchState | None = None,
) -> SearchState:
    rng = random.Random(config.seed)
    state = resume or initial_state(codec, initial_configs, config)
    for generation in range(state.generation, config.generations):
        candidates = spawn_population(state.elites, codec, config, rng)
        screened = evaluate_population(candidates, SCREENING_SEEDS, league, weights)
        survivors = top_eligible(screened, count=config.elites * 2)
        developed = evaluate_population(survivors, DEVELOPMENT_SEEDS, league, weights)
        state = state.advance(generation, developed, rng.getstate())
        save_state_atomic(state, output)
    return state
```

Use `SCREENING_SEEDS = range(820_000, 820_008)` and `DEVELOPMENT_SEEDS = range(830_000, 830_032)`. Assert at import time that they do not overlap each other, frontier seeds, or promotion seeds. Use atomic temp-file replacement for resumable state. Reject a resume whose manifest hash, engine, seed sets, genome schema hash, weights, or evolution config differs.

- [ ] **Step 5: Test determinism, resume, progressive counts, and no GPU use**

Mock arena outcomes with a deterministic function of genome and seed. Assert an uninterrupted four-generation search equals a two-generation save plus resume byte-for-byte. Record calls and assert every candidate sees all league members, screening uses 8 seeds, only survivors use 32, and no code imports or calls `torch.cuda`.

Run: `.venv/bin/python -m pytest tests/search/test_fitness.py tests/search/test_evolution.py tests/search/test_arena.py -q`

Expected: PASS.

- [ ] **Step 6: Add the CPU-only CLI**

```python
def main() -> None:
    args = parser().parse_args()
    if args.workers < 1 or args.workers > 16:
        raise SystemExit("--workers must be between 1 and 16 while Toad is running")
    frontier = verify_frontier(args.manifest, args.artifact_root)
    report = FrontierReport.model_validate_json(args.frontier_report.read_text())
    state = evolve(
        codec=GenomeCodec.default(),
        initial_configs=seed_configs(),
        league=frontier.opponents,
        weights=strength_weights(report),
        config=EvolutionConfig(workers=args.workers, seed=args.seed),
        output=args.output,
    )
    write_finalists(state, args.finalists)
```

- [ ] **Step 7: Commit Task 8**

```bash
git add src/kaggriculture/search/arena.py src/kaggriculture/search/fitness.py src/kaggriculture/search/evolution.py src/kaggriculture/search/scripts/hybrid_search.py tests/search/test_arena.py tests/search/test_fitness.py tests/search/test_evolution.py
git commit -m "feat: evolve hybrid policies against the league"
```

---

### Task 9: Paired Promotion Gate and Packaged Hybrid Smoke

**Files:**

- Create: `src/kaggriculture/search/promotion.py`
- Create: `src/kaggriculture/search/scripts/hybrid_holdout.py`
- Create: `src/kaggriculture/search/scripts/freeze_hybrid.py`
- Modify: `src/kaggriculture/scripts/package.py:62-130`
- Test: `tests/search/test_promotion.py`
- Test: `tests/search/test_freeze_hybrid.py`
- Test: `tests/test_package.py`
- Test: `tests/test_submission.py`

**Interfaces:**

- Consumes: finalist config, incumbent, Boatlee v14, verified frontier, exact weights, and arena per-game outcomes.
- Produces: `bootstrap_interval`, `PromotionVerdict`, `evaluate_promotion(candidate: HybridOpponent, incumbent: Opponent, boatlee: Opponent, frontier: Opponent, league: Mapping[str, Opponent], seeds: Sequence[int], workers: int) -> PromotionVerdict`, `freeze_runtime(config: HybridConfig, output: Path) -> Path`, and package-build overrides used only by tests until approval.

- [ ] **Step 1: Write exact promotion-ruling tests**

```python
def test_promotion_requires_every_gate_not_only_the_aggregate() -> None:
    verdict = PromotionVerdict(
        boatlee=Interval(0.55, 0.51, 0.59),
        frontier=Interval(0.54, 0.501, 0.58),
        league_delta=Interval(0.03, 0.01, 0.05),
        matchup_deltas={"old_meta": -0.051},
        failures=0,
        deterministic=True,
    )
    assert verdict.passed is False
    assert verdict.reasons == ("old_meta regressed by 0.051 (> 0.050)",)


def test_bootstrap_pairs_both_seats_of_each_seed() -> None:
    samples = ((1.0, 0.0), (0.5, 0.5), (1.0, 1.0))
    first = bootstrap_interval(samples, draws=10_000, seed=20260825)
    second = bootstrap_interval(samples, draws=10_000, seed=20260825)
    assert first == second


def test_holdout_seeds_are_disjoint_from_every_search_seed() -> None:
    assert set(PROMOTION_SEEDS).isdisjoint(FRONTIER_SEEDS)
    assert set(PROMOTION_SEEDS).isdisjoint(SCREENING_SEEDS)
    assert set(PROMOTION_SEEDS).isdisjoint(DEVELOPMENT_SEEDS)
    assert len(PROMOTION_SEEDS) == 128
```

- [ ] **Step 2: Run RED**

Run: `.venv/bin/python -m pytest tests/search/test_promotion.py tests/search/test_freeze_hybrid.py tests/test_package.py -q`

Expected: FAIL because promotion and freeze modules do not exist and package build has no alternate-entrypoint seam.

- [ ] **Step 3: Implement paired seed-level intervals and atomic rulings**

```python
PROMOTION_SEEDS = tuple(range(840_000, 840_128))


def bootstrap_interval(
    paired_seat_points: Sequence[tuple[float, float]],
    *,
    draws: int = 10_000,
    seed: int = 20260825,
) -> Interval:
    per_seed = tuple(statistics.mean(pair) for pair in paired_seat_points)
    rng = random.Random(seed)
    estimates = sorted(
        statistics.mean(rng.choices(per_seed, k=len(per_seed))) for _ in range(draws)
    )
    return Interval(
        mean=statistics.mean(per_seed),
        low=estimates[int(draws * 0.025)],
        high=estimates[int(draws * 0.975)],
    )
```

`evaluate_promotion` must compute direct candidate-vs-Boatlee and candidate-vs-frontier intervals, paired candidate-minus-incumbent league deltas on identical opponent/seed/seat rows, and individual matchup point deltas. Build the entire verdict before setting `passed`; never mutate a partial verdict after one gate raises.

- [ ] **Step 4: Generate a dependency-free frozen runtime module**

```python
def freeze_runtime(config: HybridConfig, output: Path) -> Path:
    runtime = to_runtime(config)
    source = (
        '"""Generated validated hybrid runtime configuration."""\n\n'
        "from kaggriculture.hybrid.runtime import RuntimeConfig\n\n"
        f"RUNTIME = RuntimeConfig.from_payload({runtime.to_payload()!r})\n"
    )
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(source)
    temporary.replace(output)
    loaded = load_runtime_module(output)
    if loaded.RUNTIME != runtime:
        raise RuntimeError("frozen runtime round trip changed the configuration")
    return output
```

Tests write to `tmp_path`; this task must not create `src/kaggriculture/hybrid/winner.py` in the working tree.

- [ ] **Step 5: Add an alternate-entrypoint package seam and real archive smoke**

Change `build` to accept keyword-only `entrypoint: Path = ENTRYPOINT` and `required: Mapping[Path, str] = REQUIRED`. Existing CLI behavior stays unchanged. In the test, write a temporary `main.py` that constructs `RuntimeConfig.from_payload(payload_literal)` from `runtime_config().to_payload()` rendered with `repr` and passes it to `build_agent`, call `build(output, entrypoint=hybrid_main, required={})`, extract the archive, and run a full reference-engine episode against `economic_policy` in a subprocess. The temporary entrypoint is self-contained, so the archive does not depend on importing a configuration file outside the package root.

Assert:

```python
assert archive_size < 4 * 1024 * 1024
assert all(state.status == "DONE" for state in final_states)
assert max(per_turn_seconds) < 1.0
assert percentile(per_turn_seconds, 99) < 0.100
assert "torch" not in agent_imported_modules
assert "pydantic" not in agent_imported_modules
assert not any("/search/" in name for name in archive_names)
```

- [ ] **Step 6: Run GREEN and package regressions**

Run: `.venv/bin/python -m pytest tests/search/test_promotion.py tests/search/test_freeze_hybrid.py tests/test_package.py tests/test_submission.py -q`

Expected: PASS without changing the default packaged Boatlee entrypoint.

- [ ] **Step 7: Commit Task 9**

```bash
git add src/kaggriculture/search/promotion.py src/kaggriculture/search/scripts/hybrid_holdout.py src/kaggriculture/search/scripts/freeze_hybrid.py src/kaggriculture/scripts/package.py tests/search/test_promotion.py tests/search/test_freeze_hybrid.py tests/test_package.py tests/test_submission.py
git commit -m "feat: gate and package hybrid candidates"
```

---

### Task 10: Run Frontier, Search, and Untouched Holdout; Stop Before Serving

**Files:**

- Create: `docs/experiments/2026-08-25-hybrid-frontier-and-search.md`
- Create only if the gate passes: `docs/experiments/2026-08-25-hybrid-candidate.json`
- Do not modify: `main.py`

**Interfaces:**

- Consumes: all previous CLIs and the exact external public artifacts.
- Produces: reproducible frontier matrix, search evidence, promotion verdict, and—only on PASS—a reviewed candidate JSON outside the submission package.

- [ ] **Step 1: Run the exact frontier round robin at low priority**

```bash
nice -n 10 .venv/bin/python -m kaggriculture.search.scripts.frontier_round_robin --manifest src/kaggriculture/search/frontier_manifest.json --artifact-root /data/kaggriculture/search/public-frontier --workers 16 --output run/hybrid/frontier.json
```

Expected: one 128-game-per-pair matrix, no failures, and an explicit `frontier_name`.

- [ ] **Step 2: Run resumable progressive evolution**

```bash
nice -n 10 .venv/bin/python -m kaggriculture.search.scripts.hybrid_search --manifest src/kaggriculture/search/frontier_manifest.json --artifact-root /data/kaggriculture/search/public-frontier --frontier-report run/hybrid/frontier.json --workers 16 --seed 20260825 --output run/hybrid/search-state.json --finalists run/hybrid/finalists.json
```

Expected: completed configured generations, every generation recording all league matchups, no CUDA context, and a finalist list containing decoded Pydantic configurations plus source/manifest/schema hashes.

- [ ] **Step 3: Run the untouched promotion gate exactly once per finalist**

```bash
nice -n 10 .venv/bin/python -m kaggriculture.search.scripts.hybrid_holdout --finalists run/hybrid/finalists.json --manifest src/kaggriculture/search/frontier_manifest.json --artifact-root /data/kaggriculture/search/public-frontier --frontier-report run/hybrid/frontier.json --workers 16 --output run/hybrid/promotion.json
```

Expected: 128 untouched seeds, both seats, direct Boatlee and measured-frontier intervals, paired full-league deltas, every matchup delta, failures, determinism, runtime, and one atomic PASS/FAIL verdict per finalist.

- [ ] **Step 4: Run the complete verification gate**

```bash
.venv/bin/python -m pytest tests/test_action_codec.py tests/test_features.py tests/hybrid tests/search tests/learn/test_encoding.py tests/learn/test_mask.py tests/learn/test_play.py tests/learn/test_rollout.py tests/sim/test_observe.py tests/sim/test_differential.py tests/test_package.py tests/test_submission.py -q
.venv/bin/ruff format --check src/kaggriculture/action_codec.py src/kaggriculture/features.py src/kaggriculture/hybrid src/kaggriculture/search tests/hybrid tests/search tests/test_action_codec.py tests/test_features.py
.venv/bin/ruff check src/kaggriculture/action_codec.py src/kaggriculture/features.py src/kaggriculture/hybrid src/kaggriculture/search tests/hybrid tests/search tests/test_action_codec.py tests/test_features.py
.venv/bin/ty check
git diff --check
```

Expected: all focused tests and static checks pass. Run the corpus parity separately with `pytest tests/test_features.py -m slow -q`; run real socket/GPU-marked tests only when their required hardware/sandbox permissions are available.

- [ ] **Step 5: Write the evidence report and candidate artifact**

The report must contain:

- exact commit and engine version;
- public URLs, notebook versions, source hashes, and historical scores;
- complete frontier matrix and strength weights;
- fixed seed ranges and proof of disjointness;
- evolution settings, generation count, and finalist configurations;
- Boatlee/frontier/full-league paired intervals and individual matchup deltas;
- illegal action, exception, forfeit, determinism, package size, and p99 runtime results;
- whether the active Toad process remained live and on its original GPU;
- the exact promotion ruling with every failed condition named.

If and only if a finalist passes, copy its decoded Pydantic JSON from the immutable search artifact to `docs/experiments/2026-08-25-hybrid-candidate.json` and record both SHA-256 values. Do not generate `hybrid/winner.py`, edit `main.py`, package for upload, push a Kaggle kernel, or spend a submission slot.

- [ ] **Step 6: Commit evidence only**

Always stage the evidence report:

```bash
git add docs/experiments/2026-08-25-hybrid-frontier-and-search.md
```

If the promotion verdict is PASS, additionally stage the immutable candidate:

```bash
git add docs/experiments/2026-08-25-hybrid-candidate.json
```

Then commit the staged evidence:

```bash
git commit -m "docs: report the hybrid frontier gate"
```

- [ ] **Step 7: Stop for the serving decision**

Present the report, candidate hash, and PASS/FAIL result to the user. A PASS permits a new, separately designed change to freeze the candidate, switch `main.py`, build the real submission, and optionally submit. A FAIL returns to development/search seeds without inspecting or recycling failed holdout seeds.

---

## Final Verification Summary

Before claiming the implementation is complete:

1. Confirm `git status --short` contains no unintended tracked changes.
2. Confirm the active Toad tmux session and PID are still live and its `CUDA_VISIBLE_DEVICES` binding is unchanged.
3. Confirm fresh-process imports of `kaggriculture.action_codec`, `kaggriculture.features`, `kaggriculture.hybrid.runtime`, and `kaggriculture.hybrid.policy` load neither Torch nor Pydantic.
4. Confirm the default `main.py` and default package build still serve Boatlee v14.
5. Confirm search/frontier sources are absent from the alternate hybrid archive.
6. Confirm all three seed sets and the frontier seed set are disjoint.
7. Confirm the promotion result was computed once from the untouched 128-seed holdout and was not used to mutate another candidate.
