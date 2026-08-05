# Market Head Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the cloned policy a market action space, so it can sell what it grows and bank more than its opening 3,000.

**Architecture:** A second head on the Phase 2 trunk. Market decisions are global, so this head reads the pooled trunk — unlike the unit head, which gathers at each unit's tile. Orders are modelled as a quantity per `(verb, item)` pair rather than as a sequence of order slots, and quantities are classified into buckets rather than regressed.

**Tech Stack:** PyTorch, Lightning (`seed_everything` only), numpy `.npz` shards, `kaggle_environments`, Weights & Biases.

## Global Constraints

- Run everything with `uv run`. Add dependencies with `uv add`. Never `pip`, never bare `python`.
- `uv run pre-commit run -a` must pass before every commit. Fix type errors; never add ignore comments.
- Google-style docstrings on every public function, class and module. Absolute imports. Never `from __future__ import annotations`.
- `logging.info` / `logging.warning`, never `print()`. `tqdm` for long iterables.
- Always include the batch dimension. Prefer `nn.functional` over an aliased `F`. Prefer `.tile` over `.unsqueeze(1).expand`. Prefer `.flatten` when collapsing dimensions. Prefer `.numpy(force=True)`. Never `einops.rearrange`.
- Required argparse arguments take no `--` prefix. Prefer CONSTANTS over CLI arguments.
- Training and inference code is shared, never duplicated.
- No defensive code, no unused code paths. Raise a concise error for unsupported cases rather than adding a fallback.
- Shape and vocabulary constants live in `kaggriculture.learn.encoding` and are imported, never restated as literals.
- Tests reading `/data/kaggriculture/episodes` carry `pytest.mark.slow` and skip cleanly without it. The default run has `addopts = "-m 'not slow'"`. The fast suite must never read the corpus.
- **Every test guarding load-bearing behaviour must be observed to fail when that behaviour is broken.** Break it, watch it go red, restore it, and report what you broke. This repo has shipped four tests that passed while agreeing with the bug they named.
- **Any corpus statistic must be measured across all six archives**, or reported per archive. Measuring one archive and generalising has produced two wrong committed claims.
- The submitted agent must never import `wandb` or anything under `scripts/`; the sandbox has no network.
- Do not mention Claude or AI assistance in commit messages.

## Context

Phase 2 (`docs/superpowers/plans/2026-08-05-imitation-dataset.md`) built the pipeline: corpus selection, observation and action encoding, sharded datasets, a 10.1M-parameter residual CNN, behaviour cloning, and a league gate. It reached 0.8566 holdout accuracy and then lost 0–100 across 400 league games, banking exactly its starting 3,000 every time.

The cause is structural. The engine increases money in exactly one place — `farm["money"] += price` in `_commit_unit`, crediting a completed `SELL` — and none of the 22 `UNIT_OPS` verbs is a market verb.

Spec: `docs/superpowers/specs/2026-08-05-market-head.md`.

### Measured facts this plan depends on

All from six archives, 10 episodes each, 86,400 seat-turns:

- Orders per turn: `{0: 42333, 1: 22981, 2: 8861, 3: 3179, 4: 3975, 5: 2201, 6: 801, 7: 538, 8: 426, 9: 202, 10: 903}`. The spike at 10 is `maxMarketOrdersPerTurn` clipping.
- Quantities: 62 distinct values to a max of 85; 34.5% are exactly 1, 93.5% are ≤ 12.
- Items per verb: `BUY_SEED` all 5 crops; `SELL` all 9 products; `BUY_ANIMAL` all 3 animals; `BUY_PRODUCT` is `WHEAT` 27,635 against `FERTILIZER` twice.
- 22.3% of order-bearing turns repeat a `(verb, item)` pair, but almost all is `HIRE`. Non-`HIRE` repeats ≈1.8%.

### Interfaces this plan consumes

From `kaggriculture.learn.encoding`: `BOARD` (10), `TILE_PLANES` (38), `SCALARS` (28), `MAX_UNITS` (20), `IGNORE` (-100), `UNIT_OPS` (22), `encode_board`, `encode_scalars`, `encode_positions`, `encode_units(action, units)`, `decode_units(logits, units)`, `unit_count(observation, seat)`, `TooManyUnitsError`.

From `kaggriculture.learn.dataset`: `build_shard(samples, destination, stride)`, `Shards`, `ROWS_PER_SHARD`.

From `kaggriculture.learn.scripts.build`: `SHARDS`, `TRAIN`, `HOLDOUT`, `write_dataset`, `clear_shards`.

From `kaggriculture.learn.model`: `Policy.forward(board, scalars, positions) -> (batch, MAX_UNITS, len(UNIT_OPS))`.

## File Structure

| File                                                                                                | Responsibility                                                                                                  |
| --------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `src/kaggriculture/learn/encoding.py`                                                               | **Modify.** Add the market vocabulary, bucket table, `encode_market`, `decode_market`. Unit encoders untouched. |
| `src/kaggriculture/learn/dataset.py`                                                                | **Modify.** Rows carry market labels; `Shards` returns five tensors.                                            |
| `src/kaggriculture/learn/model.py`                                                                  | **Modify.** A pooled market head; `forward` returns both heads.                                                 |
| `src/kaggriculture/learn/scripts/train.py`                                                          | **Modify.** Two loss terms, two accuracies.                                                                     |
| `src/kaggriculture/learn/scripts/play.py`                                                           | **Modify.** Merge both decodes into one action dict.                                                            |
| `src/kaggriculture/package.py`                                                                      | **Modify.** Ship the checkpoint.                                                                                |
| `tests/learn/test_encoding.py`, `test_dataset.py`, `test_model.py`, `test_train.py`, `test_play.py` | **Modify.**                                                                                                     |
| `tests/test_submission.py`                                                                          | **Modify.** Pin that the archive carries weights and no `scripts/`.                                             |

---

### Task 1: The market vocabulary and quantity buckets

**Files:**

- Modify: `src/kaggriculture/learn/encoding.py`
- Test: `tests/learn/test_encoding.py`

**Interfaces:**

- Consumes: `PRODUCTS`, `CROPS`, `ANIMALS` from `kaggriculture.constants`.
- Produces: `MARKET_SLOTS: tuple[tuple[str, str], ...]`, `QUANTITIES: tuple[int, ...]`, `MAX_ORDERS: int`, `bucket_of(n) -> int`, `quantity_of(bucket) -> int`, `encode_market(action) -> torch.Tensor`, `decode_market(logits) -> list[list]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_every_verb_item_pair_the_corpus_uses_has_a_slot() -> None:
    """A missing slot silently drops an order the teacher actually played."""
    verbs = {verb for verb, _ in MARKET_SLOTS}

    assert verbs == {"SELL", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL"}
    assert {item for verb, item in MARKET_SLOTS if verb == "SELL"} == set(PRODUCTS)
    assert {item for verb, item in MARKET_SLOTS if verb == "BUY_SEED"} == set(CROPS)
    assert {item for verb, item in MARKET_SLOTS if verb == "BUY_ANIMAL"} == set(ANIMALS)


def test_buckets_are_exact_where_the_corpus_is_dense() -> None:
    """93.5% of orders are 12 or fewer, so those must not be lumped into ranges."""
    assert QUANTITIES[:13] == tuple(range(13))
    assert bucket_of(0) == 0
    assert bucket_of(7) == 7
    assert bucket_of(12) == 12
    assert bucket_of(13) == bucket_of(16) > 12
    assert bucket_of(85) == len(QUANTITIES) - 1


def test_a_bucket_round_trips_to_a_quantity_that_lands_in_it() -> None:
    """Decoding must not emit a count outside the bucket it came from."""
    for n in (0, 1, 5, 12, 14, 20, 30, 85):
        assert bucket_of(quantity_of(bucket_of(n))) == bucket_of(n)


def test_hire_is_a_count_and_buy_land_is_a_flag() -> None:
    """HIRE is atomic, so hiring three hands is three orders, not a quantity."""
    action = {"market": [["HIRE"], ["HIRE"], ["HIRE"], ["BUY_LAND"]]}

    labels = encode_market(action)

    assert labels[0, HIRE_SLOT].item() == bucket_of(3)
    assert labels[0, LAND_SLOT].item() == 1


def test_repeated_orders_for_one_item_sum() -> None:
    """22% of order-bearing turns repeat a pair; dropping one loses a real sale."""
    action = {"market": [["SELL", "WHEAT", 3], ["SELL", "WHEAT", 4]]}

    labels = encode_market(action)

    assert labels[0, MARKET_SLOTS.index(("SELL", "WHEAT"))].item() == bucket_of(7)


def test_decoding_never_exceeds_the_engine_s_order_cap() -> None:
    """The engine truncates past MAX_ORDERS, so anything beyond it is discarded."""
    logits = torch.zeros(1, len(MARKET_SLOTS) + 2, len(QUANTITIES))
    logits[0, :, 1] = 10.0

    orders = decode_market(logits)

    assert len(orders) <= MAX_ORDERS


def test_decoding_puts_sells_before_buys() -> None:
    """Orders fill in sequence, so a sale must fund the purchase it precedes."""
    logits = torch.zeros(1, len(MARKET_SLOTS) + 2, len(QUANTITIES))
    logits[0, MARKET_SLOTS.index(("SELL", "WHEAT")), 2] = 10.0
    logits[0, MARKET_SLOTS.index(("BUY_SEED", "MELON")), 2] = 10.0

    orders = decode_market(logits)

    assert [order[0] for order in orders] == ["SELL", "BUY_SEED"]


def test_an_empty_market_decodes_to_no_orders() -> None:
    """Half of all turns trade nothing; emitting a zero order would be rejected."""
    logits = torch.zeros(1, len(MARKET_SLOTS) + 2, len(QUANTITIES))
    logits[0, :, 0] = 10.0

    assert decode_market(logits) == []
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/learn/test_encoding.py -q`
Expected: FAIL on `ImportError` for `MARKET_SLOTS`.

- [ ] **Step 3: Implement**

`MARKET_SLOTS` is every `(verb, item)` pair the engine accepts: `SELL` × `sorted(PRODUCTS)`, `BUY_SEED` × `sorted(CROPS)`, `BUY_PRODUCT` × `sorted(PRODUCTS)`, `BUY_ANIMAL` × `sorted(ANIMALS)`. Build it from the constants, never by hand — `BUY_PRODUCT` is almost always `WHEAT` in the corpus, but the engine accepts any product and a hand-written list is how `DROP` went missing in Phase 2.

Two extra slots follow the pairs: `HIRE_SLOT` (a count) and `LAND_SLOT` (a flag). Define them as `len(MARKET_SLOTS)` and `len(MARKET_SLOTS) + 1`.

`QUANTITIES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 24, 40, 64)`. The first thirteen are exact; the rest are representatives of the ranges `13-20`, `21-32`, `33-52`, `53+`. `bucket_of(n)` returns the index of the largest quantity not exceeding `n`; `quantity_of(bucket)` returns `QUANTITIES[bucket]`.

`encode_market(action)` returns `(1, len(MARKET_SLOTS) + 2)` int64 bucket indices. Sum quantities for repeated pairs. Count `HIRE` orders into `HIRE_SLOT` as a bucket. Set `LAND_SLOT` to 1 if any `BUY_LAND` order is present, else 0.

`decode_market(logits)` takes `(1, len(MARKET_SLOTS) + 2, len(QUANTITIES))`, argmaxes each slot, and emits orders for the non-zero ones: sells first, then buys, then `HIRE` repeated by its count, then `BUY_LAND`. Truncate to `MAX_ORDERS`, read from the engine's `maxMarketOrdersPerTurn` default of 10 — define it as a module constant with the engine reference in its docstring.

Note in the docstring that no slot is ever `IGNORE`: unlike unit slots, every market slot is a real decision, and "trade nothing" is the meaningful class 0. The market loss therefore masks nothing.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/learn/test_encoding.py -q`
Expected: PASS.

- [ ] **Step 5: Prove the guards discriminate**

Break each of these, confirm the named test fails, restore:

| Break                                        | Test that must go red                                  |
| -------------------------------------------- | ------------------------------------------------------ |
| Drop `BUY_PRODUCT` from `MARKET_SLOTS`       | `test_every_verb_item_pair_the_corpus_uses_has_a_slot` |
| Make `QUANTITIES` start `(0, 1, 2, 4, 8, …)` | `test_buckets_are_exact_where_the_corpus_is_dense`     |
| Overwrite rather than sum repeated pairs     | `test_repeated_orders_for_one_item_sum`                |
| Emit buys before sells                       | `test_decoding_puts_sells_before_buys`                 |
| Drop the `MAX_ORDERS` truncation             | `test_decoding_never_exceeds_the_engine_s_order_cap`   |

- [ ] **Step 6: Commit**

```bash
git add src/kaggriculture/learn/encoding.py tests/learn/test_encoding.py
git commit -m "feat: a market vocabulary the policy can express"
```

---

### Task 2: A corpus-measured check on the vocabulary

**Files:**

- Test: `tests/learn/test_encoding.py`

**Interfaces:**

- Consumes: `MARKET_SLOTS`, `encode_market`, `decode_market`, `bucket_of` from Task 1; `CORPUS` from `kaggriculture.learn.corpus`.

Task 1's tests are all hand-built dicts, which is exactly the shape of test this repo has shipped four bugs behind. This task checks the vocabulary against the corpus itself.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.slow
@pytest.mark.skipif(not CORPUS.is_dir(), reason="needs the replay corpus")
def test_every_order_in_the_corpus_encodes() -> None:
    """A verb or item with no slot is dropped silently, not loudly.

    Checked across every archive rather than one, because the ladder's agent
    mix changes daily -- the verb mix inverts between 07-30 and 08-04, and two
    committed claims in this repo have already come from generalising a single
    archive.
    """
    seen = set()
    for archive in sorted(CORPUS.glob("*.zip")):
        with zipfile.ZipFile(archive) as bundle:
            name = next(n for n in bundle.namelist() if n.endswith(".json"))
            with bundle.open(name) as member:
                steps = json.load(member)["steps"]
        for step in steps:
            for seat in (0, 1):
                action = step[seat].get("action") or {}
                for order in action.get("market", []) or []:
                    seen.add(order[0] if len(order) < 3 else (order[0], order[1]))
                encode_market(action)

    unknown = {s for s in seen if isinstance(s, tuple) and s not in MARKET_SLOTS}
    assert not unknown, f"corpus plays {unknown}, which no slot can express"
    assert {s for s in seen if isinstance(s, str)} <= {"HIRE", "BUY_LAND"}
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/learn/test_encoding.py -m slow -q`
Expected: PASS, having read six archives.

- [ ] **Step 3: Confirm it can fail**

Temporarily remove `("SELL", "WHEAT")` from `MARKET_SLOTS`; the test must report it as unknown. Restore.

- [ ] **Step 4: Commit**

```bash
git add tests/learn/test_encoding.py
git commit -m "test: check the market vocabulary against every archive"
```

---

### Task 3: Market labels in the dataset

**Files:**

- Modify: `src/kaggriculture/learn/dataset.py`
- Test: `tests/learn/test_dataset.py`

**Interfaces:**

- Consumes: `encode_market` from Task 1.
- Produces: rows of `(board, scalars, positions, labels, market)`; `Shards.__getitem__` returns five tensors.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.slow
@pytest.mark.skipif(not CORPUS.is_dir(), reason="needs the replay corpus")
def test_a_shard_carries_market_labels_of_the_declared_width(tmp_path: Path) -> None:
    """A ragged market label would fail at train time, not build time."""
    destination = tmp_path / "shard.npz"

    build_shard([one_sample()], destination, stride=64)

    board, scalars, positions, labels, market = Shards([destination])[0]
    assert market.shape == (len(MARKET_SLOTS) + 2,)
    assert market.dtype == torch.int64


@pytest.mark.slow
@pytest.mark.skipif(not CORPUS.is_dir(), reason="needs the replay corpus")
def test_market_labels_come_from_the_same_action_as_the_unit_labels(
    tmp_path: Path,
) -> None:
    """Both heads must be trained on one decision, not two adjacent ones.

    The unit labels are the action that follows a row's observation. If the
    market labels were taken from the row's own index instead, the model would
    learn to trade one turn behind its own farming, and nothing would raise.
    """
    sample = one_sample()
    with zipfile.ZipFile(ARCHIVE) as bundle, bundle.open(sample.name) as member:
        steps = json.load(member)["steps"]

    turn = next(
        i
        for i in range(1, len(steps) - 1)
        if not torch.equal(
            encode_market(steps[i][sample.seat]["action"] or {})[0],
            encode_market(steps[i + 1][sample.seat]["action"] or {})[0],
        )
    )
    destination = tmp_path / "aligned.npz"
    build_shard([sample], destination, stride=1)

    _, _, _, _, market = Shards([destination])[turn]

    assert torch.equal(market, encode_market(steps[turn + 1][sample.seat]["action"])[0])
    assert not torch.equal(market, encode_market(steps[turn][sample.seat]["action"])[0])
```

- [ ] **Step 2: Run and watch them fail**

Run: `uv run pytest tests/learn/test_dataset.py -m slow -q`
Expected: FAIL — `Shards` returns four tensors.

- [ ] **Step 3: Implement**

Add a fifth array to the shard. The market labels come from the **same action as the unit labels** — `action[i + 1]` — because both are one decision. Do not reach for `action[i]`.

- [ ] **Step 4: Run**

Run: `uv run pytest tests/learn/test_dataset.py -m slow -q`
Expected: PASS.

- [ ] **Step 5: Prove the alignment guard discriminates**

Source the market labels from `steps[index]` instead of `steps[index + 1]`; `test_market_labels_come_from_the_same_action_as_the_unit_labels` must fail. Restore.

- [ ] **Step 6: Rebuild and commit**

```bash
uv run python -m kaggriculture.learn.scripts.build
git add src/kaggriculture/learn/dataset.py tests/learn/test_dataset.py
git commit -m "feat: carry the turn's market orders alongside its unit orders"
```

Report the row counts and the skipped-turn count. `write_dataset` clears the directory first, so the previous build's shards cannot survive into this one.

---

### Task 4: The market head

**Files:**

- Modify: `src/kaggriculture/learn/model.py`
- Test: `tests/learn/test_model.py`

**Interfaces:**

- Consumes: `MARKET_SLOTS`, `QUANTITIES` from Task 1.
- Produces: `Policy.forward(board, scalars, positions) -> tuple[torch.Tensor, torch.Tensor]` — unit logits `(batch, MAX_UNITS, len(UNIT_OPS))` and market logits `(batch, len(MARKET_SLOTS) + 2, len(QUANTITIES))`.

- [ ] **Step 1: Write the failing tests**

```python
def test_forward_returns_both_heads() -> None:
    """One trunk, two decisions: what the units do and what the farm trades."""
    model = Policy()
    board = torch.zeros(2, TILE_PLANES, BOARD, BOARD)

    units, market = model(board, torch.zeros(2, SCALARS), _positions(2))

    assert units.shape == (2, MAX_UNITS, len(UNIT_OPS))
    assert market.shape == (2, len(MARKET_SLOTS) + 2, len(QUANTITIES))


def test_the_market_head_reads_the_whole_board() -> None:
    """What to sell depends on the whole farm, not on any one tile.

    The unit head deliberately reads only its own unit's tile. The market head
    must not: a harvest anywhere changes what there is to sell.
    """
    torch.manual_seed(0)
    model = Policy().eval()
    board = torch.randn(1, TILE_PLANES, BOARD, BOARD)
    elsewhere = board.clone()
    elsewhere[0, :, 9, 9] += 5.0

    with torch.no_grad():
        _, before = model(board, torch.zeros(1, SCALARS), _positions(1))
        _, after = model(elsewhere, torch.zeros(1, SCALARS), _positions(1))

    assert not torch.equal(before, after)


def test_unit_positions_do_not_move_the_market_head() -> None:
    """Where a hand stands is not a reason to trade differently."""
    torch.manual_seed(0)
    model = Policy().eval()
    board = torch.randn(1, TILE_PLANES, BOARD, BOARD)
    moved = _positions(1)
    moved[0, 0] = BOARD * BOARD - 1

    with torch.no_grad():
        _, here = model(board, torch.zeros(1, SCALARS), _positions(1))
        _, there = model(board, torch.zeros(1, SCALARS), moved)

    assert torch.equal(here, there)


def test_the_model_still_fits_the_size_every_winner_used() -> None:
    """A second head must not turn a 10M-parameter trunk into something else."""
    parameters = sum(p.numel() for p in Policy().parameters())

    assert 1e6 < parameters < 20e6
```

- [ ] **Step 2: Run and watch them fail**

Run: `uv run pytest tests/learn/test_model.py -q`
Expected: FAIL — `forward` returns one tensor.

- [ ] **Step 3: Implement**

Pool the trunk with `features.mean(dim=(2, 3))` and project to `(batch, len(MARKET_SLOTS) + 2, len(QUANTITIES))` through one `Linear`. Pooling is right here and wrong for units, for the reason the existing `forward` docstring already gives — say so in the new docstring so the asymmetry reads as deliberate.

Every existing test in `test_model.py` and `test_train.py` that calls `forward` now unpacks two values. Update them; do not leave a call site taking the tuple as a tensor.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 5: Prove the guards discriminate**

Feed the market head the gathered per-unit columns instead of the pooled trunk; `test_unit_positions_do_not_move_the_market_head` must fail. Restore. Then report the new parameter count broken down by component.

- [ ] **Step 6: Commit**

```bash
git add src/kaggriculture/learn/model.py tests/learn/
git commit -m "feat: a market head on the trunk the units already read"
```

---

### Task 5: Train both heads

**Files:**

- Modify: `src/kaggriculture/learn/scripts/train.py`
- Test: `tests/learn/test_train.py`

**Interfaces:**

- Consumes: five-tensor rows from Task 3, two-headed `forward` from Task 4.
- Produces: `market_loss(logits, labels) -> torch.Tensor`, `MARKET_WEIGHT: float`.

- [ ] **Step 1: Write the failing tests**

```python
def test_the_market_loss_scores_every_slot() -> None:
    """Unlike unit slots, no market slot is padding -- "trade nothing" is a choice.

    Masking a zero here would teach the model that not trading is unobserved
    rather than chosen, and it chooses it on half of all turns.
    """
    logits = torch.zeros(1, len(MARKET_SLOTS) + 2, len(QUANTITIES))
    labels = torch.zeros(1, len(MARKET_SLOTS) + 2, dtype=torch.int64)

    loss = market_loss(logits, labels)

    assert loss > 0


def test_the_market_loss_is_weighted_against_the_unit_loss() -> None:
    """Roughly four acting units a turn against mostly-zero market slots.

    Summed unweighted, the unit head dominates and the market head learns the
    prior. The weight is a constant so a run's numbers can be reproduced from
    the repository.
    """
    assert 0.0 < MARKET_WEIGHT <= 10.0


def test_both_accuracies_are_reported_separately() -> None:
    """A combined number hides which of the two heads is failing."""
    model = Policy(blocks=1, channels=8).eval()
    loader = DataLoader(_rows(6), batch_size=4)

    metrics = evaluate(model, loader, "cpu")

    assert {"loss/units", "accuracy/units", "loss/market", "accuracy/market"} <= set(
        metrics
    )
```

- [ ] **Step 2: Run and watch them fail**

Run: `uv run pytest tests/learn/test_train.py -q`
Expected: FAIL on `ImportError` for `market_loss`.

- [ ] **Step 3: Implement**

`market_loss` is `cross_entropy` over `(batch * slots, len(QUANTITIES))` with **no `ignore_index`** — every slot is a real decision. Say that in the docstring, contrasting it with `unit_loss`, which masks because most unit slots hold nobody.

Total loss is `unit_loss + MARKET_WEIGHT * market_loss`. Set `MARKET_WEIGHT = 1.0` and record in the docstring that it is a starting point measured against nothing yet — the first run's two accuracy curves are the evidence for changing it.

`evaluate` returns a dict of the four metrics rather than a pair. Update `main` and every call site.

- [ ] **Step 4: Run**

Run: `uv run pytest -q` and `uv run pytest -m slow -q`
Expected: PASS.

- [ ] **Step 5: Prove the guards discriminate**

Add `ignore_index=0` to `market_loss`; `test_the_market_loss_scores_every_slot` must fail. Restore.

- [ ] **Step 6: Train**

```bash
git add -A && git commit -m "feat: train the market head alongside the unit head"
uv run python -m kaggriculture.learn.scripts.train
```

Tracking refuses a dirty tree, so commit first. Report both holdout accuracies, both losses, and the wandb URL.

---

### Task 6: Play, package, and gate

**Files:**

- Modify: `src/kaggriculture/learn/scripts/play.py`, `src/kaggriculture/package.py`
- Test: `tests/learn/test_play.py`, `tests/test_submission.py`

**Interfaces:**

- Consumes: `decode_market` from Task 1, two-headed `forward` from Task 4, the checkpoint from Task 5.

- [ ] **Step 1: Write the failing tests**

```python
def test_the_agent_emits_both_unit_and_market_orders() -> None:
    """An action with an empty market cannot bank a coin; that lost 400 games."""
    action = agent(_observation(), _configuration())

    assert set(action) == {"farmer", "hands", "market"}
    assert isinstance(action["market"], list)


def test_the_submission_carries_the_weights() -> None:
    """A packaged agent that cannot load its checkpoint plays untrained."""
    archive = build_archive(tmp_path)

    names = zipfile.ZipFile(archive).namelist()
    assert any(name.endswith(".pt") for name in names)


def test_the_submission_ships_no_scripts() -> None:
    """The sandbox has no network; a wandb import forfeits the episode on turn 0."""
    archive = build_archive(tmp_path)

    names = zipfile.ZipFile(archive).namelist()
    assert not any("/scripts/" in name for name in names)
```

- [ ] **Step 2: Run and watch them fail**

Run: `uv run pytest tests/learn/test_play.py tests/test_submission.py -q`
Expected: FAIL — the market list is empty and the archive has no `.pt`.

- [ ] **Step 3: Implement**

`play.py` calls both decodes and merges them into one dict. `decode_units` already returns `{"farmer": …, "hands": …, "market": []}`; fill that list from `decode_market`.

`package.py` currently drops `learn/` wholesale through `EXCLUDED`, so it has no mechanism to ship weights at all. Add one: include the checkpoint and the modules `play.py` imports, while still excluding everything under `scripts/` that reaches the network. The submitted agent must import neither `wandb` nor `tracking`.

- [ ] **Step 4: Run**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 5: Gate against the league — this is the deliverable**

```bash
uv run python -m kaggriculture.scripts.evaluate <agent> --track
```

Report, in this order and without tuning toward any of them:

1. Bank per episode, and whether it exceeds the opening 3,000 at all. That single number is what Phase 2 could not move.
2. Win rate against `starter` with its Wilson interval.
3. Win rate against `heuristic-v2` and against `economic_policy`.
4. The league figure over all opponents.

**Gate:** beats `starter`. Rungs beyond that — `heuristic-v2` at 54.6k, `economic_policy` at 118k — are reported, not required; the corpus is dominated by one kernel, so cloning it well means approaching it rather than beating it. State the rung reached and stop.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: play and package the two-headed policy"
```

Record the result in `STRATEGY.md` alongside Phase 2's, including the bank, so the two phases are comparable at a glance.

---

## Self-Review

**Spec coverage.** The action space and its two shapes — Task 1. Corpus-measured verification of the vocabulary — Task 2. Per-item quantity buckets rather than order slots, with the rejected alternative recorded — Task 1, Step 3. Dataset rebuild — Task 3. A second head on the existing trunk reading the pooled features — Task 4. Two loss terms with a stated weighting and separately reported accuracies — Task 5. Serialisation order, the `MAX_ORDERS` cap, play, packaging, and the three-rung gate — Tasks 1 and 6. Shipping weights, which the spec flags as newly necessary — Task 6.

**Type consistency.** `MARKET_SLOTS`, `QUANTITIES`, `MAX_ORDERS`, `HIRE_SLOT` and `LAND_SLOT` are defined in Task 1 and consumed in Tasks 2–6. `encode_market` returns `(1, len(MARKET_SLOTS) + 2)` int64, which Task 3 unbatches into shards and Task 5 flattens for the loss. `forward` returns a two-tuple from Task 4 onward, and Tasks 5 and 6 unpack it. `evaluate` returns a dict from Task 5 onward.

**Known risks, stated rather than solved.**

- Cloning caps at the teacher. Beating `economic_policy` needs RL; this phase produces a policy worth fine-tuning, not a winner.
- Lockstep pricing is unmodelled. The head emits quantities against observed prices while the fill price moves and the opponent sells into the same book on the same turn.
- Aggregating repeated `(verb, item)` pairs changes their interleaving against the opponent's queue. Measured at ~1.8% of order-bearing turns once `HIRE` is excluded, and accepted.
- `MARKET_WEIGHT = 1.0` is a guess until the first run's two curves exist.
