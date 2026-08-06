# Route Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bank more than 118,000 by replaying production routes harvested from the top decile of our own replay corpus, matched to the live board, with the market layer recomputed live.

**Architecture:** A prototype store built offline from the corpus, a retrieval layer that picks the nearest prototype by an identity-free state signature with hysteresis, a realignment step that maps the prototype's unit actions onto our actual units, and `economic_policy` supplying both the market orders and the fallback when nothing matches.

**Tech Stack:** Python stdlib plus the existing package — `kaggriculture.learn.corpus`, `kaggriculture.economic_policy`, `kaggriculture.observation`, `kaggle_environments`. **No torch on the agent path**: importing it costs 10.7 s of the 60 s sandbox overage pool, and this design does not need it.

## Global Constraints

- Run everything with `uv run`. Add dependencies with `uv add`. Never `pip`, never bare `python`.
- `uv run pre-commit run -a` must pass before every commit. Fix type errors; never add ignore comments.
- Google-style docstrings on every public function, class and module. Absolute imports. Never `from __future__ import annotations`.
- `logging.info` / `logging.warning`, never `print()`. `tqdm` for long iterables. Required argparse arguments take no `--` prefix. Prefer CONSTANTS over CLI arguments.
- No defensive code, no unused code paths. Raise a concise error for unsupported cases rather than adding a fallback. The one sanctioned exception is this design's own `economic_policy` fallback, which is a deliberate, counted, reported behaviour — not a silent rescue.
- **Every test guarding load-bearing behaviour must be observed to fail when that behaviour is broken.** Break it, watch it go red, restore it, and report what you broke. This repo has shipped four tests that passed while agreeing with the bug they named.
- **Any corpus statistic must be measured across all archives**, or reported per archive. Measuring one archive and generalising has produced three wrong committed claims in this project.
- The submitted agent must never import `wandb`, `torch`, or anything under `scripts/`.
- Tests reading `/data/kaggriculture/episodes` carry `pytest.mark.slow` and skip cleanly without it; the default run has `addopts = "-m 'not slow'"`.
- Do not mention Claude or AI assistance in commit messages.

## Context

Spec: `docs/superpowers/specs/2026-08-06-route-memory.md`. Ladder measurements: `docs/competition.md`.

The vendored `economic_policy` banks ~118,000 and peaks at 1289.6 on the public leaderboard. The corpus already on disk has a median seat bank of 125,773 and a top decile above 149,120. The strongest published agent scores 2863.9 and is 96.7% memorised route tables.

Phase 3's behaviour clone banked 0 because it re-decides every turn and compounding error destroys it — it hired on nearly every turn where the teacher hires on 14.6% of turns, only in a day's first four hours. A route carries the sequence a per-turn policy cannot represent.

### Interfaces this plan consumes

- `kaggriculture.learn.corpus`: `CORPUS`, `Sample(archive, name, seat, rating)`, `select(archives, min_rating, per_archive)`, `read_manifest`.
- `kaggriculture.economic_policy`: `agent(observation)` returns a full action dict; `_market_actions` supplies market orders alone.
- `kaggriculture.observation`: `Observation.parse(raw)`.
- `kaggriculture.config`: `LEAGUE`, `HarnessConfig`.
- `kaggriculture.scripts.tracking`: refuses to record a run from a dirty tree.

## File Structure

| File                                                              | Responsibility                                                          |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `src/kaggriculture/routes/store.py`                               | **Create.** `Prototype`, harvest, dedupe, save/load.                    |
| `src/kaggriculture/routes/signature.py`                           | **Create.** The identity-free state signature and its distance.         |
| `src/kaggriculture/routes/play.py`                                | **Create.** Retrieval with hysteresis, realignment, the agent callable. |
| `src/kaggriculture/routes/scripts/harvest.py`                     | **Create.** Offline prototype build. Excluded from the archive.         |
| `tests/routes/test_store.py`, `test_signature.py`, `test_play.py` | **Create.**                                                             |
| `src/kaggriculture/scripts/package.py`                            | **Modify.** Ship the prototype store.                                   |
| `main.py`                                                         | **Modify, only if the gate earns it.**                                  |

---

### Task 1: The state signature

**Files:**

- Create: `src/kaggriculture/routes/signature.py`, `src/kaggriculture/routes/__init__.py`
- Test: `tests/routes/test_signature.py`

**Interfaces:**

- Consumes: `kaggriculture.constants` (`CROPS`, `ANIMALS`, `PRODUCTS`, `BOARD_SIZE`, `TURNS_PER_DAY`).
- Produces: `SIGNATURE_FIELDS: tuple[str, ...]`, `signature(observation, seat) -> tuple[float, ...]`, `distance(a, b, day) -> float`.

- [ ] **Step 1: Write the failing tests**

```python
def test_the_signature_ignores_who_the_opponent_is() -> None:
    """A prototype must transfer across matchups or the store is per-opponent.

    Nothing identifying the opponent may enter the signature -- only the public
    shape of the board. Two observations differing solely in the opponent's
    identity fields must produce the same signature.
    """
    ours = empty_observation()
    theirs = empty_observation()
    theirs["farms"][1]["money"] = 99_999.0

    assert signature(ours, 0) == signature(theirs, 0)


def test_the_signature_moves_with_our_own_board() -> None:
    """It is a description of our farm; if our farm changes it must change."""
    before = empty_observation()
    after = empty_observation()
    after["farms"][0]["tiles"][2][3] = _plant("MELON")

    assert signature(before, 0) != signature(after, 0)


def test_distance_is_phase_dominated_early_and_board_dominated_late() -> None:
    """Routes are phase-locked early and diverge in composition later.

    On day 0 an hour apart matters more than a crop apart, because every good
    route is doing the same thing at the same time. By day 25 the reverse holds.
    """
    base = empty_observation()
    an_hour = empty_observation()
    an_hour["hour"] = 3
    a_crop = empty_observation()
    a_crop["farms"][0]["tiles"][2][3] = _plant("MELON")

    early_hour = distance(signature(base, 0), signature(an_hour, 0), day=0)
    early_crop = distance(signature(base, 0), signature(a_crop, 0), day=0)
    late_hour = distance(signature(base, 0), signature(an_hour, 0), day=25)
    late_crop = distance(signature(base, 0), signature(a_crop, 0), day=25)

    assert early_hour > early_crop
    assert late_crop > late_hour


def test_distance_to_itself_is_zero() -> None:
    """A prototype must match its own recorded state exactly."""
    one = signature(empty_observation(), 0)

    assert distance(one, one, day=10) == 0.0
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/routes/test_signature.py -q`
Expected: FAIL on `ImportError`.

- [ ] **Step 3: Implement**

`signature` returns a fixed-width tuple: counts of our tiles by crop and by animal, weed count, locked count, our money, our hand count, our hires today, our unlocked quadrant count, day, hour, and our shed total per product. **Only `farms[seat]` and global state** — nothing from `farms[1 - seat]`.

`distance` is a weighted L1 over the fields, with the phase weight decaying and the composition weight rising as `day / SEASON_DAYS`. Name the weights as constants and say in the docstring that the crossover point is a guess until Task 5 measures it.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/routes/test_signature.py -q`
Expected: PASS.

- [ ] **Step 5: Prove the guards discriminate**

| Break                                               | Test that must go red                                             |
| --------------------------------------------------- | ----------------------------------------------------------------- |
| Include `farms[1 - seat]["money"]` in the signature | `test_the_signature_ignores_who_the_opponent_is`                  |
| Make the phase and composition weights constant     | `test_distance_is_phase_dominated_early_and_board_dominated_late` |

- [ ] **Step 6: Commit**

```bash
git add src/kaggriculture/routes/ tests/routes/
git commit -m "feat: an identity-free signature for matching a board to a route"
```

---

### Task 2: The prototype store

**Files:**

- Create: `src/kaggriculture/routes/store.py`
- Test: `tests/routes/test_store.py`

**Interfaces:**

- Consumes: `signature` from Task 1; `CORPUS`, `Sample`, `select` from `kaggriculture.learn.corpus`.
- Produces: `Prototype` (a pydantic model carrying `bank`, `opponent_bank`, `rating`, `actions: list[dict]`, `signatures: list[tuple[float, ...]]`), `harvest(samples, floor) -> list[Prototype]`, `dedupe(prototypes, tolerance) -> list[Prototype]`, `save(prototypes, path)`, `load(path) -> list[Prototype]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_prototype_keeps_the_action_for_every_acting_turn() -> None:
    """A route is the sequence; a gap in it is a turn the agent cannot replay."""
    prototype = _prototype(turns=719)

    assert len(prototype.actions) == 719
    assert len(prototype.signatures) == 719


def test_harvest_keeps_only_routes_above_the_bank_floor() -> None:
    """The corpus median is 125,773 and our current agent banks ~118,000.

    Replaying a median route would be no better than what we already ship, so
    the floor is the point of the exercise, not a detail.
    """
    kept = harvest([_sample(bank=150_000), _sample(bank=90_000)], floor=149_120)

    assert [p.bank for p in kept] == [150_000]


def test_dedupe_collapses_near_identical_routes_keeping_the_richest() -> None:
    """The corpus is dominated by a few public kernels.

    A thousand copies of one route is a thousand times the storage and none of
    the coverage. When two routes are within tolerance the richer one survives,
    so the store trends toward the best exemplar of each strategy.
    """
    twin_a = _prototype(bank=150_000, seed=1)
    twin_b = _prototype(bank=155_000, seed=1)
    other = _prototype(bank=151_000, seed=99)

    kept = dedupe([twin_a, twin_b, other], tolerance=1e-6)

    assert sorted(p.bank for p in kept) == [151_000, 155_000]


def test_a_saved_store_round_trips() -> None:
    """The store ships inside the submission archive and is read there."""
    original = [_prototype(bank=150_000)]

    save(original, path)
    loaded = load(path)

    assert loaded[0].actions == original[0].actions
    assert loaded[0].signatures == original[0].signatures
```

- [ ] **Step 2: Run and watch them fail**

Run: `uv run pytest tests/routes/test_store.py -q`
Expected: FAIL on `ImportError`.

- [ ] **Step 3: Implement**

`harvest` streams each `Sample`'s episode from its zip — **never extract an archive**, they are ~27 MB per episode — recomputes the signature at every turn from the recorded observation, and keeps the route when the seat's final bank clears `floor`.

Record `opponent_bank` alongside `bank`. A route that banked 158k against a weak opponent is not better than one that banked 150k against a strong one, and Task 5 needs both numbers to say which criterion it used.

`dedupe` compares signature trajectories, keeping the richest of each near-identical group.

Serialise compactly — gzipped JSON. The store ships in the submission archive, so report its size.

- [ ] **Step 4: Run**

Run: `uv run pytest tests/routes/test_store.py -q`
Expected: PASS.

- [ ] **Step 5: Prove the guards discriminate**

Lower the floor to 0 and confirm `test_harvest_keeps_only_routes_above_the_bank_floor` fails. Make `dedupe` keep the first rather than the richest and confirm its test fails. Restore both.

- [ ] **Step 6: Commit**

```bash
git add src/kaggriculture/routes/store.py tests/routes/test_store.py
git commit -m "feat: harvest and dedupe route prototypes from the corpus"
```

---

### Task 3: Harvest the store

**Files:**

- Create: `src/kaggriculture/routes/scripts/harvest.py`, `src/kaggriculture/routes/scripts/__init__.py`

**Interfaces:**

- Consumes: `harvest`, `dedupe`, `save` from Task 2; `select` from `kaggriculture.learn.corpus`.

- [ ] **Step 1: Write the script**

```python
STORE = Path("/data/kaggriculture/routes/prototypes.json.gz")
FLOOR = 149_120.0
TOLERANCE = 1e-3
```

Select from every archive, harvest, dedupe, save. Log how many routes each archive contributed, how many survived the floor, and how many survived dedupe. **Report the per-archive counts, not just the total** — if one archive contributes nothing the store is quietly narrower than it looks.

- [ ] **Step 2: Run it**

```bash
uv run python -m kaggriculture.routes.scripts.harvest
```

Report the store's size on disk, the prototype count, and the bank distribution of what survived.

- [ ] **Step 3: Commit**

```bash
git add src/kaggriculture/routes/scripts/
git commit -m "feat: build the prototype store from every archive"
```

---

### Task 4: Retrieval, realignment and the agent

**Files:**

- Create: `src/kaggriculture/routes/play.py`
- Test: `tests/routes/test_play.py`

**Interfaces:**

- Consumes: `signature`, `distance` from Task 1; `load` from Task 2; `economic_policy.agent` and `_market_actions`.
- Produces: `agent(observation) -> dict`, `RouteAgent` carrying the hysteresis state, `realign(action, prototype_units, our_units) -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
def test_the_market_orders_come_from_the_live_policy_not_the_route() -> None:
    """Prices depend on both players' cumulative sales.

    A replayed SELL quantity is priced for a market that no longer exists, so
    the production plan is replayed and the market is recomputed. This is the
    single most important division in this design.
    """
    prototype = _prototype(market=[["SELL", "WHEAT", 40]])

    action = RouteAgent([prototype]).act(_observation_with_empty_shed())

    assert action["market"] != [["SELL", "WHEAT", 40]]
    assert action["market"] == economic_policy_market(_observation_with_empty_shed())


def test_an_unmatched_board_falls_back_to_the_whole_policy() -> None:
    """The floor of this design is the agent we already ship, not zero."""
    agent = RouteAgent([_prototype(seed=1)])

    action = agent.act(_observation_unlike(seed=1))

    assert agent.fallbacks == 1
    assert action == economic_policy.agent(_observation_unlike(seed=1))


def test_hysteresis_keeps_the_current_route_unless_another_is_clearly_better() -> None:
    """Thrashing between routes yields a sequence no route would ever play.

    Each prototype is individually coherent; alternating between them is not.
    A new prototype must beat the incumbent by a margin, not by a hair.
    """
    agent = RouteAgent([_prototype(seed=1), _prototype(seed=2)])
    agent.act(_observation_matching(seed=1))

    agent.act(_observation_marginally_closer_to(seed=2))

    assert agent.current == 0


def test_realignment_issues_orders_to_the_units_we_actually_have() -> None:
    """A replayed action assumes a unit stands where the prototype's unit stood.

    When the boards diverge the action is a no-op at best and an order to the
    wrong hand at worst. Orders map onto our nearest unit, and orders for units
    we do not have are dropped rather than issued to somebody else.
    """
    prototype_units = [(4, 4), (7, 1), (0, 9)]
    ours = [(4, 4), (7, 2)]

    realigned = realign({"farmer": ["WATER"], "hands": [["DIG"], ["HARVEST"]]},
                        prototype_units, ours)

    assert realigned["farmer"] == ["WATER"]
    assert len(realigned["hands"]) == 1
```

- [ ] **Step 2: Run and watch them fail**

Run: `uv run pytest tests/routes/test_play.py -q`
Expected: FAIL on `ImportError`.

- [ ] **Step 3: Implement**

`RouteAgent.act` computes the signature, finds the nearest prototype, and holds the incumbent unless a challenger beats it by `HYSTERESIS_MARGIN`. It replays that prototype's unit actions for the current turn after realignment, and takes its market orders from `economic_policy`.

Falling back when the nearest distance exceeds `MATCH_THRESHOLD` is deliberate and **counted** — expose `fallbacks` and log the total at episode end. A route memory that falls back 90% of the time is `economic_policy` with extra steps, and only the count will say so.

`MATCH_THRESHOLD` and `HYSTERESIS_MARGIN` are constants with docstrings saying they are unmeasured until Task 5.

- [ ] **Step 4: Run**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 5: Prove the guards discriminate**

| Break                                                         | Test that must go red                                                      |
| ------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Replay the prototype's market orders                          | `test_the_market_orders_come_from_the_live_policy_not_the_route`           |
| Set `MATCH_THRESHOLD` to infinity                             | `test_an_unmatched_board_falls_back_to_the_whole_policy`                   |
| Set `HYSTERESIS_MARGIN` to 0                                  | `test_hysteresis_keeps_the_current_route_unless_another_is_clearly_better` |
| Issue every prototype hand order regardless of our unit count | `test_realignment_issues_orders_to_the_units_we_actually_have`             |

- [ ] **Step 6: Commit**

```bash
git add src/kaggriculture/routes/play.py tests/routes/test_play.py
git commit -m "feat: replay a matched route and price the market live"
```

---

### Task 5: Gate it, and measure the two guesses

**Files:**

- Modify: `src/kaggriculture/scripts/package.py`, `main.py` (only if earned), `STRATEGY.md`

- [ ] **Step 1: Ship the store**

`package.py` excludes `scripts` and, until Phase 3's change, `learn`. Add `routes` minus `routes/scripts`, plus the prototype store. Pin with a test that the archive carries the store and no training scripts, using `tarfile` — **the archive is a gzipped tar, not a zip**.

- [ ] **Step 2: Measure the sandbox budget**

Play one full episode from the **built archive** at two threads. Report import cost, per-turn mean and max, and total overage consumed. No torch here, so this should be far cheaper than Phase 3's agent — confirm that rather than assume it.

- [ ] **Step 3: Gate against the league**

```bash
uv run python -m kaggriculture.scripts.run --agent <agent> --track
```

Report, in this order:

1. **Mean bank per episode**, against `economic_policy`'s ~118,000.
2. **Fallback rate.** If it is above 50%, say so plainly — the result is mostly the fallback's.
3. Win rate against each league opponent with Wilson intervals.
4. The league figure overall.

**Gate:** beats `economic_policy` on bank. Below that there is no reason to ship.

- [ ] **Step 4: Measure the two guessed constants**

`MATCH_THRESHOLD` and `HYSTERESIS_MARGIN` were set blind. Sweep each over three values, holding the other fixed, and report bank and fallback rate for each. **Report the sweep, not just the winner** — a constant chosen from an unreported sweep is indistinguishable from one chosen by luck.

- [ ] **Step 5: Switch `main.py` only if it earns it**

`main.py` serves `economic_policy`. Repoint it only if route memory beats it on bank. Otherwise leave it and say so.

- [ ] **Step 6: Commit and record**

```bash
git add -A
git commit -m "feat: gate route memory against the league"
```

Record the result in `STRATEGY.md` beside Phase 2's and Phase 3's, with the bank, so all three are comparable at a glance.

---

## Self-Review

**Spec coverage.** Prototypes with a bank floor and dedupe — Task 2. Identity-free signature — Task 1. Retrieval with hysteresis — Task 4. Realignment — Task 4. Market recomputed not replayed — Task 4, and the guard for it is the first test in that task. Counted fallback — Task 4. Harvest across every archive — Task 3. The three-rung gate, the fallback rate, the shipping gap, and the two unmeasured constants — Task 5.

**Type consistency.** `signature` returns `tuple[float, ...]` in Task 1 and is consumed by `distance`, `Prototype.signatures` and `RouteAgent.act`. `Prototype` is produced in Task 2 and consumed in Tasks 3 and 4. `agent(observation) -> dict` matches the competition entrypoint that `main.py` already exposes.

**Deliberately deferred.** Prototype decay as the meta moves — the spec names it and Task 5's sweep does not cover it; it needs a second archive-vintage comparison and belongs in its own plan once the store exists. RL fine-tuning. Re-running the Phase 3 clone as the fallback instead of `economic_policy`, which is only worth doing if the retrain beats 118,000.

**Known risks, stated rather than solved.** Replay is open-loop and an opponent that recognises a route can counter it. Selection by final bank rewards luck, which is why `opponent_bank` is recorded. The store's size against the submission archive is unknown until Task 3 runs.
