# Route Arena and Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Search for a stronger 720-turn route by replaying candidates against a league of decoded opponents on the batched GPU simulator.

**Architecture:** A route is one seat's 720 turns of actions in the engine's own grammar. It is encoded to action tensors once and replayed across a batch of seeds with no host synchronisation, which is the only workload the simulator is fast at. Candidates are scored by seat-swapped win rate against a league, and improved by single-edit hill-climbing.

**Tech Stack:** Python 3.11, PyTorch, `kaggle-environments` 1.32.7, pytest, `uv`.

## Global Constraints

- Engine is `kaggle-environments>=1.32.7`. Do not upgrade it inside this plan.
- `tests/sim` must stay green, including `test_reference_tripwire.py`. If a
  change makes the simulator disagree with the reference, the change is wrong —
  the reference is not to be re-recorded to accommodate it.
- The shipped agent must not import torch. Anything under
  `src/kaggriculture/search/` is offline-only and must be excluded from the
  submission archive (Task 8 enforces this with a test).
- Run everything with `uv run` from the repository root.
- `uv run pre-commit run -a` must pass before every commit. It runs the full
  pytest suite and takes ~2 minutes; run it in the background and do not edit
  files while it runs, because pre-commit stashes unstaged changes and restores
  them afterwards, destroying concurrent edits.
- Stage explicit paths. Never `git add -A`.

---

### Task 1: Route representation and harvesting

A route is the unit everything else operates on, and the cheapest source of
strong ones is the episodes we already have on disk.

**Files:**

- Create: `src/kaggriculture/search/__init__.py`
- Create: `src/kaggriculture/search/route.py`
- Test: `tests/search/__init__.py`, `tests/search/test_route.py`

**Interfaces:**

- Consumes: nothing from earlier tasks.
- Produces: `Route = list[dict]` (720 entries, each with keys `farmer`,
  `hands`, `market`); `from_episode(episode: dict, seat: int) -> Route`;
  `load(path: Path) -> Route`; `save(route: Route, path: Path) -> None`.

- [ ] **Step 1: Write the failing test**

```python
"""Routes are harvested from episodes and survive a round trip."""

import json
import zipfile
from pathlib import Path

import pytest

from kaggriculture.search import route as route_module

ARCHIVE = Path("/data/kaggriculture/episodes/kaggriculture-episodes-2026-08-15.zip")


@pytest.mark.skipif(not ARCHIVE.exists(), reason="replay corpus not on this machine")
def test_a_harvested_route_has_one_entry_per_turn() -> None:
    """An episode's seat is 720 turns of actions in the engine's own grammar."""
    with zipfile.ZipFile(ARCHIVE) as bundle:
        episode = json.loads(bundle.read("93454366.json"))

    harvested = route_module.from_episode(episode, seat=0)

    assert len(harvested) == 720
    assert all(set(turn) == {"farmer", "hands", "market"} for turn in harvested)
    assert any(turn["market"] for turn in harvested)


def test_a_route_survives_a_round_trip(tmp_path: Path) -> None:
    """Saving and loading must not alter a single order."""
    original = [
        {"farmer": ["PASS"], "hands": [["WATER"]], "market": [["SELL", "WHEAT", 3]]}
    ] * 720

    route_module.save(original, tmp_path / "route.json")

    assert route_module.load(tmp_path / "route.json") == original
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/search/test_route.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kaggriculture.search'`

- [ ] **Step 3: Write minimal implementation**

Create `src/kaggriculture/search/__init__.py` holding only a docstring:

```python
"""Offline route search. Never imported by the shipped agent."""
```

Create `src/kaggriculture/search/route.py`:

```python
"""A route: one seat's whole season, in the engine's own action grammar.

The engine reads an action as ``{"farmer": [...], "hands": [[...], ...],
"market": [[...], ...]}``, and a route is 720 of them. Keeping that grammar
rather than inventing one means a route can be harvested from any replay, handed
straight back to the reference engine, and diffed against a competitor's tape
without a translation layer in between.
"""

import json
from pathlib import Path
from typing import Any

Route = list[dict[str, Any]]

TURNS = 720


def from_episode(episode: dict[str, Any], seat: int) -> Route:
    """Return one seat's actions from a decoded episode replay.

    Args:
        episode: A replay as published in the daily episode archives.
        seat: Which player to harvest, 0 or 1.

    Returns:
        One action per turn, normalised so every entry has all three keys.
    """
    if seat not in (0, 1):
        raise ValueError(f"seat must be 0 or 1, got {seat}")
    return [_normalise(step[seat].get("action")) for step in episode["steps"]]


def load(path: Path) -> Route:
    """Return the route stored at ``path``."""
    return [_normalise(turn) for turn in json.loads(Path(path).read_text())]


def save(route: Route, path: Path) -> None:
    """Write ``route`` to ``path`` as JSON."""
    Path(path).write_text(json.dumps(route))


def _normalise(action: dict[str, Any] | None) -> dict[str, Any]:
    """Return one turn's action with every key present and every order a list."""
    action = action or {}
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(order or ["PASS"]) for order in (action.get("hands") or [])],
        "market": [list(order) for order in (action.get("market") or [])],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/search/test_route.py -v`
Expected: PASS (2 passed, or 1 passed 1 skipped without the corpus)

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/search/__init__.py src/kaggriculture/search/route.py tests/search/__init__.py tests/search/test_route.py
git commit -m "feat: harvest routes from episode replays"
```

---

### Task 2: Extract turn encoding so one grammar reader serves both callers

`sim/rollout.scripted_actions` already turns an action dict into simulator
codes, but it does it per batch row inside a loop. The route encoder needs the
same mapping applied once. Extract it rather than writing a second copy: this
project already carries three copies of the price curve and two of them were
wrong.

**Files:**

- Modify: `src/kaggriculture/sim/rollout.py:148-208`
- Test: `tests/sim/test_rollout.py`

**Interfaces:**

- Consumes: nothing from earlier tasks.
- Produces: `encode_turn(action: Mapping[str, Any]) -> TurnActions`, where
  `TurnActions` is a frozen dataclass with `units: tuple[int, ...]` of length
  `MAX_UNITS` and `orders: tuple[tuple[int, int, int], ...]` of length 10, each
  `(order_type, order_item, order_qty)` and `order_item` `-1` when unused.

- [ ] **Step 1: Write the failing test**

```python
def test_encode_turn_maps_the_action_grammar_to_simulator_codes() -> None:
    """One reader of the grammar, used by both the scripted bridge and routes."""
    from kaggriculture.sim.rollout import encode_turn

    encoded = encode_turn(
        {
            "farmer": ["PLANT", "WHEAT"],
            "hands": [["WATER"], ["PASS"]],
            "market": [["SELL", "WHEAT", 3], ["HIRE"]],
        }
    )

    assert encoded.units[0] == UNIT_OPS.index("PLANT:WHEAT")
    assert encoded.units[1] == UNIT_OPS.index("WATER")
    # SELL is order type 1 and WHEAT is index 0 of PRODUCT_NAMES.
    assert encoded.orders[0] == (1, PRODUCT_NAMES.index("WHEAT"), 3)
    # HIRE carries no item, so the item slot stays at the unused sentinel.
    assert encoded.orders[1] == (5, -1, 1)
    # Unused slots are the "no order" code.
    assert encoded.orders[2] == (0, -1, 0)
```

Add `from kaggriculture.sim.state import PRODUCT_NAMES` and
`from kaggriculture.learn.encoding import UNIT_OPS` to the test module's imports
if they are not already present.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sim/test_rollout.py::test_encode_turn_maps_the_action_grammar_to_simulator_codes -v`
Expected: FAIL with `ImportError: cannot import name 'encode_turn'`

- [ ] **Step 3: Write minimal implementation**

In `src/kaggriculture/sim/rollout.py`, add above `scripted_actions`:

```python
@dataclass(frozen=True)
class TurnActions:
    """One turn's actions as simulator codes, independent of batch and seat.

    Attributes:
        units: One op index per unit slot, ``PASS`` where the turn named none.
        orders: ``(order_type, order_item, order_qty)`` per market slot, in the
            queue position the reference reads them in. Type 0 is "no order",
            and ``order_item`` is -1 wherever the verb carries no item.
    """

    units: tuple[int, ...]
    orders: tuple[tuple[int, int, int], ...]


def encode_turn(action: Mapping[str, Any]) -> TurnActions:
    """Return one action dict as simulator codes.

    Args:
        action: A turn in the engine's action grammar.

    Returns:
        The encoded turn. Malformed orders become the "aborted" code 7, which is
        what the reference does with an order it cannot parse.
    """
    units = [UNIT_OPS.index("PASS")] * MAX_UNITS
    units[0] = _unit_index(action.get("farmer", ["PASS"]))
    hands = action.get("hands") or []
    if isinstance(hands, list):
        for unit, unit_action in enumerate(hands[: MAX_UNITS - 1], start=1):
            units[unit] = _unit_index(unit_action)

    orders: list[tuple[int, int, int]] = [(0, -1, 0)] * MAX_MARKET_ORDERS_PER_TURN
    given = action.get("market") or []
    if isinstance(given, list):
        for slot, order in enumerate(given[:MAX_MARKET_ORDERS_PER_TURN]):
            orders[slot] = _encode_order(order)
    return TurnActions(units=tuple(units), orders=tuple(orders))


def _encode_order(order: object) -> tuple[int, int, int]:
    """Return one market order's ``(type, item, quantity)`` codes."""
    if not isinstance(order, (list, tuple)) or not order:
        return (7, -1, 0)
    verb = str(order[0])
    if verb == "HIRE":
        return (5, -1, 1)
    if verb == "BUY_LAND":
        return (6, -1, 1)
    if verb in _MARKET_TYPES and len(order) >= 3:
        kind, catalogue = _MARKET_TYPES[verb]
        item, quantity = str(order[1]), order[2]
        if item not in catalogue or not isinstance(quantity, int):
            return (7, -1, 0)
        return (kind, catalogue.index(item), max(0, min(64, quantity)))
    return (7, -1, 0)
```

Add `MAX_MARKET_ORDERS_PER_TURN` to the imports from
`kaggriculture.sim.engine`, and `dataclass` to the `dataclasses` import.

Then replace the body of `scripted_actions`'s per-row loop so it calls
`encode_turn` instead of re-reading the grammar:

```python
    for batch in range(state.batch_size):
        encoded = encode_turn(opponent(unpack(state, batch, seat)))
        for unit, op in enumerate(encoded.units):
            units[batch, unit] = op
        for slot, (kind, item, quantity) in enumerate(encoded.orders):
            markets.order_type[batch, seat, slot] = kind
            markets.order_item[batch, seat, slot] = item
            markets.order_qty[batch, seat, slot] = quantity
    return units, markets
```

- [ ] **Step 4: Run the whole simulator suite, not just the new test**

Run: `uv run pytest tests/sim -q`
Expected: PASS, 61 passed. `scripted_actions` is exercised by the differential
tests, so this is what proves the extraction changed no behaviour.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/sim/rollout.py tests/sim/test_rollout.py
git commit -m "refactor: one reader of the action grammar, not two"
```

---

### Task 3: Encode a whole route to tensors once

**Files:**

- Create: `src/kaggriculture/search/encode.py`
- Test: `tests/search/test_encode.py`

**Interfaces:**

- Consumes: `route.Route` from Task 1; `sim.rollout.encode_turn` from Task 2.
- Produces: `encode_route(route: Route, device: torch.device | str) -> EncodedRoute`,
  where `EncodedRoute` is a frozen dataclass with `units: torch.Tensor` of shape
  `(720, MAX_UNITS)` int16 and `order_type` / `order_item` / `order_qty` each of
  shape `(720, 10)` and dtypes int8 / int8 / int32.

- [ ] **Step 1: Write the failing test**

```python
"""A route becomes tensors once, not once per batch row."""

import torch

from kaggriculture.learn.encoding import MAX_UNITS, UNIT_OPS
from kaggriculture.search.encode import encode_route
from kaggriculture.sim.state import PRODUCT_NAMES


def test_encode_route_produces_one_row_per_turn() -> None:
    """Shapes and dtypes must match what `step` consumes."""
    route = [{"farmer": ["PASS"], "hands": [], "market": []}] * 720
    route[5] = {"farmer": ["WATER"], "hands": [], "market": [["SELL", "WHEAT", 4]]}

    encoded = encode_route(route, device="cpu")

    assert encoded.units.shape == (720, MAX_UNITS)
    assert encoded.units.dtype == torch.int16
    assert encoded.order_type.shape == (720, 10)
    assert encoded.units[5, 0] == UNIT_OPS.index("WATER")
    assert encoded.order_type[5, 0] == 1
    assert encoded.order_item[5, 0] == PRODUCT_NAMES.index("WHEAT")
    assert encoded.order_qty[5, 0] == 4
    assert encoded.order_type[4, 0] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/search/test_encode.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kaggriculture.search.encode'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Encoding a route into the tensors ``sim.engine.step`` consumes.

A route is the same for every game in a batch, so it is encoded once and the
per-turn rows are broadcast across the batch at replay time. That is the whole
reason a route replays without host synchronisation and a scripted Python
opponent does not.
"""

import dataclasses

import torch

from kaggriculture.learn.encoding import MAX_UNITS
from kaggriculture.search.route import Route
from kaggriculture.sim.rollout import encode_turn


@dataclasses.dataclass(frozen=True)
class EncodedRoute:
    """One route's whole season as device tensors, indexed by turn.

    Attributes:
        units: ``(turns, MAX_UNITS)`` int16 op indices.
        order_type: ``(turns, 10)`` int8 market order kinds.
        order_item: ``(turns, 10)`` int8 catalogue indices, -1 where unused.
        order_qty: ``(turns, 10)`` int32 quantities.
    """

    units: torch.Tensor
    order_type: torch.Tensor
    order_item: torch.Tensor
    order_qty: torch.Tensor


def encode_route(route: Route, device: torch.device | str) -> EncodedRoute:
    """Return ``route`` as per-turn tensors on ``device``.

    Args:
        route: One seat's season in the engine's action grammar.
        device: Where the tensors are wanted.

    Returns:
        The encoded route.
    """
    turns = [encode_turn(action) for action in route]
    units = torch.tensor(
        [turn.units for turn in turns], dtype=torch.int16, device=device
    )
    orders = [[list(order) for order in turn.orders] for turn in turns]
    stacked = torch.tensor(orders, dtype=torch.int64, device=device)
    return EncodedRoute(
        units=units.view(len(turns), MAX_UNITS),
        order_type=stacked[:, :, 0].to(torch.int8),
        order_item=stacked[:, :, 1].to(torch.int8),
        order_qty=stacked[:, :, 2].to(torch.int32),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/search/test_encode.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/search/encode.py tests/search/test_encode.py
git commit -m "feat: encode a route to simulator tensors once"
```

---

### Task 4: Replay two routes on the reference engine, and prove it

**This task was rewritten after its first attempt returned BLOCKED.** The
original built the arena on the batched simulator and its fidelity test refused
to pass: replaying a recorded episode's own actions on its own seed banked
`[13342, 13078]` against the recorded `[71961, 73382]`, matching turn by turn to
turn 42 and diverging at a `PLACE FERTILIZER 3`.

The cause is structural, not a bug. The simulator's `unit_actions` tensor holds
one op index per unit and has **no quantity lane**; `learn/encoding.py:660` sets
`TRANSFER_QUANTITY = 1` and its own docstring calls that lossy on purpose. It is
the right call for the RL policy's action space, which never moves more than one
item, and every `tests/sim` differential test passes because they all drive the
simulator from exactly that subset. The reference engine reads
`n = int(action[2])` for `PICKUP`/`PLACE`, and in top episode 93454366, **283 of
362 PICKUP/PLACE actions (78%) carry a quantity other than 1**.

So the simulator faithfully replays the RL action space, and a recorded route is
not in it. Rather than add a quantity lane — which cascades through
`sim/units.py`, `sim/rollout.py`, `step`'s signature, `learn/encoding.py`,
`learn/mask.py` and every differential test — the arena moves to the reference
engine. It costs throughput we do not need: a candidate scored against a
five-route league over 64 seat-swapped seeds runs in about a minute across this
machine's cores, while a single ladder verdict costs ~15 hours. And it removes
fidelity risk entirely, because the arena then measures on the engine that
scores the competition.

**Two defects in the previous brief were found and are already corrected below.**
The episode seed lives at `episode["info"]["seed"]`; `episode["configuration"]["seed"]`
is null in the archives. And indexing is avoided altogether here by having the
replay agent read `observation["step"]` rather than counting its own turns, which
is self-aligning against the recorded route no matter which convention the engine
uses.

**Files:**

- Create: `src/kaggriculture/search/arena.py`
- Modify: `src/kaggriculture/sim/rollout.py` (`_unit_index`)
- Delete: `src/kaggriculture/search/encode.py`, `tests/search/test_encode.py`
- Test: `tests/search/test_arena.py`, `tests/sim/test_rollout.py`

**Also make the simulator's domain boundary loud.** The batched simulator's
design spec §2.5 says it "raises on ... an input outside that domain", and it
does not: `_unit_index` drops `action[2]`, so `PLACE FERTILIZER 3` is silently
executed as a transfer of one. Silence there cost a full agent context on a
turn-by-turn diff before the cause surfaced. `_unit_index` must raise
`ValueError` naming the verb, item and quantity when a `PICKUP` or `PLACE`
carries an explicit quantity other than 1, with a test that pins it. This is the
project's stated rule — raise on unsupported cases rather than add fallback
behaviour — and it costs nothing: `learn/encoding.py` emits
`TRANSFER_QUANTITY = 1` and every in-domain caller already satisfies it.

`encode_route` was written for the simulator arena and nothing calls it under
this design. It goes rather than lingering as an unused path; git history keeps
it if a quantity lane is ever built. `encode_turn` in `sim/rollout.py` stays —
`scripted_actions` still uses it.

**Interfaces:**

- Consumes: `route.Route` and `route.from_episode` from Task 1.
- Produces: `play(seat_zero: Route, seat_one: Route, seeds: Sequence[int],
workers: int | None = None) -> list[tuple[int, int]]`, one `(bank_seat_zero,
bank_seat_one)` per seed, in the order given.

- [ ] **Step 1: Write the failing test**

```python
"""Replaying a real episode's own actions must reproduce its own result."""

import json
import zipfile
from pathlib import Path

import pytest

from kaggriculture.search.arena import play
from kaggriculture.search.route import from_episode

ARCHIVE = Path("/data/kaggriculture/episodes/kaggriculture-episodes-2026-08-15.zip")


@pytest.mark.skipif(not ARCHIVE.exists(), reason="replay corpus not on this machine")
def test_replaying_an_episode_reproduces_its_recorded_banks() -> None:
    """The engine is deterministic given seed and actions, so this is exact.

    This is the whole warrant for the arena. It covers the action grammar, seat
    assignment, market queue-position coupling and turn alignment in one
    assertion, and it is the difference between an arena that is faithful and
    one that is merely plausible. Do not weaken it to a tolerance.
    """
    with zipfile.ZipFile(ARCHIVE) as bundle:
        episode = json.loads(bundle.read("93454366.json"))
    seed = int(episode["info"]["seed"])
    expected = [(int(episode["rewards"][0]), int(episode["rewards"][1]))]

    banks = play(from_episode(episode, seat=0), from_episode(episode, seat=1), [seed])

    assert banks == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/search/test_arena.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kaggriculture.search.arena'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Replaying routes against each other on the reference engine.

The engine is deterministic given a seed and both seats' actions, so a route
replayed here reproduces the episode it was harvested from exactly. That is the
property the arena rests on, and it is why the arena runs on the reference
engine rather than on the batched simulator: the simulator's action encoding
carries no quantity for PICKUP/PLACE, which 78% of a real route's transfers use.

Episodes are independent, so seeds are played across a process pool. Each worker
runs one whole episode; the routes are plain lists of dicts and pickle without
help.
"""

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor

from kaggle_environments import make

from kaggriculture.constants import ENVIRONMENT, EPISODE_STEPS
from kaggriculture.search.route import Route


def play(
    seat_zero: Route,
    seat_one: Route,
    seeds: Sequence[int],
    workers: int | None = None,
) -> list[tuple[int, int]]:
    """Play two routes against each other across ``seeds``.

    Args:
        seat_zero: The route to play in seat 0.
        seat_one: The route to play in seat 1.
        seeds: One episode seed per game.
        workers: Processes to spread the games over, or None for the default.

    Returns:
        One ``(bank_seat_zero, bank_seat_one)`` per seed, in the order given.
    """
    work = [(seat_zero, seat_one, int(seed)) for seed in seeds]
    if not work:
        return []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_one, work))


def _one(work: tuple[Route, Route, int]) -> tuple[int, int]:
    """Play a single episode. Runs in a subprocess."""
    seat_zero, seat_one, seed = work
    environment = make(
        ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS, "seed": seed}
    )
    environment.run([_replay(seat_zero), _replay(seat_one)])
    final = environment.steps[-1]
    return (int(final[0].reward or 0), int(final[1].reward or 0))


def _replay(route: Route):
    """Return an agent that plays ``route``, indexed by the engine's own clock.

    Reading ``observation["step"]`` rather than counting turns internally is what
    keeps the replay aligned with the recording: the route was harvested by step
    index, so it is replayed by step index, and no assumption about whether an
    action belongs to the state before or after it can creep in.
    """

    def agent(observation, configuration=None):
        step = int(observation["step"])
        return route[step] if step < len(route) else {"farmer": ["PASS"], "hands": [], "market": []}

    return agent
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/search/test_arena.py -v`
Expected: PASS — banks exactly equal to the episode's recorded `rewards`.

If it fails, diagnose against a single seed by comparing `environment.steps[t]`
with the recorded episode's `steps[t]` and finding the first turn whose farm
state differs. Do not weaken the assertion, and do not compare only one seat. If
it cannot be made to pass, stop and report what diverged and when.

- [ ] **Step 5: Delete the simulator encoder and commit**

```bash
git rm src/kaggriculture/search/encode.py tests/search/test_encode.py
git add src/kaggriculture/search/arena.py tests/search/test_arena.py
```

Commit with a message written to a file and `git commit -F`, explaining why the
arena runs on the reference engine and what the fidelity test establishes. Do
not mention Claude.

---

### Task 5: Seat-swapped win rate against a league

**Files:**

- Modify: `src/kaggriculture/search/arena.py`
- Test: `tests/search/test_arena.py`

**Interfaces:**

- Consumes: `play(seat_zero, seat_one, seeds, workers=None) -> list[tuple[int, int]]`
  from Task 4.
- Produces: `evaluate(candidate: Route, league: Mapping[str, Opponent],
seeds: Sequence[int], workers: int | None = None) -> dict[str, float]`,
  returning one win rate per league member over `2 * len(seeds)` games.
- Also produces: `Opponent = Route | str`, and `play` extended to accept a `str`
  on either side.

**Policy opponents, which the reference engine makes possible.** The earlier
simulator design could only host routes, and the spec called that out as the
arena's blind spot: it could never see an opponent that reacts to us. The
reference engine runs _agents_, so an opponent may now be either a route or a
path to an agent file — `src/kaggriculture/economic_policy.py`,
`src/kaggriculture/boatlee_v14_policy.py`, or a decoded competitor kernel.
`kaggle_environments`' `env.run` already accepts a file path in the same
position as a callable, so this costs a type union and a branch.

Keep both kinds. A tape is what most of the field actually submits; a policy is
the only thing that can respond to what our candidate does, and a route that only
beats frozen recordings is the overfitting the spec's gate 3 exists to catch.

- [ ] **Step 1: Write the failing test**

```python
def test_a_route_against_itself_scores_exactly_half() -> None:
    """Seat-swapping makes the mirror exactly 0.5, which catches a seat bug.

    A win rate that is not 0.5 here means one seat is being favoured -- by an
    unswapped replay, a seed reused between the two orderings, or ties counted
    for one side only.
    """
    route = [{"farmer": ["PASS"], "hands": [], "market": []}] * 720

    scores = evaluate(route, {"mirror": route}, seeds=[1, 2, 3, 4])

    assert scores["mirror"] == pytest.approx(0.5)


def test_a_policy_can_stand_in_for_a_route() -> None:
    """An opponent may be an agent path, so the league can hold reacting play.

    A route is a recording and cannot respond to us. `economic_policy` can, and
    scoring against something that reacts is the only in-arena check on a
    candidate that has merely learned to beat frozen tapes.
    """
    route = [{"farmer": ["PASS"], "hands": [], "market": []}] * 720

    scores = evaluate(
        route, {"econ": "src/kaggriculture/economic_policy.py"}, seeds=[7]
    )

    assert 0.0 <= scores["econ"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/search/test_arena.py -v`
Expected: FAIL with `NameError: name 'evaluate' is not defined`

- [ ] **Step 3: Write minimal implementation**

Extend `play` so either side may be an agent path, and add `evaluate` beneath it:

```python
Opponent = Route | str


def _side(opponent: Opponent):
    """Return what ``env.run`` should be handed for one seat.

    A ``str`` is a path to an agent file, which the engine loads itself; a route
    is replayed by our own closure. Both occupy the same position in ``run``.
    """
    return opponent if isinstance(opponent, str) else _replay(opponent)


def evaluate(
    candidate: Route,
    league: Mapping[str, Opponent],
    seeds: Sequence[int],
    workers: int | None = None,
) -> dict[str, float]:
    """Return the candidate's win rate against each league member.

    Every seed is played twice, once with the candidate in each seat, because
    the seats are not symmetric: they hold different quadrants and their market
    orders pair by queue position. Scoring one ordering only would measure the
    seat as much as the route.

    A tie counts as half a win, which is what the ladder's rating does with it.

    Args:
        candidate: The route being scored.
        league: Opponents by name; each a route or a path to an agent file.
        seeds: Episode seeds; each is played in both orderings.
        workers: Processes to spread games over, or None for the default.

    Returns:
        One win rate per league member, over ``2 * len(seeds)`` games each.
    """
    scores = {}
    for name, opponent in league.items():
        first = play(candidate, opponent, seeds, workers)
        second = play(opponent, candidate, seeds, workers)
        ours = [a for a, _ in first] + [b for _, b in second]
        theirs = [b for _, b in first] + [a for a, _ in second]
        wins = sum(
            1.0 if us > them else 0.5 if us == them else 0.0
            for us, them in zip(ours, theirs, strict=True)
        )
        scores[name] = wins / len(ours)
    return scores
```

`play`'s signature widens to `seat_zero: Opponent, seat_one: Opponent`, and its
worker builds each seat with `_side(...)`. Add
`from collections.abc import Mapping, Sequence` to the imports.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/search/test_arena.py -v`
Expected: PASS, all three tests including the Task 4 fidelity test.

The mirror test must be **exactly** 0.5. If it is not, a seat is being favoured
and the cause is in the swap, not in the tolerance — do not relax the assertion.

- [ ] **Step 5: Commit**

Stage `src/kaggriculture/search/arena.py` and `tests/search/test_arena.py`.
Commit with `git commit -F <file>`, explaining why seat-swapping is required and
why the league holds both tapes and policies. Do not mention Claude.

---

### Task 6: Single-edit mutations

Every mutation changes exactly one thing inside one day. With a noisy binary
fitness, a candidate that differs from its parent in several places cannot tell
us which edit paid.

**Files:**

- Create: `src/kaggriculture/search/mutate.py`
- Test: `tests/search/test_mutate.py`

**Interfaces:**

- Consumes: `route.Route` from Task 1.
- Produces: `mutate(route: Route, rng: random.Random) -> tuple[Route, str]`
  returning a new route and a human-readable description of the single edit.

- [ ] **Step 1: Write the failing test**

```python
"""A mutation changes one thing, and leaves a route the engine can still read."""

import random

from kaggriculture.search.mutate import mutate


def _route() -> list[dict]:
    route = [{"farmer": ["PASS"], "hands": [["PASS"]], "market": []} for _ in range(720)]
    for turn in range(0, 720, 7):
        route[turn]["market"] = [["SELL", "WHEAT", 3]]
    return route


def test_a_mutation_changes_exactly_one_turn() -> None:
    """Anything more and a fitness difference cannot be attributed."""
    original = _route()

    for seed in range(40):
        mutated, description = mutate(original, random.Random(seed))

        differing = [i for i, (a, b) in enumerate(zip(original, mutated, strict=True)) if a != b]
        assert len(differing) == 1, description
        assert len(mutated) == 720
        assert description


def test_the_mutation_set_can_reach_the_hinge_products() -> None:
    """The search must be able to plant a tomato and buy a goose.

    The town drains 264 tomato a season into a market the whole field supplies
    with 15 units, and capturing that is the reason this search exists. A
    mutation set that cannot express `PLANT:TOMATO` or `BUY_SEED TOMATO` would
    hill-climb forever without reaching it and return a negative that looked
    honest. This asserts the reachability directly rather than trusting the
    constant lists to stay right.
    """
    route = _route()
    route[3]["market"] = [["BUY_SEED", "WHEAT", 2], ["BUY_ANIMAL", "COW", 1]]
    seen = set()

    for seed in range(400):
        mutated, _ = mutate(route, random.Random(seed))
        for turn in mutated:
            seen.add(tuple(turn["farmer"]))
            for order in turn["market"]:
                seen.add(tuple(order[:2]))

    # Engine grammar specifically: ["PLANT", "TOMATO"], not our ["PLANT:TOMATO"].
    assert ("PLANT", "TOMATO") in seen
    assert ("BUY_SEED", "TOMATO") in seen
    assert ("BUY_ANIMAL", "GOOSE") in seen


def test_a_mutation_does_not_alter_its_parent() -> None:
    """The search keeps the incumbent; an in-place edit would corrupt it."""
    original = _route()
    before = [dict(turn) for turn in original]

    mutate(original, random.Random(0))

    assert original == before
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/search/test_mutate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kaggriculture.search.mutate'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Single-edit mutations over a route.

Four edits, each confined to one turn: retime a market order within its day,
resize one, retarget one, or replace a unit's op. Compound edits are excluded
deliberately -- the fitness is a win rate over a few hundred noisy games, and a
candidate differing in several places tells us nothing about which edit paid.

Legality is not checked here. The engine masks an op the board does not permit
and partially fills an order larger than the shed holds, so an illegal edit
simply scores badly and the search discards it. Checking here would mean a
second implementation of the rules, which is the defect this project has hit
most often.
"""

import copy
import random

from kaggriculture.search.route import Route

TURNS_PER_DAY = 24
SELLABLE = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
# Engine grammar, not our internal op names. `encode_turn` would accept
# ["PLANT:TOMATO"] because that is how UNIT_OPS spells it, but the reference
# engine reads ["PLANT", "TOMATO"] and a route has to replay there -- the final
# gate is on the reference engine, not on the simulator.
#
# The PLANT entries are in the pool on purpose. Without them the search cannot
# put a tomato in the ground, and the whole reason for building it is that the
# town drains 264 tomato a season into a market the entire field supplies with
# 15 units. A mutation set that cannot represent the answer would hill-climb
# forever and return an honest-looking negative.
UNIT_OPS_POOL = (
    ("PASS",), ("WATER",), ("HARVEST",), ("COLLECT_FERTILIZER",),
    ("NORTH",), ("SOUTH",), ("EAST",), ("WEST",),
    ("PLANT", "WHEAT"), ("PLANT", "CARROT"), ("PLANT", "TOMATO"),
    ("PLANT", "STRAWBERRY"), ("PLANT", "MELON"),
)


def mutate(route: Route, rng: random.Random) -> tuple[Route, str]:
    """Return a copy of ``route`` with exactly one turn changed.

    Args:
        route: The parent route, left untouched.
        rng: The stream deciding the edit, so a run is reproducible.

    Returns:
        The mutated route and a description of the edit.
    """
    mutated = copy.deepcopy(route)
    with_orders = [turn for turn, action in enumerate(route) if action["market"]]
    edits = ["unit"]
    if with_orders:
        edits += ["retime", "resize", "retarget"]
    choice = rng.choice(edits)

    if choice == "unit":
        turn = rng.randrange(len(route))
        op = list(rng.choice(UNIT_OPS_POOL))
        mutated[turn]["farmer"] = op
        return mutated, f"turn {turn}: farmer -> {' '.join(op)}"

    turn = rng.choice(with_orders)
    slot = rng.randrange(len(mutated[turn]["market"]))
    order = list(mutated[turn]["market"][slot])

    if choice == "resize" and len(order) >= 3:
        quantity = max(1, min(64, int(order[2]) + rng.choice((-4, -2, -1, 1, 2, 4))))
        order[2] = quantity
        mutated[turn]["market"][slot] = order
        return mutated, f"turn {turn} slot {slot}: quantity -> {quantity}"

    # Retargeting covers BUY_SEED and BUY_ANIMAL as well as SELL, for the same
    # reason the PLANT ops are in the pool: buying tomato seed and buying a
    # goose are the two market edits that make the hinge products reachable at
    # all, and a SELL-only retarget can never emit either.
    catalogues = {"SELL": SELLABLE, "BUY_SEED": CROPS, "BUY_ANIMAL": ANIMALS}
    if choice == "retarget" and len(order) >= 3 and order[0] in catalogues:
        item = rng.choice(catalogues[order[0]])
        order[1] = item
        mutated[turn]["market"][slot] = order
        return mutated, f"turn {turn} slot {slot}: item -> {item}"

    # Retiming is a reorder within the turn, not a move to another turn. Moving
    # an order across turns would change two turns at once, and then a fitness
    # difference could not be attributed to one edit. Queue position is what
    # pairs the two seats' orders in the reference engine, so a reorder inside
    # one turn is a real edit rather than a no-op.
    slots = mutated[turn]["market"]
    other = rng.randrange(len(slots))
    slots[slot], slots[other] = slots[other], slots[slot]
    return mutated, f"turn {turn}: slots {slot} and {other} swapped"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/search/test_mutate.py -v`
Expected: PASS, both tests.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/search/mutate.py tests/search/test_mutate.py
git commit -m "feat: single-edit route mutations"
```

---

### Task 7: Build the league

The searcher needs opponents on disk before it can run, and the two sources are
already available: boatlee's V16-RC5 tape, decoded from its published notebook,
and routes harvested from top-rated episodes in the daily archives.

**Files:**

- Create: `src/kaggriculture/search/scripts/__init__.py`
- Create: `src/kaggriculture/search/scripts/build_league.py`
- Test: `tests/search/test_build_league.py`

**Interfaces:**

- Consumes: `route.from_episode` and `route.save` from Task 1.
- Produces: `harvest(archive: Path, count: int, output: Path) -> list[Path]`,
  writing one JSON route per harvested seat and returning the paths written.

- [ ] **Step 1: Write the failing test**

```python
"""The league is built from the strongest seats of the strongest episodes."""

from pathlib import Path

import pytest

from kaggriculture.search.scripts.build_league import harvest

ARCHIVE = Path("/data/kaggriculture/episodes/kaggriculture-episodes-2026-08-15.zip")


@pytest.mark.skipif(not ARCHIVE.exists(), reason="replay corpus not on this machine")
def test_harvest_writes_one_route_per_requested_opponent(tmp_path: Path) -> None:
    """Each harvested file is a 720-turn route the arena can load."""
    from kaggriculture.search.route import load

    written = harvest(ARCHIVE, count=2, output=tmp_path)

    assert len(written) == 2
    for path in written:
        assert len(load(path)) == 720
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/search/test_build_league.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kaggriculture.search.scripts'`

- [ ] **Step 3: Write minimal implementation**

Create `src/kaggriculture/search/scripts/__init__.py` with a docstring, then
`src/kaggriculture/search/scripts/build_league.py`:

```python
"""Harvest league opponents from the published episode archives.

The winning seat of a top-rated episode is a route by construction: the archive
records every action it took. Harvesting several days rather than one is
deliberate -- Kaito Fukami's own advice is that optimising against the latest
Top-30 alone loses to older meta generations still active on the ladder.
"""

import argparse
import json
import logging
import zipfile
from pathlib import Path

from kaggriculture.learn.corpus import read_manifest
from kaggriculture.search.route import from_episode, save

LOGGER = logging.getLogger(__name__)
ENGINE = "1.32.7"


def harvest(archive: Path, count: int, output: Path) -> list[Path]:
    """Write the winning seats of the archive's best episodes as routes.

    Args:
        archive: A daily episode ``.zip``.
        count: How many opponents to write.
        output: Directory to write them into.

    Returns:
        The paths written, best episode first.
    """
    output.mkdir(parents=True, exist_ok=True)
    rows = sorted(read_manifest(archive), key=lambda row: -row.avg_score)
    written: list[Path] = []
    with zipfile.ZipFile(archive) as bundle:
        for row in rows:
            if len(written) >= count:
                break
            episode = json.loads(bundle.read(f"{row.episode_id}.json"))
            if str(episode.get("module_version")) != ENGINE:
                continue
            rewards = episode.get("rewards") or [0, 0]
            seat = 0 if rewards[0] >= rewards[1] else 1
            path = output / f"{archive.stem}-{row.episode_id}-seat{seat}.json"
            save(from_episode(episode, seat), path)
            written.append(path)
            LOGGER.info("%s rating %.0f bank %s", path.name, row.avg_score, rewards[seat])
    return written


def main() -> None:
    """Harvest a league from one or more daily archives."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="directory for league routes")
    parser.add_argument("archives", type=Path, nargs="+", help="daily episode zips")
    parser.add_argument("--per-archive", type=int, default=3)
    arguments = parser.parse_args()
    for archive in arguments.archives:
        harvest(archive, arguments.per_archive, arguments.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/search/test_build_league.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/search/scripts/__init__.py src/kaggriculture/search/scripts/build_league.py tests/search/test_build_league.py
git commit -m "feat: harvest a league of opponents from the episode archives"
```

Note for whoever runs it: boatlee's V16-RC5 tape is a second source, decoded
from its published notebook by base85 + zlib into a 720-entry `_ACTIONS` list,
which is already a route and needs only `save`.

---

### Task 8: The hill-climb loop, and keeping it out of the submission

`package.py` copies the package tree with `EXCLUDED` applied at every directory
level. A new `search` package would ship into the submission unless excluded,
and it imports torch — which the packaging notes measure at 10.7 s of the 60 s
overage pool. A tuple entry alone is not evidence; the test builds the archive
and looks inside it.

**Files:**

- Create: `src/kaggriculture/search/scripts/hillclimb.py`
- Modify: `src/kaggriculture/scripts/package.py:44`
- Test: `tests/test_package.py`

**Interfaces:**

- Consumes: `arena.evaluate` from Task 5, `mutate.mutate` from Task 6,
  `route.load` / `route.save` from Task 1, and a league directory built by
  Task 7.
- Produces: a command-line entry point; no importable interface later tasks rely
  on.

- [ ] **Step 1: Write the failing test**

```python
def test_the_archive_does_not_ship_the_search_package(tmp_path: Path) -> None:
    """Search code imports torch, which costs 10.7s of the 60s overage pool."""
    import tarfile

    from kaggriculture.scripts.package import build

    archive = build(tmp_path / "submission.tar.gz")

    with tarfile.open(archive) as bundle:
        names = bundle.getnames()
    assert not [name for name in names if "/search/" in name or name.endswith("/search")]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_package.py::test_the_archive_does_not_ship_the_search_package -v`
Expected: FAIL — the archive contains `kaggriculture/search/...`

- [ ] **Step 3: Write minimal implementation**

In `src/kaggriculture/scripts/package.py:44`, add `"search"` to the excluded
names and extend the comment above it to say why:

```python
EXCLUDED = shutil.ignore_patterns(
    "__pycache__", "scripts", "learn", "search", "*.pt", STORE.name
)
```

Then create `src/kaggriculture/search/scripts/hillclimb.py`:

```python
"""Hill-climb a route against a league, one edit at a time.

Accepts a candidate only when its mean league win rate beats the incumbent's by
more than the standard error of the comparison, so a run does not walk uphill on
noise. Every accepted route is written out, because the interesting artefact is
the sequence of edits that paid, not only the final tape.
"""

import argparse
import json
import logging
import random
from pathlib import Path

from kaggriculture.search.arena import evaluate
from kaggriculture.search.mutate import mutate
from kaggriculture.search.route import load, save

LOGGER = logging.getLogger(__name__)
SEEDS = tuple(range(500_000, 500_064))


def main() -> None:
    """Run the hill-climb until the candidate budget is exhausted."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed_route", type=Path, help="route to start from")
    parser.add_argument("league", type=Path, help="directory of opponent routes")
    parser.add_argument("output", type=Path, help="directory for accepted routes")
    parser.add_argument("--candidates", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rng", type=int, default=0)
    arguments = parser.parse_args()

    league = {path.stem: load(path) for path in sorted(arguments.league.glob("*.json"))}
    if not league:
        raise SystemExit(f"no opponent routes in {arguments.league}")
    incumbent = load(arguments.seed_route)
    arguments.output.mkdir(parents=True, exist_ok=True)

    scores = evaluate(incumbent, league, SEEDS, arguments.device)
    best = sum(scores.values()) / len(scores)
    LOGGER.info("incumbent %.4f %s", best, json.dumps(scores))

    rng = random.Random(arguments.rng)
    for candidate in range(arguments.candidates):
        mutated, description = mutate(incumbent, rng)
        scores = evaluate(mutated, league, SEEDS, arguments.device)
        mean = sum(scores.values()) / len(scores)
        # The standard error of a win rate over this many games, doubled because
        # incumbent and candidate are both estimates.
        games = 2 * len(SEEDS) * len(league)
        threshold = 2 * (0.25 / games) ** 0.5
        if mean > best + threshold:
            incumbent, best = mutated, mean
            save(incumbent, arguments.output / f"accepted-{candidate:04d}.json")
            LOGGER.info("accepted %.4f (%s) %s", mean, description, json.dumps(scores))
        else:
            LOGGER.info("rejected %.4f (%s)", mean, description)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test and the full suite**

Run: `uv run pytest tests/test_package.py -v`
Expected: PASS

Run: `uv run pre-commit run -a` in the background, and wait for it.
Expected: all hooks pass, including the full pytest suite.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/search/scripts/hillclimb.py src/kaggriculture/scripts/package.py tests/test_package.py
git commit -m "feat: hill-climb routes, and keep the searcher out of the archive"
```

---

## After the plan

The plan ends with a searcher, not a submission. Before anything is submitted,
run the spec's §5 gate: **128 seat-swapped games on the reference engine**
against the agent we currently serve, with the win rate's 95% interval excluding
0.5. The simulator is fidelity-proven and Task 4 proves the arena on top of it,
but the thing we submit is measured on the thing that scores it.

Two failure modes worth watching for, both of which have already happened once
on this project today:

- **A candidate that wins in the arena and loses on the engine.** That means the
  arena and the reference have diverged, and Task 4's test should be re-run on a
  fresh episode before trusting any further search output.
- **A search that accepts steadily and improves nothing.** The threshold in
  `hillclimb` guards against walking uphill on noise, but it assumes the league
  win rates are independent; they are not, since one seed set is reused. If
  acceptances stop correlating with the gate result, widen the seed set before
  widening the threshold.
