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


def test_animals_lost_counts_every_disappearance() -> None:
    """A starved animal is a bought asset walking off the farm, and must be visible."""
    steps = [
        [
            {
                "observation": {
                    "day": 0,
                    "farms": [
                        {
                            "tiles": [[{"animal": "COW"}, {"animal": "COW"}]],
                            "money": 3000,
                        }
                    ],
                    "private": {"shed": {}},
                    "market": {"prices": {}},
                },
                "reward": 3000,
            }
        ],
        [
            {
                "observation": {
                    "day": 1,
                    "farms": [{"tiles": [[{"animal": "COW"}, None]], "money": 3000}],
                    "private": {"shed": {}},
                    "market": {"prices": {}},
                },
                "reward": 3000,
            }
        ],
    ]

    season = summarise(steps, player=0)

    assert season.animals_lost == 1


def test_animals_lost_counts_a_loss_masked_by_a_same_turn_placement() -> None:
    """A starvation and a same-turn purchase elsewhere must not cancel out.

    Unit actions, including placing a newly bought animal, apply before the
    end-of-day starvation check within the same recorded turn. A player who
    places one animal the same turn another starves shows no net change in
    herd size, so a net-count implementation would miss the loss entirely.
    """
    steps = [
        [
            {
                "observation": {
                    "day": 0,
                    "farms": [
                        {
                            "tiles": [
                                [
                                    {"kind": "PASTURE", "animal": "COW"},
                                    {"kind": "PASTURE"},
                                    {"kind": "PASTURE", "animal": "COW"},
                                ]
                            ],
                            "money": 3000,
                        }
                    ],
                    "private": {"shed": {}},
                    "market": {"prices": {}},
                },
                "reward": 3000,
            }
        ],
        [
            {
                "observation": {
                    "day": 1,
                    "farms": [
                        {
                            "tiles": [
                                [
                                    {"kind": "PASTURE"},
                                    {"kind": "PASTURE", "animal": "COW"},
                                    {"kind": "PASTURE", "animal": "COW"},
                                ]
                            ],
                            "money": 3000,
                        }
                    ],
                    "private": {"shed": {}},
                    "market": {"prices": {}},
                },
                "reward": 3000,
            }
        ],
    ]

    season = summarise(steps, player=0)

    assert season.animals == [2, 2]
    assert season.animals_lost == 1


def test_summarise_reads_the_right_seat_s_private_state(tmp_path: Path) -> None:
    """Public state comes from index 0, private state from the player's own index.

    At `player=0` those two indices coincide, so every other test in this file
    passes whether or not the distinction is honoured. Seat 1 is the only place
    the module's one non-obvious decision is actually exercised: reading a
    player's shed from index 0 would silently return seat 0's produce for
    everybody, and nothing would raise.
    """
    env = make(ENVIRONMENT, configuration={"episodeSteps": 240, "seed": 9})
    env.run(["baselines/heuristic_v1.py", "baselines/heuristic_v2.py"])
    path = tmp_path / "episode.json"
    path.write_text(json.dumps(env.toJSON()))
    steps = load(path)

    first, second = summarise(steps, player=0), summarise(steps, player=1)

    assert first.bank != second.bank
    assert first.crops != second.crops
    raw = steps[-1][1]["observation"]["private"]["shed"]
    assert second.final_shed == dict(raw)
