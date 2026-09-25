"""Verify season timing, observation visibility, defaults, and final rewards."""

from __future__ import annotations

from _common import PASS, assert_exact_engine, emit, new_env


def config_defaults() -> dict[str, object]:
    env = new_env(seed=1)
    names = (
        "episodeSteps",
        "actTimeout",
        "boardSize",
        "startingMoney",
        "maxMarketOrdersPerTurn",
        "turnsPerDay",
        "shedCapacity",
        "weedSpawnChance",
        "townShopUnlockInterval",
        "townShopSellInterval",
        "townCenterSellInterval",
        "farmHandCostMult",
    )
    result = {name: getattr(env.configuration, name) for name in names}
    assert result == {
        "episodeSteps": 720,
        "actTimeout": 1,
        "boardSize": 10,
        "startingMoney": 3000,
        "maxMarketOrdersPerTurn": 10,
        "turnsPerDay": 24,
        "shedCapacity": 100,
        "weedSpawnChance": 0.005,
        "townShopUnlockInterval": 3,
        "townShopSellInterval": 4,
        "townCenterSellInterval": 24,
        "farmHandCostMult": 1,
    }
    return result


def timing_boundaries() -> dict[str, object]:
    env = new_env(seed=1, weedSpawnChance=0)
    env.reset(2)
    samples = [
        {
            "recorded_step": env.state[0].observation.step,
            "day": env.state[0].observation.day,
            "hour": env.state[0].observation.hour,
        }
    ]
    for _ in range(25):
        env.step([PASS, PASS])
        step = env.state[0].observation.step
        if step in (1, 23, 24, 25):
            obs = env.state[0].observation
            samples.append({"recorded_step": step, "day": obs.day, "hour": obs.hour})
    assert samples == [
        {"recorded_step": 0, "day": 0, "hour": 0},
        {"recorded_step": 1, "day": 0, "hour": 1},
        {"recorded_step": 23, "day": 0, "hour": 23},
        {"recorded_step": 24, "day": 1, "hour": 0},
        {"recorded_step": 25, "day": 1, "hour": 1},
    ]
    return {"boundaries": samples, "end_of_day_resulting_recorded_steps": [24, 48, 72]}


def end_of_day_reset() -> dict[str, object]:
    env = new_env(seed=1, weedSpawnChance=0)
    env.reset(2)
    env.step(
        [
            {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]]},
            PASS,
        ]
    )
    farm = env.state[0].observation.farms[0]
    private = env.state[0].observation.private
    private.inventories[0]["WHEAT"] = 2
    assert len(farm.hands) == 1 and farm.hires_today == 1
    for _ in range(23):
        env.step([PASS, PASS])
    obs = env.state[0].observation
    farm = obs.farms[0]
    private = obs.private
    assert (obs.step, obs.day, obs.hour) == (24, 1, 0)
    assert list(farm.farmer) == [4, 4]
    assert len(farm.hands) == 0 and farm.hires_today == 0
    assert len(private.inventories) == 1 and dict(private.inventories[0]) == {}
    assert private.shed.WHEAT == 2
    return {
        "transition_at_recorded_step": 24,
        "farmer_spawn": list(farm.farmer),
        "hands_after": len(farm.hands),
        "hires_today_after": farm.hires_today,
        "inventories_after": len(private.inventories),
        "auto_dropped_wheat": private.shed.WHEAT,
    }


def visibility() -> dict[str, object]:
    env = new_env(seed=1, weedSpawnChance=0)
    env.reset(2)
    o0 = env.state[0].observation
    o1 = env.state[1].observation
    assert o0.player == 0 and o1.player == 1
    # The framework serializes separate wrappers, but the shared values agree
    # and interpreter mutations are copied to both seats after every step.
    assert o0.farms == o1.farms
    assert o0.market == o1.market
    assert o0.town == o1.town
    assert o0.private is not o1.private
    assert "private" not in o0.farms[1]
    assert set(o0.farms[1].keys()) == {
        "money",
        "tiles",
        "farmer",
        "hands",
        "unlocked_quadrants",
        "hires_today",
    }
    assert set(o0.private.keys()) == {"shed", "seeds", "inventories"}
    assert "step" in o0 and "step" not in o1
    env.step(
        [
            {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 1]]},
            PASS,
        ]
    )
    assert env.state[0].observation.farms == env.state[1].observation.farms
    assert env.state[0].observation.private != env.state[1].observation.private
    return {
        "players": [o0.player, o1.player],
        "shared_fields": ["farms", "market", "town", "day", "hour"],
        "public_farm_fields": sorted(o0.farms[1].keys()),
        "own_private_fields": sorted(o0.private.keys()),
        "opponent_private_visible": False,
        "framework_step_key_by_seat": [True, False],
    }


def full_episode() -> dict[str, object]:
    calls = [0, 0]
    seen_steps: list[list[int]] = [[], []]

    def agent0(observation, configuration):
        calls[0] += 1
        seen_steps[0].append(observation.step)
        market = [["BUY_SEED", "WHEAT", 1]] if observation.step == 0 else []
        return {"farmer": ["PASS"], "hands": [], "market": market}

    def agent1(observation, configuration):
        calls[1] += 1
        # Seat 1 has no framework-added `step`; shared day/hour are reliable.
        seen_steps[1].append(
            observation.day * configuration.turnsPerDay + observation.hour
        )
        return PASS

    env = new_env(seed=1, weedSpawnChance=0)
    steps = env.run([agent0, agent1])
    assert len(steps) == 720
    assert calls == [719, 719]
    assert seen_steps[0] == list(range(719))
    assert seen_steps[1] == list(range(719))
    assert [s.status for s in env.state] == ["DONE", "DONE"]
    assert [s.reward for s in env.state] == [2990.0, 3000.0]
    assert env.state[0].observation.step == 719
    assert (env.state[0].observation.day, env.state[0].observation.hour) == (29, 23)
    eod_recorded_steps = [step + 1 for step in seen_steps[0] if (step + 1) % 24 == 0]
    assert len(eod_recorded_steps) == 29 and eod_recorded_steps[-1] == 696
    return {
        "recorded_states": len(steps),
        "recorded_step_range": [0, env.state[0].observation.step],
        "policy_calls_per_player": calls,
        "action_step_range": [seen_steps[0][0], seen_steps[0][-1]],
        "final_day_hour": [env.state[0].observation.day, env.state[0].observation.hour],
        "completed_end_of_day_transitions": len(eod_recorded_steps),
        "last_end_of_day_resulting_recorded_step": eod_recorded_steps[-1],
        "final_banks_and_rewards": [s.reward for s in env.state],
        "winner_by_relative_bank": 1,
        "margin_used_by_engine_reward": True,
        "competition_match_result": "player 1 win; player 0 loss",
    }


if __name__ == "__main__":
    emit(
        {
            "engine": assert_exact_engine(),
            "configuration": config_defaults(),
            "timing": timing_boundaries(),
            "end_of_day": end_of_day_reset(),
            "visibility": visibility(),
            "episode": full_episode(),
        }
    )
