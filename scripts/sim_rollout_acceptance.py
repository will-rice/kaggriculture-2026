"""Run rollout legality and scripted-opponent acceptance gates."""

import argparse
import json

import torch
from kaggle_environments import make

from kaggriculture import economic_policy
from kaggriculture.learn.encoding import MARKET_SLOTS, MAX_UNITS, QUANTITIES, UNIT_OPS
from kaggriculture.sim.config import Config
from kaggriculture.sim.engine import reset, step, unit_quantity_ones
from kaggriculture.sim.rollout import collect_segment, scripted_actions


class _UniformPolicy(torch.nn.Module):
    def forward(
        self,
        board: torch.Tensor,
        scalars: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = len(board)
        device = board.device
        return (
            torch.zeros(batch, MAX_UNITS, len(UNIT_OPS), device=device),
            torch.zeros(batch, MAX_UNITS, len(QUANTITIES), device=device),
            torch.zeros(batch, len(MARKET_SLOTS) + 2, len(QUANTITIES), device=device),
            torch.zeros(batch, 1, device=device),
        )


def legality_gate(device: torch.device, batch: int, turns: int) -> dict[str, int]:
    """Collect at least one million masked decisions and require zero illegal."""
    seeds = torch.arange(batch, dtype=torch.int64, device=device) + 300_001
    state = reset(Config(), seeds)
    _, trajectory = collect_segment(
        state,
        _UniformPolicy().to(device),
        turns=turns,
        generator=torch.Generator(device=device).manual_seed(20260810),
    )
    decisions = turns * batch * 2 * (MAX_UNITS + len(MARKET_SLOTS) + 2)
    if decisions < 1_000_000:
        raise AssertionError(f"only {decisions} decisions; one million required")
    # `Trajectory.illegal` stays on the device so that `collect_segment` holds
    # no synchronisation; the gate is the place that genuinely needs the number,
    # and reading it here costs one synchronisation for the whole run.
    illegal = int(trajectory.illegal)
    if illegal:
        raise AssertionError(f"rollout produced {illegal} illegal decisions")
    return {"decisions": decisions, "illegal": illegal}


def economic_parity(device: torch.device, games: int) -> dict[str, object]:
    """Require economic_policy's terminal banks to match reference game by game."""
    seeds = list(range(400_001, 400_001 + games))
    environments = []
    for seed in seeds:
        environment = make("kaggriculture", configuration={"seed": seed}, debug=True)
        environment.reset(2)
        environments.append(environment)
    state = reset(Config(), torch.tensor(seeds, dtype=torch.int64, device=device))
    units = torch.full(
        (games, 2, MAX_UNITS),
        UNIT_OPS.index("PASS"),
        dtype=torch.int16,
        device=device,
    )
    quantities = unit_quantity_ones(games, device)
    for _ in range(719):
        opponent_units, opponent_quantities, markets = scripted_actions(
            state, economic_policy.agent
        )
        units[:, 1].copy_(opponent_units)
        quantities[:, 1].copy_(opponent_quantities)
        for environment in environments:
            environment.step(
                [
                    {"farmer": ["PASS"], "hands": [], "market": []},
                    economic_policy.agent(environment.state[1].observation),
                ]
            )
        state = step(state, units, markets, quantities)
    reference_banks = [
        int(environment.state[1].observation.farms[1].money)
        for environment in environments
    ]
    simulator_banks = state.money[:, 1].cpu().tolist()
    if simulator_banks != reference_banks:
        index = next(
            index
            for index, pair in enumerate(
                zip(simulator_banks, reference_banks, strict=True)
            )
            if pair[0] != pair[1]
        )
        raise AssertionError(
            f"economic parity diverged at seed {seeds[index]}: "
            f"{simulator_banks[index]} != {reference_banks[index]}"
        )
    return {
        "games": games,
        "divergences": 0,
        "mean_bank": sum(reference_banks) / games,
    }


def main() -> None:
    """Run selected rollout gates and print one machine-readable record."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--turns", type=int, default=25)
    parser.add_argument("--economic-games", type=int, default=0)
    parser.add_argument("--skip-legality", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    record: dict[str, object] = {"device": str(device)}
    if not args.skip_legality:
        record["legality"] = legality_gate(device, args.batch, args.turns)
    if args.economic_games:
        record["economic_policy"] = economic_parity(device, args.economic_games)
    print(json.dumps(record))


if __name__ == "__main__":
    main()
