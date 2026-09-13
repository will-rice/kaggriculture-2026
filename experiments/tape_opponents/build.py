"""Turn the recorded ladder into opponents.

The campaign's pool is 83 agents and 69 of them descend from one program, so a
candidate is mostly being measured against its own family. Meanwhile 17,679
qualifying episodes sit on disk, each holding two recorded seats, and a
recorded seat replayed verbatim is what most of the ladder actually plays.
Those are realistic opponents rather than strawmen, and they are already paid
for.

Each kept seat becomes a directory holding a small ``main.py`` and the route it
replays. Opponents are loaded from a path and are not bound by the one-file
rule a submission is, so the route stays JSON beside the code rather than a
compressed literal inside it.

Deduplicated by opening. Replay waves are narrow -- the field converges on a
handful of strong openings and most recorded seats in a given week are the same
strategy -- so keeping every episode would buy 500 copies of one opponent. The
fingerprint is the first `PREFIX` actions, which is how the teams tracking this
publicly separate strategy families.

Written outside `config.AGENTS` on purpose. That path is a `copycheck` corpus
root, and adding hundreds of megabytes of recorded actions to the corpus would
cost every worker that builds it for no protection worth having: a candidate
cannot read a file at runtime, and without `base64` or `zlib` a route it
embedded as a literal would be megabytes of visible JSON.
"""

import argparse
import hashlib
import json
import logging
import pathlib

from tqdm import tqdm

from kaggriculture.campaign import tapes

LOGGER = logging.getLogger(__name__)
DESTINATION = pathlib.Path("/data/kaggriculture/tapes")
# Actions that define a strategy family. The opening commits the farm -- what
# is bought, hired and built in the first days decides the economy -- so two
# seats agreeing this far are the same strategy however they end.
PREFIX = 120

PLAYER = '''"""Replay of a recorded ladder seat: {name}.

Final bank {bank:,.0f} against {other:,.0f}, engine {engine}, episode seed
{seed}. An opponent for the campaign's pool, not a submission.

The route is fixed, which is the point: this is what most of the ladder plays,
so a candidate measured against it is measured against the real field. It does
not answer what the candidate does, and a pool of these alone would teach a
candidate nothing about being reacted to.
"""

import json
import pathlib

ROUTE = json.loads((pathlib.Path(__file__).resolve().parent / "route.json").read_text())


def agent(observation, configuration=None):
    """The action this seat took on this turn of the game it was recorded in."""
    step = int(observation.get("day", 0)) * 24 + int(observation.get("hour", 0))
    hands = observation["farms"][int(observation.get("player", 0))].get("hands") or []
    if step >= len(ROUTE):
        return {{"farmer": ["PASS"], "hands": [["PASS"] for _ in hands], "market": []}}
    recorded = ROUTE[step]
    # The recording had its own hands; this seat has whatever it has now. A
    # route that stayed legal keeps the same count, and one that drifted gets
    # padded rather than handing the engine a list of the wrong length.
    theirs = [list(hand) for hand in (recorded.get("hands") or [])]
    theirs = (theirs + [["PASS"]] * len(hands))[: len(hands)]
    return {{
        "farmer": list(recorded.get("farmer") or ["PASS"]),
        "hands": theirs,
        "market": [list(order) for order in (recorded.get("market") or [])],
    }}
'''


def main() -> None:
    """Walk episodes, keep one seat per distinct opening, write each as an agent."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", type=int, help="episodes to walk")
    parser.add_argument(
        "--min-bank",
        type=float,
        default=120_000.0,
        help="skip seats that finished below this, so the pool gains strength",
    )
    parser.add_argument("--out", type=pathlib.Path, default=DESTINATION)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    seen: dict[str, str] = {}
    roster: dict[str, str] = {}
    walked = weak = duplicate = 0
    for i in tqdm(range(args.episodes), desc="episodes"):
        try:
            episode = tapes.qualifying(i)
        except FileNotFoundError:
            LOGGER.info("corpus exhausted after %d episodes", walked)
            break
        walked += 1
        banks = [record.get("reward") or 0.0 for record in episode.steps[-1]]
        for seat, bank in enumerate(banks):
            if bank < args.min_bank:
                weak += 1
                continue
            route = [step[seat].get("action") for step in episode.steps[1:]]
            if any(action is None for action in route):
                continue
            fingerprint = hashlib.sha256(
                json.dumps(route[:PREFIX], sort_keys=True).encode()
            ).hexdigest()[:16]
            if fingerprint in seen:
                duplicate += 1
                continue
            name = f"tape_{fingerprint}"
            seen[fingerprint] = name
            folder = args.out / name
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "route.json").write_text(
                json.dumps(route, separators=(",", ":")), encoding="utf-8"
            )
            (folder / "main.py").write_text(
                PLAYER.format(
                    name=name,
                    bank=bank,
                    other=banks[1 - seat],
                    engine=episode.engine_version,
                    seed=episode.seed,
                ),
                encoding="utf-8",
            )
            roster[name] = str((folder / "main.py").resolve())

    listing = args.out / "roster.json"
    listing.write_text(json.dumps(roster, indent=1, sort_keys=True), encoding="utf-8")
    LOGGER.info(
        "%d episodes walked: %d distinct openings kept, %d seats below %s, "
        "%d duplicate openings. Roster at %s",
        walked,
        len(roster),
        weak,
        f"{args.min_bank:,.0f}",
        duplicate,
        listing,
    )


if __name__ == "__main__":
    main()
