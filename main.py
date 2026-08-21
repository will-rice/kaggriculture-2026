"""Competition entrypoint: Kaggle loads this file and calls ``agent``.

The loader ``exec``s this file with an empty globals dict and takes the *last*
callable defined in it, so ``agent`` must stay the final import — and nothing
here may reference ``__file__``, which is not defined under that exec.

The submission tarball places this file next to the ``kaggriculture`` package,
and the loader puts that directory on ``sys.path`` while this module executes,
so the import below resolves both locally and on the competition runner.

**Everything below this paragraph is void and the import no longer matches it.**
The agent actually served is ``kaggriculture.kaito_policy``, vendored kaito
v21.1. Every number below was measured on ``kaggle-environments`` 1.32.3; the
ladder moved to 1.32.6 on 2026-08-07, late-season town-centre demand fell about
eightfold, and every bank quoted here roughly halved. On the re-measured
reference table v21.1 is the *weakest* of the four vendored agents — it loses
512-0 to ``boatlee_v14_policy`` and 512-0 to ``kaito_v23_policy`` over held-out
seeds. See ``docs/research/2026-08-08-engine-1326-rebaseline.md``. What to serve
instead is a human decision and has deliberately not been taken here.

The agent the text below describes is route memory: it replays the nearest of 190 routes
harvested from strong seats in the replay corpus, realigned onto the units we
actually have, and falls back to the vendored economic policy on the 0.1% of
turns where no route is near. It replaces that economic policy, which is what
this file served until the gate below was measured.

Over 128 seeded league games each, on 100 further seeds, and on 32 held-out
seeds, all three runs agree:

    agent            mean bank    league win rate     vs the recorded tape
    economic policy    145,604    0.750 [0.71, 0.79]    0 of 32
    route memory       162,251    0.922 [0.87, 0.95]   22 of 32

The tape is the whole story. Against the three reactive opponents both agents
win every game, so the economic policy's ceiling is exactly the one opponent it
cannot beat: it banks 119,380 against a recording that banks 139,997, having
re-decided all 719 turns from the live board. That gap is the reason this phase
exists, and it is also visible on the real ladder, where a blind single-episode
tape replay scores 1468.6 against the economic policy's 1025.4. Replaying a
route that is *chosen* per board beats both — but only while it stays free to
change its mind, which is what ``HYSTERESIS_MARGIN`` buys and what a lock-on
replay, banking more coins and losing every game to the tape, does not.

The store the agent replays is 3.6 MiB inside the archive and costs 3.7 s to
decode, once, inside turn zero: 4.5% of the sandbox's 60-second overage pool,
measured out of the built archive by ``kaggriculture.scripts.budget``. Nothing
on this path imports torch.
"""

# Serves `boatlee_v14_policy`, restored 2026-08-21 after the ladder falsified
# the serve decision of 2026-08-20. `searched_route_policy` passed the held-out
# gate against this agent (0.734 [0.652, 0.803] over 128 games) and that result
# was real on the ladder too -- doubled overall win rate, +10k median bank --
# but rating is set at the frontier, and above 1400 the searched route loses to
# the current meta 4-to-1 (4/19 and 1/9 across its two copies). It converged at
# ~1370 against this agent's ~1500. The gate examined against what we served,
# not against the frontier; until the gate's opponent set includes fresh
# top-band tapes, "PASS" does not mean "worth fielding".
#
# The searched route remains vendored as `searched_route_policy` -- it is a
# gate opponent and a record, not a serve.
from kaggriculture.boatlee_v14_policy import agent

__all__ = ["agent"]
