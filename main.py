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

# Serves `searched_route_policy`: the route search's first gate-passing output,
# measured 2026-08-20 on kaggle-environments 1.32.7. A top-30 route harvested
# from episode 94106275 (2026-08-18 archive) plus one accepted edit
# (SELL WOOL 3 -> 5 at turn 646, +1.95% +/- 0.50% paired), embedded as a
# standalone tape and proven bank-identical to the gated artifact.
#
# The evidence is the held-out gate, one pre-registered run, no gate-shopping:
#
#   rate 0.7344  Wilson 95% [0.6518, 0.8032]  over 128 seat-swapped games
#   vs boatlee_v14_policy (the previous serve) on seeds 700,000-700,063,
#   disjoint from every seed the search ever saw.
#
# boatlee_v14 stays vendored below as the reference opponent; its own serve
# block (0.960 league win rate on 1.32.6, 2026-08-08) is in git history at
# 730d9e2 and before.
from kaggriculture.searched_route_policy import agent

__all__ = ["agent"]
