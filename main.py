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

# Serves `kaito_v56_policy` as of 2026-09-01, replacing `kaito_v54_policy`,
# which had been served since 2026-08-31. Held-out seeds 700000-700063, each
# played in both seat orderings, on kaggle-environments 1.32.7:
#
#     vs kaito v54 (the agent it replaces)  116/128 = 0.906  [0.843, 0.946]
#
# The served bar is 0.5 and the interval does not come near it. Only that arm
# was run live: the frontier pool was scored in its replay form (0.903, lower
# 0.890, against v54's 0.847) and is a diagnostic, not the pass -- it grades
# against recorded tapes rather than opponents that re-decide. The serve switch
# of 2026-08-20 was made on a gate that only asked "does the candidate beat what
# we serve", and the ladder falsified it the next day, which is why that leg
# exists at all. Here the ladder is the confirmation rather than the refutation:
# two v56 copies submitted 2026-09-01 reached 2340.7 and 2302.1 within hours,
# past the 2270.5 that v54's better copy took seventeen hours to reach, and they
# track each other to 102 points where the v54 pair differed by 512. See
# `kaggriculture.search.scripts.holdout`, whose `SERVED` constant tracks this
# import line and is tested against it.
#
# **Imported by the author's own entry-point name, not as `agent`.** That file
# defines `agent` twice and `_kaggle_submission_entrypoint` twice; rebinding a
# name does not move it to the end of the namespace, so the author added
# `kaggle_agent_v56` as the last callable and documented that it must stay
# there. That is what `kaggle_environments` runs when the file is loaded as an
# agent path -- which is how `search.scripts.holdout` played the 128 games above
# -- so it is what this line imports, and every path that serves this kernel
# lands on the same function object.
#
# `kaito_v54_policy`, `boatlee_v14_policy` and `searched_route_policy` remain
# vendored -- they are gate opponents and records, not serves.
from kaggriculture.kaito_v56_policy import kaggle_agent_v56 as agent

__all__ = ["agent"]
