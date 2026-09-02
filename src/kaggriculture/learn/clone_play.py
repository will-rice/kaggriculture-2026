"""The cloned teacher as an agent file the reference engine can load.

``arena`` plays a path, and ``kaggle_environments`` execs that path and takes
the last callable the file leaves behind. ``learn.play`` is already that agent
-- same encoders, same masks, same argmax -- and the only thing that differs
here is which weights it loads: the clone of our own teacher under ``/data``
rather than the corpus clone beside the package.

The redirection is a rebind of ``play.CHECKPOINT`` rather than a second copy of
``play.agent``. ``play.model`` reads the name out of its own module globals and
caches the result, which is exactly the seam ``play``'s docstring describes for
pointing a test at its own file; a forked copy of the agent would be a second
implementation that could drift from the one that ships, and the whole point of
the gate is to measure the deployed path.

``model.cache_clear()`` runs because the engine execs this file inside a worker
that may already have loaded the packaged checkpoint for another seat. Without
it, a process that had played ``learn/play.py`` once would keep serving those
weights here and the gate would quietly score the wrong agent.

Nothing is defined after ``agent``: the engine's loader takes the last callable
in the namespace, so anything below would be played instead of it.
"""

from kaggriculture.learn import CLONE_CHECKPOINT, play

play.CHECKPOINT = CLONE_CHECKPOINT
play.model.cache_clear()

agent = play.agent
