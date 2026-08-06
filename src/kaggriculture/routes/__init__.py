"""Route memory: replaying production routes harvested from the replay corpus.

Matching a live board to a recorded route needs a key that describes the
board without describing who is sitting across the table -- see
``kaggriculture.routes.signature`` for the signature and the distance used to
find the nearest prototype.

``STORE`` lives here rather than in ``play.py`` or ``scripts/harvest.py`` for
the reason ``learn/__init__.py`` gives for ``CHECKPOINT``: the module that
writes the file, the module that reads it and the module that packages it must
all name the same path. They did not. ``harvest`` wrote ``/data`` and ``play``
read beside itself, so a freshly harvested store was invisible to the agent
that needed it and a submission built from it raised ``FileNotFoundError`` on
turn zero.
"""

from pathlib import Path

# Beside this package, so one path serves both the local league and the
# unpacked submission archive: the agent imports `kaggriculture.routes.play` in
# each case and `__file__` resolves to whichever tree it was imported from. A
# path under `/data` would exist only on the workstation, and the submitted
# agent would find nothing there.
STORE = Path(__file__).parent / "prototypes.json.gz"
