"""Puts the repo root on ``sys.path`` so tests can import ``baselines``.

``baselines/`` lives outside ``src/`` on purpose: it is not part of the
installed package and never ships in the submission archive. Pytest's default
import mode only adds a test file's own package root to ``sys.path``, not the
invocation directory, so without this file ``from baselines.heuristic_v1
import STRATEGY`` fails under a plain ``pytest`` invocation even though it
works under ``python -m pytest``. An empty conftest.py at the repo root is
pytest's own mechanism for putting that root on ``sys.path``.
"""
