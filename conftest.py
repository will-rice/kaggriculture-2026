"""Puts the repo root on ``sys.path`` for a plain ``pytest`` invocation.

Pytest's default import mode adds a test file's own package root to
``sys.path``, not the invocation directory, so ``main.py`` and anything else
living at the repository root would not import under a bare ``pytest`` even
though it works under ``python -m pytest``. An empty conftest.py at the repo
root is pytest's own mechanism for putting that root on ``sys.path``.
"""
