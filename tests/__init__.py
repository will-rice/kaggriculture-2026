"""Test package.

Made a package -- along with ``tests/learn`` and ``tests/routes`` -- so that
two test modules may share a basename. ``tests/learn/test_play.py`` and
``tests/routes/test_play.py`` both exist, and under pytest's default prepend
import mode a bare ``test_play`` name would collide and fail collection.
"""
