"""What ``BaselineIdentity`` proves, and what a fresh copy has to be.

Three properties, and each of them is a failure that would otherwise be silent.

A drifted controller still imports and still plays a season, so the digest check
is the only thing standing between "the agent we gated" and "the agent that
happens to be on disk under that name".

The agent is resolved by the author's own entry-point name rather than by any
positional rule, and this file pins why: ``agent`` is defined twice in this
controller and neither definition is the entry point. A loader that took "the
attribute called agent" would get the second one, which behaves correctly today
and would go on doing so until it did not; a loader that took the shadowed first
one would get the v51 backbone, which plays a different season on any board
where v56's shop branch fires and raises nothing on the boards where it does
not.

And a fresh copy that shared state with the last one would only show up as a
wrong reward on a slightly wrong trajectory, which is exactly the kind of thing
counterfactual collection would then train on.
"""

import importlib.util
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import FunctionType
from typing import cast

import pytest

from kaggriculture.market_residual.baseline import (
    ENTRYPOINT,
    SERVED,
    BaselineIdentity,
    BaselineIntegrityError,
    load_verified_baseline,
)
from tests.market_residual.conftest import Turn

# How much of a real episode the two fresh copies are replayed over. Long
# enough to cross the controller's own phase changes -- it repairs drift and
# re-times the market as the season runs -- and short enough to stay off the
# slow marker.
TRACE_TURNS = 200


def copy_of_the_served_module(tmp_path: Path, name: str) -> Path:
    """Return an importable copy of the served controller under a new name.

    Args:
        tmp_path: A directory that will be put on ``sys.path`` by the caller.
        name: The module name the copy should answer to.

    Returns:
        The path the copy was written to.
    """
    copy = tmp_path / f"{name}.py"
    shutil.copyfile(SERVED.source_path(), copy)
    return copy


def test_the_served_identity_matches_the_module_on_disk() -> None:
    """The digest carried as data must be the digest of what we actually serve.

    ``SERVED`` restates the notebook's checksum so that a controller swap is a
    change to four strings. Restating it is only safe while it agrees with the
    file; this is the test that keeps it honest, and it is the same region
    ``tests/test_vendored_policies.py`` hashes from the other direction.
    """
    body = SERVED.verified_source()

    assert SERVED.source_path().name == "kaito_v56_policy.py"
    assert len(body.encode("utf-8")) == 199_613
    assert body.startswith(SERVED.body_start)


def test_a_drifted_source_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One appended comment is enough; a payload edit would be no louder."""
    source = copy_of_the_served_module(tmp_path, "drifted_v56")
    source.write_text(source.read_text(encoding="utf-8") + "\n# drift\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    identity = BaselineIdentity(
        module="drifted_v56", body_start=SERVED.body_start, sha256=SERVED.sha256
    )

    with pytest.raises(BaselineIntegrityError, match="sha256"):
        load_verified_baseline(identity)


def test_an_undrifted_copy_under_another_name_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise the digest check could be passing on the module name alone.

    Without this, ``test_a_drifted_source_is_refused`` would still pass if
    ``load_verified_baseline`` refused every module it had not heard of, and the
    digest would never have been computed at all.
    """
    copy_of_the_served_module(tmp_path, "undrifted_v56")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    identity = BaselineIdentity(
        module="undrifted_v56", body_start=SERVED.body_start, sha256=SERVED.sha256
    )

    assert callable(load_verified_baseline(identity))


def test_a_missing_marker_is_refused_rather_than_hashed_from_the_top() -> None:
    """A marker that no longer matches must fail, not silently move the region.

    The digest is taken from a line of the author's file to the end. If that
    line were ever edited away, hashing from wherever the search landed -- or
    from byte zero -- would produce a mismatch that reads as "the payload
    changed", pointing an investigation at the wrong half of the file.
    """
    identity = BaselineIdentity(
        module=SERVED.module,
        body_start='"""not a line of this file.',
        sha256=SERVED.sha256,
    )

    with pytest.raises(BaselineIntegrityError, match="does not contain"):
        identity.verified_source()


def test_an_unresolvable_module_is_refused() -> None:
    """A controller swap that names a module nobody shipped fails at load."""
    identity = BaselineIdentity(
        module="kaggriculture.kaito_v99_policy",
        body_start=SERVED.body_start,
        sha256=SERVED.sha256,
    )

    with pytest.raises(BaselineIntegrityError, match="no source file"):
        identity.source_path()


def test_the_loaded_agent_is_the_entry_point_and_not_either_agent() -> None:
    """Two functions here are called ``agent`` and neither one is served.

    Rebinding a name leaves it in its original namespace position, so the
    controller's second ``def agent`` does not move to the end and the author
    added ``kaggle_agent_v56`` below both to be the callable Kaggle picks. That
    is what ``ENTRYPOINT`` names, what ``main.py`` imports, and what the gate
    played, so it is what a counterfactual collected here has to be about.

    Both wrong answers are asserted against, because they fail differently. The
    surviving ``agent`` is what a loader reaching for the obvious attribute
    would get; it is correct today and would stop being correct the moment a
    later kernel put anything in the wrapper, which is a change we would not
    see. The shadowed ``agent`` is the v51 backbone, and it is wrong now --
    silently, on most seeds, because the route it differs by is a prior the
    residual controller mostly overrules.
    ``tests/test_vendored_policies.py`` plays a branching season to show that.
    """
    source = SERVED.verified_source()
    namespace: dict[str, object] = {"__name__": "trap", "__file__": "<trap>"}
    exec(compile(source, "<trap>", "exec"), namespace)  # noqa: S102

    loaded = cast(FunctionType, load_verified_baseline(SERVED))
    entry = cast(FunctionType, namespace[ENTRYPOINT])
    surviving = cast(FunctionType, namespace["agent"])
    definitions = [
        line
        for line, text in enumerate(source.splitlines(), 1)
        if text.startswith("def agent(")
    ]

    assert len(definitions) == 2, "the shadowing this test is about is gone"
    assert loaded.__name__ == ENTRYPOINT
    assert loaded.__code__.co_firstlineno == entry.__code__.co_firstlineno
    assert surviving.__code__.co_firstlineno == definitions[-1]
    assert loaded.__code__.co_firstlineno > definitions[-1]


def test_fresh_instances_do_not_share_state(kaito_turns: tuple[Turn, ...]) -> None:
    """Two copies replayed over the same real season must agree exactly.

    The controller keeps per-episode state in module-level closures. If a second
    copy inherited any of it -- through the bare modules the payload registers in
    ``sys.modules``, which the second load overwrites -- the two traces would
    diverge somewhere mid-season, and every counterfactual replay collected in
    one process would be built on whichever copy ran last.

    Replayed over seat zero's own observations, in order, so this is the season
    that seat actually played rather than a sequence assembled here.
    """
    left = load_verified_baseline(SERVED)
    right = load_verified_baseline(SERVED)
    seat_zero = [turn for turn in kaito_turns if turn.seat == 0][:TRACE_TURNS]

    left_trace = [left(turn.observation) for turn in seat_zero]
    right_trace = [right(turn.observation) for turn in seat_zero]

    assert left is not right
    assert len(seat_zero) == TRACE_TURNS
    assert left_trace == right_trace
    assert any(action["market"] for action in left_trace), "trace saw no market orders"


def test_concurrent_loads_are_serialised_and_still_independent(
    kaito_turns: tuple[Turn, ...],
) -> None:
    """Loading is the unsafe part, and the lock is what makes it safe.

    The payload registers modules under bare top-level names, so two loads
    running at once can bind a half-initialised module. The existing
    fresh-instance test loads serially and so cannot exercise that race at all;
    this one runs the loads through the pool this repository reaches for by
    default, and demands both a completed load and a controller that plays the
    same season as one loaded alone.
    """
    with ThreadPoolExecutor(max_workers=4) as pool:
        loaded = list(pool.map(load_verified_baseline, [SERVED] * 4))
    seat_zero = [turn for turn in kaito_turns if turn.seat == 0][:TRACE_TURNS]
    alone = load_verified_baseline(SERVED)

    expected = [alone(turn.observation) for turn in seat_zero]
    traces = [[agent(turn.observation) for turn in seat_zero] for agent in loaded]

    assert len({id(agent) for agent in loaded}) == len(loaded)
    assert all(trace == expected for trace in traces)
