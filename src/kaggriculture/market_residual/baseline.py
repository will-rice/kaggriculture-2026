"""Loading the frozen controller as data, and one fresh copy per episode.

Two things are wrong with importing the frozen controller directly.

The first is that ``import`` is not a check. The controller this residual is
trained against, gated against and merged over is a specific 199,613-byte
payload whose SHA-256 the author's notebook publishes; an import gives back
whatever is on disk under that name. So the baseline is named by
``BaselineIdentity`` -- a module path, the first line of the verified region,
and the digest -- and ``load_verified_baseline`` hashes the bytes before it runs
them. Swapping controllers is then a change to four strings, not a change to
code, which is the shape the promotion ledger asked for: the served controller
has already moved three times this month.

The marker is doing less work than it looks like it is. v54 and v56 open their
verified regions with the *same* line, because both bundle the same v51 ancestor
and inherit its module docstring; only the digest separates them. A swap that
changed ``SERVED_MODULE`` and forgot ``SERVED_SHA256`` therefore fails loudly at
the hash rather than passing on a matching marker, which is the order those two
checks have to happen in.

The second is that a module is a singleton. The controller keeps per-episode
state in module-level closures, and counterfactual collection replays the same
season from different points, in threads, in the same process. A shared instance
would make those replays talk to each other, and the symptom would not be an
exception -- it would be a slightly wrong reward on a slightly wrong trajectory.
So every call to ``load_verified_baseline`` ``exec``s the verified bytes into a
fresh namespace and returns that namespace's own entry point.

That ``exec`` is not thread-safe, and the reason is not the namespace -- it is
``sys.modules``. Executing the payload decodes and registers a bundled ancestor
under *bare* top-level names (``v48``, ``v50``, ``scripts`` and others), and two
loads running at once interleave those registrations: the second load can bind a
half-initialised module the first is still filling in, and the import machinery
inside the payload then reads attributes that are not there yet. So the load is
serialised on ``LOAD_LOCK``. It is a lock rather than a documented rule because
this repository's own convention reaches for ``ThreadPoolExecutor`` by default,
and collection -- which wants one fresh controller per branch row -- is exactly
where that reach happens. The lock covers only the load; the returned callables
share nothing and are played concurrently.

**The agent is resolved by the author's own entry-point name, and ``agent`` is
not it.** This controller defines ``agent`` twice -- once at the end of the
bundled v51 ancestor, once for v56's shop branch -- and ``ENTRYPOINT`` names
neither: it names ``kaggle_agent_v56``, the third function the author added
because rebinding a name does not move it to the end of a namespace, and
documented as the one that must stay last. That is what the Kaggle runner picks
out of the file, what ``main.py`` imports, and what the 128-game gate played, so
loading it here is what makes the counterfactuals this module feeds be about the
agent we actually serve rather than about one that happens to behave like it.

The reason to pin a name rather than follow the runner's positional rule is the
shadowed ``agent``, and the reason to pin it rather than test for it is that it
is almost undetectable. It is the v51 backbone -- no shop branch, no seed
redaction -- and its route differs from v56's in 690 of 719 entries without that
changing anything on most boards, because the route is a prior the residual
controller mostly overrules. On the gate's own reference seed it banks the same
number v56 banks, to the coin. It only diverges where the shop branch fires, and
there it diverges on most turns. So a collection run built on the wrong function
would look right in every summary it produced. The two survivors agree today --
``kaggle_agent_v56`` delegates straight to the surviving ``agent`` -- which is
precisely why picking between them cannot be left to a rule that a later kernel
could quietly move out from under us.
``tests/market_residual/test_baseline.py`` pins which of the three we load.
"""

import hashlib
import importlib.util
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import CodeType
from typing import Any, Mapping, Protocol

SERVED_MODULE = "kaggriculture.kaito_v56_policy"

# The first line of the author's own file, where our vendoring header stops.
# Restated from `tests/test_vendored_policies.V56_BODY_START`, which pins the
# same boundary against the same digest. It is the same line v54 opened with --
# v56 bundles the same v51 ancestor and inherits its module docstring -- so the
# marker alone does not identify the controller and the digest below is what
# does.
SERVED_BODY_START = '"""v51: current-meta capital-flow hybrid.'

# The checksum the source notebook publishes over its own payload.
SERVED_SHA256 = "e7f0502537ea1a79f4ec971235627930fa4ef233156547da942369679141f10c"

# The author's own final callable, and the name every path that serves this
# kernel resolves: `main.py` imports it, and `kaggle_environments` picks it out
# of the file by position when `search.scripts.holdout` plays the file as an
# agent path. Controller-specific, so it moves with SERVED_MODULE.
ENTRYPOINT = "kaggle_agent_v56"

# Serialises the payload ``exec`` below, whose ``sys.modules`` registrations are
# global, and guards the two caches beside it. See this module's docstring for
# why a rule was not enough.
LOAD_LOCK = threading.Lock()

# Compiled payloads, keyed by the digest that was verified to produce them. The
# source is still read and hashed on every load, so a controller that changed on
# disk still raises before anything cached is reached -- the cache is keyed by
# the digest, so changed bytes address a different entry and can never be served
# from an old one. Measured on this payload the saving is small (compiling
# 171 KiB costs ~1.2 ms of a ~890 ms load; the rest is the payload's own
# module-level work) and it is kept because it is exactly free, not because it
# is the lever collection needed.
CODE_CACHE: dict[str, CodeType] = {}


@dataclass(frozen=True)
class LoadReport:
    """How much of this process's time has gone into loading controllers.

    Counterfactual collection pays two fresh loads per branch arm, and a
    throughput budget that cannot separate that cost from the simulation it
    surrounds is a budget nobody can act on. The counter lives here because
    this is the only place that knows a load happened.
    """

    count: int
    seconds: float


LOADS = LoadReport(count=0, seconds=0.0)


class BaselineIntegrityError(RuntimeError):
    """Raised when the frozen controller on disk is not the verified one."""


class BaselineAgent(Protocol):
    """The engine's agent calling convention, as the controller exposes it."""

    def __call__(
        self,
        observation: Mapping[str, Any],
        configuration: object | None = None,
    ) -> dict[str, Any]:
        """Return one canonical engine action for a raw observation."""
        ...


@dataclass(frozen=True)
class BaselineIdentity:
    """Which frozen controller a run is built on, and how to prove it.

    Held as data so that promoting a different controller is a change to these
    three strings. Every artifact this project writes carries the identity it
    was produced under, so a model trained against one controller cannot be
    served over another without the mismatch being visible.
    """

    module: str
    body_start: str
    sha256: str

    @classmethod
    def served(cls) -> "BaselineIdentity":
        """Return the identity of the controller ``main.py`` currently serves."""
        return SERVED

    def source_path(self) -> Path:
        """Return the file the module resolves to, without importing it.

        Returns:
            The path ``module`` would be loaded from.

        Raises:
            BaselineIntegrityError: If the module resolves to nothing, or to
                something with no source file.
        """
        spec = importlib.util.find_spec(self.module)
        if spec is None or spec.origin is None:
            raise BaselineIntegrityError(f"no source file for module {self.module!r}")
        return Path(spec.origin)

    def verified_source(self) -> str:
        """Return the controller's source, having checked its digest first.

        Returns:
            Everything from ``body_start`` to the end of the file -- exactly the
            region ``sha256`` is taken over, and a self-contained module.

        Raises:
            BaselineIntegrityError: If the marker is absent or the digest of the
                region it opens is not ``sha256``.
        """
        source = self.source_path().read_text(encoding="utf-8")
        start = source.find(self.body_start)
        if start < 0:
            raise BaselineIntegrityError(
                f"{self.module} does not contain {self.body_start!r}"
            )
        body = source[start:]
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        if digest != self.sha256:
            raise BaselineIntegrityError(
                f"{self.module} has sha256 {digest}, expected {self.sha256}"
            )
        return body


SERVED = BaselineIdentity(
    module=SERVED_MODULE,
    body_start=SERVED_BODY_START,
    sha256=SERVED_SHA256,
)


def load_verified_baseline(identity: BaselineIdentity) -> BaselineAgent:
    """Return a private copy of the verified frozen controller.

    The returned callable shares no module-level state with any other copy or
    with the imported module, so one per episode is safe to run concurrently.
    The loading itself is not: executing this controller registers modules in
    ``sys.modules`` under bare names -- the same side effect importing it has,
    with the names pinned by ``tests/test_vendored_policies.py`` -- so the
    ``exec`` is serialised on ``LOAD_LOCK`` and concurrent callers queue.

    Args:
        identity: Which controller to load, and the digest it must have.

    Returns:
        That controller's ``ENTRYPOINT``, fresh.

    Raises:
        BaselineIntegrityError: If the source fails its digest check, or the
            verified source binds no callable under ``ENTRYPOINT``.
    """
    global LOADS
    started = time.perf_counter()
    source = identity.verified_source()
    namespace: dict[str, Any] = {
        "__name__": f"verified:{identity.module}",
        "__file__": str(identity.source_path()),
        "__package__": "",
    }
    with LOAD_LOCK:
        code = CODE_CACHE.get(identity.sha256)
        if code is None:
            code = compile(source, namespace["__file__"], "exec")
            CODE_CACHE[identity.sha256] = code
        exec(code, namespace)  # noqa: S102
        LOADS = replace(
            LOADS,
            count=LOADS.count + 1,
            seconds=LOADS.seconds + time.perf_counter() - started,
        )
    agent = namespace.get(ENTRYPOINT)
    if not callable(agent):
        raise BaselineIntegrityError(
            f"{identity.module} binds no callable {ENTRYPOINT!r}"
        )
    return agent


def load_report() -> LoadReport:
    """Return how many controllers this process has loaded, and at what cost.

    Returns:
        The running total since the last reset.
    """
    return LOADS


def reset_load_report() -> None:
    """Zero the load counter, so one phase's cost can be attributed to it."""
    global LOADS
    with LOAD_LOCK:
        LOADS = LoadReport(count=0, seconds=0.0)
