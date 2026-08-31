"""Loading the frozen controller as data, and one fresh copy per episode.

Two things are wrong with importing the frozen controller directly.

The first is that ``import`` is not a check. The controller this residual is
trained against, gated against and merged over is a specific 171,508-byte
payload whose SHA-256 the author's notebook publishes; an import gives back
whatever is on disk under that name. So the baseline is named by
``BaselineIdentity`` -- a module path, the first line of the verified region,
and the digest -- and ``load_verified_baseline`` hashes the bytes before it runs
them. Swapping controllers is then a change to three strings, not a change to
code, which is the shape the promotion ledger asked for: the served controller
has already moved twice this month.

The second is that a module is a singleton. The controller keeps per-episode
state in module-level closures, and counterfactual collection replays the same
season from different points, in threads, in the same process. A shared instance
would make those replays talk to each other, and the symptom would not be an
exception -- it would be a slightly wrong reward on a slightly wrong trajectory.
So every call to ``load_verified_baseline`` ``exec``s the verified bytes into a
fresh namespace and returns that namespace's own agent.

**The agent is resolved by name, not as the last callable defined.** The Kaggle
runner and ``scripts.package`` both take the last callable in a namespace, and
for this controller that rule picks the wrong function: its final statements
bind ``_V54_INNER_AGENT = agent`` and *then* rebind ``agent``, and rebinding an
existing key leaves it in its original dict position. The last-inserted callable
is therefore the inner agent -- which is also named ``agent``, so even checking
``__name__`` would not notice. It differs from the real entry point by the seed
redaction the header describes, so playing it would be a controller that can see
the engine seed, parity against the served agent would fail, and nothing would
raise. ``tests/market_residual/test_baseline.py`` pins that trap.
"""

import hashlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

SERVED_MODULE = "kaggriculture.kaito_v54_policy"

# The first line of the author's own file, where our vendoring header stops.
# Restated from `tests/test_vendored_policies.V54_BODY_START`, which pins the
# same boundary against the same digest.
SERVED_BODY_START = '"""v51: current-meta capital-flow hybrid.'

# The checksum the source notebook publishes over its own payload.
SERVED_SHA256 = "9f21735aaf0354e064e1d9bab1b2e186fad12cf6446e7ecf83f5014915e04a4f"

ENTRYPOINT = "agent"


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
    Note that executing this controller registers modules in ``sys.modules``
    under bare names; that is the same side effect importing it has, and the
    names are pinned by ``tests/test_vendored_policies.py``.

    Args:
        identity: Which controller to load, and the digest it must have.

    Returns:
        That controller's ``agent``, fresh.

    Raises:
        BaselineIntegrityError: If the source fails its digest check, or the
            verified source binds no callable ``agent``.
    """
    source = identity.verified_source()
    namespace: dict[str, Any] = {
        "__name__": f"verified:{identity.module}",
        "__file__": str(identity.source_path()),
        "__package__": "",
    }
    exec(compile(source, namespace["__file__"], "exec"), namespace)  # noqa: S102
    agent = namespace.get(ENTRYPOINT)
    if not callable(agent):
        raise BaselineIntegrityError(
            f"{identity.module} binds no callable {ENTRYPOINT!r}"
        )
    return agent
