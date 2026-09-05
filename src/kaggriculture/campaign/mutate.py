"""One mutation: a codex session in a sandbox, or a fake that edits a constant.

The codex command is a class attribute so a test can replace it with ``true``
or ``sleep``; the rest of the module never changes between the fake and the
real thing.

Codex 0.147's ``--json`` output is one JSON object per line. Token usage
lives on the ``turn.completed`` event, under ``usage.input_tokens`` and
``usage.output_tokens`` (verified against a real session log); other events
carry no usage and are ignored. Codex sometimes ends a turn with a question
instead of writing ``child.py``; ``_last_message`` recovers the last
``agent_message`` text so the caller knows why.
"""

import json
import logging
import os
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel

from kaggriculture.campaign import config

LOGGER = logging.getLogger(__name__)


class Mutation(BaseModel):
    """The outcome of one mutation call.

    Attributes:
        program_id: The child program id the sandbox was built for.
        child: Path to the written child program, or None if there is
            nothing to evaluate. A timed-out session that had already
            written ``child.py`` still yields it: the file is what gets
            evaluated, however the session ended.
        status: "ok", "no_output" (ran but wrote nothing usable), "timeout",
            or "exec_error".
        reason: Free-form explanation; empty on "ok".
        seconds: Wall-clock time the call took.
        input_tokens: Total input tokens billed, summed across the call.
        output_tokens: Total output tokens billed, summed across the call.
    """

    program_id: str
    child: Path | None
    status: Literal["ok", "no_output", "timeout", "exec_error"]
    reason: str
    seconds: float
    input_tokens: int
    output_tokens: int


class Mutator(Protocol):
    """Something that turns a sandbox into a child program, or a failure."""

    def __call__(self, sandbox: Path, program_id: str) -> Mutation:
        """Mutates the parent in ``sandbox`` into a child program.

        Args:
            sandbox: A directory built by ``prompt.build_sandbox``.
            program_id: The child program id.

        Returns:
            A `Mutation` describing the outcome.
        """
        ...


class CodexMutator:
    """Runs ``codex exec`` in a prebuilt sandbox and reports the result.

    ``COMMAND`` is overridden by tests (e.g. to ``["true"]`` or
    ``["sleep", "N"]``) to exercise the no-output and timeout paths without
    spending a real codex call; the ``-m``/``-C`` flags are only appended
    when the command actually is codex.

    Codex spawns shell commands as tool calls in ``workspace-write`` mode; on
    a timeout, killing only the direct child would orphan any grandchild
    still running, leaking a core the harness's ``config.CORE_BUDGET``
    assumes is free. The process runs in its own session
    (``start_new_session=True``) so a timeout can kill the whole process
    group with ``os.killpg``, not just codex itself.
    """

    COMMAND = [
        "codex",
        "exec",
        "-s",
        "workspace-write",
        "-c",
        "approval_policy=never",
        "--json",
        "-",
    ]

    def __init__(
        self,
        model: str = "gpt-5.6-sol",
        timeout: float = config.MUTATION_TIMEOUT_SECONDS,
    ) -> None:
        """Initializes the mutator.

        Args:
            model: The codex model to request.
            timeout: Seconds to allow the codex call before killing it.
        """
        self.model = model
        self.timeout = timeout

    def __call__(self, sandbox: Path, program_id: str) -> Mutation:
        """Runs one codex session in ``sandbox`` and returns its outcome.

        Args:
            sandbox: A directory built by ``prompt.build_sandbox``, holding
                ``AGENTS.md``, ``parent.py``, and ``PROMPT.md``.
            program_id: The child program id.

        Returns:
            A `Mutation` describing what happened.
        """
        started = time.perf_counter()
        prompt = (sandbox / "PROMPT.md").read_text(encoding="utf-8")
        command = [*self.COMMAND]
        if command[0] == "codex":
            command += ["-m", self.model, "-C", str(sandbox)]
        log = sandbox / "codex.jsonl"
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=handle,
                stderr=subprocess.PIPE,
                text=True,
                cwd=sandbox,
                start_new_session=True,
            )
            try:
                _, stderr = process.communicate(input=prompt, timeout=self.timeout)
            except subprocess.TimeoutExpired:
                # The process can exit between the timeout firing and this
                # lookup, and a reaped pid has no process group: that is a
                # call that finished too late to count, not a kill that
                # failed, so it is recorded as the timeout it is.
                try:
                    pgid: int | None = os.getpgid(process.pid)
                except ProcessLookupError:
                    pgid = None
                if pgid is not None:
                    os.killpg(pgid, signal.SIGKILL)
                process.wait()
                child = _written(sandbox)
                LOGGER.warning(
                    "codex call for %s timed out after %ss, killed pgid %s; %s",
                    program_id,
                    self.timeout,
                    pgid,
                    "keeping the child it had written" if child else "no child",
                )
                tokens_in, tokens_out = _tokens(log)
                return Mutation(
                    program_id=program_id,
                    child=child,
                    status="timeout",
                    reason=f"{self.timeout}s (pgid {pgid})",
                    seconds=time.perf_counter() - started,
                    input_tokens=tokens_in,
                    output_tokens=tokens_out,
                )
        if process.returncode != 0:
            LOGGER.warning(
                "codex call for %s exited %s", program_id, process.returncode
            )
            return Mutation(
                program_id=program_id,
                child=None,
                status="exec_error",
                reason=(stderr or "")[-500:],
                seconds=time.perf_counter() - started,
                input_tokens=0,
                output_tokens=0,
            )
        tokens_in, tokens_out = _tokens(log)
        child = _written(sandbox)
        if child is None:
            reason = _last_message(log) or "child.py missing or empty"
            return Mutation(
                program_id=program_id,
                child=None,
                status="no_output",
                reason=reason,
                seconds=time.perf_counter() - started,
                input_tokens=tokens_in,
                output_tokens=tokens_out,
            )
        return Mutation(
            program_id=program_id,
            child=child,
            status="ok",
            reason="",
            seconds=time.perf_counter() - started,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
        )


def _written(sandbox: Path) -> Path | None:
    """``sandbox/child.py`` if the session wrote something there, else None."""
    child = sandbox / "child.py"
    if child.exists() and child.read_text(encoding="utf-8").strip():
        return child
    return None


def _tokens(log: Path) -> tuple[int, int]:
    """Sums token counts from codex's ``--json`` log.

    Args:
        log: Path to the ``codex.jsonl`` transcript.

    Returns:
        The total (input_tokens, output_tokens) across every
        ``turn.completed`` event; (0, 0) if the log has none.
    """
    tokens_in = tokens_out = 0
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "turn.completed":
            continue
        usage = event.get("usage") or {}
        tokens_in += int(usage.get("input_tokens", 0))
        tokens_out += int(usage.get("output_tokens", 0))
    return tokens_in, tokens_out


def _last_message(log: Path) -> str:
    """Returns the last ``agent_message`` text in a codex ``--json`` log.

    Args:
        log: Path to the ``codex.jsonl`` transcript.

    Returns:
        The first 200 characters of the last agent message, or "" if the
        log has none.
    """
    last = ""
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = event.get("item") or {}
        if (
            event.get("type") == "item.completed"
            and item.get("type") == "agent_message"
        ):
            last = item.get("text", "")
    return last[:200]


class FakeMutator:
    """Copies ``parent.py`` to ``child.py`` through ``edit``. For dry runs and tests."""

    def __init__(self, edit: Callable[[str], str]) -> None:
        """Initializes the mutator.

        Args:
            edit: Transforms the parent's source text into the child's.
        """
        self.edit = edit

    def __call__(self, sandbox: Path, program_id: str) -> Mutation:
        """Writes ``sandbox / "child.py"`` as ``edit`` of the parent's source.

        Args:
            sandbox: A directory holding ``parent.py``.
            program_id: The child program id.

        Returns:
            A `Mutation` with status "ok".
        """
        child = sandbox / "child.py"
        child.write_text(
            self.edit((sandbox / "parent.py").read_text(encoding="utf-8")),
            encoding="utf-8",
        )
        return Mutation(
            program_id=program_id,
            child=child,
            status="ok",
            reason="",
            seconds=0.0,
            input_tokens=0,
            output_tokens=0,
        )
