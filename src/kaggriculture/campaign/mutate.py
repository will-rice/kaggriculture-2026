"""One mutation: a codex session in a sandbox, or a fake that edits a constant.

The codex command is a class attribute so a test can replace it with ``true``
or ``sleep``; the rest of the module never changes between the fake and the
real thing.

Both mutators are awaitable, because the loop runs many sessions at once on
one event loop: codex is an ``asyncio`` child process, and the fake's file
work goes to a thread so an ``edit`` that sleeps cannot stall the loop.

A session's transcript runs to hundreds of kilobytes, so reading and parsing
it goes to a thread like everything else that blocks: the loop thread is
dispatching seven other sessions while this one is being totted up.

Codex 0.147's ``--json`` output is one JSON object per line. Token usage
lives on the ``turn.completed`` event, under ``usage.input_tokens`` and
``usage.output_tokens`` (verified against a real session log); other events
carry no usage and are ignored. Codex sometimes ends a turn with a question
instead of writing ``child.py``; ``_last_message`` recovers the last
``agent_message`` text so the caller knows why.
"""

import asyncio
import json
import logging
import os
import signal
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
        reason: Free-form explanation; empty on "ok". An "exec_error" carries
            the provider's own failure message when the transcript has one.
        seconds: Wall-clock time the call took.
        input_tokens: Total input tokens billed, summed across the call.
        output_tokens: Total output tokens billed, summed across the call.
        model: The model that produced this outcome.
        fallback: Whether that model was the fallback, after the first
            model's session failed.
    """

    program_id: str
    child: Path | None
    status: Literal["ok", "no_output", "timeout", "exec_error"]
    reason: str
    seconds: float
    input_tokens: int
    output_tokens: int
    model: str
    fallback: bool = False


class Mutator(Protocol):
    """Something that turns a sandbox into a child program, or a failure."""

    async def __call__(self, sandbox: Path, program_id: str) -> Mutation:
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
    (``start_new_session=True``) so a timeout -- or the cancellation that
    shutting the loop down delivers -- can kill the whole process group with
    ``os.killpg``, not just codex itself. Without the cancellation arm, a
    killed loop would leave ``CODEX_CONCURRENCY`` sessions running.
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
        model: str = config.CODEX_MODEL,
        fallback: str = config.CODEX_FALLBACK_MODEL,
        timeout: float = config.SESSION_LIMIT_SECONDS,
    ) -> None:
        """Initializes the mutator.

        Args:
            model: The codex model to request.
            fallback: The model to retry on, once, when a session on
                ``model`` fails without a verdict (the provider refused or
                codex crashed); "" to never retry.
            timeout: Seconds to allow the codex call before killing it.
        """
        self.model = model
        self.fallback = fallback
        self.timeout = timeout

    async def __call__(self, sandbox: Path, program_id: str) -> Mutation:
        """Runs one codex session in ``sandbox``, retrying once on the fallback.

        A session that ends in ``exec_error`` -- the provider failed the turn
        (a model at capacity, a rate limit) or codex itself died -- is run
        again on the fallback model in the same sandbox. The failed
        transcript is kept beside the new one as ``codex.<model>.failed.jsonl``.

        Args:
            sandbox: A directory built by ``prompt.build_sandbox``, holding
                ``AGENTS.md``, ``parent.py``, and ``PROMPT.md``.
            program_id: The child program id.

        Returns:
            A `Mutation` describing what happened, on whichever model
            produced it.
        """
        mutation = await self.session(sandbox, program_id, self.model)
        if mutation.status != "exec_error" or not self.fallback:
            return mutation
        LOGGER.warning(
            "codex call for %s failed on %s (%s); retrying on %s",
            program_id,
            self.model,
            mutation.reason[:120],
            self.fallback,
        )
        log = sandbox / "codex.jsonl"
        log.replace(log.with_name(f"codex.{self.model}.failed.jsonl"))
        retried = await self.session(sandbox, program_id, self.fallback)
        return retried.model_copy(
            update={"fallback": True, "seconds": mutation.seconds + retried.seconds}
        )

    async def session(self, sandbox: Path, program_id: str, model: str) -> Mutation:
        """Runs one codex session in ``sandbox`` on ``model``.

        The model is also exported to the child as ``CAMPAIGN_CODEX_MODEL``,
        which codex ignores and a stand-in command in a test can read.

        Args:
            sandbox: A directory built by ``prompt.build_sandbox``.
            program_id: The child program id.
            model: The codex model to request.

        Returns:
            A `Mutation` describing what happened.
        """
        started = time.perf_counter()
        prompt = (sandbox / "PROMPT.md").read_text(encoding="utf-8")
        command = [*self.COMMAND]
        if command[0] == "codex":
            command += ["-m", model, "-C", str(sandbox)]
        log = sandbox / "codex.jsonl"
        with log.open("w", encoding="utf-8") as handle:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=handle,
                stderr=asyncio.subprocess.PIPE,
                cwd=sandbox,
                start_new_session=True,
                env={**os.environ, "CAMPAIGN_CODEX_MODEL": model},
            )
            try:
                _, stderr = await asyncio.wait_for(
                    process.communicate(prompt.encode()), self.timeout
                )
            except TimeoutError:
                pgid = kill_group(process)
                await process.wait()
                child = _written(sandbox)
                LOGGER.warning(
                    "codex call for %s timed out after %ss, killed pgid %s; %s",
                    program_id,
                    self.timeout,
                    pgid,
                    "keeping the child it had written" if child else "no child",
                )
                tokens_in, tokens_out = await asyncio.to_thread(_tokens, log)
                return Mutation(
                    program_id=program_id,
                    child=child,
                    status="timeout",
                    reason=f"{self.timeout}s (pgid {pgid})",
                    seconds=time.perf_counter() - started,
                    input_tokens=tokens_in,
                    output_tokens=tokens_out,
                    model=model,
                )
            except asyncio.CancelledError:
                # The loop is shutting down. The session goes with it, whole
                # process group and all, or a restart would find eight codex
                # sessions still running against sandboxes nothing owns.
                LOGGER.warning(
                    "codex call for %s cancelled, killed pgid %s",
                    program_id,
                    kill_group(process),
                )
                await process.wait()
                raise
        if process.returncode != 0:
            reason = (
                await asyncio.to_thread(_failure, log)
                or stderr.decode(errors="replace")[-500:]
            )
            LOGGER.warning(
                "codex call for %s exited %s on %s: %s",
                program_id,
                process.returncode,
                model,
                reason[:200],
            )
            return Mutation(
                program_id=program_id,
                child=None,
                status="exec_error",
                reason=reason,
                seconds=time.perf_counter() - started,
                input_tokens=0,
                output_tokens=0,
                model=model,
            )
        tokens_in, tokens_out = await asyncio.to_thread(_tokens, log)
        child = _written(sandbox)
        if child is None:
            reason = (
                await asyncio.to_thread(_last_message, log)
                or "child.py missing or empty"
            )
            return Mutation(
                program_id=program_id,
                child=None,
                status="no_output",
                reason=reason,
                seconds=time.perf_counter() - started,
                input_tokens=tokens_in,
                output_tokens=tokens_out,
                model=model,
            )
        return Mutation(
            program_id=program_id,
            child=child,
            status="ok",
            reason="",
            seconds=time.perf_counter() - started,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            model=model,
        )


def kill_group(process: asyncio.subprocess.Process) -> int | None:
    """SIGKILL a session's whole process group and return the group it killed.

    Args:
        process: The codex child, started with ``start_new_session=True``.

    Returns:
        The process group killed, or None if the process had already been
        reaped -- a call that finished too late to count, not a kill that
        failed, so the caller reports it as whatever ended the wait.
    """
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        return None
    os.killpg(pgid, signal.SIGKILL)
    return pgid


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


def _failure(log: Path) -> str:
    """The provider's message from a ``turn.failed`` event, or "".

    Codex wraps the provider's error as a JSON string inside the event's own
    message (``{"type":"error","error":{"message":...}}``); a plain string
    is returned as it is.

    Args:
        log: Path to the ``codex.jsonl`` transcript.

    Returns:
        The first 300 characters of the failure message, or "" if the log
        has no ``turn.failed`` event.
    """
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "turn.failed":
            continue
        message = (event.get("error") or {}).get("message", "")
        try:
            inner = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            return str(message)[:300]
        if isinstance(inner, dict):
            error = inner.get("error") or {}
            message = error.get("message") or inner.get("message") or message
        return str(message)[:300]
    return ""


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

    async def __call__(self, sandbox: Path, program_id: str) -> Mutation:
        """Writes ``sandbox / "child.py"`` as ``edit`` of the parent's source.

        The edit runs in a thread, as the codex session it stands in for runs
        in a child process: a dry run with a slow ``edit`` must exercise the
        loop's concurrency rather than stall its single thread.

        Args:
            sandbox: A directory holding ``parent.py``.
            program_id: The child program id.

        Returns:
            A `Mutation` with status "ok".
        """
        started = time.perf_counter()
        child = sandbox / "child.py"
        await asyncio.to_thread(self.write, sandbox, child)
        return Mutation(
            program_id=program_id,
            child=child,
            status="ok",
            reason="",
            seconds=time.perf_counter() - started,
            input_tokens=0,
            output_tokens=0,
            model="fake",
        )

    def write(self, sandbox: Path, child: Path) -> None:
        """Read the parent, apply ``edit``, write the child. Runs in a thread.

        Args:
            sandbox: A directory holding ``parent.py``.
            child: Where to write the edited source.
        """
        child.write_text(
            self.edit((sandbox / "parent.py").read_text(encoding="utf-8")),
            encoding="utf-8",
        )
