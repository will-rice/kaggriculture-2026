"""One mutation: one model call on one file, or a fake that edits a constant.

A call is given a directory holding ``child.py`` and nothing else, and the
whole message -- on standard input for codex, as an argument for agy. It edits
that file and stops; the loop reads it back, scores it, and composes the next
round's message from the result.

Two programs can drive a round. `config.MUTATOR` chooses, and `build` returns
it. They exist side by side because an entitlement ran out rather than because
either is better: codex has no quota on this account until 2026-09-22, and agy
bills a different one.

Neither program's own report of how it went is read. Both will exit 0 having
done nothing -- codex ends a turn with a question instead of an edit, and agy
returns ``"status": "SUCCESS"`` when its ``--print-timeout`` expires mid-turn,
after announcing that it finished. `_written` compares the file against what
the round was handed, and that comparison is the whole verdict.

Each command is a class attribute so a test can replace it with ``true``
or ``sleep``; the rest of the module never changes between the fakes and the
real things.

Both mutators are awaitable, because the loop runs many rounds at once on
one event loop: codex is an ``asyncio`` child process, and the fake's file
work goes to a thread so an ``edit`` that sleeps cannot stall the loop.

A call's transcript runs to hundreds of kilobytes, so reading and parsing
it goes to a thread like everything else that blocks: the loop thread is
dispatching seven other rounds while this one is being totted up.

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
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

from dotenv import load_dotenv
from pydantic import BaseModel

from kaggriculture.campaign import config, plan

LOGGER = logging.getLogger(__name__)

# Reads the model catalog this login's codex is entitled to -- a local
# lookup, not a model turn, so no quota is spent running it. A test replaces
# this with a command that prints a small catalog of its own.
MODEL_CATALOG_COMMAND = ["codex", "debug", "models"]
# The same for agy. Also a local lookup rather than a model turn: it spends
# no quota, and `agy -p /usage` is how the quota itself is read.
AGY_CATALOG_COMMAND = ["agy", "models"]
# And OpenCode's, which lists every provider this install is signed
# into at once, so the catalog is the union rather than one seller's.
OPENCODE_CATALOG_COMMAND = ["opencode", "models"]


def known_models() -> set[str]:
    """The model slugs this codex login's catalog reports.

    Returns:
        Every ``slug`` in ``MODEL_CATALOG_COMMAND``'s JSON output.
    """
    output = subprocess.run(
        MODEL_CATALOG_COMMAND, capture_output=True, check=True, text=True
    ).stdout
    return {model["slug"] for model in json.loads(output)["models"]}


# The file the model is read from, absolute on purpose. `load_dotenv` with no
# path searches relative to something -- the caller's module, or the working
# directory -- and the campaign moves its working directory: every worker that
# runs a program is given a scratch one. A relative search would find a
# different file depending on who asked.
ENV = config.ROOT / ".env"


def model() -> str:
    """The model to ask for, read fresh at every call.

    Read here rather than captured at startup so that it can change without
    stopping the campaign. `.env` is reloaded first, which is what makes that
    true: a process's environment is fixed when it is spawned, so exporting a
    variable in a shell cannot reach a loop that is already running, and the
    file is the part of the environment a running process can re-read.

    So: edit `CAMPAIGN_CODEX_MODEL` in `.env` and the next round uses it. The
    wandb run keeps the name it started with, because that is what it started
    with; `calls/model` is logged per call and is the truth about any one of
    them.

    Returns:
        The slug from the environment, or `config.CODEX_MODEL`.
    """
    load_dotenv(ENV, override=True)
    return os.environ.get("CAMPAIGN_CODEX_MODEL") or config.CODEX_MODEL


def fallback() -> str:
    """The model retried once when the first fails without a verdict.

    Read at the call for the same reason, and "" to never retry.

    Returns:
        The slug from the environment, or `config.CODEX_FALLBACK_MODEL`.
    """
    load_dotenv(ENV, override=True)
    value = os.environ.get("CAMPAIGN_CODEX_FALLBACK_MODEL")
    return config.CODEX_FALLBACK_MODEL if value is None else value


def validate_model(model: str) -> None:
    """Fails fast when ``model`` is not a slug this codex login recognizes.

    A typo'd model is hundreds of failed sessions discovered one at a time --
    this login accepts ``gpt-6-astra`` but refuses ``gpt-5.6-astra``, so it is
    not hypothetical. Called once at startup, before any session spends a
    call on a name that was never going to work.

    Args:
        model: The model to check.

    Raises:
        SystemExit: ``model`` is not in ``known_models()``.
    """
    if model not in known_models():
        raise SystemExit(
            f"{model!r} is not a model this codex login knows about "
            "(run `codex debug models` to check the spelling)"
        )


def agy_model() -> str:
    """The agy model to ask for, read fresh at every call.

    Read from `.env` at the call for the reason `model` is: a running
    campaign cannot be reached by exporting a variable, and the file is the
    part of its environment it can re-read.

    Returns:
        The slug from the environment, or `config.AGY_MODEL`.
    """
    load_dotenv(ENV, override=True)
    return os.environ.get("CAMPAIGN_AGY_MODEL") or config.AGY_MODEL


def agy_fallback() -> str:
    """The agy model retried once when the first fails without a verdict.

    Returns:
        The slug from the environment, or `config.AGY_FALLBACK_MODEL`.
    """
    load_dotenv(ENV, override=True)
    value = os.environ.get("CAMPAIGN_AGY_FALLBACK_MODEL")
    return config.AGY_FALLBACK_MODEL if value is None else value


def known_agy_models() -> set[str]:
    """The model slugs this agy login is entitled to.

    ``agy models`` prints one ``slug<TAB>label`` line per model on standard
    output and its progress line on standard error, so the parse is the first
    field of every line. It documents an ``--output-format json`` that agy
    1.2.4 does not have -- the flag is refused outright -- which is why this
    reads the text.

    Returns:
        Every slug the catalog lists.
    """
    output = subprocess.run(
        AGY_CATALOG_COMMAND, capture_output=True, check=True, text=True
    ).stdout
    return {
        line.split("\t", 1)[0].strip() for line in output.splitlines() if line.strip()
    }


def validate_agy_model(model: str) -> None:
    """Fails fast when ``model`` is not a slug this agy login recognizes.

    agy does fail a bad slug itself, loudly and non-zero. It fails it once
    per round, though, and a campaign that discovers its model was misspelled
    one session at a time has spent the night finding out.

    Args:
        model: The model to check.

    Raises:
        SystemExit: ``model`` is not in `known_agy_models`.
    """
    if model not in known_agy_models():
        raise SystemExit(
            f"{model!r} is not a model this agy login knows about "
            "(run `agy models` to check the spelling)"
        )


class Mutation(BaseModel):
    """The outcome of one mutation call.

    Attributes:
        program_id: The child program id this call was made for.
        child: Path to the written child program, or None if there is
            nothing to evaluate. A timed-out call that had already
            written ``child.py`` still yields it: the file is what gets
            evaluated, however the call ended.
        status: "ok", "no_output" (ran but wrote nothing usable), or
            "exec_error".
        reason: Free-form explanation; empty on "ok". An "exec_error" carries
            the provider's own failure message when the transcript has one.
        seconds: Wall-clock time the call took.
        input_tokens: Total input tokens billed, summed across the call.
        output_tokens: Total output tokens billed, summed across the call.
        model: The model that produced this outcome.
        fallback: Whether that model was the fallback, after the first
            model's call failed.
    """

    program_id: str
    child: Path | None
    status: Literal["ok", "no_output", "exec_error"]
    reason: str
    seconds: float
    input_tokens: int
    output_tokens: int
    model: str
    fallback: bool = False


class Mutator(Protocol):
    """Something that turns one file and one message into a child program."""

    # Where this program discovers skills, relative to the workspace. The
    # loop copies `config.SKILLS` there, and the two programs do not agree:
    # codex reads `.codex/skills` under its working directory, agy walks up
    # from it looking for `.agents`. Hardcoding either in the loop is how a
    # round silently loses the schema and the queries worth running -- it
    # still runs, it just never finds them -- so the mutator names its own.
    SKILLS_DIR: Path
    # And what this driver calls the transcript it leaves behind, as a glob,
    # because a retry writes a second one beside the first. Named here for the
    # same reason `SKILLS_DIR` is: the loop copies it out and has no business
    # knowing which program wrote it.
    TRANSCRIPTS: str

    async def __call__(
        self, workspace: Path, message: str, program_id: str
    ) -> Mutation:
        """Mutates ``workspace / "child.py"`` in place.

        Args:
            workspace: A directory holding ``child.py`` and nothing else.
            message: The whole prompt, composed by ``prompt.compose``.
            program_id: The child program id.

        Returns:
            A `Mutation` describing the outcome.
        """
        ...


class CodexMutator:
    """Runs one ``codex exec`` over one file and reports the result.

    ``COMMAND`` is overridden by tests (e.g. to ``["true"]``) to exercise the
    no-output and failure paths without spending a real codex call; the
    ``-m``/``-C`` flags are only appended when the command actually is codex.

    A call runs until it is done. There is no cap: a round runs the skills
    this login has installed, and the only cap this ever had cut calls off
    before they had written anything. So the one thing that ends a call early
    is the cancellation that shutting the loop down delivers. Codex spawns
    shell commands as tool calls in ``workspace-write`` mode, and killing only
    the direct child would orphan any grandchild still running, leaking a core
    that ``config.CORE_BUDGET`` assumes is free; the process runs in its own
    session (``start_new_session=True``) so the cancellation arm can take the
    whole process group with ``os.killpg``. Without it, a killed loop would
    leave ``SESSIONS`` codex calls running.
    """

    # `--skip-git-repo-check` because a call runs in a temporary directory
    # holding one file, not in a repository: codex refuses an untrusted
    # directory otherwise, and the directory is deliberately not one.
    SKILLS_DIR = Path(".codex") / "skills"
    TRANSCRIPTS = "codex*.jsonl"

    COMMAND = [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "-s",
        "workspace-write",
        "-c",
        "approval_policy=never",
        "--json",
        "-",
    ]

    def __init__(self, model: str = "", fallback: str | None = None) -> None:
        """Initializes the mutator.

        Both are normally left unset and read from the environment at every
        call, so the campaign's model can change without stopping it. A test
        pins them instead.

        Args:
            model: The codex model to request, or "" to read it per call.
            fallback: The model to retry on, once, when a call on ``model``
                fails without a verdict (the provider refused or codex
                crashed). "" never retries; None reads it per call.
        """
        self.model = model
        self.fallback = fallback

    async def __call__(
        self, workspace: Path, message: str, program_id: str
    ) -> Mutation:
        """Runs one codex call in ``workspace``, retrying once on the fallback.

        A call that ends in ``exec_error`` -- the provider failed the turn
        (a model at capacity, a rate limit) or codex itself died -- is run
        again on the fallback model over the same file. The failed transcript
        is kept beside the new one as ``codex.<model>.failed.jsonl``.

        Args:
            workspace: A directory holding ``child.py`` and nothing else.
            message: The whole prompt, composed by ``prompt.compose``.
            program_id: The child program id.

        Returns:
            A `Mutation` describing what happened, on whichever model
            produced it.
        """
        # Read here, not at startup: the model is allowed to change under a
        # running campaign, and a round asks for whatever it is now.
        asked = self.model or model()
        retry = fallback() if self.fallback is None else self.fallback
        mutation = await self.call(workspace, message, program_id, asked)
        if mutation.status != "exec_error" or not retry:
            return mutation
        LOGGER.warning(
            "codex call for %s failed on %s (%s); retrying on %s",
            program_id,
            asked,
            mutation.reason[:120],
            retry,
        )
        log = workspace / "codex.jsonl"
        log.replace(log.with_name(f"codex.{asked}.failed.jsonl"))
        retried = await self.call(workspace, message, program_id, retry)
        return retried.model_copy(
            update={"fallback": True, "seconds": mutation.seconds + retried.seconds}
        )

    async def call(
        self, workspace: Path, message: str, program_id: str, model: str
    ) -> Mutation:
        """Runs one codex call in ``workspace`` on ``model``.

        The model is also exported to the child as ``CAMPAIGN_CODEX_MODEL``,
        which codex ignores and a stand-in command in a test can read.

        Args:
            workspace: A directory holding ``child.py`` and nothing else.
            message: The whole prompt, written to codex's standard input.
            program_id: The child program id.
            model: The codex model to request.

        Returns:
            A `Mutation` describing what happened.
        """
        started = time.perf_counter()
        # What the round was handed. A call that ends with the file exactly
        # as it found it has written nothing, however cleanly it exited.
        given = plan.gather(workspace)
        command = [*self.COMMAND]
        if command[0] == "codex":
            command += ["-m", model, "-C", str(workspace)]
            # Passed per call rather than left to `~/.codex/config.toml`, so
            # the campaign's effort is the campaign's decision and does not
            # move when the host edits its own codex settings.
            command += ["-c", f"model_reasoning_effort={config.CODEX_REASONING}"]
        log = workspace / "codex.jsonl"
        with log.open("w", encoding="utf-8") as handle:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=handle,
                stderr=asyncio.subprocess.PIPE,
                cwd=workspace,
                start_new_session=True,
                env={**os.environ, "CAMPAIGN_CODEX_MODEL": model},
            )
            try:
                _, stderr = await process.communicate(message.encode())
            except asyncio.CancelledError:
                # The loop is shutting down. The call goes with it, whole
                # process group and all, or a restart would find eight codex
                # calls still running against directories nothing owns.
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
        child = _written(workspace, given)
        if child is None:
            reason = (
                await asyncio.to_thread(_last_message, log)
                or "child.py missing, empty or unchanged"
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


class AgyMutator:
    """Runs one ``agy --print`` over one file and reports the result.

    The second way to run a round, because the first one ran out: codex bills
    an entitlement that is empty until 2026-09-22 and `agy` bills a different
    one. Everything about the shape is the same -- one directory, one file,
    one message, a verdict the loop decides for itself -- and three things
    about the mechanics are not.

    The message goes on the command line, not standard input: ``--print``
    takes its prompt as a value, and a valueless ``-p`` is an error rather
    than a read from stdin. (``--input-format stream-json`` does read stdin,
    one turn per NDJSON line, which is a session driver and not this.)

    ``--add-dir`` is not optional. Without it the workspace is outside
    anything the call may touch, and the failure is quiet in the worst way:
    on 2026-09-16 a probe run without it reported having read a file it
    never opened and set a value it had invented. The directory is passed
    both as ``cwd`` -- which is where agy starts walking to find
    ``.agents`` -- and as ``--add-dir``, which is what actually grants it.

    ``"status": "SUCCESS"`` and an exit code of 0 mean neither that the round
    finished nor that it did anything. Measured on agy 1.2.4: a call whose
    ``--print-timeout`` expired mid-turn returned its partial answer, said
    ``SUCCESS``, exited 0, and left ``child.py`` exactly as it was found --
    having announced in its own last message that it had completed the task.
    So the verdict here is what it is for codex and for the same reason:
    ``_written`` compares the file against what the round was handed, and
    nothing the round says about itself is read.

    Two flags are deliberately absent. ``--sandbox`` restricts the terminal,
    and a round has to be able to run ``measure.py``; commands already run
    without prompting because the host's settings say
    ``toolPermission: proceed-in-sandbox``. ``--disable-slash-commands``
    would stop skill expansion, and the workspace skill in ``SKILLS_DIR`` is
    the point -- it is discovered even in a temporary directory that is not a
    repository (probed 2026-09-16). Skills are disclosed progressively, so
    the host's own eight cost their descriptions and nothing more unless a
    round chooses to open one.
    """

    SKILLS_DIR = Path(".agents") / "skills"
    TRANSCRIPTS = "agy*.jsonl"

    # Overridden by tests with something like ``["true"]``; the message and
    # the flags are only appended when the command actually is agy.
    COMMAND = ["agy", "--print"]

    def __init__(self, model: str = "", fallback: str | None = None) -> None:
        """Initializes the mutator.

        Args:
            model: The agy model to request, or "" to read it per call.
            fallback: The model to retry on, once, when a call on ``model``
                fails without a verdict. "" never retries; None reads it per
                call.
        """
        self.model = model
        self.fallback = fallback

    async def __call__(
        self, workspace: Path, message: str, program_id: str
    ) -> Mutation:
        """Runs one agy call in ``workspace``, retrying once on the fallback.

        Args:
            workspace: A directory holding ``child.py`` and nothing else.
            message: The whole prompt, composed by ``prompt.compose``.
            program_id: The child program id.

        Returns:
            A `Mutation` describing what happened, on whichever model
            produced it.
        """
        asked = self.model or agy_model()
        retry = agy_fallback() if self.fallback is None else self.fallback
        mutation = await self.call(workspace, message, program_id, asked)
        if mutation.status != "exec_error" or not retry:
            return mutation
        LOGGER.warning(
            "agy call for %s failed on %s (%s); retrying on %s",
            program_id,
            asked,
            mutation.reason[:120],
            retry,
        )
        log = workspace / "agy.jsonl"
        log.replace(log.with_name(f"agy.{asked}.failed.jsonl"))
        retried = await self.call(workspace, message, program_id, retry)
        return retried.model_copy(
            update={"fallback": True, "seconds": mutation.seconds + retried.seconds}
        )

    def invocation(self, workspace: Path, message: str, model: str) -> list[str]:
        """The whole command line one round runs as.

        Separate from `call` so that the flags can be asserted on without
        spending a call. Losing one of them does not fail loudly -- it produces
        a round that reads its file, cannot measure anything, and reports
        `no_output` -- so they are worth a test.

        Args:
            workspace: The directory the round works in.
            message: The whole prompt, passed as ``--print``'s value.
            model: The agy model to request.

        Returns:
            The argument vector, unchanged when ``COMMAND`` is a stand-in.
        """
        command = [*self.COMMAND]
        if command[0] != "agy":
            return command
        return command + [
            message,
            # One JSON object per line: typed `init` and `step_update` events
            # as the round works, and a terminal `result` event carrying the
            # answer and the token usage. The plain `json` format returns that
            # same result object and nothing of the steps.
            "--output-format",
            "stream-json",
            "--mode",
            "accept-edits",
            "--add-dir",
            str(workspace),
            # Without this a round can read and edit its own file and run
            # nothing that matters. agy confines a command to the directories
            # its workspace grants, and the interpreter the loop runs under
            # lives outside them: `python measure.py` came back "command not
            # found" while that interpreter was on the PATH the round had
            # inherited, and `python3` found the system 3.10 with no
            # `kaggriculture` in it. A round that cannot measure is worth
            # nothing -- the one measured this way spent 226 seconds and 75,000
            # tokens hunting the filesystem for the package and edited nothing.
            #
            # Granting the directories instead was measured and is worse: it
            # takes the round's own box out of the middle of its workspace, and
            # the round stops being able to find `child.py` at all, hunting the
            # repository instead at 265,000 tokens a round. It also hands a
            # round `gate.py`, which is the program that scores it.
            #
            # So this, which is what a codex round has always run as: `-s
            # workspace-write` with `approval_policy=never` is the same posture
            # and a wider one. What either reaches is a throwaway directory
            # holding one agent, and the loop still plays every scored game
            # itself against opponents the round never sees.
            "--dangerously-skip-permissions",
            "--model",
            model,
            "--print-timeout",
            config.AGY_TIMEOUT,
        ]

    async def call(
        self, workspace: Path, message: str, program_id: str, model: str
    ) -> Mutation:
        """Runs one agy call in ``workspace`` on ``model``.

        Args:
            workspace: A directory holding ``child.py`` and nothing else.
            message: The whole prompt, passed as ``--print``'s value.
            program_id: The child program id.
            model: The agy model to request.

        Returns:
            A `Mutation` describing what happened.
        """
        started = time.perf_counter()
        given = plan.gather(workspace)
        command = self.invocation(workspace, message, model)
        log = workspace / "agy.jsonl"
        with log.open("w", encoding="utf-8") as handle:
            process = await asyncio.create_subprocess_exec(
                *command,
                # The prompt is an argument here, so nothing is written in.
                # Closed rather than inherited: agy reads a controlling
                # terminal for an OAuth code when its stdin is a pipe, and a
                # round that waits on one nobody is watching waits for hours.
                stdin=asyncio.subprocess.DEVNULL,
                stdout=handle,
                stderr=asyncio.subprocess.PIPE,
                cwd=workspace,
                start_new_session=True,
                env={**os.environ, "CAMPAIGN_AGY_MODEL": model},
            )
            try:
                _, stderr = await process.communicate()
            except asyncio.CancelledError:
                # agy runs shell commands, schedules background tasks and can
                # invoke subagents of its own, and it only terminates those on
                # a clean exit -- which a killed parent never reaches. So the
                # group goes, or a restart finds a round still playing seasons
                # against a directory nothing owns and a core the budget has
                # already promised to somebody else.
                LOGGER.warning(
                    "agy call for %s cancelled, killed pgid %s",
                    program_id,
                    kill_group(process),
                )
                await process.wait()
                raise
        result = await asyncio.to_thread(_agy_result, log)
        if process.returncode != 0:
            # Only a cascade-level failure gets here: a model slug agy cannot
            # resolve, an interrupt, a crash. A refused tool or a failed
            # command inside the round is not one of them and exits 0.
            reason = _agy_reason(result) or stderr.decode(errors="replace")[-500:]
            LOGGER.warning(
                "agy call for %s exited %s on %s: %s",
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
        usage = result.get("usage") or {}
        if result.get("status") not in ("SUCCESS", None):
            # The run failed without reaching a verdict, and said so in the
            # one place that carries it: agy still exits 0. Measured on
            # 2026-09-16, when a round that had read the program, played 32
            # seasons and queried the corpus ended on `"status": "ERROR"` with
            # "Individual quota reached ... Resets in 4h18m40s" -- eight
            # hundred seconds of real work, and as `no_output` it would have
            # been spent for nothing and never retried. This is what the
            # fallback is for, and why the fallback is on the other pool.
            reason = _agy_reason(result)
            LOGGER.warning(
                "agy call for %s failed without a verdict on %s: %s",
                program_id,
                model,
                reason[:200],
            )
            return Mutation(
                program_id=program_id,
                child=None,
                status="exec_error",
                reason=reason,
                seconds=time.perf_counter() - started,
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
                model=model,
            )
        child = _written(workspace, given)
        if child is None:
            return Mutation(
                program_id=program_id,
                child=None,
                status="no_output",
                reason=_agy_reason(result) or "child.py missing, empty or unchanged",
                seconds=time.perf_counter() - started,
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
                model=model,
            )
        return Mutation(
            program_id=program_id,
            child=child,
            status="ok",
            reason="",
            seconds=time.perf_counter() - started,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            model=model,
        )


class OpenCodeMutator:
    """Runs one ``opencode run`` over one file and reports the result.

    The third driver, and the first whose limit is money rather than a clock.
    codex bills an entitlement that is empty until 2026-09-22, and agy bills two
    five-hour buckets that eight sessions drained in twenty minutes; OpenCode
    reaches a seller that charges per token, so a round costs about five cents
    and how many rounds are left in a day is a question about the budget.

    Three things it needs, all of them documented rather than discovered.

    ``--dir`` names the round's directory, and it is the whole of the
    containment. Without it OpenCode resolves its project by walking up from
    wherever the caller happened to be standing, and the caller is the loop,
    standing in the repository: the first probe read the campaign's source, ran
    its git history, listed ``run/campaign`` and played 32 real games there,
    while the directory it had been handed sat empty in /tmp. ``--dir`` is
    authoritative over both the process's own directory and the ``PWD`` it
    inherits, measured against a deliberately hostile pair of both.

    ``external_directory`` stays denied. ``--auto`` would approve it along with
    everything else, and a round that can be talked into reading one path
    outside its box can be talked into reading the champion archive. Denying it
    by policy means containment does not rest on the round's goodwill; what a
    round still runs freely is its own directory, which is all a round is.

    Skills are left alone. ``.agents/skills`` is one of OpenCode's own discovery
    paths -- the same directory agy walks up for -- so a round finds the schema
    and the queries worth running without being told, and the host's own skills
    arrive with them. That costs a few thousand tokens and is worth paying:
    those are the skills that tell a model to measure before it concludes.
    """

    SKILLS_DIR = Path(".agents") / "skills"
    TRANSCRIPTS = "opencode*.ndjson"

    # Overridden by tests; the flags are only appended when it really is
    # opencode.
    COMMAND = ["opencode", "run"]

    # The campaign's say over the host's, merged last of every config source.
    # The host's own `opencode.jsonc` is tuned for a different project
    # altogether, and a round should not inherit a file the campaign does not
    # own -- the same reason the codex effort is passed per call.
    POLICY = {
        "permission": {
            "bash": "allow",
            "edit": "allow",
            "read": "allow",
            "glob": "allow",
            "grep": "allow",
            "skill": {"*": "allow"},
            # The one that is not a convenience.
            "external_directory": "deny",
            # A round that asks a question in a room with nobody in it has
            # spent its turn; better it never reaches for the tool.
            "question": "deny",
        }
    }

    def __init__(self, model: str = "", fallback: str | None = None) -> None:
        """Initializes the mutator.

        Args:
            model: The ``provider/model`` to request, or "" to read it per call.
            fallback: The model to retry on, once, when a call fails without a
                verdict. "" never retries; None reads it per call.
        """
        self.model = model
        self.fallback = fallback

    async def __call__(
        self, workspace: Path, message: str, program_id: str
    ) -> Mutation:
        """Runs one opencode call in ``workspace``, retrying once on the fallback.

        Args:
            workspace: A directory holding ``child.py`` and nothing else.
            message: The whole prompt, composed by ``prompt.compose``.
            program_id: The child program id.

        Returns:
            A `Mutation` describing what happened, on whichever model produced
            it.
        """
        asked = self.model or opencode_model()
        retry = opencode_fallback() if self.fallback is None else self.fallback
        mutation = await self.call(workspace, message, program_id, asked)
        if mutation.status != "exec_error" or not retry:
            return mutation
        LOGGER.warning(
            "opencode call for %s failed on %s (%s); retrying on %s",
            program_id,
            asked,
            mutation.reason[:120],
            retry,
        )
        log = workspace / "opencode.ndjson"
        spent = asked.replace("/", "-")
        log.replace(log.with_name(f"opencode.{spent}.failed.ndjson"))
        retried = await self.call(workspace, message, program_id, retry)
        return retried.model_copy(
            update={"fallback": True, "seconds": mutation.seconds + retried.seconds}
        )

    def invocation(self, workspace: Path, message: str, model: str) -> list[str]:
        """The whole command line one round runs as.

        Separate from `call` so the flags can be asserted on without spending a
        call, because none of them fails loudly: a round that loses ``--dir``
        still runs, and reports on a directory that is not its own.

        Args:
            workspace: The directory the round works in.
            message: The whole prompt.
            model: The ``provider/model`` to request.

        Returns:
            The argument vector, unchanged when ``COMMAND`` is a stand-in.
        """
        command = [*self.COMMAND]
        if command[0] != "opencode":
            return command
        return command + [
            message,
            "--dir",
            str(workspace),
            "--format",
            "json",
            "-m",
            model,
        ]

    async def call(
        self, workspace: Path, message: str, program_id: str, model: str
    ) -> Mutation:
        """Runs one opencode call in ``workspace`` on ``model``.

        Args:
            workspace: A directory holding ``child.py`` and nothing else.
            message: The whole prompt.
            program_id: The child program id.
            model: The ``provider/model`` to request.

        Returns:
            A `Mutation` describing what happened.
        """
        started = time.perf_counter()
        given = plan.gather(workspace)
        log = workspace / "opencode.ndjson"
        with log.open("w", encoding="utf-8") as handle:
            process = await asyncio.create_subprocess_exec(
                *self.invocation(workspace, message, model),
                # The prompt is an argument; nothing is written in, and a round
                # that blocks reading a terminal nobody is at blocks for hours.
                stdin=asyncio.subprocess.DEVNULL,
                stdout=handle,
                stderr=asyncio.subprocess.PIPE,
                cwd=workspace,
                start_new_session=True,
                env={
                    **os.environ,
                    "OPENCODE_CONFIG_CONTENT": json.dumps(self.POLICY),
                },
            )
            try:
                _, stderr = await process.communicate()
            except asyncio.CancelledError:
                # opencode runs shell commands of its own, so killing only the
                # direct child would leave a season playing against a directory
                # nothing owns, on a core the budget has already promised out.
                LOGGER.warning(
                    "opencode call for %s cancelled, killed pgid %s",
                    program_id,
                    kill_group(process),
                )
                await process.wait()
                raise
        spent = await asyncio.to_thread(_opencode_usage, log)
        if process.returncode != 0:
            reason = stderr.decode(errors="replace")[-500:] or "opencode exited nonzero"
            LOGGER.warning(
                "opencode call for %s exited %s on %s: %s",
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
                input_tokens=spent.input_tokens,
                output_tokens=spent.output_tokens,
                model=model,
            )
        child = _written(workspace, given)
        if child is None:
            return Mutation(
                program_id=program_id,
                child=None,
                status="no_output",
                reason=await asyncio.to_thread(_opencode_last_text, log)
                or "child.py missing, empty or unchanged",
                seconds=time.perf_counter() - started,
                input_tokens=spent.input_tokens,
                output_tokens=spent.output_tokens,
                model=model,
            )
        # The only driver that says what it charged, so it is worth saying.
        LOGGER.info(
            "opencode round %s wrote a child on %s for $%.4f",
            program_id,
            model,
            spent.dollars,
        )
        return Mutation(
            program_id=program_id,
            child=child,
            status="ok",
            reason="",
            seconds=time.perf_counter() - started,
            input_tokens=spent.input_tokens,
            output_tokens=spent.output_tokens,
            model=model,
        )


def kill_group(process: asyncio.subprocess.Process) -> int | None:
    """SIGKILL a session's whole process group and return the group it killed.

    Args:
        process: The codex process, started with ``start_new_session=True``.

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


def _written(workspace: Path, given: str) -> Path | None:
    """``workspace/child.py`` if the call left something new there, else None.

    A call that ended without touching the file leaves the very program it
    was asked to improve. Scoring that again would insert a duplicate, spend
    an evaluation on it, and tell the telemetry a child was produced when
    none was.

    Args:
        workspace: The directory the call worked in.
        given: What ``child.py`` held before the call.

    Returns:
        The child, or None if it is missing, empty, or unchanged.
    """
    child = workspace / "child.py"
    if not child.exists():
        return None
    # The whole program, which is `child.py` with whatever `plan.json` holds
    # packed back into it. A round that changed only the plan changed the
    # agent, and reading `child.py` alone would report that as nothing.
    source = plan.gather(workspace)
    if not source.strip() or source == given:
        return None
    return child


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


def _agy_result(log: Path) -> dict:
    """The payload of agy's terminal ``result`` event, or {}.

    Args:
        log: Path to the ``agy.jsonl`` transcript.

    Returns:
        The last ``result`` event's payload -- ``status``, ``response``,
        ``usage``, and ``denied_actions`` when the round was refused
        something -- or {} if the stream ended before one was written.
    """
    found: dict = {}
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "result":
            found = event.get("result") or {}
    return found


def _agy_reason(result: dict) -> str:
    """Why a call is worth no child, read off its ``result`` event.

    ``denied_actions`` comes first when it is there. It is the difference
    between a round that would not make the edit and a round that was not
    allowed to, and it names the rule that would have permitted it -- which
    is the distinction a day was lost to on 2026-09-16, when a missing
    ``--add-dir`` was read as a model inventing an answer.

    Args:
        result: The payload from `_agy_result`.

    Returns:
        A short explanation, or "" when the result says nothing useful.
    """
    parts = []
    status = result.get("status", "")
    if status and status != "SUCCESS":
        parts.append(str(status))
    error = result.get("error")
    if error:
        parts.append(str(error))
    denied = result.get("denied_actions")
    if denied:
        parts.append(f"denied: {denied}")
    response = str(result.get("response", "")).strip()
    if response:
        parts.append(response[:200])
    return "; ".join(parts)[:300]


class Spent(BaseModel):
    """What one opencode call billed.

    Attributes:
        input_tokens: Prompt tokens, summed across the call's steps.
        output_tokens: Completion tokens, summed the same way.
        dollars: What the seller charged, which only this driver reports.
    """

    input_tokens: int
    output_tokens: int
    dollars: float


def _opencode_usage(log: Path) -> Spent:
    """Sums the tokens and the cost across an opencode run.

    Every ``step_finish`` event carries a `tokens` object and a `cost`, and a
    round is many steps, so the call's total is their sum. Cached prompt tokens
    are reported separately by the seller and are not added here: they are
    charged at a different rate, and what this feeds is a token count the other
    drivers report the same way.

    Args:
        log: Path to the ``opencode.ndjson`` transcript.

    Returns:
        The totals, zeroed when the stream carries none.
    """
    tokens_in = tokens_out = 0
    dollars = 0.0
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "step_finish":
            continue
        part = event.get("part") or {}
        tokens = part.get("tokens") or {}
        tokens_in += int(tokens.get("input", 0))
        tokens_out += int(tokens.get("output", 0))
        dollars += float(part.get("cost", 0.0))
    return Spent(input_tokens=tokens_in, output_tokens=tokens_out, dollars=dollars)


def _opencode_last_text(log: Path) -> str:
    """The last thing the round said, for a call that wrote nothing.

    Args:
        log: Path to the ``opencode.ndjson`` transcript.

    Returns:
        The first 200 characters of the last ``text`` event, or "".
    """
    last = ""
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "text":
            last = str((event.get("part") or {}).get("text", "")) or last
    return last[:200]


class FakeMutator:
    """Edits ``child.py`` in place through ``edit``. For dry runs and tests."""

    # Nothing reads them, but the loop still lays them out, so a dry run
    # exercises the same copy a real round does. It writes no transcript, and
    # the glob simply matches nothing.
    SKILLS_DIR = Path(".agents") / "skills"
    TRANSCRIPTS = "fake*.jsonl"

    def __init__(self, edit: Callable[[str], str]) -> None:
        """Initializes the mutator.

        Args:
            edit: Transforms the parent's source text into the child's.
        """
        self.edit = edit

    async def __call__(
        self, workspace: Path, message: str, program_id: str
    ) -> Mutation:
        """Rewrites ``workspace / "child.py"`` as ``edit`` of what it holds.

        The edit runs in a thread, as the codex call it stands in for runs in
        a child process: a dry run with a slow ``edit`` must exercise the
        loop's concurrency rather than stall its single thread.

        Args:
            workspace: A directory holding ``child.py``, the program to edit.
            message: The whole prompt. A fake reads nothing of it; it is here
                because the loop hands every mutator the same three things.
            program_id: The child program id.

        Returns:
            A `Mutation` with status "ok".
        """
        started = time.perf_counter()
        child = workspace / "child.py"
        await asyncio.to_thread(self.write, child)
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

    def write(self, child: Path) -> None:
        """Read ``child.py``, apply ``edit``, write it back. Runs in a thread.

        Args:
            child: The file to rewrite.
        """
        child.write_text(self.edit(child.read_text(encoding="utf-8")), encoding="utf-8")


def opencode_model() -> str:
    """The ``provider/model`` an opencode round asks for, read at every call.

    Returns:
        The slug from the environment, or `config.OPENCODE_MODEL`.
    """
    load_dotenv(ENV, override=True)
    return os.environ.get("CAMPAIGN_OPENCODE_MODEL") or config.OPENCODE_MODEL


def opencode_fallback() -> str:
    """The seller retried once when the first fails without a verdict.

    Returns:
        The slug from the environment, or `config.OPENCODE_FALLBACK_MODEL`.
    """
    load_dotenv(ENV, override=True)
    value = os.environ.get("CAMPAIGN_OPENCODE_FALLBACK_MODEL")
    return config.OPENCODE_FALLBACK_MODEL if value is None else value


def known_opencode_models() -> set[str]:
    """Every ``provider/model`` this opencode install can reach.

    ``opencode models`` prints one per line across every authenticated
    provider.

    Returns:
        The slugs it lists.
    """
    output = subprocess.run(
        OPENCODE_CATALOG_COMMAND, capture_output=True, check=True, text=True
    ).stdout
    return {line.strip() for line in output.splitlines() if "/" in line}


def validate_opencode_model(model: str) -> None:
    """Fails fast when ``model`` is not one this install can reach.

    Args:
        model: The ``provider/model`` to check.

    Raises:
        SystemExit: ``model`` is not in `known_opencode_models`.
    """
    if model not in known_opencode_models():
        raise SystemExit(
            f"{model!r} is not a model this opencode install can reach "
            "(run `opencode models` to check the spelling, and "
            "`opencode auth list` to see which sellers are signed in)"
        )


# Every program that can drive a round. The selection names one of these and
# `build` returns it; nothing else in the loop knows there is more than one.
DRIVERS: dict[str, type] = {
    "codex": CodexMutator,
    "agy": AgyMutator,
    "opencode": OpenCodeMutator,
}


def selected() -> str:
    """Which program drives a round, read fresh so it can change.

    Returns:
        "codex" or "agy".

    Raises:
        SystemExit: The name is neither.
    """
    load_dotenv(ENV, override=True)
    kind = os.environ.get("CAMPAIGN_MUTATOR") or config.MUTATOR
    if kind not in DRIVERS:
        named = ", ".join(repr(name) for name in DRIVERS)
        raise SystemExit(f"{kind!r} is not a mutator; use one of {named}")
    return kind


def build() -> Mutator:
    """The mutator `selected` names.

    Returns:
        A mutator reading its own model per call.
    """
    return DRIVERS[selected()]()


def asked_model() -> str:
    """The model the selected mutator will ask for.

    Returns:
        The slug, for the run's own record of what it opened on.
    """
    return {
        "codex": model,
        "agy": agy_model,
        "opencode": opencode_model,
    }[selected()]()


def validate_models() -> None:
    """Checks the selected mutator's models against its own catalog.

    Each program has its own vocabulary and neither knows the other's:
    `gpt-5.6-luna` is not a slug agy has ever heard of, and validating it
    against the wrong catalog would refuse a run that was going to work.

    Raises:
        SystemExit: Either model is not one the login knows.
    """
    check, asked, retry = {
        "codex": (validate_model, model, fallback),
        "agy": (validate_agy_model, agy_model, agy_fallback),
        "opencode": (validate_opencode_model, opencode_model, opencode_fallback),
    }[selected()]
    check(asked())
    if retry():
        check(retry())
