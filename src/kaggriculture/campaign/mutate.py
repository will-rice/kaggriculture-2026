"""One mutation: one model call on one file, or a fake that edits a constant.

A call is given a directory holding ``child.py`` and nothing else, and the
whole message. It edits that file and stops; the loop reads it back, scores
it, and composes the next round's message from the result.

Three programs can drive a round -- codex, agy and opencode -- and they exist
side by side because each bills a different entitlement, not because any is
better. `Driver` is the one shape they share: read the model at the call,
run the program in the workspace as its own process group, read the
transcript back, and decide the verdict by comparing the file against what
the round was handed. What differs between them is a table of class
attributes and three hooks -- the argument vector, the environment, and how
the transcript is read.

No program's own report of how it went decides anything. All of them will
exit 0 having done nothing -- codex ends a turn with a question instead of an
edit, agy returns ``"status": "SUCCESS"`` when its ``--print-timeout`` expires
mid-turn, after announcing that it finished. `_written` compares the file
against what the round was handed, and that comparison is the whole verdict.

Each command is a class attribute so a test can replace it with ``true`` or
``sleep``; the flags are appended only when the command really is the
program, so a stand-in is run exactly as given.

Every driver is awaitable, because the loop runs many rounds at once on one
event loop: a program is an ``asyncio`` child process, and the fake's file
work goes to a thread so an ``edit`` that sleeps cannot stall the loop. A
call's transcript runs to hundreds of kilobytes, so reading it goes to a
thread too.
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, Protocol

from dotenv import load_dotenv
from pydantic import BaseModel

from kaggriculture.campaign import config, plan

LOGGER = logging.getLogger(__name__)

# Which program drives a round: a key of `DRIVERS`. Read per round through
# `selected`, so it can change under a running campaign the way the model can.
#
# It is "agy" because the codex quota this account had is exhausted until
# 2026-09-22 08:09 and the deadline is 2026-09-30. `agy` bills a different
# entitlement entirely -- and two of them: `agy -p /usage` reports a Gemini
# pool and a separate "Claude and GPT models" pool. The rotation asks the
# others when the selected one is out.
MUTATOR = "agy"

# How long a refused entitlement is left alone before it is asked again.
#
# The rotation used to rediscover the same exhaustion every round: opus
# refused, sonnet refused, the gemini pool answered, and the next round opened
# by asking opus again. Measured 2026-09-21 the pair took about two and a half
# minutes to refuse, against rounds composing every three -- so most of a
# round's setup was spent confirming a wall that agy already reports, with a
# reset time attached, in the refusal itself.
#
# Thirty minutes rather than that reset time, which arrives as prose ("Resets
# in 2h48m38s") and would have to be parsed to be trusted. The cost of being
# wrong is bounded and small in both directions: at worst half an hour of not
# using an entitlement that came back early, against a couple of minutes a
# round saved while it is genuinely out.
QUOTA_COOLDOWN = 30 * 60

# How long the rotation waits before asking again when every program is out of
# quota.
#
# It used to return the refusal and let the round fail, which spun: sixteen
# refusals inside three minutes on 2026-09-20, as fast as four sessions could
# ask. Failing is worse than idling -- a round that produced nothing counts
# toward `STAGNATION_SESSIONS`, so an outage would have the campaign decide its
# champion had gone stale when nothing had run, and the failures are shown to
# the next round as though they were its own.
#
# Five minutes because a refusal costs seconds, so the poll is free next to the
# ten-to-twenty-five minutes an answer takes, and the shortest reset seen so far
# is a five-hour window.
QUOTA_WAIT = 5 * 60

# The file the models are read from, absolute on purpose. `load_dotenv` with no
# path searches relative to something -- the caller's module, or the working
# directory -- and the campaign moves its working directory: every worker that
# runs a program is given a scratch one. A relative search would find a
# different file depending on who asked.
ENV = config.ROOT / ".env"


def setting(key: str) -> str | None:
    """One `.env` value, read fresh so it can change under a running campaign.

    A process's environment is fixed when it is spawned, so exporting a
    variable in a shell cannot reach a loop that is already running; the file
    is the part of the environment a running process can re-read. Edit
    `.env` and the next round uses it. The wandb run keeps the name it
    started with, because that is what it started with; `calls/model` is
    logged per call and is the truth about any one of them.

    Args:
        key: The variable to read.

    Returns:
        Its value, or None when neither the file nor the environment sets it.
    """
    load_dotenv(ENV, override=True)
    return os.environ.get(key)


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


class Report(BaseModel):
    """What a transcript says about the call that wrote it.

    Attributes:
        input_tokens: Prompt tokens billed, summed across the call.
        output_tokens: Completion tokens, summed the same way.
        failure: The program's own account of a run that reached no verdict
            -- a quota wall, a provider refusing the turn -- or "" when it
            reports none. Non-empty is an ``exec_error`` whatever the exit
            code, because agy exits 0 on one.
        said: The last thing the round said, which is the reason a call that
            wrote nothing is given.
        dollars: What the seller charged, where the program reports it.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    failure: str = ""
    said: str = ""
    dollars: float | None = None


class Mutator(Protocol):
    """Something that turns one file and one message into a child program."""

    # Where this program discovers skills, relative to the workspace. The
    # loop copies `config.SKILLS` into every one of them, and the programs do
    # not agree: codex reads `.codex/skills` under its working directory, agy
    # and opencode walk up from it looking for `.agents`. Hardcoding either in
    # the loop is how a round silently loses the schema and the queries worth
    # running -- it still runs, it just never finds them.
    #
    # Plural because a call can be handed from one program to another when the
    # first has no quota left, and the workspace is prepared before anybody
    # knows which will serve it.
    SKILLS_DIRS: tuple[Path, ...]
    # And what this driver calls the transcript it leaves behind, as globs,
    # because a retry writes a second one beside the first and a rotation can
    # leave one from each program it tried. Named here for the same reason
    # `SKILLS_DIRS` is: the loop copies them out and has no business knowing
    # which program wrote which.
    TRANSCRIPTS: tuple[str, ...]

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


class Driver:
    """One program that drives a round: the shape, with the program left open.

    A call runs until it is done, or until `TIMEOUT` if the program is one
    that can hang before its first byte. There is no other cap: a round runs
    the skills this login has installed, and the only cap this ever had cut
    calls off before they had written anything. So the one other thing that
    ends a call early is the cancellation that shutting the loop down
    delivers. Every program spawns shell commands of its own, and killing only
    the direct child would orphan any grandchild still running, leaking a core
    that ``config.CORE_BUDGET`` assumes is free; the process runs in its own
    session so the cancellation arm can take the whole group with
    ``os.killpg``.

    Attributes:
        NAME: The program's binary, which is also the transcript's stem and
            the log's word for it.
        COMMAND: How it is started. Replaced by tests with a stand-in, which
            is then run exactly as given.
        EXTENSION: The transcript's suffix.
        CATALOG: The command that lists the models this login may ask for --
            a local lookup, not a model turn, so it spends no quota.
        MODEL_KEY: The `.env` variable naming the model, exported to the
            child so a stand-in can read which it was asked for.
        MODEL: The model when nothing names one.
        FALLBACK_KEY: The `.env` variable naming the model retried once when
            a call fails without a verdict. Set empty to never retry.
        FALLBACK: The fallback when nothing names one.
        MESSAGE_ON_STDIN: Whether the prompt is written to the process or
            passed in `invocation`. Standard input is closed otherwise: agy
            reads a controlling terminal for an OAuth code when its stdin is
            a pipe, and a round that waits on one nobody is watching waits
            for hours.
        TIMEOUT: Seconds before a call is killed as hung, or None to wait.
    """

    NAME: str
    COMMAND: list[str]
    EXTENSION = "jsonl"
    SKILLS_DIRS: tuple[Path, ...]
    TRANSCRIPTS: tuple[str, ...]
    CATALOG: list[str]
    MODEL_KEY: str
    MODEL: str
    FALLBACK_KEY: str
    FALLBACK: str
    MESSAGE_ON_STDIN = False
    TIMEOUT: float | None = None

    def __init__(
        self,
        model: str = "",
        fallback: str | None = None,
        source: "Callable[[], str] | None" = None,
    ) -> None:
        """Initializes the driver.

        Both models are normally left unset and read from `.env` at every
        call, so the campaign's model can change without stopping it. A test
        pins them instead.

        Args:
            model: The model to request, or "" to read it per call.
            fallback: The model to retry on, once, when a call on ``model``
                fails without a verdict. "" never retries; None reads it per
                call.
            source: Where the per-call model is read from when ``model`` is
                unset; `asked` by default. agy meters two entitlements apart
                and the model name alone decides which a call bills, so the
                rotation holds two agy drivers reading different keys.
        """
        self.model = model
        self.fallback = fallback
        self.source = source or type(self).asked

    @classmethod
    def asked(cls) -> str:
        """The model to ask for, read fresh at every call."""
        return setting(cls.MODEL_KEY) or cls.MODEL

    @classmethod
    def retry(cls) -> str:
        """The model retried once when the first fails without a verdict."""
        value = setting(cls.FALLBACK_KEY)
        return cls.FALLBACK if value is None else value

    @classmethod
    def catalog(cls) -> set[str]:
        """Every model this login may ask for, as `CATALOG` reports them."""
        raise NotImplementedError

    @classmethod
    def validate(cls, model: str) -> None:
        """Fails fast when ``model`` is not one this login's catalog lists.

        A typo'd model is hundreds of failed sessions discovered one at a time
        -- codex accepts ``gpt-6-astra`` and refuses ``gpt-5.6-astra``, so it
        is not hypothetical. Called once at startup, before any session
        spends a call on a name that was never going to work.

        Args:
            model: The model to check.

        Raises:
            SystemExit: ``model`` is not in `catalog`.
        """
        if model not in cls.catalog():
            raise SystemExit(
                f"{model!r} is not a model this {cls.NAME} login knows about "
                f"(run `{' '.join(cls.CATALOG)}` to check the spelling)"
            )

    def invocation(self, workspace: Path, message: str, model: str) -> list[str]:
        """The whole command line one round runs as.

        Separate from `call` so that the flags can be asserted on without
        spending a call. Losing one of them does not fail loudly -- it
        produces a round that reads its file, cannot measure anything, and
        reports `no_output` -- so they are worth a test.

        Args:
            workspace: The directory the round works in.
            message: The whole prompt.
            model: The model to request.

        Returns:
            The argument vector, unchanged when `COMMAND` is a stand-in.
        """
        raise NotImplementedError

    def environment(self, model: str) -> dict[str, str]:
        """What the child is told beyond the environment it inherits."""
        return {self.MODEL_KEY: model}

    def report(self, log: Path) -> Report:
        """Reads the transcript back. Runs in a thread.

        Args:
            log: The transcript this call wrote.

        Returns:
            What it says about the call.
        """
        raise NotImplementedError

    def transcript(self, workspace: Path) -> Path:
        """Where this program's transcript goes."""
        return workspace / f"{self.NAME}.{self.EXTENSION}"

    async def __call__(
        self, workspace: Path, message: str, program_id: str
    ) -> Mutation:
        """Runs one call in ``workspace``, retrying once on the fallback.

        A call that ends in ``exec_error`` -- the provider failed the turn (a
        model at capacity, a rate limit, a quota wall) or the program itself
        died -- is run again on the fallback model over the same file. The
        failed transcript is kept beside the new one, named for the model
        that failed.

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
        asked = self.model or self.source()
        retry = self.retry() if self.fallback is None else self.fallback
        mutation = await self.call(workspace, message, program_id, asked)
        if mutation.status != "exec_error" or not retry:
            return mutation
        LOGGER.warning(
            "%s call for %s failed on %s (%s); retrying on %s",
            self.NAME,
            program_id,
            asked,
            mutation.reason[:120],
            retry,
        )
        log = self.transcript(workspace)
        spent = asked.replace("/", "-")
        log.replace(log.with_name(f"{self.NAME}.{spent}.failed.{self.EXTENSION}"))
        retried = await self.call(workspace, message, program_id, retry)
        return retried.model_copy(
            update={"fallback": True, "seconds": mutation.seconds + retried.seconds}
        )

    async def call(
        self, workspace: Path, message: str, program_id: str, model: str
    ) -> Mutation:
        """Runs one call in ``workspace`` on ``model``.

        Args:
            workspace: A directory holding ``child.py`` and nothing else.
            message: The whole prompt.
            program_id: The child program id.
            model: The model to request.

        Returns:
            A `Mutation` describing what happened.
        """
        started = time.perf_counter()
        # What the round was handed. A call that ends with the file exactly
        # as it found it has written nothing, however cleanly it exited.
        given = plan.gather(workspace)
        log = self.transcript(workspace)
        with log.open("w", encoding="utf-8") as handle:
            process = await asyncio.create_subprocess_exec(
                *self.invocation(workspace, message, model),
                stdin=(
                    asyncio.subprocess.PIPE
                    if self.MESSAGE_ON_STDIN
                    else asyncio.subprocess.DEVNULL
                ),
                stdout=handle,
                stderr=asyncio.subprocess.PIPE,
                cwd=workspace,
                start_new_session=True,
                env={**os.environ, **self.environment(model)},
            )
            written = message.encode() if self.MESSAGE_ON_STDIN else None
            try:
                _, stderr = await asyncio.wait_for(
                    process.communicate(written), self.TIMEOUT
                )
            except TimeoutError:
                # A hung program has the same children a cancelled one does.
                LOGGER.warning(
                    "%s call for %s gave up after %ss, killed pgid %s",
                    self.NAME,
                    program_id,
                    self.TIMEOUT,
                    kill_group(process),
                )
                return self.outcome(
                    program_id,
                    model,
                    started,
                    Report(),
                    "exec_error",
                    f"{self.NAME} wrote nothing in {self.TIMEOUT}s and was killed",
                )
            except asyncio.CancelledError:
                # The loop is shutting down. The call goes with it, whole
                # process group and all -- the programs run shell commands,
                # schedule background tasks and invoke subagents of their own,
                # and terminate those only on a clean exit, which a killed
                # parent never reaches. Otherwise a restart finds a round
                # still playing seasons against a directory nothing owns, on
                # a core the budget has already promised to somebody else.
                LOGGER.warning(
                    "%s call for %s cancelled, killed pgid %s",
                    self.NAME,
                    program_id,
                    kill_group(process),
                )
                await process.wait()
                raise
        report = await asyncio.to_thread(self.report, log)
        if process.returncode != 0 or report.failure:
            # A failure the program reports is one whatever the exit code:
            # agy exits 0 on a quota wall, saying so only in its `result`
            # event, after eight hundred seconds of real work that as
            # `no_output` would be thrown away instead of retried.
            reason = (
                report.failure
                or stderr.decode(errors="replace")[-500:]
                or f"{self.NAME} exited {process.returncode}"
            )
            LOGGER.warning(
                "%s call for %s failed on %s (exit %s): %s",
                self.NAME,
                program_id,
                model,
                process.returncode,
                reason[:200],
            )
            return self.outcome(
                program_id, model, started, report, "exec_error", reason
            )
        child = _written(workspace, given)
        if child is None:
            reason = report.said or "child.py missing, empty or unchanged"
            return self.outcome(program_id, model, started, report, "no_output", reason)
        if report.dollars is not None:
            LOGGER.info(
                "%s round %s wrote a child on %s for $%.4f",
                self.NAME,
                program_id,
                model,
                report.dollars,
            )
        return self.outcome(program_id, model, started, report, "ok", "", child)

    @staticmethod
    def outcome(
        program_id: str,
        model: str,
        started: float,
        report: Report,
        status: Literal["ok", "no_output", "exec_error"],
        reason: str,
        child: Path | None = None,
    ) -> Mutation:
        """One `Mutation`, stamped with the call's time and what it billed."""
        return Mutation(
            program_id=program_id,
            child=child,
            status=status,
            reason=reason,
            seconds=time.perf_counter() - started,
            input_tokens=report.input_tokens,
            output_tokens=report.output_tokens,
            model=model,
        )


class CodexMutator(Driver):
    """``codex exec`` over one file, the prompt on standard input.

    Codex 0.147's ``--json`` output is one JSON object per line. Token usage
    lives on the ``turn.completed`` event, under ``usage.input_tokens`` and
    ``usage.output_tokens`` (verified against a real session log); other
    events carry no usage and are ignored. Codex sometimes ends a turn with a
    question instead of writing ``child.py``; ``_last_message`` recovers the
    last ``agent_message`` text so the caller knows why.
    """

    NAME = "codex"
    # `--skip-git-repo-check` because a call runs in a temporary directory
    # holding one file, not in a repository: codex refuses an untrusted
    # directory otherwise, and the directory is deliberately not one.
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
    SKILLS_DIRS: tuple[Path, ...] = (Path(".codex") / "skills",)
    TRANSCRIPTS: tuple[str, ...] = ("codex*.jsonl",)
    CATALOG = ["codex", "debug", "models"]
    MODEL_KEY = "CAMPAIGN_CODEX_MODEL"
    # The cheap model. `gpt-6-astra` ran for three hours on 2026-09-09 and
    # produced one promotion that had two candidate causes -- the vendored
    # opponent draw landed in the same restart -- at a quota cost the first
    # campaign had already spent nearly all of one on. A second trial wants a
    # measurement that can attribute: promotions per session against a luna
    # baseline on the same code.
    #
    # `validate` checks this against the login's own catalog at startup,
    # because a typo here is hundreds of failed sessions discovered one at a
    # time -- not hypothetical: this login accepts `gpt-6-astra` but refuses
    # `gpt-5.6-astra` (probed 2026-09-05 on codex 0.153).
    MODEL = "gpt-5.6-luna"
    FALLBACK_KEY = "CAMPAIGN_CODEX_FALLBACK_MODEL"
    # Astra answered "Selected model is at capacity" some of the time (two
    # calls in the first live hour), which is why a fallback exists at all.
    FALLBACK = "gpt-5.6-sol"
    MESSAGE_ON_STDIN = True

    # How hard the model is asked to think, passed on every call.
    #
    # Astra offers low, medium, high, xhigh, max and ultra, and defaults to
    # medium. The campaign was not running at medium, though, and not at
    # anything it chose: `~/.codex/config.toml` sets `model_reasoning_effort
    # = "high"` for the host's own interactive use, and every campaign call
    # inherited it. That is the same shape of coupling as a round inheriting
    # the host's skills -- the loop's behaviour changing because a file it
    # does not own changed -- and it is worth closing whatever the value is.
    #
    # `max` is "maximum reasoning depth for the hardest problems". Above it
    # sits `ultra`, which adds automatic task delegation; that is a different
    # execution shape rather than more thinking, and a round already has a
    # shape.
    REASONING = "max"

    @classmethod
    def catalog(cls) -> set[str]:
        """Every ``slug`` in the catalog command's JSON output."""
        output = subprocess.run(
            cls.CATALOG, capture_output=True, check=True, text=True
        ).stdout
        return {model["slug"] for model in json.loads(output)["models"]}

    def invocation(self, workspace: Path, message: str, model: str) -> list[str]:
        """`COMMAND` with the model, the directory and the effort."""
        command = [*self.COMMAND]
        if command[0] != self.NAME:
            return command
        # The effort is passed per call rather than left to
        # `~/.codex/config.toml`, so the campaign's effort is the campaign's
        # decision and does not move when the host edits its own settings.
        return command + [
            "-m",
            model,
            "-C",
            str(workspace),
            "-c",
            f"model_reasoning_effort={self.REASONING}",
            # `workspace-write` closes the network by default, and a round's
            # one instrument opens a socket: `measure.py --against` asks the
            # games database which opponent an episode was, and the day tables
            # the message points at live there. Measured over 46 codex rounds
            # on 2026-09-25: 36 `--against` runs died on `PermissionError`, 44
            # queries were refused, and the rounds reported the required test
            # as blocked -- which the loop then filed as their failure.
            "-c",
            "sandbox_workspace_write.network_access=true",
        ]

    def report(self, log: Path) -> Report:
        """Tokens, the provider's failure if any, and the last thing said."""
        tokens_in, tokens_out = _tokens(log)
        return Report(
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            failure=_failure(log),
            said=_last_message(log),
        )


class AgyMutator(Driver):
    """``agy --print`` over one file, the prompt on the command line.

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
    Only the file decides. A status that is not ``SUCCESS`` is the one thing
    the result event is trusted about: it is how a quota wall arrives, on an
    exit code of 0, and it is an ``exec_error`` so the fallback -- on the
    other entitlement -- is tried.

    Two flags are deliberately absent. ``--sandbox`` restricts the terminal,
    and a round has to be able to run ``measure.py``; commands already run
    without prompting because the host's settings say
    ``toolPermission: proceed-in-sandbox``. ``--disable-slash-commands``
    would stop skill expansion, and the workspace skill in `SKILLS_DIRS` is
    the point -- it is discovered even in a temporary directory that is not a
    repository (probed 2026-09-16). Skills are disclosed progressively, so
    the host's own eight cost their descriptions and nothing more unless a
    round chooses to open one.
    """

    NAME = "agy"
    COMMAND = ["agy", "--print"]
    SKILLS_DIRS: tuple[Path, ...] = (Path(".agents") / "skills",)
    TRANSCRIPTS: tuple[str, ...] = ("agy*.jsonl",)
    # `agy -p /usage` is how the quota itself is read; this spends none.
    CATALOG = ["agy", "models"]
    MODEL_KEY = "CAMPAIGN_AGY_MODEL"
    # Sonnet rather than a flash model because a round reads the champion and
    # its opponents and edits a program, and the cheap end of the catalog has
    # already failed that once: `gemini-3.6-flash-low` produced garbage on a
    # two-step shell-and-edit probe that `-medium` completed. It also spends
    # the pool that has quota rather than the one that is 2% down.
    MODEL = "claude-sonnet-4-6"
    FALLBACK_KEY = "CAMPAIGN_AGY_FALLBACK_MODEL"
    # Retried across pools on purpose: a Claude-pool refusal (rate limit,
    # capacity) is exactly the failure a same-pool retry would hit again.
    FALLBACK = "gemini-3.8-flash-medium"
    # The model asked for on agy's other entitlement.
    #
    # `agy -p /usage` reports two pools -- "Gemini Models" and "Claude and GPT
    # models" -- and meters them apart. The model name alone decides which a
    # call bills, so naming only one leaves the other unspent: on 2026-09-20
    # the Gemini pool was down to 34% while Claude and GPT sat at 67%, and agy
    # had already stopped both lineages twice for want of quota. A Gemini
    # model here because `MODEL` names a Claude one; the pair is what matters.
    SECOND_KEY = "CAMPAIGN_AGY_SECOND_MODEL"
    SECOND = "gemini-3.1-pro-high"
    # How long one round may run.
    #
    # Deliberately far above the 5m default, for the reason a round cap was
    # removed from codex: the only cap this ever had cut calls off before
    # they had written anything. It has to be said out loud here because an
    # expired `--print-timeout` does not look like a failure -- agy returns
    # the partial answer, reports `"status": "SUCCESS"` and exits 0 (measured
    # on 1.2.4, with `child.py` untouched). Nothing but the file says whether
    # the round worked, which is why `_written` is what decides the verdict.
    PRINT_TIMEOUT = "3h"

    @classmethod
    def second(cls) -> str:
        """The model to ask for on the other entitlement, read at every call."""
        return setting(cls.SECOND_KEY) or cls.SECOND

    @classmethod
    def catalog(cls) -> set[str]:
        """Every slug the catalog lists.

        ``agy models`` prints one ``slug<TAB>label`` line per model on
        standard output and its progress line on standard error, so the parse
        is the first field of every line. It documents an ``--output-format
        json`` that agy 1.2.4 does not have -- the flag is refused outright --
        which is why this reads the text.
        """
        output = subprocess.run(
            cls.CATALOG, capture_output=True, check=True, text=True
        ).stdout
        return {
            line.split("\t", 1)[0].strip()
            for line in output.splitlines()
            if line.strip()
        }

    def invocation(self, workspace: Path, message: str, model: str) -> list[str]:
        """`COMMAND` with the prompt, the grants and the model."""
        command = [*self.COMMAND]
        if command[0] != self.NAME:
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
            self.PRINT_TIMEOUT,
        ]

    def report(self, log: Path) -> Report:
        """The terminal `result` event: usage, status and the answer."""
        result = _agy_result(log)
        usage = result.get("usage") or {}
        said = _agy_reason(result)
        return Report(
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            failure=said if result.get("status") not in ("SUCCESS", None) else "",
            said=said,
        )


class OpenCodeMutator(Driver):
    """``opencode run`` over one file: the driver whose limit is money.

    OpenCode reaches a seller that charges per token, so a round costs about
    five cents and how many rounds are left in a day is a question about the
    budget rather than a five-hour bucket.

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

    NAME = "opencode"
    COMMAND = ["opencode", "run"]
    EXTENSION = "ndjson"
    SKILLS_DIRS: tuple[Path, ...] = (Path(".agents") / "skills",)
    TRANSCRIPTS: tuple[str, ...] = ("opencode*.ndjson",)
    # Lists every provider this install is signed into at once, so the
    # catalog is the union rather than one seller's.
    CATALOG = ["opencode", "models"]
    MODEL_KEY = "CAMPAIGN_OPENCODE_MODEL"
    # As `provider/model`. `gpt-5.6-luna` because it is the model this lineage
    # was already climbing with through codex, and OpenRouter sells it by the
    # token with no window at all: $0.20 a million input, about five cents for
    # a round of the size measured here.
    MODEL = "openrouter/openai/gpt-5.6-luna"
    FALLBACK_KEY = "CAMPAIGN_OPENCODE_FALLBACK_MODEL"
    # The same model from a different seller, which is the only fallback that
    # answers the failure a fallback is for: a provider refusing,
    # rate-limiting or dropping the turn is a fact about that seller and not
    # about the model.
    FALLBACK = "opencode-go/gpt-5.6-luna"
    # `opencode run` has no timeout flag of its own, and on 2026-09-20 four
    # calls hung for four and a half hours apiece -- three minutes of CPU
    # between them, zero-byte transcripts -- while the loop waited, because
    # nothing told it not to. Its config's `timeout`/`headerTimeout`/
    # `chunkTimeout` were already at their five-minute defaults and did not
    # fire: nothing had streamed, so the hang was upstream of the request, in
    # opencode's own server startup.
    #
    # Forty minutes rather than something tight. A round that is working takes
    # ten to twenty-five, so a shorter cap would throw away good calls to
    # catch a rare bad one; this is a backstop against a hang, not a limit on
    # a round.
    TIMEOUT = 40 * 60

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

    # What the provider is allowed to take, in milliseconds. Documented
    # defaults are 300000 for each and they did not save us -- a hung call
    # produced nothing for four and a half hours, which means it never got as
    # far as a request -- so these are the inner of two layers, and `TIMEOUT`
    # is the outer one that actually caught it.
    #
    # `timeout` caps the request, `headerTimeout` the wait for response
    # headers, `chunkTimeout` the gap between streamed chunks. Named per
    # provider, so the provider is read off the model: an opencode model is
    # `provider/model`, sometimes `provider/vendor/model`, and the first
    # segment is the seller.
    LIMITS = {"timeout": 600_000, "headerTimeout": 120_000, "chunkTimeout": 120_000}

    def policy(self, model: str) -> dict:
        """The config this call runs under, with the provider's limits in it.

        Args:
            model: The ``provider/model`` being requested.

        Returns:
            `POLICY` with a `provider` section for this call's seller.
        """
        seller = model.split("/", 1)[0]
        return {**self.POLICY, "provider": {seller: {"options": dict(self.LIMITS)}}}

    @classmethod
    def catalog(cls) -> set[str]:
        """Every ``provider/model`` listed, one per line across every seller."""
        output = subprocess.run(
            cls.CATALOG, capture_output=True, check=True, text=True
        ).stdout
        return {line.strip() for line in output.splitlines() if "/" in line}

    @classmethod
    def validate(cls, model: str) -> None:
        """As `Driver.validate`, naming the auth list a missing seller needs."""
        if model not in cls.catalog():
            raise SystemExit(
                f"{model!r} is not a model this opencode install can reach "
                "(run `opencode models` to check the spelling, and "
                "`opencode auth list` to see which sellers are signed in)"
            )

    def invocation(self, workspace: Path, message: str, model: str) -> list[str]:
        """`COMMAND` with the prompt, the directory and the model."""
        command = [*self.COMMAND]
        if command[0] != self.NAME:
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

    def environment(self, model: str) -> dict[str, str]:
        """The model, and the policy this call runs under."""
        return {
            **super().environment(model),
            "OPENCODE_CONFIG_CONTENT": json.dumps(self.policy(model)),
        }

    def report(self, log: Path) -> Report:
        """Sums the tokens and the cost across the run, and keeps its last word.

        Every ``step_finish`` event carries a `tokens` object and a `cost`,
        and a round is many steps, so the call's total is their sum. Cached
        prompt tokens are reported separately by the seller and are not added
        here: they are charged at a different rate, and what this feeds is a
        token count the other drivers report the same way.
        """
        tokens_in = tokens_out = 0
        dollars = 0.0
        said = ""
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            part = event.get("part") or {}
            if event.get("type") == "text":
                said = str(part.get("text", "")) or said
            if event.get("type") != "step_finish":
                continue
            tokens = part.get("tokens") or {}
            tokens_in += int(tokens.get("input", 0))
            tokens_out += int(tokens.get("output", 0))
            dollars += float(part.get("cost", 0.0))
        return Report(
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            said=said[:200],
            dollars=dollars,
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


class FakeMutator:
    """Edits ``child.py`` in place through ``edit``. For dry runs and tests."""

    # Nothing reads them, but the loop still lays them out, so a dry run
    # exercises the same copy a real round does. It writes no transcript, and
    # the glob simply matches nothing.
    SKILLS_DIRS: tuple[Path, ...] = (Path(".agents") / "skills",)
    TRANSCRIPTS: tuple[str, ...] = ("fake*.jsonl",)

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


# Every program that can drive a round. The selection names one of these and
# `build` returns it; nothing else in the loop knows there is more than one.
DRIVERS: dict[str, type[Driver]] = {
    "codex": CodexMutator,
    "agy": AgyMutator,
    "opencode": OpenCodeMutator,
}


def selected() -> str:
    """Which program drives a round, read fresh so it can change.

    Returns:
        A key of `DRIVERS`.

    Raises:
        SystemExit: The name is none of them.
    """
    kind = setting("CAMPAIGN_MUTATOR") or MUTATOR
    if kind not in DRIVERS:
        named = ", ".join(repr(name) for name in DRIVERS)
        raise SystemExit(f"{kind!r} is not a mutator; use one of {named}")
    return kind


# What a program says when the entitlement is gone rather than the call being
# wrong. agy says "Individual quota reached", codex "You've hit your usage
# limit"; matched loosely because the wording is theirs to change and the cost
# of a false match is one call handed to the next program, which is where a
# true match sends it anyway.
EXHAUSTED = ("quota", "usage limit", "rate limit", "insufficient_quota")


def named(driver: "Mutator") -> str:
    """What to call a program in the log.

    Two of the rotation's entries are `AgyMutator`, differing only in which
    entitlement their model bills, so the class name alone cannot say which
    one refused or which one answered -- and that is exactly the question the
    log is read to answer.

    Args:
        driver: The program to name.

    Returns:
        The class name, and the model where the program chooses one.
    """
    source = getattr(driver, "source", None)
    return f"{type(driver).__name__}({source()})" if source else type(driver).__name__


def out_of_quota(mutation: "Mutation") -> bool:
    """Whether this call failed because the entitlement is spent.

    Args:
        mutation: What the call came back with.

    Returns:
        True when the failure is an exhausted entitlement rather than a round
        that ran and wrote nothing.
    """
    if mutation.status != "exec_error":
        return False
    said = mutation.reason.lower()
    return any(phrase in said for phrase in EXHAUSTED)


class Rotating:
    """Tries each program in turn until one still has quota to answer with.

    Three programs can drive a round and each bills a different entitlement,
    which is the only reason there are three. Pinned to one, a run stops when
    that one stops: agy ran out at 00:44 on 2026-09-19 and both lineages were
    idle until morning, and on 2026-09-20 all three were out at once.

    Nothing can be checked in advance. They are separate CLIs against separate
    accounts, and the only tool that reports quota sees just the providers
    opencode is signed into. So this asks, which is cheap -- a refusal comes
    back in seconds, where an answer takes ten minutes.

    Order matters and the configured program leads it, so a run with quota does
    exactly what it did before this existed.

    Attributes:
        SKILLS_DIRS: Every place any of them looks for skills, so the workspace
            suits whichever ends up serving the call.
        TRANSCRIPTS: Every pattern any of them writes, so a call that was
            passed along leaves both transcripts behind and the log shows it.
    """

    def __init__(self, drivers: "Sequence[Mutator]") -> None:
        """Initializes the rotation.

        Args:
            drivers: The programs to try, in the order to try them.
        """
        self.drivers = drivers
        # When each program may be asked again, as a monotonic deadline. An
        # entitlement that just refused will refuse for hours, and asking it
        # anyway costs the round the couple of minutes the refusal takes to
        # arrive -- paid per round, against rounds that compose every three.
        self.spent = [0.0] * len(drivers)
        self.SKILLS_DIRS: tuple[Path, ...] = tuple(
            dict.fromkeys(one for driver in drivers for one in driver.SKILLS_DIRS)
        )
        self.TRANSCRIPTS: tuple[str, ...] = tuple(
            dict.fromkeys(one for driver in drivers for one in driver.TRANSCRIPTS)
        )

    async def __call__(
        self, workspace: Path, message: str, program_id: str
    ) -> "Mutation":
        """Asks each program in turn until one answers with something.

        Args:
            workspace: A directory holding ``child.py`` and nothing else.
            message: The whole prompt.
            program_id: The child program id.

        Waits and asks again when every one of them is out, rather than
        failing the round: a round that produced nothing counts toward
        `STAGNATION_SESSIONS` and is shown to the next round as its own
        history, so spinning against a wall would have the campaign mistake an
        outage for a champion that had gone stale.

        Returns:
            The first outcome that is not an exhausted entitlement. Does not
            return while every program is refusing; cancellation is what ends
            the wait, which is what shutting the loop down delivers.
        """
        assert self.drivers, "a rotation with no drivers in it"
        waited = 0
        while True:
            # None when every program was still on cooldown this pass, which
            # is the one case there is no fresh refusal to report.
            refused: Mutation | None = None
            for index, driver in enumerate(self.drivers):
                if self.spent[index] > time.monotonic():
                    continue
                outcome = await driver(workspace, message, program_id)
                if not out_of_quota(outcome):
                    if waited:
                        LOGGER.info("%s: quota is back after %ds", program_id, waited)
                    # On success too, not only on failure. Every line this
                    # wrote used to be an error, so the program that was
                    # working was the one thing the log never mentioned.
                    LOGGER.info("%s: %s answered", program_id, named(driver))
                    self.spent[index] = 0.0
                    return outcome
                self.spent[index] = time.monotonic() + QUOTA_COOLDOWN
                refused = outcome
                LOGGER.warning(
                    "%s: %s has no quota (%s); leaving it alone for %ds",
                    program_id,
                    named(driver),
                    outcome.reason[:80],
                    QUOTA_COOLDOWN,
                )
            # Everything is out. Waiting rather than failing, because a round
            # that produced nothing counts toward stagnation and is shown to the
            # next round as its own history -- so spinning would have the
            # campaign mistake an outage for a champion that had gone stale.
            LOGGER.error(
                "%s: every program is out of quota; waiting %ds (%s)",
                program_id,
                QUOTA_WAIT,
                refused.reason[:120] if refused else "all of them on cooldown",
            )
            # To the soonest deadline when nothing was even asked, because
            # waking on `QUOTA_WAIT` would find every program still cooling
            # and sleep again -- and with one program in the rotation it would
            # never ask anybody at all.
            nap = (
                QUOTA_WAIT
                if refused is not None
                else max(0.0, min(self.spent) - time.monotonic())
            )
            await asyncio.sleep(nap)
            waited += nap


def build() -> "Rotating":
    """The program `selected` names, and the others behind it.

    One entitlement running out used to stop a run: agy's did at 00:44 on
    2026-09-19 and both lineages were idle until morning. The three bill
    separate accounts, so the others are still worth asking.

    Returns:
        A rotation, the selected program first, each reading its own model per
        call.
    """
    first = selected()
    order = [first, *(name for name in DRIVERS if name != first)]
    drivers: list[Mutator] = []
    for name in order:
        drivers.append(DRIVERS[name]())
        if name == "agy":
            # Twice, because agy meters two entitlements apart and the model
            # name alone decides which a call bills. Naming one left the other
            # unspent while agy stopped both lineages for want of quota.
            drivers.append(AgyMutator(source=AgyMutator.second))
    return Rotating(drivers)


def asked_model() -> str:
    """The model the selected program will ask for.

    Returns:
        The slug, for the run's own record of what it opened on.
    """
    return DRIVERS[selected()].asked()


def validate_models() -> None:
    """Checks the selected program's models against its own catalog.

    Each program has its own vocabulary and none knows another's:
    `gpt-5.6-luna` is not a slug agy has ever heard of, and validating it
    against the wrong catalog would refuse a run that was going to work.

    Raises:
        SystemExit: Either model is not one the login knows.
    """
    driver = DRIVERS[selected()]
    driver.validate(driver.asked())
    if driver.retry():
        driver.validate(driver.retry())
    # The other entitlement's slug too, whichever program leads: the rotation
    # holds it either way, and a typo there is a pool that silently never gets
    # asked rather than a run that fails loudly.
    AgyMutator.validate(AgyMutator.second())
