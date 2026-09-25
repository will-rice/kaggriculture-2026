"""The mutation call, with a fake for every test that is not about codex itself.

Every codex stand-in here is a real subprocess -- ``true``, ``sleep``, a
``bash`` line -- because what these tests are about is what the mutator does
to a live process group, which no fake process could show.

A call is given a directory holding ``child.py`` and the whole prompt on
standard input, so these build the first and pass the second.
"""

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from kaggriculture.campaign import mutate

# A codex that says nothing but a well-formed transcript, so the no-output
# path can be read without spending a call.
TRANSCRIPT = (
    '{"type":"thread.started","thread_id":"t"}\n'
    '{"type":"item.completed","item":{"id":"i1","type":"agent_message",'
    '"text":"I have a question about the interface."}}\n'
    '{"type":"turn.completed","usage":{"input_tokens":10,"cached_input_tokens":0,'
    '"cache_write_input_tokens":0,"output_tokens":5,"reasoning_output_tokens":0}}\n'
)


MESSAGE = (
    "Write child.py implementing def agent(observation, configuration=None).\n"
    "Keep it a complete, self-contained agent file.\n"
)


# The program a round is handed: a call has written something when what it
# leaves behind is not this.
PARENT = "def agent(o, c=None):\n    return {'farmer': ['PASS']}\n"


def workspace(tmp_path: Path) -> Path:
    """The directory a call works in: ``child.py``, the program to improve.

    That is the whole contract -- one file and the prompt on standard input --
    so a call that leaves this file as it found it has written nothing.
    """
    (tmp_path / "child.py").write_text(PARENT, encoding="utf-8")
    return tmp_path


def test_fake_mutator_writes_a_child_with_the_edit_applied(tmp_path: Path) -> None:
    """FakeMutator rewrites child.py through the caller's edit."""
    box = workspace(tmp_path)
    (box / "child.py").write_text(
        "LIMIT = 1\n"
        "def agent(o, c=None):\n"
        "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
    )
    mutator = mutate.FakeMutator(edit=lambda s: s.replace("LIMIT = 1", "LIMIT = 2"))
    result = asyncio.run(mutator(box, MESSAGE, "p1"))
    assert result.status == "ok" and result.child == box / "child.py"
    assert result.child is not None
    assert "LIMIT = 2" in result.child.read_text()


def test_the_whole_prompt_reaches_the_call_on_standard_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The message is the prompt, and there is no file to read it from.

    Everything the model is told -- the rules, the verdict, the day table,
    the instruction -- travels this way, so the directory can hold one file
    and nothing a path could leak through.
    """
    box = workspace(tmp_path)
    monkeypatch.setattr(
        mutate.CodexMutator,
        "COMMAND",
        [
            "bash",
            "-c",
            "cat > stdin.txt; printf 'def agent(o, c=None):\\n    return {}\\n' "
            "> child.py",
        ],
    )

    result = asyncio.run(mutate.CodexMutator()(box, MESSAGE, "p12"))

    assert result.status == "ok"
    assert (box / "stdin.txt").read_text() == MESSAGE


def test_codex_mutator_reports_no_output_when_the_file_is_left_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A codex that exits clean having touched nothing is "no_output", not "ok".

    The file it was handed is a complete program, so it would validate and
    score; taking it would insert a copy of the round's own parent and count
    a child the model never wrote.
    """
    box = workspace(tmp_path)
    monkeypatch.setattr(
        mutate.CodexMutator, "COMMAND", ["true"]
    )  # a codex that says nothing
    result = asyncio.run(mutate.CodexMutator()(box, MESSAGE, "p2"))
    assert result.status == "no_output" and result.child is None
    assert (box / "child.py").read_text() == PARENT
    assert (box / "codex.jsonl").exists()


# The provider's capacity refusal, as codex records it: the provider's error
# is a JSON string inside the event's own message.
CAPACITY = json.dumps(
    {
        "type": "turn.failed",
        "error": {
            "message": json.dumps(
                {
                    "type": "error",
                    "status": 503,
                    "error": {
                        "message": "Selected model is at capacity. "
                        "Please try a different model."
                    },
                }
            )
        },
    }
)
# A stand-in codex that fails the turn as the provider does when a model is
# at capacity, or writes a child, depending on the model it was asked for.
CAPACITY_OR_CHILD = f"""
if [ "$CAMPAIGN_CODEX_MODEL" = "$1" ]; then
  printf '%s\\n' '{CAPACITY}'
  exit 1
fi
printf 'def agent(o, c=None):\\n    return {{}}\\n' > child.py
printf '%s\\n' '{{"type":"turn.completed","usage":{{"input_tokens":7}}}}'
"""


def test_a_session_that_fails_on_the_first_model_is_retried_on_the_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One retry on the fallback, over the same file; the first transcript kept."""
    box = workspace(tmp_path)
    monkeypatch.setattr(
        mutate.CodexMutator, "COMMAND", ["bash", "-c", CAPACITY_OR_CHILD, "_", "first"]
    )
    result = asyncio.run(
        mutate.CodexMutator(model="first", fallback="second")(box, MESSAGE, "p10")
    )
    assert result.status == "ok" and result.model == "second" and result.fallback
    assert result.input_tokens == 7
    assert (box / "child.py").exists()
    assert "turn.failed" in (box / "codex.first.failed.jsonl").read_text()
    assert "turn.completed" in (box / "codex.jsonl").read_text()


def test_a_failure_on_both_models_is_an_exec_error_naming_the_provider_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reason is the provider's message, not the empty stderr."""
    box = workspace(tmp_path)
    monkeypatch.setattr(
        mutate.CodexMutator, "COMMAND", ["bash", "-c", CAPACITY_OR_CHILD, "_", "same"]
    )
    result = asyncio.run(
        mutate.CodexMutator(model="same", fallback="same")(box, MESSAGE, "p11")
    )
    assert result.status == "exec_error" and result.fallback
    assert "at capacity" in result.reason
    assert (box / "child.py").read_text() == PARENT


def test_codex_mutator_reports_exec_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A codex process that exits non-zero is reported as "exec_error"."""
    box = workspace(tmp_path)
    monkeypatch.setattr(
        mutate.CodexMutator, "COMMAND", ["sh", "-c", "echo boom 1>&2; exit 1"]
    )
    result = asyncio.run(mutate.CodexMutator()(box, MESSAGE, "p6"))
    assert result.status == "exec_error"
    assert "boom" in result.reason
    assert (box / "codex.jsonl").exists()


def test_codex_mutator_no_output_reason_carries_the_last_agent_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A no_output reason names the last thing codex said, and tokens still parse."""
    box = workspace(tmp_path)
    # `printf %s` writes its argument verbatim, so the transcript reaches
    # `codex.jsonl` exactly as a real session would have written it.
    monkeypatch.setattr(mutate.CodexMutator, "COMMAND", ["printf", "%s", TRANSCRIPT])
    result = asyncio.run(mutate.CodexMutator()(box, MESSAGE, "p5"))
    assert result.status == "no_output"
    assert "question about the interface" in result.reason
    assert result.input_tokens == 10 and result.output_tokens == 5


def test_cancelling_a_call_kills_the_whole_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shutting the loop down takes the session with it, grandchildren included.

    A cancelled call that only closed its pipes would leave codex running
    against a directory nothing owns, holding cores the budget counts as
    free; a restart would then race eight orphans.
    """
    box = workspace(tmp_path)
    pgid_file = tmp_path / "pgid"
    monkeypatch.setattr(
        mutate.CodexMutator,
        "COMMAND",
        [
            "bash",
            "-c",
            f"ps -o pgid= -p $$ | tr -d ' ' > {pgid_file}; sleep 30 & sleep 30",
        ],
    )

    async def cancel_mid_session() -> None:
        """Start the call, wait for the group to exist, then cancel it."""
        call = asyncio.ensure_future(mutate.CodexMutator()(box, MESSAGE, "p9"))
        # Non-empty, not merely present: the shell's `>` creates the file
        # before `ps` has written a word into it, so waiting on existence
        # alone reads "" back on a loaded box and the test fails parsing an
        # empty string rather than on the process group it is about.
        while not pgid_file.exists() or not pgid_file.read_text().strip():
            await asyncio.sleep(0.05)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call

    asyncio.run(cancel_mid_session())

    pgid = int(pgid_file.read_text().strip())
    for _ in range(20):
        survivors = subprocess.run(
            ["pgrep", "-g", str(pgid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if survivors.returncode != 0:
            break
        time.sleep(0.1)
    else:
        pytest.fail(f"process group {pgid} survived the cancellation")


# A stand-in for `codex debug models`: a local catalog lookup, not a model
# turn, so a real one costs nothing to run -- but these tests replace it
# anyway, so a typo in the constant cannot make a whole suite depend on
# whatever this login happens to be entitled to today.
def catalog(*slugs: str) -> list[str]:
    """A command printing a catalog of exactly ``slugs``, standing in for codex."""
    body = json.dumps({"models": [{"slug": slug} for slug in slugs]})
    return ["printf", "%s", body]


def test_the_catalog_is_exactly_the_commands_slugs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`catalog` reads slugs from the catalog command, nothing else."""
    monkeypatch.setattr(
        mutate.CodexMutator, "CATALOG", catalog("gpt-6-astra", "gpt-5.6-luna")
    )
    assert mutate.CodexMutator.catalog() == {"gpt-6-astra", "gpt-5.6-luna"}


def test_validate_model_accepts_a_slug_the_catalog_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model this login's catalog knows about passes without complaint."""
    monkeypatch.setattr(mutate.CodexMutator, "CATALOG", catalog("gpt-5.6-luna"))
    mutate.CodexMutator.validate("gpt-5.6-luna")


def test_validate_model_refuses_a_typo_and_names_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo'd model is refused before it costs a single session.

    This is not hypothetical: this login accepts ``gpt-6-astra`` but refuses
    ``gpt-5.6-astra``.
    """
    monkeypatch.setattr(mutate.CodexMutator, "CATALOG", catalog("gpt-6-astra"))
    with pytest.raises(SystemExit, match="gpt-5.6-astra"):
        mutate.CodexMutator.validate("gpt-5.6-astra")


@pytest.mark.skipif(
    not os.environ.get("CAMPAIGN_CODEX"),
    reason="set CAMPAIGN_CODEX=1 to spend a real codex call",
)
def test_real_codex_writes_a_child(tmp_path: Path) -> None:
    """One real codex call turns a trivial parent into a working child.py."""
    box = workspace(tmp_path)
    # codex refuses to run outside a trusted directory, and a round's
    # directory is a fresh one under the system temporary tree, so make this
    # tmp_path trusted rather than relaxing the mutator's real command.
    subprocess.run(["git", "init", "-q"], cwd=box, check=True)
    result = asyncio.run(mutate.CodexMutator()(box, MESSAGE, "p4"))
    assert result.status == "ok"
    assert result.child is not None
    assert "def agent" in (box / "child.py").read_text()
    assert result.input_tokens > 0 and result.output_tokens > 0


def test_the_reasoning_effort_is_passed_on_every_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A constant nothing passes is a setting that does not exist.

    The effort was never passed at all: codex took it from
    ``~/.codex/config.toml``, which belongs to the host and is edited for the
    host's own interactive use. So the loop's reasoning depth was whatever
    somebody last set for themselves, and a campaign could change behaviour
    between restarts with nothing in its own tree changing.

    Exercised through a real ``codex`` on PATH rather than a substituted
    COMMAND, because the flags are only appended when the command *is* codex
    -- which is exactly the branch a substituted COMMAND skips.

    `.env` is pointed at an empty file of this test's own and the variable is
    removed from the environment, so `mutate.model` falls through to the
    constant and the assertion below is about the flag being passed rather
    than about whichever slug the host happens to be running today. Both
    halves are needed: `load_dotenv` only ever adds to `os.environ`, so any
    earlier read of the real `.env` -- this process makes one per call --
    leaves the host's slug set for the life of the interpreter, and moving
    `ENV` afterwards cannot take it back out.
    """
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.setattr(mutate, "ENV", tmp_path / ".env")
    monkeypatch.delenv("CAMPAIGN_CODEX_MODEL", raising=False)
    monkeypatch.delenv("CAMPAIGN_CODEX_FALLBACK_MODEL", raising=False)
    box = workspace(tmp_path)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    shim = binaries / "codex"
    shim.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" > "$ARGV_OUT"\ncat > /dev/null\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)
    argv = tmp_path / "argv.txt"
    monkeypatch.setenv("PATH", f"{binaries}:{os.environ['PATH']}")
    monkeypatch.setenv("ARGV_OUT", str(argv))

    asyncio.run(mutate.CodexMutator()(box, MESSAGE, "p12"))

    passed = argv.read_text(encoding="utf-8").splitlines()
    assert f"model_reasoning_effort={mutate.CodexMutator.REASONING}" in passed
    # And the network, which `workspace-write` closes by default: without it
    # `measure.py --against` and every query the message points at die on a
    # socket, which is what 36 of the first codex rounds did.
    assert "sandbox_workspace_write.network_access=true" in passed
    # Beside the model, so a call names both rather than inheriting either.
    assert mutate.CodexMutator.MODEL in passed


def test_the_model_changes_under_a_running_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Edit `.env` and the next round asks for the new model. No restart.

    A process's environment is fixed when it is spawned, so exporting a
    variable in a shell cannot reach a loop that is already running. The file
    is the part of the environment a running process can re-read, so it is
    reloaded at every call rather than once at startup -- which is the whole
    difference between "set the model" and "set the model without stopping
    the campaign".
    """
    monkeypatch.setattr(mutate, "ENV", tmp_path / ".env")
    monkeypatch.delenv("CAMPAIGN_CODEX_MODEL", raising=False)
    env = tmp_path / ".env"

    env.write_text("CAMPAIGN_CODEX_MODEL=first-model\n", encoding="utf-8")
    before = mutate.CodexMutator.asked()
    # The same process, no restart, nothing re-imported.
    env.write_text("CAMPAIGN_CODEX_MODEL=second-model\n", encoding="utf-8")
    after = mutate.CodexMutator.asked()

    assert before == "first-model"
    assert after == "second-model", "the change did not reach a running process"


def test_an_unset_model_is_the_campaigns_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing in the environment means the value the repository commits to.

    So a campaign started with no configuration runs the model the code says,
    and a typo in `.env` is a changed model rather than a missing one.
    """
    monkeypatch.setattr(mutate, "ENV", tmp_path / ".env")
    monkeypatch.delenv("CAMPAIGN_CODEX_MODEL", raising=False)
    monkeypatch.delenv("CAMPAIGN_CODEX_FALLBACK_MODEL", raising=False)
    (tmp_path / ".env").write_text("", encoding="utf-8")

    assert mutate.CodexMutator.asked() == mutate.CodexMutator.MODEL
    assert mutate.CodexMutator.retry() == mutate.CodexMutator.FALLBACK


def test_a_round_asks_for_whatever_the_model_is_now(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mutator reads it at the call, not at construction.

    Binding it in `__init__` was the bug this replaces: the default argument
    evaluated once at import, so the campaign's model was fixed before the
    first round and could not be anything else without a restart.
    """
    monkeypatch.setattr(mutate, "ENV", tmp_path / ".env")
    monkeypatch.delenv("CAMPAIGN_CODEX_MODEL", raising=False)
    box = workspace(tmp_path)
    monkeypatch.setattr(
        mutate.CodexMutator,
        "COMMAND",
        [
            "bash",
            "-c",
            "printf 'def agent(o, c=None):\\n    return {}\\n' > child.py",
        ],
    )
    mutator = mutate.CodexMutator()

    (tmp_path / ".env").write_text("CAMPAIGN_CODEX_MODEL=alpha\n", encoding="utf-8")
    first = asyncio.run(mutator(box, MESSAGE, "p20"))
    (tmp_path / ".env").write_text("CAMPAIGN_CODEX_MODEL=beta\n", encoding="utf-8")
    second = asyncio.run(mutator(box, MESSAGE, "p21"))

    assert first.model == "alpha"
    assert second.model == "beta"


# What agy 1.2.4 writes under `--output-format stream-json`: one JSON object
# per line, and a terminal `result` event carrying the answer and the usage.
# This one is the case the whole verdict turns on -- a round that reports
# having done the work, with `status` SUCCESS and an exit code of 0, having
# left the file alone. Measured rather than invented: a real call whose
# `--print-timeout` expired mid-turn did exactly this, and said so.
AGY_SUCCESS_SAYING_NOTHING = (
    '{"event":"init","init":{"model":"claude-sonnet-4-6"}}\n'
    '{"event":"step_update","step_update":{"step_index":0,"state":"DONE",'
    '"step_type":"user_input"}}\n'
    '{"event":"result","result":{"status":"SUCCESS","response":'
    '"I have updated child.py as requested.","duration_seconds":9.4,'
    '"num_turns":1,"usage":{"input_tokens":41,"output_tokens":13,'
    '"thinking_tokens":7,"cache_read_tokens":0,"total_tokens":61}}}\n'
)

# The same, for a round the harness would not let touch the file. agy reports
# these rather than failing, so a forbidden workspace and a model that
# declined to edit are the same event unless this field is read.
AGY_DENIED = (
    '{"event":"result","result":{"status":"SUCCESS","response":'
    '"I was unable to modify the file.","denied_actions":'
    '["write_file(child.py) needs allow rule write_file(/tmp)"],'
    '"usage":{"input_tokens":5,"output_tokens":2}}}\n'
)

# A round that did the work.
AGY_WROTE_A_CHILD = (
    '{"event":"result","result":{"status":"SUCCESS","response":"done",'
    '"usage":{"input_tokens":31,"output_tokens":9}}}\n'
)

# And a run that failed without reaching a verdict. This is the quota wall as
# agy reported it on 2026-09-16, after a round had read the program, played 32
# seasons and queried the corpus over eight hundred seconds: the exit code is
# still 0 and the only thing that says otherwise is `status`.
AGY_OUT_OF_QUOTA = (
    '{"event":"result","result":{"status":"ERROR","error":'
    '"Individual quota reached. Please upgrade your subscription to increase '
    'your limits. Resets in 4h18m40s.","response":"Let me read child.py",'
    '"usage":{"input_tokens":208584,"output_tokens":8200}}}\n'
)


def test_a_run_that_failed_without_a_verdict_is_an_exec_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A quota wall is not a round that chose to write nothing.

    The distinction pays for itself: `no_output` spends the round and moves on,
    while `exec_error` retries on the fallback -- which is on the other quota
    pool for exactly this reason. Measured once at eight hundred seconds of real
    work, which as `no_output` would have been thrown away.
    """
    box = workspace(tmp_path)
    monkeypatch.setattr(
        mutate.AgyMutator, "COMMAND", ["printf", "%s", AGY_OUT_OF_QUOTA]
    )

    result = asyncio.run(
        mutate.AgyMutator(model="asked", fallback="")(box, MESSAGE, "p26")
    )

    assert result.status == "exec_error"
    assert "quota reached" in result.reason
    # Billed even though it produced nothing, so the telemetry says what it cost.
    assert result.input_tokens == 208584 and result.output_tokens == 8200


def test_a_quota_wall_is_retried_on_the_other_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wall the campaign actually hits, and the round it need not lose.

    agy bills two entitlements that run down separately -- the Gemini models
    share one, the Claude and GPT models another -- so a wall is a property of
    the pool rather than of the account, and the fallback is chosen on the other
    side of it. Measured: four sonnet rounds on a 310KB champion exhausted the
    Claude five-hour limit while the Gemini one was untouched.
    """
    box = workspace(tmp_path)
    monkeypatch.setattr(
        mutate.AgyMutator,
        "COMMAND",
        ["bash", "-c", AGY_WALL_OR_CHILD, "_", "claude-pool"],
    )

    result = asyncio.run(
        mutate.AgyMutator(model="claude-pool", fallback="gemini-pool")(
            box, MESSAGE, "p27"
        )
    )

    assert result.status == "ok" and result.fallback
    assert result.model == "gemini-pool"
    assert (box / "child.py").read_text() != PARENT
    assert (box / "agy.claude-pool.failed.jsonl").exists()


# A stand-in agy that walks into the quota wall on one pool -- exiting 0, as it
# does -- and writes a child on the other.
AGY_WALL_OR_CHILD = f"""
if [ "$CAMPAIGN_AGY_MODEL" = "$1" ]; then
  printf '%s' '{AGY_OUT_OF_QUOTA}'
  exit 0
fi
printf 'def agent(o, c=None):\\n    return {{}}\\n' > child.py
printf '%s' '{AGY_WROTE_A_CHILD}'
"""

# A stand-in agy that writes a child and reports it.
AGY_CHILD = f"""
printf 'def agent(o, c=None):\\n    return {{}}\\n' > child.py
printf '%s' '{AGY_WROTE_A_CHILD}'
"""

# And one that fails outright, or writes a child, depending on the model it
# was asked for. A cascade failure puts its message on standard error, where
# agy puts it -- so the transcript of a failed call is empty, and the reason
# has to come from the stream the process wrote it to.
AGY_CAPACITY_OR_CHILD = f"""
if [ "$CAMPAIGN_AGY_MODEL" = "$1" ]; then
  printf 'error: model is at capacity\\n' 1>&2
  exit 1
fi
{AGY_CHILD}
"""


def test_agy_success_having_written_nothing_is_no_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Agy saying SUCCESS is not evidence that the round did anything.

    This is the difference between the two programs that matters. A codex call
    that writes nothing at least ends its turn saying so; agy returns its
    partial answer, reports SUCCESS and exits 0 when its timeout expires --
    in the measured case claiming in the same breath to have finished. Taking
    that at its word would score the round's own parent as its child.
    """
    box = workspace(tmp_path)
    monkeypatch.setattr(
        mutate.AgyMutator, "COMMAND", ["printf", "%s", AGY_SUCCESS_SAYING_NOTHING]
    )

    result = asyncio.run(mutate.AgyMutator()(box, MESSAGE, "p20"))

    assert result.status == "no_output" and result.child is None
    assert (box / "child.py").read_text() == PARENT
    # The claim is kept, because reading it back is how the next person learns
    # it cannot be trusted.
    assert "updated child.py" in result.reason
    assert result.input_tokens == 41 and result.output_tokens == 13


def test_agy_mutator_reports_ok_with_the_tokens_the_result_event_carries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A round that edits the file is "ok", billed by its own result event."""
    box = workspace(tmp_path)
    monkeypatch.setattr(mutate.AgyMutator, "COMMAND", ["bash", "-c", AGY_CHILD])

    result = asyncio.run(mutate.AgyMutator()(box, MESSAGE, "p21"))

    assert result.status == "ok" and result.reason == ""
    assert result.child == box / "child.py"
    assert result.input_tokens == 31 and result.output_tokens == 9
    assert (box / "agy.jsonl").exists()


def test_a_denied_action_is_named_in_the_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A round the harness forbade reads differently from one that declined.

    On 2026-09-16 a probe run without ``--add-dir`` was read as a model
    inventing a number, and the question that got asked was the wrong one.
    agy says which it was; the reason has to carry it or the log cannot.
    """
    box = workspace(tmp_path)
    monkeypatch.setattr(mutate.AgyMutator, "COMMAND", ["printf", "%s", AGY_DENIED])

    result = asyncio.run(mutate.AgyMutator()(box, MESSAGE, "p22"))

    assert result.status == "no_output"
    assert "denied" in result.reason and "allow rule" in result.reason


def test_agy_mutator_reports_exec_error_on_a_non_zero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a cascade failure exits non-zero, and it is an exec_error.

    A refused tool or a command that failed inside the round is not one: agy
    exits 0 for those, which is why they arrive as "no_output" instead.
    """
    box = workspace(tmp_path)
    monkeypatch.setattr(
        mutate.AgyMutator, "COMMAND", ["sh", "-c", "echo 'error: boom' 1>&2; exit 1"]
    )

    result = asyncio.run(mutate.AgyMutator()(box, MESSAGE, "p23"))

    assert result.status == "exec_error" and "boom" in result.reason
    assert (box / "agy.jsonl").exists()


def test_an_agy_call_that_fails_is_retried_on_the_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One retry, over the same file, with the failed transcript kept beside it.

    Kept even though it is empty: what a cascade failure writes goes to
    standard error, and a file that exists and holds nothing is itself the
    record of a call that produced no stream.
    """
    box = workspace(tmp_path)
    monkeypatch.setattr(
        mutate.AgyMutator,
        "COMMAND",
        ["bash", "-c", AGY_CAPACITY_OR_CHILD, "_", "first"],
    )

    result = asyncio.run(
        mutate.AgyMutator(model="first", fallback="second")(box, MESSAGE, "p24")
    )

    assert result.status == "ok" and result.model == "second" and result.fallback
    assert result.input_tokens == 7 or result.input_tokens == 31
    assert (box / "agy.first.failed.jsonl").exists()
    assert "SUCCESS" in (box / "agy.jsonl").read_text()


def test_an_agy_failure_on_both_models_names_the_cause_from_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both models failing is an exec_error carrying what agy printed."""
    box = workspace(tmp_path)
    monkeypatch.setattr(
        mutate.AgyMutator,
        "COMMAND",
        ["bash", "-c", AGY_CAPACITY_OR_CHILD, "_", "same"],
    )

    result = asyncio.run(
        mutate.AgyMutator(model="same", fallback="same")(box, MESSAGE, "p25")
    )

    assert result.status == "exec_error" and result.fallback
    assert "at capacity" in result.reason
    assert (box / "child.py").read_text() == PARENT


def test_each_mutator_names_the_directory_its_own_program_reads() -> None:
    """The loop copies the skills where the driving program will look.

    codex reads ``.codex/skills`` under its working directory; agy walks up
    from it for ``.agents``. A round handed the other one still runs, which is
    why this is asserted rather than left to be noticed: it just never finds
    the schema or the queries worth running.
    """
    assert mutate.CodexMutator.SKILLS_DIRS == (Path(".codex") / "skills",)
    assert mutate.AgyMutator.SKILLS_DIRS == (Path(".agents") / "skills",)


def test_the_selection_decides_which_program_and_which_vocabulary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`build` and `asked_model` follow ``CAMPAIGN_MUTATOR`` together.

    Together is the point: the two catalogs share no slug, so a selection that
    moved one without the other would check a codex model against agy's
    catalog and refuse a run that was going to work.
    """
    monkeypatch.setattr(mutate, "ENV", tmp_path / ".env")
    monkeypatch.delenv("CAMPAIGN_MUTATOR", raising=False)
    env = tmp_path / ".env"

    env.write_text("CAMPAIGN_MUTATOR=codex\n", encoding="utf-8")
    assert isinstance(mutate.build().drivers[0], mutate.CodexMutator)
    assert mutate.asked_model() == mutate.CodexMutator.asked()

    env.write_text("CAMPAIGN_MUTATOR=agy\n", encoding="utf-8")
    assert isinstance(mutate.build().drivers[0], mutate.AgyMutator)
    assert mutate.asked_model() == mutate.AgyMutator.asked()


def test_a_mutator_that_is_neither_is_refused_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A misspelled driver fails once, not one round at a time."""
    monkeypatch.setattr(mutate, "ENV", tmp_path / ".env")
    monkeypatch.delenv("CAMPAIGN_MUTATOR", raising=False)
    (tmp_path / ".env").write_text("CAMPAIGN_MUTATOR=antigravity\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="antigravity"):
        mutate.selected()


def test_an_unset_agy_model_is_the_campaigns_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing in the environment means the slugs the repository commits to."""
    monkeypatch.setattr(mutate, "ENV", tmp_path / ".env")
    monkeypatch.delenv("CAMPAIGN_AGY_MODEL", raising=False)
    monkeypatch.delenv("CAMPAIGN_AGY_FALLBACK_MODEL", raising=False)
    (tmp_path / ".env").write_text("", encoding="utf-8")

    assert mutate.AgyMutator.asked() == mutate.AgyMutator.MODEL
    assert mutate.AgyMutator.retry() == mutate.AgyMutator.FALLBACK


def test_validation_checks_the_selected_programs_own_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A typo is caught against agy's catalog when agy is the one driving.

    The two catalogs share no slug, so this is the failure that would go
    unnoticed in either direction: a codex model checked against agy's list
    refuses a run that was going to work, and an agy model checked against
    codex's passes a run that cannot make a single call.
    """
    env = tmp_path / ".env"
    env.write_text(
        "CAMPAIGN_MUTATOR=agy\nCAMPAIGN_AGY_MODEL=claude-sonnet-4-7\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mutate, "ENV", env)
    monkeypatch.setattr(
        mutate.AgyMutator,
        "CATALOG",
        ["printf", "%s", "claude-sonnet-4-6\tClaude Sonnet 4.6 (Thinking)\n"],
    )

    with pytest.raises(SystemExit, match="claude-sonnet-4-7"):
        mutate.validate_models()


def test_the_agy_catalog_is_read_as_the_text_agy_prints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Slugs are the first tab-separated field, and the label is not one.

    agy 1.2.4 documents an ``--output-format json`` for ``models`` that it
    does not have -- the flag is refused outright -- so this parses the
    two-column text it does print, and a label containing a space must not
    become part of a slug.
    """
    monkeypatch.setattr(
        mutate.AgyMutator,
        "CATALOG",
        [
            "printf",
            "%s",
            "gemini-3.8-flash-medium\tGemini 3.8 Flash (Medium)\n"
            "claude-opus-4-6-thinking\tClaude Opus 4.6 (Thinking)\n",
        ],
    )

    assert mutate.AgyMutator.catalog() == {
        "gemini-3.8-flash-medium",
        "claude-opus-4-6-thinking",
    }


def test_a_round_is_invoked_with_everything_it_needs_to_measure() -> None:
    """The flags a round cannot work without, asserted rather than assumed.

    None of these fails loudly. A round missing ``--add-dir`` reports reading a
    file it never opened; a round that cannot reach the interpreter reads its
    own file, runs no measurement at all, and comes back ``no_output`` having
    spent a whole call. Both were measured, and neither looks like a missing
    flag from the outside.
    """
    box = Path("/tmp/a-round")
    invocation = mutate.AgyMutator().invocation(box, MESSAGE, "claude-sonnet-4-6")

    assert invocation[0] == "agy"
    # The prompt is an argument, and a valueless `--print` is an error rather
    # than a read from standard input.
    assert invocation[1:3] == ["--print", MESSAGE]
    assert invocation[invocation.index("--add-dir") + 1] == str(box)
    assert invocation[invocation.index("--model") + 1] == "claude-sonnet-4-6"
    assert (
        invocation[invocation.index("--print-timeout") + 1]
        == mutate.AgyMutator.PRINT_TIMEOUT
    )
    assert invocation[invocation.index("--output-format") + 1] == "stream-json"
    assert invocation[invocation.index("--mode") + 1] == "accept-edits"
    # The grant that lets a round run `measure.py`, which is the whole point of
    # a round: without it the call reads its file and measures nothing.
    assert any("skip-permissions" in argument for argument in invocation)
    # `--sandbox` restricts the terminal, which is the one thing a round needs.
    assert "--sandbox" not in invocation
    # And skills stay discoverable: the workspace copy is the point.
    assert "--disable-slash-commands" not in invocation


def test_a_stand_in_command_is_left_exactly_as_it_is() -> None:
    """A test's own command gets no flags, or it would not be a stand-in."""
    mutator = mutate.AgyMutator()
    mutator.COMMAND = ["true"]

    assert mutator.invocation(Path("/tmp/a-round"), MESSAGE, "any") == ["true"]


# What `opencode run --format json` writes: one JSON object per line. Token
# counts and the price live on `step_finish`, one per step, so a call's total is
# their sum -- and the price is the only one of the three drivers that reports
# what it charged.
OPENCODE_WROTE_A_CHILD = (
    '{"type":"step_start","part":{"type":"step-start"}}\n'
    '{"type":"tool_use","part":{"tool":"bash","state":{"status":"completed"}}}\n'
    '{"type":"step_finish","part":{"type":"step-finish","reason":"tool-calls",'
    '"tokens":{"input":900,"output":40,"cache":{"read":17204,"write":110}},'
    '"cost":0.0201}}\n'
    '{"type":"text","part":{"text":"Measured, then made one change."}}\n'
    '{"type":"step_finish","part":{"type":"step-finish","reason":"stop",'
    '"tokens":{"input":100,"output":60,"cache":{"read":9000,"write":0}},'
    '"cost":0.0212}}\n'
)

# And one that read its file, said something, and edited nothing.
OPENCODE_SAID_NOTHING_USEFUL = (
    '{"type":"text","part":{"text":"I could not find a change worth making."}}\n'
    '{"type":"step_finish","part":{"type":"step-finish","reason":"stop",'
    '"tokens":{"input":50,"output":10},"cost":0.0009}}\n'
)

# A stand-in opencode that writes a child and reports it.
OPENCODE_CHILD = f"""
printf 'def agent(o, c=None):\\n    return {{}}\\n' > child.py
printf '%s' '{OPENCODE_WROTE_A_CHILD}'
"""

# And one that fails outright on one seller, or writes a child on the other.
OPENCODE_SELLER_OR_CHILD = f"""
if [ "$1" = "fail" ]; then
  printf 'provider error: 429 rate limited\\n' 1>&2
  exit 1
fi
{OPENCODE_CHILD}
"""


def test_a_round_is_confined_to_its_own_directory() -> None:
    """``--dir`` is the containment, and it is asserted because losing it is quiet.

    A round without it resolves its project by walking up from wherever the
    caller stood, and the caller is the loop standing in the repository: the
    first probe read the campaign's source, ran its git history, listed
    `run/campaign` and played 32 real games there while the directory it had
    been handed sat empty. Nothing about that looked like a missing flag.
    """
    box = Path("/tmp/a-round")
    invocation = mutate.OpenCodeMutator().invocation(box, MESSAGE, "seller/model")

    assert invocation[:2] == ["opencode", "run"]
    assert invocation[2] == MESSAGE
    assert invocation[invocation.index("--dir") + 1] == str(box)
    assert invocation[invocation.index("--format") + 1] == "json"
    assert invocation[invocation.index("-m") + 1] == "seller/model"


def test_a_round_may_not_reach_outside_its_box_even_if_it_asks() -> None:
    """Containment is policy, not the round's good manners.

    `--auto` would approve `external_directory` along with everything else, and
    a round that can be talked into reading one path outside its box can be
    talked into reading the champion archive. So the policy denies it and the
    round runs freely only where it lives.
    """
    policy = mutate.OpenCodeMutator.POLICY["permission"]

    assert policy["external_directory"] == "deny"
    # A question in an empty room ends the turn; better not to reach for it.
    assert policy["question"] == "deny"
    # And the tools a round actually works with are its own.
    assert policy["bash"] == "allow" and policy["edit"] == "allow"


def test_opencode_sums_the_tokens_and_the_price_across_the_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A round is many steps, so what it billed is their sum."""
    box = workspace(tmp_path)
    monkeypatch.setattr(
        mutate.OpenCodeMutator, "COMMAND", ["bash", "-c", OPENCODE_CHILD]
    )

    result = asyncio.run(mutate.OpenCodeMutator()(box, MESSAGE, "p30"))

    assert result.status == "ok" and result.child == box / "child.py"
    assert result.input_tokens == 1000 and result.output_tokens == 100
    assert (box / "opencode.ndjson").exists()


def test_the_price_of_a_call_is_read_even_though_no_field_carries_it(
    tmp_path: Path,
) -> None:
    """`Report.dollars` is what the transcript says the seller charged.

    Cached prompt tokens are deliberately not added to the input count: the
    seller bills them at another rate, and what this feeds is a number the other
    two drivers report the same way. The dollars are the honest total.
    """
    log = tmp_path / "opencode.ndjson"
    log.write_text(OPENCODE_WROTE_A_CHILD, encoding="utf-8")

    spent = mutate.OpenCodeMutator().report(log)

    assert spent.input_tokens == 1000 and spent.output_tokens == 100
    assert spent.dollars == pytest.approx(0.0413)


def test_a_round_that_edits_nothing_is_no_output_whatever_it_said(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file decides, and the round's own account of itself is kept but unread."""
    box = workspace(tmp_path)
    monkeypatch.setattr(
        mutate.OpenCodeMutator,
        "COMMAND",
        ["printf", "%s", OPENCODE_SAID_NOTHING_USEFUL],
    )

    result = asyncio.run(mutate.OpenCodeMutator()(box, MESSAGE, "p31"))

    assert result.status == "no_output" and result.child is None
    assert (box / "child.py").read_text() == PARENT
    assert "worth making" in result.reason


def test_opencode_reports_exec_error_on_a_non_zero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A seller that refuses the turn is an exec_error carrying what it said."""
    box = workspace(tmp_path)
    monkeypatch.setattr(
        mutate.OpenCodeMutator,
        "COMMAND",
        ["sh", "-c", "echo 'provider error: 429' 1>&2; exit 1"],
    )

    result = asyncio.run(mutate.OpenCodeMutator()(box, MESSAGE, "p32"))

    assert result.status == "exec_error" and "429" in result.reason


def test_a_refusing_seller_is_retried_at_another_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fallback is the same model bought elsewhere, which is the point.

    A rate limit or a dropped turn is a fact about the seller rather than about
    the model, so the retry keeps the model and changes the shop: the campaign
    reaches `gpt-5.6-luna` through OpenRouter and through OpenCode's own plan,
    and they do not run out together.
    """
    box = workspace(tmp_path)
    monkeypatch.setattr(
        mutate.OpenCodeMutator,
        "COMMAND",
        ["bash", "-c", OPENCODE_SELLER_OR_CHILD, "_", "fail"],
    )

    result = asyncio.run(
        mutate.OpenCodeMutator(
            model="openrouter/openai/gpt-5.6-luna",
            fallback="opencode-go/gpt-5.6-luna",
        )(box, MESSAGE, "p33")
    )

    assert result.status == "exec_error" and result.fallback
    # The first seller's transcript is kept beside the retry's, named for it.
    assert (box / "opencode.openrouter-openai-gpt-5.6-luna.failed.ndjson").exists()


def test_every_driver_is_reachable_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`CAMPAIGN_MUTATOR` names one of three, and the loop knows no more than that."""
    monkeypatch.setattr(mutate, "ENV", tmp_path / ".env")
    monkeypatch.delenv("CAMPAIGN_MUTATOR", raising=False)
    env = tmp_path / ".env"

    for name, driver in (
        ("codex", mutate.CodexMutator),
        ("agy", mutate.AgyMutator),
        ("opencode", mutate.OpenCodeMutator),
    ):
        env.write_text(f"CAMPAIGN_MUTATOR={name}\n", encoding="utf-8")
        assert isinstance(mutate.build().drivers[0], driver)

    env.write_text("CAMPAIGN_MUTATOR=opencode\n", encoding="utf-8")
    assert mutate.asked_model() == mutate.OpenCodeMutator.asked()


def test_an_unset_opencode_model_is_the_campaigns_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing in the environment means the sellers the repository commits to."""
    monkeypatch.setattr(mutate, "ENV", tmp_path / ".env")
    monkeypatch.delenv("CAMPAIGN_OPENCODE_MODEL", raising=False)
    monkeypatch.delenv("CAMPAIGN_OPENCODE_FALLBACK_MODEL", raising=False)
    (tmp_path / ".env").write_text("", encoding="utf-8")

    assert mutate.OpenCodeMutator.asked() == mutate.OpenCodeMutator.MODEL
    assert mutate.OpenCodeMutator.retry() == mutate.OpenCodeMutator.FALLBACK
    # The same model from two shops, which is what makes the retry worth making.
    assert mutate.OpenCodeMutator.MODEL != mutate.OpenCodeMutator.FALLBACK
    assert (
        mutate.OpenCodeMutator.MODEL.rsplit("/", 1)[-1]
        == (mutate.OpenCodeMutator.FALLBACK.rsplit("/", 1)[-1])
    )


def spent(reason: str) -> mutate.Mutation:
    """A call that came back because the entitlement is gone."""
    return mutate.Mutation(
        program_id="p1",
        child=None,
        status="exec_error",
        reason=reason,
        seconds=0.2,
        input_tokens=0,
        output_tokens=0,
        model="whichever",
    )


def answered() -> mutate.Mutation:
    """A call that actually ran."""
    return mutate.Mutation(
        program_id="p1",
        child=Path("/tmp/child.py"),
        status="ok",
        reason="",
        seconds=90.0,
        input_tokens=10,
        output_tokens=10,
        model="whichever",
    )


class Stub:
    """A driver that returns what it was told to, and records being asked."""

    def __init__(self, name: str, gives: mutate.Mutation, asked: list) -> None:
        self.SKILLS_DIRS: tuple[Path, ...] = (Path(f".{name}") / "skills",)
        self.TRANSCRIPTS: tuple[str, ...] = (f"{name}*.jsonl",)
        self.name = name
        self.gives = gives
        self.asked = asked

    async def __call__(
        self, workspace: Path, message: str, program_id: str
    ) -> mutate.Mutation:
        """Records that this driver was asked, and answers as told."""
        self.asked.append(self.name)
        return self.gives


def test_a_program_out_of_quota_hands_the_round_to_the_next() -> None:
    """What cost two days about ten hours.

    agy's entitlement ran out at 00:44 on 2026-09-19 and both lineages sat idle
    until morning. The three bill separate accounts, so the others were still
    worth asking.
    """
    asked: list[str] = []
    rotation = mutate.Rotating(
        [
            Stub(
                "agy", spent("ERROR; Individual quota reached. Resets in 2h48m"), asked
            ),
            Stub("codex", answered(), asked),
        ]
    )

    got = asyncio.run(rotation(Path("/tmp"), "message", "p1"))

    assert asked == ["agy", "codex"]
    assert got.status == "ok"


def test_a_refused_program_is_not_asked_again_while_it_cools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The waste this exists to stop, measured on 2026-09-21.

    Every round opened by asking the claude entitlement, waiting about two and
    a half minutes for opus and sonnet to refuse, and then getting its answer
    from the gemini pool -- against rounds composing every three minutes. The
    refusal is not news after the first one: an entitlement that just ran out
    stays out for hours, and agy says so in the refusal itself.
    """
    monkeypatch.setattr(mutate, "QUOTA_COOLDOWN", 10_000)
    asked: list[str] = []
    rotation = mutate.Rotating(
        [
            Stub(
                "agy", spent("ERROR; Individual quota reached. Resets in 2h48m"), asked
            ),
            Stub("gemini", answered(), asked),
        ]
    )

    first = asyncio.run(rotation(Path("/tmp"), "message", "p1"))
    second = asyncio.run(rotation(Path("/tmp"), "message", "p2"))

    assert first.status == "ok" and second.status == "ok"
    # Asked once, then skipped: the second round goes straight to the one that
    # answered.
    assert asked == ["agy", "gemini", "gemini"]


def test_a_cooled_program_is_asked_again_once_the_cooldown_expires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skipping is a delay, never a retirement.

    An entitlement comes back, and when it does it is the one the rotation
    prefers -- it leads the order for a reason. A cooldown that never expired
    would quietly demote the best program for the rest of the run.
    """
    monkeypatch.setattr(mutate, "QUOTA_COOLDOWN", 0)
    asked: list[str] = []
    rotation = mutate.Rotating(
        [
            Stub(
                "agy", spent("ERROR; Individual quota reached. Resets in 2h48m"), asked
            ),
            Stub("gemini", answered(), asked),
        ]
    )

    asyncio.run(rotation(Path("/tmp"), "message", "p1"))
    asyncio.run(rotation(Path("/tmp"), "message", "p2"))

    assert asked == ["agy", "gemini", "agy", "gemini"]


def test_a_program_that_answers_stops_cooling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deadline left behind an answer would skip a program that works."""
    monkeypatch.setattr(mutate, "QUOTA_COOLDOWN", 0)
    asked: list[str] = []
    rotation = mutate.Rotating(
        [Reviving(refusals=1, asked=[]), Stub("gemini", answered(), asked)]
    )

    asyncio.run(rotation(Path("/tmp"), "message", "p1"))
    # Refused, so it carries a deadline -- an expired one here, but a deadline.
    assert rotation.spent[0] > 0.0

    asyncio.run(rotation(Path("/tmp"), "message", "p2"))

    # Answered, so the deadline is gone rather than left behind to be compared
    # against on every later round.
    assert rotation.spent[0] == 0.0


def test_the_two_agy_entries_are_told_apart_by_name() -> None:
    """The log's whole job here, and what it could not do before.

    Two of the rotation's programs are `AgyMutator`; only the model says which
    entitlement a call bills. With the class name alone a fallback to the
    gemini pool and a loop stuck on the claude one wrote the same line, and
    telling them apart on 2026-09-21 needed `/proc/<pid>/cmdline`.
    """
    names = [mutate.named(driver) for driver in mutate.build().drivers]

    agy = [one for one in names if one.startswith("AgyMutator")]
    assert len(agy) == 2, names
    assert agy[0] != agy[1]
    assert len(set(names)) == len(names), names


def test_a_program_with_quota_is_the_only_one_asked() -> None:
    """A run with quota must behave exactly as it did before rotation existed."""
    asked: list[str] = []
    rotation = mutate.Rotating(
        [Stub("agy", answered(), asked), Stub("codex", answered(), asked)]
    )

    asyncio.run(rotation(Path("/tmp"), "message", "p1"))

    assert asked == ["agy"]


def test_a_round_that_ran_and_wrote_nothing_is_not_passed_along() -> None:
    """Only an exhausted entitlement moves a call. A bad round is a bad round.

    Passing those along would spend three entitlements on one failure and hide
    which program produced it.
    """
    asked: list[str] = []
    nothing = mutate.Mutation(
        program_id="p1",
        child=None,
        status="no_output",
        reason="I will review the results shortly",
        seconds=48.0,
        input_tokens=5,
        output_tokens=5,
        model="whichever",
    )
    rotation = mutate.Rotating(
        [Stub("agy", nothing, asked), Stub("codex", answered(), asked)]
    )

    got = asyncio.run(rotation(Path("/tmp"), "message", "p1"))

    assert asked == ["agy"]
    assert got.status == "no_output"


def test_the_workspace_suits_whichever_program_serves_the_call() -> None:
    """Which one answers is not known until it is asked.

    codex reads `.codex/skills`; agy and opencode walk up for `.agents`. The
    loop prepares the directory first, so it has to prepare all of them -- a
    round that finds no skills still runs, it just never finds the schema.
    """
    asked: list[str] = []
    rotation = mutate.Rotating(
        [Stub("agy", answered(), asked), Stub("codex", answered(), asked)]
    )

    assert set(rotation.SKILLS_DIRS) == {
        Path(".agy") / "skills",
        Path(".codex") / "skills",
    }
    assert set(rotation.TRANSCRIPTS) == {"agy*.jsonl", "codex*.jsonl"}


def test_the_real_rotation_leads_with_the_configured_program() -> None:
    """`CAMPAIGN_MUTATOR` still decides who is asked first."""
    built = mutate.build()

    assert type(built.drivers[0]).__name__.lower().startswith(mutate.selected()[:3])
    # Every program, and agy twice: it meters two entitlements apart and the
    # model name alone decides which a call bills.
    assert len(built.drivers) == len(mutate.DRIVERS) + 1
    assert sum(isinstance(one, mutate.AgyMutator) for one in built.drivers) == 2


def test_agy_is_asked_on_both_of_its_entitlements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`agy -p /usage` meters two pools and the model name picks one.

    Measured 2026-09-20: "Gemini Models" at 34% weekly and "Claude and GPT
    models" at 67%, and agy had already stopped both lineages twice for want of
    quota while more than half of it sat unspent behind a slug nothing asked
    for. So the rotation holds agy twice, once per pool.
    """
    monkeypatch.setattr(mutate, "ENV", tmp_path / ".env")
    monkeypatch.delenv("CAMPAIGN_MUTATOR", raising=False)
    monkeypatch.delenv("CAMPAIGN_AGY_MODEL", raising=False)
    monkeypatch.delenv("CAMPAIGN_AGY_SECOND_MODEL", raising=False)
    (tmp_path / ".env").write_text(
        "CAMPAIGN_MUTATOR=agy\n"
        "CAMPAIGN_AGY_MODEL=claude-opus-4-6-thinking\n"
        "CAMPAIGN_AGY_SECOND_MODEL=gemini-3.1-pro-high\n",
        encoding="utf-8",
    )

    drivers = mutate.build().drivers
    agy = [one for one in drivers if isinstance(one, mutate.AgyMutator)]

    assert len(agy) == 2
    assert [one.source() for one in agy] == [
        "claude-opus-4-6-thinking",
        "gemini-3.1-pro-high",
    ]


def test_the_second_entitlement_follows_its_own_key_at_the_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both slugs are read per call, so either pool can be moved mid-run.

    A campaign cannot be reached by exporting a variable, and stopping one to
    change a model costs a full champion re-evaluation.
    """
    monkeypatch.setattr(mutate, "ENV", tmp_path / ".env")
    monkeypatch.delenv("CAMPAIGN_AGY_SECOND_MODEL", raising=False)
    env = tmp_path / ".env"

    env.write_text("CAMPAIGN_AGY_SECOND_MODEL=gemini-3.8-flash-low\n", encoding="utf-8")
    assert mutate.AgyMutator.second() == "gemini-3.8-flash-low"

    env.write_text("CAMPAIGN_AGY_SECOND_MODEL=gpt-oss-120b-medium\n", encoding="utf-8")
    assert mutate.AgyMutator.second() == "gpt-oss-120b-medium"


def test_the_two_agy_entries_are_not_the_same_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One out of quota must not take the other down with it."""
    monkeypatch.setattr(mutate, "ENV", tmp_path / ".env")
    monkeypatch.delenv("CAMPAIGN_MUTATOR", raising=False)
    (tmp_path / ".env").write_text("CAMPAIGN_MUTATOR=agy\n", encoding="utf-8")

    agy = [one for one in mutate.build().drivers if isinstance(one, mutate.AgyMutator)]

    assert agy[0] is not agy[1]
    assert agy[0].source is not agy[1].source


class Reviving:
    """A driver that refuses for a while and then answers."""

    def __init__(self, refusals: int, asked: list) -> None:
        self.SKILLS_DIRS: tuple[Path, ...] = (Path(".x") / "skills",)
        self.TRANSCRIPTS: tuple[str, ...] = ("x*.jsonl",)
        self.refusals = refusals
        self.asked = asked

    async def __call__(
        self, workspace: Path, message: str, program_id: str
    ) -> mutate.Mutation:
        """Refuses the first `refusals` asks, then answers."""
        self.asked.append(len(self.asked))
        if len(self.asked) <= self.refusals:
            return spent("ERROR; Individual quota reached. Resets in 2h48m38s.")
        return answered()


def test_an_exhausted_rotation_waits_instead_of_failing_the_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spinning against a wall is worse than idling against it.

    On 2026-09-20 every program was out and the rotation returned the refusal,
    so the round failed and the loop composed another -- sixteen refusals in
    three minutes. A round that produced nothing counts toward
    `STAGNATION_SESSIONS` and is shown to the next round as its own history, so
    an outage would have the campaign decide its champion had gone stale when
    nothing had run.
    """
    monkeypatch.setattr(mutate, "QUOTA_WAIT", 0)
    monkeypatch.setattr(mutate, "QUOTA_COOLDOWN", 0)
    asked: list[int] = []
    rotation = mutate.Rotating([Reviving(refusals=2, asked=asked)])

    got = asyncio.run(rotation(Path("/tmp"), "message", "p1"))

    assert got.status == "ok"
    assert len(asked) == 3


def test_the_wait_is_what_it_is_configured_to_be(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Polling is free next to an answer, but it is not free of a clock."""
    monkeypatch.setattr(mutate, "QUOTA_WAIT", 7)
    monkeypatch.setattr(mutate, "QUOTA_COOLDOWN", 0)
    slept: list[float] = []

    async def note(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(mutate.asyncio, "sleep", note)
    asked: list[int] = []
    rotation = mutate.Rotating([Reviving(refusals=2, asked=asked)])

    asyncio.run(rotation(Path("/tmp"), "message", "p1"))

    assert slept == [7, 7]


def test_waiting_ends_when_the_loop_is_shut_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation is what ends the wait, because nothing else does.

    The rotation does not give up on its own -- that is the point -- so the only
    thing that must be able to end it is the cancellation a stopping loop
    delivers to every round in flight.
    """
    monkeypatch.setattr(mutate, "QUOTA_COOLDOWN", 0)
    asked: list[int] = []
    # Never revives, so only cancellation can end this.
    rotation = mutate.Rotating([Reviving(refusals=10**6, asked=asked)])

    async def stopped() -> None:
        task = asyncio.create_task(rotation(Path("/tmp"), "message", "p1"))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(stopped())
