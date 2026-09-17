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

from kaggriculture.campaign import config, mutate

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


def test_known_models_is_exactly_the_catalogs_slugs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``known_models`` reads slugs from the catalog command, nothing else."""
    monkeypatch.setattr(
        mutate, "MODEL_CATALOG_COMMAND", catalog("gpt-6-astra", "gpt-5.6-luna")
    )
    assert mutate.known_models() == {"gpt-6-astra", "gpt-5.6-luna"}


def test_validate_model_accepts_a_slug_the_catalog_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model this login's catalog knows about passes without complaint."""
    monkeypatch.setattr(mutate, "MODEL_CATALOG_COMMAND", catalog("gpt-5.6-luna"))
    mutate.validate_model("gpt-5.6-luna")


def test_validate_model_refuses_a_typo_and_names_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo'd model is refused before it costs a single session.

    This is not hypothetical: this login accepts ``gpt-6-astra`` but refuses
    ``gpt-5.6-astra``.
    """
    monkeypatch.setattr(mutate, "MODEL_CATALOG_COMMAND", catalog("gpt-6-astra"))
    with pytest.raises(SystemExit, match="gpt-5.6-astra"):
        mutate.validate_model("gpt-5.6-astra")


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
    assert f"model_reasoning_effort={config.CODEX_REASONING}" in passed
    # Beside the model, so a call names both rather than inheriting either.
    assert config.CODEX_MODEL in passed


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
    before = mutate.model()
    # The same process, no restart, nothing re-imported.
    env.write_text("CAMPAIGN_CODEX_MODEL=second-model\n", encoding="utf-8")
    after = mutate.model()

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

    assert mutate.model() == config.CODEX_MODEL
    assert mutate.fallback() == config.CODEX_FALLBACK_MODEL


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
    assert mutate.CodexMutator.SKILLS_DIR == Path(".codex") / "skills"
    assert mutate.AgyMutator.SKILLS_DIR == Path(".agents") / "skills"


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
    assert isinstance(mutate.build(), mutate.CodexMutator)
    assert mutate.asked_model() == mutate.model()

    env.write_text("CAMPAIGN_MUTATOR=agy\n", encoding="utf-8")
    assert isinstance(mutate.build(), mutate.AgyMutator)
    assert mutate.asked_model() == mutate.agy_model()


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

    assert mutate.agy_model() == config.AGY_MODEL
    assert mutate.agy_fallback() == config.AGY_FALLBACK_MODEL


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
        mutate,
        "AGY_CATALOG_COMMAND",
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
        mutate,
        "AGY_CATALOG_COMMAND",
        [
            "printf",
            "%s",
            "gemini-3.8-flash-medium\tGemini 3.8 Flash (Medium)\n"
            "claude-opus-4-6-thinking\tClaude Opus 4.6 (Thinking)\n",
        ],
    )

    assert mutate.known_agy_models() == {
        "gemini-3.8-flash-medium",
        "claude-opus-4-6-thinking",
    }
