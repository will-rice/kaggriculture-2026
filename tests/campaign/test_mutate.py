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


def test_codex_mutator_reports_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A codex call that outlives its timeout is reported as "timeout"."""
    box = workspace(tmp_path)
    monkeypatch.setattr(mutate.CodexMutator, "COMMAND", ["sleep", "5"])
    result = asyncio.run(mutate.CodexMutator(timeout=1)(box, MESSAGE, "p3"))
    assert result.status == "timeout"
    assert (box / "codex.jsonl").exists()


def test_a_timed_out_session_still_yields_the_child_it_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cap bounds a round's wall clock, not whether its work counts.

    Whatever ``child.py`` holds when the cap fires is what the loop scores,
    so a call killed after it had written a complete program must not have
    that thrown away.
    """
    box = workspace(tmp_path)
    monkeypatch.setattr(
        mutate.CodexMutator,
        "COMMAND",
        [
            "bash",
            "-c",
            "printf 'def agent(o, c=None):\\n    return {}\\n' > child.py; sleep 30",
        ],
    )
    result = asyncio.run(mutate.CodexMutator(timeout=2)(box, MESSAGE, "p8"))
    assert result.status == "timeout"
    assert result.child == box / "child.py"
    assert "def agent" in (box / "child.py").read_text()


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


def test_codex_mutator_kills_the_whole_process_group_on_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A timeout kills the process group, not just codex, so no orphan survives."""
    box = workspace(tmp_path)
    monkeypatch.setattr(
        mutate.CodexMutator, "COMMAND", ["bash", "-c", "sleep 30 & sleep 30"]
    )
    result = asyncio.run(mutate.CodexMutator(timeout=1)(box, MESSAGE, "p7"))
    assert result.status == "timeout"
    pgid = int(result.reason.split("pgid ")[1].rstrip(")"))
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
        pytest.fail(f"process group {pgid} still has a member after the timeout kill")


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
        call = asyncio.ensure_future(
            mutate.CodexMutator(timeout=30)(box, MESSAGE, "p9")
        )
        while not pgid_file.exists():
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


SKILL = """---
name: {name}
description: {description}
---

Body.
"""


def test_the_command_turns_off_every_skill_the_round_must_not_follow() -> None:
    """One inline table per name, and the whole list, or the rest run anyway."""
    override = mutate.skills_off()

    assert override.startswith("skills.config=[")
    for name in config.SKILLS_OFF:
        assert f'{{name="{name}",enabled=false}}' in override
    assert override.count("enabled=false") == len(config.SKILLS_OFF)


def test_only_a_real_codex_call_is_given_the_per_call_arguments(tmp_path: Path) -> None:
    """The model, the directory and the skills go on a codex line and no other.

    Tests replace ``COMMAND`` with an ordinary shell command, which would take
    ``-m`` and ``-c`` as files to read and fail on all of them.
    """
    codex = mutate.CodexMutator(model="m", fallback="", timeout=1).arguments(
        "chosen", tmp_path
    )

    assert codex[:2] == ["codex", "exec"]
    assert codex[-6:-2] == ["-m", "chosen", "-C", str(tmp_path)]
    assert codex[-2] == "-c" and codex[-1] == mutate.skills_off()

    class Shell(mutate.CodexMutator):
        COMMAND = ["bash", "-c", "true"]

    assert Shell(model="m", timeout=1).arguments("chosen", tmp_path) == Shell.COMMAND


def test_codex_drops_the_named_skills_and_keeps_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex's own answer, not ours: what it would put in front of the model.

    ``codex debug prompt-input`` renders the model-visible input without
    spending a call, so this is the real config key parsed by the real binary.
    A skills directory of this test's own, reached through ``HOME``, because
    the machine's own inventory is not the campaign's to depend on.
    """
    skills = tmp_path / ".agents" / "skills"
    for name in ("gatekeeper-fixture", "worker-fixture"):
        (skills / name).mkdir(parents=True)
        (skills / name / "SKILL.md").write_text(
            SKILL.format(name=name, description=f"Use when {name} applies."),
            encoding="utf-8",
        )
    monkeypatch.setattr(config, "SKILLS_OFF", ["gatekeeper-fixture"])
    workspace = tmp_path / "round"
    workspace.mkdir()
    (workspace / "child.py").write_text("x = 1\n", encoding="utf-8")

    before = _prompt_input(workspace, tmp_path, [])
    after = _prompt_input(workspace, tmp_path, ["-c", mutate.skills_off()])

    assert "gatekeeper-fixture" in before and "worker-fixture" in before
    assert "gatekeeper-fixture" not in after
    assert "worker-fixture" in after


def _prompt_input(workspace: Path, home: Path, extra: list[str]) -> str:
    """What codex would show the model in ``workspace``, with ``home`` as HOME."""
    return subprocess.run(
        ["codex", "debug", "prompt-input", *extra, "improve child.py"],
        capture_output=True,
        text=True,
        check=True,
        cwd=workspace,
        env={**os.environ, "HOME": str(home)},
    ).stdout


def test_the_round_home_holds_every_host_skill_but_the_blocked_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unlisted is not enough: the blocked ones must not be on disk to read.

    Disabling a skill stops codex offering it, and the round read it anyway --
    `using-superpowers` names `superpowers:brainstorming` in its own text, so
    the model went and read the file with `sed`. Symlinks, and one per skill,
    so a skill the developer adds is offered to the next campaign.
    """
    host = tmp_path / "host"
    for name in ("gatekeeper-fixture", "worker-fixture", "helper-fixture"):
        (host / name).mkdir(parents=True)
        (host / name / "SKILL.md").write_text(f"name: {name}\n", encoding="utf-8")
    (host / "loose-file.md").write_text("not a skill\n", encoding="utf-8")
    monkeypatch.setattr(config, "HOST_SKILLS", host)
    monkeypatch.setattr(config, "SKILLS_OFF", ["gatekeeper-fixture"])
    monkeypatch.setattr(config, "ROUND_HOME", tmp_path / "round-home")

    mutate.prepare_home()

    skills = config.ROUND_HOME / ".agents" / "skills"
    assert sorted(p.name for p in skills.iterdir()) == [
        "helper-fixture",
        "worker-fixture",
    ]
    assert (skills / "worker-fixture" / "SKILL.md").read_text() == (
        "name: worker-fixture\n"
    )
    assert not (skills / "gatekeeper-fixture").exists()


def test_the_round_home_is_rebuilt_rather_than_added_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A skill blocked today must not survive from a campaign that allowed it."""
    host = tmp_path / "host"
    for name in ("kept-fixture", "blocked-later-fixture"):
        (host / name).mkdir(parents=True)
        (host / name / "SKILL.md").write_text(f"name: {name}\n", encoding="utf-8")
    monkeypatch.setattr(config, "HOST_SKILLS", host)
    monkeypatch.setattr(config, "ROUND_HOME", tmp_path / "round-home")
    monkeypatch.setattr(config, "SKILLS_OFF", [])
    mutate.prepare_home()
    assert (config.ROUND_HOME / ".agents" / "skills" / "blocked-later-fixture").exists()

    monkeypatch.setattr(config, "SKILLS_OFF", ["blocked-later-fixture"])
    mutate.prepare_home()

    skills = config.ROUND_HOME / ".agents" / "skills"
    assert sorted(p.name for p in skills.iterdir()) == ["kept-fixture"]


def test_a_call_runs_under_the_round_home_and_the_real_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HOME moves so the skills move; CODEX_HOME stays so the login does not.

    Codex derives `CODEX_HOME` from `HOME`, so moving one and not the other
    would take `auth.json` and the model catalog with it and every call in the
    campaign would fail unauthenticated.
    """
    monkeypatch.setattr(config, "ROUND_HOME", tmp_path / "round-home")
    box = tmp_path / "box"
    box.mkdir()
    (box / "child.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        mutate.CodexMutator,
        "COMMAND",
        ["bash", "-c", 'printf "# %s %s\\n" "$HOME" "$CODEX_HOME" >> child.py'],
    )

    mutation = asyncio.run(
        mutate.CodexMutator(model="m", fallback="", timeout=30)(box, "go", "p1")
    )

    assert mutation.status == "ok"
    written = (box / "child.py").read_text(encoding="utf-8")
    assert written.endswith(f"# {config.ROUND_HOME} {mutate.codex_home()}\n")
    assert mutate.codex_home() != config.ROUND_HOME / ".codex"


def test_codex_home_is_read_from_the_environment_when_it_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A login kept somewhere other than `~/.codex` is still the login."""
    monkeypatch.setenv("CODEX_HOME", "/elsewhere/codex")
    assert mutate.codex_home() == Path("/elsewhere/codex")

    monkeypatch.delenv("CODEX_HOME")
    assert mutate.codex_home() == Path.home() / ".codex"
