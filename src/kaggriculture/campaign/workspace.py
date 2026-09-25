"""The directory a round works in: laid out, guarded, and read back.

A round is given a directory holding the program to improve and nothing a
path could leak through. `prepare` lays it out, `kept` and `restored` put
back anything the round changed in the campaign's own source while it ran,
`transcripts` keeps what the round wrote about itself, and `packed` writes
the round's program back as the one self-contained file everything
downstream expects.
"""

import logging
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from kaggriculture.campaign import archive, config, measure, plan, prompt
from kaggriculture.campaign.mutate import Mutation

LOGGER = logging.getLogger(__name__)

# What a round is not allowed to change: the campaign's own code, and the
# messages it is asked with. Not the engine binary or anything compiled, which
# no round has reason to touch and which would make this expensive.
GUARDED = ("*.py", "*.md")


def with_skills(box: Path, wanted: tuple[Path, ...]) -> None:
    """Copy the skills into every place a driver might look for them.

    Which program serves this call is not settled until it is made -- one out
    of quota hands it to the next -- and the workspace is prepared first. A
    round that finds no skills still runs; it just never finds the schema or
    the queries worth running, which is a failure that does not announce
    itself.

    Args:
        box: The round's directory.
        wanted: Where each driver looks, relative to it.
    """
    for where in wanted:
        shutil.copytree(config.SKILLS, box / where)


def kept(root: Path = config.ROOT) -> dict[Path, bytes]:
    """The campaign's own source, as it stands before a round runs.

    Fifty files and about seven hundred kilobytes, read once against a call that
    takes minutes.

    Args:
        root: The checkout to read. A parameter so a test can name its own.

    Returns:
        The bytes of every guarded file, by path.
    """
    return {
        one: one.read_bytes()
        for pattern in GUARDED
        for one in sorted((root / "src").rglob(pattern))
        if "__pycache__" not in one.parts
    }


def restored(
    before: dict[Path, bytes], mutation: Mutation, program_id: str
) -> Mutation:
    """Put back anything the round changed in the campaign's own source.

    A round gets a throwaway directory holding one agent. On 2026-09-20 one
    edited the campaign instead -- `REFERENCE_SAMPLE` to 0.0, turning off the
    reference-engine cross-check, and `roster.path`'s `raise KeyError` into a
    constructed path, making any string resolve to a file. Neither was random:
    both weaken a guard in the direction that makes the round's own job easier,
    and the second was provoked by `measure.py --against` refusing a mistyped
    name. Given an objective and a writable grader, editing the grader is the
    cheaper way to satisfy it.

    Prevention is not on offer. agy's `--sandbox` restricts the terminal and a
    round has to run `measure.py`, and the driver runs with
    `--dangerously-skip-permissions`, which the comment on that flag wrongly
    calls the same posture as codex's `-s workspace-write`.

    Restored from bytes rather than from git, because the version that asked git
    reverted all 250 tracked files when a pre-commit hook's `GIT_DIR` outranked
    the repository it was handed. This writes back only what it read, so the
    worst it can do is undo a change to one of the files it named.

    The round is failed rather than scored: a program measured against guards it
    removed is not measured at all.

    Args:
        before: What `kept` read before the round.
        mutation: What the call produced.
        program_id: The child program id, for the log and the reason.

    Returns:
        The mutation, or a copy recording that the round wrote out of bounds.
    """
    moved = sorted(
        one
        for one, was in before.items()
        if not one.exists() or one.read_bytes() != was
    )
    if not moved:
        return mutation
    for one in moved:
        one.write_bytes(before[one])
    named = ", ".join(one.name for one in moved)
    LOGGER.warning("%s changed the campaign and was put back: %s", program_id, named)
    return mutation.model_copy(
        update={
            "child": None,
            "status": "no_output",
            "reason": (
                f"this round changed the campaign's own source -- {named} -- and "
                "it has been put back. The directory you are given holds the "
                "program to edit; the campaign that measures it is not yours to "
                "change, and a program measured against guards it removed is not "
                "measured at all."
            ),
        }
    )


def packed(mutation: Mutation, box: Path, program_id: str) -> Mutation:
    """Write the round's program back as one file, or record that it cannot be.

    The gate, the archive, the pool, the validator and the submission all expect
    one self-contained file, and none of them has to learn otherwise, so the
    plan is packed back into the program here.

    A round that wrote a `plan.json` the schema refuses produced nothing
    runnable. That is an ordinary outcome of asking a model to edit a file, and
    it was a crash until 2026-09-20: `gather` validated, nothing caught what it
    raised, and the error went up through the worker and the task group and took
    both lineages down. The reason travels onto the mutation, so the next round
    on this lineage is told what broke.

    Args:
        mutation: What the call produced.
        box: The round's directory.
        program_id: The child program id, for the log.

    Returns:
        The mutation, or a copy recording that nothing usable was written.
    """
    if mutation.child is None:
        return mutation
    try:
        mutation.child.write_text(plan.gather(box), encoding="utf-8")
    except plan.BrokenPlanError as broken:
        LOGGER.warning("%s wrote an unusable plan: %s", program_id, broken)
        return mutation.model_copy(
            update={"child": None, "status": "no_output", "reason": str(broken)}
        )
    return mutation


def prepare(
    box: Path,
    source: Path,
    siblings: Sequence[archive.Program],
    database: archive.Database,
    skills: tuple[Path, ...],
) -> None:
    """Lay out one round's directory: the program to edit and what it needs.

    ``child.py`` and ``plan.json`` apart, so the round can read and edit the
    plan: the controller with one import where 94,490 characters of base85
    used to be, and the strategy as readable JSON. No round had ever changed
    the plan while it was a blob -- champions 17 to 24 carry a byte-identical
    one -- because as text it is unreadable and nothing said it was data.

    ``parent.py`` and ``measure.py``: the same program again, and the script
    that plays one against the other. The spec had the model run nothing --
    "the loop plays; the model never does" -- which made every round an edit
    shipped blind and waited on. A round can now change one thing and measure
    it before spending a gate on it, which is what every improvement found by
    hand on 2026-09-10 came from. The verdict is still the loop's: it plays
    every scored game itself, against opponents this never sees, and nothing
    a round reports is read. The script is copied with a shebang naming the
    interpreter this loop is running, because a round's own guess is wrong
    and it pays to guess: `python` is not on the PATH a round's commands see,
    `python3` is the system 3.10 with no `kaggriculture` in it, and one round
    spent its whole call on "I will wait for the search for `kaggriculture`
    to complete".

    ``tried_N.py``: the edits already made to this program, which the message
    names and scores. A score says a direction lost ground; the file is what
    says which direction it was. ``attempts.jsonl``: the whole campaign, one
    line per program, for whatever question the round brings to it.

    And the skills, at whichever path the driving program looks for one, so a
    round that wants the schema and the queries worth running opens them and
    a round with a different question pays nothing for them.

    Args:
        box: The round's directory, empty.
        source: The program this round starts from.
        siblings: Programs already written from ``source``, best first.
        database: The campaign's record, for ``attempts.jsonl``.
        skills: Where each driver looks for skills, relative to the box.
    """
    plan.lay_out(source.read_text(encoding="utf-8"), box)
    # The gate writes a champion read-only so nothing can edit the file the
    # pool plays, and `shutil.copy` carries that mode across. This copy is the
    # one file the call must be able to write.
    (box / "child.py").chmod(0o644)
    shutil.copy(source, box / "parent.py")
    runner = box / "measure.py"
    runner.write_text(
        f"#!{sys.executable}\n" + Path(measure.__file__).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    runner.chmod(0o755)
    for number, program in enumerate(siblings[: prompt.RECENT_ATTEMPTS], start=1):
        shutil.copy(program.source_path, box / f"tried_{number}.py")
    database.attempts(box / "attempts.jsonl")
    with_skills(box, skills)


def transcripts(box: Path, patterns: tuple[str, ...], into: Path) -> None:
    """Keep what the round wrote about itself, before the workspace goes.

    A child whose plan is unchanged can mean the round never opened
    `plan.json`, or that it edited it, measured the edit worse and backed it
    out -- which is the round doing what it was told. The score cannot tell
    those apart and the transcript can.

    Args:
        box: The round's directory.
        patterns: What the drivers call their transcripts, as globs.
        into: Where to keep them; created on the first one.
    """
    written = {one for pattern in patterns for one in box.glob(pattern)}
    for transcript in sorted(written):
        into.mkdir(parents=True, exist_ok=True)
        shutil.copy(transcript, into / transcript.name)
