"""The wandb run a campaign logs to, and the two axes it logs on."""

import wandb
from git import Git, Repo

from kaggriculture.campaign import config
from kaggriculture.campaign.mutate import asked_model, selected

# Metrics go to one wandb run per campaign, resumed across restarts by its
# fixed id. Starting a fresh campaign (a new `run/campaign`) means a new id
# here, or its curves land on top of the old run's.
WANDB_ENTITY = "will-rice"
WANDB_PROJECT = "kaggriculture-2026"


def open_run(dry_run: bool, tag: str, configuration: dict[str, int]) -> wandb.Run:
    """Open the run this campaign logs to, named for the revision that made it.

    Two axes, because a session is many rounds now: everything a round
    produces is stepped by ``calls`` and everything a session or a gate
    produces by ``sessions``. One axis for both would file five rounds under
    one session number and keep the last.

    The revision alone. The model used to be in the name too, from when it
    was a constant compiled into the run; it is chosen in `.env` at the call
    now and can change without a restart, so a name carrying it would be
    wrong from the first round that moved it -- and a wandb id is fixed for
    the life of the run, so there is no renaming it afterwards. The model
    this run opened on is recorded in the run config, and `calls/model` is
    logged per call and is the truth about any one of them.

    Exits on uncommitted changes under ``src/``: a run named by a hash has to
    be that hash.

    Args:
        dry_run: Whether to open the run disabled, logging nothing.
        tag: What distinguishes this lineage from another on the same
            revision, or "".
        configuration: The loop's hyperparameters, recorded on the run.
    """
    # `Git` is bound to the directory and runs there. `Repo.git` is not: in a
    # linked worktree `Repo` resolves to the shared git directory with no
    # working tree of its own, and `git status` fails outright there.
    dirty = Git(config.ROOT).status("--porcelain", "--", "src")
    if dirty:
        raise SystemExit(f"uncommitted changes under src/:\n{dirty}")
    started_on = asked_model()
    # The revision, and what distinguishes one lineage on it from another.
    name = Repo(config.ROOT).head.commit.hexsha[:7] + (f"-{tag}" if tag else "")
    log = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        id=name,
        name=name,
        resume="allow",
        mode="disabled" if dry_run else "online",
        config={
            **configuration,
            # Not `CODEX_MODEL`: it has held a codex slug and now holds
            # whatever `MUTATOR` names, and a key that lies about which
            # program produced a run is how the switch hides.
            "MUTATOR": selected(),
            "MUTATOR_MODEL": started_on,
        },
    )
    log.define_metric("sessions")
    log.define_metric("calls")
    log.define_metric("sessions/*", step_metric="sessions")
    log.define_metric("gate/*", step_metric="sessions")
    log.define_metric("calls/*", step_metric="calls")
    log.define_metric("database/*", step_metric="calls")
    return log
