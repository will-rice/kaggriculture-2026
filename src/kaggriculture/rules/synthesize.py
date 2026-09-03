"""Ask codex for candidate rule sets and keep the ones that validate.

An LLM proposal that does not validate is dropped rather than repaired: the
schema is the contract, and silently fixing a malformed proposal would measure
our repair rather than its idea.

Two shapes are deliberately not repaired. A fence with no JSON inside logs
what actually arrived and moves on -- worth knowing about, since a codex
change that stops fencing its output would otherwise read as a silent string
of empty batches. A single fence containing more than one JSON object is
*not* split and salvaged: the contract is one candidate per fenced block, and
guessing where one object ends and the next begins is exactly the kind of
repair this module exists to avoid. It shows up as one dropped, logged parse
failure, same as any other malformed block.
"""

import json
import logging
import re
import subprocess

from pydantic import ValidationError

from kaggriculture.rules.spec import RuleSet

LOGGER = logging.getLogger(__name__)

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_candidates(text: str) -> list[RuleSet]:
    """Return every valid rule set in the model's output.

    Args:
        text: Raw stdout from the model.

    Returns:
        The candidates that parsed and validated, in order.
    """
    blocks = _FENCE.findall(text)
    if not blocks:
        LOGGER.info("no fenced JSON block in model output: %r", text[:200])
    candidates: list[RuleSet] = []
    for block in blocks:
        try:
            candidates.append(RuleSet.model_validate(json.loads(block)))
        except (json.JSONDecodeError, ValidationError) as error:
            LOGGER.info("dropping an unparseable candidate: %s", str(error)[:160])
    return candidates


def propose(prompt: str, timeout: float = 600.0) -> list[RuleSet]:
    """Run ``codex exec`` with the prompt and return the candidates it wrote.

    Args:
        prompt: The full prompt, including the schema and prior results.
        timeout: Seconds to allow before giving up.

    Returns:
        Validated candidate rule sets. Empty only means codex ran and wrote
        no valid candidate -- a failed or timed-out invocation is never
        folded into this list, so an empty return always means "codex
        answered, but with nothing usable".

    Raises:
        subprocess.CalledProcessError: If ``codex exec`` exits nonzero. A
            broken invocation is not a legitimate empty result.
        subprocess.TimeoutExpired: If ``codex exec`` exceeds ``timeout``
            seconds. A timeout is not an empty result either.
    """
    try:
        result = subprocess.run(
            ["codex", "exec", "--sandbox", "read-only", prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        LOGGER.warning("codex exec failed: %s", error.stderr.strip()[:200])
        raise
    return parse_candidates(result.stdout)
