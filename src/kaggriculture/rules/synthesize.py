"""Ask codex for candidate rule sets and keep the ones that validate.

An LLM proposal that does not validate is dropped rather than repaired: the
schema is the contract, and silently fixing a malformed proposal would measure
our repair rather than its idea.
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
    candidates: list[RuleSet] = []
    for block in _FENCE.findall(text):
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
        Validated candidate rule sets, possibly empty.
    """
    result = subprocess.run(
        ["codex", "exec", "--sandbox", "read-only", prompt],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        LOGGER.info("codex exec failed: %s", result.stderr.strip()[:200])
        return []
    return parse_candidates(result.stdout)
