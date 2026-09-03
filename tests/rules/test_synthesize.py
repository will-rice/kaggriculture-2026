"""Parsing is tested without calling the LLM; the LLM is tested by hand once."""

import subprocess

import pytest

from kaggriculture.rules import synthesize
from kaggriculture.rules.synthesize import parse_candidates, propose


def test_a_fenced_json_block_is_parsed() -> None:
    """Codex wraps output in prose and fences; the parser must survive both."""
    text = """Here is my proposal.

```json
{"rules": [{"name": "prebuy",
  "condition": {"field": "step", "op": "eq", "value": 0},
  "effect": {"kind": "market_insert", "order": ["BUY_PRODUCT", "WHEAT", 51], "at": 0}}]}
```

That should squeeze their opening."""

    candidates = parse_candidates(text)

    assert len(candidates) == 1
    assert candidates[0].rules[0].name == "prebuy"


def test_an_invalid_candidate_is_dropped_not_raised() -> None:
    """One bad proposal in a batch must not lose the good ones."""
    text = """
```json
{"rules": [{"name": "bad", "condition": {"field": "opponent.private.seeds",
  "op": "ge", "value": 1}, "effect": {"kind": "market_insert", "order": [], "at": 0}}]}
```
```json
{"rules": []}
```
"""

    assert len(parse_candidates(text)) == 1


def test_output_with_no_json_yields_nothing() -> None:
    """A refusal or an error message is not a candidate."""
    assert parse_candidates("I could not determine a good rule.") == []


def test_output_with_no_fence_at_all_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A codex response that stops fencing must not read as silent nothing.

    ``test_output_with_no_json_yields_nothing`` only checks the return value,
    which a parser that always returns ``[]`` would also satisfy. This test
    additionally requires a diagnostic naming the actual shape that arrived,
    so an always-empty parser -- or one that only logs on a parse failure,
    never on a total absence of fences -- fails it.
    """
    text = "I could not determine a good rule."

    with caplog.at_level("INFO", logger=synthesize.__name__):
        candidates = parse_candidates(text)

    assert candidates == []
    assert "no fenced JSON block" in caplog.text
    assert text in caplog.text


def test_two_objects_in_one_fence_are_dropped_as_a_pair() -> None:
    """The contract is one candidate per fence.

    This module does not guess where one JSON object ends and the next
    begins inside a single fence.
    """
    text = """
```json
{"rules": []}
{"rules": [{"name": "second",
  "condition": {"field": "step", "op": "eq", "value": 0},
  "effect": {"kind": "market_drop", "order": [], "at": 0}}]}
```
"""

    assert parse_candidates(text) == []


def test_propose_raises_when_codex_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crashed invocation must not be indistinguishable from an empty answer.

    ``fake_run`` mimics the real ``subprocess.run`` contract -- it only
    raises when called with ``check=True`` -- so this test fails (as it
    should) against a ``propose`` that calls ``subprocess.run`` with
    ``check=False`` and folds a nonzero return code into ``[]`` by hand
    instead of letting the call raise.
    """

    def fake_run(*_args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        if kwargs.get("check"):
            raise subprocess.CalledProcessError(1, ["codex"], output="", stderr="boom")
        return subprocess.CompletedProcess(
            args=["codex"], returncode=1, stdout="", stderr="boom"
        )

    monkeypatch.setattr(synthesize.subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        propose("anything")


def test_propose_returns_parsed_candidates_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful invocation still runs its stdout through the same parser."""
    stdout = """
```json
{"rules": [{"name": "prebuy",
  "condition": {"field": "step", "op": "eq", "value": 0},
  "effect": {"kind": "market_insert", "order": ["BUY_PRODUCT", "WHEAT", 51], "at": 0}}]}
```
"""

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["codex"], returncode=0, stdout=stdout, stderr=""
        )

    monkeypatch.setattr(synthesize.subprocess, "run", fake_run)

    candidates = propose("anything")

    assert len(candidates) == 1
    assert candidates[0].rules[0].name == "prebuy"
