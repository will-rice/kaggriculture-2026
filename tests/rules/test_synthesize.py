"""Parsing is tested without calling the LLM; the LLM is tested by hand once."""

from kaggriculture.rules.synthesize import parse_candidates


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
