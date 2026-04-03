"""
llm_filter.py
-------------
LLM-based filter to determine whether a code diff contains ONLY
security-related changes (no unrelated refactoring, style changes, etc.).

Used by collect_from_datasets.py and collect_augmented.py as an optional
quality filter (--llm-filter flag).
"""

from __future__ import annotations

import os
import json
import difflib
import logging

log = logging.getLogger(__name__)

# ── Prompt template ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a security code reviewer. Your task is to determine whether a code diff
contains ONLY changes related to fixing a security vulnerability.

Answer in JSON: {"security_only": true/false, "reason": "brief explanation"}

Rules:
- "security_only": true  → ALL changed lines are directly related to fixing
  the specified vulnerability type (CWE).
- "security_only": false → The diff contains changes UNRELATED to the security
  fix, such as: refactoring, adding features, fixing non-security bugs,
  changing comments/docs unrelated to the fix, code style changes, etc.
- Minor whitespace/formatting changes around the security fix are acceptable
  (still return true).
- If unsure, lean towards true.
"""

_USER_PROMPT_TEMPLATE = """\
Vulnerability type: {cwe}

```diff
{diff}
```

Are all the changes in this diff related to fixing the {cwe} vulnerability?
Answer in JSON only."""


# ── Client management ────────────────────────────────────────────────────────

_client = None


def _get_client():
    """Lazy-init the OpenAI client."""
    global _client
    if _client is None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for --llm-filter. "
                "Install with: pip install openai"
            )
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set. "
                "Set it with: export OPENAI_API_KEY='sk-...'"
            )
        _client = OpenAI(api_key=api_key)
    return _client


# ── Public API ───────────────────────────────────────────────────────────────

def count_diff_lines(before: str, after: str) -> int:
    """
    Count the number of changed lines in a unified diff.
    Only counts lines starting with '+' or '-' (excluding the header lines
    '+++' and '---').
    """
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    diff_lines = list(difflib.unified_diff(
        before_lines, after_lines,
        fromfile="a/file", tofile="b/file", n=0,
    ))

    count = 0
    for line in diff_lines:
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+") or line.startswith("-"):
            count += 1
    return count


def is_security_only_change(
    before: str,
    after: str,
    cwe: str,
    model: str = "gpt-4o-mini",
) -> bool:
    """
    Use an LLM to determine whether the diff between `before` and `after`
    contains ONLY security-related changes for the given CWE.

    Returns True if the changes are security-only (or on API error).
    Returns False if unrelated changes are detected.
    """
    # Generate diff
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    diff_str = "".join(difflib.unified_diff(
        before_lines, after_lines,
        fromfile="a/file", tofile="b/file", n=3,
    ))

    if not diff_str:
        return True  # no diff = nothing to filter

    # Truncate very long diffs to avoid token limits
    max_chars = 4000
    if len(diff_str) > max_chars:
        diff_str = diff_str[:max_chars] + "\n... (truncated)"

    user_prompt = _USER_PROMPT_TEMPLATE.format(cwe=cwe.upper(), diff=diff_str)

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        content = response.choices[0].message.content.strip()

        # Parse JSON response
        # Handle cases where the model wraps JSON in markdown code blocks
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        result = json.loads(content)
        is_sec = result.get("security_only", True)
        reason = result.get("reason", "")

        log.debug("[LLM] security_only=%s  reason=%s", is_sec, reason)
        return bool(is_sec)

    except json.JSONDecodeError as e:
        log.warning("[LLM] Could not parse response as JSON: %s – keeping entry.", e)
        return True  # fail-open: keep the entry

    except Exception as e:
        log.warning("[LLM] API error: %s – keeping entry.", e)
        return True  # fail-open: keep the entry
