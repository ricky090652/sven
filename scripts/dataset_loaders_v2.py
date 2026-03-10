"""
dataset_loaders_v2.py
---------------------
Load vulnerable/patched code pairs from BigVul, CrossVul, and CVEfixes,
normalise them into a common structure for the SVEN training pipeline.

This is a NEW file that does NOT modify the original dataset_loaders.py.
It extends the supported CWEs to cover both SVEN original CWEs and the
2025 OWASP Top 10:

  SVEN original : cwe-022 078 079 089 125 190 416 476 787
  2025 Top 10   : cwe-022 078 079 094 122 288 306 416 502 787
  Union (target): cwe-022 078 079 089 094 122 125 190 288 306 416 476 502 787

Returned items are dicts with keys:
  func_before   – vulnerable code string
  func_after    – patched / secure code string
  cwe           – normalised cwe-id  (e.g. "cwe-022")
  lang          – "py" or "c"
  commit_link   – URL or dataset reference
  file_name     – suggested filename with correct extension
  source        – "bigvul" | "crossvul" | "cvefixes"
"""

from __future__ import annotations
import re
from typing import Generator

# ──────────────────────────────────────────────────────────────────────────
# Supported CWEs: union of SVEN original + 2025 Top 10
# ──────────────────────────────────────────────────────────────────────────
SUPPORTED_CWES: set[str] = {
    "cwe-022",   # Path Traversal
    "cwe-078",   # OS Command Injection
    "cwe-079",   # XSS
    "cwe-089",   # SQL Injection         (SVEN original)
    "cwe-094",   # Code Injection        (2025 Top 10)
    "cwe-122",   # Heap Buffer Overflow  (2025 Top 10, C only)
    "cwe-125",   # Out-of-bounds Read    (SVEN original, C only)
    "cwe-190",   # Integer Overflow      (SVEN original, C only)
    "cwe-288",   # Auth Bypass           (2025 Top 10)
    "cwe-306",   # Missing Authentication(2025 Top 10)
    "cwe-416",   # Use-After-Free        (C only)
    "cwe-476",   # NULL Pointer Deref    (SVEN original, C only)
    "cwe-502",   # Unsafe Deserialization(2025 Top 10)
    "cwe-787",   # Out-of-bounds Write   (C only)
}

# CWEs where only C samples make sense (no Python QL queries, no Bandit rules)
C_ONLY_CWES: set[str] = {
    "cwe-122", "cwe-125", "cwe-190", "cwe-416", "cwe-476", "cwe-787",
}

# CWEs where only Python/scripting samples make sense
PY_ONLY_CWES: set[str] = {
    "cwe-288", "cwe-306", "cwe-502",
}

_C_LANGS  = {"c", "c++", "cpp"}
_PY_LANGS = {"python", "py"}


def _normalise_cwe(raw: str | int | None) -> str | None:
    """Return 'cwe-NNN' (lowercase, zero-padded to 3 digits) or None."""
    if raw is None:
        return None
    s = str(raw).strip().upper()
    # Handle "CWE-22" or bare "22"
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    num = int(m.group(1))
    return f"cwe-{num:03d}"


def _normalise_lang(raw: str | None) -> str | None:
    if raw is None:
        return None
    low = raw.strip().lower()
    if low in _C_LANGS:
        return "c"
    if low in _PY_LANGS:
        return "py"
    return None


def _is_usable_pair(before: str, after: str, cwe: str | None, lang: str | None) -> bool:
    if not before or not after:
        return False
    if not before.strip() or not after.strip():
        return False
    if before.strip() == after.strip():
        return False
    if cwe not in SUPPORTED_CWES:
        return False
    if lang not in ("py", "c"):
        return False
    # Reject cross-language combinations that have no validation support
    if lang == "py" and cwe in C_ONLY_CWES:
        return False
    if lang == "c" and cwe in PY_ONLY_CWES:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────
# BigVul  (HuggingFace: bstee615/bigvul)
# ─────────────────────────────────────────────────────────────────────────
def load_bigvul(
    cwe_filter: set[str] | None = None,
    lang_filter: set[str] | None = None,
    max_per_cwe_lang: int = 25,
) -> Generator[dict, None, None]:
    """Yield normalised code pairs from the BigVul dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    print("[BigVul] Loading dataset (bstee615/bigvul)…")
    ds = load_dataset("bstee615/bigvul", split="train")

    count: dict[tuple, int] = {}
    for row in ds:
        before   = row.get("func_before") or ""
        after    = row.get("func_after")  or ""
        cwe      = _normalise_cwe(row.get("CWE ID"))
        lang     = _normalise_lang(row.get("lang"))

        if not _is_usable_pair(before, after, cwe, lang):
            continue
        if cwe_filter  and cwe  not in cwe_filter:
            continue
        if lang_filter and lang not in lang_filter:
            continue

        key = (cwe, lang)
        if count.get(key, 0) >= max_per_cwe_lang:
            continue
        count[key] = count.get(key, 0) + 1

        commit_link = row.get("codeLink") or row.get("CVE Page") or "bigvul"
        ext = ".py" if lang == "py" else ".c"
        yield {
            "func_before": before,
            "func_after":  after,
            "cwe":         cwe,
            "lang":        lang,
            "commit_link": commit_link,
            "file_name":   f"bigvul_{cwe.replace('-','_')}{ext}",
            "source":      "bigvul",
        }


# ─────────────────────────────────────────────────────────────────────────
# CrossVul  (HuggingFace: hitoshura25/crossvul)
# ─────────────────────────────────────────────────────────────────────────
def load_crossvul(
    cwe_filter: set[str] | None = None,
    lang_filter: set[str] | None = None,
    max_per_cwe_lang: int = 25,
) -> Generator[dict, None, None]:
    """Yield normalised code pairs from the CrossVul dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    print("[CrossVul] Loading dataset (hitoshura25/crossvul)…")
    ds = load_dataset("hitoshura25/crossvul", split="train")

    count: dict[tuple, int] = {}
    for row in ds:
        before   = row.get("vulnerable_code") or ""
        after    = row.get("fixed_code")      or ""
        cwe      = _normalise_cwe(row.get("cwe_id"))
        lang     = _normalise_lang(row.get("language"))

        if not _is_usable_pair(before, after, cwe, lang):
            continue
        if cwe_filter  and cwe  not in cwe_filter:
            continue
        if lang_filter and lang not in lang_filter:
            continue

        key = (cwe, lang)
        if count.get(key, 0) >= max_per_cwe_lang:
            continue
        count[key] = count.get(key, 0) + 1

        commit_link = row.get("file_pair_id") or "crossvul"
        ext = ".py" if lang == "py" else ".c"
        yield {
            "func_before": before,
            "func_after":  after,
            "cwe":         cwe,
            "lang":        lang,
            "commit_link": commit_link,
            "file_name":   f"crossvul_{cwe.replace('-','_')}{ext}",
            "source":      "crossvul",
        }


# ─────────────────────────────────────────────────────────────────────────
# CVEfixes  (HuggingFace: AhmedShahriarSaki/CVEfixes_v1.0.7)
#
# Schema (from CVEfixes v1.0.7 paper):
#   func_before, func_after, language, cwe_id, repo_url, commit_hash, ...
#
# Fallback: auto-detect columns if schema differs between releases.
# ─────────────────────────────────────────────────────────────────────────
_CVEFIXES_DATASET_ID = "AhmedShahriarSaki/CVEfixes_v1.0.7"

# Alternative column name mappings to handle schema variations
_BEFORE_COLS = ["func_before", "vulnerable_func", "code_before", "before"]
_AFTER_COLS  = ["func_after",  "patched_func",    "code_after",  "after"]
_LANG_COLS   = ["language", "lang", "programming_language"]
_CWE_COLS    = ["cwe_id", "CWE_ID", "cwe", "CWE"]


def _pick_col(row: dict, candidates: list[str]) -> str | None:
    """Return the first non-empty value from candidate column names."""
    for col in candidates:
        v = row.get(col)
        if v and str(v).strip():
            return str(v).strip()
    return None


def load_cvefixes(
    cwe_filter: set[str] | None = None,
    lang_filter: set[str] | None = None,
    max_per_cwe_lang: int = 25,
    dataset_id: str = _CVEFIXES_DATASET_ID,
    streaming: bool = True,
) -> Generator[dict, None, None]:
    """
    Yield normalised code pairs from the CVEfixes dataset.

    Uses streaming=True by default to avoid downloading the full dataset
    (138K functions) before processing.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    print(f"[CVEfixes] Loading dataset ({dataset_id}, streaming={streaming})…")
    try:
        ds = load_dataset(dataset_id, split="train", streaming=streaming)
    except Exception as e:
        print(f"[CVEfixes] Failed to load '{dataset_id}': {e}")
        return

    count: dict[tuple, int] = {}
    seen_any = False

    for row in ds:
        if not seen_any:
            seen_any = True
            # Log the columns we found for debugging
            print(f"[CVEfixes] Schema columns: {list(row.keys())[:12]}")

        before = _pick_col(row, _BEFORE_COLS) or ""
        after  = _pick_col(row, _AFTER_COLS)  or ""
        lang   = _normalise_lang(_pick_col(row, _LANG_COLS))
        cwe    = _normalise_cwe(_pick_col(row, _CWE_COLS))

        if not _is_usable_pair(before, after, cwe, lang):
            continue
        if cwe_filter  and cwe  not in cwe_filter:
            continue
        if lang_filter and lang not in lang_filter:
            continue

        key = (cwe, lang)
        if count.get(key, 0) >= max_per_cwe_lang:
            continue
        count[key] = count.get(key, 0) + 1

        repo_url     = _pick_col(row, ["repo_url", "repo", "repository"]) or "cvefixes"
        commit_hash  = _pick_col(row, ["commit_hash", "commit_id", "commit"]) or ""
        commit_link  = f"{repo_url}/commit/{commit_hash}" if commit_hash else repo_url
        ext = ".py" if lang == "py" else ".c"

        yield {
            "func_before": before,
            "func_after":  after,
            "cwe":         cwe,
            "lang":        lang,
            "commit_link": commit_link,
            "file_name":   f"cvefixes_{cwe.replace('-','_')}{ext}",
            "source":      "cvefixes",
        }

    if not seen_any:
        print("[CVEfixes] WARNING: no rows were yielded from the dataset.")


# ─────────────────────────────────────────────────────────────────────────
# Combined loader
# ─────────────────────────────────────────────────────────────────────────
def load_all_pairs(
    datasets: list[str] | None = None,
    cwe_filter: set[str] | None = None,
    lang_filter: set[str] | None = None,
    max_per_cwe_lang: int = 25,
) -> Generator[dict, None, None]:
    """Yield pairs from all requested datasets (bigvul / crossvul / cvefixes)."""
    datasets = datasets or ["bigvul", "crossvul", "cvefixes"]

    loader_map = {
        "bigvul":   load_bigvul,
        "crossvul": load_crossvul,
        "cvefixes": load_cvefixes,
    }

    for name in datasets:
        if name not in loader_map:
            print(f"[WARN] Unknown dataset '{name}', skipping.")
            continue
        yield from loader_map[name](
            cwe_filter=cwe_filter,
            lang_filter=lang_filter,
            max_per_cwe_lang=max_per_cwe_lang,
        )
