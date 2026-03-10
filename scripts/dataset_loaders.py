"""
dataset_loaders.py
------------------
Load vulnerable/patched code pairs from HuggingFace public datasets
(BigVul, CrossVul) and normalise them into a common structure
suitable for the SVEN training pipeline.

Returned items are dicts with keys:
  func_before  – vulnerable code
  func_after   – patched / secure code
  cwe          – normalised cwe-id  (e.g. "cwe-022")
  lang         – "py" or "c"
  commit_link  – URL or dataset reference
  file_name    – suggested filename with correct extension
"""

from __future__ import annotations
import re
from typing import Generator

# ──────────────────────────────────────────────────────────────────────────
# Supported (cwe, lang) pairs must match SUPPORTED_CWE_QL in codeql_validator
# ──────────────────────────────────────────────────────────────────────────
SUPPORTED_CWES = {
    "cwe-022", "cwe-078", "cwe-079", "cwe-089",
    "cwe-125", "cwe-190", "cwe-416", "cwe-476", "cwe-787",
}

_C_LANGS  = {"c", "c++", "cpp"}
_PY_LANGS = {"python", "py"}

# CWEs that only have C QL queries → Python samples will be skipped
_C_ONLY_CWES = {"cwe-125", "cwe-190", "cwe-416", "cwe-476", "cwe-787"}


def _normalise_cwe(raw: str | int | None) -> str | None:
    """Return 'cwe-NNN' (lowercase, zero-padded to 3 digits) or None."""
    if raw is None:
        return None
    s = str(raw).strip().upper()
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


def _is_usable_pair(before: str, after: str, cwe: str, lang: str) -> bool:
    if not before or not after:
        return False
    if before.strip() == after.strip():
        return False
    if cwe not in SUPPORTED_CWES:
        return False
    if lang not in ("py", "c"):
        return False
    if lang == "py" and cwe in _C_ONLY_CWES:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────
# BigVul  (HuggingFace: bstee615/bigvul)
#
# Actual columns (confirmed):
#   'CVE ID', 'CVE Page', 'CWE ID', 'codeLink', 'commit_id',
#   'commit_message', 'func_after', 'func_before', 'lang', 'project', 'vul'
# ─────────────────────────────────────────────────────────────────────────

def load_bigvul(
    cwe_filter: set[str] | None = None,
    lang_filter: set[str] | None = None,
    max_per_cwe_lang: int = 500,
) -> Generator[dict, None, None]:
    """Yield normalised code pairs from the BigVul dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    print("[BigVul] Loading dataset from HuggingFace (bstee615/bigvul)…")
    # Note: trust_remote_code is not supported in datasets >= 3.0
    ds = load_dataset("bstee615/bigvul", split="train")

    count: dict[tuple, int] = {}

    for row in ds:
        before    = row.get("func_before") or ""
        after     = row.get("func_after")  or ""
        raw_cwe   = row.get("CWE ID")      or ""   # capital letters!
        raw_lang  = row.get("lang")        or ""

        cwe  = _normalise_cwe(raw_cwe)
        lang = _normalise_lang(raw_lang)

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
#
# Actual columns (confirmed from dataset card):
#   'cwe_id', 'cwe_description', 'language', 'vulnerable_code',
#   'fixed_code', 'file_pair_id', 'source', 'language_dir'
# ─────────────────────────────────────────────────────────────────────────

def load_crossvul(
    cwe_filter: set[str] | None = None,
    lang_filter: set[str] | None = None,
    max_per_cwe_lang: int = 500,
) -> Generator[dict, None, None]:
    """Yield normalised code pairs from the CrossVul dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    print("[CrossVul] Loading dataset from HuggingFace (hitoshura25/crossvul)…")
    ds = load_dataset("hitoshura25/crossvul", split="train")

    count: dict[tuple, int] = {}

    for row in ds:
        before   = row.get("vulnerable_code") or ""
        after    = row.get("fixed_code")       or ""  # 'fixed_code' not 'patched_code'
        raw_cwe  = row.get("cwe_id")           or ""
        raw_lang = row.get("language")         or ""

        cwe  = _normalise_cwe(raw_cwe)
        lang = _normalise_lang(raw_lang)

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
# Combined loader
# ─────────────────────────────────────────────────────────────────────────

def load_all_pairs(
    datasets: list[str] | None = None,
    cwe_filter: set[str] | None = None,
    lang_filter: set[str] | None = None,
    max_per_cwe_lang: int = 500,
) -> Generator[dict, None, None]:
    """Yield pairs from all requested datasets."""
    datasets = datasets or ["bigvul", "crossvul"]

    loader_map = {
        "bigvul":   load_bigvul,
        "crossvul": load_crossvul,
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
