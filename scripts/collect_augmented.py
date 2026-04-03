"""
collect_augmented.py
--------------------
Augmented pipeline that extends collect_from_datasets.py to cover additional
CWEs (2025 OWASP Top 10) and an additional dataset (CVEfixes).

Collection strategy (simple & transparent):
  - For each dataset (bigvul / crossvul / cvefixes) × each CWE, collect at
    most --max-samples (default 50) pairs **per dataset**.
  - If a dataset has fewer than --max-samples for a given CWE, all of them
    are used (no silent skip).
  - Validation follows the same logic as collect_from_datasets.py:
      * --trust-dataset-labels  skip the "before" scan (recommended for C)
      * --no-codeql             skip all CodeQL scans

Target CWEs (union of SVEN original + 2025 OWASP Top 10):
  cwe-022 078 079 089 094 122 125 190 288 306 416 476 502 787

Usage:
  python collect_augmented.py --datasets bigvul crossvul cvefixes \\
      --trust-dataset-labels --max-samples 50 \\
      --output-dir ../data_train_val/augmented/train
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import difflib
import logging
from collections import defaultdict
from typing import Any, Generator

# ── Paths ────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SVEN_ROOT  = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SVEN_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

from sven.utils import parse_diff                            # SVEN diff parser
from codeql_validator import scan_with_codeql, SUPPORTED_CWE_QL
from llm_filter import count_diff_lines, is_security_only_change

# ── Extend CodeQL support at runtime for new CWEs ────────────────────────────
_QL_REPO = os.path.join(_SVEN_ROOT, "codeql", "codeql-repo")

def _ql(rel: str) -> str:
    """Return absolute QL path; only register if the file actually exists."""
    return os.path.join(_QL_REPO, rel)

_CWE_QL_EXTENSIONS: dict[str, dict[str, list[str]]] = {
    "cwe-094": {
        "py": [_ql("python/ql/src/Security/CWE-094/CodeInjection.ql")],
    },
    "cwe-502": {
        "py": [_ql("python/ql/src/Security/CWE-502/UnsafeDeserialization.ql")],
    },
    "cwe-122": {
        "c": [_ql("cpp/ql/src/Security/CWE/CWE-119/OverflowBuffer.ql"),
              _ql("cpp/ql/src/Critical/OverflowStatic.ql")],
    },
    # cwe-288, cwe-306: no CodeQL queries → Bandit-only / trust-label
}

for _cwe, _langs in _CWE_QL_EXTENSIONS.items():
    if _cwe not in SUPPORTED_CWE_QL:
        SUPPORTED_CWE_QL[_cwe] = {}
    for _lang, _paths in _langs.items():
        SUPPORTED_CWE_QL[_cwe][_lang] = [p for p in _paths if os.path.exists(p)]

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ── CWEs targeted by this script ─────────────────────────────────────────────
# Union of SVEN original + 2025 OWASP Top 10
SUPPORTED_CWES: set[str] = {
    "cwe-022",   # Path Traversal
    "cwe-078",   # OS Command Injection
    "cwe-079",   # XSS
    "cwe-089",   # SQL Injection
    "cwe-094",   # Code Injection        (new)
    "cwe-122",   # Heap Buffer Overflow  (new, C only)
    "cwe-125",   # Out-of-bounds Read
    "cwe-190",   # Integer Overflow
    "cwe-288",   # Auth Bypass           (new)
    "cwe-306",   # Missing Auth          (new)
    "cwe-416",   # Use-After-Free
    "cwe-476",   # NULL Pointer Deref
    "cwe-502",   # Unsafe Deserialization(new)
    "cwe-787",   # Out-of-bounds Write
}

_C_LANGS  = {"c", "c++", "cpp"}
_PY_LANGS = {"python", "py"}
_C_ONLY_CWES  = {"cwe-122", "cwe-125", "cwe-190", "cwe-416", "cwe-476", "cwe-787"}
_PY_ONLY_CWES = {"cwe-288", "cwe-306", "cwe-502"}

DEFAULT_OUTPUT_DIR = os.path.join(_SVEN_ROOT, "data_train_val", "augmented", "train")
DEFAULT_MAX = 50    # per dataset per (cwe, lang)


# ── Normalisation helpers ─────────────────────────────────────────────────────

def _normalise_cwe(raw: str | int | None) -> str | None:
    import re
    if raw is None:
        return None
    s = str(raw).strip()
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    return f"cwe-{int(m.group(1)):03d}"


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
    if before.strip() == after.strip():
        return False
    if cwe not in SUPPORTED_CWES:
        return False
    if lang not in ("py", "c"):
        return False
    if lang == "py" and cwe in _C_ONLY_CWES:
        return False
    if lang == "c" and cwe in _PY_ONLY_CWES:
        return False
    return True



# ── Dataset loaders ───────────────────────────────────────────────────────────

def _all_buckets_full(
    count: dict,
    cwe_filter: set[str] | None,
    lang_filter: set[str] | None,
    max_per_cwe_lang: int,
) -> bool:
    """Return True when every (cwe, lang) bucket that could be filled is full."""
    cwes  = cwe_filter  if cwe_filter  else SUPPORTED_CWES
    langs = lang_filter if lang_filter else {"py", "c"}
    for cwe in cwes:
        for lang in langs:
            if lang == "py" and cwe in _C_ONLY_CWES:
                continue
            if lang == "c" and cwe in _PY_ONLY_CWES:
                continue
            if count.get((cwe, lang), 0) < max_per_cwe_lang:
                return False
    return True

def _load_bigvul(
    cwe_filter: set[str] | None,
    lang_filter: set[str] | None,
    max_per_cwe_lang: int,
) -> Generator[dict, None, None]:
    from datasets import load_dataset
    log.info("[BigVul] Loading dataset…")
    ds = load_dataset("bstee615/bigvul", split="train")
    count: dict[tuple, int] = {}
    for row in ds:
        before = row.get("func_before") or ""
        after  = row.get("func_after")  or ""
        cwe    = _normalise_cwe(row.get("CWE ID"))
        lang   = _normalise_lang(row.get("lang"))
        if not _is_usable_pair(before, after, cwe, lang):
            continue
        if cwe_filter  and cwe  not in cwe_filter:
            continue
        if lang_filter and lang not in lang_filter:
            continue
        key = (cwe, lang)
        if count.get(key, 0) >= max_per_cwe_lang:
            # Early exit: check if all requested buckets are full
            if _all_buckets_full(count, cwe_filter, lang_filter, max_per_cwe_lang):
                log.info("[BigVul] All buckets full, stopping early.")
                return
            continue
        count[key] = count.get(key, 0) + 1
        ext = ".py" if lang == "py" else ".c"
        yield {
            "func_before": before,
            "func_after":  after,
            "cwe":         cwe,
            "lang":        lang,
            "commit_link": row.get("codeLink") or row.get("CVE Page") or "bigvul",
            "file_name":   f"bigvul_{cwe.replace('-','_')}{ext}",
            "source":      "bigvul",
        }


def _load_crossvul(
    cwe_filter: set[str] | None,
    lang_filter: set[str] | None,
    max_per_cwe_lang: int,
) -> Generator[dict, None, None]:
    from datasets import load_dataset
    log.info("[CrossVul] Loading dataset…")
    ds = load_dataset("hitoshura25/crossvul", split="train")
    count: dict[tuple, int] = {}
    for row in ds:
        before = row.get("vulnerable_code") or ""
        after  = row.get("fixed_code")      or ""
        cwe    = _normalise_cwe(row.get("cwe_id"))
        lang   = _normalise_lang(row.get("language"))
        if not _is_usable_pair(before, after, cwe, lang):
            continue
        if cwe_filter  and cwe  not in cwe_filter:
            continue
        if lang_filter and lang not in lang_filter:
            continue
        key = (cwe, lang)
        if count.get(key, 0) >= max_per_cwe_lang:
            if _all_buckets_full(count, cwe_filter, lang_filter, max_per_cwe_lang):
                log.info("[CrossVul] All buckets full, stopping early.")
                return
            continue
        count[key] = count.get(key, 0) + 1
        ext = ".py" if lang == "py" else ".c"
        yield {
            "func_before": before,
            "func_after":  after,
            "cwe":         cwe,
            "lang":        lang,
            "commit_link": row.get("file_pair_id") or "crossvul",
            "file_name":   f"crossvul_{cwe.replace('-','_')}{ext}",
            "source":      "crossvul",
        }


_CVEFIXES_BEFORE_COLS = ["func_before", "vulnerable_func", "vulnerable_code", "code_before", "before"]
_CVEFIXES_AFTER_COLS  = ["func_after",  "patched_func",    "fixed_code",      "code_after",  "after"]
_CVEFIXES_LANG_COLS   = ["language", "lang", "programming_language"]
_CVEFIXES_CWE_COLS    = ["cwe_id", "CWE_ID", "cwe", "CWE"]


def _pick(row: dict, candidates: list[str]) -> str | None:
    for col in candidates:
        v = row.get(col)
        if v and str(v).strip():
            return str(v).strip()
    return None


def _load_cvefixes(
    cwe_filter: set[str] | None,
    lang_filter: set[str] | None,
    max_per_cwe_lang: int,
    dataset_id: str = "AhmedShahriarSaki/CVEfixes_v1.0.7",
    streaming: bool = True,
) -> Generator[dict, None, None]:
    from datasets import load_dataset
    log.info("[CVEfixes] Loading dataset (%s, streaming=%s)…", dataset_id, streaming)
    try:
        ds = load_dataset(dataset_id, split="train", streaming=streaming)
    except Exception as e:
        log.warning("[CVEfixes] Could not load dataset: %s – skipping.", e)
        return

    count: dict[tuple, int] = {}
    seen = False
    for row in ds:
        if not seen:
            seen = True
            log.info("[CVEfixes] columns: %s", list(row.keys())[:12])
        before = _pick(row, _CVEFIXES_BEFORE_COLS) or ""
        after  = _pick(row, _CVEFIXES_AFTER_COLS)  or ""
        cwe    = _normalise_cwe(_pick(row, _CVEFIXES_CWE_COLS))
        lang   = _normalise_lang(_pick(row, _CVEFIXES_LANG_COLS))
        if not _is_usable_pair(before, after, cwe, lang):
            continue
        if cwe_filter  and cwe  not in cwe_filter:
            continue
        if lang_filter and lang not in lang_filter:
            continue
        key = (cwe, lang)
        if count.get(key, 0) >= max_per_cwe_lang:
            if _all_buckets_full(count, cwe_filter, lang_filter, max_per_cwe_lang):
                log.info("[CVEfixes] All buckets full, stopping early.")
                return
            continue
        count[key] = count.get(key, 0) + 1
        repo_hash = _pick(row, ["commit_hash", "commit_id", "commit"]) or ""
        repo_url  = _pick(row, ["repo_url", "repo", "repository"]) or "cvefixes"
        ext = ".py" if lang == "py" else ".c"
        yield {
            "func_before": before,
            "func_after":  after,
            "cwe":         cwe,
            "lang":        lang,
            "commit_link": f"{repo_url}/commit/{repo_hash}" if repo_hash else repo_url,
            "file_name":   f"cvefixes_{cwe.replace('-','_')}{ext}",
            "source":      "cvefixes",
        }
    if not seen:
        log.warning("[CVEfixes] No rows returned from dataset.")


_LOADERS = {
    "bigvul":   _load_bigvul,
    "crossvul": _load_crossvul,
    "cvefixes": _load_cvefixes,
}


# ── Core conversion (identical to collect_from_datasets.py) ──────────────────

def pair_to_sven_entry(
    before: str,
    after: str,
    cwe: str,
    lang: str,
    commit_link: str,
    file_name: str,
) -> dict[str, Any] | None:
    """
    Identical to the same function in collect_from_datasets.py:
    compute unified diff, run parse_diff, return the first entry.
    Returns None if no function-level change is found.
    """
    before_lines = before.splitlines(keepends=True)
    after_lines  = after.splitlines(keepends=True)
    diff_str = "".join(difflib.unified_diff(
        before_lines, after_lines,
        fromfile="a/" + file_name,
        tofile="b/"   + file_name,
        n=3,
    ))

    if not diff_str:
        return None

    entries = parse_diff(file_name, before, after, diff_str)
    if not entries:
        return None

    # Take the first (largest / most significant) function change
    entry = entries[0]

    # Guard: parse_diff may return a function from the file whose body is
    # unchanged (the actual patch was in a different function).  Skip these.
    if entry.get("func_src_before", "").strip() == entry.get("func_src_after", "").strip():
        return None

    entry["commit_link"] = commit_link
    entry["file_name"]   = file_name
    entry["vul_type"]    = cwe.lower()
    return entry



# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Augmented collection: BigVul + CrossVul + CVEfixes → SVEN JSONL"
    )
    p.add_argument("--datasets", nargs="+", default=["bigvul", "crossvul", "cvefixes"],
                   choices=list(_LOADERS), help="Datasets to pull from.")
    p.add_argument("--cwe", nargs="+", metavar="CWE",
                   help="Only collect these CWEs (e.g. cwe-022 cwe-094).")
    p.add_argument("--lang", choices=["py", "c", "all"], default="all",
                   help="Language filter.")
    p.add_argument("--max-samples", type=int, default=DEFAULT_MAX, metavar="N",
                   help=f"Max samples per (dataset, CWE, lang) bucket. Default: {DEFAULT_MAX}.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}")
    p.add_argument("--no-codeql", action="store_true",
                   help="Skip all CodeQL validation.")
    p.add_argument("--trust-dataset-labels", action="store_true",
                   help="Skip CodeQL scan on vulnerable-before code (recommended for C).")
    p.add_argument("--max-diff-lines", type=int, default=None, metavar="N",
                   help="Skip pairs whose diff exceeds N changed lines (e.g. 40). "
                        "Default: no limit.")
    p.add_argument("--llm-filter", action="store_true",
                   help="Use an LLM to filter out diffs that contain non-security changes. "
                        "Requires OPENAI_API_KEY env var.")
    p.add_argument("--llm-model", default="gpt-4o-mini", metavar="MODEL",
                   help="LLM model to use for --llm-filter. Default: gpt-4o-mini.")
    p.add_argument("--dry-run", action="store_true",
                   help="Process pairs but do not write files.")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = get_args()

    cwe_filter: set[str] | None = None
    if args.cwe:
        cwe_filter = {c.lower() for c in args.cwe}

    lang_filter: set[str] | None = None
    if args.lang != "all":
        lang_filter = {args.lang}

    log.info("=" * 62)
    log.info("Augmented dataset collection")
    log.info("  Datasets  : %s", args.datasets)
    log.info("  CWEs      : %s", sorted(cwe_filter) if cwe_filter else "all (%d)" % len(SUPPORTED_CWES))
    log.info("  Max/bucket: %d  (per dataset × CWE × lang)", args.max_samples)
    log.info("  CodeQL    : %s  trust-labels: %s  dry-run: %s",
             not args.no_codeql, args.trust_dataset_labels, args.dry_run)
    log.info("  Output    : %s", args.output_dir)
    log.info("=" * 62)

    stats: dict[str, dict] = defaultdict(
        lambda: {"loaded": 0, "vul_ok": 0, "sec_ok": 0,
                 "diff_skip": 0, "llm_skip": 0,
                 "saved": 0, "skipped": 0}
    )
    out_handles: dict[str, Any] = {}

    def get_fh(cwe: str):
        if cwe in out_handles:
            return out_handles[cwe]
        if args.dry_run:
            out_handles[cwe] = None
            return None
        os.makedirs(args.output_dir, exist_ok=True)
        path = os.path.join(args.output_dir, f"{cwe}.jsonl")
        fh = open(path, "a", encoding="utf-8")   # append – don't overwrite existing data
        out_handles[cwe] = fh
        log.info("Opened: %s", path)
        return fh

    try:
        for ds_name in args.datasets:
            loader = _LOADERS.get(ds_name)
            if loader is None:
                log.warning("Unknown dataset '%s', skipping.", ds_name)
                continue

            for pair in loader(
                cwe_filter=cwe_filter,
                lang_filter=lang_filter,
                max_per_cwe_lang=args.max_samples,
            ):
                cwe    = pair["cwe"]
                lang   = pair["lang"]
                before = pair["func_before"]
                after  = pair["func_after"]
                key    = f"{ds_name}/{cwe}/{lang}"
                st     = stats[key]
                st["loaded"] += 1

                # ── 1. Validate vulnerable code ──────────────────────────
                if not args.no_codeql:
                    if args.trust_dataset_labels:
                        st["vul_ok"] += 1
                    else:
                        vul_status = scan_with_codeql(before, lang, cwe)
                        if vul_status == "VULNERABLE":
                            st["vul_ok"] += 1
                        elif vul_status == "ERROR":
                            log.debug("[%s] CodeQL error on before-code, skipping.", key)
                            st["skipped"] += 1
                            continue
                        else:
                            log.debug("[%s] Before-code not flagged by CodeQL, skipping.", key)
                            st["skipped"] += 1
                            continue
                else:
                    st["vul_ok"] += 1

                # ── 2. Validate patched code ─────────────────────────────
                if not args.no_codeql:
                    sec_status = scan_with_codeql(after, lang, cwe)
                    if sec_status == "SECURE":
                        st["sec_ok"] += 1
                    elif sec_status == "ERROR":
                        log.debug("[%s] CodeQL error on after-code, accepting.", key)
                        st["sec_ok"] += 1
                    else:
                        log.debug("[%s] After-code still vulnerable, skipping.", key)
                        st["skipped"] += 1
                        continue
                else:
                    st["sec_ok"] += 1

                # ── 3. Diff line-count filter (opt-in) ───────────────────
                if args.max_diff_lines is not None:
                    n_diff = count_diff_lines(before, after)
                    if n_diff > args.max_diff_lines:
                        log.debug("[%s] diff too large (%d lines > %d), skipping.",
                                  key, n_diff, args.max_diff_lines)
                        st["diff_skip"] += 1
                        continue

                # ── 4. LLM security-relevance filter (opt-in) ─────────────
                if args.llm_filter:
                    if not is_security_only_change(before, after, cwe, model=args.llm_model):
                        log.info("[%s] LLM says diff contains non-security changes, skipping.", key)
                        st["llm_skip"] += 1
                        continue

                # ── 5. Convert to SVEN JSONL (identical to original) ─────
                entry = pair_to_sven_entry(
                    before=before,
                    after=after,
                    cwe=cwe,
                    lang=lang,
                    commit_link=pair.get("commit_link", ""),
                    file_name=pair.get("file_name", f"file.{'py' if lang=='py' else 'c'}"),
                )
                if entry is None:
                    log.debug("[%s] parse_diff: no function-level change found, skipping.", key)
                    st["skipped"] += 1
                    continue

                # ── 6. Write ─────────────────────────────────────────────
                st["saved"] += 1
                if args.dry_run:
                    log.info("[DRY-RUN] %s  func=%s", key, entry.get("func_name", "?"))
                else:
                    fh = get_fh(cwe)
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    fh.flush()
                    log.info("[%s] saved func=%-30s (saved=%d)",
                             key, entry.get("func_name", "?")[:30], st["saved"])

    finally:
        for fh in out_handles.values():
            if fh is not None:
                fh.close()

    # ── Summary ──────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 62)
    log.info("Collection complete")
    log.info("=" * 62)
    log.info("%-35s %7s %7s %7s %7s %7s %7s %7s",
             "bucket", "loaded", "vul_ok", "sec_ok", "d_skip", "l_skip", "saved", "skip")
    total = 0
    for key in sorted(stats):
        s = stats[key]
        log.info("%-35s %7d %7d %7d %7d %7d %7d %7d",
                 key, s["loaded"], s["vul_ok"], s["sec_ok"],
                 s["diff_skip"], s["llm_skip"],
                 s["saved"], s["skipped"])
        total += s["saved"]
    log.info("%-35s %7s %7s %7s %7d", "TOTAL", "", "", "", total)
    log.info("Output: %s", args.output_dir)
    if args.dry_run:
        log.info("(DRY RUN – no files written)")


if __name__ == "__main__":
    main()
