"""
collect_from_datasets.py
------------------------
Automated pipeline to collect real vulnerable/patched code pairs
from public HuggingFace datasets, validate them with CodeQL, and
append to SVEN's per-CWE training JSONL files.

Usage examples
--------------
# Dry-run: show what would be processed, write nothing
python collect_from_datasets.py --dry-run

# Collect CWE-022 Python samples, cap at 50 per bucket
python collect_from_datasets.py --cwe cwe-022 --lang py --max-samples 50

# Collect all supported CWEs, all languages, all datasets
python collect_from_datasets.py --datasets bigvul crossvul --max-samples 200

# Skip CodeQL validation (faster, lower quality)
python collect_from_datasets.py --no-codeql --max-samples 100
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import difflib
import logging
from collections import defaultdict
from typing import Any

# ── import project utilities ──────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SVEN_ROOT  = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SVEN_ROOT)

from sven.utils import parse_diff                       # SVEN diff parser
from codeql_validator import scan_with_codeql, SUPPORTED_CWE_QL
from dataset_loaders   import load_all_pairs, SUPPORTED_CWES
from llm_filter import count_diff_lines, is_security_only_change

# ─────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = os.path.join(_SVEN_ROOT, "data_train_val", "train")


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect real vulnerability patches from public datasets "
                    "and append them to SVEN training JSONL files."
    )
    p.add_argument(
        "--cwe", nargs="+", metavar="CWE",
        help="CWE(s) to collect (e.g. cwe-022 cwe-089). Default: all supported.",
    )
    p.add_argument(
        "--lang", choices=["py", "c", "all"], default="all",
        help="Language filter. Default: all.",
    )
    p.add_argument(
        "--datasets", nargs="+", default=["bigvul", "crossvul"],
        choices=["bigvul", "crossvul"],
        help="Which HuggingFace datasets to pull from.",
    )
    p.add_argument(
        "--max-samples", type=int, default=200, metavar="N",
        help="Maximum samples per (CWE, lang) bucket. Default: 200.",
    )
    p.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR, metavar="DIR",
        help=f"Directory to write JSONL files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    p.add_argument(
        "--no-codeql", action="store_true",
        help="Skip CodeQL validation entirely (fastest, no quality filter).",
    )
    p.add_argument(
        "--trust-dataset-labels", action="store_true",
        help="Trust the dataset's vulnerability label (skip CodeQL check on the BEFORE "
             "code). Still validates that AFTER code is clean. Recommended for BigVul/C "
             "where isolated function snippets rarely have explicit taint sources.",
    )
    p.add_argument(
        "--max-diff-lines", type=int, default=None, metavar="N",
        help="Skip pairs whose diff exceeds N changed lines (e.g. 40). "
             "Default: no limit.",
    )
    p.add_argument(
        "--llm-filter", action="store_true",
        help="Use an LLM to filter out diffs that contain non-security changes. "
             "Requires OPENAI_API_KEY env var.",
    )
    p.add_argument(
        "--llm-model", default="gpt-4o-mini", metavar="MODEL",
        help="LLM model to use for --llm-filter. Default: gpt-4o-mini.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be processed without writing any files.",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────
# Core conversion: code pair → SVEN JSONL entry
# ─────────────────────────────────────────────────────────────────────────

def pair_to_sven_entry(
    before: str,
    after: str,
    cwe: str,
    lang: str,
    commit_link: str,
    file_name: str,
) -> dict[str, Any] | None:
    """
    Compute unified diff, call parse_diff from sven/utils.py to extract
    function-level changes, and return a dict ready to be serialised
    as one JSONL line.

    Returns None if parse_diff finds no function-level changes.
    """
    before_lines = before.splitlines(keepends=True)
    after_lines  = after.splitlines(keepends=True)
    diff_str = "".join(difflib.unified_diff(
        before_lines, after_lines,
        fromfile="a/" + file_name,
        tofile="b/" + file_name,
        n=3,
    ))

    if not diff_str:
        return None

    entries = parse_diff(file_name, before, after, diff_str)
    if not entries:
        return None

    # Take the first (largest / most significant) function change
    entry = entries[0]
    entry["commit_link"] = commit_link
    entry["file_name"]   = file_name
    entry["vul_type"]    = cwe.lower()
    return entry


# ─────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = get_args()

    # ── Determine which (CWE, lang) pairs to process ─────────────────────
    cwe_filter: set[str] | None = None
    if args.cwe:
        cwe_filter = {c.lower() for c in args.cwe}
        unsupported = cwe_filter - SUPPORTED_CWES
        if unsupported:
            log.warning("These CWEs have no QL queries and will be skipped: %s", unsupported)
        cwe_filter -= unsupported

    lang_filter: set[str] | None = None
    if args.lang != "all":
        lang_filter = {args.lang}

    # ── Stat counters ────────────────────────────────────────────────────
    stats: dict[str, dict] = defaultdict(lambda: {
        "loaded": 0, "vul_ok": 0, "sec_ok": 0,
        "diff_skip": 0, "llm_skip": 0,
        "saved": 0, "skipped": 0,
    })

    # ── Output file handles (lazy open) ─────────────────────────────────
    out_handles: dict[str, Any] = {}

    def get_out_file(cwe: str):
        if cwe in out_handles:
            return out_handles[cwe]
        out_path = os.path.join(args.output_dir, f"{cwe}.jsonl")
        if args.dry_run:
            out_handles[cwe] = None
            return None
        os.makedirs(args.output_dir, exist_ok=True)
        fh = open(out_path, "a", encoding="utf-8")
        out_handles[cwe] = fh
        log.info("Opened output: %s", out_path)
        return fh

    # ── Stream through datasets ──────────────────────────────────────────
    log.info("Starting collection  datasets=%s  lang=%s  max=%d  dry_run=%s",
             args.datasets, args.lang, args.max_samples, args.dry_run)

    try:
        for pair in load_all_pairs(
            datasets=args.datasets,
            cwe_filter=cwe_filter,
            lang_filter=lang_filter,
            max_per_cwe_lang=args.max_samples,
        ):
            cwe    = pair["cwe"]
            lang   = pair["lang"]
            before = pair["func_before"]
            after  = pair["func_after"]
            key    = f"{cwe}/{lang}"
            st     = stats[key]
            st["loaded"] += 1

            log.debug("[%s | %s] source=%s", cwe, lang, pair.get("source"))

            # ── 1. Validate vulnerable code ──────────────────────────────
            if not args.no_codeql:
                if args.trust_dataset_labels:
                    # Trust the dataset's CWE label; skip the BEFORE scan.
                    # CodeQL often can't find sources in isolated library snippets.
                    st["vul_ok"] += 1
                else:
                    vul_status = scan_with_codeql(before, lang, cwe)
                    if vul_status == "VULNERABLE":
                        st["vul_ok"] += 1
                    elif vul_status == "ERROR":
                        log.debug("[%s] CodeQL error on vulnerable code, skipping.", key)
                        st["skipped"] += 1
                        continue
                    else:  # SECURE → CodeQL disagrees, skip
                        log.debug("[%s] Vulnerable code not detected by CodeQL, skipping.", key)
                        st["skipped"] += 1
                        continue
            else:
                st["vul_ok"] += 1

            # ── 2. Validate patched code ─────────────────────────────────
            if not args.no_codeql:
                sec_status = scan_with_codeql(after, lang, cwe)
                if sec_status == "SECURE":
                    st["sec_ok"] += 1
                elif sec_status == "ERROR":
                    # CodeQL could not analyse – trust the dataset's patch
                    log.debug("[%s] CodeQL error on patched code, accepting anyway.", key)
                    st["sec_ok"] += 1
                else:  # VULNERABLE → patch didn't fix the issue
                    log.debug("[%s] Patched code still vulnerable, skipping.", key)
                    st["skipped"] += 1
                    continue
            else:
                st["sec_ok"] += 1

            # ── 3. Diff line-count filter (opt-in) ────────────────────────
            if args.max_diff_lines is not None:
                n_diff = count_diff_lines(before, after)
                if n_diff > args.max_diff_lines:
                    log.debug("[%s] diff too large (%d lines > %d), skipping.",
                              key, n_diff, args.max_diff_lines)
                    st["diff_skip"] += 1
                    continue

            # ── 4. LLM security-relevance filter (opt-in) ─────────────────
            if args.llm_filter:
                if not is_security_only_change(before, after, cwe, model=args.llm_model):
                    log.info("[%s] LLM says diff contains non-security changes, skipping.", key)
                    st["llm_skip"] += 1
                    continue

            # ── 5. Convert to SVEN JSONL format ──────────────────────────
            entry = pair_to_sven_entry(
                before=before,
                after=after,
                cwe=cwe,
                lang=lang,
                commit_link=pair.get("commit_link", ""),
                file_name=pair.get("file_name", f"file.{'py' if lang=='py' else 'c'}"),
            )

            if entry is None:
                log.debug("[%s] parse_diff found no function-level change, skipping.", key)
                st["skipped"] += 1
                continue

            # ── 6. Write entry ────────────────────────────────────────────
            st["saved"] += 1
            if args.dry_run:
                log.info("[DRY-RUN] Would write 1 entry → %s.jsonl  (func: %s)",
                         cwe, entry.get("func_name", "?"))
            else:
                fh = get_out_file(cwe)
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                fh.flush()
                log.info("[%s | %s/%s] Saved func=%s  (saved=%d)",
                         pair.get("source", "?"), cwe, lang,
                         entry.get("func_name", "?"), st["saved"])

    finally:
        for fh in out_handles.values():
            if fh is not None:
                fh.close()

    # ── Summary ──────────────────────────────────────────────────────────
    log.info("\n%s\nCollection complete.\n%s", "=" * 60, "=" * 60)
    total_saved = 0
    for key, st in sorted(stats.items()):
        log.info(
            "  %-20s  loaded=%-5d  vul_ok=%-5d  sec_ok=%-5d  diff_skip=%-5d  llm_skip=%-5d  saved=%-5d  skipped=%d",
            key, st["loaded"], st["vul_ok"], st["sec_ok"],
            st["diff_skip"], st["llm_skip"],
            st["saved"], st["skipped"],
        )
        total_saved += st["saved"]
    log.info("  %-20s  total=%d", "TOTAL", total_saved)


if __name__ == "__main__":
    main()
