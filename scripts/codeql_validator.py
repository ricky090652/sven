"""
codeql_validator.py
-------------------
CodeQL validation helpers for the automated dataset collection pipeline.
Mirrors the logic in sec_eval.py (codeql_create_db / codeql_analyze),
but operates on individual code snippets and returns 'VULNERABLE'/'SECURE'/'ERROR'.
"""

import os
import shutil
import tempfile
import subprocess
import random

# ──────────────────────────────────────────────
# Relative (to this script) paths to QL queries
# ──────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SVEN_ROOT  = os.path.dirname(_SCRIPT_DIR)
_CODEQL_BIN = os.path.join(_SVEN_ROOT, "codeql", "codeql")
_QL_REPO    = os.path.join(_SVEN_ROOT, "codeql", "codeql-repo")

def _ql(rel):
    return os.path.join(_QL_REPO, rel)

# ─────────────────────────────────────────────────────────────────────;
# CWE → language → list of QL query paths
# All 9 training CWEs are covered; C-only CWEs only have a 'c' entry.
# ─────────────────────────────────────────────────────────────────────
SUPPORTED_CWE_QL = {
    "cwe-022": {
        "py": [_ql("python/ql/src/Security/CWE-022/PathInjection.ql"),
               _ql("python/ql/src/Security/CWE-022/TarSlip.ql")],
        "c":  [_ql("cpp/ql/src/Security/CWE/CWE-022/TaintedPath.ql")],
    },
    "cwe-078": {
        "py": [_ql("python/ql/src/Security/CWE-078/CommandInjection.ql")],
        "c":  [_ql("cpp/ql/src/Security/CWE/CWE-078/ExecTainted.ql")],
    },
    "cwe-079": {
        "py": [_ql("python/ql/src/Security/CWE-079/ReflectedXss.ql"),
               _ql("python/ql/src/Security/CWE-079/Jinja2WithoutEscaping.ql")],
        "c":  [_ql("cpp/ql/src/Security/CWE/CWE-079/CgiXss.ql")],
    },
    "cwe-089": {
        "py": [_ql("python/ql/src/Security/CWE-089/SqlInjection.ql")],
        "c":  [_ql("cpp/ql/src/Security/CWE/CWE-089/SqlTainted.ql")],
    },
    # C-only CWEs below
    "cwe-125": {
        "c":  [_ql("cpp/ql/src/Security/CWE/CWE-119/OverflowBuffer.ql"),
               _ql("cpp/ql/src/Critical/OverflowStatic.ql"),
               _ql("cpp/ql/src/Critical/OverflowCalculated.ql")],
    },
    "cwe-190": {
        "c":  [_ql("cpp/ql/src/Security/CWE/CWE-190/ArithmeticTainted.ql"),
               _ql("cpp/ql/src/Security/CWE/CWE-190/ArithmeticUncontrolled.ql"),
               _ql("cpp/ql/src/Security/CWE/CWE-190/IntegerOverflowTainted.ql")],
    },
    "cwe-416": {
        "c":  [_ql("cpp/ql/src/Critical/UseAfterFree.ql")],
    },
    "cwe-476": {
        "c":  [_ql("cpp/ql/src/Critical/MissingNullTest.ql")],
    },
    "cwe-787": {
        "c":  [_ql("cpp/ql/src/Security/CWE/CWE-120/BadlyBoundedWrite.ql"),
               _ql("cpp/ql/src/Security/CWE/CWE-120/OverrunWrite.ql"),
               os.path.join(_QL_REPO, "cpp/ql/src/Likely Bugs/Memory Management/PotentialBufferOverflow.ql")],
    },
}


# Common C stub headers to help CodeQL compile isolated function snippets
_C_STUB_HEADER = """#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif
"""


def _create_db(src_dir: str, db_dir: str, lang: str) -> bool:
    """
    Create a CodeQL database.  For C, we:
      1. Prepend a stub header to the source file so that common symbols resolve.
      2. Copy the Makefile and try 'make -B' as in sec_eval.py.
      3. If compilation still fails, fall back to 'autobuild' which can handle
         syntactically-valid code without linking (good for isolated snippets).
    Returns True on success.
    """
    if lang == "py":
        cmd = (
            f'"{_CODEQL_BIN}" database create "{db_dir}"'
            f' --quiet --language=python --overwrite'
            f' --source-root "{src_dir}"'
        )
        result = subprocess.run(
            cmd, shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120
        )
        return result.returncode == 0

    elif lang == "c":
        # Prepend stub headers to each .c file to help CodeQL parse isolated snippets
        for fname in os.listdir(src_dir):
            if fname.endswith(".c"):
                fpath = os.path.join(src_dir, fname)
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    original = f.read()
                if "#include" not in original:
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.write(_C_STUB_HEADER + original)

        # Try 1: Makefile-based compilation (same as sec_eval.py)
        makefile_src = os.path.join(_SCRIPT_DIR, "Makefile")
        if os.path.exists(makefile_src):
            shutil.copy(makefile_src, os.path.join(src_dir, "Makefile"))

        cmd_make = (
            f'"{_CODEQL_BIN}" database create "{db_dir}"'
            f' --quiet --language=cpp --overwrite'
            f' --command="make -B"'
            f' --source-root "{src_dir}"'
        )
        result = subprocess.run(
            cmd_make, shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120
        )
        if result.returncode == 0 and os.path.isdir(db_dir):
            return True

        # Try 2: force make even if it fails ('make -B; true' always exits 0),
        #         so CodeQL can extract from whatever GCC managed to parse.
        cmd_auto = (
            f'"{_CODEQL_BIN}" database create "{db_dir}"'
            f' --quiet --language=cpp --overwrite'
            f' --command="make -B; true"'
            f' --source-root "{src_dir}"'
        )
        result = subprocess.run(
            cmd_auto, shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120
        )
        return result.returncode == 0 and os.path.isdir(db_dir)

    return False



def _analyze(db_dir: str, ql_path: str, csv_path: str) -> bool:
    """
    Run a CodeQL query against a database.
    Returns True if the analysis produced any output (= vulnerability found).
    """
    additional_packs = os.path.expanduser("~/.codeql/packages/codeql/")
    cmd = (
        f'"{_CODEQL_BIN}" database analyze "{db_dir}"'
        f' "{ql_path}"'
        f' --quiet --format=csv --output="{csv_path}"'
        f' --additional-packs="{additional_packs}"'
    )
    result = subprocess.run(
        cmd, shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=180
    )
    if result.returncode != 0:
        return False
    return os.path.exists(csv_path) and os.path.getsize(csv_path) > 0


def scan_with_codeql(code_str: str, lang: str, cwe_id: str) -> str:
    """
    Scan a code snippet with CodeQL.

    Parameters
    ----------
    code_str : source code text
    lang     : 'py' or 'c'
    cwe_id   : e.g. 'cwe-022'

    Returns
    -------
    'VULNERABLE' | 'SECURE' | 'ERROR'
    """
    cwe_id = cwe_id.lower()
    if cwe_id not in SUPPORTED_CWE_QL:
        return "ERROR"
    lang_map = SUPPORTED_CWE_QL[cwe_id]
    if lang not in lang_map:
        return "ERROR"

    ql_paths = [p for p in lang_map[lang] if os.path.exists(p)]
    if not ql_paths:
        return "ERROR"

    file_ext = ".py" if lang == "py" else ".c"

    with tempfile.TemporaryDirectory() as tmp:
        src_dir = os.path.join(tmp, "src")
        db_dir  = os.path.join(tmp, "db")
        os.makedirs(src_dir)

        src_file = os.path.join(src_dir, f"target{file_ext}")
        with open(src_file, "w", encoding="utf-8") as f:
            f.write(code_str)

        try:
            ok = _create_db(src_dir, db_dir, lang)
        except subprocess.TimeoutExpired:
            return "ERROR"
        if not ok:
            return "ERROR"

        found_vuln = False
        ran_ok = False
        for ql_path in ql_paths:
            csv_path = os.path.join(tmp, f"res_{random.randint(0, 99999)}.csv")
            try:
                hit = _analyze(db_dir, ql_path, csv_path)
                ran_ok = True
                if hit:
                    found_vuln = True
                    break          # one positive result is enough
            except subprocess.TimeoutExpired:
                continue

        if not ran_ok:
            return "ERROR"
        return "VULNERABLE" if found_vuln else "SECURE"
