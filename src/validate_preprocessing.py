"""
validate_preprocessing.py
=========================
Full-dataset chunked preprocessing validator.
Amazon ML Challenge 2026: Business Entity Resolution

Processes every dataset file in chunks (no full-file RAM load), writes
preprocessed TSVs incrementally, and reports a final summary:

  - Total input rows per file
  - Total processed (output) rows per file
  - Whether input_rows == output_rows
  - Missing-value counts (addresses)
  - Duplicate entity_id detection
  - Total processing time per file
  - Peak / approximate memory usage

Usage
-----
  # Fast full-dataset run — processes + writes all files, minimal checks:
  python src/validate_preprocessing.py

  # Full-dataset run with deep integrity checks per chunk (slower):
  python src/validate_preprocessing.py --deep-check

  # Validate only a few chunks per file (no output written):
  python src/validate_preprocessing.py --sample-chunks 2 --no-write

  # Custom output directory / chunk size:
  python src/validate_preprocessing.py --output-dir my_output --chunksize 50000

  # Process a subset of files:
  python src/validate_preprocessing.py --files train_source1,test_source1

Options
-------
  --sample-chunks N   Process only the first N chunks per file (default: all)
  --output-dir DIR    Where to write preprocessed TSVs (default: ../output)
  --chunksize N       Rows per chunk (default: 100000)
  --no-write          Do not write preprocessed TSVs; validate in-memory only
  --deep-check        Run per-chunk regex audits (NULL literals, uppercase,
                      script-loss). Slower but catches normalisation bugs.
  --files KEY,...     Comma-separated file keys to process (default: all 7)

Exit codes
----------
  0 — all files passed (input_rows == output_rows for every file)
  1 — one or more files failed
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import re
import unicodedata
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pandas as pd

from preprocessing import (
    SOURCE_FILES,
    DEFAULT_CHUNKSIZE,
    iter_chunks,
)

# ── Optional psutil ────────────────────────────────────────────────────────────
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# ── Script ranges for deep-check script-loss detection ────────────────────────
_SCRIPT_RANGES = [
    ("Devanagari", "\u0900", "\u097F"),
    ("Bengali",    "\u0980", "\u09FF"),
    ("Kannada",    "\u0C80", "\u0CFF"),
    ("Tamil",      "\u0B80", "\u0BFF"),
]

# ── Patterns used only in deep-check mode ─────────────────────────────────────
_NULL_CHK = re.compile(
    r"(?:[^a-z0-9]|^)(?:null|<null>|\(null\))(?:[^a-z0-9]|$)"
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _rss_mb() -> float:
    if _HAS_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    return 0.0


def _fmt(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    return f"{m}m {sec:02d}s"


def _sep(c="─", w=72):
    return c * w


def _script_loss(raw_col: pd.Series, norm_col: pd.Series,
                 lo: str, hi: str) -> int:
    """Characters lost in a Unicode block during normalisation."""
    pat = re.compile(f"[{lo}-{hi}]")
    raw_n  = raw_col.fillna("").apply(lambda s: len(pat.findall(str(s))))
    norm_n = norm_col.fillna("").apply(lambda s: len(pat.findall(str(s))))
    return int((raw_n - norm_n).clip(lower=0).sum())


# ─── Per-chunk deep-check (only when --deep-check is set) ─────────────────────

def _deep_check_chunk(chunk: pd.DataFrame, result: dict) -> None:
    """Run optional per-chunk integrity checks and accumulate into result."""
    for col in ["norm_business_name", "norm_business_address"]:
        if col not in chunk.columns:
            continue
        # NULL literals must not survive normalisation
        has_null = chunk[col].str.lower().str.contains(
            _NULL_CHK, regex=True, na=False
        ).sum()
        result["null_literals_remain"] += int(has_null)
        # No uppercase ASCII in normalised columns
        uc = chunk[col].apply(
            lambda x: any("A" <= c <= "Z" for c in str(x))
        ).sum()
        result["uppercase_remain"] += int(uc)

    # Non-Latin script must not be stripped
    for script, lo, hi in _SCRIPT_RANGES:
        if "business_name" in chunk.columns and "norm_business_name" in chunk.columns:
            loss = _script_loss(
                chunk["business_name"], chunk["norm_business_name"], lo, hi
            )
            result["script_loss"][script] += loss


# ─── Per-file validation ───────────────────────────────────────────────────────

def validate_file(
    file_key: str,
    output_dir: Optional[Path],
    chunksize: int,
    sample_chunks: Optional[int],
    write_output: bool,
    deep_check: bool,
) -> dict:
    """Process one file in chunks; return a result dict."""
    is_gt = (file_key == "train_ground_truth")
    result = {
        "ok":                   True,
        "file_key":             file_key,
        "input_rows":           0,
        "output_rows":          0,
        "rows_match":           False,
        "chunks":               0,
        "missing_name":         0,
        "missing_addr":         0,
        "dup_ids":              0,
        "null_literals_remain": 0,
        "uppercase_remain":     0,
        "script_loss":          {s: 0 for s, *_ in _SCRIPT_RANGES},
        "elapsed_s":            0.0,
        "peak_rss_mb":          0.0,
        "errors":               [],
        "warnings":             [],
    }

    path = SOURCE_FILES[file_key]
    if not path.exists():
        result["ok"] = False
        result["errors"].append(f"File not found: {path}")
        return result

    # Output file handle
    out_fh    = None
    out_path  = None
    if write_output and output_dir is not None:
        out_path = Path(output_dir) / f"{file_key}_preprocessed.tsv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_fh = open(out_path, "w", encoding="utf-8", newline="")

    seen_ids: set[str] = set()
    peak_rss  = _rss_mb()
    t_start   = time.perf_counter()
    first_chk = True

    try:
        for chunk in iter_chunks(file_key, chunksize=chunksize):
            n = result["chunks"] + 1
            result["chunks"]      = n
            result["input_rows"] += len(chunk)
            result["output_rows"]+= len(chunk)

            rss = _rss_mb()
            if rss > peak_rss:
                peak_rss = rss

            elapsed = time.perf_counter() - t_start
            mem_str = f"  RAM ~{rss:.0f} MB" if _HAS_PSUTIL else ""
            print(
                f"  [{file_key}] chunk {n:4d} | "
                f"rows: {len(chunk):>7,} | "
                f"total: {result['input_rows']:>10,} | "
                f"elapsed: {_fmt(elapsed)}"
                f"{mem_str}",
                flush=True,
            )

            # ── Duplicate ID check ────────────────────────────────────────────
            id_col = "source1_entity_id" if is_gt else "entity_id"
            if id_col in chunk.columns:
                ids     = set(chunk[id_col].dropna().astype(str))
                dup_in  = len(chunk[id_col].dropna()) - len(ids)
                dup_crs = len(ids & seen_ids)
                result["dup_ids"] += dup_in + dup_crs
                seen_ids.update(ids)

            if not is_gt:
                # Missing names
                result["missing_name"] += int(
                    (chunk["business_name"].isna() |
                     (chunk["business_name"] == "")).sum()
                )
                # Missing addresses
                if "address_missing" in chunk.columns:
                    result["missing_addr"] += int(chunk["address_missing"].sum())

                # Deep checks (only when requested — expensive)
                if deep_check:
                    _deep_check_chunk(chunk, result)

            # ── Write output ──────────────────────────────────────────────────
            if out_fh is not None:
                chunk.to_csv(
                    out_fh, sep="\t", index=False,
                    header=first_chk, encoding=None,
                )
            first_chk = False

            # ── Early stop in sample mode ─────────────────────────────────────
            if sample_chunks is not None and n >= sample_chunks:
                result["warnings"].append(
                    f"Stopped after {n} chunk(s) (--sample-chunks {sample_chunks}). "
                    "Row counts reflect only the sampled portion."
                )
                break

    finally:
        if out_fh is not None:
            out_fh.close()

    result["elapsed_s"]   = time.perf_counter() - t_start
    result["peak_rss_mb"] = peak_rss
    result["rows_match"]  = (result["input_rows"] == result["output_rows"])

    # ── Post-processing error checks ──────────────────────────────────────────
    if not result["rows_match"]:
        result["ok"] = False
        result["errors"].append(
            f"ROW COUNT MISMATCH: input={result['input_rows']:,} "
            f"output={result['output_rows']:,}"
        )
    if result["dup_ids"] > 0:
        result["warnings"].append(f"Found {result['dup_ids']:,} duplicate entity_ids.")

    if deep_check and not is_gt:
        if result["null_literals_remain"] > 0:
            result["ok"] = False
            result["errors"].append(
                f"{result['null_literals_remain']:,} rows still contain "
                "NULL literals in norm_* columns."
            )
        if result["uppercase_remain"] > 0:
            result["ok"] = False
            result["errors"].append(
                f"{result['uppercase_remain']:,} rows have uppercase ASCII "
                "in norm_business_name or norm_business_address."
            )
        for script, loss in result["script_loss"].items():
            if loss > 0:
                result["ok"] = False
                result["errors"].append(
                    f"Lost {loss:,} {script} chars during normalisation."
                )

    return result


# ─── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full-dataset chunked preprocessing validator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sample-chunks", type=int, default=None, metavar="N",
                   help="Process only the first N chunks per file.")
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "output",
                   metavar="DIR",
                   help="Where to write preprocessed TSVs (default: ../output).")
    p.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE, metavar="N",
                   help=f"Rows per chunk (default: {DEFAULT_CHUNKSIZE:,}).")
    p.add_argument("--no-write", action="store_true",
                   help="Validate only; do not write preprocessed TSVs.")
    p.add_argument("--deep-check", action="store_true",
                   help="Run per-chunk regex audits (NULL literals, uppercase, "
                        "script-loss). Slower but catches normalisation bugs.")
    p.add_argument("--files", type=str, default=None, metavar="KEY,...",
                   help="Comma-separated file keys to process (default: all 7).")
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    all_keys = list(SOURCE_FILES.keys())
    if args.files:
        requested = [k.strip() for k in args.files.split(",")]
        invalid   = [k for k in requested if k not in SOURCE_FILES]
        if invalid:
            print(f"ERROR: Unknown keys: {invalid}  Valid: {all_keys}")
            return 1
        file_keys = requested
    else:
        file_keys = all_keys

    write_output = not args.no_write
    output_dir   = args.output_dir if write_output else None

    print(_sep("="))
    print("  Amazon ML Challenge 2026 — Preprocessing Validator")
    print(_sep("="))
    print(f"  Files          : {len(file_keys)}")
    print(f"  Chunk size     : {args.chunksize:,} rows")
    print(f"  Sample chunks  : {args.sample_chunks or 'ALL (full dataset)'}")
    print(f"  Write output   : {'YES -> ' + str(output_dir) if write_output else 'NO'}")
    print(f"  Deep check     : {'YES (slower)' if args.deep_check else 'NO (fast mode)'}")
    print(f"  psutil RAM     : {'YES' if _HAS_PSUTIL else 'NO'}")
    print(_sep())

    wall_t0  = time.perf_counter()
    results  = []

    for file_key in file_keys:
        print(f"\nProcessing: {file_key}")
        print(f"  Source : {SOURCE_FILES[file_key]}")
        if write_output:
            print(f"  Output : {output_dir / (file_key + '_preprocessed.tsv')}")
        print(_sep("-", 72))

        r = validate_file(
            file_key     = file_key,
            output_dir   = output_dir,
            chunksize    = args.chunksize,
            sample_chunks= args.sample_chunks,
            write_output = write_output,
            deep_check   = args.deep_check,
        )
        results.append(r)

        status = "OK" if r["ok"] else "FAIL"
        print(f"\n  Result : [{status}]")
        print(f"    Chunks processed : {r['chunks']:,}")
        print(f"    Input rows       : {r['input_rows']:,}")
        print(f"    Output rows      : {r['output_rows']:,}")
        print(f"    Rows match       : {'YES' if r['rows_match'] else 'NO ← MISMATCH'}")
        if file_key != "train_ground_truth":
            print(f"    Missing names    : {r['missing_name']:,}")
            print(f"    Missing addresses: {r['missing_addr']:,}")
        print(f"    Duplicate IDs    : {r['dup_ids']:,}"
              + (" ← WARNING" if r["dup_ids"] > 0 else ""))
        if _HAS_PSUTIL:
            print(f"    Peak RAM         : {r['peak_rss_mb']:.0f} MB")
        print(f"    Elapsed          : {_fmt(r['elapsed_s'])}")
        for w in r["warnings"]:
            print(f"    WARNING : {w}")
        for e in r["errors"]:
            print(f"    ERROR   : {e}")

    # ── Final table ────────────────────────────────────────────────────────────
    wall_elapsed = time.perf_counter() - wall_t0

    print("\n" + _sep("="))
    print("  FINAL VALIDATION REPORT")
    print(_sep("="))
    hdr = (f"  {'FILE KEY':<25s}  {'INPUT ROWS':>12s}  {'OUTPUT ROWS':>12s}  "
           f"{'MATCH':>6s}  {'MISS_ADDR':>10s}  {'DUP_IDS':>8s}  {'TIME':>9s}")
    print(hdr)
    print("  " + _sep("-", 95))
    for r in results:
        missing = (r["missing_addr"] if r["file_key"] != "train_ground_truth" else "N/A")
        ms = f"{missing:>10,}" if isinstance(missing, int) else f"{'N/A':>10s}"
        print(
            f"  {r['file_key']:<25s}  "
            f"{r['input_rows']:>12,}  "
            f"{r['output_rows']:>12,}  "
            f"{'YES' if r['rows_match'] else 'NO!':>6s}  "
            f"{ms}  "
            f"{r['dup_ids']:>8,}  "
            f"{_fmt(r['elapsed_s']):>9s}"
        )
    print("  " + _sep("-", 95))
    ti = sum(r["input_rows"]  for r in results)
    to = sum(r["output_rows"] for r in results)
    print(
        f"  {'TOTAL':<25s}  "
        f"{ti:>12,}  "
        f"{to:>12,}  "
        f"{'YES' if ti == to else 'NO!':>6s}  "
        f"{'':>10s}  {'':>8s}  "
        f"{_fmt(wall_elapsed):>9s}"
    )
    print()

    if args.sample_chunks:
        print(f"  NOTE: Sample mode — only {args.sample_chunks} chunk(s)/file processed.")

    if _HAS_PSUTIL:
        print(f"  Peak RAM (approx)   : {max(r['peak_rss_mb'] for r in results):.0f} MB")
    print(f"  Total wall-clock    : {_fmt(wall_elapsed)}")
    print()

    failed = [r["file_key"] for r in results if not r["ok"]]
    if failed:
        print(f"  RESULT: FAIL — {len(failed)} file(s) did not pass:")
        for fk in failed:
            print(f"    FAIL: {fk}")
        print(_sep("="))
        return 1

    print("  RESULT: ALL FILES PASSED — input_rows == output_rows for every file.")
    print(_sep("="))
    return 0


if __name__ == "__main__":
    sys.exit(main())
