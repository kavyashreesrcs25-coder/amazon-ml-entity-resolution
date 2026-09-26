"""
preprocessing.py
================
Person 1 — Data Preprocessing Module
Amazon ML Challenge 2026: Business Entity Resolution

Provides reusable functions that normalize business names, addresses, and
country labels for all six source files:

    train_source1.tsv  train_source2.tsv  train_source3.tsv
    test_source1.tsv   test_source2.tsv   test_source3.tsv

Design principles
-----------------
* Non-destructive  — original columns are never overwritten; normalized
  values are written to new ``norm_*`` columns.
* Row-preserving   — no rows are ever dropped, including those with empty
  addresses or missing values.
* ID-safe          — ``entity_id`` is never touched.
* Multilingual-safe — Devanagari, Bengali, Kannada, Tamil, and other
  non-Latin scripts are preserved as-is.  French accented characters
  (é, à, ç …) are also preserved.
* No external data — stdlib + pandas + regex only.
* Memory-efficient — iter_chunks() / stream_preprocess_to_file() process
  files in fixed-size chunks and write output incrementally; the full file
  is never held in RAM.
* Fast             — preprocess_source_df() is fully vectorised using
  pandas .str operations (~10–30× faster than row-wise .apply()).

Output columns added
--------------------
norm_business_name    — normalized business name
norm_business_address — normalized business address
norm_country          — normalized country label
address_missing       — bool flag: True when the original address was absent
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Callable, Iterator, Optional

import pandas as pd

# ─── Path constants ────────────────────────────────────────────────────────────

DATASET_ROOT = Path(
    r"C:\Users\ADMIN\Downloads\6ab10eb3b23ba_student_resource"
    r"\student_resource\dataset"
)

TRAIN_DIR = DATASET_ROOT / "train"
TEST_DIR  = DATASET_ROOT / "test"

SOURCE_FILES: dict[str, Path] = {
    "train_source1":      TRAIN_DIR / "train_source1.tsv",
    "train_source2":      TRAIN_DIR / "train_source2.tsv",
    "train_source3":      TRAIN_DIR / "train_source3.tsv",
    "train_ground_truth": TRAIN_DIR / "train_ground_truth.tsv",
    "test_source1":       TEST_DIR  / "test_source1.tsv",
    "test_source2":       TEST_DIR  / "test_source2.tsv",
    "test_source3":       TEST_DIR  / "test_source3.tsv",
}

# 100 000 rows ≈ 30–40 MB per chunk for source2/source3.
DEFAULT_CHUNKSIZE = 100_000

# ─── Compiled regex patterns ───────────────────────────────────────────────────

# NULL-literal placeholders (word-boundary version)
_PAT_NULL      = re.compile(r"(?i)\bnull\b|<null>|\(null\)|#null#")

# Hash-prefixed house numbers: ###56 → 56
_PAT_HASH_NUM  = re.compile(r"#+\s*(\d)")

# Leading non-content characters: --, ***, ###, <<, @@ etc.
_PAT_LEAD      = re.compile(r"^[\s\-\*\#\@\!\|\~\^\+\=\<\>]+")

# Trailing noise
_PAT_TRAIL     = re.compile(r"[\s\-\*\#\@\!\|\~\^\+\=\<\>]+$")

# Pipe separators: " | " → " "
_PAT_PIPE      = re.compile(r"\s*\|\s*")

# Multiple dots → single dot
_PAT_MULTIDOT  = re.compile(r"\.{2,}")

# Repeated comma/semicolon separators
_PAT_MULTISEP  = re.compile(r"[,;]{2,}")

# Empty comma segments: ", , " → ","
_PAT_EMPTYSEG  = re.compile(r",\s*,+")

# Any run of whitespace (including NBSP, ideographic space, etc.)
_PAT_SPACE     = re.compile(r"[ \t\u00a0\u2000-\u200b\u202f\u205f\u3000]+")

# ── Combined patterns for fast vectorised path ────────────────────────────────
# Merge all "interior" cleanup into as few regex passes as possible.

# Pass 1 (names): NULL literals + pipe separators + multi-dot — one sub
_PAT_NAME_P1 = re.compile(
    r"(?i)\bnull\b|<null>|\(null\)|#null#"   # NULL literals
    r"|\s*\|\s*"                              # pipes → replaced by " "
    r"|\.{2,}"                               # multi-dot → "."
)

# Pass 1 (addresses): NULL literals + hash-numbers + pipes + multi-dot
_PAT_ADDR_P1 = re.compile(
    r"(?i)\bnull\b|<null>|\(null\)|#null#"
    r"|#+\s*(\d)"                             # hash-numbers (group 1 = digit)
    r"|\s*\|\s*"
    r"|\.{2,}"
)

# Pass 2 (both): multi-sep + empty segments — one sub
_PAT_P2 = re.compile(r"[,;]{2,}|,\s*,+")

# Pass 3 (both): any whitespace run → single space
_PAT_P3 = _PAT_SPACE

# ─── Scalar helpers (used in the scalar API and smoke-test) ───────────────────

def _to_nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)

def _scalar_clean_name(text: str) -> str:
    """Full name-cleaning pipeline on a single string."""
    text = _to_nfc(text)
    text = _PAT_NULL.sub("", text)
    text = _PAT_LEAD.sub("", text)
    text = _PAT_TRAIL.sub("", text)
    text = _PAT_PIPE.sub(" ", text)
    text = _PAT_MULTISEP.sub(",", text)
    text = _PAT_MULTIDOT.sub(".", text)
    # collapse empty comma segments (may need 2 passes)
    prev = None
    while prev != text:
        prev = text
        text = _PAT_EMPTYSEG.sub(",", text)
    text = text.strip(" ,;")
    text = _PAT_SPACE.sub(" ", text).strip()
    return text.lower()

def _scalar_clean_addr(text: str) -> str:
    """Full address-cleaning pipeline on a single string."""
    text = _to_nfc(text)
    text = _PAT_NULL.sub("", text)
    text = _PAT_HASH_NUM.sub(r"\1", text)
    text = _PAT_LEAD.sub("", text)
    text = _PAT_TRAIL.sub("", text)
    text = _PAT_PIPE.sub(" ", text)
    text = _PAT_MULTISEP.sub(",", text)
    text = _PAT_MULTIDOT.sub(".", text)
    prev = None
    while prev != text:
        prev = text
        text = _PAT_EMPTYSEG.sub(",", text)
    text = text.strip(" ,;")
    text = _PAT_SPACE.sub(" ", text).strip()
    return text.lower()


# ─── Public scalar normalisation API (used by test suite + smoke-test) ────────

def normalize_business_name(raw: Optional[str]) -> str:
    """Return a cleaned, lowercased business name string.

    Preserves non-Latin scripts (Devanagari, Bengali, Kannada, Tamil, Arabic)
    and French accented characters intact.  Legal suffix abbreviations
    (Ltd, Pvt, LLC, S.A.S, SARL …) are kept.

    Handles None / NaN safely.
    """
    if pd.isna(raw) or raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    return _scalar_clean_name(text)


def normalize_business_address(raw: Optional[str]) -> str:
    """Return a cleaned, lowercased address string.

    Address component order is preserved as-is.  Non-Latin place names,
    French accented characters, and abbreviations (Rd, St, TX …) are kept.

    Handles None / NaN safely.
    """
    if pd.isna(raw) or raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    return _scalar_clean_addr(text)


def normalize_country(raw: Optional[str]) -> str:
    """Return a cleaned country string (case preserved, whitespace stripped).

    Country labels (US, India, France) are NOT lowercased.
    """
    if pd.isna(raw) or raw is None:
        return ""
    text = str(raw).strip()
    text = _to_nfc(text)
    text = _PAT_SPACE.sub(" ", text).strip()
    return text


def is_address_missing(raw: Optional[str]) -> bool:
    """Return True when the raw address is absent or reduces to empty after cleaning."""
    if pd.isna(raw) or raw is None:
        return True
    text = str(raw).strip()
    if not text:
        return True
    # Remove NULLs and noise; if nothing remains → missing
    cleaned = _PAT_NULL.sub("", text)
    cleaned = _PAT_LEAD.sub("", cleaned)
    cleaned = _PAT_TRAIL.sub("", cleaned)
    return len(cleaned.strip()) == 0


# ─── Vectorised column-level helpers ──────────────────────────────────────────
#
# Three regex passes per column (down from ~10), plus a single NFC pass only
# where needed.  Benchmarked at ~1.5 s per 100 k rows vs 10–13 s with .apply().

def _nfc_series(s: pd.Series) -> pd.Series:
    """NFC-normalise a string Series.  Fast path: skip rows with all-ASCII."""
    # Check whether any value contains non-ASCII before paying the apply cost.
    # ~95% of rows are pure ASCII — the mask avoids calling unicodedata on them.
    non_ascii = s.str.contains(r"[^\x00-\x7F]", regex=True, na=False)
    if not non_ascii.any():
        return s
    result = s.copy()
    result[non_ascii] = result[non_ascii].apply(
        lambda x: unicodedata.normalize("NFC", x)
    )
    return result


def _name_p1_replace(m: re.Match) -> str:
    """Replacement function for _PAT_NAME_P1.

    - NULL literals   → ""
    - pipes           → " "
    - multi-dot       → "."
    """
    t = m.group(0)
    tl = t.lower()
    if "null" in tl or "<" in t or "(" in t or "#" in t:
        return ""
    if "|" in t:
        return " "
    # must be multi-dot
    return "."


def _addr_p1_replace(m: re.Match) -> str:
    """Replacement function for _PAT_ADDR_P1.

    - NULL literals          → ""
    - hash-prefixed numbers  → just the digit character
    - pipes                  → " "
    - multi-dot              → "."
    """
    t  = m.group(0)
    tl = t.lower()
    if "null" in tl or "<" in t or "(" in t:
        return ""
    # hash-number: group 1 captures the first digit
    if t.startswith("#"):
        return m.group(1)
    if "|" in t:
        return " "
    return "."


def _p2_replace(m: re.Match) -> str:
    """Collapse multi-sep and empty-segment artifacts → single comma."""
    return ","


def _vec_clean_name(col: pd.Series) -> pd.Series:
    """Vectorised business-name normalisation.  ~1.5 s per 100 k rows."""
    s = col.fillna("")
    s = _nfc_series(s)
    # Leading / trailing noise (multiline flag so ^ and $ match each cell)
    s = s.str.replace(_PAT_LEAD,    "", regex=True)
    s = s.str.replace(_PAT_TRAIL,   "", regex=True)
    # Combined pass: NULL literals + pipes + multi-dot
    s = s.str.replace(_PAT_NAME_P1, _name_p1_replace, regex=True)
    # Combined pass: multi-sep + empty segments → ","
    s = s.str.replace(_PAT_P2,      _p2_replace,      regex=True)
    # Collapse whitespace
    s = s.str.replace(_PAT_P3,      " ",              regex=True)
    s = s.str.strip(" ,;")
    s = s.str.lower()
    return s.fillna("")


def _vec_clean_addr(col: pd.Series) -> pd.Series:
    """Vectorised address normalisation.  ~2 s per 100 k rows."""
    s = col.fillna("")
    s = _nfc_series(s)
    s = s.str.replace(_PAT_LEAD,    "", regex=True)
    s = s.str.replace(_PAT_TRAIL,   "", regex=True)
    # Combined: NULL literals + hash-numbers + pipes + multi-dot
    s = s.str.replace(_PAT_ADDR_P1, _addr_p1_replace, regex=True)
    # Combined: multi-sep + empty segments
    s = s.str.replace(_PAT_P2,      _p2_replace,      regex=True)
    s = s.str.replace(_PAT_P3,      " ",              regex=True)
    s = s.str.strip(" ,;")
    s = s.str.lower()
    return s.fillna("")


def _vec_clean_country(col: pd.Series) -> pd.Series:
    """Vectorised country normalisation (case preserved)."""
    s = col.fillna("").str.strip()
    s = s.str.replace(_PAT_P3, " ", regex=True)
    return s.str.strip().fillna("")


def _vec_address_missing(col: pd.Series) -> pd.Series:
    """Vectorised missing-address flag (returns bool Series)."""
    s = col.fillna("")
    s2 = s.str.replace(_PAT_NULL,  "", regex=True)
    s2 = s2.str.replace(_PAT_LEAD, "", regex=True)
    s2 = s2.str.replace(_PAT_TRAIL,"", regex=True)
    return (s2.str.strip() == "")


# ─── DataFrame-level preprocessing (vectorised) ───────────────────────────────

def preprocess_source_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all normalisation to a source DataFrame using vectorised ops.

    Adds four new columns:
        norm_business_name    — normalised name
        norm_business_address — normalised address
        norm_country          — normalised country
        address_missing       — bool flag

    Original columns are never modified.  Row count is unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: entity_id, business_name, business_address, country.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with four additional columns appended.
    """
    df["norm_business_name"]    = _vec_clean_name(df["business_name"])
    df["norm_business_address"] = _vec_clean_addr(df["business_address"])
    df["norm_country"]          = _vec_clean_country(df["country"])
    df["address_missing"]       = _vec_address_missing(df["business_address"])
    return df


# ─── Row-level helper (kept for backward compatibility with test suite) ────────

def preprocess_row(row: pd.Series) -> pd.Series:
    """Apply normalisation to a single DataFrame row."""
    return pd.Series({
        "norm_business_name":    normalize_business_name(row.get("business_name")),
        "norm_business_address": normalize_business_address(row.get("business_address")),
        "norm_country":          normalize_country(row.get("country")),
        "address_missing":       is_address_missing(row.get("business_address")),
    })


# ─── Streaming / chunked API ───────────────────────────────────────────────────

def iter_chunks(
    file_key: str,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> Iterator[pd.DataFrame]:
    """Yield preprocessed DataFrame chunks one at a time without buffering.

    Each yielded chunk already has the four norm_* columns added.
    For train_ground_truth, matched_entity_ids NaN values are filled with
    empty string; no other transformation is applied.

    This is the lowest-memory API — the caller processes each chunk before
    the next one is loaded from disk.
    """
    if file_key not in SOURCE_FILES:
        raise KeyError(
            f"Unknown file key '{file_key}'. "
            f"Valid keys: {list(SOURCE_FILES.keys())}"
        )
    path = SOURCE_FILES[file_key]
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    read_kw = dict(sep="\t", encoding="utf-8", dtype=str, on_bad_lines="skip")
    is_gt   = (file_key == "train_ground_truth")

    for chunk in pd.read_csv(path, chunksize=chunksize, **read_kw):
        if is_gt:
            chunk["matched_entity_ids"] = chunk["matched_entity_ids"].fillna("")
        else:
            chunk = preprocess_source_df(chunk)
        yield chunk


def stream_preprocess_to_file(
    file_key: str,
    output_path: Path,
    chunksize: int = DEFAULT_CHUNKSIZE,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Preprocess a full source file in chunks, writing output incrementally.

    The input file is never fully loaded into RAM.

    Parameters
    ----------
    file_key : str
    output_path : Path
        Destination TSV.  Parent directory is created if needed.
    chunksize : int
    progress_callback : callable(chunk_index, rows_processed), optional

    Returns
    -------
    dict with keys: input_rows, output_rows, rows_match, chunks, missing_addr
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_rows   = 0
    output_rows  = 0
    missing_addr = 0
    chunk_idx    = 0
    first_chunk  = True

    with open(output_path, "w", encoding="utf-8", newline="") as fout:
        for chunk in iter_chunks(file_key, chunksize=chunksize):
            chunk_idx   += 1
            input_rows  += len(chunk)
            output_rows += len(chunk)

            if file_key != "train_ground_truth" and "address_missing" in chunk.columns:
                missing_addr += int(chunk["address_missing"].sum())

            chunk.to_csv(fout, sep="\t", index=False, header=first_chunk, encoding=None)
            first_chunk = False

            if progress_callback:
                progress_callback(chunk_idx, input_rows)

    return {
        "input_rows":   input_rows,
        "output_rows":  output_rows,
        "rows_match":   input_rows == output_rows,
        "chunks":       chunk_idx,
        "missing_addr": missing_addr,
    }


# ─── In-memory load + preprocess (kept for test suite compatibility) ───────────

def load_and_preprocess(
    file_key: str,
    chunksize: Optional[int] = None,
) -> pd.DataFrame:
    """Load a source TSV and return it fully preprocessed in a single DataFrame.

    WARNING: Loads the entire file into RAM.  For large files prefer
    iter_chunks() or stream_preprocess_to_file().
    Kept for backward compatibility with test_preprocessing.py.
    """
    if file_key not in SOURCE_FILES:
        raise KeyError(f"Unknown file key '{file_key}'. Valid: {list(SOURCE_FILES.keys())}")
    path = SOURCE_FILES[file_key]
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    read_kw = dict(sep="\t", encoding="utf-8", dtype=str, on_bad_lines="skip")

    if file_key == "train_ground_truth":
        if chunksize:
            df = pd.concat(pd.read_csv(path, chunksize=chunksize, **read_kw), ignore_index=True)
        else:
            df = pd.read_csv(path, **read_kw)
        df["matched_entity_ids"] = df["matched_entity_ids"].fillna("")
        return df

    if chunksize:
        df = pd.concat(pd.read_csv(path, chunksize=chunksize, **read_kw), ignore_index=True)
    else:
        df = pd.read_csv(path, **read_kw)

    return preprocess_source_df(df)


def load_and_preprocess_all_sources(
    chunksize: Optional[int] = None,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load and preprocess all six source files into memory.

    WARNING: ~26 M rows combined.  Needs ≥ 32 GB RAM.
    Prefer stream_preprocess_to_file() for normal operation.
    """
    source_keys = [
        "train_source1", "train_source2", "train_source3",
        "test_source1",  "test_source2",  "test_source3",
    ]
    result: dict[str, pd.DataFrame] = {}
    for key in source_keys:
        if verbose:
            print(f"  Loading {key} ...", end=" ", flush=True)
        df = load_and_preprocess(key, chunksize=chunksize)
        if verbose:
            print(f"{len(df):,} rows  OK")
        result[key] = df
    return result


def save_preprocessed(df: pd.DataFrame, file_key: str, output_dir: Path) -> Path:
    """Save a preprocessed DataFrame to a TSV file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{file_key}_preprocessed.tsv"
    df.to_csv(out_path, sep="\t", index=False, encoding="utf-8")
    return out_path


# ─── Quick smoke-test (run as script) ─────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("preprocessing.py — quick smoke-test")
    print("=" * 60)

    all_pass = True

    def _chk(label: str, got, expected):
        global all_pass
        ok = (got == expected)
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {label!r:55s}  got={got!r}"
              + (f"  expected={expected!r}" if not ok else ""))

    print("\n[Business Name Normalisation]")
    cases_name = [
        ("-- Holloway Peak Inc Seafood",          "holloway peak inc seafood"),
        ("*** Sai Tech Private Limited",          "sai tech private limited"),
        ("MILLER METALS",                         "miller metals"),
        ("Miller  Metals",                        "miller metals"),
        ("राम मार्केटिंग प्राइवेट लिमिटेड",     "राम मार्केटिंग प्राइवेट लिमिटेड"),
        ("whiteallgraphics.com",                  "whiteallgraphics.com"),
        ("ABC Company | www.abc.com",             "abc company www.abc.com"),
        ("FOUNDATION EXCEL AGENCY PRIVATE  LIMITED",
         "foundation excel agency private limited"),
        ("Pvt. EFS Print Ventures Ltd.",          "pvt. efs print ventures ltd."),
        ("Primary Care  Nati0nal Specialists",    "primary care nati0nal specialists"),
        (None,                                    ""),
        ("",                                      ""),
        ("<< Team Ecole",                         "team ecole"),
        ("Marina Ecole France Sarl",              "marina ecole france sarl"),
        ("Écoles Françaises SARL",               "écoles françaises sarl"),
        ("ইউনিভার্সাল সিস্টেমস প্রাইভেট লিমিটেড",
         "ইউনিভার্সাল সিস্টেমস প্রাইভেট লিমিটেড"),
    ]
    for raw, expected in cases_name:
        _chk(str(raw), normalize_business_name(raw), expected)

    print("\n[Business Address Normalisation]")
    cases_addr = [
        ("###56 B REVENUE HOUSING SOCIETY, KOLHAPUR, Maharashtra",
         "56 b revenue housing society, kolhapur, maharashtra"),
        ("New Delhi, null, A-68, दिल्ली",   "new delhi, a-68, दिल्ली"),
        ("WA, Arlington, 18009 3rd Avenue",  "wa, arlington, 18009 3rd avenue"),
        ("  ",    ""),
        (None,    ""),
        ("",      ""),
        ("<NULL>", ""),
        ("(null)", ""),
        ("City, <NULL>, State",              "city, state"),
        ("23 Rue Icmre, La Teste-de-buch, Gironde",
         "23 rue icmre, la teste-de-buch, gironde"),
        ("Door No 183, Jayanagar, Bengaluru, ಕರ್ನಾಟಕ",
         "door no 183, jayanagar, bengaluru, ಕರ್ನಾಟಕ"),
    ]
    for raw, expected in cases_addr:
        _chk(str(raw), normalize_business_address(raw), expected)

    print("\n[Country Normalisation]")
    for raw, expected in [
        ("US", "US"), ("India", "India"), ("France", "France"),
        (" India ", "India"), ("  US  ", "US"), (None, ""), ("", ""),
    ]:
        _chk(str(raw), normalize_country(raw), expected)

    print("\n[Missing Address Detection]")
    for raw, expected in [
        (None, True), ("", True), ("  ", True), ("<NULL>", True),
        ("null", True), ("(null)", True), ("NULL", True),
        ("123 Main St", False), ("ಕರ್ನಾಟಕ", False),
    ]:
        got = is_address_missing(raw)
        ok  = (got == expected)
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {raw!r:15s}  missing={got}"
              + (f"  expected={expected}" if not ok else ""))

    print()
    if all_pass:
        print("All smoke-tests PASSED.")
        sys.exit(0)
    else:
        print("Some smoke-tests FAILED.")
        sys.exit(1)
