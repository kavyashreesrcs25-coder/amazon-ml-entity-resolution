"""
test_preprocessing.py
=====================
Person 1 — Preprocessing Verification Suite
Amazon ML Challenge 2026: Business Entity Resolution

Runs two layers of tests:

Layer 1 — Unit tests
    Pure-function tests against hand-crafted inputs covering every noise
    pattern observed in the dataset inspection.  No files are read.

Layer 2 — Integration tests
    Loads each of the six source TSVs (plus ground truth) from the real
    dataset, applies preprocessing, and verifies structural invariants:
      - row count unchanged
      - entity_id column unchanged
      - original columns still present
      - normalized columns created with correct names
      - no normalized column accidentally deletes non-Latin text
      - missing-address flag is consistent with normalised address content
      - address_missing rates match expected dataset ranges

Usage
-----
    python src/test_preprocessing.py

Exit code 0  → all tests passed
Exit code 1  → one or more tests failed (failures printed inline)

No third-party test framework required — stdlib only.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Callable

import pandas as pd

# Allow running from project root OR from src/
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from preprocessing import (
    TRAIN_DIR,
    TEST_DIR,
    normalize_business_name,
    normalize_business_address,
    normalize_country,
    is_address_missing,
    preprocess_source_df,
    load_and_preprocess,
)

# ─── Minimal test harness ──────────────────────────────────────────────────────

_PASS = 0
_FAIL = 0
_ERRORS: list[str] = []


def _check(name: str, condition: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"    PASS  {name}")
    else:
        _FAIL += 1
        msg = f"    FAIL  {name}" + (f"\n          {detail}" if detail else "")
        print(msg)
        _ERRORS.append(name + (f": {detail}" if detail else ""))


def _section(title: str) -> None:
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")


def _run(label: str, fn: Callable) -> None:
    """Run a test function, catching unexpected exceptions."""
    global _FAIL
    try:
        fn()
    except Exception:
        _FAIL += 1
        msg = f"    ERROR  {label} raised an unexpected exception"
        print(msg)
        traceback.print_exc()
        _ERRORS.append(label)


# ─── Layer 1: Unit tests ───────────────────────────────────────────────────────


def test_name_noise_prefix_removal():
    """Leading noise characters are stripped from business names."""
    cases = [
        ("-- Holloway Peak Inc Seafood",  "holloway peak inc seafood"),
        ("*** Sai Tech Private Limited",  "sai tech private limited"),
        ("### Corp",                      "corp"),
        ("<< Team Ecole",                 "team ecole"),
        ("@@ Bright Solutions",          "bright solutions"),
    ]
    for raw, expected in cases:
        got = normalize_business_name(raw)
        _check(
            f"name_noise_prefix: {raw!r}",
            got == expected,
            f"expected {expected!r}, got {got!r}",
        )


def test_name_whitespace_collapse():
    """Multiple internal spaces are collapsed to one."""
    cases = [
        ("Miller  Metals",                    "miller metals"),
        ("FOUNDATION EXCEL AGENCY PRIVATE  LIMITED",
         "foundation excel agency private limited"),
        ("A   B   C",                         "a b c"),
        ("  leading and trailing  ",          "leading and trailing"),
    ]
    for raw, expected in cases:
        got = normalize_business_name(raw)
        _check(
            f"name_whitespace: {raw!r}",
            got == expected,
            f"expected {expected!r}, got {got!r}",
        )


def test_name_uppercase_lowercased():
    """ALLCAPS names from Source 2 are lowercased."""
    cases = [
        ("MILLER METALS",        "miller metals"),
        ("COBALT  (LLC)",         "cobalt (llc)"),
        ("NATIONAL BANK OF INDIA", "national bank of india"),
    ]
    for raw, expected in cases:
        got = normalize_business_name(raw)
        _check(
            f"name_uppercase: {raw!r}",
            got == expected,
            f"expected {expected!r}, got {got!r}",
        )


def test_name_pipe_separator():
    """Pipe characters used as separators are replaced by a space."""
    cases = [
        ("SHIVSHAKTI | www.shivshakti.com", "shivshakti www.shivshakti.com"),
        ("ABC Company | xyz.com",            "abc company xyz.com"),
    ]
    for raw, expected in cases:
        got = normalize_business_name(raw)
        _check(
            f"name_pipe: {raw!r}",
            got == expected,
            f"expected {expected!r}, got {got!r}",
        )


def test_name_devanagari_preserved():
    """Devanagari script is preserved intact after normalisation."""
    cases = [
        "राम मार्केटिंग प्राइवेट लिमिटेड",
        "आदित्य प्रॉपर्टीज एलएलपी",
        "मॉडर्न फाइनेंस",
    ]
    for raw in cases:
        got = normalize_business_name(raw)
        # Every Devanagari character in the original must still be present
        original_devanagari = [c for c in raw if "\u0900" <= c <= "\u097F"]
        remaining_devanagari = [c for c in got if "\u0900" <= c <= "\u097F"]
        _check(
            f"name_devanagari_preserved: {raw!r}",
            len(original_devanagari) == len(remaining_devanagari),
            f"lost {len(original_devanagari)-len(remaining_devanagari)} Devanagari chars",
        )


def test_name_bengali_preserved():
    """Bengali script is preserved intact."""
    raw = "ইউনিভার্সাল সিস্টেমস প্রাইভেট লিমিটেড"
    got = normalize_business_name(raw)
    original_bengali = [c for c in raw if "\u0980" <= c <= "\u09FF"]
    remaining_bengali = [c for c in got if "\u0980" <= c <= "\u09FF"]
    _check(
        "name_bengali_preserved",
        len(original_bengali) == len(remaining_bengali),
        f"lost {len(original_bengali)-len(remaining_bengali)} Bengali chars",
    )


def test_name_kannada_preserved():
    """Kannada script is preserved intact."""
    raw = "ಕರ್ನಾಟಕ ಮಾರ್ಕೆಟಿಂಗ್"
    got = normalize_business_name(raw)
    original = [c for c in raw if "\u0C80" <= c <= "\u0CFF"]
    remaining = [c for c in got if "\u0C80" <= c <= "\u0CFF"]
    _check(
        "name_kannada_preserved",
        len(original) == len(remaining),
        f"lost {len(original)-len(remaining)} Kannada chars",
    )


def test_name_french_accents_preserved():
    """French accented characters are preserved."""
    cases = [
        ("Écoles Françaises SARL",   "écoles françaises sarl"),
        ("Société Générale",          "société générale"),
        ("Marina École France Sarl",  "marina école france sarl"),
        ("Fractales Amis Groupe S.A.S", "fractales amis groupe s.a.s"),
    ]
    for raw, expected in cases:
        got = normalize_business_name(raw)
        _check(
            f"name_french_accents: {raw!r}",
            got == expected,
            f"expected {expected!r}, got {got!r}",
        )


def test_name_null_and_empty():
    """None and empty string inputs return empty string."""
    for raw in [None, "", "   "]:
        got = normalize_business_name(raw)
        _check(
            f"name_null_empty: {raw!r}",
            got == "",
            f"expected '', got {got!r}",
        )


def test_name_legal_suffixes_kept():
    """Legal entity suffixes are preserved — they discriminate entities."""
    cases = [
        "pvt. efs print ventures ltd.",
        "primary care national specialists l.l.c.",
        "marina ecole france sarl",
        "znb club sarl",
        "fractales amis groupe s.a.s",
    ]
    for expected_fragment in cases:
        got = normalize_business_name(expected_fragment)
        _check(
            f"name_legal_suffix_kept: {expected_fragment!r}",
            got == expected_fragment,
            f"got {got!r}",
        )


def test_name_entity_id_not_touched():
    """entity_id-style strings are not mangled."""
    # These should pass through lowercased, untouched otherwise
    raw = "S1-925783039"
    got = normalize_business_name(raw)
    _check(
        "name_entity_id_passthrough",
        got == "s1-925783039",
        f"got {got!r}",
    )


# ── Address unit tests ──────────────────────────────────────────────────────


def test_addr_hash_number_cleanup():
    """Hash prefixes before house numbers are removed."""
    cases = [
        ("###56 B REVENUE HOUSING SOCIETY, KOLHAPUR, Maharashtra",
         "56 b revenue housing society, kolhapur, maharashtra"),
        ("#18009 THIRD AVE, ARLINGTON, WA",
         "18009 third ave, arlington, wa"),
        ("##100 MAIN ST",   "100 main st"),
    ]
    for raw, expected in cases:
        got = normalize_business_address(raw)
        _check(
            f"addr_hash_number: {raw!r}",
            got == expected,
            f"expected {expected!r}, got {got!r}",
        )


def test_addr_null_literal_removal():
    """Literal 'null' / '<NULL>' / '(null)' placeholders are removed."""
    cases = [
        ("New Delhi, null, A-68, दिल्ली",   "new delhi, a-68, दिल्ली"),
        ("G.T. Karnal Road, null, A-68, दिल्ली", "g.t. karnal road, a-68, दिल्ली"),
        ("<NULL>",                             ""),
        ("(null)",                             ""),
        ("City, <NULL>, State",               "city, state"),
    ]
    for raw, expected in cases:
        got = normalize_business_address(raw)
        _check(
            f"addr_null_literal: {raw!r}",
            got == expected,
            f"expected {expected!r}, got {got!r}",
        )


def test_addr_component_reorder_preserved():
    """Address component order is preserved as-is (not forced to a standard form)."""
    cases = [
        ("WA, Arlington, 18009 3rd Avenue",        "wa, arlington, 18009 3rd avenue"),
        ("ME, EVERETT ROAD, POLAND",                "me, everett road, poland"),
        ("GREENSBORO, NC, 19 1/2 STARDUST TRAIL",  "greensboro, nc, 19 1/2 stardust trail"),
    ]
    for raw, expected in cases:
        got = normalize_business_address(raw)
        _check(
            f"addr_reorder_preserved: {raw!r}",
            got == expected,
            f"expected {expected!r}, got {got!r}",
        )


def test_addr_non_latin_preserved():
    """Non-Latin characters in addresses are preserved."""
    cases = [
        ("Door No 183, Jayanagar, Bengaluru, ಕರ್ನಾಟಕ",
         "door no 183, jayanagar, bengaluru, ಕರ್ನಾಟಕ"),
        ("H.no 910 A 3503, Mumbai, महाराष्ट्र",
         "h.no 910 a 3503, mumbai, महाराष्ट्र"),
        ("No 10 Enkay Square, Gurugram, HR",
         "no 10 enkay square, gurugram, hr"),
    ]
    for raw, expected in cases:
        got = normalize_business_address(raw)
        _check(
            f"addr_non_latin: {repr(raw)[:40]}",
            got == expected,
            f"expected {expected!r}, got {got!r}",
        )


def test_addr_french_preserved():
    """French place names with accented characters are preserved."""
    cases = [
        ("23 Rue Icmre, La Teste-de-buch, Gironde",
         "23 rue icmre, la teste-de-buch, gironde"),
        ("175 Boulevard du Président Franklin Roosevelt, Bordeaux",
         "175 boulevard du président franklin roosevelt, bordeaux"),
        ("63 R. DE DIEPPE, LILLE, Hauts-de-France",
         "63 r. de dieppe, lille, hauts-de-france"),
    ]
    for raw, expected in cases:
        got = normalize_business_address(raw)
        _check(
            f"addr_french: {repr(raw)[:45]}",
            got == expected,
            f"expected {expected!r}, got {got!r}",
        )


def test_addr_empty_and_none():
    """Empty / None addresses normalise to empty string."""
    for raw in [None, "", "  ", "<NULL>", "(null)", "null", "NULL"]:
        got = normalize_business_address(raw)
        _check(
            f"addr_empty: {raw!r}",
            got == "",
            f"expected '', got {got!r}",
        )


def test_addr_house_numbers_kept():
    """Numeric house/plot numbers in addresses are not stripped."""
    cases = [
        ("1795 Westchester Drive, High Point, NC",
         "1795 westchester drive, high point, nc"),
        ("2621 Cotten Road, Tyler, TX",
         "2621 cotten road, tyler, tx"),
        ("018009 Third Ave, Arlington, Washington",
         "018009 third ave, arlington, washington"),
    ]
    for raw, expected in cases:
        got = normalize_business_address(raw)
        _check(
            f"addr_house_numbers: {raw!r}",
            got == expected,
            f"expected {expected!r}, got {got!r}",
        )


# ── Country unit tests ──────────────────────────────────────────────────────


def test_country_known_values():
    """Training countries US and India are preserved correctly."""
    _check("country_US",    normalize_country("US")    == "US")
    _check("country_India", normalize_country("India") == "India")


def test_country_france_test_set():
    """France (test-set only) passes through unchanged."""
    _check("country_France", normalize_country("France") == "France")


def test_country_whitespace_stripped():
    """Surrounding whitespace is stripped."""
    cases = [(" US ", "US"), ("  India  ", "India"), ("\tFrance\t", "France")]
    for raw, expected in cases:
        got = normalize_country(raw)
        _check(
            f"country_whitespace: {raw!r}",
            got == expected,
            f"expected {expected!r}, got {got!r}",
        )


def test_country_case_preserved():
    """Country case is NOT lowercased — exact labels are preserved."""
    # 'US' must stay 'US', not become 'us'
    _check("country_case_US",    normalize_country("US")    == "US")
    _check("country_case_India", normalize_country("India") == "India")


def test_country_null_empty():
    """None / empty returns empty string."""
    for raw in [None, "", "   "]:
        got = normalize_country(raw)
        _check(
            f"country_null: {raw!r}",
            got == "",
            f"expected '', got {got!r}",
        )


# ── Missing-address detection ──────────────────────────────────────────────


def test_missing_address_flags():
    """is_address_missing returns correct bool for all edge cases."""
    missing = [None, "", "  ", "<NULL>", "(null)", "null", "NULL", "<null>"]
    present = [
        "123 Main St",
        "ಕರ್ನಾಟಕ",
        "Paris, France",
        "KH NO. -570/13, NEW DELHI",
        "0",            # edge: a lone zero is a valid (if odd) address fragment
    ]
    for raw in missing:
        _check(
            f"missing_flag_true: {raw!r}",
            is_address_missing(raw) is True,
            f"expected True, got False",
        )
    for raw in present:
        _check(
            f"missing_flag_false: {raw!r}",
            is_address_missing(raw) is False,
            f"expected False, got True",
        )


# ─── Layer 2: Integration tests against real dataset files ────────────────────

# Expected row counts from dataset inspection
EXPECTED_ROWS = {
    "train_source1":      2_206_821,
    "train_source2":      5_034_616,
    "train_source3":      5_285_603,
    "train_ground_truth": 2_206_821,
    "test_source1":       1_732_544,
    "test_source2":       4_887_273,
    "test_source3":       5_082_316,
}

SOURCE_KEYS = [
    "train_source1", "train_source2", "train_source3",
    "test_source1",  "test_source2",  "test_source3",
]

# From dataset inspection: source2/3 have ~2.6–3.4% missing addresses
MISSING_ADDR_BOUNDS = {
    "train_source1": (0.0,  0.5),   # source1 is clean
    "train_source2": (2.0,  5.0),
    "train_source3": (2.0,  5.0),
    "test_source1":  (0.0,  0.5),
    "test_source2":  (1.0,  5.0),
    "test_source3":  (1.0,  5.0),
}

# Script Unicode ranges used to detect non-Latin script preservation
SCRIPT_RANGES = {
    "Devanagari": ("\u0900", "\u097F"),
    "Bengali":    ("\u0980", "\u09FF"),
    "Kannada":    ("\u0C80", "\u0CFF"),
    "Tamil":      ("\u0B80", "\u0BFF"),
}


def _count_script_chars(text: str, lo: str, hi: str) -> int:
    return sum(1 for c in text if lo <= c <= hi)


def run_integration_tests_for(file_key: str, sample_size: int = 50_000) -> None:
    """Load a source file, preprocess it, and run all structural checks.

    Uses a sample for speed on the very large files (source2/source3 ~500MB).
    Row-count check still uses the full file count from the known table.
    """
    print(f"\n  [{file_key}]")

    # ── Load ──────────────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        df_full = load_and_preprocess(file_key)
    except FileNotFoundError as e:
        _check(f"{file_key}_file_exists", False, str(e))
        return
    elapsed = time.time() - t0
    print(f"    loaded {len(df_full):,} rows in {elapsed:.1f}s")

    is_gt = (file_key == "train_ground_truth")
    df = df_full  # use full for checks; sample only for content probes

    # ── T1: Row count unchanged ───────────────────────────────────────────────
    expected = EXPECTED_ROWS[file_key]
    _check(
        f"{file_key}_row_count",
        len(df) == expected,
        f"expected {expected:,}, got {len(df):,}",
    )

    # ── T2: entity_id / source1_entity_id unchanged ───────────────────────────
    id_col = "source1_entity_id" if is_gt else "entity_id"
    _check(
        f"{file_key}_id_col_present",
        id_col in df.columns,
        f"column '{id_col}' missing",
    )
    if id_col in df.columns:
        _check(
            f"{file_key}_no_null_ids",
            df[id_col].notna().all(),
            "some entity_ids are NaN",
        )
        _check(
            f"{file_key}_unique_ids",
            df[id_col].nunique() == len(df),
            f"duplicate IDs found: {len(df) - df[id_col].nunique()}",
        )
        # Prefix check
        if not is_gt:
            prefix = file_key.split("_")[0][0].upper() + "1-" if "source1" in file_key else \
                     file_key.split("_")[0][0].upper() + "2-" if "source2" in file_key else \
                     file_key.split("_")[0][0].upper() + "3-"
            # Derive correct prefix from file_key
            src_num = file_key[-1]  # "1", "2", or "3"
            expected_prefix = f"S{src_num}-"
            bad = (~df[id_col].str.startswith(expected_prefix)).sum()
            _check(
                f"{file_key}_id_prefix_{expected_prefix}",
                bad == 0,
                f"{bad:,} IDs do not start with '{expected_prefix}'",
            )

    if is_gt:
        # Ground truth: just verify columns and row count (no norm cols expected)
        _check(
            f"{file_key}_original_cols",
            {"source1_entity_id", "matched_entity_ids"}.issubset(df.columns),
            f"columns: {list(df.columns)}",
        )
        # Singletons: matched_entity_ids empty string (not NaN after preprocessing)
        _check(
            f"{file_key}_no_nan_matched_ids",
            df["matched_entity_ids"].notna().all(),
            "NaN found in matched_entity_ids after fillna",
        )
        return

    # ── T3: Original columns still present ───────────────────────────────────
    original_cols = {"entity_id", "business_name", "business_address", "country"}
    _check(
        f"{file_key}_original_cols_present",
        original_cols.issubset(df.columns),
        f"missing: {original_cols - set(df.columns)}",
    )

    # ── T4: Normalised columns created ────────────────────────────────────────
    norm_cols = {"norm_business_name", "norm_business_address",
                 "norm_country", "address_missing"}
    _check(
        f"{file_key}_norm_cols_created",
        norm_cols.issubset(df.columns),
        f"missing: {norm_cols - set(df.columns)}",
    )

    # ── T5: No NaN in normalised columns ─────────────────────────────────────
    for col in ["norm_business_name", "norm_business_address", "norm_country"]:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            _check(
                f"{file_key}_{col}_no_nan",
                nan_count == 0,
                f"{nan_count:,} NaN values found",
            )

    # ── T6: address_missing dtype is bool ─────────────────────────────────────
    if "address_missing" in df.columns:
        _check(
            f"{file_key}_address_missing_is_bool",
            df["address_missing"].dtype == bool,
            f"dtype is {df['address_missing'].dtype}",
        )

    # ── T7: Missing-address rate within expected bounds ───────────────────────
    if "address_missing" in df.columns:
        lo, hi = MISSING_ADDR_BOUNDS[file_key]
        rate = df["address_missing"].mean() * 100
        _check(
            f"{file_key}_missing_addr_rate_{lo:.0f}pct_to_{hi:.0f}pct",
            lo <= rate <= hi,
            f"rate={rate:.2f}% (expected {lo}–{hi}%)",
        )

    # ── T8: Normalised address is empty iff address_missing is True ───────────
    if {"norm_business_address", "address_missing"}.issubset(df.columns):
        inconsistent = (
            (df["address_missing"]) & (df["norm_business_address"] != "")
            | (~df["address_missing"]) & (df["norm_business_address"] == "")
        ).sum()
        _check(
            f"{file_key}_missing_flag_consistent",
            inconsistent == 0,
            f"{inconsistent:,} rows where flag and norm_address disagree",
        )

    # ── Content probe on a sample (expensive operations) ─────────────────────
    sample = df.sample(n=min(sample_size, len(df)), random_state=42)

    # ── T9: Normalised names are all lowercase (ASCII portion) ────────────────
    def _has_uppercase_ascii(s: str) -> bool:
        return any("A" <= c <= "Z" for c in s)

    upper_count = sample["norm_business_name"].apply(_has_uppercase_ascii).sum()
    _check(
        f"{file_key}_norm_name_lowercase",
        upper_count == 0,
        f"{upper_count:,} rows still have uppercase ASCII in norm_business_name",
    )

    # ── T10: Normalised addresses are all lowercase (ASCII portion) ───────────
    upper_addr = (
        sample["norm_business_address"]
        .apply(_has_uppercase_ascii)
        .sum()
    )
    _check(
        f"{file_key}_norm_addr_lowercase",
        upper_addr == 0,
        f"{upper_addr:,} rows still have uppercase ASCII in norm_business_address",
    )

    # ── T11: Country values unchanged in norm_country ─────────────────────────
    unique_raw    = set(df["country"].dropna().unique())
    unique_normed = set(df["norm_country"].unique()) - {""}
    _check(
        f"{file_key}_norm_country_values_match",
        unique_normed == unique_raw,
        f"raw={sorted(unique_raw)}, normed={sorted(unique_normed)}",
    )

    # ── T12: Non-Latin script characters not accidentally deleted ─────────────
    # For each script, find rows with that script in the RAW name, then verify
    # the SAME number of script characters appear in the normalised name.
    for script, (lo, hi) in SCRIPT_RANGES.items():
        raw_has_script = sample["business_name"].apply(
            lambda x: _count_script_chars(str(x) if pd.notna(x) else "", lo, hi) > 0
        )
        n_rows = raw_has_script.sum()
        if n_rows == 0:
            continue  # script not present in this file's sample
        subset = sample[raw_has_script]
        char_loss = 0
        for _, row in subset.iterrows():
            raw_chars  = _count_script_chars(str(row["business_name"]),    lo, hi)
            norm_chars = _count_script_chars(str(row["norm_business_name"]), lo, hi)
            char_loss += max(0, raw_chars - norm_chars)
        _check(
            f"{file_key}_script_{script}_preserved",
            char_loss == 0,
            f"lost {char_loss} {script} chars across {n_rows} rows",
        )

    # ── T13: Norm columns do not contain literal 'null' placeholders ──────────
    for col in ["norm_business_name", "norm_business_address"]:
        if col not in sample.columns:
            continue
        has_null_literal = sample[col].str.lower().str.contains(
            r"(?<![a-z0-9])(null|<null>|\(null\))(?![a-z0-9])",
            regex=True,
            na=False,
        ).sum()
        _check(
            f"{file_key}_{col}_no_null_literal",
            has_null_literal == 0,
            f"{has_null_literal} rows still contain 'null' literals",
        )

    # ── T14: No leading/trailing whitespace in norm columns ───────────────────
    for col in ["norm_business_name", "norm_business_address", "norm_country"]:
        if col not in sample.columns:
            continue
        leading_trailing = sample[col].apply(
            lambda x: x != x.strip()
        ).sum()
        _check(
            f"{file_key}_{col}_no_edge_whitespace",
            leading_trailing == 0,
            f"{leading_trailing} rows have leading/trailing whitespace",
        )

    # ── T15: No double-spaces in norm columns ─────────────────────────────────
    for col in ["norm_business_name", "norm_business_address"]:
        if col not in sample.columns:
            continue
        double_space = sample[col].str.contains("  ", na=False).sum()
        _check(
            f"{file_key}_{col}_no_double_space",
            double_space == 0,
            f"{double_space} rows still have double spaces",
        )

    # ── T16: entity_id values are identical before and after preprocessing ────
    # (Verifies preprocess_source_df does not touch entity_id)
    id_changed = (df["entity_id"] != df_full["entity_id"]).sum()
    _check(
        f"{file_key}_entity_id_unchanged",
        id_changed == 0,
        f"{id_changed} entity_ids were modified",
    )

    # ── T17: Row count identical between raw and preprocessed ─────────────────
    _check(
        f"{file_key}_row_count_stable",
        len(df) == len(df_full),
        f"before={len(df_full):,}, after={len(df):,}",
    )


def run_dataframe_api_test() -> None:
    """Test preprocess_source_df directly on a hand-crafted DataFrame."""
    _section("preprocess_source_df — API contract test")

    data = {
        "entity_id":        ["S1-001", "S1-002", "S1-003", "S1-004", "S1-005"],
        "business_name":    [
            "-- Holloway Peak Inc",
            "MILLER METALS",
            "राम मार्केटिंग",
            None,
            "SCI Ptit Àmicale",
        ],
        "business_address": [
            "###56 Main St, City",
            "New Delhi, null, A-68",
            "",
            None,
            "18 RUE JEN ZAY, Dunkerque",
        ],
        "country":          ["US", "US", "India", "India", "France"],
    }
    df = pd.DataFrame(data)
    original_entity_ids = df["entity_id"].copy()
    original_len = len(df)

    df = preprocess_source_df(df)

    _check("api_row_count_unchanged",  len(df) == original_len)
    _check("api_entity_id_unchanged",  (df["entity_id"] == original_entity_ids).all())
    _check("api_norm_name_present",    "norm_business_name"    in df.columns)
    _check("api_norm_addr_present",    "norm_business_address" in df.columns)
    _check("api_norm_country_present", "norm_country"          in df.columns)
    _check("api_address_missing_present", "address_missing"    in df.columns)

    # Specific value checks
    _check("api_noise_prefix_removed",
           df.loc[0, "norm_business_name"] == "holloway peak inc")
    _check("api_allcaps_lowered",
           df.loc[1, "norm_business_name"] == "miller metals")
    _check("api_devanagari_preserved",
           "राम" in df.loc[2, "norm_business_name"])
    _check("api_null_name_empty",
           df.loc[3, "norm_business_name"] == "")
    _check("api_french_accents_preserved",
           "àmicale" in df.loc[4, "norm_business_name"])

    _check("api_hash_number_removed",
           df.loc[0, "norm_business_address"].startswith("56"))
    _check("api_null_literal_removed",
           "null" not in df.loc[1, "norm_business_address"])
    _check("api_empty_addr_flag_true",  df.loc[2, "address_missing"] == True)
    _check("api_none_addr_flag_true",   df.loc[3, "address_missing"] == True)
    _check("api_present_addr_flag_false", df.loc[4, "address_missing"] == False)

    _check("api_original_cols_intact",
           {"entity_id", "business_name", "business_address", "country"}
           .issubset(df.columns))
    _check("api_france_country",
           df.loc[4, "norm_country"] == "France")


# ─── Main runner ───────────────────────────────────────────────────────────────


def main() -> int:
    print("=" * 70)
    print("  Amazon ML Challenge 2026 — Preprocessing Test Suite")
    print("  Person 1: Data Preprocessing Verification")
    print("=" * 70)

    # ── Unit tests ─────────────────────────────────────────────────────────────
    _section("Unit: Business Name Normalisation")
    _run("name_noise_prefix_removal",   test_name_noise_prefix_removal)
    _run("name_whitespace_collapse",    test_name_whitespace_collapse)
    _run("name_uppercase_lowercased",   test_name_uppercase_lowercased)
    _run("name_pipe_separator",         test_name_pipe_separator)
    _run("name_devanagari_preserved",   test_name_devanagari_preserved)
    _run("name_bengali_preserved",      test_name_bengali_preserved)
    _run("name_kannada_preserved",      test_name_kannada_preserved)
    _run("name_french_accents",         test_name_french_accents_preserved)
    _run("name_null_and_empty",         test_name_null_and_empty)
    _run("name_legal_suffixes_kept",    test_name_legal_suffixes_kept)
    _run("name_entity_id_not_touched",  test_name_entity_id_not_touched)

    _section("Unit: Business Address Normalisation")
    _run("addr_hash_number_cleanup",    test_addr_hash_number_cleanup)
    _run("addr_null_literal_removal",   test_addr_null_literal_removal)
    _run("addr_component_reorder",      test_addr_component_reorder_preserved)
    _run("addr_non_latin_preserved",    test_addr_non_latin_preserved)
    _run("addr_french_preserved",       test_addr_french_preserved)
    _run("addr_empty_and_none",         test_addr_empty_and_none)
    _run("addr_house_numbers_kept",     test_addr_house_numbers_kept)

    _section("Unit: Country Normalisation")
    _run("country_known_values",        test_country_known_values)
    _run("country_france_test_set",     test_country_france_test_set)
    _run("country_whitespace_stripped", test_country_whitespace_stripped)
    _run("country_case_preserved",      test_country_case_preserved)
    _run("country_null_empty",          test_country_null_empty)

    _section("Unit: Missing-Address Detection")
    _run("missing_address_flags",       test_missing_address_flags)

    # ── DataFrame API test ─────────────────────────────────────────────────────
    _run("dataframe_api",               run_dataframe_api_test)

    # ── Integration tests ──────────────────────────────────────────────────────
    _section("Integration: Real Dataset Files")
    print("\n  NOTE: Large files (source2/source3 ~500MB) may take a few minutes.")
    print("        A 50,000-row sample is used for content probes;")
    print("        row-count and structural checks use the full file.\n")

    for key in SOURCE_KEYS:
        _run(f"integration_{key}", lambda k=key: run_integration_tests_for(k))

    # Ground truth (different schema, lighter checks)
    _section("Integration: Ground Truth File")
    _run("integration_ground_truth",
         lambda: run_integration_tests_for("train_ground_truth"))

    # ── Summary ────────────────────────────────────────────────────────────────
    total = _PASS + _FAIL
    print("\n" + "=" * 70)
    print(f"  RESULTS:  {_PASS} passed  /  {_FAIL} failed  /  {total} total")
    print("=" * 70)

    if _FAIL > 0:
        print("\nFailed tests:")
        for name in _ERRORS:
            print(f"  ✗  {name}")
        return 1

    print("\n  All tests PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
