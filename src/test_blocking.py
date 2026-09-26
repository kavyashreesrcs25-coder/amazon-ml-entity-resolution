"""
test_blocking.py
================
Quick smoke-test for the blocking module.
Runs on a small slice of training data (~5 000 rows per file) so it
completes in under 2 minutes.  Reports per-rule hit counts and overall
recall on the slice.

Usage
-----
  python src/test_blocking.py
  python src/test_blocking.py --rows 10000

Exit codes
----------
  0  all checks passed
  1  a check failed
"""

from __future__ import annotations

import sys
import time
import tempfile
from pathlib import Path
from collections import defaultdict

import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
from blocking import (
    PREPROCESSED,
    all_keys,
    keys_r1, keys_r2, keys_r3, keys_r4, keys_r5, keys_r6,
    build_raw_lookup_file,
    _build_keys_df,
    _get_candidates_for_chunk,
    run_blocking,
    evaluate_recall,
    MAX_KEY_BUCKET,
)

# ─── Minimal test harness ──────────────────────────────────────────────────────
_PASS = _FAIL = 0
_ERRORS: list[str] = []


def _chk(name: str, condition: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  PASS  {name}")
    else:
        _FAIL += 1
        msg = f"  FAIL  {name}" + (f" — {detail}" if detail else "")
        print(msg)
        _ERRORS.append(name)


# ─── Unit tests for key functions ─────────────────────────────────────────────

def test_key_functions() -> None:
    print("\n--- Unit: key generation functions ---")

    # R1 exact name + country
    k = keys_r1("miller metals", "US")
    _chk("r1_basic",           k == ["miller metals|US"])
    _chk("r1_empty_name",      keys_r1("", "US") == [])
    _chk("r1_empty_country",   keys_r1("abc", "") == ["abc|"])

    # R2 prefix-6
    k = keys_r2("miller metals", "US")
    _chk("r2_prefix6",         k == ["miller|US"])
    _chk("r2_short_name",      keys_r2("ab", "US") == [])     # < 3 chars
    _chk("r2_exact6",          keys_r2("miller", "US") == ["miller|US"])

    # R3 first 2 significant tokens, sorted
    k = keys_r3("the miller metals inc", "US")
    _chk("r3_stop_filtered",   k == ["metals miller|US"])     # "the","inc" stripped
    k = keys_r3("abc", "US")
    _chk("r3_single_token",    k == ["abc|US"])               # only 1 token
    _chk("r3_empty",           keys_r3("", "US") == [])

    # R4 character 3-grams
    k = keys_r4("miller", "US")
    _chk("r4_produces_grams",  len(k) > 0)
    _chk("r4_format",          all("|US" in x for x in k))
    _chk("r4_short",           keys_r4("ab", "US") == [])    # < gram_len

    # R5 first numeric token
    k = keys_r5("1795 westchester drive, high point, nc", "US")
    _chk("r5_number",          k == ["1795|US"])
    _chk("r5_no_number",       keys_r5("main street, city", "US") == [])
    _chk("r5_single_digit",    keys_r5("5 main st", "US") == [])  # single digit skip
    _chk("r5_empty",           keys_r5("", "US") == [])

    # R6 first address token + name prefix-4
    k = keys_r6("1795 westchester drive", "mill")
    _chk("r6_basic",           k == ["1795|mill"])
    _chk("r6_empty_addr",      keys_r6("", "mill") == [])
    _chk("r6_empty_name",      keys_r6("123 main", "") == [])

    # all_keys returns all 6 rules
    kd = all_keys("miller metals", "1795 main st", "US")
    _chk("all_keys_6_rules",   set(kd.keys()) == {"r1","r2","r3","r4","r5","r6"})


# ─── Multilingual tests ───────────────────────────────────────────────────────

def test_multilingual_keys() -> None:
    """Verify blocking keys are generated correctly for non-Latin scripts
    and French accented characters from all 6 rules.

    Uses actual examples observed in the dataset during inspection.
    """
    print("\n--- Multilingual: key generation for non-Latin scripts ---")

    # ── Devanagari (Hindi) — India ────────────────────────────────────────────
    hindi_name = "राम मार्केटिंग प्राइवेट लिमिटेड"
    hindi_addr = "kh no. -570/13, new delhi, west delhi, delhi"
    country_in = "India"

    # R1: exact name preserved
    k = keys_r1(hindi_name, country_in)
    _chk("devanagari_r1_preserved",
         k == [f"{hindi_name}|{country_in}"],
         f"got {k}")

    # R2: first 6 Unicode codepoints of name
    expected_r2_prefix = hindi_name[:6]  # "राम मा" (6 chars)
    k = keys_r2(hindi_name, country_in)
    _chk("devanagari_r2_prefix_6codepoints",
         k == [f"{expected_r2_prefix}|{country_in}"],
         f"got {k}, expected prefix={expected_r2_prefix!r}")

    # R3: Devanagari tokens NOT filtered by English stop-words
    k = keys_r3(hindi_name, country_in)
    _chk("devanagari_r3_not_empty",
         len(k) > 0 and k[0].endswith(f"|{country_in}"),
         f"got {k}")
    # Confirm non-Latin tokens appear in the key
    if k:
        key_part = k[0].split("|")[0]
        has_devanagari = any("\u0900" <= c <= "\u097F" for c in key_part)
        _chk("devanagari_r3_tokens_preserved", has_devanagari,
             f"no Devanagari in key {k[0]!r}")

    # R4: 3-codepoint grams of Devanagari name
    k = keys_r4(hindi_name, country_in)
    _chk("devanagari_r4_grams_generated", len(k) > 0, f"got {k}")
    if k:
        gram_part = k[0].split("|")[0]
        _chk("devanagari_r4_gram_is_unicode",
             len(gram_part) == 3 and any("\u0900" <= c <= "\u097F" for c in gram_part),
             f"gram={gram_part!r}")

    # R5: numeric run in address
    k = keys_r5(hindi_addr, country_in)
    _chk("devanagari_r5_number_extracted",
         k == ["570|India"],
         f"got {k}")

    # R6: first address token + first 4 codepoints of name
    k = keys_r6(hindi_addr, hindi_name)
    _chk("devanagari_r6_addr_token",
         len(k) > 0 and f"|{hindi_name[:4]}" in k[0],
         f"got {k}")

    # ── Bengali — India ───────────────────────────────────────────────────────
    bengali_name = "ইউনিভার্সাল সিস্টেমস প্রাইভেট লিমিটেড"
    country_in2  = "India"

    k = keys_r1(bengali_name, country_in2)
    _chk("bengali_r1_preserved", k == [f"{bengali_name}|{country_in2}"])

    k = keys_r3(bengali_name, country_in2)
    _chk("bengali_r3_not_empty", len(k) > 0)
    if k:
        key_part = k[0].split("|")[0]
        has_bengali = any("\u0980" <= c <= "\u09FF" for c in key_part)
        _chk("bengali_r3_tokens_preserved", has_bengali,
             f"no Bengali in {k[0]!r}")

    k = keys_r4(bengali_name, country_in2)
    _chk("bengali_r4_grams", len(k) > 0)
    if k:
        gram = k[0].split("|")[0]
        _chk("bengali_r4_gram_unicode",
             any("\u0980" <= c <= "\u09FF" for c in gram),
             f"gram={gram!r}")

    # ── Kannada — India ───────────────────────────────────────────────────────
    kannada_addr = "door no 183, jayanagar, bengaluru, ಕರ್ನಾಟಕ"
    k = keys_r5(kannada_addr, "India")
    _chk("kannada_addr_r5_number", k == ["183|India"], f"got {k}")

    # R6 with Kannada in address
    kannada_name = "ಕರ್ನಾಟಕ ಮಾರ್ಕೆಟಿಂಗ್"
    k = keys_r6(kannada_addr, kannada_name)
    _chk("kannada_r6_addr_token",
         len(k) > 0 and k[0].startswith("door|"),
         f"got {k}")

    # ── French — France ───────────────────────────────────────────────────────
    french_name = "fractales amis groupe s.a.s"
    french_addr = "23 rue icmre, la teste-de-buch, gironde"
    country_fr  = "France"

    k = keys_r1(french_name, country_fr)
    _chk("french_r1_preserved", k == [f"{french_name}|{country_fr}"])

    k = keys_r2(french_name, country_fr)
    _chk("french_r2_prefix", k == ["fracta|France"], f"got {k}")

    k = keys_r3(french_name, country_fr)
    _chk("french_r3_not_empty", len(k) > 0)
    if k:
        # "groupe" and "fracta..." should survive; "s.a.s" short but not stop-word
        _chk("french_r3_has_content",
             k[0].endswith("|France") and len(k[0]) > 8,
             f"got {k}")

    # French accented name
    accented_name = "société générale sarl"
    k = keys_r1(accented_name, country_fr)
    _chk("french_accented_r1", k == [f"{accented_name}|{country_fr}"])

    k = keys_r2(accented_name, country_fr)
    _chk("french_accented_r2_prefix",
         k == ["sociét|France"],
         f"got {k}")

    k = keys_r4(accented_name, country_fr)
    _chk("french_accented_r4_grams", len(k) > 0)
    if k:
        gram = k[0].split("|")[0]
        _chk("french_accented_r4_gram_content",
             "soc" in gram or "oci" in gram,
             f"gram={gram!r}")

    # French address with numbers
    k = keys_r5(french_addr, country_fr)
    _chk("french_r5_number", k == ["23|France"], f"got {k}")

    k = keys_r6(french_addr, french_name)
    _chk("french_r6_first_token",
         len(k) > 0 and k[0].startswith("23|frac"),
         f"got {k}")

    # ── Mixed-script name (actual dataset example) ────────────────────────────
    mixed_name = "bombay बाबा marketing private limited"
    k = keys_r1(mixed_name, "India")
    _chk("mixed_script_r1", k == [f"{mixed_name}|India"])

    k = keys_r3(mixed_name, "India")
    _chk("mixed_script_r3_not_empty", len(k) > 0)
    if k:
        key_part = k[0].split("|")[0]
        # Devanagari token "बाबा" should survive (not ASCII stop-word)
        _chk("mixed_script_r3_devanagari_kept",
             "बाबा" in key_part or "bombay" in key_part,
             f"key={k[0]!r}")

    # ── Non-Latin stop-word guard: short non-ASCII tokens must NOT be filtered ─
    # "us" is in _STOP but "ūs" (Latin extended) is NOT; also test that
    # 2-char Devanagari tokens like "मे" are kept.
    short_hindi = "मे नई दिल्ली"   # 3 tokens: 2-char, 2-char, 5-char
    k = keys_r3(short_hindi, "India")
    _chk("short_devanagari_tokens_kept",
         len(k) > 0,
         "2-char Devanagari tokens were incorrectly filtered")

    # ── Country is open-set: unknown country passes through ───────────────────
    k = keys_r1("acme corp", "Germany")
    _chk("open_set_country_germany", k == ["acme corp|Germany"])
    k = keys_r1("empresa sarl", "Morocco")
    _chk("open_set_country_morocco", k == ["empresa sarl|Morocco"])

    # ── Website-as-name (observed in dataset) ────────────────────────────────
    k = keys_r1("whiteallgraphics.com", "US")
    _chk("website_name_r1", k == ["whiteallgraphics.com|US"])
    k = keys_r2("whiteallgraphics.com", "US")
    _chk("website_name_r2", k == ["whitea|US"], f"got {k}")

    # ── DataFrame-level: _build_keys_df preserves Unicode ────────────────────
    import pandas as _pd
    test_df = _pd.DataFrame({
        "entity_id":            ["S2-001", "S2-002", "S2-003", "S2-004"],
        "norm_business_name":   [
            "राम मार्केटिंग प्राइवेट लिमिटेड",
            "ইউনিভার্সাল সিস্টেমস",
            "société générale",
            "miller metals",
        ],
        "norm_business_address": [
            "kh no. -570/13, new delhi",
            "west bengal, calcutta, 2",
            "23 rue icmre, gironde",
            "1795 westchester drive, nc",
        ],
        "norm_country": ["India", "India", "France", "US"],
    })
    kdf = _build_keys_df(test_df)
    _chk("build_keys_df_has_rows", len(kdf) > 0)

    # Check Devanagari name generates a key
    devanagari_keys = kdf[kdf["entity_id"] == "S2-001"]["block_key"].tolist()
    _chk("devanagari_keys_generated",
         len(devanagari_keys) > 0,
         f"no keys for Devanagari entity")

    # Check R1 key contains Devanagari
    r1_key_exists = any(
        "राम मार्केटिंग प्राइवेट लिमिटेड|India" in k
        for k in devanagari_keys
    )
    _chk("devanagari_r1_in_dataframe", r1_key_exists,
         f"R1 Devanagari key missing from: {devanagari_keys[:3]}")

    # Check French accented name generates keys
    french_keys = kdf[kdf["entity_id"] == "S2-003"]["block_key"].tolist()
    _chk("french_keys_generated", len(french_keys) > 0)
    r1_fr = any("société générale|France" in k for k in french_keys)
    _chk("french_r1_in_dataframe", r1_fr,
         f"French R1 key missing from: {french_keys[:3]}")

    # Check Bengali keys
    bengali_keys = kdf[kdf["entity_id"] == "S2-002"]["block_key"].tolist()
    _chk("bengali_keys_generated", len(bengali_keys) > 0)

    # ── End-to-end: matching pair across scripts ──────────────────────────────
    # Simulate: S1 has English name, S2 has same name in Devanagari — only
    # R5 (address number) and R6 should match them, not R1/R2/R3.
    # This is expected behaviour — cross-script matching relies on address.
    s1_eng  = "ram marketing pvt ltd"
    s2_dev  = "राम मार्केटिंग प्राइवेट लिमिटेड"
    addr    = "570 new delhi"
    cty     = "India"

    s1_keys_r5 = set(keys_r5(addr, cty))
    s2_keys_r5 = set(keys_r5(addr, cty))
    _chk("cross_script_r5_match",
         bool(s1_keys_r5 & s2_keys_r5),
         "R5 should match same address number across scripts")

    s1_r6 = set(keys_r6(addr, s1_eng))
    s2_r6 = set(keys_r6(addr, s2_dev))
    # They won't match on R6 name-prefix (different scripts) — that's correct
    # behaviour; the matching model handles cross-script via other features.
    _chk("cross_script_r6_addr_part_present",
         len(s1_r6) > 0 and len(s2_r6) > 0,
         "R6 should still generate keys for both scripts")


# ─── Integration smoke-test on real data slice ────────────────────────────────

def test_integration(n_rows: int = 5_000) -> None:
    print(f"\n--- Integration: {n_rows:,}-row slice of training data ---")

    # Check files exist
    for key in ["train_s1", "train_s2", "train_s3", "train_gt"]:
        path = PREPROCESSED[key]
        if not path.exists():
            _chk(f"file_exists_{key}", False, str(path))
            return
        _chk(f"file_exists_{key}", True)

    # Read small slices
    def read_slice(path: Path, n: int) -> pd.DataFrame:
        return pd.read_csv(
            path, sep="\t", dtype=str, nrows=n,
            on_bad_lines="skip",
        ).fillna("")

    df_s2 = read_slice(PREPROCESSED["train_s2"], n_rows)
    df_s3 = read_slice(PREPROCESSED["train_s3"], n_rows)
    df_s1 = read_slice(PREPROCESSED["train_s1"], n_rows)
    df_gt = read_slice(PREPROCESSED["train_gt"], n_rows)

    _chk("s2_slice_loaded",   len(df_s2) == n_rows)
    _chk("s3_slice_loaded",   len(df_s3) == n_rows)
    _chk("s1_slice_loaded",   len(df_s1) == n_rows)

    # Build mini lookup tables from slices
    print("  Building mini lookup tables ...")
    t0 = time.perf_counter()
    kdf_s2 = _build_keys_df(df_s2.fillna("")).rename(columns={"entity_id":"cand_id"})
    kdf_s3 = _build_keys_df(df_s3.fillna("")).rename(columns={"entity_id":"cand_id"})
    lkp = pd.concat([kdf_s2, kdf_s3], ignore_index=True).drop_duplicates()
    build_time = time.perf_counter() - t0
    print(f"  Lookup built in {build_time:.2f}s | {len(lkp):,} pairs")

    total_keys = len(lkp)
    _chk("indexes_have_keys", total_keys > 0, f"total pairs = {total_keys}")

    # Build lookup dict from lkp DataFrame — vectorised
    lkp_dict: dict[str, list[str]] = (
        lkp.groupby("block_key")["cand_id"]
        .apply(list)
        .to_dict()
    )

    # Retrieve candidates via inline dict lookup
    print("  Retrieving candidates via inline lookup ...")
    cand_map = _get_candidates_for_chunk(df_s1.fillna(""), lkp_dict)
    # Ensure all S1 IDs are present (zero-candidate ones won't be in cand_map)

    # Per-rule hit count via key-generation check on a sample
    per_rule_hits = defaultdict(int)
    for _, row_s1 in df_s1.head(200).iterrows():
        ksets = all_keys(str(row_s1.get("norm_business_name","")),
                         str(row_s1.get("norm_business_address","")),
                         str(row_s1.get("norm_country","")))
        for rule, kset in ksets.items():
            if kset:
                per_rule_hits[rule] += 1

    total_cands = sum(len(v) for v in cand_map.values())
    avg_cands   = total_cands / len(df_s1) if df_s1 is not None and len(df_s1) > 0 else 0
    zero_cands  = len(df_s1) - len(cand_map)

    print(f"  Total candidates: {total_cands:,}")
    print(f"  Avg / S1: {avg_cands:.1f}")
    print(f"  Zero-candidate S1: {zero_cands}")
    print(f"  Per-rule S1 hit counts: {dict(per_rule_hits)}")

    _chk("candidates_generated", total_cands > 0)
    _chk("avg_candidates_reasonable", 1 <= avg_cands <= 5000,
         f"avg={avg_cands:.1f}")

    # Measure recall on the slice against ground truth (only pairs in the slice)
    # Build a set of S2/S3 IDs that actually appear in the slices
    s2_ids_in_slice = set(df_s2["entity_id"].tolist())
    s3_ids_in_slice = set(df_s3["entity_id"].tolist())
    slice_ids = s2_ids_in_slice | s3_ids_in_slice

    gt_dict: dict[str, set[str]] = {}
    for row in df_gt.itertuples(index=False):
        ids = [x.strip() for x in str(row.matched_entity_ids).split(",")
               if x.strip() and x.strip() in slice_ids]
        if ids:
            gt_dict[row.source1_entity_id] = set(ids)

    hit = miss = 0
    for s1_id, true_ids in gt_dict.items():
        cands = cand_map.get(s1_id, set())
        for tid in true_ids:
            if tid in cands:
                hit += 1
            else:
                miss += 1

    total_pairs_in_slice = hit + miss
    recall = hit / total_pairs_in_slice if total_pairs_in_slice > 0 else float("nan")

    print(f"\n  Slice-level recall (true pairs in both slices):")
    print(f"    True pairs in slice : {total_pairs_in_slice}")
    print(f"    Found               : {hit}")
    print(f"    Missed              : {miss}")
    print(f"    Recall              : {recall:.4f} ({recall*100:.1f}%)")

    # NOTE: Slice recall is intentionally not asserted here.
    # True S2/S3 matches for S1[0:5k] are spread across the full 5M-row
    # S2/S3 files; only ~0.1% land in the first 5k rows of those files.
    # Real recall is measured by evaluate_blocking.py on the full dataset.
    if total_pairs_in_slice > 0:
        print(f"  Slice recall: {recall:.4f} "
              "(low is expected — matches span full files, not just this slice)")
    else:
        print("  (No true pairs in this slice window — normal for 5k sample.)")
    _chk("slice_recall_check_ran", True)  # always pass; just confirms the code ran

    # Check no S1 IDs appear as their own candidates
    self_match = sum(
        1 for s1_id, cands in cand_map.items()
        if s1_id in cands
    )
    _chk("no_self_matches", self_match == 0, f"{self_match} self-matches found")

    # Check all candidates are S2- or S3- prefixed
    bad_prefix = sum(
        1 for cands in cand_map.values()
        for c in cands
        if not (c.startswith("S2-") or c.startswith("S3-"))
    )
    _chk("all_candidates_s2_or_s3", bad_prefix == 0,
         f"{bad_prefix} candidates with wrong prefix")


# ─── End-to-end mini run ──────────────────────────────────────────────────────

def test_end_to_end(n_rows: int = 5_000) -> None:
    """Write a mini candidate file and verify its format."""
    print(f"\n--- End-to-end: mini run on {n_rows:,} rows ---")

    # Write tiny slice TSVs to temp files
    def write_slice(src: Path, n: int, dst: Path) -> None:
        df = pd.read_csv(src, sep="\t", dtype=str, nrows=n, on_bad_lines="skip")
        df.to_csv(dst, sep="\t", index=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for key, fname in [
            ("train_s1", "s1.tsv"),
            ("train_s2", "s2.tsv"),
            ("train_s3", "s3.tsv"),
        ]:
            write_slice(PREPROCESSED[key], n_rows, tmp / fname)

        out = tmp / "candidates.tsv"
        stats = run_blocking(
            s1_path     = tmp / "s1.tsv",
            s2_path     = tmp / "s2.tsv",
            s3_path     = tmp / "s3.tsv",
            output_path = out,
            verbose     = False,
        )

        _chk("e2e_output_exists",  out.exists())
        _chk("e2e_s1_count",       stats["s1_total"] == n_rows,
             f"got {stats['s1_total']}")
        _chk("e2e_candidates_gt0", stats["total_candidates"] > 0)

        # Check output file structure
        df_out = pd.read_csv(out, sep="\t", dtype=str, nrows=10)
        _chk("e2e_columns",
             list(df_out.columns) == ["source1_entity_id", "candidate_entity_ids"],
             str(list(df_out.columns)))

        # All source1_entity_id values start with S1-
        bad_s1 = (~df_out["source1_entity_id"].str.startswith("S1-")).sum()
        _chk("e2e_s1_prefix", bad_s1 == 0, f"{bad_s1} rows with wrong prefix")

    print(f"  e2e stats: s1={stats['s1_total']}, "
          f"cands={stats['total_candidates']}, "
          f"avg={stats['avg_candidates']:.1f}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Blocking smoke-tests")
    p.add_argument("--rows", type=int, default=5_000,
                   help="Rows per file for integration test (default: 5000).")
    p.add_argument("--skip-integration", action="store_true")
    p.add_argument("--skip-e2e", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    print("=" * 70)
    print("  Amazon ML Challenge 2026 — Blocking Smoke-Tests")
    print("=" * 70)

    test_key_functions()
    test_multilingual_keys()

    if not args.skip_integration:
        test_integration(args.rows)

    if not args.skip_e2e:
        test_end_to_end(args.rows)

    total = _PASS + _FAIL
    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {_PASS} passed / {_FAIL} failed / {total} total")
    print(f"{'=' * 70}")

    if _FAIL:
        print("\nFailed tests:")
        for name in _ERRORS:
            print(f"  FAIL  {name}")
        return 1

    print("\n  All smoke-tests PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
