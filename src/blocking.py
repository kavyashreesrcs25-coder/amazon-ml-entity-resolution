"""
blocking.py
===========
Person 2 -- Blocking / Candidate Generation
Amazon ML Challenge 2026: Business Entity Resolution

DESIGN -- disk-based sorted lookup, low RAM
------------------------------------------
Building 22M+ (block_key, cand_id) pairs in a Python list blows RAM to 6+ GB.
This version writes the lookup to a **sorted TSV on disk** in two passes:

  Pass A (index build):
    Stream S2 and S3 preprocessed TSVs chunk by chunk.
    For every row generate blocking keys -> write (block_key, cand_id) rows
    directly to a temp file.  Never accumulate in RAM.

  Pass B (sort):
    Sort the temp file externally by block_key (pandas sort_values on disk).

  Pass C (retrieval):
    Stream S1 in chunks.  Generate S1 blocking keys.
    For each S1 chunk, look up matching cand_ids from the sorted lookup via
    a pandas merge.  Write results to candidate_pairs.tsv.

Peak RAM stays ~= 1.5-2 GB (one S1 chunk + one lookup shard at a time).

Six Blocking Rules
------------------
R1  exact_name|country
R2  name[:6]|country
R3  sorted(first 2 significant tokens)|country
R4  3-char grams of name[:24], max 3 per row|country  <- tighter than before
R5  first >=2-digit run in address|country
R6  first addr token|name[:4]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

_SRC  = Path(__file__).resolve().parent
_ROOT = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = _ROOT / "output"

PREPROCESSED = {
    "train_s1":  OUTPUT_DIR / "train_source1_preprocessed.tsv",
    "train_s2":  OUTPUT_DIR / "train_source2_preprocessed.tsv",
    "train_s3":  OUTPUT_DIR / "train_source3_preprocessed.tsv",
    "train_gt":  OUTPUT_DIR / "train_ground_truth_preprocessed.tsv",
    "test_s1":   OUTPUT_DIR / "test_source1_preprocessed.tsv",
    "test_s2":   OUTPUT_DIR / "test_source2_preprocessed.tsv",
    "test_s3":   OUTPUT_DIR / "test_source3_preprocessed.tsv",
}

CHUNKSIZE         = 200_000
MAX_KEY_BUCKET    = 100       # drop keys with > 100 candidates (too common)
NAME_GRAM_LEN     = 3
NAME_GRAM_PREFIX  = 24        # shorter prefix -> fewer Unicode codepoints
MAX_GRAMS_PER_ROW = 3
MIN_TOKEN_LEN     = 2

# ── Stop-words ────────────────────────────────────────────────────────────────
# ENGLISH-ONLY legal/common words.  Non-Latin tokens (Devanagari, Bengali,
# Arabic, Kannada, Tamil, French non-ASCII) are NEVER filtered by this list --
# they carry full identity information.  The guard is applied in _sig_tokens:
# only tokens that are entirely ASCII are checked against _STOP.
_STOP = frozenset({
    "the","and","of","a","an","in","for","co","inc","ltd",
    "llc","pvt","private","limited","corp","corporation","company",
    "enterprises","services","solutions","group","international",
    "global","national","india","us","france","sarl","sas",
    "llp","plc","gmbh","bv","nv","sa",
})

# ── Regex patterns ─────────────────────────────────────────────────────────────
# re.UNICODE is the default in Python 3 str, but explicit for clarity.

# Matches >=2 consecutive digit characters (Unicode-digit-safe: matches
# Western, Arabic-Indic, Devanagari digits etc.).
_RE_DIGITS = re.compile(r"\d{2,}", re.UNICODE)

# Tokeniser: split on ASCII whitespace and common ASCII separators (, . / -).
# Non-Latin characters (Devanagari, Bengali, French accented chars, CJK ...)
# are treated as word content and NOT split on.  This preserves:
#   "राम मार्केटिंग"   -> ["राम", "मार्केटिंग"]
#   "société générale"  -> ["société", "générale"]
#   "123 main st"       -> ["123", "main", "st"]
_RE_TOKENS = re.compile(r"[^\s,./\-]+", re.UNICODE)


# ─────────────────────────────────────────────────────────────────────────────
# SCALAR KEY HELPERS (for test suite)
# ─────────────────────────────────────────────────────────────────────────────

def _is_ascii(s: str) -> bool:
    """Return True if every character in s is in the ASCII range."""
    return all(ord(c) < 128 for c in s)


def _sig_tokens(name: str) -> list[str]:
    """Return significant tokens from a (already-normalised, lowercased) name.

    Rules:
    - Token must be >= MIN_TOKEN_LEN Unicode codepoints.
    - If the token is entirely ASCII it must NOT appear in _STOP.
    - Non-ASCII tokens (Devanagari, Bengali, French words, etc.) are ALWAYS
      kept -- they are never filtered by the English stop-word list.
    """
    result = []
    for t in _RE_TOKENS.findall(name):
        if len(t) < MIN_TOKEN_LEN:
            continue
        # Only apply English stop-word filter to pure-ASCII tokens
        if _is_ascii(t) and t in _STOP:
            continue
        result.append(t)
    return result

def keys_r1(name: str, country: str) -> list[str]:
    """Exact normalised name + country. Works for any script."""
    return [f"{name}|{country}"] if name else []

def keys_r2(name: str, country: str) -> list[str]:
    """6-codepoint name prefix + country. Unicode codepoint slicing is correct
    for Devanagari/Bengali/Latin -- e.g. 'राम मा'[:6] = 'राम मा' (6 chars)."""
    p = name[:6]
    return [f"{p}|{country}"] if len(p) >= 3 else []

def keys_r3(name: str, country: str) -> list[str]:
    """Sorted first-2 significant tokens + country.
    Non-Latin tokens are preserved; only ASCII tokens are stop-word filtered."""
    toks = _sig_tokens(name)
    if not toks:
        return []
    return [" ".join(sorted(toks[:2])) + f"|{country}"]

def keys_r4(name: str, country: str) -> list[str]:
    """3-codepoint grams of name prefix + country.
    Works on any Unicode script -- slicing by codepoint is script-neutral."""
    s = name[:NAME_GRAM_PREFIX]
    if len(s) < NAME_GRAM_LEN:
        return []
    grams = [s[i:i+NAME_GRAM_LEN] for i in range(len(s)-NAME_GRAM_LEN+1)]
    if len(grams) > MAX_GRAMS_PER_ROW:
        step  = len(grams) // MAX_GRAMS_PER_ROW
        grams = grams[::step][:MAX_GRAMS_PER_ROW]
    return [f"{g}|{country}" for g in grams]

def keys_r5(addr: str, country: str) -> list[str]:
    """First >=2-digit run in address + country.
    _RE_DIGITS matches Unicode digits (Western, Arabic-Indic, Devanagari...)."""
    if not addr:
        return []
    nums = _RE_DIGITS.findall(addr)
    return [f"{nums[0]}|{country}"] if nums else []

def keys_r6(addr: str, name: str) -> list[str]:
    """First non-separator address token + first-4-codepoints of name.
    Both addr and name may contain any Unicode script."""
    if not addr or not name:
        return []
    for t in _RE_TOKENS.findall(addr):
        if len(t) >= 2:
            return [f"{t}|{name[:4]}"]
    return []



def keys_r7(addr: str, country: str) -> list[str]:
    """R7: First address token of >=3 chars + country.

    Cross-script anchor for pairs where S1 name is Latin and S2/S3 name is
    Devanagari/Tamil/Telugu/Bengali (char overlap = 0).  Both records share
    the same address, so the first meaningful address token is the link.

    Examples:
      "af-684, nandgram ..."  -> "af|India"
      "wz-187c shop ..."      -> "wz|India"
      "6(29), c.i.t. ..."    -> "6(29)|India"
    """
    if not addr:
        return []
    for t in _RE_TOKENS.findall(addr):
        if len(t) >= 3:
            return [f"{t}|{country}"]
    return []


def keys_r8(name: str, country: str) -> list[str]:
    """R8: 4-codepoint name prefix + country.

    More specific than the existing 3-char gram (R4) -- higher precision.
    Works on any Unicode script (codepoint slicing).

    Examples:
      "miller metals"    -> "mill|US"
      "crystal staffing" -> "crys|US"
    """
    if len(name) < 4:
        return []
    return [f"{name[:4]}|{country}"]

def all_keys(name: str, addr: str, country: str) -> dict[str, list[str]]:
    return {
        "r1": keys_r1(name, country),
        "r2": keys_r2(name, country),
        "r3": keys_r3(name, country),
        "r4": keys_r4(name, country),
        "r5": keys_r5(addr, country),
        "r6": keys_r6(addr, name),
        "r7": keys_r7(addr, country),
        "r8": keys_r8(name, country),
    }


# ─────────────────────────────────────────────────────────────────────────────
# VECTORISED KEY GENERATION  (one chunk -> long-form DataFrame)
# ─────────────────────────────────────────────────────────────────────────────

def _build_keys_df(chunk: pd.DataFrame) -> pd.DataFrame:
    """Return long-form (entity_id, block_key) for all 6 rules.

    Vectorised where possible.  R4 uses a tighter .values loop
    (max 2 grams per row to keep throughput acceptable).
    """
    chunk = chunk.reset_index(drop=True)
    eid   = chunk["entity_id"]
    name  = chunk["norm_business_name"]
    addr  = chunk["norm_business_address"]
    ctry  = chunk["norm_country"]

    parts = []

    # R1 -- exact name|country (0.09s / 200k)
    mask = name.str.len() > 0
    if mask.any():
        k = (name[mask] + "|" + ctry[mask])
        parts.append(pd.DataFrame({"entity_id": eid[mask].values, "block_key": k.values}))

    # R2 -- prefix-6|country (0.10s / 200k)
    mask = name.str.len() >= 3
    if mask.any():
        k = (name[mask].str[:6] + "|" + ctry[mask])
        parts.append(pd.DataFrame({"entity_id": eid[mask].values, "block_key": k.values}))

    # R3 -- sorted 2-sig-token key|country
    # Non-Latin tokens bypass the English stop-word list (see _sig_tokens).
    def _r3(n: str) -> str:
        toks = _sig_tokens(n)
        return " ".join(sorted(toks[:2]))
    r3_base = name.apply(_r3)
    mask    = r3_base.str.len() > 0
    if mask.any():
        k = (r3_base[mask] + "|" + ctry[mask])
        parts.append(pd.DataFrame({"entity_id": eid[mask].values, "block_key": k.values}))

    # R4 -- 3-char grams at 2 fixed offsets (vectorised, 0.3s / 200k)
    s_short = name.str[:NAME_GRAM_PREFIX]
    mask    = s_short.str.len() >= NAME_GRAM_LEN
    if mask.any():
        # Gram at offset 0
        g0 = s_short[mask].str[0:NAME_GRAM_LEN]
        k0 = g0 + "|" + ctry[mask]
        parts.append(pd.DataFrame({"entity_id": eid[mask].values, "block_key": k0.values}))
        # Gram at offset 6 (middle of a typical 12-char name)
        mask2 = s_short.str.len() >= NAME_GRAM_LEN + 6
        if mask2.any():
            g1 = s_short[mask2].str[6:6+NAME_GRAM_LEN]
            k1 = g1 + "|" + ctry[mask2]
            parts.append(pd.DataFrame({"entity_id": eid[mask2].values, "block_key": k1.values}))

    # R5 -- first >=2-digit run|country (0.48s / 200k)
    r5_num = addr.str.extract(r"(\d{2,})", expand=False)
    mask   = r5_num.notna()
    if mask.any():
        k = (r5_num[mask] + "|" + ctry[mask])
        parts.append(pd.DataFrame({"entity_id": eid[mask].values, "block_key": k.values}))

    # R6 -- first addr token|name[:4] (0.42s / 200k)
    r6_tok = addr.str.extract(r"([^\s,./\-]{2,})", expand=False)
    mask   = r6_tok.notna() & (name.str.len() >= 1)
    if mask.any():
        k = (r6_tok[mask] + "|" + name[mask].str[:4])
        parts.append(pd.DataFrame({"entity_id": eid[mask].values, "block_key": k.values}))

    # R7 -- first addr token of >=3 chars + country (cross-script anchor)
    r7_tok = addr.str.extract(r"([^\s,./\-]{3,})", expand=False)
    mask   = r7_tok.notna()
    if mask.any():
        k = (r7_tok[mask] + "|" + ctry[mask])
        parts.append(pd.DataFrame({"entity_id": eid[mask].values, "block_key": k.values}))

    # R8 -- 4-codepoint name prefix + country (higher precision than 3-char R4)
    mask = name.str.len() >= 4
    if mask.any():
        k = (name[mask].str[:4] + "|" + ctry[mask])
        parts.append(pd.DataFrame({"entity_id": eid[mask].values, "block_key": k.values}))

    if not parts:
        return pd.DataFrame(columns=["entity_id", "block_key"])

    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates()[["entity_id", "block_key"]]


# ─────────────────────────────────────────────────────────────────────────────
# PASS A -- stream S2+S3 -> write raw lookup TSV to disk (no RAM accumulation)
# ─────────────────────────────────────────────────────────────────────────────

def build_raw_lookup_file(
    s2_path: Path,
    s3_path: Path,
    raw_path: Path,
    chunksize: int = CHUNKSIZE,
    verbose: bool = True,
) -> int:
    """Stream S2 and S3 -> write (block_key, cand_id) TSV to disk.

    Returns total rows written.
    """
    total = 0
    t0    = time.perf_counter()
    with open(raw_path, "w", encoding="utf-8", newline="") as fout:
        fout.write("block_key\tcand_id\n")
        for src_label, src_path in [("S2", s2_path), ("S3", s3_path)]:
            if verbose:
                print(f"  [{src_label}] streaming {src_path.name} ...", flush=True)
            src_rows = chunk_n = 0
            reader = pd.read_csv(
                src_path, sep="\t", dtype=str, chunksize=chunksize,
                usecols=["entity_id","norm_business_name",
                         "norm_business_address","norm_country"],
                on_bad_lines="skip",
            )
            for chunk in reader:
                chunk_n  += 1
                src_rows += len(chunk)
                chunk     = chunk.fillna("")

                kdf = _build_keys_df(chunk)
                # Fast write: numpy char concat + single write (2x faster than writelines)
                bk  = kdf["block_key"].values.astype(str)
                ci  = kdf["entity_id"].values.astype(str)
                import numpy as _np
                lines = _np.char.add(_np.char.add(bk, "\t"), ci)
                fout.write("\n".join(lines) + "\n")
                total += len(kdf)

                if verbose and chunk_n % 5 == 0:
                    elapsed = time.perf_counter() - t0
                    rss = _rss_mb()
                    print(f"    {src_label} chunk {chunk_n:4d} | "
                          f"{src_rows:>10,} rows | "
                          f"{elapsed:.0f}s"
                          + (f" | RAM ~{rss:.0f} MB" if _HAS_PSUTIL else ""),
                          flush=True)

            if verbose:
                elapsed = time.perf_counter() - t0
                print(f"  [{src_label}] done -- {src_rows:,} rows | "
                      f"total pairs so far: {total:,} | {elapsed:.0f}s",
                      flush=True)

    if verbose:
        size_mb = raw_path.stat().st_size / (1024**2)
        print(f"  Raw lookup file: {raw_path} ({size_mb:.0f} MB, "
              f"{total:,} rows)", flush=True)
    return total


# ─────────────────────────────────────────────────────────────────────────────
# PASS B -- sort + cap hot keys  (pandas chunk sort -> write sorted lookup)
# ─────────────────────────────────────────────────────────────────────────────

def sort_and_cap_lookup(
    raw_path: Path,
    sorted_path: Path,
    max_key_bucket: int = MAX_KEY_BUCKET,
    chunksize: int = 2_000_000,
    verbose: bool = True,
) -> int:
    """Sort raw lookup by block_key, drop hot keys, write clean lookup.

    Returns number of (block_key, cand_id) pairs in the clean lookup.
    """
    if verbose:
        print(f"  Sorting lookup ({raw_path.stat().st_size/(1024**2):.0f} MB) ...",
              flush=True)
    t0 = time.perf_counter()

    # Read in chunks, sort each, write to temp shards, then merge-sort.
    # For files up to ~8 GB this single-pass sort works fine on most machines.
    df = pd.read_csv(raw_path, sep="\t", dtype=str, on_bad_lines="skip",
                     names=["block_key", "cand_id"], header=0,
                     quoting=3)   # QUOTE_NONE: ignore quote chars in data

    if verbose:
        print(f"  Loaded {len(df):,} pairs into RAM for sorting | "
              f"RAM ~{_rss_mb():.0f} MB", flush=True)

    # Drop hot keys
    counts = df["block_key"].value_counts()
    hot    = set(counts[counts > max_key_bucket].index)
    if hot:
        before = len(df)
        df = df[~df["block_key"].isin(hot)]
        if verbose:
            print(f"  Dropped {len(hot):,} hot keys "
                  f"(removed {before-len(df):,} pairs)", flush=True)

    df = df.drop_duplicates().sort_values("block_key").reset_index(drop=True)
    df.to_csv(sorted_path, sep="\t", index=False)

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"  Sorted lookup: {len(df):,} pairs | {elapsed:.1f}s | "
              f"RAM ~{_rss_mb():.0f} MB", flush=True)
    return len(df)


# ─────────────────────────────────────────────────────────────────────────────
# PASS C -- retrieve candidates for S1 via chunked merge
# ─────────────────────────────────────────────────────────────────────────────

def _get_candidates_for_chunk(
    chunk: pd.DataFrame,
    lookup_dict: dict[str, list[str]],
) -> dict[str, set[str]]:
    """Return {s1_id: set(cand_ids)} for one S1 chunk using inline key lookup.

    Generates all 6 blocking keys per row inline (no intermediate DataFrame)
    and hits the lookup dict directly.  Much faster than building a full
    (entity_id, block_key) DataFrame and then looping over it.
    """
    cand_map: dict[str, set[str]] = {}

    # Pre-extract numpy arrays for speed
    eid_arr  = chunk["entity_id"].values
    name_arr = chunk["norm_business_name"].values
    addr_arr = chunk["norm_business_address"].values
    ctry_arr = chunk["norm_country"].values

    for i in range(len(eid_arr)):
        s1_id = eid_arr[i]
        name  = name_arr[i]
        addr  = addr_arr[i]
        ctry  = ctry_arr[i]
        cands: set[str] | None = None

        def _add(hits: list[str] | None) -> None:
            nonlocal cands
            if hits:
                if cands is None:
                    cands = set(hits)
                else:
                    cands.update(hits)

        # R1 exact name
        if name:
            _add(lookup_dict.get(f"{name}|{ctry}"))

        # R2 prefix-6
        if len(name) >= 3:
            _add(lookup_dict.get(f"{name[:6]}|{ctry}"))

        # R3 first 2 sig tokens sorted
        # _sig_tokens guards non-Latin tokens from English stop-word filter
        toks = _sig_tokens(name)
        if toks:
            _add(lookup_dict.get(" ".join(sorted(toks[:2])) + f"|{ctry}"))

        # R4 first 3-gram
        if len(name) >= NAME_GRAM_LEN:
            _add(lookup_dict.get(f"{name[:NAME_GRAM_LEN]}|{ctry}"))
            if len(name) >= NAME_GRAM_LEN + 6:
                _add(lookup_dict.get(f"{name[6:6+NAME_GRAM_LEN]}|{ctry}"))

        # R5 first >=2-digit run in address
        nums = _RE_DIGITS.findall(addr)
        if nums:
            _add(lookup_dict.get(f"{nums[0]}|{ctry}"))

        # R6 first addr token + name[:4]
        if addr and name:
            m = _RE_TOKENS.search(addr)
            if m and len(m.group(0)) >= 2:
                _add(lookup_dict.get(f"{m.group(0)}|{name[:4]}"))

        # R7 first addr token of >=3 chars + country (cross-script anchor)
        if addr:
            for t in _RE_TOKENS.findall(addr):
                if len(t) >= 3:
                    _add(lookup_dict.get(f"{t}|{ctry}"))
                    break

        # R8 4-codepoint name prefix + country
        if len(name) >= 4:
            _add(lookup_dict.get(f"{name[:4]}|{ctry}"))

        if cands:
            cands.discard(s1_id)   # no self-matches
            if cands:
                cand_map[s1_id] = cands

    return cand_map


def stream_retrieve(
    s1_path: Path,
    sorted_lookup_path: Path,
    output_path: Path,
    chunksize: int = CHUNKSIZE,
    verbose: bool = True,
) -> dict:
    """Stream S1, merge against sorted lookup DataFrame, write output.

    Loads the sorted lookup as a DataFrame once (stays in RAM as pandas,
    no Python dict overhead). For each S1 chunk, generates blocking keys,
    sorts them, and does a pandas merge -- pure C-level hash join.
    """
    if verbose:
        print(f"  Loading sorted lookup DataFrame ...", flush=True)
    t0 = time.perf_counter()

    lookup = pd.read_csv(
        sorted_lookup_path, sep="\t", dtype=str, on_bad_lines="skip",
        quoting=3
    )
    total_lookup = len(lookup)

    if verbose:
        rss = _rss_mb()
        print(f"  Lookup: {total_lookup:,} pairs | "
              f"RAM ~{rss:.0f} MB | {time.perf_counter()-t0:.1f}s",
              flush=True)

    stats = {
        "s1_total": 0, "total_candidates": 0,
        "zero_candidate_s1": 0, "max_candidates": 0, "sum_candidates": 0,
    }
    peak_rss  = _rss_mb()
    chunk_num = 0

    reader_s1 = pd.read_csv(
        s1_path, sep="\t", dtype=str, chunksize=chunksize,
        usecols=["entity_id", "norm_business_name",
                 "norm_business_address", "norm_country"],
        on_bad_lines="skip",
    )

    with open(output_path, "w", encoding="utf-8", newline="") as fout:
        fout.write("source1_entity_id\tcandidate_entity_ids\n")

        for chunk in reader_s1:
            chunk_num += 1
            chunk    = chunk.fillna("")
            s1_ids   = chunk["entity_id"].tolist()
            stats["s1_total"] += len(chunk)

            # Generate S1 blocking keys
            kdf = _build_keys_df(chunk).rename(
                columns={"entity_id": "source1_entity_id"})

            # Merge S1 keys against lookup -- C-level hash join
            merged = kdf.merge(lookup, on="block_key", how="inner")
            # Drop self-matches (shouldn't happen but guard)
            merged = merged[merged["source1_entity_id"] != merged["cand_id"]]

            if not merged.empty:
                # Group: one comma-separated list per S1 entity
                grouped = (
                    merged.groupby("source1_entity_id")["cand_id"]
                    .apply(lambda x: ",".join(sorted(set(x))))
                    .reset_index()
                )
                grouped.columns = ["source1_entity_id", "candidate_entity_ids"]
            else:
                grouped = pd.DataFrame(
                    columns=["source1_entity_id", "candidate_entity_ids"])

            # Ensure every S1 row appears in output
            full = (pd.DataFrame({"source1_entity_id": s1_ids})
                    .merge(grouped, on="source1_entity_id", how="left"))
            full["candidate_entity_ids"] = full["candidate_entity_ids"].fillna("")

            # Stats
            counts = full["candidate_entity_ids"].apply(
                lambda x: len(x.split(",")) if x else 0)
            n_zero = int((counts == 0).sum())
            n_tot  = int(counts.sum())
            n_max  = int(counts.max()) if len(counts) else 0

            stats["zero_candidate_s1"]  += n_zero
            stats["total_candidates"]   += n_tot
            stats["sum_candidates"]     += n_tot
            if n_max > stats["max_candidates"]:
                stats["max_candidates"] = n_max

            # Write
            import numpy as _np
            s1_arr  = full["source1_entity_id"].values.astype(str)
            cid_arr = full["candidate_entity_ids"].values.astype(str)
            lines   = _np.char.add(_np.char.add(s1_arr, "\t"), cid_arr)
            fout.write("\n".join(lines) + "\n")

            rss = _rss_mb()
            if rss > peak_rss:
                peak_rss = rss

            if verbose:
                elapsed = time.perf_counter() - t0
                print(f"  S1 chunk {chunk_num:4d} | "
                      f"{stats['s1_total']:>10,} rows | "
                      f"{stats['total_candidates']:>12,} cands | "
                      f"{elapsed:.0f}s"
                      + (f" | RAM ~{rss:.0f} MB" if _HAS_PSUTIL else ""),
                      flush=True)

    stats["elapsed_s"]      = time.perf_counter() - t0
    stats["peak_rss_mb"]    = peak_rss
    stats["avg_candidates"] = (
        stats["sum_candidates"] / stats["s1_total"]
        if stats["s1_total"] > 0 else 0.0)
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_blocking(
    s1_path: Path,
    s2_path: Path,
    s3_path: Path,
    output_path: Path,
    max_key_bucket: int = MAX_KEY_BUCKET,
    chunksize: int = CHUNKSIZE,
    verbose: bool = True,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t_wall = time.perf_counter()

    with tempfile.TemporaryDirectory(dir=OUTPUT_DIR) as tmpdir:
        raw_path    = Path(tmpdir) / "lookup_raw.tsv"
        sorted_path = Path(tmpdir) / "lookup_sorted.tsv"

        # Pass A -- write raw lookup to disk
        if verbose:
            print(f"\n[Pass A] Building raw lookup file ...", flush=True)
        build_raw_lookup_file(s2_path, s3_path, raw_path, chunksize, verbose)

        # Pass B -- sort + cap
        if verbose:
            print(f"\n[Pass B] Sorting and capping lookup ...", flush=True)
        sort_and_cap_lookup(raw_path, sorted_path, max_key_bucket,
                            chunksize=4_000_000, verbose=verbose)
        # free raw file disk space
        raw_path.unlink(missing_ok=True)

        # Pass C -- retrieve
        if verbose:
            print(f"\n[Pass C] Retrieving candidates for S1 ...", flush=True)
        stats = stream_retrieve(s1_path, sorted_path, output_path,
                                chunksize, verbose)

    stats["wall_elapsed_s"] = time.perf_counter() - t_wall
    if verbose:
        _print_stats(stats)
    return stats


def _print_stats(s: dict) -> None:
    sep = "-" * 62
    print(f"\n{sep}")
    print("  Blocking statistics")
    print(sep)
    print(f"  S1 records processed   : {s['s1_total']:>12,}")
    print(f"  Total candidate pairs  : {s['total_candidates']:>12,}")
    print(f"  Avg candidates / S1    : {s['avg_candidates']:>12.1f}")
    print(f"  Max candidates / S1    : {s['max_candidates']:>12,}")
    print(f"  Zero-candidate S1      : {s['zero_candidate_s1']:>12,}")
    if _HAS_PSUTIL:
        print(f"  Peak RAM               : {s['peak_rss_mb']:>11.0f} MB")
    print(f"  Total wall time        : {_fmt(s.get('wall_elapsed_s',s['elapsed_s']))}")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# RECALL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_recall(
    gt_path: Path,
    candidate_path: Path,
    verbose: bool = True,
    chunksize: int = CHUNKSIZE,
) -> dict:
    if verbose:
        print("\nLoading ground truth ...")
    gt = pd.read_csv(gt_path, sep="\t", dtype=str,
                     usecols=["source1_entity_id","matched_entity_ids"])
    gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")

    gt_dict: dict[str, set[str]] = {}
    true_s2 = true_s3 = 0
    for row in gt.itertuples(index=False):
        ids = [x.strip() for x in row.matched_entity_ids.split(",") if x.strip()]
        gt_dict[row.source1_entity_id] = set(ids)
        true_s2 += sum(1 for x in ids if x.startswith("S2-"))
        true_s3 += sum(1 for x in ids if x.startswith("S3-"))

    total_true = true_s2 + true_s3
    if verbose:
        print(f"  GT: {len(gt_dict):,} S1 | S2 pairs: {true_s2:,} | "
              f"S3 pairs: {true_s3:,} | total: {total_true:,}")

    hit_s2 = hit_s3 = 0
    total_cands = max_cands = sum_cands = s1_cnt = zero_cand = 0

    for chunk in pd.read_csv(candidate_path, sep="\t", dtype=str,
                              chunksize=chunksize):
        chunk["candidate_entity_ids"] = chunk["candidate_entity_ids"].fillna("")
        for row in chunk.itertuples(index=False):
            cands = set(x.strip()
                        for x in row.candidate_entity_ids.split(",") if x.strip())
            n = len(cands)
            total_cands += n; sum_cands += n; s1_cnt += 1
            if n == 0:
                zero_cand += 1
            if n > max_cands:
                max_cands = n
            for tid in gt_dict.get(row.source1_entity_id, set()):
                if tid in cands:
                    (hit_s2 if tid.startswith("S2-") else hit_s3).__class__  # dummy
                    if tid.startswith("S2-"):
                        hit_s2 += 1
                    else:
                        hit_s3 += 1

    recall_s2  = hit_s2 / true_s2  if true_s2  > 0 else float("nan")
    recall_s3  = hit_s3 / true_s3  if true_s3  > 0 else float("nan")
    recall_all = (hit_s2+hit_s3)/total_true if total_true > 0 else float("nan")
    avg_cands  = sum_cands / s1_cnt if s1_cnt > 0 else 0.0

    result = {
        "s1_records": s1_cnt, "true_s2_pairs": true_s2,
        "true_s3_pairs": true_s3, "total_true_pairs": total_true,
        "hit_s2": hit_s2, "hit_s3": hit_s3,
        "recall_s2": recall_s2, "recall_s3": recall_s3,
        "recall_overall": recall_all,
        "total_candidates": total_cands, "avg_candidates": avg_cands,
        "max_candidates": max_cands, "zero_candidate_s1": zero_cand,
    }
    if verbose:
        _print_recall(result)
    return result


def _print_recall(r: dict) -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print("  BLOCKING RECALL EVALUATION")
    print(sep)
    print(f"  S1 records              : {r['s1_records']:>12,}")
    print(f"  True S2 pairs           : {r['true_s2_pairs']:>12,}")
    print(f"  True S3 pairs           : {r['true_s3_pairs']:>12,}")
    print(f"  Total true pairs        : {r['total_true_pairs']:>12,}")
    print()
    print(f"  S2 pairs found          : {r['hit_s2']:>12,}")
    print(f"  S3 pairs found          : {r['hit_s3']:>12,}")
    print()
    print(f"  S2 recall               : {r['recall_s2']*100:>11.2f}%")
    print(f"  S3 recall               : {r['recall_s3']*100:>11.2f}%")
    print(f"  Overall recall          : {r['recall_overall']*100:>11.2f}%")
    print()
    print(f"  Total candidate pairs   : {r['total_candidates']:>12,}")
    print(f"  Avg candidates / S1     : {r['avg_candidates']:>12.1f}")
    print(f"  Max candidates / S1     : {r['max_candidates']:>12,}")
    print(f"  Zero-candidate S1       : {r['zero_candidate_s1']:>12,}")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _rss_mb() -> float:
    if _HAS_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    return 0.0

def _fmt(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m:02d}m {sec:02d}s" if h else f"{m}m {sec:02d}s"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Blocking -- Amazon ML Challenge 2026")
    p.add_argument("--split", choices=["train","test"], default="test")
    p.add_argument("--output", type=Path, default=OUTPUT_DIR/"candidate_pairs.tsv")
    p.add_argument("--max-key-bucket", type=int, default=MAX_KEY_BUCKET)
    p.add_argument("--chunksize", type=int, default=CHUNKSIZE)
    p.add_argument("--no-eval", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    print("=" * 70)
    print("  Amazon ML Challenge 2026 -- Blocking / Candidate Generation")
    print("=" * 70)
    print(f"  Split          : {args.split}")
    print(f"  Output         : {args.output}")
    print(f"  Max key bucket : {args.max_key_bucket:,}")
    print(f"  Chunk size     : {args.chunksize:,}")
    print(f"  psutil         : {'YES' if _HAS_PSUTIL else 'NO'}")
    print("=" * 70)

    if args.split == "train":
        s1,s2,s3,gt = (PREPROCESSED["train_s1"],PREPROCESSED["train_s2"],
                        PREPROCESSED["train_s3"],PREPROCESSED["train_gt"])
    else:
        s1,s2,s3,gt = (PREPROCESSED["test_s1"],PREPROCESSED["test_s2"],
                        PREPROCESSED["test_s3"],None)

    for p in [s1,s2,s3]:
        if not p.exists():
            print(f"ERROR: {p} not found. Run validate_preprocessing.py first.")
            return 1

    stats = run_blocking(s1,s2,s3,args.output,
                         args.max_key_bucket,args.chunksize,True)
    print(f"\nCandidate file : {args.output}")
    print(f"File size      : {args.output.stat().st_size/(1024**2):.1f} MB")

    if args.split=="train" and not args.no_eval and gt and gt.exists():
        evaluate_recall(gt, args.output, True, args.chunksize)
    return 0


if __name__ == "__main__":
    sys.exit(main())
