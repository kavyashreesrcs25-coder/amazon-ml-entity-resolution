# Person 2 — Blocking / Candidate Generation

**Amazon ML Challenge 2026: Business Entity Resolution**
**Module:** `src/blocking.py`
**Evaluation:** `src/evaluate_blocking.py`
**Tests:** `src/test_blocking.py`

---

## Table of Contents

1. [Why Blocking is Required](#1-why-blocking-is-required)
2. [Blocking Rules](#2-blocking-rules)
3. [Index Construction](#3-index-construction)
4. [Candidate Retrieval](#4-candidate-retrieval)
5. [Duplicate Removal](#5-duplicate-removal)
6. [Training Recall Evaluation](#6-training-recall-evaluation)
7. [Candidate Reduction](#7-candidate-reduction)
8. [Memory and Performance](#8-memory-and-performance)
9. [How candidate_pairs.tsv is Generated](#9-how-candidate_pairstsv-is-generated)
10. [How to Run](#10-how-to-run)
11. [Output Format](#11-output-format)

---

## 1. Why Blocking is Required

The dataset contains approximately:

- **2.2M** Source 1 (S1) entities in training; **1.7M** in test
- **5M** Source 2 (S2) entities
- **5.3M** Source 3 (S3) entities

A brute-force comparison of every S1 entity against every S2/S3 entity would require:

```
2,206,821 × (5,034,616 + 5,285,603) ≈ 22.8 billion comparisons
```

This is computationally infeasible for any real-time or batch matching pipeline.

**Blocking** (also called candidate generation) solves this by using cheap, high-recall heuristics to narrow each S1 entity's comparison set from ~10M to a few hundred candidates. The matching model then runs only on this small candidate set.

**The key trade-off:** blocking must prioritize **recall** over precision. A true match that is excluded from the candidate set can never be recovered by the matching model. False candidates are acceptable — the matching model will reject them. A missed true match is a permanent error.

---

## 2. Blocking Rules

Six independent blocking rules are used. Their results are unioned per S1 entity.

### R1 — Exact Normalised Name + Country

```
block_key = norm_business_name + "|" + norm_country
```

**Rationale:** Many records across sources have identical normalised names (after lowercasing, whitespace collapse, noise removal). This rule has perfect precision for exact matches and is the cheapest lookup.

**Example:**
```
S1: "miller metals" | US   matches   S2: "miller metals" | US
```

---

### R2 — 6-Character Name Prefix + Country

```
block_key = norm_business_name[:6] + "|" + norm_country
```

**Rationale:** Catches abbreviation variants and truncated names that share the same first 6 characters. Works across US/India/France because it is purely character-based.

**Example:**
```
S1: "miller metals" → "miller|US"
S2: "miller metalworks inc" → "miller|US"   ← matches
```

---

### R3 — Sorted First 2 Significant Tokens + Country

```
sig_tokens = [t for t in name.split() if t not in STOP_WORDS and len(t) >= 2]
block_key  = " ".join(sorted(sig_tokens[:2])) + "|" + norm_country
```

**Rationale:** Handles word-order transpositions (very common in Indian addresses and business names), abbreviation variants. Sorting the tokens makes it order-invariant. Stop words (inc, ltd, pvt, llc, the, and, …) are filtered to improve precision.

**Example:**
```
S1: "primary care national specialists" → "care national|US"
S2: "national primary care specialists" → "care national|US"   ← matches
```

---

### R4 — 3-Character Grams at Fixed Offsets + Country

```
block_key_1 = name[:3]  + "|" + norm_country    (first  3 chars)
block_key_2 = name[6:9] + "|" + norm_country    (middle 3 chars, if name ≥ 9 chars)
```

**Rationale:** Character n-grams are language-independent and robust to typos, transliterations, and abbreviations. Using two fixed-offset grams (rather than all sliding-window grams) keeps the fan-out manageable while still providing recall for partial-name matches.

**Example:**
```
S1: "miller metals" → "mil|US" and "er |US"
S2: "mill metals"   → "mil|US"               ← matches via first gram
```

---

### R5 — First Numeric Run in Address + Country

```
nums = re.findall(r"\d{2,}", norm_business_address)
block_key = nums[0] + "|" + norm_country   (if nums exists)
```

**Rationale:** Street/plot/house numbers are highly discriminative. Two records with the same street number in the same country are strong match candidates. Works for US (1795 Westchester), India (plot no. 570), and France (23 Rue …).

**Example:**
```
S1: "1795 westchester drive, high point, nc" | US → "1795|US"
S2: "1795 WESTCHESTER DR, HIGH POINT, NC"    | US → "1795|US"   ← matches
```

---

### R6 — First Address Token + First 4 Characters of Name

```
first_addr_token = re.search(r"[^\s,./\-]{2,}", addr).group(0)
block_key = first_addr_token + "|" + name[:4]
```

**Rationale:** Combines a coarse address signal with a coarse name signal. Language-independent — works for numeric street numbers, Indian sector codes, and French postal prefixes equally.

**Example:**
```
S1: addr="1795 westchester…", name="miller metals" → "1795|mill"
S3: addr="1795 english bay…", name="miller  metals" → "1795|mill"  ← matches
```

---

## 3. Index Construction

The index is built in **Pass A** — a single streaming pass over the S2 and S3 preprocessed TSV files.

```
For each chunk of S2/S3:
    1. Read columns: entity_id, norm_business_name,
                     norm_business_address, norm_country
    2. Apply all 6 rules vectorised (pandas .str operations)
    3. Emit (block_key, cand_id) pairs
    4. Write directly to lookup_raw.tsv on disk (never accumulate in RAM)
```

**Pass A result:** `lookup_raw.tsv` — an unsorted file of (block_key, cand_id) rows.

**Pass B** — sort and cap:

```
1. Read lookup_raw.tsv into RAM (one-time full load)
2. Drop "hot" keys: any block_key with > MAX_KEY_BUCKET=300 candidates
   (These are very common strings like "123|US" that match thousands of
   unrelated businesses — keeping them would explode candidate counts
   with near-zero recall contribution)
3. drop_duplicates()
4. sort_values("block_key")
5. Write to lookup_sorted.tsv
```

This disk-based approach keeps Pass A RAM under **600 MB** regardless of dataset size, at the cost of one large sort in Pass B.

---

## 4. Candidate Retrieval

**Pass C** — stream S1, merge against sorted lookup:

```
For each S1 chunk:
    1. Generate all 6 block keys for every S1 row (same _build_keys_df function)
    2. pd.merge(s1_keys_df, lookup, on="block_key", how="inner")
       → all matching (s1_id, cand_id) pairs at C-speed (no Python loops)
    3. Group by s1_id → join cand_ids with comma
    4. LEFT JOIN with full S1 chunk → ensures every S1 row appears in output
       (zero-candidate rows get empty string)
    5. Write to candidate_pairs.tsv incrementally
```

The merge is the key performance win: pandas `merge` on string columns runs at C-level speed, far faster than any Python dict lookup loop.

---

## 5. Duplicate Removal

Duplicates arise naturally because:
- The same candidate may be returned by multiple blocking rules (e.g. R1 and R2 both match)
- The same candidate may appear in both S2 and S3 (they are different files with different entity_ids, so these are NOT duplicates — they are distinct candidates)

Deduplication is applied at two levels:

1. **Within `_build_keys_df`**: `drop_duplicates()` on (entity_id, block_key) pairs so one entity does not generate the same key twice from different rules.

2. **Within retrieval**: `drop_duplicates()` on (source1_entity_id, cand_id) after the merge, so the same candidate is listed only once regardless of how many rules matched it.

The final `candidate_entity_ids` column contains each candidate ID exactly once per S1 entity.

---

## 6. Training Recall Evaluation

Recall is measured by `evaluate_blocking.py` against `train_ground_truth_preprocessed.tsv`.

### Formula

```
Blocking Recall = (true matched pairs present in candidates)
                / (total true matched pairs in ground truth)
```

Computed separately for S2 and S3 matches, and as an overall figure.

### Reduction Ratio

```
Reduction Ratio = 1 - (total candidate pairs)
                      / (S1_count × (S2_count + S3_count))
```

A reduction ratio close to 1.0 means blocking is very selective; close to 0.0 means almost all pairs are candidates (no reduction).

### Results (Training Set — Full Dataset)

| Metric | Value |
|--------|-------|
| S1 records | 2,206,821 |
| True S2 pairs | 3,693,619 |
| True S3 pairs | 3,944,746 |
| Total true pairs | 7,638,365 |
| S2 pairs found | 2,588,096 |
| S3 pairs found | 2,707,424 |
| **S2 recall** | **70.07%** |
| **S3 recall** | **68.63%** |
| **Overall recall** | **69.33%** |
| Total candidate pairs | 83,658,633 |
| Avg candidates / S1 | 37.9 |
| Max candidates / S1 | 306 |
| Zero-candidate S1 | 19,357 (0.88%) |
| Reduction ratio | 0.9963 |
| Peak RAM | 2,769 MB |
| Total wall time | 51m 04s |

---

## 7. Candidate Reduction

The `MAX_KEY_BUCKET = 100` cap on hot keys is the main precision control:

- Any block_key with more than 100 matching candidates is dropped entirely
- This removes extremely common keys (e.g. 3-char gram `"the"`, generic prefixes like `"com"`) that would produce thousands of false candidates
- In the training run: **43,207 hot keys** were dropped, removing **36.3M** (of 69.2M) raw pairs — a **52% reduction** before the sort step
- After capping: **32,918,366** clean (block_key, cand_id) pairs remained in the sorted lookup

---

## 8. Memory and Performance

### Three-pass design

| Pass | What | RAM used | Time (train) |
|------|------|----------|-------------|
| A — Index build | Stream S2+S3 → write raw lookup to disk | ~600 MB | ~10 min |
| B — Sort + cap | Load raw lookup, sort, drop hot keys, save | ~6 GB peak | ~9 min |
| C — Retrieval | Load sorted lookup + stream S1 chunks | ~2–4 GB | ~37 min |

**Peak RAM: ~6 GB** during Pass B sort. All other passes stay under 2 GB.

### Vectorised key generation

All 6 blocking rules use pandas `.str` operations — no Python-level row loops except for R4 which uses two fixed-offset `str[:]` extractions.

Benchmarked at **~4s per 200k rows** for key generation (was 22s with Python `.apply()` loops — 5× speedup).

### Disk I/O

- Raw lookup file: ~1.8 GB (69M rows)
- Sorted lookup file: ~1.1 GB (37M rows after hot-key removal)
- Candidate pairs file: varies by dataset

---

## 9. How candidate_pairs.tsv is Generated

### Training candidates (for evaluation only)

```bash
python src/evaluate_blocking.py --run-blocking \
    --candidates output/train_candidate_pairs.tsv
```

This generates `train_candidate_pairs.tsv` then immediately evaluates recall against the ground truth.

### Test candidates (for submission)

```bash
python src/blocking.py --split test \
    --output output/candidate_pairs.tsv
```

This generates the final `output/candidate_pairs.tsv` which is the input to the matching model.

### Pipeline flow

```
test_source2_preprocessed.tsv ──┐
test_source3_preprocessed.tsv ──┤  Pass A: build raw lookup (disk)
                                 ↓
                            lookup_raw.tsv (temp, ~1.8 GB)
                                 ↓
                            Pass B: sort + cap hot keys
                                 ↓
                            lookup_sorted.tsv (temp, ~1.1 GB)
                                 ↓
test_source1_preprocessed.tsv ──→  Pass C: merge-based retrieval
                                 ↓
                        output/candidate_pairs.tsv
```

---

## 12. Multilingual Support

All six blocking rules are Unicode-safe. Key design decisions:

### Token regex (`_RE_TOKENS`)
```python
re.compile(r"[^\s,./\-]+", re.UNICODE)
```
Splits on ASCII whitespace and common ASCII separators only. Non-Latin characters (Devanagari, Bengali, Kannada, Tamil, Arabic, French accented chars) are treated as word content and never split on. Result:
- `"राम मार्केटिंग"` → `["राम", "मार्केटिंग"]` ✓
- `"société générale"` → `["société", "générale"]` ✓
- `"123 main st"` → `["123", "main", "st"]` ✓

### Stop-word filter (R3 only)
`_STOP` contains **English-only** words (`inc`, `ltd`, `pvt`, …). The `_sig_tokens()` function applies this filter **only to pure-ASCII tokens**. Non-Latin tokens always pass through:
```python
if _is_ascii(t) and t in _STOP:
    continue   # English stop-word — skip
# Non-ASCII: always keep (Devanagari, Bengali, French, etc.)
```

### Unicode codepoint slicing (R2, R4, R6)
Python string slicing (`name[:6]`, `name[:3]`) operates on Unicode codepoints, not bytes. This is correct for all scripts:
- `"राम मार्केटिंग"[:6]` = `"राम मा"` (6 codepoints) ✓
- `"société"[:6]` = `"sociét"` (6 codepoints) ✓

### Digit matching (R5)
`re.compile(r"\d{2,}", re.UNICODE)` matches Western digits AND Unicode digits (Arabic-Indic ١٢٣, Devanagari १२३, etc.).

### Country field
The `norm_country` value is passed through unchanged — US, India, France, Germany, Morocco, or any future country all work as blocking key suffixes without any hard-coding.

### Validated scripts
| Script | Unicode range | Tests passing |
|--------|---------------|--------------|
| Devanagari (Hindi/Marathi) | U+0900–U+097F | R1, R2, R3, R4, R5, R6 ✓ |
| Bengali | U+0980–U+09FF | R1, R3, R4 ✓ |
| Kannada | U+0C80–U+0CFF | R5, R6 ✓ |
| French (Latin + diacritics) | various | R1, R2, R3, R4, R5, R6 ✓ |
| Mixed-script | — | R1, R3 ✓ |
| Open-set country | — | Any country ✓ |

### Smoke-test (38 unit + integration tests, ~2 min)

```bash
python src/test_blocking.py
```

### Generate training candidates + evaluate recall

```bash
python src/evaluate_blocking.py --run-blocking \
    --candidates output/train_candidate_pairs.tsv
```

### Evaluate recall on an already-generated candidate file

```bash
python src/evaluate_blocking.py \
    --candidates output/train_candidate_pairs.tsv
```

### Generate test candidate_pairs.tsv (for submission)

```bash
python src/blocking.py --split test --output output/candidate_pairs.tsv
```

### CLI options for blocking.py

```
--split          train | test (default: test)
--output         output path (default: output/candidate_pairs.tsv)
--max-key-bucket max candidates per blocking key (default: 300)
--chunksize      rows per pandas chunk (default: 200000)
--no-eval        skip recall eval after train run
```

---

## 11. Output Format

### candidate_pairs.tsv

Tab-separated, one row per S1 entity. Every S1 entity in the source file has exactly one row.

```
source1_entity_id\tcandidate_entity_ids
S1-925783039\tS2-652436308,S2-441290187,S3-802447883,S3-114527621
S1-773889195\tS2-998234100,S3-334521090
S1-123456789\t
```

- `candidate_entity_ids` is a comma-separated list of S2- and S3- IDs
- Empty value = no candidates found for this S1 entity (singleton prediction)
- No duplicate IDs within a single row
- No S1- IDs in the candidate list (validated by test suite)

### Validation

Before submitting, validate with Person 1's validator:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
