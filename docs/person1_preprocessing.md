# Person 1 — Data Preprocessing

**Amazon ML Challenge 2026: Business Entity Resolution**
**Module:** `src/preprocessing.py`
**Test suite:** `src/test_preprocessing.py`

---

## Table of Contents

1. [Overview](#1-overview)
2. [Files Created](#2-files-created)
3. [How to Run](#3-how-to-run)
4. [Design Principles](#4-design-principles)
5. [Preprocessing Steps — Business Name](#5-preprocessing-steps--business-name)
6. [Preprocessing Steps — Business Address](#6-preprocessing-steps--business-address)
7. [Preprocessing Steps — Country](#7-preprocessing-steps--country)
8. [Missing Value Handling](#8-missing-value-handling)
9. [Multilingual Data Handling](#9-multilingual-data-handling)
10. [Output Columns](#10-output-columns)
11. [Before / After Examples](#11-before--after-examples)
12. [Test Suite Coverage](#12-test-suite-coverage)
13. [Important Decisions and Rationale](#13-important-decisions-and-rationale)

---

## 1. Overview

Three independent databases (Source 1, Source 2, Source 3) contain records for the same real-world businesses but with widely varying formatting: uppercase vs title-case, abbreviated vs full legal names, reordered address components, non-Latin scripts, literal NULL placeholders, and hash-prefixed house numbers.

Before any blocking or matching can happen, every record needs to be brought to a consistent surface form so that "MILLER METALS" and "Miller  Metals" look the same to a string-similarity function, while "राम मार्केटिंग" and a Bengali transliteration of the same name remain distinct and intact.

This module provides that normalization layer. It:

- adds four new columns (`norm_business_name`, `norm_business_address`, `norm_country`, `address_missing`) to each source DataFrame
- never modifies or deletes any original column or row
- works correctly for US, India, and France data (and any future country)
- requires only `pandas` and Python's standard library

---

## 2. Files Created

```
Amazon/
├── src/
│   ├── preprocessing.py        ← normalization module (this work)
│   └── test_preprocessing.py   ← verification test suite (this work)
├── docs/
│   └── person1_preprocessing.md  ← this document
└── output/                     ← empty; preprocessed TSVs saved here if needed
```

The original dataset files are never touched:

```
[dataset root]/
├── train/
│   ├── train_source1.tsv       ← read-only
│   ├── train_source2.tsv       ← read-only
│   ├── train_source3.tsv       ← read-only
│   └── train_ground_truth.tsv  ← read-only
└── test/
    ├── test_source1.tsv        ← read-only
    ├── test_source2.tsv        ← read-only
    └── test_source3.tsv        ← read-only
```

---

## 3. How to Run

### Run the full test suite (unit + integration against real files)

```bash
# From the project root
python src/test_preprocessing.py
```

Expected output ends with:

```
======================================================================
  RESULTS:  NNN passed  /  0 failed  /  NNN total

  All tests PASSED.
```

### Use the module in your own code

```python
from src.preprocessing import load_and_preprocess, load_and_preprocess_all_sources

# Load and preprocess a single file
df = load_and_preprocess("train_source1")
print(df.columns.tolist())
# ['entity_id', 'business_name', 'business_address', 'country',
#  'norm_business_name', 'norm_business_address', 'norm_country', 'address_missing']

# Load all six source files at once
sources = load_and_preprocess_all_sources(verbose=True)
train1 = sources["train_source1"]
test2  = sources["test_source2"]
```

### Save preprocessed files to disk

```python
from src.preprocessing import load_and_preprocess, save_preprocessed
from pathlib import Path

df = load_and_preprocess("train_source1")
save_preprocessed(df, "train_source1", Path("output/"))
# writes: output/train_source1_preprocessed.tsv
```

### Memory-efficient chunked loading (for large source2/source3 files)

```python
df = load_and_preprocess("train_source2", chunksize=200_000)
```

### Apply normalization to an already-loaded DataFrame

```python
from src.preprocessing import preprocess_source_df
df = preprocess_source_df(df)   # adds norm_* columns in-place
```

### Run the built-in smoke-test for the core functions

```bash
python src/preprocessing.py
```

---

## 4. Design Principles

| Principle | Implementation |
|-----------|---------------|
| Non-destructive | Original columns are never overwritten. Normalized values go into new `norm_*` columns. |
| Row-preserving | No rows are ever dropped, including rows with empty addresses or NULL placeholders. |
| ID-safe | `entity_id` is read but never modified. |
| Multilingual-safe | Non-Latin scripts pass through unchanged. Only ASCII noise characters and formatting artifacts are cleaned. |
| No external data | stdlib + pandas + regex only. No internet lookups, no translation APIs, no geocoding. |
| Open country set | The country field is not assumed to be `{US, India}`. France (test-only) and any future country pass through correctly. |

---

## 5. Preprocessing Steps — Business Name

Applied by `normalize_business_name(raw)` in this order:

### Step 1 — Guard against None / NaN
If the value is missing, return `""` immediately. This prevents downstream crashes on null cells.

### Step 2 — NFC Unicode normalization
```python
unicodedata.normalize("NFC", text)
```
NFC (Canonical Decomposition, Canonical Composition) makes canonically equivalent Unicode characters compare equal. For example, `é` can be encoded as a single codepoint (U+00E9) or as `e` + combining acute accent (U+0065 U+0301). NFC ensures they always become the single-codepoint form.

This is the only Unicode normalization that is safe for multilingual text. NFKC/NFKD would decompose ligatures and strip combining characters, corrupting Devanagari vowel marks, Bengali conjuncts, and French diacritics.

### Step 3 — Remove literal NULL placeholders
```
"null"  "<null>"  "<NULL>"  "(null)"  "#null#"
```
These appear in Source 3 business names (rarely). They are removed, and the surrounding whitespace is collapsed.

### Step 4 — Strip leading / trailing noise characters
```
"-- Holloway Peak Inc"   →   "Holloway Peak Inc"
"*** Sai Tech Pvt Ltd"   →   "Sai Tech Pvt Ltd"
"<< Team Ecole"          →   "Team Ecole"
"@@ Bright Solutions"    →   "Bright Solutions"
```
Characters stripped from the edges: `- * # @ ! | ~ ^ + = < > ` and whitespace.
Only leading/trailing positions are targeted. A hyphen inside a name (e.g. `La Teste-de-Buch`) is kept.

### Step 5 — Replace pipe separators with a space
```
"SHIVSHAKTI | www.shivshakti.com"   →   "SHIVSHAKTI www.shivshakti.com"
"ABC Company | xyz.com"              →   "ABC Company xyz.com"
```
Pipes appeared in Source 2 names as a visual separator between the business name and its website. The website fragment is kept (it can help matching) but the pipe is replaced with a space so tokens don't merge.

### Step 6 — Normalize separators
- Repeated dots (`...`) → single dot (`.`)
- Repeated commas/semicolons (`,,`) → single comma (`,`)
- Empty comma segments left by NULL removal (`"City, , State"`) → `"City, State"`

### Step 7 — Collapse whitespace
All runs of horizontal whitespace (space, tab, non-breaking space, Unicode space variants) are collapsed to a single ASCII space. Leading and trailing space is removed.

### Step 8 — Lowercase
The entire string is lowercased. This makes `"MILLER METALS"`, `"Miller Metals"`, and `"miller metals"` identical to a string-comparison function.

Lowercase is applied **after** all noise removal, so the original `business_name` column still preserves the original case for audit purposes.

---

## 6. Preprocessing Steps — Business Address

Applied by `normalize_business_address(raw)` in this order:

### Step 1 — Guard against None / NaN
Return `""` immediately.

### Step 2 — Detect effectively-empty addresses
If the string is whitespace-only after stripping, return `""`.

### Step 3 — NFC Unicode normalization
Same as for names (see above).

### Step 4 — Remove literal NULL placeholders
```
"New Delhi, null, A-68, दिल्ली"      →   "New Delhi, A-68, दिल्ली"
"G.T. Karnal Road, null, A-68"       →   "G.T. Karnal Road, A-68"
"City, <NULL>, State"                →   "City, State"
```
Source 3 frequently contains the literal string `null` or `<NULL>` embedded inside an otherwise valid address. Removing these and collapsing the resulting empty comma segment avoids confusing a matching model that might treat "null" as a city name.

### Step 5 — Remove hash prefixes from house numbers
```
"###56 B Revenue Housing Society"    →   "56 B Revenue Housing Society"
"#18009 THIRD AVE, ARLINGTON, WA"    →   "18009 THIRD AVE, ARLINGTON, WA"
```
Source 2 uses `#` / `##` / `###` before street/plot numbers. This is formatting noise; the number itself is the identity-relevant part.

### Step 6 — Strip leading / trailing noise characters
Same character set as names. Removes stray dashes or hashes at the very start or end of the address string.

### Step 7 — Normalize separators
Same collapse logic as names. Also cleans up empty comma segments that appear after NULL removal.

### Step 8 — Collapse whitespace

### Step 9 — Lowercase

### What is deliberately NOT done to addresses

| Not done | Reason |
|----------|--------|
| Address component reordering | Source 2 sometimes reverses component order (`WA, Arlington, 18009 3rd Ave`). Normalizing order requires reliable parsing of US/India/France addresses simultaneously — error-prone. Downstream models handle this with token-level features. |
| Abbreviation expansion (`Rd` → `Road`) | Would introduce errors for French street abbreviations and Indian landmark-based addresses. String similarity on character n-grams handles abbreviations naturally. |
| Geocoding / coordinate lookup | Prohibited by challenge rules and not needed. |
| Transliteration | Non-Latin place names (`ಕರ್ನಾಟಕ`, `महाराष्ट्र`) are preserved — they are the actual identity of the location. |

---

## 7. Preprocessing Steps — Country

Applied by `normalize_country(raw)`:

1. Guard against None / NaN → return `""`
2. Strip surrounding whitespace
3. NFC normalization (handles future countries with diacritics, e.g. `Réunion`)
4. Collapse internal whitespace runs

The value is **not lowercased**. Country labels in this dataset are already clean (`US`, `India`, `France`). Preserving case avoids surprises if downstream code does exact-string matching on country labels.

---

## 8. Missing Value Handling

### In source files

The `business_address` column has missing values in Source 2 and Source 3:

| File | Missing addresses |
|------|------------------|
| train_source1 | 0 (0%) |
| train_source2 | ~168,967 (3.4%) |
| train_source3 | ~175,916 (3.3%) |
| test_source1  | 0 (0%) |
| test_source2  | ~129,408 (2.7%) |
| test_source3  | ~136,098 (2.7%) |

All of these rows are kept. Missing addresses are represented as `""` in `norm_business_address` and `True` in `address_missing`.

### The `address_missing` flag

`is_address_missing(raw)` returns `True` when:
- The raw value is `NaN` / `None`
- The raw value is an empty string
- The raw value is whitespace-only
- The raw value contains only NULL placeholder strings (`null`, `<NULL>`, `(null)`) and nothing else

The flag is consistent with `norm_business_address`: whenever `address_missing` is `True`, `norm_business_address` is `""`, and vice versa. This invariant is verified in the test suite.

### In ground truth

The `matched_entity_ids` column is empty for **123,247 Source 1 entities** (5.6%). These are legitimate singletons — businesses that genuinely have no match in Source 2 or Source 3. They are loaded with `fillna("")` so the column is always a string, never NaN. Downstream code can detect singletons with `row["matched_entity_ids"] == ""`.

---

## 9. Multilingual Data Handling

The dataset contains business names and addresses in multiple scripts:

| Script | Range | Example | Source |
|--------|-------|---------|--------|
| Devanagari (Hindi/Marathi) | U+0900–U+097F | `राम मार्केटिंग प्राइवेट लिमिटेड` | Source 2 India |
| Bengali | U+0980–U+09FF | `ইউনিভার্সাল সিস্টেমস` | Source 2 India |
| Kannada | U+0C80–U+0CFF | `ಕರ್ನಾಟಕ` | Source 3 India |
| Tamil | U+0B80–U+0BFF | appears in addresses | Source 3 India |
| Latin with diacritics | various | `é à ç ü ñ` | Source 3 France |

### Safety rules applied

1. **NFC only** — the only Unicode transformation applied. It is script-neutral and lossless: it rearranges codepoints into canonical composed form without removing any characters.

2. **Lowercase is ASCII-only in effect** — Python's `str.lower()` lowercases the ASCII range correctly and is safe for Devanagari/Bengali/Kannada (which have no case distinction). French accented characters (`É → é`, `À → à`) are also lowercased correctly by Python's Unicode-aware `lower()`.

3. **Regex patterns target only ASCII noise** — all regexes (`_RE_LEADING_NOISE`, `_RE_HASH_NUMBER`, etc.) match ASCII punctuation characters only. They cannot accidentally match Devanagari vowel signs or French diacritics.

4. **Character preservation is verified** — the test suite counts non-Latin characters in the raw column and verifies the same count appears in the normalized column for Devanagari, Bengali, Kannada, and Tamil scripts.

5. **No transliteration** — transliterating `राम मार्केटिंग` to `Ram Marketing` would make the normalized name unverifiable against the original and would lose information for the 10% of records where the same entity is written in different scripts across sources (Source 1 English, Source 2 Devanagari). The downstream matching model sees both and must handle cross-script matching directly.

---

## 10. Output Columns

After `preprocess_source_df(df)` runs, each source DataFrame has 8 columns:

| Column | Type | Description | Modified? |
|--------|------|-------------|-----------|
| `entity_id` | str | Original unique ID (`S1-`, `S2-`, `S3-` prefix) | Never |
| `business_name` | str | Original raw name | Never |
| `business_address` | str / NaN | Original raw address | Never |
| `country` | str | Original country label | Never |
| `norm_business_name` | str | Normalized name (lowercase, noise removed) | Added |
| `norm_business_address` | str | Normalized address (lowercase, NULL removed) | Added |
| `norm_country` | str | Normalized country (whitespace stripped) | Added |
| `address_missing` | bool | True when address is absent/null | Added |

The ground truth DataFrame (`train_ground_truth`) is loaded as-is with `matched_entity_ids` NaN-filled to `""`. No `norm_*` columns are added to it.

---

## 11. Before / After Examples

### Business names

| Raw (original) | `norm_business_name` | Noise removed |
|---|---|---|
| `-- Holloway Peak Inc Seafood` | `holloway peak inc seafood` | Leading `--` |
| `*** Sai Tech Private Limited` | `sai tech private limited` | Leading `***` |
| `MILLER METALS` | `miller metals` | ALLCAPS |
| `Miller  Metals` | `miller metals` | Double space |
| `FOUNDATION EXCEL AGENCY PRIVATE  LIMITED` | `foundation excel agency private limited` | ALLCAPS + double space |
| `राम मार्केटिंग प्राइवेट लिमिटेड` | `राम मार्केटिंग प्राइवेट लिमिटेड` | Nothing (preserved) |
| `ইউনিভার্সাল সিস্টেমস প্রাইভেট লিমিটেড` | `ইউনিভার্সাল সিস্টেমস প্রাইভেট লিমিটেড` | Nothing (preserved) |
| `ABC Company \| www.abc.com` | `abc company www.abc.com` | Pipe separator |
| `Pvt. EFS Print Ventures Ltd.` | `pvt. efs print ventures ltd.` | Nothing (legal suffix kept) |
| `<< Team Ecole` | `team ecole` | Leading `<<` |
| `Marina Ecole France Sarl` | `marina ecole france sarl` | ALLCAPS of source |
| `Société Générale` | `société générale` | Accents preserved, lowercased |
| `Fractales Amis Groupe S.A.S` | `fractales amis groupe s.a.s` | Legal suffix kept |
| `None` / NaN | `""` | Safe empty |

### Business addresses

| Raw (original) | `norm_business_address` | Noise removed |
|---|---|---|
| `###56 B REVENUE HOUSING SOCIETY, KOLHAPUR, Maharashtra` | `56 b revenue housing society, kolhapur, maharashtra` | `###` prefix, ALLCAPS |
| `New Delhi, null, A-68, दिल्ली` | `new delhi, a-68, दिल्ली` | `null` literal + empty segment |
| `G.T. Karnal Road, null, A-68, दिल्ली` | `g.t. karnal road, a-68, दिल्ली` | `null` literal + empty segment |
| `City, <NULL>, State` | `city, state` | `<NULL>` literal + empty segment |
| `WA, Arlington, 18009 3rd Avenue` | `wa, arlington, 18009 3rd avenue` | ALLCAPS; order preserved |
| `Door No 183, Jayanagar, Bengaluru, ಕರ್ನಾಟಕ` | `door no 183, jayanagar, bengaluru, ಕರ್ನಾಟಕ` | Kannada preserved |
| `H.no 910 A 3503, Mumbai, महाराष्ट्र` | `h.no 910 a 3503, mumbai, महाराष्ट्र` | Devanagari preserved |
| `175 Boulevard du Président Franklin Roosevelt, Bordeaux` | `175 boulevard du président franklin roosevelt, bordeaux` | French accent preserved |
| `63 R. DE DIEPPE, LILLE, Hauts-de-France` | `63 r. de dieppe, lille, hauts-de-france` | ALLCAPS, hyphenated region preserved |
| `018009 Third Ave, Arlington, Washington` | `018009 third ave, arlington, washington` | Leading zero kept |
| `<NULL>` | `""` + `address_missing=True` | Entire value is NULL |
| `""` / None | `""` + `address_missing=True` | Safe empty |

### Country

| Raw | `norm_country` | Note |
|-----|----------------|------|
| `US` | `US` | Unchanged |
| `India` | `India` | Unchanged |
| `France` | `France` | Test-set country, passes through |
| `" India "` | `India` | Whitespace stripped |
| `None` | `""` | Safe empty |

---

## 12. Test Suite Coverage

`src/test_preprocessing.py` runs two layers:

### Layer 1 — Unit tests (101 tests, no file I/O)

| Category | Tests | What is verified |
|----------|-------|-----------------|
| Name: noise prefix removal | 5 | `--`, `***`, `###`, `<<`, `@@` stripped from start |
| Name: whitespace collapse | 4 | Double/triple spaces → single |
| Name: ALLCAPS → lowercase | 3 | Source 2 ALLCAPS lowercased |
| Name: pipe separator | 2 | `\|` replaced by space |
| Name: Devanagari preserved | 3 | Character count unchanged |
| Name: Bengali preserved | 1 | Character count unchanged |
| Name: Kannada preserved | 1 | Character count unchanged |
| Name: French accents preserved | 4 | `é à ç ü` survive lowercasing |
| Name: None / empty | 3 | Returns `""` safely |
| Name: legal suffixes kept | 5 | Ltd, LLC, SARL, S.A.S not stripped |
| Name: entity_id passthrough | 1 | `S1-925783039` untouched |
| Address: hash prefix removal | 3 | `###56` → `56` |
| Address: NULL literal removal | 5 | `null`, `<NULL>`, `(null)` removed |
| Address: component order preserved | 3 | Reordered US addresses unchanged |
| Address: non-Latin preserved | 3 | Kannada and Devanagari in addresses |
| Address: French preserved | 3 | Accented French place names |
| Address: empty/None | 7 | All null patterns → `""` |
| Address: house numbers kept | 3 | Leading zeros, fractions kept |
| Country: known values | 3 | US, India, France |
| Country: whitespace | 3 | Stripped correctly |
| Country: case preserved | 2 | `US` stays `US` not `us` |
| Country: None/empty | 3 | Returns `""` safely |
| Missing flag: True cases | 8 | All null/empty patterns flagged |
| Missing flag: False cases | 5 | Valid addresses not flagged |
| DataFrame API | 18 | All column names, dtypes, values |

### Layer 2 — Integration tests (17 checks × 7 files)

Each of the six source files and ground truth file is loaded in full and checked:

| Check | What is verified |
|-------|-----------------|
| Row count | Exact match against known counts from dataset inspection |
| ID column present | `entity_id` / `source1_entity_id` exists |
| No null IDs | All entity IDs are non-null |
| Unique IDs | No duplicate entity IDs |
| ID prefix | All IDs have correct `S1-` / `S2-` / `S3-` prefix |
| Original columns present | `entity_id`, `business_name`, `business_address`, `country` |
| Norm columns created | All four `norm_*` / `address_missing` columns exist |
| No NaN in norm columns | Normalized string columns are fully non-null |
| `address_missing` is bool dtype | Not object/string |
| Missing-address rate in bounds | 0–0.5% for source1; 1–5% for source2/3 |
| Missing flag consistent | `address_missing==True` ↔ `norm_address==""` for every row |
| Norm names are lowercase | No uppercase ASCII in `norm_business_name` |
| Norm addresses are lowercase | No uppercase ASCII in `norm_business_address` |
| Country values match | `norm_country` unique values == `country` unique values |
| Non-Latin script preserved | Character counts unchanged for Devanagari/Bengali/Kannada/Tamil |
| No NULL literals in norm cols | `"null"` / `"<null>"` not present after normalization |
| No edge whitespace | Leading/trailing space absent |
| No double spaces | Internal double spaces absent |
| `entity_id` unchanged | Raw and preprocessed values identical |
| Row count stable | `len(df)` before == after preprocessing |

---

## 13. Important Decisions and Rationale

### Why lowercase names and addresses but not country?

Names and addresses vary freely in case between sources (`MILLER METALS` vs `Miller Metals` vs `miller metals`). Lowercasing makes all three identical to a string-similarity function. Country labels are already consistent (`US`, `India`, `France`) and downstream code may use exact-string matching on them, so casing is preserved.

### Why preserve legal suffixes (Ltd, LLC, SARL, S.A.S)?

Two businesses called "Global Trade" and "Global Trade LLC" are different entities. Stripping legal suffixes would merge them incorrectly. The suffixes are kept and the matching model will learn how much weight to assign them.

### Why not expand abbreviations (Rd → Road, Pvt → Private)?

Abbreviation expansion requires a language-specific lookup table. The dataset contains US English, Indian English, Hindi/Devanagari, and French. A US-centric abbreviation list would corrupt French addresses (e.g. `Rue` should not become `Road`). String similarity on character n-grams handles abbreviation differences naturally without expansion.

### Why not reorder address components to a canonical form?

Source 2 frequently reverses component order (`WA, Arlington, 18009 3rd Ave` vs `18009 3rd Ave, Arlington, WA`). Canonical reordering requires reliable city/state/country parsing, which is language- and country-specific. A French address parser does not apply to Indian landmark-based addresses. Downstream token-overlap features handle reordering more robustly than a parsing approach.

### Why NFC and not NFKC?

NFKC performs compatibility decomposition, which collapses stylistic variants (ligatures, circled letters, full-width digits) but also strips combining characters. This would corrupt Devanagari vowel signs (matras) and Bengali conjuncts, rendering those names unrecognizable. NFC is the safe choice for multilingual text.

### Why keep websites-used-as-names?

Some Source 2 records use the business website as the `business_name` field (e.g. `whiteallgraphics.com`). Removing these would produce empty names and eliminate a strong matching signal. The corresponding Source 1 record has the real name (`White All Graphics LLC`), and the website in Source 2 is a reliable identifier for that exact business.

### Why a separate `address_missing` flag instead of imputing?

Imputing a missing address (e.g. filling with "unknown") would create false address-similarity signals between unrelated businesses. The flag lets downstream models explicitly learn that "both records have no address" is a different situation from "addresses match" or "addresses differ". The flag is also useful for analysis: 3% of Source 2/3 records have no address and must be matched on name alone.
