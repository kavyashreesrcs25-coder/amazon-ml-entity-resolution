"""
evaluate_blocking.py
====================
Standalone blocking recall evaluation against training ground truth.

Usage
-----
  # Evaluate an already-generated candidate file:
  python src/evaluate_blocking.py

  # Or specify paths explicitly:
  python src/evaluate_blocking.py \
      --candidates output/train_candidate_pairs.tsv \
      --ground-truth output/train_ground_truth_preprocessed.tsv

  # Generate train candidates first, then evaluate:
  python src/evaluate_blocking.py --run-blocking

Exit codes
----------
  0  all OK
  1  file not found or evaluation failed
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from blocking import (
    PREPROCESSED,
    OUTPUT_DIR,
    CHUNKSIZE,
    run_blocking,
    evaluate_recall,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate blocking recall on training data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--candidates", type=Path,
        default=OUTPUT_DIR / "train_candidate_pairs.tsv",
        help="Path to the candidate_pairs TSV to evaluate.",
    )
    p.add_argument(
        "--ground-truth", type=Path,
        default=PREPROCESSED["train_gt"],
        help="Path to train_ground_truth_preprocessed.tsv.",
    )
    p.add_argument(
        "--run-blocking", action="store_true",
        help="Generate train candidate pairs before evaluating "
             "(uses default blocking settings).",
    )
    p.add_argument(
        "--chunksize", type=int, default=CHUNKSIZE,
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    print("=" * 70)
    print("  Amazon ML Challenge 2026 — Blocking Recall Evaluation")
    print("=" * 70)

    if args.run_blocking:
        print("\nGenerating training candidates first ...")
        for key in ["train_s1", "train_s2", "train_s3"]:
            if not PREPROCESSED[key].exists():
                print(f"ERROR: {PREPROCESSED[key]} not found. "
                      "Run validate_preprocessing.py first.")
                return 1
        run_blocking(
            s1_path     = PREPROCESSED["train_s1"],
            s2_path     = PREPROCESSED["train_s2"],
            s3_path     = PREPROCESSED["train_s3"],
            output_path = args.candidates,
            verbose     = True,
        )

    if not args.candidates.exists():
        print(f"ERROR: Candidate file not found: {args.candidates}")
        print("Run with --run-blocking to generate it, or pass --candidates PATH.")
        return 1

    if not args.ground_truth.exists():
        print(f"ERROR: Ground truth not found: {args.ground_truth}")
        return 1

    t0 = time.perf_counter()
    result = evaluate_recall(
        gt_path        = args.ground_truth,
        candidate_path = args.candidates,
        verbose        = True,
        chunksize      = args.chunksize,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nEvaluation completed in {elapsed:.1f}s")

    # Exit with error if recall is dangerously low
    recall = result.get("recall_overall", 0.0)
    if recall < 0.5:
        print(f"\nWARNING: Overall recall {recall:.3f} is below 0.5 — "
              "blocking rules need improvement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
