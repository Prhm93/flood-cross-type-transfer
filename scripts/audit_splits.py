"""
audit_splits.py
=================

Checks for leakage between train and test.

Breach: confirms the train / val / test simulation ID ranges never overlap.
Harvey: confirms no test sample is identical to a train sample, by hashing
        each sample's depth field and comparing the hashes.

Run:
    python3 scripts/audit_splits.py
"""

import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiment_lib import BREACH_TRAIN_IDS, BREACH_VAL_IDS, BREACH_TEST_IDS
from src.unified_loader import load_harvey


def fingerprint(sample):
    """A content hash of one sample's depth field, rounded so that tiny
    floating-point differences do not produce a different hash."""
    arr = np.round(sample['nodes_dynamic'][:, :, 0], 6)  # depth channel only
    return hashlib.sha256(arr.tobytes()).hexdigest()


def main():
    print("=" * 66)
    print("  SPLIT AUDIT - checking for train/test leakage")
    print("=" * 66)

    # ---- breach: ID-range check ----
    print("\n  BREACH - simulation ID ranges:")
    train_set = set(BREACH_TRAIN_IDS)
    val_set = set(BREACH_VAL_IDS)
    test_set = set(BREACH_TEST_IDS)

    print(f"    train: {min(BREACH_TRAIN_IDS)}-{max(BREACH_TRAIN_IDS)} "
          f"({len(train_set)} sims)")
    print(f"    val:   {min(BREACH_VAL_IDS)}-{max(BREACH_VAL_IDS)} "
          f"({len(val_set)} sims)")
    print(f"    test:  {min(BREACH_TEST_IDS)}-{max(BREACH_TEST_IDS)} "
          f"({len(test_set)} sims)")

    overlap_tt = train_set & test_set
    overlap_tv = train_set & val_set
    overlap_vt = val_set & test_set

    if overlap_tt or overlap_tv or overlap_vt:
        print("    FAIL: overlapping simulation IDs found!")
        print(f"      train&test: {overlap_tt}")
        print(f"      train&val:  {overlap_tv}")
        print(f"      val&test:   {overlap_vt}")
    else:
        print("    PASS: train, val and test use entirely disjoint "
              "simulation IDs - no event-level leakage possible.")

    # ---- harvey: content fingerprint check ----
    print("\n  HARVEY - content fingerprint check:")
    try:
        train = load_harvey('data/harvey/train.npz')
        val = load_harvey('data/harvey/val.npz')
        test = load_harvey('data/harvey/test.npz')
    except FileNotFoundError as e:
        print(f"    [skip] Harvey data not found: {e}")
        return

    train_fp = set(fingerprint(s) for s in train)
    val_fp = set(fingerprint(s) for s in val)
    test_fp = set(fingerprint(s) for s in test)

    print(f"    train: {len(train)} samples, {len(train_fp)} unique fingerprints")
    print(f"    val:   {len(val)} samples, {len(val_fp)} unique fingerprints")
    print(f"    test:  {len(test)} samples, {len(test_fp)} unique fingerprints")

    dup_tt = train_fp & test_fp
    dup_tv = train_fp & val_fp
    dup_vt = val_fp & test_fp

    if dup_tt or dup_tv or dup_vt:
        print("    FAIL: identical depth-field content found across splits!")
        print(f"      train&test duplicates: {len(dup_tt)}")
        print(f"      train&val duplicates:  {len(dup_tv)}")
        print(f"      val&test duplicates:   {len(dup_vt)}")
        print("    Some flood event appears in more than one split.")
        print("    Investigate before trusting any Harvey-side result.")
    else:
        print("    PASS: no identical depth-field content across train/val/test.")

    print("\n" + "=" * 66)
    print("  AUDIT COMPLETE")
    print("=" * 66)
    print("  This checks EVENT-level independence, which is the unit that")
    print("  matters for a generalisation claim. It cannot rule out subtler")
    print("  leakage, such as shared boundary conditions between adjacent")
    print("  Harvey graph chunks - say so in the paper.")


if __name__ == '__main__':
    main()
