"""
eval_persistence.py
===================

EXPERIMENT 2: Score the persistence baseline on both datasets.

Persistence = "the water stays exactly where it is at the start."
Every future frame is a copy of frame 0.

This answers the reviewer's most dangerous question:
"Does your model beat doing nothing?"

If the model beats persistence at home but not away, that IS the
transfer story: learning helps at home but breaks away.

If the model doesn't beat persistence even at home, the model is
too weak and the transfer comparison is unreliable.

No GPU needed. No training. Just scoring.

Usage:
    python3 scripts/eval_persistence.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.unified_loader import load_harvey, load_breach
from src.normalise import fit_scaler, apply_scaler
from src.metrics import score, persistence_depth

BREACH_TEST_IDS = list(range(501, 520))
BREACH_TRAIN_IDS = list(range(1, 61))
BREACH_VAL_IDS = list(range(61, 81))


def eval_persistence_on(samples, label, max_samples=19, max_steps=40):
    """Score the persistence baseline on a set of samples."""
    all_scores = []

    for idx in range(min(max_samples, len(samples))):
        s = samples[idx]
        dyn = s['nodes_dynamic']  # (N, T, 4)
        N, T, _ = dyn.shape
        steps = min(max_steps, T - 1)
        if steps < 2:
            continue

        # True depth sequence: (steps+1, N)
        true_depths = dyn[:, :steps + 1, 0].T  # (steps+1, N)

        # Persistence: every frame = frame 0
        pers = persistence_depth(true_depths)

        # Reshape for metrics: (T, 1, N)
        pred_3d = pers[:, np.newaxis, :]
        true_3d = true_depths[:, np.newaxis, :]

        s_result = score(pred_3d, true_3d, dt_seconds=3600.0, threshold=None)
        all_scores.append(s_result)

    if not all_scores:
        return {}

    avg = {
        'csi_final': float(np.nanmean([s['csi_final'] for s in all_scores])),
        'csi_mean': float(np.nanmean([s['csi_mean'] for s in all_scores])),
        'arrival_mae_s': float(np.nanmean([s['arrival_mae_s'] for s in all_scores])),
        'mass_error': float(np.nanmean([s['mass_error'] for s in all_scores])),
        'n_samples': len(all_scores),
    }

    print(f"\n  PERSISTENCE on {label} ({avg['n_samples']} samples):")
    print(f"    CSI final:     {avg['csi_final']:.4f}")
    print(f"    CSI mean:      {avg['csi_mean']:.4f}")
    print(f"    Arrival MAE:   {avg['arrival_mae_s']:.0f} s")
    print(f"    Mass error:    {avg['mass_error']:.4f}")

    return avg


def main():
    print("=" * 60)
    print("  PERSISTENCE BASELINE EVALUATION")
    print("  (Does the model beat doing nothing?)")
    print("=" * 60)

    # Load data
    print("\nLoading datasets...")
    breach_test = load_breach('data/breach/raw_datasets', BREACH_TEST_IDS, dim=64)
    breach_train = load_breach('data/breach/raw_datasets', BREACH_TRAIN_IDS, dim=64)
    breach_val = load_breach('data/breach/raw_datasets', BREACH_VAL_IDS, dim=64)
    harvey_test = load_harvey('data/harvey/test.npz')
    harvey_train = load_harvey('data/harvey/train.npz')
    harvey_val = load_harvey('data/harvey/val.npz')

    # Joint normalisation (same as training)
    print("\nFitting joint scaler...")
    all_samples = breach_train + breach_val + harvey_train + harvey_val
    scaler = fit_scaler(all_samples)

    breach_test_n = apply_scaler(breach_test, scaler)
    harvey_test_n = apply_scaler(harvey_test, scaler)

    # Score persistence on both
    breach_pers = eval_persistence_on(breach_test_n, "BREACH test")
    harvey_pers = eval_persistence_on(harvey_test_n, "HARVEY test")

    # Compare with model results if available
    print(f"\n{'='*60}")
    print(f"  COMPARISON: Model vs Persistence")
    print(f"{'='*60}")

    result_files = [
        ('breach_vector', 'results/result_breach_vector_s0.json'),
        ('harvey_vector', 'results/result_harvey_vector_s0.json'),
    ]

    for label, path in result_files:
        if os.path.exists(path):
            with open(path) as f:
                r = json.load(f)
            home_csi = r['home_scores']['csi_final']
            train_on = r['train_on']

            if train_on == 'breach':
                pers_csi = breach_pers['csi_final']
                pers_label = 'breach'
            else:
                pers_csi = harvey_pers['csi_final']
                pers_label = 'harvey'

            beats = "YES" if home_csi > pers_csi else "NO"
            print(f"\n  {label} (home = {pers_label}):")
            print(f"    Model CSI:       {home_csi:.4f}")
            print(f"    Persistence CSI: {pers_csi:.4f}")
            print(f"    Model beats persistence? {beats}")

    # Save
    result = {
        'breach_persistence': breach_pers,
        'harvey_persistence': harvey_pers,
    }
    out_path = 'results/result_persistence.json'
    os.makedirs('results', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == '__main__':
    main()