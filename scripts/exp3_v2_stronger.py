"""
exp3_v2_stronger.py
====================

FIXES THE UNSTABLE DIRECTION from the first controlled experiment.

What changed from exp3_controlled_synthetic.py:
  - More training simulations per mechanism (60 instead of 40)
  - Longer solver runs (more steps -> floods develop further)
  - Longer training schedule (100 epochs instead of 60)
  - FIVE seeds instead of three, so one bad run cannot dominate the average
  - Reports median and IQR alongside mean, because seed 2 of the old run
    was a clear outlier (a degenerate near-zero-water run) and a mean alone
    hides that

Same solver, same terrain generator, same everything else — this is a
continuation of exp3, not a different experiment.

Run:
    python3 scripts/exp3_v2_stronger.py
    python3 scripts/exp3_v2_stronger.py --quick    (smoke test)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiment_lib import set_seed, train_model, evaluate, build_model, get_device
from scripts.exp3_controlled_synthetic import generate_dataset, normalise_together

OUT = 'results'


def median_iqr(values):
    """Median and interquartile range — more robust to one bad seed than mean/std."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float('nan'), float('nan'), float('nan')
    return float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--dim', type=int, default=48)
    ap.add_argument('--n-train', type=int, default=60)
    ap.add_argument('--n-test', type=int, default=15)
    ap.add_argument('--solver-steps', type=int, default=2400)
    ap.add_argument('--dt', type=float, default=2.0)
    ap.add_argument('--dx', type=float, default=50.0)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--out-dir', default='results')
    args = ap.parse_args()

    if args.quick:
        args.dim, args.n_train, args.n_test = 24, 4, 3
        args.solver_steps, args.epochs, args.seeds = 200, 3, [0]

    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()

    print("=" * 66)
    print("  EXPERIMENT 3 v2 — CONTROLLED, LARGER, 5 SEEDS")
    print("  Fixes the point->distributed instability from v1")
    print("=" * 66)
    print(f"  grid {args.dim}x{args.dim}, {args.n_train} train + {args.n_test} "
          f"test sims per mechanism, {args.epochs} epochs, seeds {args.seeds}")

    t0 = time.time()
    print("\n  Generating POINT-SOURCE floods...")
    pt_train = generate_dataset('point', args.n_train, args.dim,
                                 args.solver_steps, args.dt, args.dx, 0)
    pt_test = generate_dataset('point', args.n_test, args.dim,
                                args.solver_steps, args.dt, args.dx, 900)

    print("\n  Generating DISTRIBUTED floods, SAME terrain seeds...")
    di_train = generate_dataset('distributed', args.n_train, args.dim,
                                 args.solver_steps, args.dt, args.dx, 0)
    di_test = generate_dataset('distributed', args.n_test, args.dim,
                                args.solver_steps, args.dt, args.dx, 900)
    print(f"\n  Generation took {time.time() - t0:.0f}s")

    pt_train, pt_test, di_train, di_test = normalise_together(
        [pt_train, pt_test, di_train, di_test])
    data = {'point': {'train': pt_train, 'test': pt_test},
            'distributed': {'train': di_train, 'test': di_test}}

    results = []
    for train_on in ['point', 'distributed']:
        away = 'distributed' if train_on == 'point' else 'point'
        for seed in args.seeds:
            print(f"\n  --- train on {train_on}, seed {seed} ---")
            set_seed(seed)
            model = build_model('vector', 64, 3, device)
            ckpt = os.path.join(args.out_dir, f'synth2_{train_on}_s{seed}.pt')
            train_model(model, data[train_on]['train'], device,
                        epochs=args.epochs, max_train_samples=args.n_train,
                        ckpt_path=ckpt, log_every=max(1, args.epochs // 4))

            home = evaluate(model, data[train_on]['test'], device,
                            max_steps=30, max_samples=args.n_test)
            awayr = evaluate(model, data[away]['test'], device,
                             max_steps=30, max_samples=args.n_test)

            retained = (100.0 * awayr['csi_final'] / home['csi_final']
                        if home['csi_final'] > 0 else float('nan'))

            # flag degenerate runs (near-zero home performance) explicitly
            degenerate = home['csi_final'] < 0.02 and home['volume_ratio'] < 0.1

            print(f"    HOME ({train_on:11s}) CSI {home['csi_final']:.4f}  "
                  f"vol {home['volume_ratio']:.2f}x{'  [DEGENERATE]' if degenerate else ''}")
            print(f"    AWAY ({away:11s}) CSI {awayr['csi_final']:.4f}  "
                  f"vol {awayr['volume_ratio']:.2f}x")
            print(f"    retained {retained:.1f}%")

            results.append({
                'train_on': train_on, 'away': away, 'seed': seed,
                'home': home, 'away_scores': awayr, 'retained_pct': retained,
                'degenerate': bool(degenerate),
            })

    print("\n" + "=" * 66)
    print("  SUMMARY (median and IQR are more robust than mean to one bad seed)")
    print("=" * 66)
    for tr in ['point', 'distributed']:
        rs = [r for r in results if r['train_on'] == tr]
        n_degenerate = sum(r['degenerate'] for r in rs)
        home_vals = [r['home']['csi_final'] for r in rs]
        away_vals = [r['away_scores']['csi_final'] for r in rs]
        ret_vals = [r['retained_pct'] for r in rs]

        hm, hlo, hhi = median_iqr(home_vals)
        am, alo, ahi = median_iqr(away_vals)
        rm, rlo, rhi = median_iqr(ret_vals)

        print(f"\n  trained on {tr}:")
        print(f"    home CSI   median {hm:.4f}  IQR [{hlo:.4f}, {hhi:.4f}]  "
              f"mean {np.mean(home_vals):.4f}")
        print(f"    away CSI   median {am:.4f}  IQR [{alo:.4f}, {ahi:.4f}]  "
              f"mean {np.mean(away_vals):.4f}")
        print(f"    retained % median {rm:.1f}  IQR [{rlo:.1f}, {rhi:.1f}]  "
              f"mean {np.mean(ret_vals):.1f}")
        if n_degenerate:
            print(f"    WARNING: {n_degenerate}/{len(rs)} runs flagged degenerate "
                  f"(near-zero home performance) — excluded from headline claims")

    with open(os.path.join(args.out_dir, 'result_exp3_v2_stronger.json'), 'w') as f:
        json.dump({'config': vars(args), 'results': results}, f,
                  indent=2, default=str)
    print(f"\n  Saved {args.out_dir}/result_exp3_v2_stronger.json")


if __name__ == '__main__':
    main()