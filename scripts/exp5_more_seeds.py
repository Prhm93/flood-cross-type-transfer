"""
exp5_more_seeds.py
===================

Adds seeds 2, 3, 4 to the architecture experiment (exp5 originally only
ran seeds 0, 1). More seeds tighten the confidence interval on whether
the transfer gap really is architecture-independent.

Merges its output with the existing results/result_exp5_stronger_model.json
rather than overwriting it, so nothing from the first run is lost.

Run:
    python3 scripts/exp5_more_seeds.py
    python3 scripts/exp5_more_seeds.py --quick
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiment_lib import (load_cache, build_cache, set_seed, train_model,
                                 build_model, get_device, evaluate, CACHE_PATH)

OUT = 'results'
RESULT_PATH = os.path.join(OUT, 'result_exp5_stronger_model.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--extra-seeds', type=int, nargs='+', default=[2, 3, 4])
    ap.add_argument('--train-samples', type=int, default=200)
    ap.add_argument('--eval-samples', type=int, default=19)
    ap.add_argument('--rollout-steps', type=int, default=40)
    args = ap.parse_args()

    if args.quick:
        args.epochs, args.extra_seeds = 2, [2]
        args.train_samples, args.eval_samples, args.rollout_steps = 4, 3, 5

    device = get_device()
    print(f"Device: {device}")

    if not os.path.exists(CACHE_PATH):
        build_cache()
    data = load_cache()

    existing = []
    if os.path.exists(RESULT_PATH):
        with open(RESULT_PATH) as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing exp5 results, adding seeds "
              f"{args.extra_seeds}")

    configs = [('small', 64, 3), ('large', 160, 6)]
    new_results = []

    for name, hidden, layers in configs:
        for train_on in ['breach', 'harvey']:
            away = 'harvey' if train_on == 'breach' else 'breach'
            for seed in args.extra_seeds:
                already = any(r['arch'] == name and r['train_on'] == train_on
                             and r['seed'] == seed for r in existing)
                if already:
                    print(f"  [skip] {name} {train_on} seed {seed} already present")
                    continue

                set_seed(seed)
                m = build_model('vector', hidden, layers, device)
                npar = sum(p.numel() for p in m.parameters())
                print(f"\n  --- {name} ({npar:,} params), train {train_on}, "
                      f"seed {seed} ---")
                ck = os.path.join(OUT, f'arch_{name}_{train_on}_s{seed}.pt')
                train_model(m, data[train_on]['train'], device,
                            epochs=args.epochs,
                            max_train_samples=args.train_samples,
                            ckpt_path=ck, log_every=max(1, args.epochs // 3))

                h = evaluate(m, data[train_on]['test'], device,
                             args.rollout_steps, args.eval_samples)
                a = evaluate(m, data[away]['test'], device,
                             args.rollout_steps, args.eval_samples)
                ret = (100 * a['csi_final'] / h['csi_final']
                       if h['csi_final'] > 0 else float('nan'))
                print(f"    home CSI {h['csi_final']:.4f}  away CSI "
                      f"{a['csi_final']:.4f}  retained {ret:.1f}%")
                new_results.append({'arch': name, 'params': npar,
                                    'train_on': train_on, 'seed': seed,
                                    'home': h, 'away': a, 'retained_pct': ret})

    merged = existing + new_results
    with open(RESULT_PATH, 'w') as f:
        json.dump(merged, f, indent=2, default=str)
    print(f"\n  Merged file now has {len(merged)} total runs -> {RESULT_PATH}")

    print("\n" + "=" * 66)
    print("  UPDATED SUMMARY WITH ALL SEEDS")
    print("=" * 66)
    for name in ['small', 'large']:
        for train_on in ['breach', 'harvey']:
            rs = [r for r in merged if r['arch'] == name and r['train_on'] == train_on]
            if not rs:
                continue
            rets = [r['retained_pct'] for r in rs]
            print(f"  {name:6s} train={train_on:8s} n_seeds={len(rs)}  "
                  f"retained mean={np.mean(rets):.1f}%  "
                  f"median={np.median(rets):.1f}%  std={np.std(rets):.1f}")


if __name__ == '__main__':
    main()