"""
exp6_metric_disagreement.py
============================

Per-event CSI against per-event volume ratio, instead of averages.

Why it matters: if one single event can have good CSI and a badly wrong
water volume AT THE SAME TIME, and that happens repeatedly, the claim is
much stronger than "the two averages disagree".

Also reports the Spearman rank correlation between CSI and volume error.
A weak correlation is the numerical version of "CSI on its own does not
tell you whether the physics is right".

No retraining - reuses existing checkpoints.

Run:
    python3 scripts/exp6_metric_disagreement.py
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiment_lib import (load_cache, build_cache, build_model, get_device,
                                rollout_predictions, score_per_event, CACHE_PATH)

OUT = 'results'


def spearman(x, y):
    """Rank correlation, written by hand so scipy is not needed."""
    x, y = np.asarray(x), np.asarray(y)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float('nan')
    # ranking twice with argsort turns values into ranks
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    device = get_device()
    if not os.path.exists(CACHE_PATH):
        build_cache()
    data = load_cache()

    all_events = {}

    for train_on in ['breach', 'harvey']:
        away = 'harvey' if train_on == 'breach' else 'breach'
        ckpt = os.path.join(OUT, f'best_{train_on}_vector_s0.pt')
        if not os.path.exists(ckpt):
            print(f"  [skip] no checkpoint {ckpt}")
            continue

        model = build_model('vector', 64, 3, device)
        model.load_state_dict(torch.load(ckpt, map_location=device,
                                         weights_only=True))

        print(f"\n{'='*66}")
        print(f"  Model trained on {train_on.upper()}")
        print(f"{'='*66}")

        for split_name, split_data in [('home', data[train_on]['test']),
                                       ('away', data[away]['test'])]:
            roll = rollout_predictions(model, split_data, device, 40, 19)
            events = score_per_event(roll)

            csis = [e['csi_final'] for e in events]
            vrs = [e['volume_ratio'] for e in events]
            # distance from a perfect ratio of 1, on a log scale so that
            # 10x too much and 10x too little count the same
            log_vr_err = [abs(np.log10(max(v, 1e-6))) for v in vrs]

            rho = spearman(csis, log_vr_err)

            print(f"\n  {split_name.upper()} "
                  f"({train_on if split_name=='home' else away}), "
                  f"n={len(events)} events:")
            print(f"    CSI:          median {np.median(csis):.4f}  "
                  f"range [{min(csis):.4f}, {max(csis):.4f}]")
            print(f"    volume ratio: median {np.median(vrs):.2f}x  "
                  f"range [{min(vrs):.2f}x, {max(vrs):.2f}x]")
            verdict = ('weak/no relationship - CSI does not track physical error'
                       if abs(rho) < 0.3 else 'some relationship')
            print(f"    Spearman(CSI, |log volume error|): {rho:.3f}  ({verdict})")

            # the events that matter most: CSI looks fine, volume is wrong
            bad = [e for e in events if e['csi_final'] > np.median(csis)
                   and (e['volume_ratio'] > 5 or e['volume_ratio'] < 0.2)]
            print(f"    Events with above-median CSI but volume ratio >5x or <0.2x: "
                  f"{len(bad)}/{len(events)}")
            if bad:
                worst = max(bad, key=lambda e: abs(np.log10(max(e['volume_ratio'], 1e-6))))
                print(f"    Worst example: event {worst['event_index']}, "
                      f"CSI={worst['csi_final']:.3f}, "
                      f"true_vol={worst['true_volume']:.2f}, "
                      f"pred_vol={worst['pred_volume']:.2f}, "
                      f"ratio={worst['volume_ratio']:.2f}x")

            key = f"{train_on}_{split_name}"
            all_events[key] = {
                'train_on': train_on,
                'tested_on': train_on if split_name == 'home' else away,
                'events': events,
                'spearman_csi_vs_logvolerr': rho,
                'n_bad_disagreements': len(bad),
                'n_events': len(events),
            }

    with open(os.path.join(OUT, 'result_exp6_metric_disagreement.json'), 'w') as f:
        json.dump(all_events, f, indent=2, default=str)
    print(f"\nSaved {OUT}/result_exp6_metric_disagreement.json")
    print("Per-event CSI and volume_ratio pairs - what the scatter plot needs")
    print("(CSI on x, volume ratio on y, one dot per flood event).")


if __name__ == '__main__':
    main()
