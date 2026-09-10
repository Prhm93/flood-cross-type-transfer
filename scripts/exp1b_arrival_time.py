"""
exp1b_arrival_time.py
======================

Extends the threshold sweep with ARRIVAL TIME, which the first version
computed internally but barely reported.

The goal: show all three metrics side by side for the same rollouts —

    CSI          -> says the breach model looks GOOD on Harvey data
    volume ratio -> says it is WRONG (131x too much water)
    arrival time -> should ALSO say it is wrong, if the flood timing
                    is nonsensical

If arrival time confirms the mass-error story, the paper gets a much
stronger three-way case: not just "one metric disagrees with another" but
"two independent physical checks agree the CSI reading is false."

Needs no retraining — reuses the checkpoints already on disk.

Run:
    python3 scripts/exp1b_arrival_time.py
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiment_lib import (load_cache, build_cache, build_model, get_device,
                                rollout_predictions, score_rollouts, volume_ratio,
                                CACHE_PATH)

OUT = 'results'


def main():
    device = get_device()
    print(f"Device: {device}")

    if not os.path.exists(CACHE_PATH):
        build_cache()
    data = load_cache()

    out = {}

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

        roll_home = rollout_predictions(model, data[train_on]['test'], device,
                                         40, 19)
        roll_away = rollout_predictions(model, data[away]['test'], device,
                                         40, 19)

        h = score_rollouts(roll_home, 'auto')
        a = score_rollouts(roll_away, 'auto')
        vr_h = volume_ratio(roll_home)
        vr_a = volume_ratio(roll_away)

        print(f"\n  {'Metric':<22s} {'HOME (' + train_on + ')':<18s} "
              f"{'AWAY (' + away + ')':<18s} Verdict")
        print(f"  {'-'*70}")

        csi_verdict = "AWAY LOOKS BETTER" if a['csi_final'] > h['csi_final'] else "away worse, as expected"
        print(f"  {'CSI (flood map)':<22s} {h['csi_final']:<18.4f} "
              f"{a['csi_final']:<18.4f} {csi_verdict}")

        vol_verdict = "AWAY IS WRONG" if abs(vr_a - 1) > abs(vr_h - 1) * 3 else "comparable"
        print(f"  {'Volume ratio (1=perfect)':<22s} {vr_h:<18.2f} "
              f"{vr_a:<18.2f} {vol_verdict}")

        arr_verdict = "AWAY IS WRONG" if a['arrival_mae_s'] > h['arrival_mae_s'] * 1.2 else "comparable"
        print(f"  {'Arrival MAE (hours)':<22s} "
              f"{h['arrival_mae_s']/3600:<18.2f} {a['arrival_mae_s']/3600:<18.2f} "
              f"{arr_verdict}")

        # The three-way verdict
        csi_says_good = a['csi_final'] > h['csi_final']
        physics_says_bad = (abs(vr_a - 1) > abs(vr_h - 1) * 2) or \
                            (a['arrival_mae_s'] > h['arrival_mae_s'] * 1.2)
        three_way = csi_says_good and physics_says_bad

        print(f"\n  THREE-WAY CHECK: CSI says away is better = {csi_says_good}. "
              f"Physical metrics say away is worse = {physics_says_bad}.")
        if three_way:
            print(f"  >>> CONFIRMED: CSI disagrees with BOTH physical checks. "
                  f"This is the strongest form of the paradox.")

        out[train_on] = {
            'home': h, 'away': a,
            'home_volume_ratio': vr_h, 'away_volume_ratio': vr_a,
            'home_arrival_hours': h['arrival_mae_s'] / 3600,
            'away_arrival_hours': a['arrival_mae_s'] / 3600,
            'csi_says_away_better': bool(csi_says_good),
            'physics_says_away_worse': bool(physics_says_bad),
            'three_way_paradox_confirmed': bool(three_way),
        }

    with open(os.path.join(OUT, 'result_exp1b_arrival_time.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved {OUT}/result_exp1b_arrival_time.json")


if __name__ == '__main__':
    main()