"""
exp7_balanced_mixed.py
========================

Revisits the mixed-training failure from the first round.

The objection was fair: 1063 Harvey samples were pooled with 60 breach
samples, so of course the model forgot breach. This tests whether fixing
that imbalance changes anything.

Five training regimes, each evaluated on BOTH test sets:

    breach_only     baseline
    harvey_only     baseline
    naive_mixed     proportional pooling - what was done before
    balanced_mixed  equal numbers drawn from each type
    weighted_mixed  proportional draw, breach's loss scaled up instead

Run:
    python3 scripts/exp7_balanced_mixed.py
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiment_lib import (load_cache, build_cache, set_seed, evaluate,
                                build_model, get_device, CACHE_PATH)

OUT = 'results'


def train_epoch_weighted(model, breach, harvey, optimizer, device,
                         mode, breach_weight=1.0, samples_per_epoch=200,
                         window=20):
    """
    One training epoch under one pooling regime.

    mode='proportional' : draw from the pooled list at its natural rate
    mode='balanced'     : draw equally from breach and harvey
    mode='weighted'     : proportional draw, but breach's loss is
                          multiplied by breach_weight before backprop
    """
    model.train()
    total_loss, n_steps = 0.0, 0

    if mode == 'balanced' and breach and harvey:
        half = samples_per_epoch // 2
        # replace=True only if breach is smaller than half the batch
        b_idx = np.random.choice(len(breach), min(half, len(breach)),
                                 replace=len(breach) < half)
        h_idx = np.random.choice(len(harvey), min(half, len(harvey)), replace=False)
        batch = [(breach[i], 1.0) for i in b_idx] + [(harvey[i], 1.0) for i in h_idx]
    elif mode == 'weighted':
        pool = [(s, breach_weight) for s in breach] + [(s, 1.0) for s in harvey]
        idx = np.random.choice(len(pool), min(samples_per_epoch, len(pool)),
                               replace=False)
        batch = [pool[i] for i in idx]
    else:  # proportional, or a single-dataset regime
        pool = [(s, 1.0) for s in breach] + [(s, 1.0) for s in harvey]
        if not pool:
            return 0.0
        idx = np.random.choice(len(pool), min(samples_per_epoch, len(pool)),
                               replace=False)
        batch = [pool[i] for i in idx]

    np.random.shuffle(batch)

    for s, w in batch:
        dyn = torch.as_tensor(s['nodes_dynamic'], dtype=torch.float32, device=device)
        sta = torch.as_tensor(s['nodes_static'], dtype=torch.float32, device=device)
        ei = torch.as_tensor(s['edge_index'], dtype=torch.long, device=device)
        N, T, _ = dyn.shape
        if T < 3:
            continue
        w_len = min(window, T - 1)
        t0 = np.random.randint(0, T - w_len)

        loss_sum = 0.0
        for t in range(t0, t0 + w_len):
            cur, nxt = dyn[:, t, :], dyn[:, t + 1, :]
            target = nxt[:, :3] - cur[:, :3]   # predict the CHANGE, not the state
            pred = model(cur, sta, ei)
            loss_sum = loss_sum + nn.functional.mse_loss(pred, target)
            n_steps += 1

        loss_sum = loss_sum * w
        optimizer.zero_grad()
        (loss_sum / w_len).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss_sum.detach())

    return total_loss / max(n_steps, 1)


def main():
    device = get_device()
    print(f"Device: {device}")
    if not os.path.exists(CACHE_PATH):
        build_cache()
    data = load_cache()

    breach_train = data['breach']['train']
    harvey_train = data['harvey']['train']
    print(f"breach train: {len(breach_train)}, harvey train: {len(harvey_train)}")
    breach_weight = len(harvey_train) / max(len(breach_train), 1)
    print(f"auto loss-weight for breach (to offset the size imbalance): "
          f"{breach_weight:.1f}x")

    # name -> (mode, breach list, harvey list, samples per epoch)
    regimes = {
        'breach_only':    ('proportional', breach_train, [], 60),
        'harvey_only':    ('proportional', [], harvey_train, 200),
        'naive_mixed':    ('proportional', breach_train, harvey_train, 200),
        'balanced_mixed': ('balanced', breach_train, harvey_train, 200),
        'weighted_mixed': ('weighted', breach_train, harvey_train, 200),
    }

    results = []
    seeds = [0, 1, 2]

    for name, (mode, b, h, spe) in regimes.items():
        for seed in seeds:
            print(f"\n  --- regime={name}, seed={seed} ---")
            set_seed(seed)
            model = build_model('vector', 64, 3, device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               patience=5, factor=0.5)
            for ep in range(60):
                loss = train_epoch_weighted(model, b, h, optimizer, device,
                                            mode, breach_weight, spe)
                sched.step(loss)
                if ep % 20 == 0 or ep == 59:
                    print(f"    epoch {ep:3d}  loss={loss:.6f}")

            breach_score = evaluate(model, data['breach']['test'], device, 40, 19)
            harvey_score = evaluate(model, data['harvey']['test'], device, 40, 19)

            print(f"    breach test CSI {breach_score['csi_final']:.4f}  "
                  f"harvey test CSI {harvey_score['csi_final']:.4f}")

            results.append({'regime': name, 'seed': seed,
                            'breach_test': breach_score,
                            'harvey_test': harvey_score})

            # save as we go, so a crash late on does not lose earlier regimes
            with open(os.path.join(OUT, 'result_exp7_balanced_mixed.json'), 'w') as f:
                json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  REGIME COMPARISON MATRIX (median CSI across seeds)")
    print("=" * 70)
    print(f"  {'Regime':<18s} {'Breach test':>14s} {'Harvey test':>14s}")
    for name in regimes:
        rs = [r for r in results if r['regime'] == name]
        b_med = np.median([r['breach_test']['csi_final'] for r in rs])
        h_med = np.median([r['harvey_test']['csi_final'] for r in rs])
        print(f"  {name:<18s} {b_med:>14.4f} {h_med:>14.4f}")

    print("\n  READ THIS: does BALANCED or WEIGHTED mixing rescue breach")
    print("  performance compared with NAIVE mixing, while keeping Harvey")
    print("  performance reasonable? If yes, the earlier pooling failure was")
    print("  about imbalance, not about the two mechanisms being incompatible.")

    print(f"\n  Saved {OUT}/result_exp7_balanced_mixed.json")


if __name__ == '__main__':
    main()
