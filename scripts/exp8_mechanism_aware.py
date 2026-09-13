"""
exp8_mechanism_aware.py
==========================

Does telling the model which flood mechanism it is looking at (a simple
one-hot flag, point-source vs distributed) help a model trained on BOTH
types handle both types well?

Compares:
    universal model, NO mechanism tag
    universal model, WITH mechanism tag

BOTH arms use balanced sampling, so the only difference between them is
the tag. (Note: the untagged arm is therefore balanced_mixed from exp7,
not naive_mixed.)

Judged on CSI *and* volume ratio. Judging a fix on CSI alone would be
inconsistent with the rest of this paper, which argues CSI can look fine
while the water is badly wrong.

Run:
    python3 scripts/exp8_mechanism_aware.py
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiment_lib import load_cache, build_cache, set_seed, get_device, CACHE_PATH
from src.model_mechanism import FloodGNNMech, add_mechanism_tag

OUT = 'results'


def train_epoch(model, breach, harvey, optimizer, device, samples_per_epoch=200,
                window=20):
    """Balanced-sampling epoch. Works for tagged and untagged data alike,
    because the tag lives inside nodes_static."""
    model.train()
    total_loss, n_steps = 0.0, 0
    half = samples_per_epoch // 2
    b_idx = np.random.choice(len(breach), min(half, len(breach)),
                             replace=len(breach) < half)
    h_idx = np.random.choice(len(harvey), min(half, len(harvey)), replace=False)
    batch = [breach[i] for i in b_idx] + [harvey[i] for i in h_idx]
    np.random.shuffle(batch)

    for s in batch:
        dyn = torch.as_tensor(s['nodes_dynamic'], dtype=torch.float32, device=device)
        sta = torch.as_tensor(s['nodes_static'], dtype=torch.float32, device=device)
        ei = torch.as_tensor(s['edge_index'], dtype=torch.long, device=device)
        N, T, _ = dyn.shape
        if T < 3:
            continue
        w = min(window, T - 1)
        t0 = np.random.randint(0, T - w)

        loss_sum = 0.0
        for t in range(t0, t0 + w):
            cur, nxt = dyn[:, t, :], dyn[:, t + 1, :]
            target = nxt[:, :3] - cur[:, :3]
            pred = model(cur, sta, ei)
            loss_sum = loss_sum + nn.functional.mse_loss(pred, target)
            n_steps += 1

        optimizer.zero_grad()
        (loss_sum / w).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss_sum.detach())

    return total_loss / max(n_steps, 1)


@torch.no_grad()
def evaluate_tagged(model, samples, device, max_steps=40, max_samples=19):
    """
    Same rollout as experiment_lib.evaluate, but for the tagged model
    (nodes_static has 3 columns instead of 1).

    Returns CSI *and* volume ratio, so the tag can be judged on the
    physics as well as on the flood map.
    """
    from src.metrics import score
    model.eval()
    csis, ratios = [], []
    for idx in range(min(max_samples, len(samples))):
        s = samples[idx]
        dyn = torch.as_tensor(s['nodes_dynamic'], dtype=torch.float32, device=device)
        sta = torch.as_tensor(s['nodes_static'], dtype=torch.float32, device=device)
        ei = torch.as_tensor(s['edge_index'], dtype=torch.long, device=device)
        N, T, _ = dyn.shape
        steps = min(max_steps, T - 1)
        if steps < 2:
            continue
        cur = dyn[:, 0, :].clone()
        preds = [cur[:, 0].cpu().numpy()]
        trues = [dyn[:, 0, 0].cpu().numpy()]
        for t in range(steps):
            delta = model(cur, sta, ei)
            new = cur.clone()
            new[:, :3] = cur[:, :3] + delta
            new[:, 0] = torch.clamp(new[:, 0], min=0.0)   # water cannot go negative
            if t + 1 < T:
                new[:, 3] = dyn[:, t + 1, 3]              # rain is a known input
            cur = new
            preds.append(cur[:, 0].cpu().numpy())
            trues.append(dyn[:, min(t + 1, T - 1), 0].cpu().numpy())

        pred_arr = np.stack(preds, 0)
        true_arr = np.stack(trues, 0)
        p3 = pred_arr[:, np.newaxis, :]
        t3 = true_arr[:, np.newaxis, :]
        csis.append(score(p3, t3, threshold=None)['csi_final'])

        # volume ratio, float64 so rounding does not eat the signal
        vt = float(np.sum(true_arr, dtype=np.float64))
        vp = float(np.sum(pred_arr, dtype=np.float64))
        ratios.append(vp / vt if vt > 0 else np.nan)

    if not csis:
        return {'csi_final': 0.0, 'volume_ratio': float('nan')}
    return {'csi_final': float(np.nanmean(csis)),
            'volume_ratio': float(np.nanmedian(ratios))}


def main():
    device = get_device()
    print(f"Device: {device}")
    if not os.path.exists(CACHE_PATH):
        build_cache()
    data = load_cache()

    breach_train, harvey_train = data['breach']['train'], data['harvey']['train']
    breach_test, harvey_test = data['breach']['test'], data['harvey']['test']

    # tagged copies of everything
    breach_train_t = add_mechanism_tag(breach_train, 'point')
    harvey_train_t = add_mechanism_tag(harvey_train, 'distributed')
    breach_test_t = add_mechanism_tag(breach_test, 'point')
    harvey_test_t = add_mechanism_tag(harvey_test, 'distributed')

    results = []
    for tagged in [False, True]:
        label = 'with_tag' if tagged else 'no_tag'
        for seed in [0, 1, 2, 3, 4]:
            print(f"\n  --- {label}, seed {seed} ---")
            set_seed(seed)
            if tagged:
                model = FloodGNNMech('vector', hidden=64, num_layers=3).to(device)
                bt, ht = breach_train_t, harvey_train_t
                bte, hte = breach_test_t, harvey_test_t
            else:
                from src.experiment_lib import build_model
                model = build_model('vector', 64, 3, device)
                bt, ht = breach_train, harvey_train
                bte, hte = breach_test, harvey_test

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               patience=5, factor=0.5)
            for ep in range(60):
                loss = train_epoch(model, bt, ht, optimizer, device)
                sched.step(loss)
                if ep % 20 == 0 or ep == 59:
                    print(f"    epoch {ep:3d}  loss={loss:.6f}")

            # the untagged model also gets volume ratio, via the same helper -
            # its static input just has 1 column instead of 3
            b_score = evaluate_tagged(model, bte, device)
            h_score = evaluate_tagged(model, hte, device)

            print(f"    breach CSI {b_score['csi_final']:.4f} "
                  f"vol {b_score['volume_ratio']:.2f}x   "
                  f"harvey CSI {h_score['csi_final']:.4f} "
                  f"vol {h_score['volume_ratio']:.2f}x")

            results.append({'tagged': tagged, 'seed': seed,
                            'breach_csi': b_score['csi_final'],
                            'breach_volume_ratio': b_score['volume_ratio'],
                            'harvey_csi': h_score['csi_final'],
                            'harvey_volume_ratio': h_score['volume_ratio']})

            with open(os.path.join(OUT, 'result_exp8_mechanism_aware.json'), 'w') as f:
                json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 74)
    print("  MECHANISM-TAG COMPARISON (median across seeds)")
    print("=" * 74)
    for tagged in [False, True]:
        rs = [r for r in results if r['tagged'] == tagged]
        label = 'WITH mechanism tag' if tagged else 'NO tag (implicit)'
        print(f"  {label:20s}  "
              f"breach CSI {np.median([r['breach_csi'] for r in rs]):.4f} "
              f"vol {np.median([r['breach_volume_ratio'] for r in rs]):.2f}x   "
              f"harvey CSI {np.median([r['harvey_csi'] for r in rs]):.4f} "
              f"vol {np.median([r['harvey_volume_ratio'] for r in rs]):.2f}x")

    print("\n  READ THIS: the tag is only a real fix if it improves BOTH")
    print("  datasets on CSI AND moves the volume ratio closer to 1.")
    print("  Better CSI with a worse volume ratio is this paper's own")
    print("  warning showing up in our own experiment.")

    print(f"\n  Saved {OUT}/result_exp8_mechanism_aware.json")


if __name__ == '__main__':
    main()
