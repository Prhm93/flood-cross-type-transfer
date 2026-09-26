#!/usr/bin/env python3
"""
exp12_stronger_harvey.py - ITEM 8. Run ONLY if exp11 shows no Harvey model beats the
all-wet CSI (0.6309) at home. Trains a larger Harvey model for longer, then scores it
at home and on dam-break with the all-wet CSI and skill S next to CSI and volume.
NEW configuration: report separately; it does not replace the main results.
--quick writes only to results/_smoke_exp12/. Never overwrites.
"""
import os, sys, json, time, argparse
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.experiment_lib import (load_cache, build_model, get_device, set_seed, train_model,
                                rollout_predictions, score_rollouts, volume_ratio)
from src.metrics import auto_threshold, csi

RESULTS = os.path.join(ROOT, 'results')


def load_ckpt(model, path, device):
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict):
        for k in ('model_state_dict', 'state_dict', 'model'):
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break
    model.load_state_dict(obj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--max-train-samples', type=int, default=None,
                    help='set to the value scripts/train.py used (see exp11 output)')
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    if a.quick:
        a.epochs, a.seeds = 2, [0]
    out_dir = os.path.join(RESULTS, '_smoke_exp12') if a.quick else RESULTS
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, 'result_exp12_stronger_harvey.json')
    if os.path.exists(out_json):
        print(f"{out_json} exists - refusing to overwrite.")
        return

    t0 = time.time()
    cache = load_cache()
    device = get_device()
    rows = []
    for s in a.seeds:
        ck = os.path.join(out_dir, f'best_harvey_h{a.hidden}_e{a.epochs}_s{s}.pt')
        if not os.path.exists(ck):
            tmp = ck + '.partial'
            print(f"\n--- train harvey, hidden {a.hidden}, {a.epochs} epochs, seed {s} ---", flush=True)
            set_seed(s)
            m = build_model('vector', a.hidden, 3, device)
            train_model(m, cache['harvey']['train'], device, epochs=a.epochs,
                        max_train_samples=a.max_train_samples, ckpt_path=tmp,
                        log_every=max(1, a.epochs // 4))
            os.replace(tmp, ck)
        m = build_model('vector', a.hidden, 3, device)
        load_ckpt(m, ck, device)
        m.eval()
        row = {'seed': s, 'params': sum(p.numel() for p in m.parameters())}
        with torch.no_grad():
            for te in ('harvey', 'breach'):
                roll = rollout_predictions(m, cache[te]['test'], device, 40, 19)
                r = score_rollouts(roll, 'auto')
                aw = float(np.median([csi(np.full_like(t[-1], np.inf), t[-1],
                                          auto_threshold(t[:, np.newaxis, :])) for _, t in roll]))
                c = float(r['csi_final'])
                row[te] = {'csi': c, 'allwet_csi': aw, 'S': (c - aw) / (1 - aw) if aw < 1 else float('nan'),
                           'volume_ratio': float(volume_ratio(roll)),
                           'arrival_h': float(r['arrival_mae_s']) / 3600.0}
                print(f"  s{s} on {te}: CSI {c:.4f} (all-wet {aw:.4f}, S {row[te]['S']:+.3f})  "
                      f"vol {row[te]['volume_ratio']:.2f}x  arrival {row[te]['arrival_h']:.1f} h")
        rows.append(row)
    json.dump({'config': vars(a), 'rows': rows}, open(out_json, 'w'), indent=2, default=str)
    print(f"\nsaved {out_json}  ({time.time() - t0:.0f}s)")


if __name__ == '__main__':
    main()
