#!/usr/bin/env python3
"""exp11b: do the exp5 large models beat a trivial 'flood everything' CSI? No training.
Model size is found by trying hidden/layer sizes until the checkpoint loads exactly."""
import os, sys, json
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.experiment_lib import load_cache, build_model, get_device, rollout_predictions, score_rollouts, volume_ratio
from src.metrics import auto_threshold, csi

R = os.path.join(ROOT, 'results')
OUT = os.path.join(R, 'result_exp11b_large_models.json')

def state(path, device):
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict):
        for k in ('model_state_dict', 'state_dict', 'model'):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj

def fit_model(sd, device):
    for h in (64, 96, 128, 160, 192, 224, 256, 320, 384, 512):
        for L in (3, 4, 5, 6, 8):
            m = build_model('vector', h, L, device)
            try:
                m.load_state_dict(sd, strict=True)
                return m, h, L
            except Exception:
                continue
    raise RuntimeError('no hidden/layers combination matches the checkpoint')

if os.path.exists(OUT):
    sys.exit(f'{OUT} exists - not overwritten')
cache, device, rows = load_cache(), get_device(), []
with torch.no_grad():
    for tr in ('harvey', 'breach'):
        for s in range(5):
            p = os.path.join(R, f'arch_large_{tr}_s{s}.pt')
            if not os.path.exists(p):
                continue
            m, h, L = fit_model(state(p, device), device)
            m.eval()
            row = {'train_on': tr, 'seed': s, 'hidden': h, 'layers': L,
                   'params': sum(q.numel() for q in m.parameters())}
            for te in ('harvey', 'breach'):
                roll = rollout_predictions(m, cache[te]['test'], device, 40, 19)
                aw = float(np.median([csi(np.full_like(t[-1], np.inf), t[-1], auto_threshold(t[:, np.newaxis, :])) for _, t in roll]))
                c = float(score_rollouts(roll, 'auto')['csi_final'])
                row[te] = {'csi': c, 'allwet': aw, 'S': (c - aw) / (1 - aw), 'volume_ratio': float(volume_ratio(roll))}
            rows.append(row)
            print(f"{tr} s{s} (hidden {h}, {L} layers, {row['params']:,} params) | "
                  f"on harvey CSI {row['harvey']['csi']:.3f} S {row['harvey']['S']:+.3f} vol {row['harvey']['volume_ratio']:.2f}x | "
                  f"on breach CSI {row['breach']['csi']:.3f} S {row['breach']['S']:+.3f} vol {row['breach']['volume_ratio']:.2f}x", flush=True)
json.dump(rows, open(OUT, 'w'), indent=2)
print('saved', OUT)
