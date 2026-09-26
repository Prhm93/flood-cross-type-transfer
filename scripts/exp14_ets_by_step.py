#!/usr/bin/env python3
"""exp14: ETS by rollout step - is there skill early that is lost later? No training."""
import os, sys
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.experiment_lib import load_cache, build_model, get_device, rollout_predictions
from src.metrics import auto_threshold

R = os.path.join(ROOT, 'results')
STEPS = [1, 2, 3, 5, 10, 20, 30, 40]


def ets(pred, true, thr):
    p, t = pred > thr, true > thr
    H, F, M, N = np.sum(p & t), np.sum(p & ~t), np.sum(~p & t), p.size
    hr = (H + F) * (H + M) / N
    den = H + F + M - hr
    return float((H - hr) / den) if den > 0 else float('nan')


def load(path, h, L, device):
    m = build_model('vector', h, L, device)
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict):
        for k in ('model_state_dict', 'state_dict', 'model'):
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break
    m.load_state_dict(obj)
    return m.eval()


cache, device = load_cache(), get_device()
models = [('small breach s0', 'best_breach_vector_s0.pt', 64, 3),
          ('small harvey s0', 'best_harvey_vector_s0.pt', 64, 3),
          ('large breach s0', 'arch_large_breach_s0.pt', 160, 6),
          ('large harvey s0', 'arch_large_harvey_s0.pt', 160, 6)]
print("median ETS over 19 events at each rollout step (0 = chance)")
print(f"{'model':16s} {'test':7s} " + " ".join(f"s{s:<6d}" for s in STEPS))
with torch.no_grad():
    for name, f, h, L in models:
        m = load(os.path.join(R, f), h, L, device)
        for te in ('breach', 'harvey'):
            roll = rollout_predictions(m, cache[te]['test'], device, 40, 19)
            T = roll[0][0].shape[0]
            vals = []
            for st in STEPS:
                if st > T:
                    vals.append('   -   ')
                    continue
                e = [ets(p[st - 1], t[st - 1], auto_threshold(t[:, np.newaxis, :])) for p, t in roll]
                vals.append(f"{np.nanmedian(e):+.3f} ")
            print(f"{name:16s} {te:7s} " + " ".join(vals), flush=True)
