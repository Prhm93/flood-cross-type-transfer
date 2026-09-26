#!/usr/bin/env python3
"""exp21: our final breach models vs the lab SWE-GNN reproduction, same test floods (501-519),
same fixed 0.05 m threshold, CSI at step 40, mean over events (as the SWE-GNN script reports).
No training."""
import os, sys, json
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.experiment_lib import load_cache, build_model, get_device, rollout_predictions
from src.metrics import csi
from scripts.exp17_multistep import breach_source, ets

R = os.path.join(ROOT, 'results')
dmax = float(json.load(open(os.path.join(R, 'scaler.json')))['dynamic_max'][0])
THR = 0.05 / dmax                      # 0.05 m in scaled depth units
cache, device = load_cache(), get_device()
_, sc = breach_source(cache['breach']['train'], dmax)
test = breach_source(cache['breach']['test'], dmax, sc)[0]
print(f"0.05 m = {THR:.5f} in scaled units; {len(test)} test floods")

def score(path, h, L):
    m = build_model('vector', h, L, device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    m.eval()
    with torch.no_grad():
        roll = rollout_predictions(m, test, device, 40, 19)
    c = [float(csi(p[-1], t[-1], THR)) for p, t in roll]
    e = [ets(p[-1], t[-1], THR) for p, t in roll]
    a = [float(csi(np.full_like(t[-1], np.inf), t[-1], THR)) for p, t in roll]
    return float(np.mean(c)), float(np.nanmedian(e)), float(np.mean(a))

for label, pat, h, L, seeds in (('ours, small (42k params)', 'exp15_ms_breach_k8_pc_s{}.pt', 64, 3, range(5)),
                                ('ours, large (492k params)', 'exp15_ms_breach_k8_pc_h160L6_s{}.pt', 160, 6, range(3))):
    rows = [score(os.path.join(R, pat.format(s)), h, L) for s in seeds if os.path.exists(os.path.join(R, pat.format(s)))]
    if rows:
        print(f"  {label:26s} CSI@40 mean {np.median([r[0] for r in rows]):.3f} (seeds {[round(r[0],3) for r in rows]})  "
              f"ETS {np.median([r[1] for r in rows]):+.3f}  all-wet CSI {rows[0][2]:.3f}")

swe = json.load(open('/root/projects/hydraulic-attention/swegnn/rollout_results.json'))
for k, v in swe.items():
    try:
        print(f"  SWE-GNN reproduction {k:18s} CSI@40 mean {v['summary']['csi_mean_per_step'][-1]:.3f}  (n={v['summary']['n_sims']})")
    except Exception as ex:
        print(f"  {k}: {ex}")
