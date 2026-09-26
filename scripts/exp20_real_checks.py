#!/usr/bin/env python3
"""
exp20_real_checks.py - reviewer checks on the final real-data models. No training.
Part A: breach (5 seeds) and Harvey (3 seeds) models on both real test sets at
        thresholds 1/5/10/20% (final step): skill S, ETS, share truly wet; volume ratio.
Part B: leakage check - breach models on the breach test set with the true breach
        hydrograph versus a CONSTANT inflow at the breach cell (same total volume).
Writes results/result_exp20_real_checks.json (never overwrites).
"""
import os, sys, json
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.experiment_lib import load_cache, build_model, get_device, rollout_predictions, volume_ratio
from src.metrics import csi
from scripts.exp17_multistep import breach_source, panel, ets

R = os.path.join(ROOT, 'results')
OUT = os.path.join(R, 'result_exp20_real_checks.json')
FRACS = (0.01, 0.05, 0.10, 0.20)


def load(path, device):
    m = build_model('vector', 64, 3, device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return m.eval()


def med(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.median(x)) if x else float('nan')


def scores(p, t, thr):
    c = float(csi(p, t, thr))
    a = float(csi(np.full_like(t, np.inf), t, thr))
    return {'S': (c - a) / (1 - a) if a < 1 else float('nan'), 'ets': ets(p, t, thr), 'wet': float((t > thr).mean())}


def by_threshold(model, samples, device):
    roll = rollout_predictions(model, samples, device, 40, 19)
    res = {str(fr): {k: med([scores(p[-1], t[-1], max(fr * float(t.max()), 1e-9))[k] for p, t in roll])
                     for k in ('S', 'ets', 'wet')} for fr in FRACS}
    res['volume_ratio'] = float(volume_ratio(roll))
    return res


def constant_source(samples):
    out = []
    for s in samples:
        d = s['nodes_dynamic'].copy()
        src = d[:, :, 3]
        cell = int(np.argmax(src.max(1)))
        d[cell, :, 3] = src[cell].mean()
        out.append(dict(s, nodes_dynamic=d))
    return out


def main():
    if os.path.exists(OUT):
        sys.exit(f'{OUT} exists - not overwritten')
    device, cache, out = get_device(), load_cache(), {}
    dmax = float(json.load(open(os.path.join(R, 'scaler.json')))['dynamic_max'][0])
    _, scale = breach_source(cache['breach']['train'], dmax)
    tests = {'breach': breach_source(cache['breach']['test'], dmax, scale)[0], 'harvey': cache['harvey']['test']}
    models = {'breach': [f'exp15_ms_breach_k8_pc_s{s}.pt' for s in range(5)],
              'harvey': [f'exp15_ms_harvey_k8_pc_s{s}.pt' for s in range(3)]}

    print("=" * 72 + "\n[A] real models by threshold (final step), median over seeds\n" + "=" * 72, flush=True)
    with torch.no_grad():
        for tr, files in models.items():
            for te, smp in tests.items():
                rows = [by_threshold(load(os.path.join(R, f), device), smp, device)
                        for f in files if os.path.exists(os.path.join(R, f))]
                cell = {fr: {k: med([r[str(fr)][k] for r in rows]) for k in ('S', 'ets', 'wet')} for fr in FRACS}
                vol = med([r['volume_ratio'] for r in rows])
                out[f'{tr}->{te}'] = {'by_threshold': {str(k): v for k, v in cell.items()}, 'volume_ratio': vol, 'n': len(rows)}
                print(f"  {tr:7s}->{te:7s} vol {vol:6.2f}x | " + "  ".join(
                    f"{int(fr*100)}%: S {cell[fr]['S']:+.3f} ETS {cell[fr]['ets']:+.3f} wet {cell[fr]['wet']:.2f}" for fr in FRACS), flush=True)

        print("\n" + "=" * 72 + "\n[B] leakage check: breach models, true vs constant inflow\n" + "=" * 72, flush=True)
        for name, smp in (('true hydrograph', tests['breach']), ('constant inflow', constant_source(tests['breach']))):
            rows = []
            for f in models['breach']:
                m = load(os.path.join(R, f), device)
                r = panel(m, smp, device, 40, 19)
                r['t10'] = by_threshold(m, smp, device)['0.1']
                rows.append(r)
            summ = {k: med([r[k] for r in rows]) for k in ('csi', 'S', 'ets', 'volume_ratio')}
            summ['S10'] = med([r['t10']['S'] for r in rows])
            summ['ets10'] = med([r['t10']['ets'] for r in rows])
            out[f'leak_{name}'] = summ
            print(f"  {name:16s} 1%: S {summ['S']:+.3f} ETS {summ['ets']:+.3f} | 10%: S {summ['S10']:+.3f} "
                  f"ETS {summ['ets10']:+.3f} | volume {summ['volume_ratio']:.2f}x", flush=True)
    json.dump(out, open(OUT, 'w'), indent=2)
    print(f"\n  saved {OUT}")


if __name__ == '__main__':
    main()
