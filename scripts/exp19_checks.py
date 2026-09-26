#!/usr/bin/env python3
"""
exp19_checks.py - two reviewer checks on the final (pre-clamp) models. No training.
Part A, units confound: the 5 breach models scored on Harvey
  A0 as in the paper | A1 Harvey depth rescaled so its training maximum equals breach's
  A2 Harvey rain channel set to zero | A3 both
Part B, saturation: every final synthetic model scored by rollout step (1% threshold)
  and by threshold (final step), with the share of cells truly wet in each test set.
Writes results/result_exp19_checks.json (never overwrites).
"""
import os, sys, json, pickle
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.experiment_lib import load_cache, build_model, get_device, rollout_predictions
from src.metrics import auto_threshold, csi
from scripts.exp17_multistep import panel, ets

R = os.path.join(ROOT, 'results')
OUT = os.path.join(R, 'result_exp19_checks.json')
SYN = ('point', 'distributed', 'inflow')
STEPS = (3, 5, 10, 15, 20, 30)
FRACS = (0.01, 0.05, 0.10, 0.20)


def load(path, device):
    m = build_model('vector', 64, 3, device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return m.eval()


def med(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.median(x)) if x else float('nan')


def variant(samples, depth_f=1.0, zero_rain=False):
    out = []
    for s in samples:
        d = s['nodes_dynamic'].copy()
        d[:, :, 0] *= depth_f
        if zero_rain:
            d[:, :, 3] = 0.0
        out.append(dict(s, nodes_dynamic=d))
    return out


def scores(p, t, thr):
    c = float(csi(p, t, thr))
    a = float(csi(np.full_like(t, np.inf), t, thr))
    return {'S': (c - a) / (1 - a) if a < 1 else float('nan'), 'ets': ets(p, t, thr),
            'wet': float((t > thr).mean())}


def main():
    if os.path.exists(OUT):
        sys.exit(f'{OUT} exists - not overwritten')
    device, out = get_device(), {}

    # ---------------- Part A ----------------
    print("=" * 72 + "\n[A] units confound: breach models on Harvey\n" + "=" * 72, flush=True)
    cache = load_cache()
    mb = max(float(s['nodes_dynamic'][:, :, 0].max()) for s in cache['breach']['train'])
    mh = max(float(s['nodes_dynamic'][:, :, 0].max()) for s in cache['harvey']['train'])
    f = mb / mh
    print(f"  training max depth (scaled units): breach {mb:.4f}, harvey {mh:.4f} -> factor {f:.3f}")
    H = cache['harvey']['test']
    variants = {'A0 as in paper': H, 'A1 depth matched': variant(H, f),
                'A2 rain removed': variant(H, 1.0, True), 'A3 both': variant(H, f, True)}
    out['A'] = {'factor': f}
    with torch.no_grad():
        for name, smp in variants.items():
            rows = []
            for s in range(5):
                p = os.path.join(R, f'exp15_ms_breach_k8_pc_s{s}.pt')
                if os.path.exists(p):
                    rows.append(panel(load(p, device), smp, device, 40, 19))
            summ = {k: med([r[k] for r in rows]) for k in ('csi', 'allwet_csi', 'S', 'ets', 'volume_ratio')}
            out['A'][name] = summ
            print(f"  {name:17s} CSI {summ['csi']:.3f} (all-wet {summ['allwet_csi']:.3f})  S {summ['S']:+.3f}  "
                  f"ETS {summ['ets']:+.4f}  volume {summ['volume_ratio']:.2f}x   [{len(rows)} seeds]", flush=True)
    del cache

    # ---------------- Part B ----------------
    print("\n" + "=" * 72 + "\n[B] saturation: synthetic models by step and by threshold\n" + "=" * 72, flush=True)
    data = pickle.load(open(os.path.join(ROOT, 'data', 'synth_src_cache.pkl'), 'rb'))['data']
    cells = {}
    with torch.no_grad():
        for tr in SYN:
            for s in range(5):
                m = load(os.path.join(R, f'exp15_ms_{tr}_k8_pc_s{s}.pt'), device)
                for te in SYN:
                    roll = rollout_predictions(m, data[te]['test'], device, 30, 15)
                    by_step = {st: [scores(p[st], t[st], auto_threshold(t[:, np.newaxis, :])) for p, t in roll] for st in STEPS}
                    by_thr = {fr: [scores(p[-1], t[-1], max(fr * float(t.max()), 1e-9)) for p, t in roll] for fr in FRACS}
                    c = cells.setdefault(f'{tr}->{te}', {'step': {st: [] for st in STEPS}, 'thr': {fr: [] for fr in FRACS}})
                    for st in STEPS:
                        c['step'][st].append({k: med([e[k] for e in by_step[st]]) for k in ('S', 'ets', 'wet')})
                    for fr in FRACS:
                        c['thr'][fr].append({k: med([e[k] for e in by_thr[fr]]) for k in ('S', 'ets', 'wet')})
    summ = {}
    for key, c in cells.items():
        summ[key] = {'step': {st: {k: med([x[k] for x in c['step'][st]]) for k in ('S', 'ets', 'wet')} for st in STEPS},
                     'thr': {fr: {k: med([x[k] for x in c['thr'][fr]]) for k in ('S', 'ets', 'wet')} for fr in FRACS}}
    out['B'] = {k: {'step': {str(a): b for a, b in v['step'].items()}, 'thr': {str(a): b for a, b in v['thr'].items()}}
                for k, v in summ.items()}

    print("\n  share of cells truly wet (median over events):")
    for te in SYN:
        w = summ[f'point->{te}']
        print(f"    {te:11s} by step " + "  ".join(f"s{st}={w['step'][st]['wet']:.2f}" for st in STEPS)
              + " | final step by threshold " + "  ".join(f"{int(fr*100)}%={w['thr'][fr]['wet']:.2f}" for fr in FRACS))
    print("\n  skill S / ETS by rollout step (1% threshold), median over 5 seeds:")
    print(f"    {'pair':24s} " + " ".join(f"{'s'+str(st):>13s}" for st in STEPS))
    for key in summ:
        print(f"    {key:24s} " + " ".join(f"{summ[key]['step'][st]['S']:+.2f}/{summ[key]['step'][st]['ets']:+.2f}".rjust(13) for st in STEPS))
    print("\n  skill S / ETS at the final step by threshold, median over 5 seeds:")
    print(f"    {'pair':24s} " + " ".join(f"{str(int(fr*100))+'%':>13s}" for fr in FRACS))
    for key in summ:
        print(f"    {key:24s} " + " ".join(f"{summ[key]['thr'][fr]['S']:+.2f}/{summ[key]['thr'][fr]['ets']:+.2f}".rjust(13) for fr in FRACS))

    json.dump(out, open(OUT, 'w'), indent=2)
    print(f"\n  saved {OUT}")


if __name__ == '__main__':
    main()
