#!/usr/bin/env python3
"""exp22 - reviewer fixes #1-#4. Final recipe (multi-step k->8, loss before clipping) with
SOURCE-ONLY scaling (scaler fitted on the training mechanism only) and a choice of
source encoding in channel 3: log = log(1+rate/1e-6) per cell | linear = rate per cell |
global = total domain inflow, same value on every cell (no location).
Saves per-event CSI, all-wet CSI, ETS, frequency bias, wet share and volume ratio at
relative 1/5/10/20% and physical 0.01/0.05/0.1/0.3 m thresholds (no physical for Harvey)."""
import os, sys, json, time, pickle, argparse
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.experiment_lib import load_cache, build_model, get_device, set_seed, rollout_predictions
import scripts.exp17_multistep as E

R = os.path.join(ROOT, 'results'); R0 = 1e-6
SYN = ('point', 'distributed', 'inflow')
REL, PHYS = (0.01, 0.05, 0.10, 0.20), (0.01, 0.05, 0.10, 0.30)


def inv(samples, sc):
    lo, hi = np.array(sc['dynamic_min']), np.array(sc['dynamic_max'])
    return [dict(s, nodes_dynamic=s['nodes_dynamic'].astype(np.float64) * (hi - lo) + lo) for s in samples]


def encode(samples, enc, area):
    out = []
    for s in samples:
        d = s['nodes_dynamic'].copy(); rate = d[:, :, 3]
        if enc == 'log': d[:, :, 3] = np.log1p(rate / R0)
        elif enc == 'global': d[:, :, 3] = np.broadcast_to(rate.sum(0) * area, rate.shape)
        out.append(dict(s, nodes_dynamic=d))
    return out


def fit(train):
    lo = np.min([s['nodes_dynamic'].reshape(-1, 4).min(0) for s in train], 0)
    hi = np.max([s['nodes_dynamic'].reshape(-1, 4).max(0) for s in train], 0)
    return lo, np.where(hi - lo == 0, 1.0, hi - lo)


def apply(samples, lo, rg):
    return [dict(s, nodes_dynamic=((s['nodes_dynamic'] - lo) / rg).astype(np.float32)) for s in samples]


def breach_rate(samples):
    out = []
    for s in samples:
        d = s['nodes_dynamic'].copy(); dep = d[:, :, 0]
        q = np.maximum(np.diff(dep.sum(0) * 1e4), 0) / 3600.0
        cell = int(np.argmax(dep[:, next(t for t in range(dep.shape[1]) if dep[:, t].max() > 1e-3)]))
        d[:, :, 3] = 0.0; d[cell, :, 3] = np.append(q, q[-1]) / 1e4
        out.append(dict(s, nodes_dynamic=d))
    return out


def ev(p, t, thr):
    P, T = p > thr, t > thr
    H, F, M, N = int((P & T).sum()), int((P & ~T).sum()), int((~P & T).sum()), P.size
    hr = (H + F) * (H + M) / N; den = H + F + M - hr
    return {'csi': H / (H + F + M) if H + F + M else float('nan'), 'allwet': (H + M) / N,
            'ets': (H - hr) / den if den > 0 else float('nan'),
            'fb': (H + F) / (H + M) if H + M else float('nan'), 'wet': (H + M) / N}


def score(model, test, device, steps, n, drange, physical):
    roll = rollout_predictions(model, test, device, steps, n)
    res = {'volume': [float(p.sum() / t.sum()) if t.sum() > 0 else float('nan') for p, t in roll]}
    for fr in REL:
        res[f'rel{int(fr*100)}'] = [ev(p[-1], t[-1], max(fr * float(t.max()), 1e-9)) for p, t in roll]
    if physical:
        for m in PHYS:
            res[f'phys{m}'] = [ev(p[-1], t[-1], m / drange) for p, t in roll]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', choices=SYN + ('breach',), required=True)
    ap.add_argument('--enc', choices=('log', 'linear', 'global'), default='log')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=100)
    a = ap.parse_args()
    name = f'exp22_{a.data}_{a.enc}_s{a.seed}'
    js = os.path.join(R, f'result_{name}.json'); ck = os.path.join(R, f'exp15_{name}.pt')
    if os.path.exists(js): sys.exit(f'{js} exists')
    device = get_device(); E.PRECLAMP = True
    if a.data == 'breach':
        c = load_cache(); sc = json.load(open(os.path.join(R, 'scaler.json')))
        tr = encode(breach_rate(inv(c['breach']['train'], sc)), 'log', 1e4)
        tests = {'breach': (encode(breach_rate(inv(c['breach']['test'], sc)), 'log', 1e4), 40, 19, True),
                 'harvey': (inv(c['harvey']['test'], sc), 40, 19, False)}
    else:
        c = pickle.load(open(os.path.join(ROOT, 'data', 'synth_src_cache.pkl'), 'rb'))
        raw = {m: {k: inv(c['data'][m][k], c['scaler']) for k in ('train', 'test')} for m in SYN}
        for m in SYN:
            for k in ('train', 'test'):
                for s in raw[m][k]: s['nodes_dynamic'][:, :, 3] = np.expm1(s['nodes_dynamic'][:, :, 3]) * R0
        tr = encode(raw[a.data]['train'], a.enc, 2500.0)
        tests = {m: (encode(raw[m]['test'], a.enc, 2500.0), 30, 15, True) for m in SYN}
    lo, rg = fit(tr)
    train = apply(tr, lo, rg)
    t0 = time.time()
    if not os.path.exists(ck):
        set_seed(a.seed); m = build_model('vector', 64, 3, device)
        E.train_multistep(m, train, device, a.epochs, 8, ck + '.partial')
        os.replace(ck + '.partial', ck)
    m = build_model('vector', 64, 3, device)
    m.load_state_dict(torch.load(ck, map_location=device, weights_only=True)); m.eval()
    out = {'config': vars(a), 'scaler_min': lo.tolist(), 'scaler_range': rg.tolist(), 'scores': {}}
    with torch.no_grad():
        for te, (smp, st, n, phys) in tests.items():
            out['scores'][te] = score(m, apply(smp, lo, rg), device, st, n, float(rg[0]), phys)
            r = out['scores'][te]
            print(f"  {'HOME' if te == a.data else 'away'} {te:11s} ETS@rel1 {np.nanmedian([e['ets'] for e in r['rel1']]):+.3f} "
                  f"ETS@rel10 {np.nanmedian([e['ets'] for e in r['rel10']]):+.3f} FB@rel10 {np.nanmedian([e['fb'] for e in r['rel10']]):.2f} "
                  f"vol {np.nanmedian(r['volume']):.2f}x", flush=True)
    json.dump(out, open(js, 'w'))
    print(f"  saved {js}  [{(time.time() - t0) / 60:.1f} min]")


if __name__ == '__main__':
    main()
