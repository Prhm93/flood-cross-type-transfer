#!/usr/bin/env python3
"""
exp17_multistep.py - recipe 3: multi-step (rollout) training + source channel.
The model is rolled forward on its OWN predictions for k steps and the loss compares
every step with the truth. k grows from 1 to kmax over the first half of training.
Same model, optimiser, learning rate and scheduler as before.
  point / distributed / inflow : data/synth_src_cache.pkl (source channel included)
  breach : main cache; channel 3 gets the breach source recovered from the simulation
           (first wet cell; outflow = rise in total water per hour), log1p(rate/1e-6),
           scaled by the training-set maximum.
Scored with the same rollout as every earlier experiment. Never overwrites.
"""
import os, sys, json, time, pickle, argparse
import numpy as np
import torch
import torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.experiment_lib import (load_cache, build_model, get_device, set_seed,
                                rollout_predictions, score_rollouts, volume_ratio)
from src.metrics import auto_threshold, csi

R = os.path.join(ROOT, 'results')
SYN = ('point', 'distributed', 'inflow')


def ets(pred, true, thr):
    p, t = pred > thr, true > thr
    H, Fa, M, N = np.sum(p & t), np.sum(p & ~t), np.sum(~p & t), p.size
    hr = (H + Fa) * (H + M) / N
    den = H + Fa + M - hr
    return float((H - hr) / den) if den > 0 else float('nan')


def panel(model, samples, device, steps, n):
    roll = rollout_predictions(model, samples, device, steps, n)
    r = score_rollouts(roll, 'auto')
    aw, ef = [], []
    for p, t in roll:
        thr = auto_threshold(t[:, np.newaxis, :])
        aw.append(float(csi(np.full_like(t[-1], np.inf), t[-1], thr)))
        ef.append(ets(p[-1], t[-1], thr))
    c, a = float(r['csi_final']), float(np.median(aw))
    return {'csi': c, 'allwet_csi': a, 'S': (c - a) / (1 - a) if a < 1 else float('nan'),
            'ets': float(np.nanmedian(ef)), 'volume_ratio': float(volume_ratio(roll)),
            'arrival_h': float(r['arrival_mae_s']) / 3600.0}


def breach_source(samples, depth_max_m, scale=None):
    """Channel 3 = breach source recovered from the simulation (cells 100 m, steps 1 h)."""
    logs = []
    for s in samples:
        d = s['nodes_dynamic'][:, :, 0] * depth_max_m          # metres
        N, T = d.shape
        vol = d.sum(0) * 1e4                                   # m3 per frame
        q = np.maximum(np.diff(vol), 0.0) / 3600.0             # m3/s into the domain
        wet = [t for t in range(T) if d[:, t].max() > 1e-3]
        cell = int(np.argmax(d[:, wet[0]])) if wet else 0
        src = np.zeros((N, T), dtype=np.float64)
        rate = np.append(q, q[-1] if len(q) else 0.0) / 1e4    # m/s at the breach cell
        src[cell, :] = np.log1p(rate / 1e-6)
        logs.append(src)
    if scale is None:
        scale = max(float(x.max()) for x in logs) or 1.0
    out = []
    for s, src in zip(samples, logs):
        dyn = s['nodes_dynamic'].copy()
        dyn[:, :, 3] = (src / scale).astype(np.float32)
        out.append(dict(s, nodes_dynamic=dyn))
    return out, scale


PRECLAMP = False


def step(model, cur, sta, ei, src_next):
    delta = model(cur, sta, ei)
    raw = cur[:, :3] + delta
    return torch.cat([torch.relu(raw[:, 0:1]), raw[:, 1:3], src_next], dim=1), raw


def train_multistep(model, samples, device, epochs, kmax, ckpt, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    best, saved = float('inf'), False
    ramp = max(1, epochs // 2)
    for ep in range(epochs):
        k = min(kmax, 1 + (ep * kmax) // ramp)
        model.train()
        tot, n = 0.0, 0
        for idx in np.random.permutation(len(samples)):
            s = samples[idx]
            dyn = torch.as_tensor(s['nodes_dynamic'], dtype=torch.float32, device=device)
            sta = torch.as_tensor(s['nodes_static'], dtype=torch.float32, device=device)
            ei = torch.as_tensor(s['edge_index'], dtype=torch.long, device=device)
            T = dyn.shape[1]
            if T <= k:
                continue
            t0 = np.random.randint(0, T - k)
            cur, loss = dyn[:, t0, :], 0.0
            for j in range(k):
                cur, raw = step(model, cur, sta, ei, dyn[:, t0 + j + 1, 3:4])
                loss = loss + F.mse_loss(raw if PRECLAMP else cur[:, :3], dyn[:, t0 + j + 1, :3])
            loss = loss / k
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot, n = tot + float(loss.detach()), n + 1
        ep_loss = tot / max(n, 1)
        sched.step(ep_loss)
        if k == kmax and ep_loss < best:
            best = ep_loss
            torch.save(model.state_dict(), ckpt)
            saved = True
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"    epoch {ep:3d}  k={k}  loss={ep_loss:.6f}  lr={opt.param_groups[0]['lr']:.1e}", flush=True)
    if not saved:
        torch.save(model.state_dict(), ckpt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', choices=SYN + ('breach', 'harvey'), required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--kmax', type=int, default=8)
    ap.add_argument('--preclamp', action='store_true')
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--layers', type=int, default=3)
    a = ap.parse_args()
    global PRECLAMP
    PRECLAMP = a.preclamp
    tag = '_pc' if a.preclamp else ''
    if (a.hidden, a.layers) != (64, 3):
        tag += f'_h{a.hidden}L{a.layers}'
    ck = os.path.join(R, f'exp15_ms_{a.data}_k{a.kmax}{tag}_s{a.seed}.pt')
    js = os.path.join(R, f'result_exp17_ms_{a.data}_k{a.kmax}{tag}_s{a.seed}.json')
    if os.path.exists(js):
        sys.exit(f'{js} exists - not overwritten')
    device = get_device()
    if a.data in ('breach', 'harvey'):
        cache = load_cache()
        dmax = float(json.load(open(os.path.join(R, 'scaler.json')))['dynamic_max'][0])
        train, scale = breach_source(cache['breach']['train'], dmax)
        if a.data == 'harvey':
            train = cache['harvey']['train']
        tests = {'breach': (breach_source(cache['breach']['test'], dmax, scale)[0], 40, 19),
                 'harvey': (cache['harvey']['test'], 40, 19)}
        print(f"  breach source channel added (log-scale max {scale:.2f})")
    else:
        data = pickle.load(open(os.path.join(ROOT, 'data', 'synth_src_cache.pkl'), 'rb'))['data']
        train = data[a.data]['train']
        tests = {m: (data[m]['test'], 30, 15) for m in SYN}
    t0 = time.time()
    if not os.path.exists(ck):
        set_seed(a.seed)
        m = build_model('vector', a.hidden, a.layers, device)
        train_multistep(m, train, device, a.epochs, a.kmax, ck + '.partial')
        os.replace(ck + '.partial', ck)
    m = build_model('vector', a.hidden, a.layers, device)
    m.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
    m.eval()
    with torch.no_grad():
        res = {te: panel(m, smp, device, st, n) for te, (smp, st, n) in tests.items()}
    for te, r in res.items():
        print(f"  {'HOME' if te == a.data else 'away'} {te:11s} CSI {r['csi']:.4f} (all-wet {r['allwet_csi']:.4f}, "
              f"S {r['S']:+.3f})  ETS {r['ets']:+.4f}  vol {r['volume_ratio']:.2f}x  arrival {r['arrival_h']:.2f}")
    h = res[a.data]
    gate = bool((a.data == 'distributed' or h['ets'] >= 0.10) and 0.5 <= h['volume_ratio'] <= 2.0)
    print(f"\n  GATE: {'PASS' if gate else 'FAIL'}   [{(time.time() - t0) / 60:.1f} min]")
    json.dump({'config': vars(a), 'scores': res, 'gate': gate}, open(js, 'w'), indent=2)


if __name__ == '__main__':
    main()
