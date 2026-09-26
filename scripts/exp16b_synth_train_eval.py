#!/usr/bin/env python3
"""
exp16b_synth_train_eval.py - train one synthetic mechanism WITH the source channel
(exp3 v2 settings: 100 epochs, all 60 simulations) and score it on all three test sets.
Panel: CSI, all-wet CSI, skill S, ETS at the final step and at step 15, volume ratio.
Distributed floods saturate at the final step, so their gate uses ETS at step 15.
Gate on the home set: ETS >= 0.10 and volume ratio within 0.5-2. Never overwrites.
"""
import os, sys, json, time, pickle, argparse
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.experiment_lib import (build_model, get_device, set_seed, train_model,
                                rollout_predictions, score_rollouts, volume_ratio)
from src.metrics import auto_threshold, csi

R = os.path.join(ROOT, 'results')
MECHS = ('point', 'distributed', 'inflow')


def ets(pred, true, thr):
    p, t = pred > thr, true > thr
    H, F, M, N = np.sum(p & t), np.sum(p & ~t), np.sum(~p & t), p.size
    hr = (H + F) * (H + M) / N
    den = H + F + M - hr
    return float((H - hr) / den) if den > 0 else float('nan')


def panel(model, samples, device):
    roll = rollout_predictions(model, samples, device, 30, 15)
    r = score_rollouts(roll, 'auto')
    aw, ef, em = [], [], []
    for p, t in roll:
        thr = auto_threshold(t[:, np.newaxis, :])
        aw.append(float(csi(np.full_like(t[-1], np.inf), t[-1], thr)))
        ef.append(ets(p[-1], t[-1], thr))
        em.append(ets(p[15], t[15], thr))
    c, a = float(r['csi_final']), float(np.median(aw))
    return {'csi': c, 'allwet_csi': a, 'S': (c - a) / (1 - a) if a < 1 else float('nan'),
            'ets_final': float(np.nanmedian(ef)), 'ets_step15': float(np.nanmedian(em)),
            'volume_ratio': float(volume_ratio(roll))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mech', choices=MECHS, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=100)
    a = ap.parse_args()
    ck = os.path.join(R, f'exp15_synthsrc_{a.mech}_s{a.seed}.pt')
    js = os.path.join(R, f'result_exp16_synthsrc_{a.mech}_s{a.seed}.json')
    if os.path.exists(js):
        sys.exit(f'{js} exists - not overwritten')
    data = pickle.load(open(os.path.join(ROOT, 'data', 'synth_src_cache.pkl'), 'rb'))['data']
    device = get_device()
    t0 = time.time()
    if not os.path.exists(ck):
        set_seed(a.seed)
        m = build_model('vector', 64, 3, device)
        train_model(m, data[a.mech]['train'], device, epochs=a.epochs, max_train_samples=60,
                    ckpt_path=ck + '.partial', log_every=max(1, a.epochs // 5))
        os.replace(ck + '.partial', ck)
    m = build_model('vector', 64, 3, device)
    m.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
    m.eval()
    with torch.no_grad():
        res = {te: panel(m, data[te]['test'], device) for te in MECHS}
    for te in MECHS:
        r = res[te]
        print(f"  {'HOME' if te == a.mech else 'away'} {te:11s} CSI {r['csi']:.4f} (all-wet {r['allwet_csi']:.4f}, "
              f"S {r['S']:+.3f})  ETS final {r['ets_final']:+.3f}  ETS s15 {r['ets_step15']:+.3f}  "
              f"vol {r['volume_ratio']:.2f}x")
    h = res[a.mech]
    e = h['ets_step15'] if a.mech == 'distributed' else h['ets_final']
    gate = bool(e >= 0.10 and 0.5 <= h['volume_ratio'] <= 2.0)
    print(f"\n  GATE ({'ETS s15' if a.mech == 'distributed' else 'ETS final'} >= 0.10, volume 0.5-2x): "
          f"{'PASS' if gate else 'FAIL'}   [{(time.time() - t0) / 60:.1f} min]")
    json.dump({'config': vars(a), 'scores': res, 'gate': gate}, open(js, 'w'), indent=2)


if __name__ == '__main__':
    main()
