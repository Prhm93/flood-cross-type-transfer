#!/usr/bin/env python3
"""
exp15_retrain.py - retrain one model with a stronger recipe, score it with the full panel.
Recipe 'allsamples': same model and loss, every training sample in every epoch
(max_train_samples=None), more epochs. Tests whether undertraining explains the weak models.
Gate on the home test set: ETS >= 0.10, skill S > 0, volume ratio within 0.5-2.
--quick = 2 epochs into results/_smoke_exp15/. Never overwrites.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.experiment_lib import (load_cache, build_model, get_device, set_seed, train_model,
                                rollout_predictions, score_rollouts, volume_ratio)
from src.metrics import auto_threshold, csi

R = os.path.join(ROOT, 'results')


def ets(pred, true, thr):
    p, t = pred > thr, true > thr
    H, F, M, N = np.sum(p & t), np.sum(p & ~t), np.sum(~p & t), p.size
    hr = (H + F) * (H + M) / N
    den = H + F + M - hr
    return float((H - hr) / den) if den > 0 else float('nan')


def panel(model, samples, device):
    roll = rollout_predictions(model, samples, device, 40, 19)
    r = score_rollouts(roll, 'auto')
    aw, e = [], []
    for p, t in roll:
        thr = auto_threshold(t[:, np.newaxis, :])
        aw.append(float(csi(np.full_like(t[-1], np.inf), t[-1], thr)))
        e.append(ets(p[-1], t[-1], thr))
    c, a = float(r['csi_final']), float(np.median(aw))
    return {'csi': c, 'allwet_csi': a, 'S': (c - a) / (1 - a) if a < 1 else float('nan'),
            'ets': float(np.nanmedian(e)), 'volume_ratio': float(volume_ratio(roll)),
            'arrival_h': float(r['arrival_mae_s']) / 3600.0}


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
    ap.add_argument('--dataset', choices=['breach', 'harvey'], required=True)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--max-train-samples', type=int, default=None, help='None = all samples every epoch')
    ap.add_argument('--tag', default='allsamples')
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    if a.quick:
        a.epochs = 2
    out_dir = os.path.join(R, '_smoke_exp15') if a.quick else R
    os.makedirs(out_dir, exist_ok=True)
    name = f'exp15_{a.dataset}_{a.tag}_h{a.hidden}_s{a.seed}'
    ck, js = os.path.join(out_dir, name + '.pt'), os.path.join(out_dir, f'result_{name}.json')
    if os.path.exists(js):
        sys.exit(f'{js} exists - not overwritten')

    cache, device = load_cache(), get_device()
    other = 'harvey' if a.dataset == 'breach' else 'breach'
    t0 = time.time()
    if not os.path.exists(ck):
        tmp = ck + '.partial'
        set_seed(a.seed)
        m = build_model('vector', a.hidden, a.layers, device)
        print(f"training {name}: {len(cache[a.dataset]['train'])} samples, "
              f"{'all' if a.max_train_samples is None else a.max_train_samples} per epoch, {a.epochs} epochs", flush=True)
        train_model(m, cache[a.dataset]['train'], device, epochs=a.epochs,
                    max_train_samples=a.max_train_samples, ckpt_path=tmp,
                    log_every=max(1, a.epochs // 10))
        if not os.path.exists(tmp):
            print("  train_model saved no checkpoint - saving final weights instead")
            torch.save(m.state_dict(), tmp)
        os.replace(tmp, ck)
    train_min = (time.time() - t0) / 60
    m = build_model('vector', a.hidden, a.layers, device)
    load_ckpt(m, ck, device)
    m.eval()
    with torch.no_grad():
        res = {te: panel(m, cache[te]['test'], device) for te in (a.dataset, other)}
    for te in (a.dataset, other):
        r = res[te]
        print(f"  {'HOME' if te == a.dataset else 'away'} {te:7s} CSI {r['csi']:.4f} (all-wet {r['allwet_csi']:.4f}, "
              f"S {r['S']:+.3f})  ETS {r['ets']:+.4f}  vol {r['volume_ratio']:.2f}x  arrival {r['arrival_h']:.1f} h")
    h = res[a.dataset]
    gate = h['ets'] >= 0.10 and h['S'] > 0 and 0.5 <= h['volume_ratio'] <= 2.0
    print(f"\n  GATE (home ETS >= 0.10, S > 0, volume 0.5-2x): {'PASS' if gate else 'FAIL'}   "
          f"[training took {train_min:.1f} min]")
    json.dump({'config': vars(a), 'train_minutes': train_min, 'scores': res, 'gate': gate},
              open(js, 'w'), indent=2)
    print(f"  saved {js}")


if __name__ == '__main__':
    main()
