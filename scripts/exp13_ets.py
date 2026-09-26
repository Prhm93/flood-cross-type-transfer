#!/usr/bin/env python3
"""
exp13_ets.py - equitable threat score (ETS, Gilbert skill score) for every model
and test set. No training.
  ETS = (H - Hr) / (H + F + M - Hr),  Hr = (H + F)(H + M) / N
per event at the final rollout step, same threshold as CSI (1% of the event's
maximum true depth). A prediction that floods every cell scores ETS = 0.
Also prints the training settings stored in the main result files.
Writes results/result_exp13_ets.json (never overwrites).
"""
import os, sys, json, pickle
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.experiment_lib import load_cache, build_model, get_device, rollout_predictions
from src.metrics import auto_threshold

R = os.path.join(ROOT, 'results')
OUT = os.path.join(R, 'result_exp13_ets.json')


def load_ckpt(model, path, device):
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict):
        for k in ('model_state_dict', 'state_dict', 'model'):
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break
    model.load_state_dict(obj)


def ets(pred, true, thr):
    p, t = pred > thr, true > thr
    H, F, M, N = np.sum(p & t), np.sum(p & ~t), np.sum(~p & t), p.size
    hr = (H + F) * (H + M) / N
    den = H + F + M - hr
    return float((H - hr) / den) if den > 0 else float('nan')


def score(model, samples, device, steps, n):
    roll = rollout_predictions(model, samples, device, steps, n)
    return float(np.nanmedian([ets(p[-1], t[-1], auto_threshold(t[:, np.newaxis, :])) for p, t in roll]))


def main():
    if os.path.exists(OUT):
        sys.exit(f"{OUT} exists - not overwritten")

    print("=" * 70 + "\n[settings] stored in the main result files\n" + "=" * 70)
    for f in ('result_breach_vector_s0.json', 'result_harvey_vector_s0.json'):
        d = json.load(open(os.path.join(R, f)))
        print(f"  {f}: keys {list(d)[:20]}")
        for k in ('config', 'args', 'params', 'hyperparameters', 'settings'):
            if k in d:
                print(f"    {k}: {json.dumps(d[k])[:700]}")

    device = get_device()
    out = {}
    print("\n  loading cache (about a minute) ...", flush=True)
    cache = load_cache()
    real = [('small', 'best_{tr}_vector_s{s}.pt', 64, 3, range(3)),
            ('large', 'arch_large_{tr}_s{s}.pt', 160, 6, range(5))]
    print("\n" + "=" * 70 + "\n[real] median ETS over 19 events, final step (all-wet = 0)\n" + "=" * 70)
    with torch.no_grad():
        for size, pat, h, L, seeds in real:
            for tr in ('breach', 'harvey'):
                for s in seeds:
                    p = os.path.join(R, pat.format(tr=tr, s=s))
                    if not os.path.exists(p):
                        continue
                    m = build_model('vector', h, L, device)
                    load_ckpt(m, p, device)
                    m.eval()
                    row = {te: score(m, cache[te]['test'], device, 40, 19) for te in ('breach', 'harvey')}
                    out[f'{size}_{tr}_s{s}'] = row
                    print(f"    {size:5s} {tr:6s} s{s}: ETS on breach {row['breach']:+.4f} | on harvey {row['harvey']:+.4f}", flush=True)

        print("\n" + "=" * 70 + "\n[synthetic] median ETS over 15 events, final step\n" + "=" * 70)
        data = pickle.load(open(os.path.join(ROOT, 'data', 'synth3_cache.pkl'), 'rb'))['data']
        mechs = ('point', 'distributed', 'inflow')
        for tr in mechs:
            for s in range(5):
                name = f'synth3_inflow_s{s}.pt' if tr == 'inflow' else f'synth2_{tr}_s{s}.pt'
                m = build_model('vector', 64, 3, device)
                load_ckpt(m, os.path.join(R, name), device)
                m.eval()
                row = {te: score(m, data[te]['test'], device, 30, 15) for te in mechs}
                out[f'synth_{tr}_s{s}'] = row
                print(f"    {tr:11s} s{s}: " + "  ".join(f"on {te} {row[te]:+.4f}" for te in mechs), flush=True)

    print("\n" + "=" * 70 + "\n  SUMMARY: median ETS across seeds (own test set marked *)\n" + "=" * 70)
    groups = [('small', 'breach', ('breach', 'harvey')), ('small', 'harvey', ('breach', 'harvey')),
              ('large', 'breach', ('breach', 'harvey')), ('large', 'harvey', ('breach', 'harvey'))]
    groups += [('synth', tr, mechs) for tr in mechs]
    for size, tr, tests in groups:
        rows = [v for k, v in out.items() if k.startswith(f'{size}_{tr}_s')]
        if rows:
            print(f"    {size:5s} {tr:11s} " + "  ".join(
                f"{te}{'*' if te == tr else ' '} {np.nanmedian([r[te] for r in rows]):+.4f}" for te in tests))
    json.dump(out, open(OUT, 'w'), indent=2)
    print(f"\n  saved {OUT}")


if __name__ == '__main__':
    main()
