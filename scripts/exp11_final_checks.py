#!/usr/bin/env python3
"""
exp11_final_checks.py - items 1-4 before submission. No training.
1a. What exp5 (large model) stored, and how train.py calls train_model
1b. CSI by rollout step vs a trivial 'flood everything' prediction, real test sets
2.  Per-event recount of 'good CSI, bad volume': raw CSI (v3 rule) and skill S
3.  The [UNVERIFIED] v3 numbers: parameter counts and stored JSON values
4.  Table 16 for seeds 0-2: CSI, volume ratio, arrival (hours), both directions
Writes results/result_exp11_final_checks.json (never overwrites).
"""
import os, sys, re, glob, json, inspect
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.experiment_lib import (load_cache, build_model, get_device, rollout_predictions,
                                score_rollouts, volume_ratio, score_per_event, train_model)
from src.metrics import auto_threshold, csi
from src.model import FloodGNN

RESULTS = os.path.join(ROOT, 'results')
OUT_JSON = os.path.join(RESULTS, 'result_exp11_final_checks.json')
REAL, SEEDS, SHOW = ['breach', 'harvey'], [0, 1, 2], [5, 10, 20, 30, 40]


def load_ckpt(model, path, device):
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict):
        for k in ('model_state_dict', 'state_dict', 'model'):
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break
    model.load_state_dict(obj)


def flat(o, pre=''):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flat(v, f'{pre}.{k}' if pre else str(k))
    elif isinstance(o, list):
        if len(o) <= 12:
            for i, v in enumerate(o):
                yield from flat(v, f'{pre}[{i}]')
        else:
            yield pre, f'list of {len(o)}'
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield pre, o


def dump(path, pattern, limit=60):
    if not os.path.exists(path):
        print(f"    MISSING {path}")
        return
    n = 0
    for k, v in flat(json.load(open(path))):
        if re.search(pattern, k, re.I):
            print(f"    {k} = {v}")
            n += 1
            if n >= limit:
                print("    ... (cut)")
                return


def med(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    return float(np.median(x)) if len(x) else float('nan')


def skill(c, a):
    return (c - a) / (1 - a) if a < 1 else float('nan')


def allwet_csi(true_step, thr):
    return float(csi(np.full_like(true_step, np.inf), true_step, thr))


def main():
    if os.path.exists(OUT_JSON):
        print(f"{OUT_JSON} exists - refusing to overwrite. Rename it first.")
        return
    out = {}

    # ---- 3. UNVERIFIED numbers (cheap, first) ----
    print("=" * 70 + "\n[3] UNVERIFIED v3 numbers\n" + "=" * 70)
    for var in ('vector', 'scalar'):
        n = sum(p.numel() for p in FloodGNN(variant=var, hidden=64, num_layers=3).parameters())
        out[f'params_{var}'] = n
        print(f"    {var} model parameters: {n:,}   (v3 says vector 42,371; scalar 42,339)")
    for f in sorted(glob.glob(os.path.join(RESULTS, 'result_harvey_*_s*.json'))):
        print(f"  {os.path.basename(f)}")
        dump(f, r'csi|retain|volume', 20)
    print("  result_exp2_rainfall_ablation.json")
    dump(os.path.join(RESULTS, 'result_exp2_rainfall_ablation.json'), r'retain|csi', 60)

    # ---- 1a. exp5 and how training was called ----
    print("\n" + "=" * 70 + "\n[1a] exp5 large model: stored values\n" + "=" * 70)
    dump(os.path.join(RESULTS, 'result_exp5_stronger_model.json'),
         r'harvey|home|csi|retain|param|hidden', 80)
    print("    look for any Harvey HOME CSI above 0.6309 (the all-wet score)")
    print("  checkpoints that look like exp5:",
          [os.path.basename(p) for p in glob.glob(os.path.join(RESULTS, '*.pt'))
           if re.search(r'large|strong|exp5|big', p, re.I)])
    print("  train_model signature:", inspect.signature(train_model))
    tp = os.path.join(ROOT, 'scripts', 'train.py')
    if os.path.exists(tp):
        for i, l in enumerate(open(tp).read().splitlines(), 1):
            if 'train_model(' in l or 'max_train' in l or 'epochs' in l:
                print(f"    train.py L{i}: {l.strip()[:120]}")

    # ---- rollouts for all real models ----
    print("\n  loading cache (about a minute) ...", flush=True)
    cache = load_cache()
    device = get_device()
    rolls = {}
    with torch.no_grad():
        for tr in REAL:
            for s in SEEDS:
                p = os.path.join(RESULTS, f'best_{tr}_vector_s{s}.pt')
                if not os.path.exists(p):
                    print(f"    missing {p}")
                    continue
                m = build_model('vector', 64, 3, device)
                load_ckpt(m, p, device)
                m.eval()
                for te in REAL:
                    rolls[(tr, s, te)] = rollout_predictions(m, cache[te]['test'], device, 40, 19)

    # ---- 1b. CSI by step ----
    print("\n" + "=" * 70 + "\n[1b] CSI by rollout step vs flooding everything (S = skill vs all-wet)\n" + "=" * 70)
    out['per_step'] = {}
    for te in REAL:
        ref = rolls[next(k for k in rolls if k[2] == te)]
        T = ref[0][1].shape[0]
        steps = [st for st in SHOW if st <= T]
        aw = {st: med([allwet_csi(t[st - 1], auto_threshold(t[:, np.newaxis, :])) for _, t in ref])
              for st in steps}
        print(f"\n  test {te} ({T} frames): all-wet CSI by step " +
              "  ".join(f"s{st}={aw[st]:.3f}" for st in steps))
        out['per_step'][te] = {'allwet': aw}
        for tr in REAL:
            line = f"    {tr}-trained:"
            res = {}
            for st in steps:
                seed_meds = []
                for s in SEEDS:
                    if (tr, s, te) in rolls:
                        seed_meds.append(med([float(csi(p[st - 1], t[st - 1], auto_threshold(t[:, np.newaxis, :])))
                                              for p, t in rolls[(tr, s, te)]]))
                c = med(seed_meds)
                res[st] = {'csi': c, 'S': skill(c, aw[st])}
                line += f"  s{st} CSI {c:.3f} S {skill(c, aw[st]):+.3f}"
            print(line)
            out['per_step'][te][tr] = res

    # ---- 2. per-event recount ----
    print("\n" + "=" * 70 + "\n[2] per-event recount (bad volume = above 5x or below 0.2x)\n" + "=" * 70)
    out['recount'] = {}
    for tr in REAL:
        for te in REAL:
            for s in SEEDS:
                key = (tr, s, te)
                if key not in rolls:
                    continue
                ev = score_per_event(rolls[key])
                c = np.array([e['csi_final'] for e in ev], dtype=float)
                v = np.array([e['volume_ratio'] for e in ev], dtype=float)
                a = np.array([allwet_csi(t[-1], auto_threshold(t[:, np.newaxis, :])) for _, t in rolls[key]])
                S = np.where(a < 1, (c - a) / np.where(a < 1, 1 - a, 1), np.nan)
                bad = (v > 5) | (v < 0.2)
                r = {'n': len(ev), 'raw_rule': int(np.sum((c > np.median(c)) & bad)),
                     'S_rule': int(np.sum((S > np.nanmedian(S)) & bad)),
                     'beat_allwet': int(np.sum(S > 0)),
                     'beat_allwet_and_bad_volume': int(np.sum((S > 0) & bad))}
                out['recount'][f'{tr}->{te}_s{s}'] = r
                print(f"    {tr}->{te} s{s}: raw rule {r['raw_rule']}/{r['n']} | S rule {r['S_rule']} | "
                      f"beat all-wet {r['beat_allwet']} | beat all-wet AND bad volume {r['beat_allwet_and_bad_volume']}")
    print("    CHECK: breach->harvey s0 raw rule should be 9/19 (v3 Table 15)")

    # ---- 4. Table 16, seeds 0-2 ----
    print("\n" + "=" * 70 + "\n[4] Table 16 for seeds 0-2 (arrival in hours)\n" + "=" * 70)
    out['table16'] = {}
    for tr in REAL:
        for s in SEEDS:
            for te in REAL:
                if (tr, s, te) not in rolls:
                    continue
                r = score_rollouts(rolls[(tr, s, te)], 'auto')
                row = {'csi': float(r['csi_final']), 'volume_ratio': float(volume_ratio(rolls[(tr, s, te)])),
                       'arrival_h': float(r['arrival_mae_s']) / 3600.0}
                out['table16'][f'{tr}_s{s}_on_{te}'] = row
                print(f"    {tr} s{s} on {te}: CSI {row['csi']:.3f}  vol {row['volume_ratio']:.2f}x  "
                      f"arrival {row['arrival_h']:.1f} h")
    print("  medians across seeds, with verdict (away vs own):")
    for tr in REAL:
        other = [x for x in REAL if x != tr][0]
        h = {k: med([out['table16'][f'{tr}_s{s}_on_{tr}'][k] for s in SEEDS if f'{tr}_s{s}_on_{tr}' in out['table16']])
             for k in ('csi', 'volume_ratio', 'arrival_h')}
        a = {k: med([out['table16'][f'{tr}_s{s}_on_{other}'][k] for s in SEEDS if f'{tr}_s{s}_on_{other}' in out['table16']])
             for k in ('csi', 'volume_ratio', 'arrival_h')}
        vflag = abs(np.log(a['volume_ratio'])) > abs(np.log(h['volume_ratio']))
        print(f"    {tr}: own CSI {h['csi']:.3f} vol {h['volume_ratio']:.2f}x arr {h['arrival_h']:.1f} h | "
              f"other CSI {a['csi']:.3f} vol {a['volume_ratio']:.2f}x arr {a['arrival_h']:.1f} h")
        print(f"      CSI flatters: {a['csi'] > h['csi']} | volume reports failure: {vflag} | "
              f"arrival reports failure: {a['arrival_h'] > 1.2 * h['arrival_h']}")
    print("  v3 seed-0: breach own 0.165/2.29x/12.5h, other 0.561/131.0x/15.4h; "
          "harvey own 0.489/12.48x/14.2h, other 0.185/0.33x/13.8h")

    json.dump(out, open(OUT_JSON, 'w'), indent=2, default=str)
    print(f"\n  saved {OUT_JSON}")


if __name__ == '__main__':
    main()
