#!/usr/bin/env python3
"""
exp9_section4_metrics.py
Section 4: metric checks on the 3 x 3 synthetic matrix. No training.
1. Print the exact threshold rule (auto_threshold source) and scorer time step
2. CSI saturation: share of truly wet cells, and CSI of an 'everything wet' guess
3. All 15 models x 3 test sets: CSI + volume ratio (must match Section 3), arrival
   (minutes), per-event scores, bad disagreements, Spearman, pairwise disagreement
Synthetic frames are 120 s apart (save_every = 2400 // 40 = 60 steps x 2 s), but the
scorer assumes 3600 s per frame, so arrival is converted: minutes = frames x 2.
Never overwrites results.
"""
import os, sys, ast, json, math, time, pickle, inspect
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.experiment_lib import (build_model, get_device, rollout_predictions,
                                score_rollouts, volume_ratio, score_per_event)
from src.metrics import auto_threshold, csi

RESULTS = os.path.join(ROOT, 'results')
CACHE = os.path.join(ROOT, 'data', 'synth3_cache.pkl')
SEC3 = os.path.join(RESULTS, 'result_exp9_inflow_matrix.json')
OUT_JSON = os.path.join(RESULTS, 'result_exp9_section4_metrics.json')
MECHS = ['point', 'distributed', 'inflow']
SEEDS = [0, 1, 2, 3, 4]
ROLL, N_TEST = 30, 15
FRAME_S = 120.0          # real seconds between synthetic frames
HOME_FLOOR = 0.15
BAD_VOL = 5.0            # exp6 rule: above-median CSI with volume ratio > 5x
TOL = 1e-3


def extract_func(path, name):
    """Pull one function out of a file without running the rest of the file."""
    src = open(path).read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            seg = ast.get_source_segment(src, node)
            ns = {'np': np, 'numpy': np, 'math': math, 'itertools': __import__('itertools')}
            exec(seg, ns)
            return ns[name], seg
    raise KeyError(f"{name} not found in {path}")


def ckpt(tr, seed):
    name = f'synth3_inflow_s{seed}.pt' if tr == 'inflow' else f'synth2_{tr}_s{seed}.pt'
    return os.path.join(RESULTS, name)


def load_ckpt(model, path, device):
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict):
        for k in ('model_state_dict', 'state_dict', 'model'):
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break
    model.load_state_dict(obj)


def minutes(arrival_s, assumed_dt):
    return float(arrival_s) / assumed_dt * FRAME_S / 60.0


def med(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    return float(np.median(v)) if len(v) else float('nan')


def main():
    if os.path.exists(OUT_JSON):
        print(f"{OUT_JSON} exists - refusing to overwrite. Rename it first.")
        return
    t0 = time.time()
    with open(CACHE, 'rb') as f:
        data = pickle.load(f)['data']
    device = get_device()
    sec3 = json.load(open(SEC3))['results']

    print("=" * 70)
    print("  EXP9 SECTION 4 - metric checks (no training)")
    print("=" * 70)

    # ---- 1. threshold rule and time step ----
    print("\n[1] auto_threshold() source:\n")
    print(inspect.getsource(auto_threshold))
    sig = inspect.signature(score_rollouts)
    dt_default = sig.parameters['dt_seconds'].default if 'dt_seconds' in sig.parameters else None
    print(f"  score_rollouts signature: {sig}")
    print(f"  score_rollouts default dt_seconds: {dt_default}  (score_per_event uses 3600.0)")
    if dt_default != 3600.0:
        print("  WARNING: default is not 3600 s - check the arrival conversion before using it")
    agg_dt = float(dt_default) if dt_default else 3600.0

    pairwise, pw_src = extract_func(os.path.join(ROOT, 'pairwise_disagreement.py'),
                                    'pairwise_disagreement')
    spearman, sp_src = extract_func(os.path.join(ROOT, 'scripts', 'exp6_metric_disagreement.py'),
                                    'spearman')
    print("\n  pairwise_disagreement() used as-is:\n" + pw_src)
    print("\n  spearman() used as-is:\n" + sp_src)

    # ---- 2. saturation check (truth does not depend on the model) ----
    print("\n[2] CSI saturation check on each test set")
    m0 = build_model('vector', 64, 3, device)
    load_ckpt(m0, ckpt('distributed', 0), device)
    m0.eval()
    sat = {}
    with torch.no_grad():
        for te in MECHS:
            roll = rollout_predictions(m0, data[te]['test'], device, ROLL, N_TEST)
            wet, allwet = [], []
            for pred, true in roll:
                thr = auto_threshold(true[:, np.newaxis, :])
                wet.append(float((true[-1] > thr).mean()))
                allwet.append(float(csi(np.full_like(true[-1], np.inf), true[-1], thr)))
            sat[te] = {'median_true_wet_fraction': med(wet),
                       'share_events_99pct_wet': float(np.mean(np.array(wet) >= 0.99)),
                       'median_allwet_csi': med(allwet),
                       'allwet_csi_per_event': allwet, 'wet_fraction_per_event': wet}
            print(f"    {te:11s} truly wet at final step: median {med(wet):.3f} "
                  f"({100*sat[te]['share_events_99pct_wet']:.0f}% of events >= 99% wet) | "
                  f"'everything wet' guess CSI median {med(allwet):.4f}")

    # ---- 3. all models x all test sets ----
    print("\n[3] scoring 15 models x 3 test sets", flush=True)
    rows, mism, n_reg = [], [], 0
    for tr in MECHS:
        for seed in SEEDS:
            model = build_model('vector', 64, 3, device)
            load_ckpt(model, ckpt(tr, seed), device)
            model.eval()
            row = {'train_on': tr, 'seed': seed, 'cells': {}}
            ref_row = next(r for r in sec3 if r['train_on'] == tr and r['seed'] == seed)
            for te in MECHS:
                with torch.no_grad():
                    roll = rollout_predictions(model, data[te]['test'], device, ROLL, N_TEST)
                agg = score_rollouts(roll, 'auto')
                vr = float(volume_ratio(roll))
                ev = score_per_event(roll)
                for key, now in (('csi_final', float(agg['csi_final'])), ('volume_ratio', vr)):
                    old = float(ref_row['scores'][te][key])
                    n_reg += 1
                    if abs(now - old) / max(abs(old), 1e-9) >= TOL:
                        mism.append(f"{tr} s{seed} on {te} {key}: now {now:.5f} sec3 {old:.5f}")
                c = np.array([e['csi_final'] for e in ev], dtype=float)
                v = np.array([e['volume_ratio'] for e in ev], dtype=float)
                ok = np.isfinite(c) & np.isfinite(v) & (v > 0)
                bad = int(np.sum((c[ok] > np.median(c[ok])) & (v[ok] > BAD_VOL))) if ok.any() else 0
                try:
                    r = spearman(c[ok], np.abs(np.log(v[ok])))
                    rho = float(r[0] if isinstance(r, (tuple, list)) else r)
                except Exception as e:
                    rho = float('nan')
                    print(f"    spearman failed: {e}")
                try:
                    _, _, pct = pairwise(ev)
                    pct = float(pct)
                except Exception as e:
                    pct = float('nan')
                    print(f"    pairwise failed: {e}")
                row['cells'][te] = {
                    'csi_final': float(agg['csi_final']), 'volume_ratio': vr,
                    'arrival_min': minutes(agg['arrival_mae_s'], agg_dt),
                    'n_events': len(ev), 'bad_disagreements': bad,
                    'spearman_csi_vs_abslogvol': rho, 'pairwise_disagreement_pct': pct,
                    'events': [dict(e, arrival_min=minutes(e['arrival_mae_s'], 3600.0)) for e in ev]}
            row['home_below_floor'] = bool(row['cells'][tr]['csi_final'] < HOME_FLOOR)
            rows.append(row)
            print(f"    {tr:11s} s{seed} done", flush=True)

    print(f"\n  regression vs Section 3: {n_reg - len(mism)}/{n_reg} values match")
    for m in mism:
        print(f"    MISMATCH {m}")

    # ---- summary ----
    print("\n" + "=" * 70)
    print("  SUMMARY - medians across 5 seeds (arrival in MINUTES, 1 frame = 2 min)")
    print("=" * 70)
    summary = {}
    for tr in MECHS:
        rs = [r for r in rows if r['train_on'] == tr]
        nf = sum(r['home_below_floor'] for r in rs)
        h = {k: med([r['cells'][tr][k] for r in rs])
             for k in ('csi_final', 'volume_ratio', 'arrival_min')}
        print(f"\n  trained on {tr} (home below floor {nf}/5)")
        summary[tr] = {'home_below_floor': nf}
        for te in MECHS:
            cells = [r['cells'][te] for r in rs]
            s = {k: med([c[k] for c in cells]) for k in
                 ('csi_final', 'volume_ratio', 'arrival_min', 'bad_disagreements',
                  'spearman_csi_vs_abslogvol', 'pairwise_disagreement_pct')}
            line = (f"    {'HOME' if te == tr else 'away'} {te:11s} CSI {s['csi_final']:.4f}  "
                    f"vol {s['volume_ratio']:.2f}x  arrival {s['arrival_min']:.1f} min  "
                    f"bad {s['bad_disagreements']:.0f}/{N_TEST}  rho {s['spearman_csi_vs_abslogvol']:+.3f}  "
                    f"pairwise {s['pairwise_disagreement_pct']:.1f}%")
            if te != tr:
                s['csi_flatters'] = bool(s['csi_final'] > h['csi_final'])
                s['volume_flags'] = bool(abs(math.log(s['volume_ratio'])) -
                                         abs(math.log(h['volume_ratio'])) > math.log(2))
                s['arrival_flags'] = bool(s['arrival_min'] > 1.2 * h['arrival_min'])
                line += (f"\n         three-way: CSI flatters {s['csi_flatters']} | "
                         f"volume flags failure {s['volume_flags']} | "
                         f"arrival flags failure {s['arrival_flags']}")
            print(line)
            summary[tr][te] = s

    with open(OUT_JSON, 'w') as f:
        json.dump({'config': dict(rollout_steps=ROLL, n_test=N_TEST, frame_seconds=FRAME_S,
                                  scorer_dt_default=dt_default, home_floor=HOME_FLOOR,
                                  bad_vol_rule=BAD_VOL),
                   'auto_threshold_source': inspect.getsource(auto_threshold),
                   'saturation': sat, 'rows': rows, 'summary': summary,
                   'regression_vs_section3': {'matched': n_reg - len(mism), 'total': n_reg,
                                              'mismatches': mism}},
                  f, indent=2, default=str)
    print(f"\n  saved {OUT_JSON}  ({time.time() - t0:.0f}s)")


if __name__ == '__main__':
    main()
