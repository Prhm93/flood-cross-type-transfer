#!/usr/bin/env python3
"""
exp9_stage2_train_inflow.py
Section 3: train the inflow mechanism (5 seeds) with EXACTLY the exp3 v2 call,
then score every synthetic model on every synthetic test set (3 x 3 matrix).
Point and distributed models are the saved synth2 checkpoints, not retrained.
--quick writes only to results/_smoke_exp9/. Never overwrites results.
"""
import os, sys, json, time, pickle, argparse
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.experiment_lib import set_seed, train_model, evaluate, build_model, get_device

RESULTS = os.path.join(ROOT, 'results')
CACHE = os.path.join(ROOT, 'data', 'synth3_cache.pkl')
STORED = os.path.join(RESULTS, 'result_exp3_v2_stronger.json')
MECHS = ['point', 'distributed', 'inflow']
EPOCHS, N_TRAIN, N_TEST, ROLL = 100, 60, 15, 30      # exp3 v2 settings
HOME_FLOOR = 0.15
TOL = 1e-3


def load_ckpt(model, path, device):
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict):
        for k in ('model_state_dict', 'state_dict', 'model'):
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break
    model.load_state_dict(obj)


def med_iqr(v):
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return float('nan'), float('nan'), float('nan')
    return float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='2 epochs, seed 0, smoke folder')
    args = ap.parse_args()
    epochs, seeds = (2, [0]) if args.quick else (EPOCHS, [0, 1, 2, 3, 4])
    out_dir = os.path.join(RESULTS, '_smoke_exp9') if args.quick else RESULTS
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, 'result_exp9_inflow_matrix.json')
    if os.path.exists(out_json):
        print(f"{out_json} exists - refusing to overwrite. Rename it first.")
        return

    t0 = time.time()
    with open(CACHE, 'rb') as f:
        data = pickle.load(f)['data']
    device = get_device()
    print("=" * 70)
    print(f"  EXP9 SECTION 3 - {'SMOKE TEST' if args.quick else 'FULL RUN'}  "
          f"epochs {epochs}, seeds {seeds}, device {device}")
    print("=" * 70, flush=True)

    # ---- train inflow, exp3 v2 call ----
    for seed in seeds:
        ck = os.path.join(out_dir, f'synth3_inflow_s{seed}.pt')
        if os.path.exists(ck):
            print(f"\n  inflow seed {seed}: finished checkpoint exists, not retrained")
            continue
        tmp = ck + '.partial'
        print(f"\n  --- train on inflow, seed {seed} ---", flush=True)
        set_seed(seed)
        model = build_model('vector', 64, 3, device)
        train_model(model, data['inflow']['train'], device, epochs=epochs,
                    max_train_samples=N_TRAIN, ckpt_path=tmp,
                    log_every=max(1, epochs // 4))
        if not os.path.exists(tmp):
            raise RuntimeError(f"train_model did not write {tmp}")
        os.replace(tmp, ck)
        print(f"    saved {ck}  ({time.time() - t0:.0f}s elapsed)", flush=True)

    # ---- 3 x 3 matrix ----
    results = []
    for tr in MECHS:
        for seed in seeds:
            if tr == 'inflow':
                ck = os.path.join(out_dir, f'synth3_inflow_s{seed}.pt')
            else:
                ck = os.path.join(RESULTS, f'synth2_{tr}_s{seed}.pt')
            model = build_model('vector', 64, 3, device)
            load_ckpt(model, ck, device)
            model.eval()
            scores = {}
            with torch.no_grad():
                for te in MECHS:
                    scores[te] = evaluate(model, data[te]['test'], device,
                                          max_steps=ROLL, max_samples=N_TEST)
            home = scores[tr]
            row = {'train_on': tr, 'seed': seed, 'ckpt': os.path.basename(ck),
                   'scores': scores,
                   'home_below_floor': bool(home['csi_final'] < HOME_FLOOR),
                   'degenerate': bool(home['csi_final'] < 0.02 and home['volume_ratio'] < 0.1),
                   'retained_pct': {te: (100.0 * scores[te]['csi_final'] / home['csi_final']
                                         if home['csi_final'] > 0 else float('nan'))
                                    for te in MECHS if te != tr}}
            results.append(row)
            print(f"\n  {tr:11s} s{seed}" + ("  [HOME BELOW FLOOR]" if row['home_below_floor'] else ""))
            for te in MECHS:
                tag = 'HOME' if te == tr else 'away'
                print(f"    {tag} {te:11s} CSI {scores[te]['csi_final']:.4f}  "
                      f"vol {scores[te]['volume_ratio']:.2f}x")

    # ---- regression against stored exp3 v2 (all seeds, home and away) ----
    stored = json.load(open(STORED))['results']
    n_ok, n_all, mism = 0, 0, []
    for r in results:
        if r['train_on'] == 'inflow':
            continue
        ref = next((s for s in stored if s['train_on'] == r['train_on'] and s['seed'] == r['seed']), None)
        if ref is None:
            continue
        pairs = [(r['scores'][r['train_on']], ref['home']), (r['scores'][ref['away']], ref['away_scores'])]
        for now, old in pairs:
            for key in ('csi_final', 'volume_ratio'):
                n_all += 1
                rel = abs(now[key] - old[key]) / max(abs(old[key]), 1e-9)
                if rel < TOL:
                    n_ok += 1
                else:
                    mism.append(f"{r['train_on']} s{r['seed']} {key}: now {now[key]:.5f} stored {old[key]:.5f}")
    print(f"\n  regression vs exp3 v2: {n_ok}/{n_all} values match")
    for m in mism:
        print(f"    MISMATCH {m}")

    # ---- summary: CSI always with volume ratio, medians + IQR ----
    print("\n" + "=" * 70)
    print("  SUMMARY - median [IQR] across seeds")
    print("=" * 70)
    summary = {}
    for tr in MECHS:
        rs = [r for r in results if r['train_on'] == tr]
        n_floor = sum(r['home_below_floor'] for r in rs)
        print(f"\n  trained on {tr}  (home below {HOME_FLOOR} CSI floor in {n_floor}/{len(rs)} seeds)")
        summary[tr] = {'home_below_floor': n_floor, 'n': len(rs)}
        for te in MECHS:
            c = med_iqr([r['scores'][te]['csi_final'] for r in rs])
            v = med_iqr([r['scores'][te]['volume_ratio'] for r in rs])
            line = (f"    {'HOME' if te == tr else 'away'} {te:11s} CSI {c[0]:.4f} [{c[1]:.4f}, {c[2]:.4f}]  "
                    f"vol {v[0]:.2f}x [{v[1]:.2f}, {v[2]:.2f}]")
            entry = {'csi': c, 'volume_ratio': v}
            if te != tr:
                rt = med_iqr([r['retained_pct'][te] for r in rs])
                line += f"  retained {rt[0]:.1f}% [{rt[1]:.1f}, {rt[2]:.1f}]"
                entry['retained_pct'] = rt
            print(line)
            summary[tr][te] = entry

    with open(out_json, 'w') as f:
        json.dump({'config': dict(epochs=epochs, seeds=seeds, n_train=N_TRAIN, n_test=N_TEST,
                                  rollout_steps=ROLL, home_floor=HOME_FLOOR, quick=args.quick),
                   'results': results, 'summary': summary,
                   'regression_vs_exp3v2': {'matched': n_ok, 'total': n_all, 'mismatches': mism}},
                  f, indent=2, default=str)
    print(f"\n  saved {out_json}  (total {time.time() - t0:.0f}s)")


if __name__ == '__main__':
    main()
