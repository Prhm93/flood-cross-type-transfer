#!/usr/bin/env python3
"""
exp9_stage1_inflow_data.py
Stage 1 of the third controlled mechanism (upstream inflow). No training.

1. Check B: time-varying solver with no hydrograph == original exp3 solver, exactly
2. Regenerate exp3 v2 point + distributed data (same code, same seeds)
3. Rebuild the exp3 v2 joint scaler from that SAME data (no new data in the fit)
4. Check A: re-score saved synth2_*_s0 checkpoints, compare with stored JSON
5. Generate the inflow mechanism: same terrain seeds, same total water volume
6. Scale inflow with the exp3 v2 scaler, report out-of-range values
7. Leakage audit, then save a cache (never overwrites)
Prints PASS or FAIL at the end. Do not train if it says FAIL.
"""
import os, sys, json, time, pickle
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.experiment_lib import evaluate, build_model, get_device
from src.unified_loader import _grid_edges
from scripts.exp3_controlled_synthetic import (make_terrain, simulate, generate_dataset,
                                               source_point, G)

# exp3 v2 settings, copied from exp3_v2_stronger.py defaults
DIM, N_TRAIN, N_TEST, STEPS, DT, DX = 48, 60, 15, 2400, 2.0, 50.0
RESULTS = os.path.join(ROOT, 'results')
CACHE = os.path.join(ROOT, 'data', 'synth3_cache.pkl')
SCALER_OUT = os.path.join(RESULTS, 'scaler_synth_exp3v2.json')
TOL = 1e-3
checks = {}


# ---- the ONLY solver change: an optional hydrograph factor on the source ----
def simulate_tv(z, source_field, n_steps, dt=2.0, dx=50.0, manning=0.05,
                depth_floor=1e-4, save_every=None, factor=None):
    """Copy of exp3 simulate(). factor[step] scales the source. factor=None = original."""
    D = z.shape[0]
    h = np.zeros((D, D), dtype=np.float64)
    qx = np.zeros((D, D + 1), dtype=np.float64)
    qy = np.zeros((D + 1, D), dtype=np.float64)
    if save_every is None:
        save_every = max(1, n_steps // 40)
    depths, vxs, vys = [], [], []
    for step in range(n_steps):
        wse = z + h
        dS = (wse[:, 1:] - wse[:, :-1]) / dx
        hf = np.maximum(wse[:, 1:], wse[:, :-1]) - np.maximum(z[:, 1:], z[:, :-1])
        hf = np.maximum(hf, 0.0)
        q_old = qx[:, 1:-1]
        num = q_old - G * hf * dt * dS
        den = 1.0 + G * dt * manning ** 2 * np.abs(q_old) / \
            np.maximum(hf, depth_floor) ** (7.0 / 3.0)
        q_new = num / den
        q_new = np.where(hf > depth_floor, q_new, 0.0)
        qx[:, 1:-1] = q_new

        dS = (wse[1:, :] - wse[:-1, :]) / dx
        hf = np.maximum(wse[1:, :], wse[:-1, :]) - np.maximum(z[1:, :], z[:-1, :])
        hf = np.maximum(hf, 0.0)
        q_old = qy[1:-1, :]
        num = q_old - G * hf * dt * dS
        den = 1.0 + G * dt * manning ** 2 * np.abs(q_old) / \
            np.maximum(hf, depth_floor) ** (7.0 / 3.0)
        q_new = num / den
        q_new = np.where(hf > depth_floor, q_new, 0.0)
        qy[1:-1, :] = q_new

        div = (qx[:, 1:] - qx[:, :-1]) / dx + (qy[1:, :] - qy[:-1, :]) / dx
        src = source_field if factor is None else source_field * factor[step]
        h = h - dt * div + dt * src
        h = np.maximum(h, 0.0)

        if step % save_every == 0:
            qxc = 0.5 * (qx[:, :-1] + qx[:, 1:])
            qyc = 0.5 * (qy[:-1, :] + qy[1:, :])
            hsafe = np.maximum(h, depth_floor)
            depths.append(h.copy())
            vxs.append(qxc / hsafe)
            vys.append(qyc / hsafe)
    return (np.stack(depths).astype(np.float32),
            np.stack(vxs).astype(np.float32),
            np.stack(vys).astype(np.float32))


def hydrograph(n_steps, peak_frac=1.0 / 3.0):
    """Triangle: 0 -> peak at peak_frac of the run -> 0. Mean scaled to exactly 1,
    so total water equals the constant-rate mechanisms."""
    t = np.arange(n_steps) / (n_steps - 1)
    f = np.where(t <= peak_frac, t / peak_frac, (1 - t) / (1 - peak_frac))
    return f / f.mean()


def source_inflow(dim, rate_m3s, dx):
    """MECHANISM C: water enters along the whole right-hand (higher) edge column."""
    field = np.zeros((dim, dim))
    field[:, dim - 2] = rate_m3s / (dx * dx * dim)
    return field


def generate_inflow(n_sims, seed_offset):
    edge_index = _grid_edges(DIM, DIM)
    fac = hydrograph(STEPS)
    samples, vol_err = [], []
    for i in range(n_sims):
        terrain_seed = seed_offset + i                       # same seeds as exp3
        z = make_terrain(DIM, terrain_seed)
        rate = np.random.RandomState(terrain_seed + 500).uniform(20.0, 60.0)
        src = source_inflow(DIM, rate, DX)
        depth, vx, vy = simulate_tv(z, src, STEPS, dt=DT, dx=DX, factor=fac)
        injected = src.sum() * DX * DX * DT * fac.sum()
        target = rate * DT * STEPS                           # what point injects
        vol_err.append(abs(injected - target) / target)
        T, N = depth.shape[0], DIM * DIM
        dyn = np.zeros((N, T, 4), dtype=np.float32)
        dyn[:, :, 0] = depth.reshape(T, N).T
        dyn[:, :, 1] = vx.reshape(T, N).T
        dyn[:, :, 2] = vy.reshape(T, N).T
        # channel 3 stays zero: water enters at a boundary, like point and breach
        zn = (z - z.min()) / max(z.max() - z.min(), 1e-9)
        samples.append({'nodes_static': zn.reshape(-1, 1).astype(np.float32),
                        'nodes_dynamic': dyn, 'edge_index': edge_index,
                        'num_nodes': N, 'num_steps': T,
                        'dataset': 'synth_inflow', 'sample_id': terrain_seed})
        if (i + 1) % 10 == 0:
            print(f"    inflow: {i+1}/{n_sims} sims (final wet fraction "
                  f"{float((depth[-1] > 0.05).mean()):.3f}, max depth {depth.max():.2f} m)",
                  flush=True)
    return samples, max(vol_err)


def fit_like_exp3(lists):
    """Exactly the min/max logic of exp3 normalise_together()."""
    d_min, d_max = np.full(4, np.inf), np.full(4, -np.inf)
    for lst in lists:
        for s in lst:
            d = s['nodes_dynamic']
            for c in range(4):
                d_min[c] = min(d_min[c], float(d[:, :, c].min()))
                d_max[c] = max(d_max[c], float(d[:, :, c].max()))
    rng_ = d_max - d_min
    rng_[rng_ == 0] = 1.0
    return d_min, d_max, rng_


def apply(lst, d_min, rng_):
    out = []
    for s in lst:
        s2 = dict(s)
        s2['nodes_dynamic'] = ((s['nodes_dynamic'] - d_min) / rng_).astype(np.float32)
        out.append(s2)
    return out


def load_ckpt(model, path, device):
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict):
        for k in ('model_state_dict', 'state_dict', 'model'):
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break
    model.load_state_dict(obj)


def summary_stats(lst, name):
    md = [float(s['nodes_dynamic'][:, :, 0].max()) for s in lst]
    wet = [float((s['nodes_dynamic'][:, -1, 0] > 0.05).mean()) for s in lst]
    print(f"    {name:12s} median max depth {np.median(md):.2f} m, "
          f"median final wet fraction {np.median(wet):.3f}")


def main():
    t0 = time.time()
    print("=" * 70)
    print("  EXP9 STAGE 1 - inflow mechanism data (no training)")
    print("=" * 70, flush=True)

    # ---- Check B: solver copy is exact ----
    print("\n[B] solver copy check", flush=True)
    z = make_terrain(DIM, 0)
    src = source_point(DIM, 40.0, DX, 0)
    a = simulate(z, src, 300, dt=DT, dx=DX)
    b = simulate_tv(z, src, 300, dt=DT, dx=DX)
    c = simulate_tv(z, src, 300, dt=DT, dx=DX, factor=np.ones(300))
    ok = all(np.array_equal(x, y) for x, y in zip(a, b)) and \
         all(np.array_equal(x, y) for x, y in zip(a, c))
    checks['B solver identical'] = ok
    print(f"    identical to exp3 simulate(): {ok}")

    # ---- regenerate exp3 v2 data ----
    print("\n  regenerating exp3 v2 point + distributed data ...", flush=True)
    raw = {}
    for mech in ('point', 'distributed'):
        raw[mech] = {'train': generate_dataset(mech, N_TRAIN, DIM, STEPS, DT, DX, 0),
                     'test': generate_dataset(mech, N_TEST, DIM, STEPS, DT, DX, 900)}
    print(f"    done in {time.time() - t0:.0f}s", flush=True)

    # ---- rebuild the exp3 v2 scaler from the SAME four lists exp3 v2 used ----
    d_min, d_max, rng_ = fit_like_exp3([raw['point']['train'], raw['point']['test'],
                                        raw['distributed']['train'], raw['distributed']['test']])
    scaler = {'dynamic_min': d_min.tolist(), 'dynamic_max': d_max.tolist(),
              'note': 'rebuilt from exp3 v2 point+distributed data; inflow NOT in the fit'}
    print(f"\n  exp3 v2 scaler: min {np.round(d_min, 4).tolist()}  max {np.round(d_max, 4).tolist()}")
    data = {m: {k: apply(v, d_min, rng_) for k, v in raw[m].items()} for m in raw}

    # ---- Check A: saved checkpoints reproduce the stored numbers ----
    print("\n[A] re-scoring saved synth2 seed-0 checkpoints", flush=True)
    device = get_device()
    stored = json.load(open(os.path.join(RESULTS, 'result_exp3_v2_stronger.json')))['results']
    a_ok = True
    for mech in ('point', 'distributed'):
        ck = os.path.join(RESULTS, f'synth2_{mech}_s0.pt')
        if not os.path.exists(ck):
            print(f"    {ck} MISSING")
            a_ok = False
            continue
        model = build_model('vector', 64, 3, device)
        load_ckpt(model, ck, device)
        model.eval()
        with torch.no_grad():
            home = evaluate(model, data[mech]['test'], device, max_steps=30, max_samples=N_TEST)
        ref = next(r for r in stored if r['train_on'] == mech and r['seed'] == 0)['home']
        for key in ('csi_final', 'volume_ratio'):
            rel = abs(home[key] - ref[key]) / max(abs(ref[key]), 1e-9)
            match = rel < TOL
            a_ok &= match
            print(f"    {mech:11s} {key:12s} now {home[key]:.5f}  stored {ref[key]:.5f}  "
                  f"{'MATCH' if match else 'MISMATCH'}")
    checks['A checkpoints reproduce stored scores'] = a_ok

    # ---- generate inflow ----
    print("\n  generating INFLOW mechanism, same terrain seeds ...", flush=True)
    in_train, e1 = generate_inflow(N_TRAIN, 0)
    in_test, e2 = generate_inflow(N_TEST, 900)
    vol_ok = max(e1, e2) < 1e-9
    checks['C same total water as point'] = vol_ok
    print(f"    worst relative volume difference vs point: {max(e1, e2):.2e}")

    print("\n  physical summary (metres, before scaling):")
    for m in ('point', 'distributed'):
        summary_stats(raw[m]['train'], m)
    summary_stats(in_train, 'inflow')

    # ---- scale inflow with the exp3 v2 scaler, no refit ----
    data['inflow'] = {'train': apply(in_train, d_min, rng_), 'test': apply(in_test, d_min, rng_)}
    print("\n  inflow after scaling: share of values outside [0, 1]")
    names = ['depth', 'vx', 'vy', 'rain']
    for c in range(4):
        vals = np.concatenate([s['nodes_dynamic'][:, :, c].ravel()
                               for s in data['inflow']['train'] + data['inflow']['test']])
        out = 100.0 * np.mean((vals < 0) | (vals > 1))
        print(f"    {names[c]:5s} min {vals.min():.3f} max {vals.max():.3f} outside {out:.3f}%")

    # ---- leakage audit ----
    leak_ok = True
    for m in data:
        tr = {s['sample_id'] for s in data[m]['train']}
        te = {s['sample_id'] for s in data[m]['test']}
        leak_ok &= not (tr & te)
    same_seeds = all({s['sample_id'] for s in data[m]['test']} ==
                     {s['sample_id'] for s in data['point']['test']} for m in data)
    checks['D no train/test terrain overlap'] = leak_ok
    checks['E test terrains matched across mechanisms'] = same_seeds
    print(f"\n  leakage audit: overlap-free {leak_ok}, matched test terrains {same_seeds}")

    # ---- save, never overwrite ----
    if os.path.exists(SCALER_OUT):
        old = json.load(open(SCALER_OUT))
        same = np.allclose(old['dynamic_min'], scaler['dynamic_min']) and \
               np.allclose(old['dynamic_max'], scaler['dynamic_max'])
        print(f"\n  {SCALER_OUT} already exists, identical: {same}")
    else:
        json.dump(scaler, open(SCALER_OUT, 'w'), indent=2)
        print(f"\n  saved {SCALER_OUT}")
    if os.path.exists(CACHE):
        print(f"  {CACHE} already exists - NOT overwritten")
    elif all(checks.values()):
        with open(CACHE, 'wb') as f:
            pickle.dump({'config': dict(dim=DIM, n_train=N_TRAIN, n_test=N_TEST, steps=STEPS,
                                        dt=DT, dx=DX), 'scaler': scaler, 'data': data}, f)
        print(f"  saved {CACHE} ({os.path.getsize(CACHE) / 1e6:.0f} MB)")
    else:
        print("  cache NOT saved because a check failed")

    print("\n" + "=" * 70)
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"\n  OVERALL: {'PASS' if all(checks.values()) else 'FAIL'}  "
          f"({time.time() - t0:.0f}s)")
    print("=" * 70)


if __name__ == '__main__':
    main()
