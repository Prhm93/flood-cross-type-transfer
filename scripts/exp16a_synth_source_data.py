#!/usr/bin/env python3
"""
exp16a_synth_source_data.py - rebuild the three synthetic mechanisms with a SOURCE channel.
Channel 3 now carries the water added to each cell by its own mechanism (rain, point source
or upstream inflow), as log(1 + rate / 1e-6 m/s), so every mechanism is told its forcing
in the same form. Same terrain seeds, solver, grid and run length as exp3 v2 / exp9.
Joint min-max scaler fitted on the three TRAINING sets only. Never overwrites.
"""
import os, sys, json, time, pickle
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.unified_loader import _grid_edges
from scripts.exp3_controlled_synthetic import make_terrain, simulate, source_point, source_distributed
from scripts.exp9_stage1_inflow_data import simulate_tv, hydrograph, source_inflow

DIM, N_TRAIN, N_TEST, STEPS, DT, DX = 48, 60, 15, 2400, 2.0, 50.0
SAVE_EVERY = max(1, STEPS // 40)
R0 = 1e-6
OUT = os.path.join(ROOT, 'data', 'synth_src_cache.pkl')
SCALER = os.path.join(ROOT, 'results', 'scaler_synth_source.json')
MECHS = ('point', 'distributed', 'inflow')


def gen(mech, n, off):
    edge_index, fac, out = _grid_edges(DIM, DIM), hydrograph(STEPS), []
    saved = [k for k in range(STEPS) if k % SAVE_EVERY == 0]
    for i in range(n):
        seed = off + i
        z = make_terrain(DIM, seed)
        rate = np.random.RandomState(seed + 500).uniform(20.0, 60.0)
        if mech == 'point':
            src = source_point(DIM, rate, DX, seed)
            depth, vx, vy = simulate(z, src, STEPS, dt=DT, dx=DX)
            srcs = [src] * len(saved)
        elif mech == 'distributed':
            src = source_distributed(DIM, rate, DX, seed)
            depth, vx, vy = simulate(z, src, STEPS, dt=DT, dx=DX)
            srcs = [src] * len(saved)
        else:
            src = source_inflow(DIM, rate, DX)
            depth, vx, vy = simulate_tv(z, src, STEPS, dt=DT, dx=DX, factor=fac)
            srcs = [src * fac[k] for k in saved]
        T, N = depth.shape[0], DIM * DIM
        assert T == len(srcs), (T, len(srcs))
        dyn = np.zeros((N, T, 4), dtype=np.float32)
        dyn[:, :, 0] = depth.reshape(T, N).T
        dyn[:, :, 1] = vx.reshape(T, N).T
        dyn[:, :, 2] = vy.reshape(T, N).T
        dyn[:, :, 3] = np.log1p(np.stack(srcs).reshape(T, N).T / R0)
        zn = (z - z.min()) / max(z.max() - z.min(), 1e-9)
        out.append({'nodes_static': zn.reshape(-1, 1).astype(np.float32), 'nodes_dynamic': dyn,
                    'edge_index': edge_index, 'num_nodes': N, 'num_steps': T,
                    'dataset': f'synth_{mech}', 'sample_id': seed})
    return out


def main():
    if os.path.exists(OUT):
        sys.exit(f'{OUT} exists - not overwritten')
    t0 = time.time()
    raw = {}
    for m in MECHS:
        raw[m] = {'train': gen(m, N_TRAIN, 0), 'test': gen(m, N_TEST, 900)}
        print(f"  {m}: generated ({time.time() - t0:.0f}s)", flush=True)
    d_min, d_max = np.full(4, np.inf), np.full(4, -np.inf)
    for m in MECHS:
        for s in raw[m]['train']:
            d = s['nodes_dynamic']
            d_min = np.minimum(d_min, d.reshape(-1, 4).min(0))
            d_max = np.maximum(d_max, d.reshape(-1, 4).max(0))
    rng_ = np.where(d_max - d_min == 0, 1.0, d_max - d_min)
    data = {}
    for m in MECHS:
        data[m] = {}
        for k in ('train', 'test'):
            data[m][k] = [dict(s, nodes_dynamic=((s['nodes_dynamic'] - d_min) / rng_).astype(np.float32))
                          for s in raw[m][k]]
    print("\n  source channel after scaling (min / median of non-zero / max), training sets:")
    for m in MECHS:
        v = np.concatenate([s['nodes_dynamic'][:, :, 3].ravel() for s in data[m]['train']])
        nz = v[v > 1e-9]
        print(f"    {m:11s} {v.min():.3f} / {np.median(nz) if nz.size else 0:.3f} / {v.max():.3f}   "
              f"cells with a source: {100 * nz.size / v.size:.2f}%")
    scaler = {'dynamic_min': d_min.tolist(), 'dynamic_max': d_max.tolist(), 'source_transform': f'log1p(rate/{R0})',
              'fitted_on': 'training sets of point, distributed, inflow'}
    json.dump(scaler, open(SCALER, 'w'), indent=2)
    pickle.dump({'scaler': scaler, 'data': data}, open(OUT, 'wb'))
    print(f"\n  saved {OUT} and {SCALER}  ({time.time() - t0:.0f}s)")


if __name__ == '__main__':
    main()
