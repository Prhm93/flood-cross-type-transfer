"""
exp3_controlled_synthetic.py
============================

THE CONFOUND-KILLER EXPERIMENT.

The reviewer's strongest objection: your two real datasets differ in flood
type AND solver AND region AND resolution AND terrain AND sample count.
So a transfer failure cannot be attributed to flood type alone.

This experiment removes every confound. We generate our own floods with:

    SAME terrain generator      SAME grid size      SAME solver
    SAME resolution             SAME time step      SAME sample count
    SAME model architecture     SAME training budget

and change EXACTLY ONE THING: how the water enters.

    Mechanism A (point source): water enters at one cell, like a breach
    Mechanism B (distributed):  water falls on every cell, like rainfall

If transfer still fails here, the flood mechanism itself is the cause,
because nothing else differs. That is the controlled result the paper needs.

THE SOLVER
----------
We use the local inertial (or "simplified shallow water") formulation of
Bates et al. (2010), which is the scheme used inside LISFLOOD-FP. It is a
standard, citable, physically grounded method — not a toy.

Flow between two neighbouring cells:

    q_new = (q_old - g * h_flow * dt * dS) / (1 + g * dt * n^2 * |q| / h_flow^(7/3))

where h_flow is the depth of water available to flow between the cells,
dS is the water-surface slope, n is Manning's roughness, g is gravity.

Then depth is updated by conservation of mass:

    dh/dt = (inflow - outflow) / cell_area  + source

The `source` term is where the two mechanisms differ, and it is the ONLY
difference between them.

Run:
    python3 scripts/exp3_controlled_synthetic.py
    python3 scripts/exp3_controlled_synthetic.py --quick   (smoke test size)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiment_lib import (set_seed, train_model, evaluate, build_model,
                                 get_device)
from src.unified_loader import _grid_edges

G = 9.81


# ------------------------------------------------------------------ #
# TERRAIN
# ------------------------------------------------------------------ #

def make_terrain(dim, seed, relief=2.0):
    """
    Generate gently undulating terrain by summing a few smooth waves.
    Identical generator for both mechanisms — only the seed varies, and
    we use THE SAME SEEDS for both, so terrain is matched pairwise.
    """
    rng = np.random.RandomState(seed)
    y, x = np.mgrid[0:dim, 0:dim] / dim

    z = np.zeros((dim, dim), dtype=np.float64)
    for _ in range(4):
        kx, ky = rng.uniform(1, 4, 2)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.3, 1.0)
        z += amp * np.sin(2 * np.pi * (kx * x + ky * y) + phase)

    z -= z.min()
    if z.max() > 0:
        z = z / z.max() * relief
    # gentle overall tilt so water has somewhere to go
    z += 0.5 * x * relief * 0.3
    return z.astype(np.float64)


# ------------------------------------------------------------------ #
# SOLVER — local inertial shallow water (Bates et al. 2010)
# ------------------------------------------------------------------ #

def simulate(z, source_field, n_steps, dt=2.0, dx=50.0, manning=0.05,
             depth_floor=1e-4, save_every=None):
    """
    Run the flood simulation.

    z            : (D,D) bed elevation, metres
    source_field : (D,D) water added per second per cell, metres/second.
                   This is the ONLY thing that differs between mechanisms.
    n_steps      : number of solver steps
    dt           : solver time step, seconds
    dx           : cell size, metres
    manning      : roughness

    Returns depth (T,D,D), vx (T,D,D), vy (T,D,D) at the saved times.
    """
    D = z.shape[0]
    h = np.zeros((D, D), dtype=np.float64)

    # discharge per unit width on cell faces
    qx = np.zeros((D, D + 1), dtype=np.float64)   # faces in x
    qy = np.zeros((D + 1, D), dtype=np.float64)   # faces in y

    if save_every is None:
        save_every = max(1, n_steps // 40)

    depths, vxs, vys = [], [], []

    for step in range(n_steps):
        wse = z + h   # water surface elevation

        # ---- x-direction faces (between column i-1 and i) ----
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

        # ---- y-direction faces ----
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

        # ---- mass conservation ----
        # net flux into each cell = (flux in from left - flux out right) etc.
        div = (qx[:, 1:] - qx[:, :-1]) / dx + (qy[1:, :] - qy[:-1, :]) / dx
        h = h - dt * div + dt * source_field
        h = np.maximum(h, 0.0)

        if step % save_every == 0:
            # cell-centred velocities from face discharges
            qxc = 0.5 * (qx[:, :-1] + qx[:, 1:])
            qyc = 0.5 * (qy[:-1, :] + qy[1:, :])
            hsafe = np.maximum(h, depth_floor)
            depths.append(h.copy())
            vxs.append(qxc / hsafe)
            vys.append(qyc / hsafe)

    return (np.stack(depths).astype(np.float32),
            np.stack(vxs).astype(np.float32),
            np.stack(vys).astype(np.float32))


# ------------------------------------------------------------------ #
# THE TWO MECHANISMS — the only difference
# ------------------------------------------------------------------ #

def source_point(dim, rate_m3s, dx, seed):
    """
    MECHANISM A — point source (breach-like).
    All the water enters through ONE cell.
    """
    rng = np.random.RandomState(seed + 10000)
    field = np.zeros((dim, dim))
    # place the source on the left edge, random row
    r = rng.randint(dim // 4, 3 * dim // 4)
    c = 1
    field[r, c] = rate_m3s / (dx * dx)   # m3/s spread over one cell area
    return field


def source_distributed(dim, rate_m3s, dx, seed):
    """
    MECHANISM B — distributed source (rainfall-like).
    The SAME total water volume, but spread over every cell.
    """
    field = np.full((dim, dim), rate_m3s / (dx * dx * dim * dim))
    return field


# ------------------------------------------------------------------ #
# DATASET GENERATION
# ------------------------------------------------------------------ #

def generate_dataset(mechanism, n_sims, dim, n_steps, dt, dx, seed_offset=0,
                     verbose=True):
    """
    Generate n_sims simulations for one mechanism, in the unified format.

    Crucially, both mechanisms use THE SAME terrain seeds, so simulation i
    of mechanism A has exactly the same terrain as simulation i of
    mechanism B. Only the water source differs.
    """
    edge_index = _grid_edges(dim, dim)
    samples = []

    for i in range(n_sims):
        terrain_seed = seed_offset + i        # SAME for both mechanisms
        z = make_terrain(dim, terrain_seed)

        rng = np.random.RandomState(terrain_seed + 500)
        rate = rng.uniform(20.0, 60.0)         # m3/s, same range both ways

        if mechanism == 'point':
            src = source_point(dim, rate, dx, terrain_seed)
        elif mechanism == 'distributed':
            src = source_distributed(dim, rate, dx, terrain_seed)
        else:
            raise ValueError(mechanism)

        depth, vx, vy = simulate(z, src, n_steps=n_steps, dt=dt, dx=dx)

        T = depth.shape[0]
        N = dim * dim

        dyn = np.zeros((N, T, 4), dtype=np.float32)
        dyn[:, :, 0] = depth.reshape(T, N).T
        dyn[:, :, 1] = vx.reshape(T, N).T
        dyn[:, :, 2] = vy.reshape(T, N).T
        # channel 3 (the 'rainfall' slot) carries the source field for the
        # distributed mechanism and zero for the point mechanism — exactly
        # mirroring the real datasets' asymmetry
        if mechanism == 'distributed':
            dyn[:, :, 3] = float(src.mean())

        zn = (z - z.min()) / max(z.max() - z.min(), 1e-9)

        samples.append({
            'nodes_static': zn.reshape(-1, 1).astype(np.float32),
            'nodes_dynamic': dyn,
            'edge_index': edge_index,
            'num_nodes': N,
            'num_steps': T,
            'dataset': f'synth_{mechanism}',
            'sample_id': terrain_seed,
        })

        if verbose and (i + 1) % 10 == 0:
            wet = float((depth[-1] > 0.05).mean())
            print(f"    {mechanism}: {i+1}/{n_sims} sims  "
                  f"(last sim final wet fraction {wet:.3f}, "
                  f"max depth {depth.max():.2f} m)")

    return samples


def normalise_together(list_of_sample_lists):
    """Joint min-max normalisation across all provided sample lists."""
    d_min = np.full(4, np.inf)
    d_max = np.full(4, -np.inf)
    for lst in list_of_sample_lists:
        for s in lst:
            d = s['nodes_dynamic']
            for c in range(4):
                d_min[c] = min(d_min[c], float(d[:, :, c].min()))
                d_max[c] = max(d_max[c], float(d[:, :, c].max()))
    rng_ = d_max - d_min
    rng_[rng_ == 0] = 1.0

    out = []
    for lst in list_of_sample_lists:
        new = []
        for s in lst:
            s2 = dict(s)
            s2['nodes_dynamic'] = ((s['nodes_dynamic'] - d_min) / rng_).astype(np.float32)
            new.append(s2)
        out.append(new)
    return out


# ------------------------------------------------------------------ #
# MAIN
# ------------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='tiny smoke-test run')
    ap.add_argument('--dim', type=int, default=48)
    ap.add_argument('--n-train', type=int, default=40)
    ap.add_argument('--n-test', type=int, default=12)
    ap.add_argument('--solver-steps', type=int, default=1600)
    ap.add_argument('--dt', type=float, default=2.0)
    ap.add_argument('--dx', type=float, default=50.0)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--out-dir', default='results')
    args = ap.parse_args()

    if args.quick:
        args.dim, args.n_train, args.n_test = 24, 4, 3
        args.solver_steps, args.epochs, args.seeds = 200, 3, [0]

    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()

    print("=" * 66)
    print("  EXPERIMENT 3 — CONTROLLED CROSS-MECHANISM TRANSFER")
    print("  Same terrain, same solver, same grid, same everything.")
    print("  ONLY the water source differs.")
    print("=" * 66)
    print(f"  grid {args.dim}x{args.dim}, {args.n_train} train + {args.n_test} "
          f"test sims per mechanism, device {device}")

    # ---- generate ----
    t0 = time.time()
    print("\n  Generating POINT-SOURCE floods (breach-like)...")
    pt_train = generate_dataset('point', args.n_train, args.dim,
                                 args.solver_steps, args.dt, args.dx, 0)
    pt_test = generate_dataset('point', args.n_test, args.dim,
                                args.solver_steps, args.dt, args.dx, 900)

    print("\n  Generating DISTRIBUTED floods (rainfall-like), SAME terrain seeds...")
    di_train = generate_dataset('distributed', args.n_train, args.dim,
                                 args.solver_steps, args.dt, args.dx, 0)
    di_test = generate_dataset('distributed', args.n_test, args.dim,
                                args.solver_steps, args.dt, args.dx, 900)

    print(f"\n  Generation took {time.time() - t0:.0f}s")

    pt_train, pt_test, di_train, di_test = normalise_together(
        [pt_train, pt_test, di_train, di_test])

    data = {'point': {'train': pt_train, 'test': pt_test},
            'distributed': {'train': di_train, 'test': di_test}}

    # ---- train and cross-test ----
    results = []
    for train_on in ['point', 'distributed']:
        away = 'distributed' if train_on == 'point' else 'point'
        for seed in args.seeds:
            print(f"\n  --- train on {train_on}, seed {seed} ---")
            set_seed(seed)
            model = build_model('vector', 64, 3, device)
            ckpt = os.path.join(args.out_dir, f'synth_{train_on}_s{seed}.pt')
            train_model(model, data[train_on]['train'], device,
                        epochs=args.epochs, max_train_samples=args.n_train,
                        ckpt_path=ckpt, log_every=max(1, args.epochs // 4))

            home = evaluate(model, data[train_on]['test'], device,
                            max_steps=30, max_samples=args.n_test)
            awayr = evaluate(model, data[away]['test'], device,
                             max_steps=30, max_samples=args.n_test)

            retained = (100.0 * awayr['csi_final'] / home['csi_final']
                        if home['csi_final'] > 0 else float('nan'))

            print(f"    HOME ({train_on:11s}) CSI {home['csi_final']:.4f}  "
                  f"vol ratio {home['volume_ratio']:.2f}")
            print(f"    AWAY ({away:11s}) CSI {awayr['csi_final']:.4f}  "
                  f"vol ratio {awayr['volume_ratio']:.2f}")
            print(f"    retained {retained:.1f}%")

            results.append({
                'train_on': train_on, 'away': away, 'seed': seed,
                'home': home, 'away_scores': awayr, 'retained_pct': retained,
            })

    # ---- summary ----
    print("\n" + "=" * 66)
    print("  CONTROLLED EXPERIMENT SUMMARY")
    print("=" * 66)
    for tr in ['point', 'distributed']:
        rs = [r for r in results if r['train_on'] == tr]
        if not rs:
            continue
        hm = np.mean([r['home']['csi_final'] for r in rs])
        aw = np.mean([r['away_scores']['csi_final'] for r in rs])
        rt = np.nanmean([r['retained_pct'] for r in rs])
        print(f"  trained on {tr:12s}: home CSI {hm:.4f}  away CSI {aw:.4f}  "
              f"retained {rt:.1f}%")

    print("\n  INTERPRETATION: terrain, grid, solver, resolution and sample")
    print("  count are IDENTICAL across the two mechanisms. Any transfer gap")
    print("  here is attributable to the flood-driving mechanism alone.")

    with open(os.path.join(args.out_dir, 'result_exp3_controlled.json'), 'w') as f:
        json.dump({'config': vars(args), 'results': results}, f,
                  indent=2, default=str)
    print(f"\n  Saved {args.out_dir}/result_exp3_controlled.json")


if __name__ == '__main__':
    main()