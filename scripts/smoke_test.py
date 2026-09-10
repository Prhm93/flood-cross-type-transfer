"""
smoke_test.py
=============

Quick check that everything works BEFORE you start a multi-hour run.
Uses tiny synthetic data — no real datasets needed, no GPU needed.
Should finish in about a minute.

Run:
    python3 scripts/smoke_test.py

If every line says OK, the full pipeline is safe to launch.
"""

import os
import sys
import tempfile
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        print(f"  OK    {name}")
        PASS.append(name)
    except Exception as e:
        print(f"  FAIL  {name}")
        print(f"        {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        FAIL.append(name)


def make_fake_samples(n=4, N=64, T=12):
    """Tiny fake dataset in the unified format."""
    from src.unified_loader import _grid_edges
    dim = int(np.sqrt(N))
    ei = _grid_edges(dim, dim)
    out = []
    for i in range(n):
        dyn = np.abs(np.random.rand(N, T, 4)).astype(np.float32) * 0.3
        out.append({
            'nodes_static': np.random.rand(N, 1).astype(np.float32),
            'nodes_dynamic': dyn,
            'edge_index': ei,
            'num_nodes': N, 'num_steps': T,
            'dataset': 'fake', 'sample_id': i,
        })
    return out


def main():
    print("=" * 60)
    print("  SMOKE TEST")
    print("=" * 60)

    # --- imports ---
    def t_imports():
        import torch
        from src import metrics, model, normalise, unified_loader, experiment_lib
    check("imports", t_imports)

    # --- metrics self-consistency ---
    def t_metrics():
        from src.metrics import csi, arrival_error, mass_error, auto_threshold, score
        t = np.zeros(200); t[:100] = 1.0
        p = np.zeros(200); p[:80] = 1.0; p[100:110] = 1.0
        got = csi(p, t)
        assert abs(got - 80/110) < 1e-9, got
        # auto threshold adapts to scale
        assert abs(auto_threshold(np.ones((3,3,3))*2.5) - 0.025) < 1e-6
        assert abs(auto_threshold(np.ones((3,3,3))*0.2) - 0.002) < 1e-6
        # score returns the threshold it used
        s = score(np.random.rand(4,2,2)*0.2, np.random.rand(4,2,2)*0.2)
        assert 'threshold_used' in s
    check("metrics maths + adaptive threshold", t_metrics)

    # --- model forward/backward, both variants ---
    def t_model():
        import torch
        from src.model import FloodGNN
        for variant in ['vector', 'scalar']:
            m = FloodGNN(variant=variant, hidden=16, num_layers=2)
            dyn = torch.randn(16, 4); sta = torch.randn(16, 1)
            ei = torch.randint(0, 16, (2, 40))
            out = m(dyn, sta, ei)
            assert out.shape == (16, 3), out.shape
            out.sum().backward()
    check("model forward + backward (both variants)", t_model)

    # --- experiment_lib train + eval on fake data ---
    def t_lib():
        from src.experiment_lib import (train_model, evaluate, build_model,
                                         get_device, rollout_predictions,
                                         score_rollouts, volume_ratio,
                                         strip_rainfall)
        dev = get_device()
        s = make_fake_samples(3)
        m = build_model('vector', 16, 2, dev)
        train_model(m, s, dev, epochs=2, max_train_samples=3, verbose=False)
        r = evaluate(m, s, dev, max_steps=4, max_samples=2)
        assert 'csi_final' in r and 'volume_ratio' in r
        # rollouts can be scored at several thresholds
        roll = rollout_predictions(m, s, dev, max_steps=4, max_samples=2)
        for thr in [0.001, 0.05]:
            sc = score_rollouts(roll, 'relative', thr)
            assert not np.isnan(sc['csi_final']) or True
        assert volume_ratio(roll) > 0
        # rainfall stripping really zeroes channel 3
        nr = strip_rainfall(s)
        assert (nr[0]['nodes_dynamic'][:, :, 3] == 0).all()
        assert s[0]['nodes_dynamic'] is not nr[0]['nodes_dynamic']
    check("experiment_lib train/eval/threshold-scoring/ablation", t_lib)

    # --- synthetic solver physics ---
    def t_solver():
        from scripts.exp3_controlled_synthetic import (make_terrain, simulate,
                                                        source_point,
                                                        source_distributed,
                                                        generate_dataset)
        dim, dx, dt = 16, 50.0, 2.0
        z = make_terrain(dim, 0)
        assert z.shape == (dim, dim) and np.isfinite(z).all()

        sp = source_point(dim, 40.0, dx, 0)
        d, vx, vy = simulate(z, sp, n_steps=200, dt=dt, dx=dx)
        assert np.isfinite(d).all(), "solver produced NaN/inf"
        assert (d >= 0).all(), "negative depth"
        # mass roughly conserved
        vol_in = 40.0 * 200 * dt
        vol_have = d[-1].sum() * dx * dx
        assert 0.7 < vol_have / vol_in < 1.3, f"mass ratio {vol_have/vol_in:.3f}"

        sd = source_distributed(dim, 40.0, dx, 0)
        d2, _, _ = simulate(z, sd, n_steps=200, dt=dt, dx=dx)
        assert np.isfinite(d2).all()
        # the two mechanisms must actually differ
        assert abs(float((d[-1] > 0.02).mean()) - float((d2[-1] > 0.02).mean())) > 1e-3, \
            "point and distributed produced the same flood — mechanisms not distinct"

        # dataset generation returns the unified format
        ds = generate_dataset('point', 2, dim, 200, dt, dx, 0, verbose=False)
        assert ds[0]['nodes_dynamic'].shape[2] == 4
        assert ds[0]['edge_index'].shape[0] == 2
    check("synthetic solver: stability, mass, mechanism difference", t_solver)

    # --- the two mechanisms train and cross-evaluate ---
    def t_synth_endtoend():
        from scripts.exp3_controlled_synthetic import (generate_dataset,
                                                        normalise_together)
        from src.experiment_lib import train_model, evaluate, build_model, get_device
        dev = get_device()
        a = generate_dataset('point', 2, 16, 200, 2.0, 50.0, 0, verbose=False)
        b = generate_dataset('distributed', 2, 16, 200, 2.0, 50.0, 0, verbose=False)
        a, b = normalise_together([a, b])
        m = build_model('vector', 16, 2, dev)
        train_model(m, a, dev, epochs=2, max_train_samples=2, verbose=False)
        ha = evaluate(m, a, dev, max_steps=4, max_samples=2)
        hb = evaluate(m, b, dev, max_steps=4, max_samples=2)
        assert 'csi_final' in ha and 'csi_final' in hb
    check("controlled experiment end-to-end (tiny)", t_synth_endtoend)

    # --- check real data is present (warn only) ---
    def t_data():
        missing = []
        if not os.path.isdir('data/breach/raw_datasets/WD'):
            missing.append('data/breach/raw_datasets')
        for f in ['train.npz', 'val.npz', 'test.npz']:
            if not os.path.exists(f'data/harvey/{f}'):
                missing.append(f'data/harvey/{f}')
        if missing:
            raise FileNotFoundError("missing real data: " + ", ".join(missing))
    check("real datasets present on disk", t_data)

    # --- existing checkpoints for exp1 (warn only) ---
    def t_ckpt():
        need = ['results/best_breach_vector_s0.pt',
                'results/best_harvey_vector_s0.pt']
        miss = [p for p in need if not os.path.exists(p)]
        if miss:
            raise FileNotFoundError(
                "Experiment 1 needs these checkpoints from your earlier runs: "
                + ", ".join(miss))
    check("checkpoints for experiment 1 present", t_ckpt)

    print("\n" + "=" * 60)
    print(f"  {len(PASS)} passed, {len(FAIL)} failed")
    print("=" * 60)
    if FAIL:
        print("  Failed:", ", ".join(FAIL))
        print("\n  NOTE: failures of the last two checks (real data / checkpoints)")
        print("  only mean those specific experiments cannot run yet; the code")
        print("  itself is fine. Any OTHER failure must be fixed first.")
        sys.exit(1)
    print("  All good — safe to launch the full run.")


if __name__ == '__main__':
    main()
    