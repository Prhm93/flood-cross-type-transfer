"""
unified_loader.py
=================

One loader, two datasets, same output format.

This is the key engineering piece for the cross-type transfer paper.
It takes either the Harvey rainfall data OR the SWE-GNN breach data
and outputs the SAME graph structure with the SAME feature columns.

That means a model trained on one can be directly tested on the other.

OUTPUT FORMAT (per sample)
--------------------------
Each sample is a dictionary with:

    nodes_static  : (N, 1)           terrain elevation (scaled 0-1)
    nodes_dynamic : (N, T, 4)        per-timestep features:
                                       channel 0 = water depth
                                       channel 1 = velocity x
                                       channel 2 = velocity y
                                       channel 3 = rainfall (0 if not available)
    edge_index    : (2, E)           which nodes connect to which
    num_nodes     : int              N
    num_steps     : int              T
    dataset       : str              'harvey' or 'breach'
    sample_id     : int              index within the dataset

WHY THIS FORMAT
---------------
- Both datasets have depth and velocity vectors. Those are channels 0-2.
- Harvey has rainfall (channel 3). Breach data does not (the breach is
  a boundary source, not rain). So channel 3 is zero for breach data.
  This is deliberate: the model sees a zero-rain field and must cope.
  That asymmetry IS the transfer challenge.
- Elevation goes into static features because it does not change over time.
- Edge index defines the graph: which cells are neighbours.

DATASETS
--------
Harvey (FloodGNN-GRU, Zenodo 10.5281/zenodo.10787632):
    - 228 test / ~val / ~train graph chunks from Hurricane Harvey, Houston TX
    - LISFLOOD-FP simulation, 30m resolution, 1-hour steps
    - Each chunk: ~600-1600 nodes, 132 timesteps, 8 dynamic features
    - Already in graph form with edges

SWE-GNN dike breach (Zenodo 10.5281/zenodo.7764418):
    - 100 simulations (60 train, 20 val, 20 test) of dike breaches
    - Delft3D-FM solver, 100m cells on a 64x64 regular grid, 1-hour steps
    - Stored as plain text: WD/VX/VY matrices + DEM xyz
    - We convert the grid into a graph (each cell = node, 4-connected edges)
"""

import os
import glob
import numpy as np


# ------------------------------------------------------------------ #
# HARVEY LOADER
# ------------------------------------------------------------------ #

def load_harvey(npz_path):
    """
    Load one Harvey npz file (train.npz, val.npz, or test.npz).

    Returns a list of sample dictionaries in the unified format.

    Feature mapping (from FloodGNN-GRU/dataset.py, confirmed by inspection):
        data[..., 0]   = water depth (scaled, non-negative)
        data[..., 1:3] = velocity (vx, vy) at current time
        data[..., 3:5] = velocity (vx, vy) at previous time
        data[..., 5:7] = velocity magnitude (scalar, two copies)
        data[..., 7]   = rainfall

    We keep: depth (col 0), current velocity (cols 1,2), rainfall (col 7).
    We drop: previous velocity and scalar magnitude (redundant for our test).

    Static features (3 columns): likely elevation, slope, Manning's n.
    We keep column 0 (elevation) as the static node feature.
    """
    print(f"Loading Harvey data from {npz_path} ...")
    raw = np.load(npz_path, allow_pickle=True)
    items = raw[list(raw.keys())[0]]  # the single key, usually 'harvey'

    samples = []
    for i, item in enumerate(items):
        data = item['data']       # (N, T, 8)
        static = item['static']   # (N, 3)
        edges = item['s_edges']   # (E, 2)

        N, T, _ = data.shape

        # Build the 4-channel dynamic array: depth, vx, vy, rain
        dynamic = np.zeros((N, T, 4), dtype=np.float32)
        dynamic[:, :, 0] = data[:, :, 0]   # water depth
        dynamic[:, :, 1] = data[:, :, 1]   # velocity x (current)
        dynamic[:, :, 2] = data[:, :, 2]   # velocity y (current)
        dynamic[:, :, 3] = data[:, :, 7]   # rainfall

        samples.append({
            'nodes_static':  static[:, :1].astype(np.float32),  # elevation
            'nodes_dynamic': dynamic,
            'edge_index':    edges.T.astype(np.int64),  # (2, E)
            'num_nodes':     N,
            'num_steps':     T,
            'dataset':       'harvey',
            'sample_id':     i,
        })

    print(f"  Loaded {len(samples)} Harvey samples. "
          f"Typical: {samples[0]['num_nodes']} nodes, "
          f"{samples[0]['num_steps']} steps.")
    return samples


# ------------------------------------------------------------------ #
# SWE-GNN BREACH LOADER
# ------------------------------------------------------------------ #

def _grid_edges(H, W):
    """
    Build edge index for a regular H x W grid (4-connected: up, down, left, right).

    Each cell is a node numbered row-major: node = row * W + col.
    Returns (2, E) int64 array.

    Example for a 3x3 grid:
        0-1-2
        |x|x|
        3-4-5
        |x|x|
        6-7-8
        Node 4 connects to 1, 3, 5, 7.
    """
    edges = []
    for r in range(H):
        for c in range(W):
            me = r * W + c
            if r > 0:     edges.append([me, (r - 1) * W + c])  # up
            if r < H - 1: edges.append([me, (r + 1) * W + c])  # down
            if c > 0:     edges.append([me, me - 1])            # left
            if c < W - 1: edges.append([me, me + 1])            # right
    return np.array(edges, dtype=np.int64).T  # (2, E)


def load_breach_sim(raw_dir, sim_id, dim=64):
    """
    Load one SWE-GNN dike-breach simulation.

    File layout (confirmed by prior inspection):
        DEM/DEM_{id}.txt  — x, y, elevation triples (dim*dim rows, 3 cols)
        WD/WD_{id}.txt    — water depth matrix (T rows, dim*dim cols)
        VX/VX_{id}.txt    — velocity x matrix (T rows, dim*dim cols)
        VY/VY_{id}.txt    — velocity y matrix (T rows, dim*dim cols)

    Returns depth (T, N), vx (T, N), vy (T, N), dem (N,) — all flat,
    where N = dim * dim.
    """
    dem_path = os.path.join(raw_dir, 'DEM', f'DEM_{sim_id}.txt')
    wd_path  = os.path.join(raw_dir, 'WD',  f'WD_{sim_id}.txt')
    vx_path  = os.path.join(raw_dir, 'VX',  f'VX_{sim_id}.txt')
    vy_path  = os.path.join(raw_dir, 'VY',  f'VY_{sim_id}.txt')

    dem_xyz = np.loadtxt(dem_path)              # (N, 3) — x, y, elevation
    elevation = dem_xyz[:, 2].astype(np.float32)  # just the height column

    depth = np.loadtxt(wd_path).astype(np.float32)  # (T, N)
    vx    = np.loadtxt(vx_path).astype(np.float32)  # (T, N)
    vy    = np.loadtxt(vy_path).astype(np.float32)  # (T, N)

    N = dim * dim
    assert elevation.shape[0] == N, f"DEM has {elevation.shape[0]} cells, expected {N}"
    assert depth.shape[1] == N, f"WD has {depth.shape[1]} cols, expected {N}"

    return depth, vx, vy, elevation


def load_breach(raw_dir, sim_ids, dim=64):
    """
    Load multiple SWE-GNN breach simulations and return unified-format samples.

    sim_ids: list of integer simulation IDs to load (e.g. range(1, 61))
    raw_dir: path to the folder containing DEM/, WD/, VX/, VY/ subfolders

    Rainfall channel is set to ZERO — breach floods have no rain,
    the water enters through the breach boundary. This is the key
    physical difference from Harvey and is the whole point of the test.
    """
    # Build the edge index once (same grid for all sims of the same dim)
    edge_index = _grid_edges(dim, dim)

    samples = []
    for sid in sim_ids:
        try:
            depth, vx, vy, elev = load_breach_sim(raw_dir, sid, dim)
        except FileNotFoundError as e:
            print(f"  [skip] sim {sid}: {e}")
            continue

        T, N = depth.shape

        # Normalise elevation to 0-1 range (same as Harvey)
        e_min, e_max = elev.min(), elev.max()
        if e_max > e_min:
            elev_norm = (elev - e_min) / (e_max - e_min)
        else:
            elev_norm = np.zeros_like(elev)

        # Build 4-channel dynamic: depth, vx, vy, rain=0
        dynamic = np.zeros((N, T, 4), dtype=np.float32)
        dynamic[:, :, 0] = depth.T    # (T, N) -> (N, T)
        dynamic[:, :, 1] = vx.T
        dynamic[:, :, 2] = vy.T
        # channel 3 stays zero — no rainfall in breach floods

        samples.append({
            'nodes_static':  elev_norm.reshape(-1, 1),   # (N, 1)
            'nodes_dynamic': dynamic,                     # (N, T, 4)
            'edge_index':    edge_index,                  # (2, E)
            'num_nodes':     N,
            'num_steps':     T,
            'dataset':       'breach',
            'sample_id':     sid,
        })

    print(f"  Loaded {len(samples)} breach samples. "
          f"Typical: {samples[0]['num_nodes']} nodes, "
          f"{samples[0]['num_steps']} steps.")
    return samples


# ------------------------------------------------------------------ #
# QUICK SUMMARY — print what you loaded so you can sanity-check
# ------------------------------------------------------------------ #

def summarise(samples, label=""):
    """Print a quick overview of a loaded dataset."""
    if not samples:
        print(f"[{label}] No samples loaded.")
        return

    nodes = [s['num_nodes'] for s in samples]
    steps = [s['num_steps'] for s in samples]
    ds    = samples[0]['dataset']

    print(f"\n{'='*60}")
    print(f"  {label or ds.upper()} — {len(samples)} samples")
    print(f"{'='*60}")
    print(f"  Nodes:     {min(nodes)} – {max(nodes)}")
    print(f"  Timesteps: {min(steps)} – {max(steps)}")
    print(f"  Edges:     {samples[0]['edge_index'].shape[1]}")
    print(f"  Dataset:   {ds}")

    # Check each channel of the first sample
    d = samples[0]['nodes_dynamic']
    names = ['depth', 'vx', 'vy', 'rain']
    print(f"  Dynamic features (sample 0):")
    for ch, name in enumerate(names):
        col = d[:, :, ch]
        print(f"    ch{ch} ({name:5s}): min={col.min():.4f} max={col.max():.4f} "
              f"mean={col.mean():.4f} frac>0={float((col > 0).mean()):.3f}")

    s = samples[0]['nodes_static']
    print(f"  Static (elevation): min={s.min():.4f} max={s.max():.4f}")
    print()


# ------------------------------------------------------------------ #
# SELF-TEST
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import sys

    print("Self-testing unified_loader.py ...\n")

    # --- Test grid edge builder ---
    edges = _grid_edges(3, 3)
    assert edges.shape[0] == 2, f"edge_index should be (2, E), got {edges.shape}"
    # 3x3 grid: 12 edges total (each of the 12 neighbour pairs, counted once each direction)
    # Actually 4-connected 3x3 = 2*(3*2) + 2*(3*2) = 24 directed edges
    # Interior edges: horizontal 3*2=6, vertical 2*3=6, total undirected=12, directed=24
    assert edges.shape[1] == 24, f"3x3 grid should have 24 directed edges, got {edges.shape[1]}"

    # Node 4 (center) should connect to 1, 3, 5, 7
    center_neighbours = set(edges[1, edges[0] == 4].tolist())
    assert center_neighbours == {1, 3, 5, 7}, f"center node 4 neighbours wrong: {center_neighbours}"
    print("  Grid edge builder: OK (3x3 grid, 24 edges, center connects to 1,3,5,7)")

    # --- Test with synthetic breach-style data ---
    import tempfile
    dim = 4
    T = 5
    N = dim * dim
    tmpdir = tempfile.mkdtemp()
    for folder in ['DEM', 'WD', 'VX', 'VY']:
        os.makedirs(os.path.join(tmpdir, folder))

    # DEM: x, y, elevation
    coords = np.array([[i, j, np.random.rand()]
                        for i in range(dim) for j in range(dim)])
    np.savetxt(os.path.join(tmpdir, 'DEM', 'DEM_1.txt'), coords)

    # WD, VX, VY: T rows, N columns
    np.savetxt(os.path.join(tmpdir, 'WD', 'WD_1.txt'),
               np.abs(np.random.rand(T, N)))
    np.savetxt(os.path.join(tmpdir, 'VX', 'VX_1.txt'),
               np.random.randn(T, N) * 0.3)
    np.savetxt(os.path.join(tmpdir, 'VY', 'VY_1.txt'),
               np.random.randn(T, N) * 0.3)

    samples = load_breach(tmpdir, [1], dim=dim)
    assert len(samples) == 1
    s = samples[0]
    assert s['nodes_dynamic'].shape == (N, T, 4)
    assert s['nodes_static'].shape == (N, 1)
    assert s['edge_index'].shape[0] == 2
    assert s['dataset'] == 'breach'
    assert (s['nodes_dynamic'][:, :, 3] == 0).all(), "rain channel should be all zeros for breach"
    summarise(samples, "SYNTHETIC BREACH")

    # Clean up
    import shutil
    shutil.rmtree(tmpdir)
    print("  Synthetic breach loader: OK")
    print("\nAll self-tests passed.")
    print("\nTo use with real data:")
    print("  harvey = load_harvey('data/harvey/test.npz')")
    print("  breach = load_breach('data/breach/raw', range(501, 520), dim=64)")