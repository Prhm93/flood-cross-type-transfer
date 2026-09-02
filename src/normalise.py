"""
normalise.py
============

Bring both datasets onto the same scale so a model trained on one
can be tested on the other without one dataset's numbers being 10x
bigger than the other's.

HOW IT WORKS
------------
For each of the 4 dynamic channels (depth, vx, vy, rain) and the
1 static channel (elevation), we find the min and max across ALL
samples in BOTH datasets, then scale everything to [0, 1].

Velocity can be negative (flow going left or down), so we scale
it to [0, 1] where 0.5 means zero flow. This preserves the sign
information — values below 0.5 mean negative, above 0.5 mean positive.

WHY JOINTLY
-----------
If we normalised each dataset on its own, "0.5" would mean different
things in each dataset. Joint normalisation means "0.5 depth" is
the same physical amount in both — which is what transfer testing needs.

USAGE
-----
    from src.normalise import fit_scaler, apply_scaler

    # Fit on BOTH datasets together
    scaler = fit_scaler(breach_samples + harvey_samples)

    # Apply to each
    breach_normed = apply_scaler(breach_samples, scaler)
    harvey_normed = apply_scaler(harvey_samples, scaler)
"""

import numpy as np
import json
import copy


def fit_scaler(samples):
    """
    Walk through all samples and find the global min/max for each channel.

    Returns a dict with min/max for dynamic channels 0-3 and static channel 0.
    """
    # Start with extreme values that will be replaced immediately
    d_min = np.full(4, np.inf)
    d_max = np.full(4, -np.inf)
    s_min = np.inf
    s_max = -np.inf

    for s in samples:
        dyn = s['nodes_dynamic']  # (N, T, 4)
        sta = s['nodes_static']   # (N, 1)

        for ch in range(4):
            col = dyn[:, :, ch]
            d_min[ch] = min(d_min[ch], col.min())
            d_max[ch] = max(d_max[ch], col.max())

        s_min = min(s_min, sta.min())
        s_max = max(s_max, sta.max())

    scaler = {
        'dynamic_min': d_min.tolist(),
        'dynamic_max': d_max.tolist(),
        'static_min': float(s_min),
        'static_max': float(s_max),
    }
    
    # Print what we found
    names = ['depth', 'vx', 'vy', 'rain']
    print("Scaler fitted on all samples:")
    for ch in range(4):
        print(f"  {names[ch]:5s}: min={d_min[ch]:.4f}  max={d_max[ch]:.4f}  "
              f"range={d_max[ch]-d_min[ch]:.4f}")
    print(f"  elev : min={s_min:.4f}  max={s_max:.4f}  "
          f"range={s_max-s_min:.4f}")

    return scaler


def apply_scaler(samples, scaler):
    """
    Scale all samples to [0, 1] using the fitted min/max.

    Returns NEW sample dicts (does not modify the originals).
    """
    d_min = np.array(scaler['dynamic_min'], dtype=np.float32)
    d_max = np.array(scaler['dynamic_max'], dtype=np.float32)
    d_range = d_max - d_min

    # Avoid division by zero (if a channel is constant, keep it at 0)
    d_range[d_range == 0] = 1.0

    s_min = scaler['static_min']
    s_range = scaler['static_max'] - s_min
    if s_range == 0:
        s_range = 1.0

    normed = []
    for s in samples:
        new_s = copy.copy(s)  # shallow copy — we replace the arrays

        # Dynamic: (N, T, 4) -> scale each channel
        dyn = s['nodes_dynamic'].copy()
        dyn = (dyn - d_min) / d_range
        new_s['nodes_dynamic'] = dyn.astype(np.float32)

        # Static: (N, 1) -> scale
        sta = s['nodes_static'].copy()
        sta = (sta - s_min) / s_range
        new_s['nodes_static'] = sta.astype(np.float32)

        normed.append(new_s)

    return normed


def save_scaler(scaler, path):
    """Save the scaler to a JSON file so it can be reloaded later."""
    with open(path, 'w') as f:
        json.dump(scaler, f, indent=2)
    print(f"Scaler saved to {path}")


def load_scaler(path):
    """Load a saved scaler."""
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------ #
# Self-test
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("Self-testing normalise.py ...\n")

    # Make two fake samples with known ranges
    s1 = {
        'nodes_dynamic': np.array([[[0.0, -2.0, -1.0, 0.0],
                                     [4.0,  2.0,  1.0, 10.0]]]),  # (1, 2, 4)
        'nodes_static': np.array([[0.0]]),
    }
    s2 = {
        'nodes_dynamic': np.array([[[2.0, 0.0, 0.0, 5.0],
                                     [2.0, 0.0, 0.0, 5.0]]]),
        'nodes_static': np.array([[100.0]]),
    }

    scaler = fit_scaler([s1, s2])

    # Check the ranges
    assert scaler['dynamic_min'] == [0.0, -2.0, -1.0, 0.0]
    assert scaler['dynamic_max'] == [4.0, 2.0, 1.0, 10.0]
    print("  fit_scaler ranges: OK")

    # Apply and check
    normed = apply_scaler([s1, s2], scaler)

    # s1 depth: [0, 4] -> [0.0, 1.0]
    assert abs(normed[0]['nodes_dynamic'][0, 0, 0] - 0.0) < 1e-6
    assert abs(normed[0]['nodes_dynamic'][0, 1, 0] - 1.0) < 1e-6

    # s1 vx: [-2, 2] -> [0.0, 1.0], so -2 -> 0.0, 2 -> 1.0
    assert abs(normed[0]['nodes_dynamic'][0, 0, 1] - 0.0) < 1e-6
    assert abs(normed[0]['nodes_dynamic'][0, 1, 1] - 1.0) < 1e-6

    # s2 static: [0, 100] -> [0.0, 1.0]
    assert abs(normed[1]['nodes_static'][0, 0] - 1.0) < 1e-6
    print("  apply_scaler values: OK")

    print("\nAll self-tests passed.")