"""
metrics.py
==========

Scoring for the cross-type transfer test.

Everything here takes plain numpy arrays, so it does not care which dataset
the numbers came from. That is deliberate: the whole point of the paper is
scoring the SAME way on two different flood types.

Shapes
------
Depth fields are (T, H, W):  T time steps, H rows, W columns, in metres.

The four things we measure
--------------------------
    csi()            did we get the flood MAP right?
    arrival_error()  did we get the TIMING right?
    mass_error()     did we invent or lose water?
    transfer_gap()   how much worse is the model away from home?

The last one is the paper.
"""

import numpy as np

WET = 0.05  # default: a cell counts as flooded above 5 cm


def auto_threshold(true_sequence):
    """
    Pick a sensible wet/dry threshold from the data itself.
    Uses 1% of the maximum true depth in the sequence.
    This adapts to whatever scale the data is on.
    """
    max_depth = float(np.max(true_sequence))
    if max_depth <= 0:
        return WET
    return max(max_depth * 0.01, 1e-6)


# ------------------------------------------------------------------ #
# 1. CSI — flood map accuracy
# ------------------------------------------------------------------ #
def csi(pred, true, threshold=WET):
    """
    Critical Success Index. Ignores exact depth, asks only: which cells are wet?

    CSI = hits / (hits + misses + false alarms)

    1.0 = perfect, 0.0 = completely wrong.
    """
    p = np.asarray(pred) > threshold      # cells we SAID were wet
    t = np.asarray(true) > threshold      # cells that ARE wet

    hits = np.logical_and(p, t).sum()          # said wet, is wet
    misses = np.logical_and(~p, t).sum()       # said dry, is wet
    false_alarms = np.logical_and(p, ~t).sum()  # said wet, is dry

    denom = hits + misses + false_alarms
    if denom == 0:
        return np.nan  # nothing wet anywhere in either field
    return float(hits / denom)


def csi_per_step(pred, true, threshold=WET):
    """CSI at every time step, so we can plot accuracy against forecast hour."""
    return np.array([csi(pred[t], true[t], threshold)
                     for t in range(pred.shape[0])])


# ------------------------------------------------------------------ #
# 2. Arrival time — timing accuracy
# ------------------------------------------------------------------ #
def arrival_index(depth, threshold=WET):
    """
    For each cell, the FIRST time step where it becomes wet.
    Cells that never get wet are marked -1.
    """
    wet = np.asarray(depth) > threshold          # (T, H, W) True/False
    ever_wet = wet.any(axis=0)                   # (H, W) did it ever flood?
    first = wet.argmax(axis=0)                   # argmax finds first True
    return np.where(ever_wet, first, -1)


def arrival_error(pred, true, dt_seconds=3600.0, threshold=WET):
    """
    Compare predicted arrival time with true arrival time, in seconds.

    Only scores cells that flood in BOTH fields — otherwise there is
    no shared event to compare.

    Returns mae (average error size), bias (positive = model is late),
    and n (how many cells were comparable).
    """
    a_pred = arrival_index(pred, threshold)
    a_true = arrival_index(true, threshold)

    both = (a_pred >= 0) & (a_true >= 0)
    if both.sum() == 0:
        return {"mae": np.nan, "bias": np.nan, "n": 0}

    diff = (a_pred[both].astype(float) - a_true[both].astype(float)) * dt_seconds
    return {"mae": float(np.abs(diff).mean()),
            "bias": float(diff.mean()),
            "n": int(both.sum())}


# ------------------------------------------------------------------ #
# 3. Mass error — did water appear from nowhere?
# ------------------------------------------------------------------ #
def mass_error(pred, true):
    """
    Relative error in TOTAL water. 0.03 means 3% too much water.

    float64 on purpose: float32 rounding is bigger than the signal
    when you add up thousands of small depths.
    """
    p = np.asarray(pred, dtype=np.float64).sum()
    t = np.asarray(true, dtype=np.float64).sum()
    if t == 0:
        return np.nan
    return float((p - t) / t)


# ------------------------------------------------------------------ #
# 4. Baselines — the sanity check the field keeps skipping
# ------------------------------------------------------------------ #
def persistence_depth(depth_sequence):
    """
    Laziest possible forecast: 'the water stays exactly where it is.'
    Every future frame = the first frame.
    """
    d = np.asarray(depth_sequence)
    return np.repeat(d[:1], d.shape[0], axis=0)


# ------------------------------------------------------------------ #
# 5. The whole score card for one run
# ------------------------------------------------------------------ #
def score(pred, true, dt_seconds=3600.0, threshold=None):
    """Run every metric at once and return a small dictionary.
    If threshold is None, it adapts to the data scale automatically."""
    pred = np.asarray(pred)
    true = np.asarray(true)
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs true {true.shape}")

    if threshold is None:
        threshold = auto_threshold(true)
    arr = arrival_error(pred, true, dt_seconds, threshold)
    per_step = csi_per_step(pred, true, threshold)

    return {
        "csi_final": csi(pred[-1], true[-1], threshold),
        "csi_mean": float(np.nanmean(per_step)),
        "csi_per_step": per_step.tolist(),
        "arrival_mae_s": arr["mae"],
        "arrival_bias_s": arr["bias"],
        "arrival_n": arr["n"],
        "mass_error": mass_error(pred, true),
        "threshold_used": float(threshold),
    }


# ------------------------------------------------------------------ #
# 6. THE PAPER'S NUMBER — how much worse is the model away from home?
# ------------------------------------------------------------------ #
def transfer_gap(home_score, away_score, key="csi_final"):
    """
    home_score : score() on the flood type the model was TRAINED on
    away_score : score() on the DIFFERENT flood type

    Returns the drop, and the drop as a percentage of home performance.

    Example: home CSI 0.80, away CSI 0.40
             -> drop 0.40, retained 50%. The model lost half its skill
                by changing flood type. That is the finding.
    """
    h = home_score[key]
    a = away_score[key]
    if h in (0, None) or np.isnan(h):
        return {"home": h, "away": a, "drop": np.nan, "retained_pct": np.nan}
    return {"home": float(h),
            "away": float(a),
            "drop": float(h - a),
            "retained_pct": float(100.0 * a / h)}


# ------------------------------------------------------------------ #
# Self-test: run this file directly to check the maths is right.
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("Self-testing metrics.py ...")

    # --- CSI against the worked example: 80 hits, 20 misses, 10 false alarms
    t = np.zeros(200); t[:100] = 1.0            # 100 wet cells
    p = np.zeros(200); p[:80] = 1.0; p[100:110] = 1.0  # 80 hits + 10 false
    got = csi(p, t)
    expect = 80 / (80 + 20 + 10)
    assert abs(got - expect) < 1e-9, (got, expect)
    print(f"  CSI worked example: {got:.4f} == {expect:.4f}  OK")

    # --- arrival: build a front that the model reports exactly 1 step late
    T, H, W = 6, 4, 4
    truth = np.zeros((T, H, W))
    for tt in range(T):
        truth[tt, :, :tt] = 1.0        # water spreads one column per step
    late = np.zeros((T, H, W))
    for tt in range(1, T):
        late[tt, :, :tt - 1] = 1.0     # same, one step behind

    a = arrival_error(late, truth, dt_seconds=3600.0)
    assert abs(a["bias"] - 3600.0) < 1e-6, a
    print(f"  arrival bias on a 1-step-late front: +{a['bias']:.0f} s  OK")

    # --- mass error
    assert abs(mass_error(np.array([1.1]), np.array([1.0])) - 0.1) < 1e-9
    print("  mass error 10% case  OK")

    # --- transfer gap
    g = transfer_gap({"csi_final": 0.80}, {"csi_final": 0.40})
    assert abs(g["retained_pct"] - 50.0) < 1e-9, g
    print(f"  transfer gap: kept {g['retained_pct']:.0f}% of skill  OK")

    # --- persistence baseline keeps frame 0 everywhere
    seq = np.random.rand(5, 3, 3)
    pers = persistence_depth(seq)
    assert np.allclose(pers[3], seq[0])
    print("  persistence baseline  OK")

    print("All self-tests passed.")
    