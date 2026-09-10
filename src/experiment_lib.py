"""
experiment_lib.py
=================

Shared code for all experiments. Two jobs:

1. CACHE THE DATA. Loading Harvey's train.npz takes several minutes because
   it is 4.3 GB. With 30+ experimental runs that is hours wasted. We load
   once, normalise once, and save a compact cache. Every experiment after
   that loads in seconds.

2. SHARED TRAIN / EVAL. One copy of the training loop and the rollout
   evaluation, used by every experiment, so they cannot drift apart.
"""

import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.unified_loader import load_harvey, load_breach
from src.normalise import fit_scaler, apply_scaler
from src.model import FloodGNN, count_params
from src.metrics import score, auto_threshold

# Split definitions — EVENT-LEVEL, never snapshot-level.
# Breach sims 1-60 train, 61-80 val, 501-519 test are entirely separate
# simulations, so no event appears in more than one split.
BREACH_TRAIN_IDS = list(range(1, 61))
BREACH_VAL_IDS = list(range(61, 81))
BREACH_TEST_IDS = list(range(501, 520))

CACHE_PATH = 'data/cache_normalised.pkl'


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------ #
# DATA CACHE
# ------------------------------------------------------------------ #

def build_cache(breach_dir='data/breach/raw_datasets',
                harvey_dir='data/harvey',
                cache_path=CACHE_PATH,
                force=False):
    """
    Load both datasets, normalise jointly, save to a cache file.
    Run once. Every experiment then loads from the cache in seconds.
    """
    if os.path.exists(cache_path) and not force:
        print(f"Cache already exists at {cache_path} (use force=True to rebuild)")
        return cache_path

    print("Building data cache (this takes a few minutes, ONCE) ...")
    t0 = time.time()

    breach_train = load_breach(breach_dir, BREACH_TRAIN_IDS, dim=64)
    breach_val = load_breach(breach_dir, BREACH_VAL_IDS, dim=64)
    breach_test = load_breach(breach_dir, BREACH_TEST_IDS, dim=64)
    harvey_train = load_harvey(os.path.join(harvey_dir, 'train.npz'))
    harvey_val = load_harvey(os.path.join(harvey_dir, 'val.npz'))
    harvey_test = load_harvey(os.path.join(harvey_dir, 'test.npz'))

    scaler = fit_scaler(breach_train + breach_val + harvey_train + harvey_val)

    data = {
        'breach': {
            'train': apply_scaler(breach_train, scaler),
            'val': apply_scaler(breach_val, scaler),
            'test': apply_scaler(breach_test, scaler),
        },
        'harvey': {
            'train': apply_scaler(harvey_train, scaler),
            'val': apply_scaler(harvey_val, scaler),
            'test': apply_scaler(harvey_test, scaler),
        },
        'scaler': scaler,
    }

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    print(f"Cache built in {time.time() - t0:.0f}s -> {cache_path} "
          f"({os.path.getsize(cache_path) / 1e9:.2f} GB)")
    return cache_path


def load_cache(cache_path=CACHE_PATH):
    """Load the pre-normalised data. Fast."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"No cache at {cache_path}. Run build_cache() first "
            f"(the runner script does this automatically).")
    with open(cache_path, 'rb') as f:
        return pickle.load(f)


def strip_rainfall(samples):
    """
    Return copies of the samples with the rainfall channel (index 3) zeroed.

    Used by the rainfall-ablation experiment: if we remove rainfall from
    BOTH datasets, neither has it, so any remaining transfer failure cannot
    be blamed on 'the rainfall input disappeared'.
    """
    out = []
    for s in samples:
        s2 = dict(s)
        dyn = s['nodes_dynamic'].copy()
        dyn[:, :, 3] = 0.0
        s2['nodes_dynamic'] = dyn
        out.append(s2)
    return out


# ------------------------------------------------------------------ #
# TRAINING
# ------------------------------------------------------------------ #

def train_one_epoch(model, samples, optimizer, device, max_samples=None,
                    window=20):
    """One pass. If optimizer is None, computes loss without updating."""
    model.train() if optimizer is not None else model.eval()
    total_loss, n_steps = 0.0, 0

    if max_samples and len(samples) > max_samples:
        indices = np.random.choice(len(samples), max_samples, replace=False)
    else:
        indices = np.arange(len(samples))

    for idx in indices:
        s = samples[idx]
        dyn = torch.as_tensor(s['nodes_dynamic'], dtype=torch.float32, device=device)
        sta = torch.as_tensor(s['nodes_static'], dtype=torch.float32, device=device)
        ei = torch.as_tensor(s['edge_index'], dtype=torch.long, device=device)

        N, T, _ = dyn.shape
        if T < 3:
            continue

        w = min(window, T - 1)
        t0 = np.random.randint(0, T - w)

        loss_sum = 0.0
        for t in range(t0, t0 + w):
            cur = dyn[:, t, :]
            nxt = dyn[:, t + 1, :]
            target = nxt[:, :3] - cur[:, :3]
            pred = model(cur, sta, ei)
            loss_sum = loss_sum + nn.functional.mse_loss(pred, target)
            n_steps += 1

        if optimizer is not None:
            optimizer.zero_grad()
            (loss_sum / w).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            loss_sum = loss_sum.detach()

        total_loss += float(loss_sum.detach()) if torch.is_tensor(loss_sum) else float(loss_sum)

    return total_loss / max(n_steps, 1)


def train_model(model, train_samples, device, epochs=60, lr=1e-3,
                max_train_samples=200, ckpt_path=None, verbose=True,
                log_every=15):
    """Full training loop. Returns the best training loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5,
                                                        factor=0.5)
    best, best_ep = float('inf'), 0

    for ep in range(epochs):
        loss = train_one_epoch(model, train_samples, optimizer, device,
                                max_samples=max_train_samples)
        sched.step(loss)
        if loss < best:
            best, best_ep = loss, ep
            if ckpt_path:
                torch.save(model.state_dict(), ckpt_path)
        if verbose and (ep % log_every == 0 or ep == epochs - 1):
            print(f"    epoch {ep:3d}  loss={loss:.6f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.1e}")

    if ckpt_path and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                          weights_only=True))
    if verbose:
        print(f"    best epoch {best_ep} (loss {best:.6f})")
    return best


# ------------------------------------------------------------------ #
# EVALUATION
# ------------------------------------------------------------------ #

@torch.no_grad()
def rollout_predictions(model, samples, device, max_steps=40, max_samples=19):
    """
    Run autoregressive rollout and return the raw predicted and true depth
    sequences, WITHOUT scoring. Scoring is separate so the same rollout can
    be scored at many thresholds (the threshold-sweep experiment).

    Returns a list of (pred_seq, true_seq), each shaped (steps+1, N).
    """
    model.eval()
    out = []

    for idx in range(min(max_samples, len(samples))):
        s = samples[idx]
        dyn = torch.as_tensor(s['nodes_dynamic'], dtype=torch.float32, device=device)
        sta = torch.as_tensor(s['nodes_static'], dtype=torch.float32, device=device)
        ei = torch.as_tensor(s['edge_index'], dtype=torch.long, device=device)

        N, T, _ = dyn.shape
        steps = min(max_steps, T - 1)
        if steps < 2:
            continue

        cur = dyn[:, 0, :].clone()
        preds = [cur[:, 0].cpu().numpy()]
        trues = [dyn[:, 0, 0].cpu().numpy()]

        for t in range(steps):
            delta = model(cur, sta, ei)
            new = cur.clone()
            new[:, :3] = cur[:, :3] + delta
            new[:, 0] = torch.clamp(new[:, 0], min=0.0)
            if t + 1 < T:
                new[:, 3] = dyn[:, t + 1, 3]   # rainfall is an external driver
            cur = new
            preds.append(cur[:, 0].cpu().numpy())
            trues.append(dyn[:, min(t + 1, T - 1), 0].cpu().numpy())

        out.append((np.stack(preds, 0), np.stack(trues, 0)))

    return out


def score_rollouts(rollouts, threshold_mode='auto', threshold_value=None,
                   dt_seconds=3600.0):
    """
    Score a list of (pred, true) rollouts.

    threshold_mode:
      'auto'     — 1% of max true depth per sample (the original default)
      'relative' — threshold_value as a FRACTION of max true depth
      'fixed'    — threshold_value used directly as an absolute depth

    Returns mean metrics across samples.
    """
    scores = []
    for pred, true in rollouts:
        p3 = pred[:, np.newaxis, :]
        t3 = true[:, np.newaxis, :]

        if threshold_mode == 'auto':
            thr = None
        elif threshold_mode == 'relative':
            mx = float(np.max(true))
            thr = max(mx * threshold_value, 1e-9)
        elif threshold_mode == 'fixed':
            thr = threshold_value
        else:
            raise ValueError(threshold_mode)

        scores.append(score(p3, t3, dt_seconds=dt_seconds, threshold=thr))

    if not scores:
        return {'csi_final': np.nan, 'csi_mean': np.nan,
                'arrival_mae_s': np.nan, 'mass_error': np.nan, 'n': 0}

    return {
        'csi_final': float(np.nanmean([s['csi_final'] for s in scores])),
        'csi_mean': float(np.nanmean([s['csi_mean'] for s in scores])),
        'arrival_mae_s': float(np.nanmean([s['arrival_mae_s'] for s in scores])),
        'mass_error': float(np.nanmean([s['mass_error'] for s in scores])),
        'csi_final_std': float(np.nanstd([s['csi_final'] for s in scores])),
        'n': len(scores),
    }


def volume_ratio(rollouts):
    """
    Predicted total volume divided by true total volume.

    The reviewer correctly noted that 'mass error of 30' is ambiguous.
    This returns an unambiguous RATIO: 1.0 is perfect, 31.0 means the model
    predicted 31 times the true volume.
    """
    ratios = []
    for pred, true in rollouts:
        vt = float(np.sum(true, dtype=np.float64))
        vp = float(np.sum(pred, dtype=np.float64))
        if vt > 0:
            ratios.append(vp / vt)
    if not ratios:
        return np.nan
    return float(np.mean(ratios))


def evaluate(model, samples, device, max_steps=40, max_samples=19):
    """Convenience: rollout + score with the default auto threshold."""
    rollouts = rollout_predictions(model, samples, device, max_steps, max_samples)
    res = score_rollouts(rollouts, 'auto')
    res['volume_ratio'] = volume_ratio(rollouts)
    return res


def build_model(variant='vector', hidden=64, layers=3, device='cpu'):
    m = FloodGNN(variant=variant, hidden=hidden, num_layers=layers)
    return m.to(device)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')