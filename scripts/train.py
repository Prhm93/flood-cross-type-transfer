"""
train.py
========

Train a flood GNN on one dataset, test on both, measure the transfer gap.

This is the script that produces the paper's central finding.

USAGE
-----
    # Train on breach data, test on breach AND harvey
    python3 scripts/train.py --train-on breach --variant vector --seed 0

    # Train on harvey, test on both
    python3 scripts/train.py --train-on harvey --variant vector --seed 0

    # Scalar variant (throws away direction)
    python3 scripts/train.py --train-on breach --variant scalar --seed 0

THE FULL EXPERIMENT (12 runs)
-----------------------------
    For each of 2 training sources (breach, harvey):
      For each of 2 variants (vector, scalar):
        For each of 3 seeds (0, 1, 2):
          Train and evaluate.

    That gives 12 runs. Each takes a few minutes on GPU.
    The paper's table comes from comparing the 12 results.

WHAT IT DOES STEP BY STEP
--------------------------
    1. Load both datasets through the unified loader
    2. Normalise jointly (same scale for both)
    3. Train the model on ONE dataset (teacher-forced, one-step prediction)
    4. Evaluate on BOTH datasets (autoregressive rollout, multi-step)
    5. Compute transfer gap (how much skill is lost away from home)
    6. Save everything to results/
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

# Add the project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.unified_loader import load_harvey, load_breach, summarise
from src.normalise import fit_scaler, apply_scaler, save_scaler, load_scaler
from src.model import FloodGNN, count_params
from src.metrics import score, transfer_gap


# ------------------------------------------------------------------ #
# CONFIG
# ------------------------------------------------------------------ #

# SWE-GNN sim ID splits (from the original paper)
BREACH_TRAIN_IDS = list(range(1, 61))      # 60 sims for training
BREACH_VAL_IDS   = list(range(61, 81))     # 20 sims for validation
BREACH_TEST_IDS  = list(range(501, 520))   # 19 sims for testing (520 absent)


def set_seed(seed):
    """Make everything reproducible."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------ #
# DATA PREPARATION
# ------------------------------------------------------------------ #

def prepare_data(args):
    """Load both datasets, normalise jointly, return train/val/test splits."""

    print("\n=== Loading datasets ===")

    # Breach data
    breach_train = load_breach(args.breach_dir, BREACH_TRAIN_IDS, dim=64)
    breach_val   = load_breach(args.breach_dir, BREACH_VAL_IDS, dim=64)
    breach_test  = load_breach(args.breach_dir, BREACH_TEST_IDS, dim=64)

    # Harvey data
    harvey_train = load_harvey(os.path.join(args.harvey_dir, 'train.npz'))
    harvey_val   = load_harvey(os.path.join(args.harvey_dir, 'val.npz'))
    harvey_test  = load_harvey(os.path.join(args.harvey_dir, 'test.npz'))

    print("\n=== Fitting joint scaler ===")
    all_samples = breach_train + breach_val + harvey_train + harvey_val
    scaler = fit_scaler(all_samples)

    # Save the scaler for reproducibility
    os.makedirs(args.out_dir, exist_ok=True)
    save_scaler(scaler, os.path.join(args.out_dir, 'scaler.json'))

    # Apply normalisation
    breach_train = apply_scaler(breach_train, scaler)
    breach_val   = apply_scaler(breach_val, scaler)
    breach_test  = apply_scaler(breach_test, scaler)
    harvey_train = apply_scaler(harvey_train, scaler)
    harvey_val   = apply_scaler(harvey_val, scaler)
    harvey_test  = apply_scaler(harvey_test, scaler)

    return {
        'breach': {'train': breach_train, 'val': breach_val, 'test': breach_test},
        'harvey': {'train': harvey_train, 'val': harvey_val, 'test': harvey_test},
        'scaler': scaler,
    }


# ------------------------------------------------------------------ #
# TRAINING LOOP
# ------------------------------------------------------------------ #

def train_one_epoch(model, samples, optimizer, device, max_samples=None):
    """
    One pass through the training data.

    Teacher-forced: at each step, the model sees the TRUE current state
    and predicts the change to the next step. It does NOT see its own
    previous predictions during training.

    This is standard practice (SWE-GNN, FloodGNN-GRU both do this).
    """
    model.train()
    total_loss = 0.0
    n_steps = 0

    if max_samples:
        indices = np.random.choice(len(samples), min(max_samples, len(samples)), replace=False)
    else:
        indices = range(len(samples))

    for idx in indices:
        s = samples[idx]
        dyn = torch.tensor(s['nodes_dynamic'], dtype=torch.float32, device=device)   # (N, T, 4)
        sta = torch.tensor(s['nodes_static'], dtype=torch.float32, device=device)    # (N, 1)
        edge_index = torch.tensor(s['edge_index'], dtype=torch.long, device=device)  # (2, E)

        N, T, _ = dyn.shape

        # Skip very short sequences
        if T < 3:
            continue

        # Pick a random window of steps to train on (keeps memory bounded)
        window = min(20, T - 1)
        t_start = np.random.randint(0, T - window)

        epoch_loss = 0.0
        for t in range(t_start, t_start + window):
            current = dyn[:, t, :]        # (N, 4) — true current state
            next_state = dyn[:, t + 1, :]  # (N, 4) — true next state

            # True change (target)
            target = next_state[:, :3] - current[:, :3]  # (N, 3) — delta depth, vx, vy

            # Predict the change
            pred_delta = model(current, sta, edge_index)  # (N, 3)

            loss = nn.functional.mse_loss(pred_delta, target)
            epoch_loss += loss

            n_steps += 1

        # Backprop over the whole window (skip if no optimizer — validation mode)
        if optimizer is not None:
            optimizer.zero_grad()
            (epoch_loss / window).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            epoch_loss = epoch_loss.detach()

        total_loss += epoch_loss.item()

    return total_loss / max(n_steps, 1)


# ------------------------------------------------------------------ #
# EVALUATION — autoregressive rollout
# ------------------------------------------------------------------ #

@torch.no_grad()
def evaluate_rollout(model, samples, device, max_steps=40, max_samples=20):
    """
    The honest test: model predicts step by step, feeding its own
    predictions back in. Small errors accumulate.

    Returns the metrics from metrics.py for each sample, then averages.
    """
    model.eval()
    all_scores = []

    indices = range(min(max_samples, len(samples)))

    for idx in indices:
        s = samples[idx]
        dyn = torch.tensor(s['nodes_dynamic'], dtype=torch.float32, device=device)
        sta = torch.tensor(s['nodes_static'], dtype=torch.float32, device=device)
        edge_index = torch.tensor(s['edge_index'], dtype=torch.long, device=device)

        N, T, _ = dyn.shape
        steps = min(max_steps, T - 1)
        if steps < 2:
            continue

        # Start from the true initial state
        current = dyn[:, 0, :].clone()  # (N, 4)

        pred_depths = [current[:, 0].cpu().numpy()]
        true_depths = [dyn[:, 0, 0].cpu().numpy()]

        for t in range(steps):
            # Predict the change
            delta = model(current, sta, edge_index)  # (N, 3)

            # Apply the change
            new_state = current.clone()
            new_state[:, :3] = current[:, :3] + delta

            # Depth can never be negative
            new_state[:, 0] = torch.clamp(new_state[:, 0], min=0.0)

            # Rainfall comes from the true data (it is an external driver, not predicted)
            if t + 1 < T:
                new_state[:, 3] = dyn[:, t + 1, 3]

            current = new_state

            pred_depths.append(current[:, 0].cpu().numpy())
            true_depths.append(dyn[:, min(t + 1, T - 1), 0].cpu().numpy())

        # Stack into (T, N) arrays and score
        pred_seq = np.stack(pred_depths, axis=0)  # (steps+1, N)
        true_seq = np.stack(true_depths, axis=0)

        # Reshape to (T, 1, N) for metrics (they expect T, H, W but 1D is fine)
        pred_3d = pred_seq[:, np.newaxis, :]
        true_3d = true_seq[:, np.newaxis, :]

        s_result = score(pred_3d, true_3d, dt_seconds=3600.0, threshold=None)
        s_result['sample_id'] = s.get('sample_id', idx)
        all_scores.append(s_result)

    # Average across samples
    if not all_scores:
        return {'csi_final': 0.0, 'csi_mean': 0.0, 'arrival_mae_s': float('inf'),
                'mass_error': float('inf'), 'n_samples': 0}

    avg = {
        'csi_final': float(np.nanmean([s['csi_final'] for s in all_scores])),
        'csi_mean': float(np.nanmean([s['csi_mean'] for s in all_scores])),
        'arrival_mae_s': float(np.nanmean([s['arrival_mae_s'] for s in all_scores])),
        'mass_error': float(np.nanmean([s['mass_error'] for s in all_scores])),
        'n_samples': len(all_scores),
    }
    return avg


# ------------------------------------------------------------------ #
# MAIN
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Train flood GNN and measure transfer gap")

    parser.add_argument('--train-on', choices=['breach', 'harvey'], required=True,
                        help="Which flood type to TRAIN on")
    parser.add_argument('--variant', choices=['vector', 'scalar'], default='vector',
                        help="vector = keep vx,vy separate; scalar = use speed only")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--max-train-samples', type=int, default=50,
                        help="Cap training samples per epoch (keeps epoch time bounded)")
    parser.add_argument('--max-eval-samples', type=int, default=19,
                        help="Cap evaluation samples")
    parser.add_argument('--rollout-steps', type=int, default=40,
                        help="How many steps to roll out during evaluation")

    # Paths
    parser.add_argument('--breach-dir', default='data/breach/raw_datasets')
    parser.add_argument('--harvey-dir', default='data/harvey')
    parser.add_argument('--out-dir', default='results')

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Training on: {args.train_on}")
    print(f"Variant: {args.variant}")
    print(f"Seed: {args.seed}")

    # Run tag for filenames
    tag = f"{args.train_on}_{args.variant}_s{args.seed}"
    print(f"Run tag: {tag}")

    # Load and normalise
    data = prepare_data(args)

    # Pick training and validation sets
    train_samples = data[args.train_on]['train']
    val_samples   = data[args.train_on]['val']

    print(f"\nTraining samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")

    # Build model
    model = FloodGNN(variant=args.variant, hidden=args.hidden, num_layers=args.layers)
    model = model.to(device)
    print(f"Model parameters: {count_params(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training loop
    print(f"\n{'='*60}")
    print(f"  TRAINING — {args.epochs} epochs")
    print(f"{'='*60}")

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(args.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_samples, optimizer, device,
            max_samples=args.max_train_samples
        )

        # Quick validation loss
        val_loss = train_one_epoch(
            model, val_samples[:10], optimizer=None, device=device,
            max_samples=10
        ) if len(val_samples) > 0 else float('inf')

        scheduler.step(train_loss)
        dt = time.time() - t0

        if train_loss < best_val_loss:
            best_val_loss = train_loss
            best_epoch = epoch
            # Save best checkpoint
            ckpt_path = os.path.join(args.out_dir, f'best_{tag}.pt')
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d}  train_loss={train_loss:.6f}  "
                  f"time={dt:.1f}s  lr={optimizer.param_groups[0]['lr']:.1e}")

    print(f"\n  Best epoch: {best_epoch} (loss {best_val_loss:.6f})")

    # Load best checkpoint
    model.load_state_dict(torch.load(os.path.join(args.out_dir, f'best_{tag}.pt'),
                                      map_location=device, weights_only=True))

    # ------------------------------------------------------------ #
    # EVALUATION — the paper's finding
    # ------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f"  EVALUATION — rollout on both datasets")
    print(f"{'='*60}")

    home_test = data[args.train_on]['test']
    away_name = 'harvey' if args.train_on == 'breach' else 'breach'
    away_test = data[away_name]['test']

    print(f"\n  Evaluating on HOME ({args.train_on}) test set...")
    home_scores = evaluate_rollout(model, home_test, device,
                                    max_steps=args.rollout_steps,
                                    max_samples=args.max_eval_samples)

    print(f"  Evaluating on AWAY ({away_name}) test set...")
    away_scores = evaluate_rollout(model, away_test, device,
                                    max_steps=args.rollout_steps,
                                    max_samples=args.max_eval_samples)

    # The paper's number
    gap = transfer_gap(home_scores, away_scores, key='csi_final')

    # Print results
    print(f"\n{'='*60}")
    print(f"  RESULTS — {tag}")
    print(f"{'='*60}")
    print(f"  HOME ({args.train_on}):")
    print(f"    CSI final:     {home_scores['csi_final']:.4f}")
    print(f"    CSI mean:      {home_scores['csi_mean']:.4f}")
    print(f"    Arrival MAE:   {home_scores['arrival_mae_s']:.0f} s")
    print(f"    Mass error:    {home_scores['mass_error']:.4f}")
    print(f"  AWAY ({away_name}):")
    print(f"    CSI final:     {away_scores['csi_final']:.4f}")
    print(f"    CSI mean:      {away_scores['csi_mean']:.4f}")
    print(f"    Arrival MAE:   {away_scores['arrival_mae_s']:.0f} s")
    print(f"    Mass error:    {away_scores['mass_error']:.4f}")
    print(f"  TRANSFER GAP:")
    print(f"    Home CSI:      {gap['home']:.4f}")
    print(f"    Away CSI:      {gap['away']:.4f}")
    print(f"    Skill retained: {gap['retained_pct']:.1f}%")
    print(f"    Drop:          {gap['drop']:.4f}")

    # Save results
    result = {
        'tag': tag,
        'train_on': args.train_on,
        'variant': args.variant,
        'seed': args.seed,
        'epochs': args.epochs,
        'best_epoch': best_epoch,
        'model_params': count_params(model),
        'home_scores': home_scores,
        'away_scores': away_scores,
        'transfer_gap': gap,
    }

    result_path = os.path.join(args.out_dir, f'result_{tag}.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Results saved to {result_path}")


# ------------------------------------------------------------------ #
# Validation-loss helper (reuses train loop with no optimizer)
# ------------------------------------------------------------------ #

def _val_loss_helper():
    """
    The train_one_epoch function can be called with optimizer=None
    to compute loss without updating weights. This is a hack but
    keeps the code short. The actual validation is the rollout.
    """
    pass


if __name__ == '__main__':
    main()