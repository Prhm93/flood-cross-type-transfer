"""
train_mixed.py
==============

EXPERIMENT 1: Train on BOTH flood types together.

The question: if we give the model breach AND rainfall data during
training, does it transfer better to each type individually?

Three possible outcomes, all publishable:
  A) Mixed training helps both → "train on diverse data" is the fix
  B) Mixed training helps one, hurts the other → interference
  C) Mixed training helps neither → the types are fundamentally incompatible

Usage:
    python3 scripts/train_mixed.py --variant vector --seed 0
    python3 scripts/train_mixed.py --variant scalar --seed 0

Full experiment (6 runs):
    for var in vector scalar; do
      for seed in 0 1 2; do
        python3 scripts/train_mixed.py --variant $var --seed $seed
      done
    done
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.unified_loader import load_harvey, load_breach, summarise
from src.normalise import fit_scaler, apply_scaler, save_scaler
from src.model import FloodGNN, count_params
from src.metrics import score, transfer_gap

BREACH_TRAIN_IDS = list(range(1, 61))
BREACH_VAL_IDS   = list(range(61, 81))
BREACH_TEST_IDS  = list(range(501, 520))


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, samples, optimizer, device, max_samples=None):
    """Same training loop as train.py."""
    model.train()
    total_loss = 0.0
    n_steps = 0

    if max_samples:
        indices = np.random.choice(len(samples), min(max_samples, len(samples)), replace=False)
    else:
        indices = range(len(samples))

    for idx in indices:
        s = samples[idx]
        dyn = torch.tensor(s['nodes_dynamic'], dtype=torch.float32, device=device)
        sta = torch.tensor(s['nodes_static'], dtype=torch.float32, device=device)
        edge_index = torch.tensor(s['edge_index'], dtype=torch.long, device=device)

        N, T, _ = dyn.shape
        if T < 3:
            continue

        window = min(20, T - 1)
        t_start = np.random.randint(0, T - window)

        epoch_loss = 0.0
        for t in range(t_start, t_start + window):
            current = dyn[:, t, :]
            next_state = dyn[:, t + 1, :]
            target = next_state[:, :3] - current[:, :3]
            pred_delta = model(current, sta, edge_index)
            loss = nn.functional.mse_loss(pred_delta, target)
            epoch_loss += loss
            n_steps += 1

        if optimizer is not None:
            optimizer.zero_grad()
            (epoch_loss / window).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            epoch_loss = epoch_loss.detach()

        total_loss += epoch_loss.item()

    return total_loss / max(n_steps, 1)


@torch.no_grad()
def evaluate_rollout(model, samples, device, max_steps=40, max_samples=20):
    """Same evaluation as train.py."""
    model.eval()
    all_scores = []

    for idx in range(min(max_samples, len(samples))):
        s = samples[idx]
        dyn = torch.tensor(s['nodes_dynamic'], dtype=torch.float32, device=device)
        sta = torch.tensor(s['nodes_static'], dtype=torch.float32, device=device)
        edge_index = torch.tensor(s['edge_index'], dtype=torch.long, device=device)

        N, T, _ = dyn.shape
        steps = min(max_steps, T - 1)
        if steps < 2:
            continue

        current = dyn[:, 0, :].clone()
        pred_depths = [current[:, 0].cpu().numpy()]
        true_depths = [dyn[:, 0, 0].cpu().numpy()]

        for t in range(steps):
            delta = model(current, sta, edge_index)
            new_state = current.clone()
            new_state[:, :3] = current[:, :3] + delta
            new_state[:, 0] = torch.clamp(new_state[:, 0], min=0.0)
            if t + 1 < T:
                new_state[:, 3] = dyn[:, t + 1, 3]
            current = new_state
            pred_depths.append(current[:, 0].cpu().numpy())
            true_depths.append(dyn[:, min(t + 1, T - 1), 0].cpu().numpy())

        pred_seq = np.stack(pred_depths, axis=0)
        true_seq = np.stack(true_depths, axis=0)
        pred_3d = pred_seq[:, np.newaxis, :]
        true_3d = true_seq[:, np.newaxis, :]

        s_result = score(pred_3d, true_3d, dt_seconds=3600.0, threshold=None)
        all_scores.append(s_result)

    if not all_scores:
        return {'csi_final': 0.0, 'csi_mean': 0.0, 'arrival_mae_s': float('inf'),
                'mass_error': float('inf'), 'n_samples': 0}

    return {
        'csi_final': float(np.nanmean([s['csi_final'] for s in all_scores])),
        'csi_mean': float(np.nanmean([s['csi_mean'] for s in all_scores])),
        'arrival_mae_s': float(np.nanmean([s['arrival_mae_s'] for s in all_scores])),
        'mass_error': float(np.nanmean([s['mass_error'] for s in all_scores])),
        'n_samples': len(all_scores),
    }


def main():
    parser = argparse.ArgumentParser(description="Train on BOTH flood types together")
    parser.add_argument('--variant', choices=['vector', 'scalar'], default='vector')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--max-train-samples', type=int, default=200)
    parser.add_argument('--max-eval-samples', type=int, default=19)
    parser.add_argument('--rollout-steps', type=int, default=40)
    parser.add_argument('--breach-dir', default='data/breach/raw_datasets')
    parser.add_argument('--harvey-dir', default='data/harvey')
    parser.add_argument('--out-dir', default='results')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tag = f"mixed_{args.variant}_s{args.seed}"

    print(f"\nDevice: {device}")
    print(f"Training on: MIXED (breach + harvey)")
    print(f"Variant: {args.variant}")
    print(f"Seed: {args.seed}")
    print(f"Run tag: {tag}")

    # Load everything
    print("\n=== Loading datasets ===")
    breach_train = load_breach(args.breach_dir, BREACH_TRAIN_IDS, dim=64)
    breach_val   = load_breach(args.breach_dir, BREACH_VAL_IDS, dim=64)
    breach_test  = load_breach(args.breach_dir, BREACH_TEST_IDS, dim=64)
    harvey_train = load_harvey(os.path.join(args.harvey_dir, 'train.npz'))
    harvey_val   = load_harvey(os.path.join(args.harvey_dir, 'val.npz'))
    harvey_test  = load_harvey(os.path.join(args.harvey_dir, 'test.npz'))

    # Joint normalisation
    print("\n=== Fitting joint scaler ===")
    all_samples = breach_train + breach_val + harvey_train + harvey_val
    scaler = fit_scaler(all_samples)
    os.makedirs(args.out_dir, exist_ok=True)

    breach_train = apply_scaler(breach_train, scaler)
    breach_val   = apply_scaler(breach_val, scaler)
    breach_test  = apply_scaler(breach_test, scaler)
    harvey_train = apply_scaler(harvey_train, scaler)
    harvey_val   = apply_scaler(harvey_val, scaler)
    harvey_test  = apply_scaler(harvey_test, scaler)

    # MIXED training set: combine both
    mixed_train = breach_train + harvey_train
    mixed_val   = breach_val + harvey_val
    np.random.shuffle(mixed_train)

    print(f"\nMixed training samples: {len(mixed_train)} "
          f"({len(breach_train)} breach + {len(harvey_train)} harvey)")

    # Build model
    model = FloodGNN(variant=args.variant, hidden=args.hidden, num_layers=args.layers)
    model = model.to(device)
    print(f"Model parameters: {count_params(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Train
    print(f"\n{'='*60}")
    print(f"  TRAINING — {args.epochs} epochs on MIXED data")
    print(f"{'='*60}")

    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, mixed_train, optimizer, device,
                                      max_samples=args.max_train_samples)
        scheduler.step(train_loss)
        dt = time.time() - t0

        if train_loss < best_loss:
            best_loss = train_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.out_dir, f'best_{tag}.pt'))

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d}  train_loss={train_loss:.6f}  "
                  f"time={dt:.1f}s  lr={optimizer.param_groups[0]['lr']:.1e}")

    print(f"\n  Best epoch: {best_epoch} (loss {best_loss:.6f})")
    model.load_state_dict(torch.load(os.path.join(args.out_dir, f'best_{tag}.pt'),
                                      map_location=device, weights_only=True))

    # Evaluate on BOTH test sets
    print(f"\n{'='*60}")
    print(f"  EVALUATION — mixed model on each dataset")
    print(f"{'='*60}")

    print(f"\n  Evaluating on BREACH test set...")
    breach_scores = evaluate_rollout(model, breach_test, device,
                                      max_steps=args.rollout_steps,
                                      max_samples=args.max_eval_samples)

    print(f"  Evaluating on HARVEY test set...")
    harvey_scores = evaluate_rollout(model, harvey_test, device,
                                      max_steps=args.rollout_steps,
                                      max_samples=args.max_eval_samples)

    # Print results
    print(f"\n{'='*60}")
    print(f"  RESULTS — {tag}")
    print(f"{'='*60}")
    print(f"  BREACH test:")
    print(f"    CSI final:     {breach_scores['csi_final']:.4f}")
    print(f"    CSI mean:      {breach_scores['csi_mean']:.4f}")
    print(f"    Arrival MAE:   {breach_scores['arrival_mae_s']:.0f} s")
    print(f"    Mass error:    {breach_scores['mass_error']:.4f}")
    print(f"  HARVEY test:")
    print(f"    CSI final:     {harvey_scores['csi_final']:.4f}")
    print(f"    CSI mean:      {harvey_scores['csi_mean']:.4f}")
    print(f"    Arrival MAE:   {harvey_scores['arrival_mae_s']:.0f} s")
    print(f"    Mass error:    {harvey_scores['mass_error']:.4f}")

    # Save
    result = {
        'tag': tag,
        'train_on': 'mixed',
        'variant': args.variant,
        'seed': args.seed,
        'epochs': args.epochs,
        'best_epoch': best_epoch,
        'model_params': count_params(model),
        'breach_scores': breach_scores,
        'harvey_scores': harvey_scores,
    }

    result_path = os.path.join(args.out_dir, f'result_{tag}.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Results saved to {result_path}")


if __name__ == '__main__':
    main()