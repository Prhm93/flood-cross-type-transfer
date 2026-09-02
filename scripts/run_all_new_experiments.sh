#!/bin/bash
# run_all_new_experiments.sh
# ==========================
# Runs all three new experiments in sequence.
# Total time: approximately 2-3 hours on one GPU.
#
# Usage:
#   cd ~/projects/floodtransfer
#   source venv/bin/activate
#   bash scripts/run_all_new_experiments.sh

set -e
echo "Starting all new experiments at $(date)"
echo ""

# ============================================
# EXPERIMENT 2: Persistence baseline (no GPU, fast)
# ============================================
echo "============================================"
echo "  EXPERIMENT 2: Persistence baseline"
echo "============================================"
python3 scripts/eval_persistence.py

# ============================================
# EXPERIMENT 1: Mixed training (6 runs)
# ============================================
for var in vector scalar; do
  for seed in 0 1 2; do
    echo ""
    echo "============================================"
    echo "  EXPERIMENT 1: Mixed training — $var seed $seed"
    echo "============================================"
    python3 scripts/train_mixed.py --variant $var --seed $seed
  done
done

# ============================================
# EXPERIMENT 3: Fine-tuning (12 runs)
# ============================================
for base in breach harvey; do
  for n in 5 10; do
    for seed in 0 1 2; do
      echo ""
      echo "============================================"
      echo "  EXPERIMENT 3: Fine-tune ${base}→target, n=$n, seed $seed"
      echo "============================================"
      python3 scripts/train_finetune.py --base $base --ft-samples $n --seed $seed
    done
  done
done

echo ""
echo "============================================"
echo "  ALL NEW EXPERIMENTS COMPLETE"
echo "============================================"
echo "Finished at $(date)"
echo ""
echo "Results:"
ls -la results/result_persistence.json
ls -la results/result_mixed_*.json
ls -la results/result_finetune_*.json