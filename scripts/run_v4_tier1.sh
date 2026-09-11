#!/bin/bash
# run_v4_tier1.sh
# ================
#   tmux new -s flood3
#   cd ~/projects/floodtransfer
#   source venv/bin/activate
#   bash scripts/run_v4_tier1.sh 2>&1 | tee results/run_v4_log.txt
#   [Ctrl+B then D to detach]
#
# Rough timings:
#   split audit            1 min   no training
#   exp6 disagreement      5 min   no training, reuses checkpoints
#   exp7 balanced mixed   2.5 hrs  5 regimes x 3 seeds x 60 epochs
#   exp8 mechanism-aware  1.5 hrs  2 conditions x 3 seeds x 60 epochs
#   TOTAL                ~4.5 hrs
#
# NO set -e on purpose: one failing step must not cancel the expensive
# ones behind it. Each step reports its own success or failure.
#
# exp6 needs results/best_breach_vector_s0.pt and best_harvey_vector_s0.pt.
# If they are missing it prints [skip] and carries on.

mkdir -p results
START=$(date +%s)
FAILED=""

step () {   # step <label> <script>
  echo ""
  echo "############################################################"
  echo "#  $1"
  echo "############################################################"
  if python3 "$2"; then
    echo "--- OK: $1"
  else
    echo "--- FAILED: $1 (continuing with the rest)"
    FAILED="$FAILED\n  - $1"
  fi
}

step "STEP 0 - SPLIT AUDIT (train/test leakage check)" scripts/audit_splits.py
step "EXP 6 - PER-EVENT METRIC DISAGREEMENT (scatter data)" scripts/exp6_metric_disagreement.py
step "EXP 7 - BALANCED / WEIGHTED MIXED TRAINING" scripts/exp7_balanced_mixed.py
step "EXP 8 - MECHANISM-AWARE MODEL (CSI + volume)" scripts/exp8_mechanism_aware.py

END=$(date +%s)
echo ""
echo "############################################################"
echo "#  BATCH COMPLETE - total $(( (END-START)/60 )) minutes"
echo "############################################################"
if [ -n "$FAILED" ]; then
  echo "Steps that failed:"
  echo -e "$FAILED"
else
  echo "All steps completed without error."
fi
echo ""
ls -la results/result_exp6_metric_disagreement.json \
       results/result_exp7_balanced_mixed.json \
       results/result_exp8_mechanism_aware.json 2>&1
