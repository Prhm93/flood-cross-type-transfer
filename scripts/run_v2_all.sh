#!/bin/bash
# run_v2_all.sh
# =============
# Everything, in order, one command. Launch it and walk away.
#
#   cd ~/projects/floodtransfer
#   source venv/bin/activate
#   bash scripts/run_v2_all.sh 2>&1 | tee results/run_v2_log.txt
#
# The tee writes a full log to results/run_v2_log.txt so nothing is lost.
#
# Rough timings on one A100:
#   smoke test          1 min
#   data cache build    5 min   (once — then every run loads in seconds)
#   Exp 1 thresholds    5 min   (no training)
#   Exp 3 controlled   40 min   (generates its own data)
#   Exp 2 no-rain      60 min
#   Exp 5 architecture 90 min
#   Exp 4 finetune     90 min
#   TOTAL             ~5 hours

set -e
START=$(date +%s)
mkdir -p results

echo "############################################################"
echo "#  STEP 0 — SMOKE TEST"
echo "############################################################"
python3 scripts/smoke_test.py || {
  echo ""
  echo "Smoke test reported failures. If the ONLY failures were"
  echo "'real datasets present' or 'checkpoints present', that is fine"
  echo "for the code itself, but those experiments will be skipped."
  echo "Continuing in 10 seconds — press Ctrl+C to stop."
  sleep 10
}

echo ""
echo "############################################################"
echo "#  STEP 1 — BUILD DATA CACHE (once, saves hours later)"
echo "############################################################"
python3 -c "
import sys; sys.path.insert(0,'.')
from src.experiment_lib import build_cache
build_cache()
"

echo ""
echo "############################################################"
echo "#  EXPERIMENT 1 — CSI THRESHOLD SENSITIVITY (no training)"
echo "#  Answers: is the CSI paradox real or a threshold artefact?"
echo "############################################################"
python3 scripts/exp_suite.py --only 1

echo ""
echo "############################################################"
echo "#  EXPERIMENT 3 — CONTROLLED CROSS-MECHANISM TRANSFER"
echo "#  Answers: does transfer still fail when terrain, solver,"
echo "#  grid and sample count are IDENTICAL and only the water"
echo "#  source changes? This kills the confounding objection."
echo "############################################################"
python3 scripts/exp3_controlled_synthetic.py

echo ""
echo "############################################################"
echo "#  EXPERIMENT 2 — RAINFALL ABLATION"
echo "#  Answers: does transfer still fail with NO rainfall channel"
echo "#  in either dataset? Rules out 'you removed an input'."
echo "############################################################"
python3 scripts/exp_suite.py --only 2

echo ""
echo "############################################################"
echo "#  EXPERIMENT 5 — STRONGER ARCHITECTURE"
echo "#  Answers: does the failure survive a bigger model?"
echo "#  Rules out 'your model was too weak'."
echo "############################################################"
python3 scripts/exp_suite.py --only 5

echo ""
echo "############################################################"
echo "#  EXPERIMENT 4 — FINE-TUNING CURVE (independent events)"
echo "#  Answers: how many independent target events are needed,"
echo "#  and does pre-training beat training from scratch?"
echo "############################################################"
python3 scripts/exp_suite.py --only 4

END=$(date +%s)
echo ""
echo "############################################################"
echo "#  ALL EXPERIMENTS COMPLETE"
echo "#  Total time: $(( (END-START)/60 )) minutes"
echo "############################################################"
ls -la results/result_exp*.json results/result_persistence.json 2>/dev/null
echo ""
echo "Full log saved if you used 'tee'. Send me the log and I will"
echo "write the results section and the revised paper."