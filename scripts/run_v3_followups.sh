#!/bin/bash
# run_v3_followups.sh
# ====================
# The three follow-up experiments, in order. Run inside tmux.
#
#   tmux new -s flood2
#   cd ~/projects/floodtransfer
#   source venv/bin/activate
#   bash scripts/run_v3_followups.sh 2>&1 | tee results/run_v3_log.txt
#   [Ctrl+B then D to detach]
#
# Rough timings on one A100:
#   Exp 1b arrival time    5 min   (no training, reuses checkpoints)
#   Exp 5 more seeds       2 hrs   (3 extra seeds x 2 archs x 2 datasets)
#   Exp 3 v2 stronger      2 hrs   (5 seeds x 2 directions, bigger data)
#   TOTAL                 ~4 hrs

set -e
mkdir -p results
START=$(date +%s)

echo "############################################################"
echo "#  SMOKE TEST (quick versions of all three)"
echo "############################################################"
python3 scripts/exp1b_arrival_time.py || echo "  [note] exp1b needs the original checkpoints — see below"
python3 scripts/exp5_more_seeds.py --quick
python3 scripts/exp3_v2_stronger.py --quick
echo ""
echo "  Smoke test done. Starting the real runs in 10s — Ctrl+C to stop."
sleep 10

echo ""
echo "############################################################"
echo "#  EXPERIMENT 1b — ARRIVAL TIME (the three-way check)"
echo "############################################################"
python3 scripts/exp1b_arrival_time.py

echo ""
echo "############################################################"
echo "#  EXPERIMENT 5 — MORE SEEDS (architecture robustness)"
echo "############################################################"
python3 scripts/exp5_more_seeds.py

echo ""
echo "############################################################"
echo "#  EXPERIMENT 3 v2 — CONTROLLED, LARGER, 5 SEEDS"
echo "#  Fixes the unstable point->distributed direction"
echo "############################################################"
python3 scripts/exp3_v2_stronger.py

END=$(date +%s)
echo ""
echo "############################################################"
echo "#  ALL FOLLOW-UP EXPERIMENTS COMPLETE"
echo "#  Total time: $(( (END-START)/60 )) minutes"
echo "############################################################"
ls -la results/result_exp1b_arrival_time.json \
       results/result_exp5_stronger_model.json \
       results/result_exp3_v2_stronger.json
echo ""
echo "Send me the log (results/run_v3_log.txt) and I will write the"
echo "final revised results section."