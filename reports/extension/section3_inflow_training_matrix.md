# Section 3 - Inflow training and the 3 x 3 synthetic transfer matrix

Date: 25 Sept 2026. Script: scripts/exp9_stage2_train_inflow.py.
Raw log: reports/exp9_stage2.txt. Results: results/result_exp9_inflow_matrix.json.
Run time 1,712 s (about 29 minutes) on the A100 MIG partition.

## What was run

- Trained the inflow mechanism for seeds 0-4 with the exact exp3 v2 call
  (vector model, hidden 64, 3 layers, 100 epochs, 60 training simulations).
- Reused the saved synth2_point_s0-s4 and synth2_distributed_s0-s4 checkpoints.
  They were not retrained.
- Scored all 15 models on all three synthetic test sets
  (15 test simulations each, 30-step rollout, CSI at 1% relative threshold, volume ratio).
- A smoke test (2 epochs, seed 0) ran first and wrote only to results/_smoke_exp9/.

## Regression check

All 40 stored exp3 v2 values (10 models x home and away x CSI and volume ratio)
reproduced within 0.1% relative. The old point and distributed results are unchanged.

## Results - median [IQR] across 5 seeds

| Trained on | Tested on | CSI | Volume ratio | Retained CSI |
|---|---|---|---|---|
| Point | Point (home) | 0.0962 [0.0961, 0.0981] | 2.25x [1.94, 2.69] | - |
| Point | Distributed | 0.9945 [0.2244, 1.0000] | 54.41x [0.16, 83.21] | 964.6% |
| Point | Inflow | 0.1581 [0.1482, 0.1649] | 2.00x [1.73, 2.39] | 161.2% |
| Distributed | Point | 0.0856 [0.0853, 0.0863] | 7.62x [3.07, 14.01] | 16.3% [13.7, 16.5] |
| Distributed | Distributed (home) | 0.5230 [0.5176, 0.6293] | 0.64x [0.50, 0.66] | - |
| Distributed | Inflow | 0.1463 [0.1454, 0.1489] | 6.82x [2.75, 12.53] | 23.1% [17.7, 28.0] |
| Inflow | Point | 0.0990 [0.0921, 0.1250] | 2.63x [2.27, 5.07] | 62.1% [60.9, 79.3] |
| Inflow | Distributed | 0.9997 [0.3007, 1.0000] | 37.96x [8.51, 73.17] | 627.1% [334.3, 634.3] |
| Inflow | Inflow (home) | 0.1514 [0.1485, 0.1577] | 2.35x [2.01, 4.52] | - |

Home-skill floor (home CSI below 0.15): point 5/5 seeds, distributed 0/5, inflow 2/5.
Per-seed values are in the raw log and the JSON.

## Findings

1. The rain-like (distributed) model fails on the new inflow floods in the same way it
   fails on point floods: 23.1% of CSI skill retained (16.3% on point), with 6.82x the true
   water volume (7.62x on point). The distributed-to-boundary failure is not specific to
   the point source.
2. Between the two boundary mechanisms (point and inflow), volume error hardly changes:
   2.00x to 2.63x against home values of 2.25x and 2.35x. Crossing between boundary and
   distributed mechanisms raises volume error by roughly 10-20 times.
3. The CSI trap appears again: the inflow-trained model scores CSI 0.9997 on distributed
   floods while predicting 37.96x the true water volume. The same pattern appears in
   breach to Harvey and point to distributed.

## Caveats

- Inflow home skill is marginal (median 0.1514, just above the floor; 2/5 seeds below).
  Retained-CSI percentages for the inflow row are fragile. Volume ratio carries that row.
- Point-trained results stay below the home-skill floor in all seeds and are reported
  but not interpreted, as in the paper.
- Seed spread on the distributed test set is wide (IQR up to 0.16x-83.21x). Medians must
  always be shown with their IQR.
- CSI values near 1.0 on the distributed test set may be saturation (true water in every
  cell at the 1% relative threshold). To be checked in Section 4 before any claim.
- Both boundary-trained models score low home CSI while the distributed model scores
  high. Part of this may be the metric itself (small, edge-fed wet areas are harder to
  match), not only the model.
- Settings follow exp3 v2 (100 epochs, 30-step rollout), not the real-data settings.

## Files

- results/result_exp9_inflow_matrix.json (committed)
- results/synth3_inflow_s0-s4.pt (on the container only; *.pt is git-ignored)

## Next section

Section 4 - CSI saturation check on the distributed test set, arrival time, per-event
scores and pairwise disagreement for the new transfer pairs. No training.
