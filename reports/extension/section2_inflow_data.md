# Section 2 - Third controlled mechanism: upstream inflow (data build and audit)

Date: 25 Sept 2026. Option A chosen after the Section 1 audit. CPU only, no training.
Script: scripts/exp9_stage1_inflow_data.py. Raw log: reports/exp9_stage1.txt. Run time 105 s.

## Why this option

Section 1 showed FloodCastBench Australia and UK are rain-driven and have no velocity data,
so they cannot enter the pipeline unchanged. Option A adds a third flood mechanism inside
the exp3 controlled setup, where terrain, solver, grid, resolution and sample counts are
already matched and only the water source changes.

## The new mechanism

Upstream inflow. Water enters along the whole right-hand edge column (column dim-2),
which is the higher side of the terrain tilt, so it runs downhill across the domain.
The inflow follows a triangular hydrograph: zero at the start, peak at one third of the
run, zero at the end. The hydrograph mean is scaled to exactly 1, so each simulation
receives the same total water as the point mechanism on the same terrain seed.
The rain channel (channel 3) is zero, as for the point mechanism and the breach data.

The terrain seeds, the inflow rate draw (20-60 m3/s, seeded by terrain seed + 500),
the solver, the grid (48 x 48, 50 m), the solver step (2 s) and the run length
(2,400 steps) are the same as exp3 v2.

## The one solver change

simulate_tv() is a line-by-line copy of the exp3 simulate() with one addition: an optional
factor[step] that scales the source over time. With no factor it is the original solver.

## Checks

| Check | What it tests | Result |
|---|---|---|
| A | Saved synth2_point_s0 and synth2_distributed_s0 re-scored on regenerated test data reproduce the stored CSI and volume ratio (within 0.1% relative) | PASS |
| B | simulate_tv() with no factor, and with a factor of all ones, gives output identical to simulate() | PASS |
| C | Inflow puts in the same total water as point (worst relative difference 3.23e-16) | PASS |
| D | No terrain seed appears in both train and test, for any mechanism | PASS |
| E | Test terrains are the same seeds across all three mechanisms | PASS |

Check A also confirms that exp3 v2 data and the exp3 v2 scaler are reproduced exactly,
even though the exp3 files were committed after the exp3 v2 run. The exact values are in
the raw log.

## Physical summary (metres, before scaling, training sets)

| Mechanism | Median max depth | Median final wet fraction |
|---|---|---|
| Point | 0.81 m | 0.085 |
| Distributed | 0.26 m | 0.169 |
| Inflow | 0.71 m | 0.121 |

The inflow floods sit between the other two mechanisms on both measures.

## Scaling

The exp3 v2 joint scaler was rebuilt from the same point and distributed data exp3 v2 used,
and saved to results/scaler_synth_exp3v2.json. The inflow data was NOT used in the fit.

Inflow after scaling:

| Channel | Min | Max | Share outside [0, 1] |
|---|---|---|---|
| depth | 0.000 | 0.904 | 0.000% |
| vx | -0.140 | 0.472 | below 0.0005% (a few values below 0) |
| vy | 0.416 | 0.801 | 0.000% |
| rain | 0.000 | 0.000 | 0.000% |

## Files produced

- results/scaler_synth_exp3v2.json (committed)
- data/synth3_cache.pkl, 335 MB, point + distributed + inflow, train and test, scaled
  (not committed, data/ is git-ignored; regenerable in about 2 minutes)

## Settings to declare in the paper

- The synthetic arm uses exp3 v2's settings (100 epochs, 30-step rollout, 5 seeds),
  not the real-data settings (60 epochs, 40-step rollout).
- The synthetic arm uses its own joint scaler, not the breach + Harvey scaler.

## Limitations

- Closed walls: no water leaves the domain, as for the other two mechanisms. This keeps
  "only the source differs", but it means inflow into a closed valley, not a full river
  with through-flow.
- No carved river channel, for the same reason.
- A reviewer may see inflow as close to the point source, since both enter at a boundary.
  Both outcomes of the transfer test are informative: point and inflow transferring to
  each other would point to "boundary source versus distributed source" as the key split;
  all three failing would show mechanism matters more finely.

## Next section

Section 3 - train the inflow mechanism (5 seeds, exp3 v2 settings) and score every
synthetic model on every synthetic test set.
