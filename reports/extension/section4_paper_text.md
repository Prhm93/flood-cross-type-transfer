# Section 4 report and draft paper text - third controlled mechanism

## Methods: third controlled mechanism (upstream inflow)

We extended the controlled synthetic experiment with a third water-entry mechanism.
Terrain seeds, solver (local inertial, Bates et al. 2010), grid (48 x 48 cells of 50 m),
solver step (2 s), run length (2,400 steps) and sample counts (60 training and 15 test
simulations) are identical to the point and distributed mechanisms. Water enters along
the whole upstream (higher) edge of the domain and follows a triangular hydrograph that
peaks at one third of the run. The hydrograph is scaled so that each simulation receives
the same total water as the point mechanism on the same terrain (largest relative
difference 3.2e-16). The only solver change is a time-varying multiplier on the source
term. With a constant multiplier the solver reproduces the original output exactly.

Inputs are scaled with the joint scaler of the two-mechanism experiment, rebuilt from the
point and distributed data only. The existing point and distributed models were reused;
re-scoring them reproduced all 90 stored values within 0.1%. The inflow model was trained
with identical settings (100 epochs, 5 seeds) and evaluated with a 30-step rollout.
CSI uses a threshold of 1% of the maximum true depth in each test sequence. Synthetic
frames are 120 s apart, so arrival error is reported in minutes.

## Results

Table X. Medians across five seeds.

| Trained on | Tested on | CSI | Volume ratio | Arrival error (min) |
|---|---|---|---|---|
| Point | Point (home) | 0.0962 | 2.25x | 16.8 |
| Point | Distributed | 0.9945 | 54.41x | 3.6 |
| Point | Inflow | 0.1581 | 2.00x | 16.8 |
| Distributed | Point | 0.0856 | 7.62x | 23.4 |
| Distributed | Distributed (home) | 0.5230 | 0.64x | 9.2 |
| Distributed | Inflow | 0.1463 | 6.82x | 23.1 |
| Inflow | Point | 0.0990 | 2.63x | 17.7 |
| Inflow | Distributed | 0.9997 | 37.96x | 3.6 |
| Inflow | Inflow (home) | 0.1514 | 2.35x | 17.5 |

The model trained on distributed floods fails on both boundary mechanisms in the same
way. It predicts 7.62 and 6.82 times the true water volume on point and inflow floods,
against 0.64 at home, and its arrival error rises from 9.2 to 23.4 and 23.1 minutes.
In 7 of 15 point events and 5 of 15 inflow events, it combines above-median CSI with
more than five times the true volume. The failure is therefore not specific to a single
point source.

Between the two boundary mechanisms, volume error changes little (2.00x to 2.63x,
against home values of 2.25x and 2.35x). Crossing between boundary and distributed
mechanisms raises volume error by roughly ten to twenty times.

CSI saturates on distributed floods. At the final step, every cell exceeds the 1%
threshold in all 15 distributed test events, and a trivial prediction that floods every
cell scores CSI 1.0000. Both boundary-trained models score 0.9945 and 0.9997 there while
predicting 54.41 and 37.96 times the true volume. CSI reports near-perfect skill for a
prediction no better than flooding everything.

The same trivial prediction bounds home skill on the boundary mechanisms. It scores
0.0799 on point and 0.1502 on inflow floods, against 0.0962 and 0.1514 for the trained
models. Under CSI, the boundary-trained models barely exceed a trivial guess at home. The
0.15 home-skill floor admits the inflow model although it matches the trivial guess.
We therefore report CSI next to the CSI of the trivial all-wet prediction.

No single metric reports the failure in both directions. When the distributed model is
tested on boundary floods, CSI is low and both volume and arrival flag the failure. When
boundary-trained models are tested on distributed floods, CSI flatters them, volume flags
the failure, and arrival does not (3.6 minutes, because every cell floods early). This
reproduces, under full experimental control and with three mechanisms, the pattern seen
between the dike-breach and Harvey datasets.

Within test sets, the Spearman correlation between event CSI and absolute log volume
error is mostly negative (-0.39 to -0.67), meaning higher-CSI events tend to have smaller
volume errors. It is near zero on the distributed test set (-0.054 and +0.096), where CSI
saturates. Pairwise ranking disagreement: [NEED NUMBER FROM ME, from the rerun].

## Revisions needed in paper v3

1. The controlled distributed-to-point claim ("loses 84% of skill") divides by a home CSI
   from a saturated test set (trained 0.523, trivial 1.0). Restate it with volume ratio
   and arrival error as the main evidence.
2. Check whether the Harvey test set is also saturated at the 1% threshold. The
   breach-to-Harvey CSI paradox may partly reflect saturation. [NOT YET RUN]
3. Check the wording of the exp6 Spearman result. A negative correlation with volume
   error means the metrics agree, not that higher CSI means worse physics.

## Limitations

- Synthetic data with closed walls and no carved channel: an inflow into a closed
  valley, not a full river with through-flow.
- Inflow home skill is marginal (2 of 5 seeds below the 0.15 floor) and matches the
  trivial all-wet CSI.
- The synthetic arm uses 100 epochs and a 30-step rollout, unlike the real-data runs.
- Seed spread is wide on distributed test floods (volume IQR up to 0.16x to 83.21x).
