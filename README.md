# Cross-type transfer in machine-learning flood surrogates

Parham Imanzadeh Charandabi, University of Salford. **Status: paper in preparation.**

## Question

Graph neural network flood surrogates are usually trained and tested on one flood type. Do they still work when the way water enters changes, and do standard flood-map metrics report the result honestly?

## Results

Equitable threat score (ETS) with bootstrap 95% intervals, 3 seeds per model. Source: `reports/exp23_stats.md`, built from `results/result_exp22_*.json`.

| Transfer | Threshold | Result |
|---|---|---|
| Rain-trained to localised floods (point and upstream inflow), 3 encodings, 6 cases | 10% of true maximum | ETS down 0.20 to 0.30, every interval below zero |
| Same 6 cases | 0.1 m | ETS down 0.16 to 0.21, every interval below zero |
| Dike-breach to Hurricane Harvey | 10% of true maximum | ETS 0.103 [0.082, 0.126] at home, -0.001 [-0.009, 0.006] on Harvey |
| Dike-breach to Hurricane Harvey | volume | median 64 times the true water volume (63.7) |
| All-wet prediction | 1% of true maximum | CSI 0.13 on dike-breach, 0.63 on Harvey |

- Rain-trained models predict far too little water on localised floods (frequency bias 0.01 to 0.68 at 10%).
- Whether localised-flood models transfer to rain depends on the input encoding. Under the log encoding, the point-trained model predicts 16.5 times the true volume on rain floods; under the linear and global encodings, the same model reaches ETS 0.25 to 0.28 on them.
- Raw CSI can reverse the apparent direction of a transfer result. Earlier dike-breach models (3 seeds, single-step training) scored CSI 0.16 to 0.19 at home but 0.44 to 0.56 on Harvey (`results/result_exp10_saturation_real.json`), because far more of Harvey is wet. CSI was therefore replaced with ETS, frequency bias and volume ratio.

## Data

- **Dike-breach:** SWE-GNN data, Zenodo DOI 10.5281/zenodo.7764418. 64 x 64 grid. Simulations 1-60 train, 61-80 validation, 501-519 test.
- **Hurricane Harvey (rain-driven):** FloodGNN-GRU data, Zenodo DOI 10.5281/zenodo.10787632. The first 19 of 228 test chunks are used.
- **Synthetic:** own generator using the local inertial shallow-water scheme of Bates et al. (2010) (`scripts/exp3_controlled_synthetic.py`). 48 x 48 grid, 50 m cells, Manning n 0.05, 60 training and 15 test runs per mechanism. Terrain seeds, grid, solver and inflow rate range (20 to 60 m3/s) are shared; only the source changes: a single cell, rain on every cell, or upstream edge inflow.

## Method

Vector FloodGNN (64 hidden units, 3 message-passing layers), trained for 100 epochs with a multi-step rollout loss (horizon up to 8 steps). Scaling is fitted on the training mechanism only. The source is given in three encodings: log, linear (rate per cell) and global (total inflow, no location).

## Metrics

ETS, frequency bias and volume ratio (summed over the rollout). Thresholds: 1, 5, 10 and 20% of the true maximum (applied to truth and prediction), plus 0.01, 0.05, 0.1 and 0.3 m where depths are in metres. Bootstrap: 2,000 resamples of seeds, then events within seeds. Every model and threshold is in one table.

## How to reproduce

Install PyTorch, then `pip install -r requirements.txt`. Put the Zenodo files in `data/breach/raw_datasets/` and `data/harvey/` (`train.npz`, `val.npz`, `test.npz`).

```bash
python -c "from src.experiment_lib import build_cache; build_cache()"   # real-data cache
python scripts/exp16a_synth_source_data.py                              # synthetic data
python scripts/exp22_rerun.py --data distributed --enc log --seed 0     # one model
python scripts/exp23_stats.py                                           # statistics table
```

`--data`: `point`, `distributed`, `inflow` or `breach`. `--enc`: `log`, `linear` or `global`. Checkpoints are in `results/exp15_exp22_*.pt`; a script stops if its result file already exists.

## Limitations

- Three seeds per model. Harvey is one storm, cut into adjacent chunks with open edges.
- Harvey depths are stored scaled by 0.1. The dike-breach model was tested without converting them to metres, so part of the Harvey failure may come from this mismatch.
- Real data were tested under the log encoding only, and the synthetic solver is not yet validated against a benchmark.
- ETS also depends on base rate: under the linear and global encodings, localised-flood models score higher ETS on rain floods than at home.
- The source channel supplies the inflow, so a home volume ratio near 1 is partly given.

## Ongoing work

Rerunning the Harvey test in metres, and a source-blind model designed to transfer across flood types.
