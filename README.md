# floodtransfer — cross-type flood transfer study

Clean folder. Nothing from the old project is carried over.

## The question

Train a flood model on ONE flood type, test it on a DIFFERENT flood type,
and measure how badly it breaks.

- Type A: **dike breach** — water enters at one edge (SWE-GNN data, already held)
- Type B: **rainfall** — water falls on every cell (Hurricane Harvey data, to download)

Then: does keeping flow direction as a **vector** help the model survive the jump?

## Why this is worth doing

The mSWE-GNN authors say in their own 2025 NHESS paper that their framework
works for dike-breach floods but they **did not evaluate it for other flood
types**. Every published transfer study moves across *regions of the same
flood type*. Nobody has moved across *types*. That is the gap.

## The claim the paper will make

> Flood models that score well on their own data fail to transfer across flood
> types, and keeping flow direction as a vector is what most helps them survive.

## Folder layout

```
floodtransfer/
  scripts/   things you run
  src/       reusable code
  data/      datasets go here (not in git)
  results/   json + figures come out here
```

## Order of work

| # | Step | GPU? | Status |
|---|------|------|--------|
| 1 | Inspect the Harvey data — confirm depth, vx, vy, rainfall | No | **DO THIS FIRST** |
| 2 | Put both datasets on the same grid and time step | No | after step 1 |
| 3 | Transfer test: train on A, test on B, and reverse | Light | after step 2 |
| 4 | Vector vs scalar direction comparison | Light | after step 3 |
| 5 | Optional second section: image-based depth transfer | Light | only if 1-4 are solid |

Step 1 decides everything. Do not build steps 2+ until it passes.

## Step 1 — run this

Download the FloodGNN-GRU Harvey dataset from Zenodo, unzip it into
`data/harvey/`, then:

```bash
cd floodtransfer
python3 scripts/inspect_dataset.py data/harvey
```

Read the **SUMMARY** block at the bottom. It prints GO or NO-GO.

Sanity-check the script against data you already know, first:

```bash
python3 scripts/inspect_dataset.py /path/to/swegnn/data/raw
```

It should report depth, vx, vy and terrain for the breach data. If it does,
you know the inspector is reading things correctly.

## What GO / NO-GO means

- **GO** — depth + both velocity components present. Full paper is possible.
- **PARTIAL (speed only)** — velocity shipped as magnitude without direction.
  The transfer test still runs; the direction half does not.
- **PARTIAL (no velocity)** — depth-only transfer test. Weaker but still novel.
- **NO-GO** — no depth field. Check the unzip, or raise `--max-files`.

Report the summary block back and step 2 gets written to match the real format.

## Missing libraries

The inspector says which one to install if it meets a format it cannot open:

```bash
pip install netCDF4     # for .nc files
pip install rasterio    # for .tif files
pip install h5py        # for .h5 files
```

## Metrics

`src/metrics.py` scores everything, the same way on both datasets:

- `csi()` — flood map accuracy
- `arrival_error()` — timing, in seconds
- `mass_error()` — did water appear from nowhere
- `transfer_gap()` — **the paper's number**: how much skill is kept away from home

Self-test it any time:

```bash
python3 src/metrics.py
```

All five checks should print OK.
