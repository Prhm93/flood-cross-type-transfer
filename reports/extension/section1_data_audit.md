# Section 1 - Data audit for the flood-type extension

Date: 25 Sept 2026. Read-only checks. No GPU. Nothing in data/ or results/ was changed.

## Goal

Find out what data is already on the container before adding new flood types to the
cross-type transfer study, and check whether the handoff plan (fluvial FloodCastBench,
same pipeline, same scaler) can run as written.

## What was run

| Check | Script or command | Raw log |
|---|---|---|
| 1a Disk scan of ~/projects and common data folders | scripts/check_new_data.py | not committed (lists other projects) |
| 1b Zip contents, frame timing, grids, water budget, scaler search | scripts/check_step1b.py | reports/check_step1b.txt |
| 1c Loader and normaliser source, saved scaler copies | shell block | reports/check_step1c.txt |
| 1d Saved scaler, model variants, georeferencing, peak depth and rain | shell block | reports/check_step1d.txt |
| 1e exp3 synthetic generator source | shell block | reports/check_step1e.txt |

## Findings

### 1. FloodCastBench on disk

The archive (FloodCastBench.zip, 20.1 GB, Zenodo 10.5281/zenodo.14017092) is in
~/projects/hydraulic-attention/data/. A full duplicate sits in hydraulic-attention-BACKUP-20260830.

| Set | Frames | Size | State |
|---|---|---|---|
| High-fidelity 30 m Australia | 2,881 | 12.4 GB | extracted |
| High-fidelity 60 m Australia | 2,881 | 3.1 GB | extracted |
| High-fidelity 30 m UK | 865 | 154.4 MB | extracted |
| High-fidelity 60 m UK | 865 | 38.5 MB | zip only |
| Low-fidelity 480 m Pakistan | 4,033 | 5.4 GB | zip only |
| Low-fidelity 480 m Mozambique | 1,729 | 137.7 MB | zip only |

Correction to check 1a: its summary marked Pakistan and Mozambique as FOUND. That was
wrong. The script matched country names on rainfall, terrain and initial-condition files.
No Pakistan or Mozambique depth frames are extracted.

Frame names are seconds from the start. The step is 300 s. Australia runs 10.00 days
(241 hourly frames). UK runs 3.00 days (73 hourly frames).

### 2. FloodCastBench has depth only

Every depth frame is a single band. The archive holds no velocity files. Both model
variants need velocity: 'vector' uses vx and vy, and 'scalar' uses speed computed from
them (src/model.py lines 45-46). So neither variant can be trained on FloodCastBench
without a declared change to the method.

### 3. The high-fidelity events are rain-driven, not river floods

Water budget over hourly 30 m frames. Rain units assumed mm/h. Outflow is not counted.

| Event | Start volume | Peak volume | Peak hour | Rain before peak | Gain / rain |
|---|---|---|---|---|---|
| Australia | 2.27e6 m3 | 3.781e8 m3 | 240 (end) | 3.762e8 m3 | 1.00 |
| UK | 1.471e6 m3 | 5.901e6 m3 | 40 | 4.546e6 m3 | 0.97 |

Rain explains almost all the water gained. Very little water enters or leaves through
the edges. Australia and UK are rainfall floods, the same mechanism as Harvey. The near
exact 1.00 also supports mm/h as the rain unit. Both events start wet: Australia max
depth 1.060 m (4.0% of cells deeper than 1 cm), UK 2.481 m (15.9%).

The FloodCast paper (arXiv 2403.12226) describes an upstream inflow boundary for the
Pakistan event (Indus River), so Pakistan is the only candidate with a real river
inflow. It is the 480 m low-fidelity set and is still in the zip.

### 4. Peak values against the joint scaler

| Event | Max depth | 99th pct depth (wet cells) | Peak rain |
|---|---|---|---|
| Australia (hour 240) | 18.248 m | 3.979 m | 26.33 mm/h |
| UK (hour 40) | 5.457 m | 3.289 m | 9.83 mm/h |

The joint scaler allows depth up to 4.076 and rain up to 1.041 (Harvey's own scaled
units). Depth mostly fits. FloodCastBench rain would sit 10 to 25 times beyond anything
the models saw in training. There is no honest unit conversion without knowing how
FloodGNN-GRU scaled Harvey rainfall.

### 5. Grid alignment

Australia: all layers are 1073 x 1073. UK: terrain, land cover and rain are 180 x 285,
but depth frames are 170 x 275. There is one world file (.tfw) per event, at 30 m, in
UTM (Australia 56S, UK 30N, Pakistan 42N, Mozambique 36S). How the UK depth grid sits
inside the terrain grid is not yet confirmed.

### 6. The joint scaler

The handoff says data/scaler.json. That file does not exist. The saved scaler is
results/scaler.json, written by scripts/train.py line 102. It matches the copy stored
inside data/cache_normalised.pkl exactly:

- dynamic_min: [0.0, -1.0, -1.0673, 0.0] (depth, vx, vy, rain)
- dynamic_max: [4.0757, 1.1943, 1.5681, 1.0410]
- static_min: -0.1643, static_max: 1.0 (elevation)

Elevation is rescaled to 0-1 per sample before the joint scaler (src/unified_loader.py
lines 203-208 for breach). Harvey depth and rain are on Harvey's internal scale, while
breach depth is in metres. The two real datasets were therefore already on mixed units
before this extension.

The cache holds breach 60/20/19 and Harvey 1063/228/228 train/val/test samples.

### 7. The controlled synthetic generator (exp3)

- scripts/exp3_controlled_synthetic.py holds the local inertial solver (Bates et al. 2010).
- All four outer edges are closed walls. No water leaves the domain.
- Every simulation uses fixed seeds, so the data can be regenerated exactly.
- exp3 v2 settings: 48 x 48 grid, 50 m cells, 2 s solver step, 2,400 steps,
  60 train and 15 test simulations per mechanism, 100 epochs, 5 seeds, 30-step rollout.
- exp3 v2 uses its own joint scaler over point and distributed data, built in memory
  and never saved.
- Checkpoints synth2_point_s0-s4 and synth2_distributed_s0-s4 exist in results/.
  From their timestamps, each training run took about 5-6 minutes.
- The exp3 files were committed (10 and 11 Sept) after the exp3 v2 run finished
  (01:39 on 10 Sept). Reproduction must be confirmed by re-scoring a saved checkpoint.

### 8. Environment

- floodtransfer venv lacks rasterio, scipy, pandas, xarray and torch_geometric.
  PIL reads the float TIFFs.
- 103.5 GB free of 495.0 GB.
- ~/projects/hat-code-backup.tar.gz is truncated (EOFError on read). The verified
  backup is the git bundle ~/projects/hydraulic-attention-full-backup.bundle.

## Handoff plan versus what the audit found

| Handoff assumption | Finding |
|---|---|
| FloodCastBench is a fluvial (river) type | Australia and UK are rain-driven (gain/rain 1.00 and 0.97) |
| Same pipeline, same model | No velocity in FloodCastBench, both variants need it |
| Same scaler, data/scaler.json | Scaler is results/scaler.json; FloodCastBench rain is 10-25x beyond its range |
| About 19 test events | 2 high-fidelity events (plus 2 low-fidelity events in the zip) |

## Options

- A: add a third controlled mechanism (upstream inflow) to the exp3 generator. Full
  depth and velocity, same solver and grid, 5 seeds, all handoff rules kept.
  Limitation: synthetic, closed walls, so an inflow rather than a full river.
- B: use FloodCastBench Australia and UK as a real-data rainfall control, with
  declared changes (no velocity, rain scaling, 2 events).
- C: extract Pakistan 480 m as the river case (depth only, one event, heavy regridding).

## Decision

[NEED FROM ME] Choose A, A+B, B or C.

## Next section

Section 2 - build and audit the data for the chosen option before any training.
