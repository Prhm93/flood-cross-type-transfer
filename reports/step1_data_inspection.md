# Step 1 Report — Data Inspection and Research Setup

Written: 1 September 2026
Status: **STEP 1 COMPLETE — GO confirmed for both datasets**


## 1. What this project is about

We are building a study that asks one question:

> **If you train a flood prediction model on one type of flood, does it
> still work on a completely different type of flood?**

Nobody has tested this. Every published flood model is trained and tested
on the same flood type. We will be the first to test across types.

The two flood types are:

| Property | Dataset A: Dike breach | Dataset B: Hurricane rainfall |
|----------|----------------------|------------------------------|
| What happens | A wall breaks, water pours in from one edge | Rain falls everywhere at once |
| Where | Synthetic Dutch polders | Houston, Texas (Hurricane Harvey 2017) |
| Water source | One breach point, constant 50 m³/s | Spatially distributed rainfall |
| Why they differ | Water enters at a boundary | Water enters from the sky onto every cell |

The physical difference is real: a breach flood spreads outward from a single
point like a puddle growing from a tap. A rainfall flood rises everywhere at
once like a bathtub filling. A model that learned one pattern has no guarantee
it understands the other.


## 2. The research gap — confirmed by the field's own words

The SWE-GNN group (Bentivoglio, Isufi, Jonkman, Taormina), who created
Dataset A, wrote in their own 2025 NHESS paper:

> "While the current model framework can work for dike-breach floods, we
> did not evaluate it for other types of floods."

They speculate river and coastal floods might work, and note that rainfall
floods need precipitation as a further input — but they never tested any of it.

Every other transfer study we found moves a model across **regions of the
same flood type**: catchment to catchment (Song & Guan, Water Research 2026),
or terrain to terrain within breach data. Nobody crosses the type boundary.

That gap is our paper.


## 3. The paper claim (one sentence)

Flood surrogate models that score well on their own flood type fail to
transfer across flood types (breach vs rainfall), and keeping flow
direction as a vector — rather than throwing it away — is what most helps
a model survive the jump.


## 4. Target journals

| Journal | Why it fits | Realistic? |
|---------|------------|------------|
| Environmental Modelling & Software (Q1) | Published a pure-generalisation paper in 2026; methods focus | **Best fit** |
| Journal of Hydrology (Q1) | Broad scope, accepts ML flood work | Good fit |
| Water Resources Research (Q1) | Wants a water-science insight, not just a model | Possible, harder |
| Geoscientific Model Development (Q1) | Wants model assessment methods; accepts evaluation papers | Good fit |


## 5. The two datasets — full details

### Dataset A: SWE-GNN dike breach

| Property | Value |
|----------|-------|
| Citation | Bentivoglio, Isufi, Jonkman, Taormina (2023). Deep learning methods for flood mapping: a review of existing applications and future research directions. HESS 27:4227-4246 |
| Data DOI | Zenodo 10.5281/zenodo.7764418 |
| Licence | CC-BY 4.0 |
| Solver | Delft3D-FM (full shallow-water equations, implicit scheme) |
| Grid | Regular, 64 × 64 cells = 4,096 nodes |
| Cell size | 100 metres (domain 6.4 km × 6.4 km) |
| Time step | 1 hour |
| Duration | 48 hours per simulation |
| Simulations | 100 total: 60 train (ids 1-60), 20 val (ids 61-80), 20 test (ids 501-519, sim 520 absent from archive) |
| Initial condition | Completely dry (water rises from nothing) |
| Boundary | Constant 50 m³/s inflow through a breach |
| Variables | Water depth (WD), velocity x (VX), velocity y (VY), terrain elevation (DEM) |
| Rainfall | None — this is a breach flood, not a rainfall flood |
| File format | Plain text matrices: WD/VX/VY are (T rows × N columns), DEM is (N × 3: x, y, z) |
| Total size | 111 MB (zipped) |
| Status | **Already held and loader verified** |

Verified properties (from prior inspection of simulation 1):
- Depth shape: (97, 64, 64), range 0 to 2.88 m
- Elevation range: -3.44 to +3.04 m
- Mean |dh| per step: 0.0022 m
- Wet fraction: 0% at t=0, rising to 30.5% at end
- 5.5% of cell-steps change by more than 1 cm (the flood actively moves)
- 99.1% of wet cells form one connected pool (confirms correct grid orientation)


### Dataset B: FloodGNN-GRU Hurricane Harvey

| Property | Value |
|----------|-------|
| Citation | Kazadi, Doss-Gollin, Sebastian, Silva (2024). FloodGNN-GRU: A Spatio-Temporal Graph Neural Network for Flood Prediction. Environmental Data Science. doi:10.1017/eds.2024.19 |
| Data DOI | Zenodo 10.5281/zenodo.10787632 |
| Licence | CC-BY 4.0 |
| Code | github.com/kanz76/FloodGNN-GRU |
| Solver | LISFLOOD-FP version 8 (simplified shallow-water scheme) |
| Original grid | 1961 × 1636 cells at 30 m (Harris County, Houston TX) |
| Data format | Pre-cut into graph chunks (variable size, 581-1568 nodes each) |
| Time steps | 132 per chunk, 1-hour resolution |
| Files | train.npz (4.3 GB), val.npz (951 MB), test.npz (905 MB), wdfp_scaler.pckl (221 B) |
| Samples | 228 in test set (train and val counts TBD after loading) |
| Total size | 6.1 GB |
| Status | **Downloaded and inspected — GO confirmed** |

Feature mapping (confirmed from source code, `dataset.py` line-by-line):

| Column | Variable | Description | Value range (scaled) |
|--------|----------|-------------|---------------------|
| data[..., 0] | wdfp | Water depth | 0.00 to 0.86 |
| data[..., 1] | vx (t) | Velocity x, current time | -1.00 to 1.00 |
| data[..., 2] | vy (t) | Velocity y, current time | -1.00 to 1.00 |
| data[..., 3] | vx (t-1) | Velocity x, previous time | -1.00 to 1.00 |
| data[..., 4] | vy (t-1) | Velocity y, previous time | -1.00 to 1.00 |
| data[..., 5] | |v| | Velocity magnitude (copy 1) | 0.00 to 0.08 |
| data[..., 6] | |v| | Velocity magnitude (copy 2) | 0.00 to 0.08 |
| data[..., 7] | rain | Rainfall | 0.00 to 0.97 |

Static features (3 columns per node):

| Column | Likely meaning | Range |
|--------|---------------|-------|
| static[:, 0] | Elevation (normalised) | 0.04 to 0.14 |
| static[:, 1] | Slope | 0.00 to 0.007 |
| static[:, 2] | Manning's roughness | 0.01 to 0.37 |

Graph structure: each chunk has an edge list (`s_edges`, shape (E, 2)) defining
which nodes are connected. Chunks vary in size (581 to 1568 nodes, 2202 to 6110
edges), because the Houston grid was cut into spatial patches.

The scaler pickle (`wdfp_scaler.pckl`) uses a custom `Scaler` class and could
not be loaded without their code. This means the depth and velocity values are
on an internal normalised scale, not physical metres. Reversing the scaling is
possible but not essential for the transfer test — what matters is whether a
model trained on one distribution copes with the other.


## 6. Alignment between the two datasets

| Property | Breach (A) | Harvey (B) | Compatible? |
|----------|-----------|------------|-------------|
| Water depth | Yes | Yes | YES |
| Velocity x | Yes | Yes | YES |
| Velocity y | Yes | Yes | YES |
| Rainfall | No (breach has no rain) | Yes | Asymmetric — this IS the test |
| Elevation | Yes | Yes | YES |
| Time step | 1 hour | 1 hour | YES |
| Format | Regular grid (64×64) | Graph chunks (variable size) | Convertible |

The format difference (grid vs graph) is handled by the unified loader
(`src/unified_loader.py`). It converts the breach grid into a graph
(each cell = node, 4-connected edges) so both datasets output the same
structure: `(nodes_dynamic, nodes_static, edge_index)`.

The one real physical difference — breach has no rainfall — is not a bug.
It IS the research question. A model trained on breach data has never seen
rain. A model trained on Harvey has never seen a dry start with a point source.
How much does each one break? That is the finding.


## 7. What we built so far

| File | Purpose | Status |
|------|---------|--------|
| `scripts/inspect_dataset.py` | Auto-detect format and report GO/NO-GO | Tested, working |
| `src/metrics.py` | CSI, arrival time, mass error, transfer gap | Tested, all 5 self-tests pass |
| `src/unified_loader.py` | Load both datasets into the same graph format | Tested on synthetic data |
| `reports/step1_data_inspection.md` | This document | Complete |


## 8. What comes next

| Step | Task | GPU needed? | Depends on |
|------|------|-------------|------------|
| 2 | Load both real datasets through unified_loader and verify shapes/ranges match | No | Breach data on server |
| 3 | Build a simple GNN model that trains on the unified format | Light | Step 2 |
| 4 | Train on breach, test on Harvey (and reverse) — measure the transfer gap | Light | Step 3 |
| 5 | Compare vector velocity vs scalar speed — does direction help transfer? | Light | Step 4 |
| 6 | Write the paper | No | Step 5 |

Step 2 requires the SWE-GNN breach data to be on the GPU server. It is
available at Zenodo 10.5281/zenodo.7764418 (111 MB zip).


## 9. Risk register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Harvey data is normalised on an internal scale; breach data is in physical metres | Models see different value ranges | Normalise both to 0-1 per feature before feeding to the model |
| Harvey graphs vary in size (581-1568 nodes); breach graphs are always 4096 nodes | Batch construction gets complicated | PyTorch Geometric handles variable-size graphs natively |
| The transfer gap might be small (both models cope fine across types) | The paper's finding disappears | Still publishable as a positive result — "flood models transfer better than expected" is also novel and useful |
| Scaler pickle cannot be reversed without their custom class | Cannot convert Harvey back to physical units | Not essential — the transfer test works on normalised values; note this as a limitation |
| GPU time is limited (~1 month from ~28 Aug 2026) | Cannot run large experiments | The GNN is small, breach data is tiny, Harvey is pre-cut — training should be minutes per run, not hours |