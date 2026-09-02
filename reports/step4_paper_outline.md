# Paper Outline

**Working title:**
Cross-Type Transfer Failure in Flood Inundation Surrogates:
Why Models Trained on One Flood Mechanism Fail on Another

**Target journals (in order of fit):**
1. Environmental Modelling & Software (Q1) — best fit, methods focus, published
   a pure-generalisation paper in 2026
2. Geoscientific Model Development (Q1) — explicitly wants model assessment
   methods; accepted evaluation papers
3. Journal of Hydrology (Q1) — broad scope, accepts ML flood work

**Length:** ~6,000 words + figures and tables


---


## Abstract (~250 words)

Paragraph 1 — the problem:
Machine-learning flood surrogates are routinely trained and evaluated on a
single flood scenario or dataset. Cross-regional transfer has begun to be
studied, but cross-TYPE transfer — training on one flood mechanism and
testing on a physically different one — has not been assessed.

Paragraph 2 — what we did:
We trained a graph neural network on dike-breach simulations (SWE-GNN data,
Delft3D-FM) and evaluated it on rainfall-driven Hurricane Harvey simulations
(LISFLOOD-FP), and vice versa. Both datasets carry depth and directional
velocity. We compared vector (direction-preserving) and scalar (speed-only)
input representations across 12 experimental runs (2 training sources ×
2 variants × 3 random seeds).

Paragraph 3 — what we found:
Three findings. (1) Harvey-trained models retain only ~30% of their flood
mapping skill on breach data, confirming severe cross-type transfer failure.
(2) In the reverse direction, CSI appears to improve while mass error
reaches 30–130×, revealing that binary extent metrics can actively mask
transfer failure on data with different value distributions. (3) Preserving
flow direction as a vector did not improve transfer; the gap is dominated
by the difference in flood driver, not by the flow representation.

Paragraph 4 — so what:
These results demonstrate that single-dataset evaluation is insufficient
for flood surrogates, and that cross-type transfer assessment requires
multiple complementary metrics including mass balance.


---


## 1. Introduction (~800 words)

### 1.1 The promise and the gap

- ML flood surrogates achieve impressive speed-ups over hydrodynamic solvers
- Standard practice: train and test on the same dataset or same flood type
- Cross-REGIONAL transfer is beginning to be studied:
  - Song & Guan (Water Research 2026): catchment-to-catchment, same flood type
  - Spatial generalisation paper (Env. Modelling & Software 2026): sub-watershed
    to unseen sub-watershed, same flood type
  - Transfer learning for coastal-estuarine systems (Cambridge 2026)
- But cross-TYPE transfer — breach to rainfall, or fluvial to pluvial — is
  UNTESTED. The mSWE-GNN group (Bentivoglio et al., NHESS 2025) states
  explicitly: "we did not evaluate for other flood types"

### 1.2 Why cross-type matters

- Different flood types have fundamentally different physics:
  - Breach: water enters at one point, spreads radially
  - Rainfall: water arrives everywhere simultaneously from above
  - Coastal: driven by tides and storm surge
- A model that learns "water comes from one edge" may encode that as a
  structural assumption, not as a learnable pattern
- Operational need: emergency responders need models that work for WHATEVER
  flood happens, not just the type in the training data

### 1.3 The direction question

- Several papers argue flow direction should be preserved as a vector, not
  collapsed to scalar speed (FloodGNN-GRU: Kazadi et al. 2024; mSWE-GNN:
  Bentivoglio et al. 2023 explicitly discarded direction)
- Hypothesis: directional information might help transfer because it
  encodes physically meaningful routing
- We test this directly

### 1.4 Contributions

State the three findings from the abstract, plus:
- First cross-type transfer evaluation in flood surrogates
- A documented failure mode of CSI under cross-scale evaluation
- Public code and reproducible experimental protocol


---


## 2. Data (~800 words)

### 2.1 Dataset A — SWE-GNN dike breach

- Full table of properties (from Step 1 report)
- Citation: Bentivoglio et al. (2023), Zenodo 10.5281/zenodo.7764418
- Delft3D-FM, 64×64 grid, 100m cells, 1h steps, 48h duration
- 60 train / 20 val / 19 test simulations
- Variables: water depth, velocity x, velocity y, terrain elevation
- Key property: constant 50 m³/s breach inflow, dry initial condition
- NO rainfall

### 2.2 Dataset B — FloodGNN-GRU Hurricane Harvey

- Full table of properties (from Step 1 report)
- Citation: Kazadi et al. (2024), Zenodo 10.5281/zenodo.10787632
- LISFLOOD-FP v8, graph chunks from 30m Houston grid, 1h steps
- 1063 train / 228 val / 228 test samples
- Variables: water depth, velocity x, velocity y, rainfall, terrain
- Key property: spatially distributed rainfall driving
- Rainfall is the FOURTH dynamic channel — breach data has this as zero

### 2.3 Alignment and differences

- Table comparing the two datasets side by side
- Same: depth + directional velocity, hourly time steps, CC-BY licence
- Different: resolution (100m vs 30m), format (grid vs graph), flood
  driver (breach vs rainfall), scale (physical metres vs internal
  normalised)
- How alignment was handled: unified graph loader, joint normalisation
- What was NOT aligned (deliberately): the rainfall channel — breach is
  zero, Harvey is active. This asymmetry IS the research question


---


## 3. Methods (~1,200 words)

### 3.1 Unified representation

- Both datasets converted to the same graph format: nodes carry 4 dynamic
  channels (depth, vx, vy, rain) + 1 static (elevation), with edge
  connectivity
- Grid-to-graph conversion for breach data: each cell is a node, 4-connected
- Joint normalisation: min/max fitted on the union of both training sets,
  applied to all splits

### 3.2 Model architecture

- Simple 3-layer message-passing GNN, ~42K parameters
- Input: current state (4 dynamic + 1 static = 5 channels for vector variant,
  4 for scalar)
- Output: predicted change in depth, vx, vy
- Two variants:
  - Vector: vx and vy as separate input channels
  - Scalar: speed = sqrt(vx² + vy²) replaces both
  - Otherwise identical architecture
- Deliberately small — the model is the measuring instrument, not the claim

### 3.3 Training protocol

- Teacher-forced, one-step prediction, MSE loss
- 60 epochs, Adam optimiser, learning rate 1e-3 with ReduceLROnPlateau
- 3 random seeds per configuration for error bars
- Each model trained on ONE dataset only

### 3.4 Evaluation protocol

- 40-step autoregressive rollout from first frame
- Model feeds its own predictions back in (no teacher forcing)
- Depth clamped to ≥ 0 (no negative water)
- Rainfall taken from true data at each step (external driver, not predicted)
- Each model evaluated on BOTH test sets (home and away)

### 3.5 Metrics

- **CSI** with adaptive threshold (1% of maximum true depth per sample) —
  measures spatial extent accuracy. Threshold adapts so it means the same
  thing on both datasets regardless of value scale
- **Mass error** — relative total water difference. No threshold needed.
  Catches the failure mode that CSI misses
- **Arrival time MAE** — timing accuracy in seconds
- **Transfer gap** — ratio of away CSI to home CSI, as percentage retained

### 3.6 Why adaptive threshold is necessary

- Breach data: max depth ~4m, fixed 0.05m threshold reasonable
- Harvey data: max depth ~0.21 after joint normalisation, 0.05 threshold
  would classify most wet cells as dry
- The 1% adaptive threshold ensures "wet" means "non-trivially flooded"
  on both scales
- Mass error serves as a threshold-free cross-check


---


## 4. Results (~1,000 words)

### 4.1 Home performance

- Table: all 12 runs, home CSI, home mass error
- Breach models converge to CSI ~0.17–0.22 at home (moderate — 40-step
  rollout from dry start is hard)
- Harvey models converge to CSI ~0.25–0.54 at home (higher, reflecting
  the different data characteristics)

### 4.2 Cross-type transfer: Harvey → Breach

**Figure 1: bar chart — home vs away CSI for Harvey-trained models**

- Vector: 0.42 → 0.12 (29% retained)
- Scalar: 0.49 → 0.14 (29% retained)
- Clear, consistent ~70% skill loss across all seeds
- Mass error stays reasonable (±1) — the model predicts a sensible amount
  of water but in the wrong places
- This is genuine transfer failure

### 4.3 Cross-type transfer: Breach → Harvey

**Figure 2: bar chart — CSI appears to improve, mass error explodes**

- CSI goes from 0.18 to 0.48 (apparent improvement)
- Mass error goes from ~0.5 to 30–130× (catastrophic)
- The model paints thin false floods across the entire Harvey domain
- Because Harvey's true depths are small after normalisation, the
  adaptive threshold is tiny, and any predicted positive depth counts
  as a "hit"
- This is a metric failure hiding a prediction failure

### 4.4 Does direction help transfer?

**Table: vector vs scalar, mean across 3 seeds**

| Direction | Variant | Harvey→Breach retained | Breach→Harvey mass err |
|-----------|---------|------------------------|------------------------|
| Harvey→Breach | vector | 29.8% | — |
| Harvey→Breach | scalar | 29.1% | — |

- No measurable difference (within seed noise)
- The dominant gap is the flood driver, not the flow representation

### 4.5 What CSI misses

**Figure 3: scatter plot — CSI vs mass error, all 12 runs, coloured by
home/away**

- Home runs cluster near the origin: CSI 0.2–0.5, mass error ±10
- Breach→Harvey away runs sit at high CSI, extreme mass error
- This plot IS Finding 2 — the visual proof that CSI and mass error
  can tell opposite stories


---


## 5. Discussion (~1,000 words)

### 5.1 Why the transfer fails

- Breach floods have a single-point source; the model learns radial
  spreading from one edge
- Rainfall floods add water everywhere simultaneously; the model sees
  a zero-rain channel and has no mechanism for distributed input
- The learned dynamics are structurally tied to the driver, not just the
  flow pattern
- This explains the asymmetry: Harvey→breach loses the rainfall
  information (it becomes irrelevant), breach→Harvey lacks it (it
  becomes essential)

### 5.2 Why direction doesn't help

- Direction is a downstream consequence of the driver, not the cause
- A model that learned "water flows radially from the left edge" has
  the correct direction for breach floods but the wrong source model
  for rainfall
- Preserving direction preserves the wrong structural assumption
  rather than correcting it
- This finding qualifies the recommendation in FloodGNN-GRU (Kazadi
  et al. 2024) and the concern in mSWE-GNN (Bentivoglio et al. 2023):
  direction matters for WITHIN-type accuracy, but does not bridge the
  cross-type gap

### 5.3 The CSI trap

- CSI is the most widely reported metric in flood model evaluation
  (cite Cohen et al. 2025 WRR on evaluation gaps)
- We show it can give a false positive on transfer: a model can score
  higher CSI on foreign data than on home data while being physically
  nonsensical
- This happens whenever the source and target have different depth
  distributions, which is inherent in cross-type transfer
- Recommendation: any cross-type or cross-scale evaluation MUST include
  mass balance as a complementary metric

### 5.4 Limitations

(See the 5 limitations from Step 3 report — write them honestly)

### 5.5 Broader implications

- Current flood-ML benchmarks (FloodSimBench, UrbanFloodBench) test
  within-type generalisation only
- Cross-type evaluation should be added to benchmark suites
- The result supports calls for more rigorous evaluation in flood ML
  (Cohen et al. 2025 WRR; the broader "baseline saturation" movement
  in ML evaluation)


---


## 6. Conclusion (~300 words)

- Restate the three findings in one sentence each
- The practical implication: do not assume a flood surrogate trained on
  one scenario will work on a different type of flood event
- The methodological implication: evaluate with multiple metrics,
  including mass balance, especially when comparing across datasets
- Future work: cross-type transfer with larger models, fine-tuning
  strategies, and inclusion of coastal/pluvial flood types


---


## Figures and tables needed

| # | Type | Contents | Status |
|---|------|----------|--------|
| 1 | Bar chart | Home vs away CSI, Harvey→breach, both variants | TO DO |
| 2 | Bar + line | Breach→Harvey: CSI appears good, mass error explodes | TO DO |
| 3 | Scatter | CSI vs mass error, all 12 runs, coloured by condition | TO DO |
| 4 | Table | Full results (already in Step 3 report) | DONE |
| 5 | Diagram | Experimental setup: train on A → test on A and B | TO DO |
| 6 | Table | Dataset comparison (from Step 1 report) | DONE |


---


## References (key citations to include)

- Bentivoglio et al. (2023) — SWE-GNN, HESS — breach dataset, discarded direction
- Bentivoglio et al. (2025) — mSWE-GNN, NHESS — "did not evaluate for other types"
- Kazadi et al. (2024) — FloodGNN-GRU, EDS — direction as vector, Harvey dataset
- Song & Guan (2026) — Water Research — cross-regional transfer (same type)
- Cohen et al. (2025) — WRR — evaluation gaps in flood inundation mapping
- Stephens et al. (2014) — CSI statistical inconsistency
- The spatial generalisation paper — Env. Modelling & Software 2026
- Transfer learning for coastal systems — Cambridge 2026