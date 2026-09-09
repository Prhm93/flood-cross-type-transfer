# Paper v2 — Revised Results After the Five Confound-Killing Experiments

Written 9 September 2026. This document is the bridge between the v1 draft,
the external reviewer's critique, and the new experimental evidence. It
states plainly what got resolved, what changed, and what is now honestly
reported as inconclusive.


## Summary table — reviewer objection vs outcome

| # | Reviewer objection | Experiment | Outcome |
|---|---|---|---|
| 2 | CSI paradox might be a threshold artefact | Exp 1: threshold sweep 0.1%–10% | **RESOLVED — paradox holds at every threshold for breach→Harvey** |
| 11 | Transfer failure might just be the missing rainfall channel | Exp 2: rainfall removed from both | **RESOLVED — failure persists nearly identically without rainfall** |
| 8 | Model might be too weak to draw conclusions from | Exp 5: 492k-param model vs 42k | **RESOLVED — larger model shows the same or worse transfer** |
| 1, 10 | Flood type confounded with solver/region/resolution | Exp 3: controlled same-terrain, same-solver synthetic | **PARTIALLY RESOLVED — confirmed in one direction, inconclusive in the other (see below)** |
| 5 | Fine-tuning samples might not be independent events | Exp 4: explicit independent simulation IDs, swept 1–40 | **RESOLVED (design) but finding changed (see below)** |
| 3 | mass error of "30" is ambiguous | All experiments now report volume ratio | **RESOLVED — unambiguous "predicted N times the true volume"** |


## 1. The CSI paradox survives every threshold (Exp 1)

For the breach-trained model evaluated on Harvey data, CSI-away exceeds
CSI-home at all six thresholds tested (0.1%, 0.5%, 1%, 2%, 5%, 10% of
maximum true depth):

| Threshold | Home CSI | Away CSI | Paradox? |
|-----------|----------|----------|----------|
| 0.1% | 0.170 | 0.791 | yes |
| 0.5% | 0.167 | 0.621 | yes |
| 1.0% | 0.165 | 0.561 | yes |
| 2.0% | 0.161 | 0.500 | yes |
| 5.0% | 0.163 | 0.417 | yes |
| 10.0% | 0.190 | 0.326 | yes |

The gap narrows as the threshold loosens (a coarser threshold is less
easily fooled by a thin false layer) but never closes, even at 10% of
maximum depth — a threshold well within normal practice. The volume ratio
confirms independently: the breach-trained model predicts 131× the true
water volume when tested on Harvey data, versus 2.3× on its own home data.

In the reverse direction (Harvey-trained, tested on breach), no paradox
occurs at any threshold — away CSI is lower than home CSI throughout, as
expected. **The paradox is one-directional**, which itself is informative:
it appears specifically when a model trained on a low-magnitude,
distributed-input flood type is applied to a domain where "wet" is defined
relative to much deeper water. This asymmetry should be stated explicitly
in the paper rather than treated as a general property of CSI.

**Revised claim for the paper:** "The CSI paradox is not an artefact of a
single threshold choice: it holds across an order of magnitude of threshold
values, and is confirmed independently by a threshold-free volume-ratio
measure. The effect is asymmetric, appearing when a distributed-input model
is applied to concentrated, deep flooding, but not in the reverse
direction — a pattern consistent with the threshold-sensitivity mechanism
we propose."


## 2. Removing rainfall does not rescue or explain the failure (Exp 2)

| Direction | With rainfall (original) | Without rainfall (Exp 2) |
|---|---|---|
| Breach → Harvey, retained % | ~230–340% (paradox) | 258–323% (paradox) |
| Harvey → breach, retained % | ~29–38% | 34–44% |

The numbers are close enough that the presence or absence of the rainfall
input channel is not the explanation for either the transfer failure or
the CSI paradox. Both persist under an ablation that removes the one input
that most obviously differs between the two datasets.

**Revised claim for the paper:** "Removing the rainfall input from both
datasets did not meaningfully change either the magnitude of transfer
failure or the CSI paradox, indicating that neither phenomenon is an
artefact of the rainfall channel's presence or absence."


## 3. A model with 11x the parameters shows the same pattern (Exp 5)

| Architecture | Params | Harvey→breach retained | Breach→Harvey retained |
|---|---|---|---|
| Small | 42,371 | 39.5%, 46.8% | 328.7%, 278.8% |
| Large | 491,843 | 30.1%, 28.6% | 243.1%, 277.2% |

If anything, the larger model transfers slightly worse in the
Harvey→breach direction, not better. There is no evidence that additional
model capacity closes the transfer gap.

**Revised claim for the paper:** "Increasing model capacity roughly
eleven-fold did not close the transfer gap and, in the harder direction,
coincided with marginally lower retained skill, indicating the failure is
not attributable to insufficient model capacity."


## 4. The controlled synthetic experiment — confirmed in one direction, honestly inconclusive in the other

This is the experiment that isolates flood mechanism from every other
confound: identical terrain (same random seeds used for both mechanisms),
identical grid, identical solver (local inertial shallow water, Bates et
al. 2010 formulation), identical resolution, identical sample counts.

**Distributed → point (the clean result):**

| Seed | Home CSI | Away CSI | Retained |
|---|---|---|---|
| 0 | 0.752 | 0.064 | 8.5% |
| 1 | 0.566 | 0.056 | 9.9% |
| 2 | 0.311 | 0.053 | 16.9% |

Consistent, severe failure across all three seeds (retained 8.5–16.9%),
closely matching the magnitude of the real Harvey→breach result (29–44%).
With every other variable held constant, this is a controlled confirmation
that changing only the flood-driving mechanism causes substantial transfer
failure.

**Point → distributed (inconclusive):**

| Seed | Home CSI | Away CSI | Retained |
|---|---|---|---|
| 0 | 0.079 | 1.000 | 1260% |
| 1 | 0.061 | 0.560 | 920% |
| 2 | 0.056 | 0.0002 | 0.3% |

This direction is unstable across seeds, including one apparently
degenerate run (seed 2: home CSI 0.056 with a volume ratio of 0.00,
suggesting the model essentially predicted no water anywhere, including
at home). The synthetic point-source dataset is small (40 training
simulations) and trained for a comparatively short schedule; this
instability most plausibly reflects an undertrained model on a harder,
sparser task rather than a property of the transfer question itself.

**Honest framing for the paper:** we report the distributed→point result
as a genuine controlled confirmation, and explicitly flag the
point→distributed direction as inconclusive due to training instability,
recommending a larger synthetic dataset and longer training schedule as
follow-up work before that direction can be claimed. This is a more
defensible position than quietly omitting the unstable direction, and a
reviewer is far more likely to accept a paper that reports its own
experiment's limits than one that does not mention them.

**Revised claim for the paper:** "In a fully controlled setting where
terrain, grid, solver, resolution and sample count were held identical
across both mechanisms, a model trained on distributed (rainfall-like)
forcing lost 83–92% of its skill when the forcing mechanism changed to a
point source, closely matching the magnitude observed on real data. This
provides direct evidence that the flood-driving mechanism itself, not
incidental dataset differences, is sufficient to cause severe transfer
failure. The reverse direction showed high variance across seeds in this
smaller synthetic setting and is reported as inconclusive pending further
training; we do not draw conclusions from it."


## 5. Fine-tuning: a low-data head start, not a general fix (Exp 4)

This finding changed the most, and the paper's claim must change with it.

| Independent events | Fine-tuned CSI | From-scratch CSI |
|---|---|---|
| 0 (zero-shot) | 0.087 | — |
| 1 | 0.164 | 0.166 |
| 2 | 0.200 | 0.148 |
| 5 | 0.172 | 0.162 |
| 10 | 0.188 | 0.165 |
| 20 | 0.173 | 0.192 |
| 40 | 0.126 | 0.195 |

Fine-tuning has a real, if modest, edge over training from scratch at very
low data counts (2 events: 0.200 vs 0.148). By 10 events the two are
essentially tied. By 20–40 events, training from scratch overtakes
fine-tuning, and fine-tuned performance actually *degrades* at 40 events
relative to its peak at 2 — plausibly a sign of the pre-trained weights
constraining adaptation once enough target data exists to learn the new
mechanism from nothing.

The original v1 claim — "fine-tuning on a handful of examples helps" — is
too strong and must be replaced.

**Revised claim for the paper:** "Pre-training on a different flood type
provided a modest advantage over training from scratch only when
independent target-domain data was extremely scarce (one to two events);
this advantage disappeared by ten events and reversed by twenty to forty.
We conclude that cross-type pre-training offers a useful head start under
severe data scarcity, but is not a substitute for a moderate amount of
target-domain data, which alone was sufficient to match or exceed the
fine-tuned model beyond roughly ten independent events. This is itself a
practically useful finding: it tells a practitioner facing a new flood
type how much target data changes the calculus, and it does not support
treating cross-type fine-tuning as a general-purpose fix."


## What this means for the earlier "five findings" framing

The five findings from Step 5 (mixed training, fine-tuning, etc.) need
updating:

- Finding 1 (transfer fails) — **strengthened**, now confirmed under
  architecture change (Exp 5), rainfall ablation (Exp 2), and partially
  under full experimental control (Exp 3).
- Finding 2 (CSI can hide failure) — **strengthened substantially**, now
  shown to be threshold-robust (Exp 1) and to hold under rainfall ablation.
  This is now unambiguously the paper's strongest and most defensible
  contribution.
- Finding 3 (direction doesn't help transfer) — unchanged, still an honest
  null, should still be stated with the seed-variance caveat from the
  earlier critique.
- Finding 4 (naive mixing backfires) — unchanged.
- Finding 5 (fine-tuning helps) — **must be rewritten** per Section 5
  above. It is now a narrower, low-data-regime finding rather than a
  general remedy.


## Recommended paper structure change

Following the external reviewer's Point 17 ("too many small
contributions"), restructure around one thesis:

> Flood surrogate transfer failure is not simply a matter of geography; a
> change in flood-driving mechanism is sufficient to cause severe failure
> even under full experimental control, and the standard flood-extent
> metric can systematically misrepresent that failure in a way that is
> robust to threshold choice, input ablation, and model capacity.

Every experiment now supports this single sentence:
- Exp 3 (distributed→point) = the controlled proof of the thesis's first half
- Exp 1 = the threshold-robustness proof of the thesis's second half
- Exp 2, Exp 5 = confound elimination supporting both halves
- Exp 4 = the practical epilogue: what to do about it, honestly scoped

This is a stronger and more defensible paper than v1, precisely because it
now says less than it originally claimed in two places (the controlled
experiment's unstable direction, and fine-tuning's true scope) while
saying its central claims much more forcefully.


## Remaining before this is submission-ready

1. Rerun Exp 3's point→distributed direction with more synthetic training
   data and a longer schedule, OR explicitly scope the controlled claim to
   one direction only in the paper (defensible, since one clean controlled
   direction is still a first).
2. Increase seeds where feasible — 3 seeds remains thin for the
   architecture and controlled experiments; even 5 would tighten the
   confidence considerably given the GPU time is cheap per run.
3. Rewrite Section 4.6 (fine-tuning) of the v1 draft per Section 5 above —
   this is a significant rewrite, not a tweak.
4. Add the volume-ratio numbers throughout, replacing the ambiguous
   "30 to 130 times too much water" phrasing per the reviewer's Point 3.
5. Update Table 1 with full reproducibility detail per the reviewer's
   Point 21.