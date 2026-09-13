# Step 7 Report — Figure Suite, Five-Seed Exp 8, and the Correlation Correction

Written: 13 September 2026
Status: **PENDING FIGURE CONFIRMATION — do not advance to paper v2 until figA is signed off**

---

## 1. What this step covers

Since Step 6 (the v3/v4 experiment report), three things happened:

1. A nine-figure suite was built, reviewed, and fixed twice for layout problems.
2. The full first paper draft was written (`paper_draft_v1.md`, ~5,700 words).
3. An external review of the response-to-reviewer letter caught five real
   problems, one of which was investigated further and turned out to be a
   genuine statistical error that needed correcting, not just softening.

This report exists so the next version of the paper starts from a single,
verified account of what changed and why.

---

## 2. The figure suite

### 2.1 Build history

| Round | Script | What it did |
|---|---|---|
| 1 | `make_paper_figures.py` | Built figures A, B, C, D, E, F from real result JSONs |
| 2 | `make_paper_figures2.py` | Added G (capacity) and H (three-way metric check) |
| 3 | `make_paper_figures3.py` | Added I (controlled experiment) |
| 4 | `make_paper_figures_fixed.py` | Fixed layout collisions in A, B, D, F, I; **redesigned** E and H, whose original versions had design problems, not just layout ones |
| 5 | `patch_figures_AH.py` | A had text still overlapping after round 4; H's caption was factually wrong (claimed all three metrics agreed in both directions, when the figure itself showed the opposite) |
| 6 | `fix_figA_disagreement.py` | See Section 4 below — replaces the Spearman-rho box with pairwise-disagreement percentages |

### 2.2 Final figure list

| # | File | Shows | Status |
|---|---|---|---|
| 1 | `fig1_concept` | The two flood types, schematic | Signed off, untouched throughout |
| 2 | `fig2_workflow` | Method overview | Signed off, untouched throughout |
| 3 | `figA_per_event_scatter` | Per-event CSI vs volume ratio — lead figure | **Round 6 just applied, awaiting confirmation** |
| 4 | `figB_threshold_robustness` | Relative and fixed threshold sweeps | Signed off |
| 5 | `figC_mixing_regimes` | Five training regimes | Signed off, no changes needed at any round |
| 6 | `figD_mechanism_tag` | Mechanism tag, judged on CSI and volume | Signed off. **Numbers now stale — see Section 3** |
| 7 | `figE_finetune_curve` | Fine-tuning vs from-scratch | Signed off after redesign (medians + signed-difference panel, not min-max bands) |
| 8 | `figF_rainfall_ablation` | Rainfall ablation | Signed off |
| 9 | `figG_capacity` | Model capacity, 5 seeds | Signed off, no changes needed at any round |
| 10 | `figH_three_way` | Three metrics, both directions | Signed off after redesign (3x2 grid, natural units, explicit verdict per panel) and one caption correction |
| 11 | `figI_controlled` | Controlled synthetic experiment | Signed off after one annotation-spacing fix |

### 2.3 Superseded figures — do not cite

Five older figures exist in `results/figures/` from before this suite was
built: `fig1_transfer_gap`, `fig2_csi_vs_mass`, `fig3_transfer_matrix`,
`fig3_csi_trap`, `fig5_finetune`. They were built from the original
12-run experiment only, before the confound-elimination and v4 tier-1
work. `fig5_finetune` in particular carries a caption claiming fine-tuning
is "a cheap, practical fix," which the exp4 curve now contradicts. These
were not deleted (nothing has been deleted from this project without
explicit confirmation) but must not appear in the paper.

---

## 3. Exp 8 — repeated at 5 seeds

**Reason:** an external review of the confound-elimination work flagged
that Exp 8 (the mechanism-tag experiment) was reported at only 3 seeds,
and recommended it be repeated at 5 before being presented as a candidate
remedy. This was the one item in that review not already satisfied by
existing work.

**Method:** `scripts/exp8_mechanism_aware.py` was edited to run seeds
`[0, 1, 2, 3, 4]` instead of `[0, 1, 2]`. The script overwrites its
result file rather than merging (confirmed before editing, to avoid
repeating the exp5 contamination bug from Step 6), so this was a full
rerun, not an extension. Runtime approximately 3 hours 45 minutes on the
GPU container, run inside tmux session `flood4`.

**Result — median across 5 seeds, `results/result_exp8_mechanism_aware.json`:**

| Condition | Dam-break CSI | Dam-break volume | Rainfall CSI | Rainfall volume |
|---|---|---|---|---|
| No tag | 0.1591 | 0.51x | 0.5399 | 25.10x |
| With tag | 0.1953 | 0.85x | 0.5219 | **3.64x** |

**Comparison with the superseded 3-seed result:**

| | 3 seeds (superseded) | 5 seeds (current) |
|---|---|---|
| Rainfall CSI, no tag -> with tag | 0.505 -> 0.397 (apparent drop) | 0.540 -> 0.522 (essentially unchanged) |
| Rainfall volume, no tag -> with tag | 31.55x -> 2.60x | 25.10x -> 3.64x |

**Assessment:** the 5-seed result is cleaner and, if anything, easier to
defend than the 3-seed one. The dramatic CSI drop seen at 3 seeds mostly
washes out with more seeds — it was largely seed noise. What survives is
the volume improvement, which remains large (6.9x) at 5 seeds. The
revised, defensible claim is: **the mechanism tag does not cost CSI and
substantially improves physical accuracy** — a more modest and more
credible claim than "CSI actively punishes a real fix," which was the
3-seed framing.

**Outstanding:** `paper_draft_v1.md` Table 9 and `figD_mechanism_tag`
still show the 3-seed numbers. Both need updating to the 5-seed result
before the next paper version is written. Figure D itself needs no
redesign, only new numbers plugged into the existing script.

---

## 4. The correlation-sign correction

This is the most important item in this report and the reason paper v2
has not yet been written.

### 4.1 What the external review of the response letter caught

A second-order review (of the response-to-reviewer letter, not the paper
itself) identified five problems. Four were style and scope corrections,
accepted without further work:

1. Section 5.4's "not jointly learnable" line contradicts the Exp 8
   result and needed rewording to connect Exp 7 and Exp 8 into one
   causal chain (imbalance ruled out -> representation implicated).
2. Section 6.4's fine-tuning claim implied a firm transition point at
   ten events that the non-monotonic, 3-seed curve does not establish.
3. The response letter's tone repeatedly asserted priority ("already
   complete," "before this review was received") in a way that reads as
   defensive rather than informative.
4. The claim that a negative Spearman correlation is "too broad," since
   one of the four settings tested shows a positive correlation, not a
   universal negative pattern.

Point 4 above led to a fifth, deeper problem when checked properly.

### 4.2 The Spearman-direction error

The review's Point 1 objected to this sentence in the response letter:

> "A rank correlation of -0.454 is a direct measure of how often, and
> how severely, the two metrics disagree on which events are
> well-modelled."

This is wrong as written — a correlation coefficient measures the
strength and direction of a monotonic relationship, not a disagreement
frequency. The review's suggested fix was to compute an actual pairwise
ranking-disagreement percentage instead.

That calculation was run (`pairwise_disagreement.py`, using the existing
`result_exp6_metric_disagreement.json`, no retraining needed):

| Direction | Pairwise disagreement |
|---|---|
| Dam-break, own data | 64.3% (110/171 pairs) |
| Dam-break, rainfall data | 35.7% (61/171 pairs) |
| Rainfall, own data | 38.0% (65/171 pairs) |
| Rainfall, dam-break data | 38.6% (66/171 pairs) |

Checking this against the underlying correlation values surfaced the
deeper issue. The paradox direction (dam-break model on rainfall data)
has the **lowest** disagreement percentage of the four, not the highest.
This is mathematically expected — a stronger-magnitude correlation
(|-0.454| > |0.351|) naturally produces fewer discordant pairs, and this
alone is not evidence of anything wrong.

The real problem is what the correlation's *sign* means for the argument
being made. `score_per_event()` correlates CSI against
`|log(volume_ratio)|`, i.e. against distance-from-perfect. A **negative**
correlation therefore means: as CSI rises, distance-from-perfect falls —
i.e. **higher CSI correlates with more accurate volume**, in that
direction, across those 19 events. Read plainly, this is the opposite of
the paper's argument in exactly the direction the argument most needs it
(dam-break model transferred to rainfall data).

### 4.3 Why the paper's central claim is not damaged

Three pieces of evidence for the same paradox do not depend on this
correlation's sign and remain fully valid:

1. **The aggregate contrast.** Median CSI rises from 0.134 (home) to
   0.605 (away) while median volume ratio worsens from 2.30x to 135.5x,
   over the same 19 events. This is a comparison of two group medians,
   not a correlation.
2. **The 9/19 count.** Nine of nineteen events combine above-median CSI
   with a volume error beyond five-fold. This is a threshold count, also
   not a correlation.
3. **The named worst case.** Event 13: CSI 0.34, volume ratio 261x.

The pairwise-disagreement percentages computed in Section 4.2 also
survive, since they measure disagreement frequency correctly (unlike the
sentence that prompted this check) and were the number the review
actually asked for.

### 4.4 What must change before paper v2

- Section 6.1 of `paper_draft_v1.md`: remove the Spearman-rho claim and
  the "CSI ranks events in the wrong order" sentence. Replace with the
  aggregate contrast, the 9/19 count, and the named worst event, plus an
  explicit note that a single correlation coefficient is not reported
  here because it is not a reliable description of a 19-point
  relationship spanning three orders of magnitude in volume ratio.
- Table 7: drop the "Spearman rho" column. Add a "pairwise disagreement"
  column using the Section 4.2 percentages instead.
- `figA_per_event_scatter`: the on-figure box quoting rho per direction
  needs to be replaced with the disagreement percentages. Script written
  (`fix_figA_disagreement.py`) and awaiting your confirmation that the
  shape is correct before being treated as final.
- The response letter's Points 1 and 2 need the same correction: Point 1
  should cite the count and aggregate evidence only; Point 2 should be
  answered with the pairwise-disagreement percentages, honestly noting
  that the paradox direction shows *fewer* pairwise disagreements than
  the home direction even though the overall picture (median CSI and
  median volume both moving sharply in opposite directions) is much
  worse there.

None of this required deleting or rerunning any experiment. It is a
correction to how one existing, correctly-computed number
(`spearman_csi_vs_logvolerr` in the exp6 JSON) is described in prose, plus
a substitution of a different, already-available number
(pairwise disagreement) in its place.

---

## 5. Full current status

| Item | Status |
|---|---|
| Exp 1-8, audit, persistence baseline | Complete and verified against source JSONs |
| Exp 5 seed-2 contamination | Fixed 11 Sept |
| Exp 8 seed count | Extended to 5, complete 13 Sept |
| Nine-figure suite | Built, fixed twice, figA update pending confirmation |
| Paper draft v1 | Written 11 Sept, ~5,700 words, 8 [NEED FROM ME] markers open |
| Paper draft v1 corrections needed | Section 5.4 rewrite, Section 6.1 rewrite, Section 6.4 softening, Table 9 refresh to 5-seed numbers, Table 7 column swap |
| Response-to-reviewer letter | Written, needs the same corrections as the paper plus a tone pass removing priority-claiming language |
| GitHub | Up to date at github.com/Prhm93/flood-cross-type-transfer as of the exp5 fix commit; figure and exp8 updates not yet pushed |

## 6. Next step

Confirm `figA_per_event_scatter` looks right after the Section 4 fix.
Once confirmed, the next paper version can be written in one pass,
applying: the Section 5.4 rewrite, the Section 6.1 rewrite, the Section
6.4 softening, the Table 9 refresh, and the Table 7 column swap. No
further experiments are planned or needed.