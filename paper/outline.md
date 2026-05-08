# VRIFA SciTech Paper — Phase 1 + Phase 2 Outline

This outline applies the *Turning Work Into a Publication* methodology (Phases 1–2 of the deck in `~/Downloads/how_to_write/deck_dense.typ`) to the VRIFA repository plus the inherited evidence package. No prose-writing happens here. The goal is to lock the contribution and the evaluation design before any paragraph is drafted.

## Format target

`bamdone-aiaa` v0.2.0 on Typst Universe (`#import "@preview/bamdone-aiaa:0.2.0": aiaa`). One-column AIAA proceedings layout, numeric citations from BibLaTeX or Hayagriva, configurable authors/affiliations. Use this rather than the existing custom theme for the SciTech submission. The current `paper/typst/theme.typ` would remain only for the working drafts.

---

## Phase 1 — Figure Out the Contribution

### Step 1. Contribution-type audit

Mapping repo + inherited evidence against the eight contribution types in the deck.

| # | Type | Color band | Evidence in the repo | Strength |
|---|---|---|---|---|
| 1 | Systems | build it | MIT-licensed CLI, ~1.4k-line `vrifa.py`, COCO/YOLOv5/Darknet exports, `run_summary.yaml`, three reproduced runs, sample data | **Strong** |
| 2 | Application | known method, new domain | Classical change-detection (Otsu, morphology, temporal persistence) brought to VARTM video; first open optical-input pipeline in this domain (lit review §02) | **Strong** |
| 3 | Algorithmic | a new technique | Directional peak-brightness rule `score = peak − current` with darken-only cutoff; closest precedents are the *symmetric* codebook (Kim 2005) and W4 (Haritaoglu 2000) envelopes (lit review §04) | **Narrow** |
| 4 | Empirical | characterize a phenomenon | 91-trial ablation showing how knobs (reference mode, color space, threshold offset, blur, morphology, min area, lock-frames) shift a multi-metric agreement objective from 0.583 to 0.807 | **Strong** |
| 5 | Benchmark | the field's measuring stick | 1,006 labeled frames, 4,689 region annotations, 20-frame human-labeled evaluation subset; lit review §02 confirms no comparable public corpus exists | **Moderate** (single-fixture) |
| 6 | Diagnostic | why methods work or fail | Implicit in the ablation but not formalized; not a primary claim | Weak |
| 7 | Theoretical | proofs, bounds | None | None |
| 8 | Conceptual | reframe the problem | "Front geometry over time is the right monitoring observable" is well-established in the LCM physics literature (Tucker & Dessenberger; Pillai & Advani; Hsiao & Advani); not a reframing | Weak |

The four strong/moderate cells (Systems, Application, Empirical, Benchmark) together are the load-bearing material. Algorithmic is real but narrow and should not headline.

### Step 2. Thesis claim — single vs umbrella

**Recommendation.** Single-contribution **Application** thesis with the **Systems** and **Empirical** material absorbed as supporting evidence. The dataset/Benchmark angle is held back as a second paper unless reviewers ask for it explicitly.

Reasoning. The deck recommends single-contribution generally, and the lit review shows the cleanest white-space gap is *open, video-input, full-field, detector-ready, reproducible* in the LCM domain. That is an Application contribution (classical change-detection retargeted to VARTM video) with Systems and Empirical evidence. Trying to also headline a Benchmark claim weakens the dataset's defensibility, since 1,006 frames on three runs from one fixture is a strong starting corpus but not yet a standard benchmark in the sense of MVTec AD or COCO.

**Strongest in three senses, applied to this paper.**

- *Defensible.* Application + Systems + Empirical is fully backed by code, data, and the 91-trial ablation. No overreach.
- *Novel.* Lit review §02 confirms no published mask IoU or boundary F1 on real VARTM infusion video; and lit review §11 confirms no comparable open tool. The novelty is in the *combination* of optical input, full-field 2D output, multi-format export, and reproducibility logging.
- *Useful.* A practitioner can clone the repo, point it at their own video, and produce reproducible front geometry plus detector-ready labels the same afternoon.

**Recommended one-sentence thesis** (Step 2 deliverable):

> VRIFA adapts directional peak-brightness change-detection to VARTM resin-infusion video and produces reproducible 2D flow-front geometry plus detector-ready supervision in COCO, YOLOv5, and Darknet formats from ordinary cameras, with a 91-trial ablation showing that domain-aware tuning of explicit pipeline knobs improves a multi-metric agreement objective on a 20-frame human-labeled subset from 0.583 to 0.807 (mask IoU 0.935, boundary F1 0.559, mean boundary distance 61.5 px).

**Backup theses if a reviewer asks for an alternative spine.**

- *Systems-only.* "VRIFA is the first open, MIT-licensed CLI that turns VARTM infusion video into reproducible flow-front geometry and detector-ready supervision in three standard formats, with a fully-logged run configuration."
- *Empirical-only.* "We characterize how a small set of explicit pipeline knobs controls flow-front segmentation quality on VARTM video, mapping a 91-trial ablation from a baseline objective of 0.583 to 0.807, and identify which knobs carry the boundary-F1 improvement (peak-brightness reference, darken-only differencing, threshold offset)."

### Step 3. Thesis-template fit

Application contributions take the template *achieves X on new domain D*. Filling it.

> *X* = pixel-accurate flow-front segmentation with mask IoU 0.935, Dice 0.966, boundary F1 0.559, mean boundary distance 61.5 px, average compute 41 ms / frame on CPU at export and 120.7 ms / frame in the full evaluation pipeline.
>
> *D* = VARTM resin infusion (transparent vacuum-bag layups, three runs, 1,006 labeled frames).

The deck calls for *evaluation = domain metrics + comparison to domain SOTA*. Domain metrics are the multi-metric set above. Domain SOTA in the optical-input regime does not exist with comparable reported numbers (lit review §02), so the comparison is to (a) VRIFA's own untuned default configuration as the null, and (b) FlowFrontNet (Stieber et al. 2021) as the closest *learned* baseline acknowledging the I/O contract is different (sensor input, simulated). Phase 2 below makes this concrete.

### Step 4. Minimum-publishable-contribution check

Three Step-4 tests applied to the recommended thesis.

- **Precise.** Yes. Specific numbers (0.583 → 0.807; mask IoU 0.935; boundary F1 0.559; mean boundary distance 61.5 px; 1,006 frames; 4,689 annotations). Named runs are identifiable in the public repo. Reproducible with `python vrifa.py --video-path …` and the inherited optimizer config.
- **Convincing.** Mostly. The ablation directly supports *tunable* and *improves agreement*. The dataset volume directly supports *produces detector-ready supervision*. One soft spot. The qualitative YOLO overlay video is weak evidence for *detector-ready* in a strict sense — it shows that exports are consumable, but does not quantify downstream detector quality. **Mitigation.** Frame the contribution as *exports detector-ready labels* (provable from artifacts) and explicitly defer the detector benchmark to follow-up work, citing the label-noise robustness literature (Frenay & Verleysen; Rolnick; Northcutt).
- **Non-obvious.** Yes, with the right framing. That classical CV can produce flow-front masks is unsurprising. The non-obvious findings are (i) the directional half-envelope rule outperforms symmetric envelopes specifically because resin wetting is a monotonic darkening transition, surfaced by the optimizer retaining peak-reference + darken-only; and (ii) the design-space sweep gives a 38% relative improvement in a multi-metric agreement objective on a held-out human-labeled subset, including a 2.7× improvement in boundary F1, the metric most tightly tied to the physically meaningful front-position observable.

**Verdict.** Phase 1 passes the minimum-publishable check. The recommended thesis is locked.

### Step 1–4 deliverable

The contribution is **Application (with Systems + Empirical evidence absorbed)**, and the thesis is the one-sentence claim above.

---

## Phase 2 — Design the Evaluation

Phase 1 is clear, so Phase 2 is included in this outline. Each step below maps onto the deck's Step-5-through-Step-8 prescription.

### Step 5. Datasets

**Match dataset to angle.** The angle is *optical-input flow-front segmentation in VARTM*. The repo's three runs are exactly this. The 20-frame human-labeled evaluation subset is the held-out reference and stays as the headline reporting surface.

**Always include the canonical benchmark.** No standard benchmark exists in this domain (lit review §02 and §11). The honest move is to name that absence in the paper and treat the released VRIFA corpus (1,006 frames, 4,689 annotations) as the surrogate canonical, while acknowledging it is single-fixture.

**Diversity through stress-test runs.** The three runs differ in frame rate and exposure, which gives some breadth, but they share fixture and lighting. Two recommended additions, both inexpensive.

1. *Held-out-run protocol.* Of the three runs, optimize on two and evaluate on the third. Report degradation. This is the *generalization-distance* dataset choice the deck recommends for an application claim.
2. *Synthetic illumination perturbation.* Apply gamma, exposure, and white-balance perturbations to the evaluation subset and report metric degradation. This stress-tests the peak-brightness reference's claimed robustness against the very phenomena it is built to absorb.

**Out-of-scope for SciTech submission.** A new fixture, a new fabric, or a new resin system would strengthen the paper but cannot land before submission unless data already exists offline.

### Step 6. Baselines

**Tier 1 — classical.** VRIFA's default configuration (first-frame reference, no peak-brightness, no darken-only, threshold offset 0). This is the natural null and is what the inherited 0.583 score measures.

**Tier 2 — domain SOTA.**

- *FlowFrontNet (Stieber et al. 2021, MIT-licensed)*. Closest learned baseline in LCM. Cannot run head-to-head because its input is a simulated sensor grid, not video pixels. Report the I/O contract difference and the published time-step accuracy explicitly so reviewers know the comparison is qualitative.
- *AI-CMCA U-Net + MobileNetV2 (Khalghollah et al. 2025)*. The cleanest learned wetting-front segmenter, trained on capillary microfluidic video, IoU 0.992. Optionally retrain its open architecture on VRIFA labels as a *learned-upper-bound* baseline. Treat as future work in this submission to keep scope tight.
- *Optical-input prior art (Pineda et al. 2010; Almazán-Lázaro et al. 2018, 2022; Mejía-Ugalde et al. 2020).* None publish mask IoU or boundary F1 on real infusion video. Tabulate this absence in the related-work section as a comparison-by-omission rather than a numeric head-to-head.

**Tier 3 — ablations.** Already covered by the inherited 91-trial study. For the SciTech version, present three named ablation slices.

1. *Reference-mode ablation.* `first` vs `running` vs `prev N` vs `peak`. Isolates the peak-brightness contribution.
2. *Directionality ablation.* `darken-only` vs symmetric. Isolates the directional half-envelope contribution, the closest thing VRIFA has to an algorithmic claim.
3. *Color-space ablation.* CIELAB vs RGB vs HSV vs grayscale. The optimizer converged on RGB; explain why.

The remaining ablation dimensions (threshold offset, blur, morphology kernel and iterations, min area, lock-frames) belong in a supplementary table.

### Step 7. Metrics

**Headline.** The multi-metric agreement objective (0.583 → 0.807). State the weighting up front.

**Efficiency.** Per-frame wall-clock on CPU, 41 ms at export and 120.7 ms in the full evaluation pipeline. Report both. State the CPU model in the protocol box.

**Breakdown.** Per-metric reporting of mask IoU, Dice/F1, boundary F1, box IoU, mean boundary distance. This is the per-condition breakdown the deck calls for. It also matches the multi-metric reporting the *Metrics Reloaded* (Maier-Hein et al. 2024) recommendation prescribes for segmentation work (lit review §12).

**Tied to the claim's angle.** Boundary F1 is the metric most tightly tied to the *front-position* observable. Lead the discussion with it. The 0.206 → 0.559 improvement, a 2.7× factor, is the cleanest single number.

**Drop these.** Any single-IoU summary would understate the contribution; explicitly omit it in favor of the multi-metric breakdown.

### Step 8. Experimental rigor

**Define the protocol.** Search space, trial budget (91), evaluation subset (20 human-labeled frames), stopping rule (mixed-variable optimization with the inherited objective). Apply identically to the baseline configuration as well. The same 20-frame subset evaluates both 0.583 and 0.807 configurations.

**Define the compute.** Report 1h21m03s total runtime, named CPU, OpenCV version, Python version. Confirm in the paper that the optimizer was run once end-to-end and that the reported best is the global best of the 91 trials.

**Seeds and confidence intervals.** OpenCV-on-CPU is deterministic given identical inputs and configs, so trial-to-trial variance is zero by construction. The relevant uncertainty is over the 20-frame evaluation subset; report 95% bootstrap confidence intervals (1,000 resamples) on each metric. This is the right uncertainty surface here.

**Stress tests.** Three concrete ones, in priority order.

1. *Held-out run.* Optimize on two of the three runs and evaluate on the third. Report degradation as the generalization headline.
2. *Synthetic illumination perturbation.* Gamma 0.7–1.4, exposure ±1 stop, white-balance shift. Report per-perturbation metrics. This is the test the peak-brightness rule was built to pass; if it fails, that itself is a publishable finding.
3. *Compute-budget downgrade.* Report metrics at `--frame-step 5` and `--frame-step 10` to bound a near-real-time deployment scenario.

**Failures discovered here become qualitative evidence in the Results section** (per the deck's Step 8). Do not paper over them.

### Phase 2 deliverable

The evaluation design is locked to (a) the three existing runs plus a held-out-run split, (b) the 20-frame human-labeled subset for headline metrics with bootstrap CIs, (c) tier-1 default-config baseline plus a qualitative comparison to FlowFrontNet and the optical-input prior art, (d) the multi-metric reporting set led by boundary F1, and (e) three named stress tests.

---

## Phase 3 (deferred)

Not drafted. Per the user's instruction, drafting begins after Phase 1 + Phase 2 are reviewed and approved.

---

## Items requiring user decision before drafting

1. **Which thesis variant.** Recommended is the Application thesis above. Confirm or pick a backup.
2. **Held-out run experiment.** Inexpensive but adds one optimization run. Approve to include.
3. **Synthetic perturbation experiment.** Inexpensive. Approve to include.
4. **FlowFrontNet handling.** Cite as closest learned baseline with I/O contract difference, *or* attempt a head-to-head by simulating a sensor grid from VRIFA's mask output (more work, stronger paper).
5. **AIAA Typst format.** Switch the SciTech submission build to `@preview/bamdone-aiaa:0.2.0` and keep the existing custom theme only for the working drafts.
6. **Two flagged citations** (lit review §02). The Li et al. 2023 YOLO paper could not be located; replace, locate, or drop. The 2025 dielectric+YOLO LCM paper needs author/venue verification. Both must be resolved before drafting the related-work paragraph.
