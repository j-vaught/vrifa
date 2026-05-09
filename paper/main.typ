//============================================================
// VRIFA — Vision-Based Flow-Front Assessment in VARTM
//
// AIAA SciTech 2027 submission, Hybrid (Application + Systems).
// Format follows paper/aiaa-example/main.typ.
//
// Build:
//   typst compile paper/main.typ paper/main.pdf
//============================================================

#import "lib.typ": *
#import "@preview/droplet:0.3.1": dropcap

#show: aiaa.with(
  title: "Vision-Based Flow-Front Assessment in Vacuum-Assisted Resin Transfer Molding",

  bibliography: bibliography("refs.bib"),
  authors-and-affiliations: (
    (
      name: "J.C. Vaught",
      job: "Graduate Research Assistant",
      department: "Department of Mechanical Engineering",
      aiaa: "AIAA Student Member",
    ),
    (
      name: "Marshall Pigford",
      job: "Undergraduate Research Assistant",
      department: "Department of Computer Science and Engineering",
      aiaa: "AIAA Student Member",
    ),
    (
      name: "Declan Johnson",
      job: "Graduate Research Assistant",
      department: "Department of Mechanical Engineering",
      aiaa: "AIAA Student Member",
    ),
    (
      name: "Darun Barazanchy",
      job: "Research Assistant Professor",
      department: "Department of Mechanical Engineering",
      aiaa: "AIAA Member",
    ),
    (
      institution: "University of South Carolina",
      city: "Columbia",
      state: "SC",
      zip: "29208",
      country: "USA",
    ),
  ),
  abstract: [
    Vacuum-Assisted Resin Transfer Molding (VARTM) operators read the
    visible advance of the resin flow front under the transparent
    vacuum bag as their primary process indicator, yet the open
    toolchain for turning that video into a reproducible,
    time-resolved flow-front segmentation is thin. Published vision
    systems for VARTM and Liquid Composite Molding are released
    either as proprietary controllers tied to a specific cell or as
    deep convolutional networks trained on simulated frames whose
    distribution does not match a typical research-bench camera,
    leaving practitioners without a starting point. This work
    presents a twelve-stage classical computer-vision pipeline
    implemented in Rust, configured via roughly fifty parameters,
    and released under the MIT license, that segments resin flow
    fronts in VARTM video at thirty frames per second end-to-end on
    a single CPU. On a fifty-five-frame hand-labeled subset spanning
    eleven distinct VARTM runs, the default configuration achieves
    a mask Intersection-over-Union of $X.X X X$ (95 % bootstrap
    confidence interval $[a, b]$) and a boundary $F_1$ of $Y.Y Y Y$
    (CI $[c, d]$). The implementation is auditable end-to-end via
    bit-exact per-stage parity tests and exports annotations in
    COCO, YOLOv5, and Darknet formats, supplying the supervision
    needed to bootstrap a downstream learned detector and serving
    as a reproducible reference for permeability-inversion and
    process-monitoring studies that today have no shared baseline.
  ],
)

#nomenclature(
  ($F_t$, "converted-colorspace frame at time index t"),
  ($G_t$, "reference image at time index t"),
  ($G_t^star$, "effective reference (peak map when peak-reference enabled, otherwise channel-zero of G_t)"),
  ($P_t$, "per-pixel peak-brightness map at time t"),
  ($D_t$, "per-pixel response field at time t"),
  ($R$, "binary region-of-interest mask"),
  ($H, W$, "frame height and width in pixels"),
  ($C$, "channel count of the working colorspace"),
  ([$alpha$], [exponential-moving-average factor for running-reference mode]),
  ([$kappa$], [sqrt-area growth factor estimated from calibration frames]),
  ([$rho$], [target wet-fraction for dynamic reference mode]),
  ([$lambda$], [user lag scale factor for dynamic reference]),
  ([$tau_t$], [elapsed time at frame index t in seconds]),
  ([$Delta tau_t$], [dynamic lag in seconds for the reference frame]),
  ($w_c$, "channel weight in the delta computation"),
  ($N_"calib"$, "number of calibration frames before dynamic reference activates"),
  ($f$, "video frame rate in frames per second"),
  ($"IoU"$, "Intersection-over-Union mask agreement score"),
  ($F_1$, [boundary $F_1$ score, mean across pixel tolerances $tau in {1, 3, 5}$]),
)

#include "sections/introduction.typ"
#include "sections/related_work.typ"
#include "sections/method.typ"
#include "sections/implementation.typ"
#include "sections/datasets.typ"
#include "sections/results.typ"
#include "sections/runtime.typ"
#include "sections/bootstrap.typ"
#include "sections/discussion.typ"
#include "sections/conclusion.typ"
