//============================================================
// Vision-Based Flow-Front Assessment in VARTM
//
// AIAA SciTech 2027 submission, System with empirical eval as support. 
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
    vacuum bag as their primary process indicator. Prior vision
    systems for VARTM and Liquid Composite Molding use the same
    classical visual primitives, namely reference-frame differencing,
    Otsu thresholding, and morphological cleanup, but each system
    uses a different subset, no system reports the joint and
    individual contribution of those primitives, and no labeled
    video benchmark exists against which to attribute that
    contribution. This work presents an integrated classical
    computer-vision pipeline that combines a peak-brightness
    reference, a darken-only difference, region-of-interest
    restriction, dynamic-lag reference selection, morphological
    cleanup, and persistence-based temporal locking, evaluated on a
    fifty-five-frame hand-labeled subset spanning eleven distinct
    VARTM infusion runs. The integrated configuration reaches mask
    Intersection-over-Union of $X.X X X$ (95 % bootstrap confidence
    interval $[a, b]$) and boundary $F_1$ of $Y.Y Y Y$
    (CI $[c, d]$). A per-sample component-removal ablation on the
    same subset characterizes the marginal IoU effect of each named
    primitive, both as an eleven-sample mean and broken down per
    sample so that primitives whose contribution depends on infusion
    regime are visible rather than averaged away. The pipeline runs
    at thirty frames per second on a single CPU and at $K$ frames
    per second on a CUDA implementation across the eleven samples.
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
#include "sections/datasets.typ"
#include "sections/results.typ"
#include "sections/runtime.typ"
#include "sections/discussion.typ"
#include "sections/conclusion.typ"
#include "sections/acknowledgments.typ"
