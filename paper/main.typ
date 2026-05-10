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
    Our work presents four contributions towards the real-time tracking
    and control of VARTM processes. The first is an integrated
    classical computer-vision pipeline that combines a peak-brightness
    reference, a darken-only difference, region-of-interest
    restriction, dynamic-lag reference selection, morphological
    cleanup, persistence-based temporal locking, and run-time
    camera-shift registration, reaching mean mask
    Intersection-over-Union of $0.921$ (95 % CI $[0.889, 0.943]$) and mean boundary $F_1$ of $0.433$
    (CI $[0.396, 0.473]$) on the benchmark introduced below. On the
    same benchmark, the two prior published classical-CV pipelines
    for VARTM flow-front segmentation, reimplemented from their
    source papers, reach mean boundary $F_1$ of $0.116$
    (Lekanidis and Vosniakos 2020) and $0.187$ (Almazán-Lázaro
    2022); their outputs lock onto distribution-medium pattern and
    bag-wrinkle artifacts rather than the resin front. The second
    is a component-removal ablation that characterizes the marginal
    IoU effect of each named primitive. The third is a
    regime-indexed configuration lookup that recommends preprocessing
    settings as a function of run circumstances so a practitioner
    can select an empirically-grounded starting point. The fourth
    is the benchmark itself, namely a 55-frame hand-labeled subset
    across 11 VARTM infusion runs with per-sample
    region-of-interest masks and a documented labeling protocol,
    against which the two prior pipelines above are evaluated as
    reference points and any future flow-front segmentation method
    can be evaluated identically. The pipeline runs at 30 frames per second on a single
    CPU and at 67 frames per second on a CUDA implementation for a
    $1920 times 1080$ input.
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
