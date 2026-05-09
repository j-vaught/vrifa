//============================================================
// VRIFA — An Open Reference Implementation for Resin
// Flow-Front Detection in VARTM Process Video
//
// AIAA SciTech 2027 submission, Hybrid (Application + Systems)
// thesis.
//
// Build:
//   typst compile paper/main.typ paper/main.pdf
//============================================================

#import "lib.typ": *
#import "@preview/droplet:0.3.1": dropcap

#show: aiaa.with(
  title: "VRIFA: An Open Reference Implementation for Resin Flow-Front Detection in VARTM Process Video",

  bibliography: bibliography("refs.bib"),
  authors-and-affiliations: (
    (
      name: "J.C. Vaught",
      job: "Graduate Research Assistant",
      department: "Department of Mechanical Engineering",
      aiaa: "AIAA Student Member",
    ),
    // TODO confirm coauthors with user before submission.
    (
      institution: "University of South Carolina",
      city: "Columbia",
      state: "SC",
      zip: "29208",
      country: "USA",
    ),
  ),
  abstract: [
    Vacuum-Assisted Resin Transfer Molding (VARTM) operators read
    the visible advance of the resin flow front under the
    transparent vacuum bag as their primary process indicator,
    yet the open toolchain for turning that video into a
    reproducible, time-resolved flow-front segmentation is thin.
    Published vision systems for VARTM and Liquid Composite
    Molding either ship as proprietary controllers tied to a
    specific cell or as deep convolutional networks trained on
    simulated frames whose distribution does not match a typical
    research-bench camera, leaving practitioners without a
    starting point. Here we show that a twelve-stage classical-CV
    pipeline implemented in Rust, configured via fifty
    command-line flags, and shipped under the MIT license,
    segments resin flow fronts in VARTM video at thirty frames
    per second end-to-end on a single CPU. On a fifty-five-frame
    hand-labeled subset spanning eleven distinct VARTM runs,
    the default configuration achieves a mask
    Intersection-over-Union of $X.X X X$ (95 % bootstrap
    confidence interval $[a, b]$) and a boundary $F_1$ of
    $Y.Y Y Y$ (CI $[c, d]$). The implementation is auditable
    end-to-end via bit-exact per-stage parity tests and exports
    annotations in COCO, YOLOv5, and Darknet formats, supplying
    the supervision needed to bootstrap a downstream learned
    detector and supplying a reproducible reference for
    permeability-inversion and process-monitoring studies that
    today have no shared baseline.
  ],
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
