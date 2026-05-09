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
    // TODO Abstract — written last, after the body and final
    // numbers are in place. Three-beat template: Context,
    // Gap, Here-we-show.
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
