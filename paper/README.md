# VRIFA SciTech Draft Set

Three overnight draft variants were generated from the current repo contents plus the inherited PDF draft at `~/Downloads/vrifa_aiaa_extended_abstract.pdf`.

Files:

- `build/vrifa_variant_a_ablation.pdf`
  - Best if you want the safest conference story.
  - Centers the 91-trial ablation and optimization study.
  - Treats detector training as downstream motivation, not the main claim.

- `build/vrifa_variant_b_dataset.pdf`
  - Best if you want to push the YOLO angle.
  - Centers VRIFA as an annotation engine that converts infusion video into detector-ready assets.
  - Still avoids a hard detector benchmark claim because the repo does not yet contain a polished comparison table.

- `build/vrifa_variant_c_monitoring.pdf`
  - Best if you want the most manufacturing-facing version.
  - Centers process observability, temporal progression, and practical deployment.

Quick previews:

- `build/previews/vrifa_variant_a_ablation.png`
- `build/previews/vrifa_variant_b_dataset.png`
- `build/previews/vrifa_variant_c_monitoring.png`

Shared evidence base used across the drafts:

- Three exported runs in the repo.
- `1,006` labeled frames.
- `4,689` exported region annotations.
- Inherited 20-frame human-labeled evaluation subset.
- Inherited 91-trial ablation and optimization study.
- Best inherited score `0.807`, baseline `0.583`.

Important caveats for the next pass:

- Author order is marked provisional in the drafts.
- The current public repo does not yet support a clean external "best on the market" claim.
- The strongest defensible immediate paper is the ablation-first or monitoring-first version unless the YOLO training results are cleaned up and documented.
