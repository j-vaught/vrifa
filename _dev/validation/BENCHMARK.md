# VRIFA ablation v2 — overnight benchmark plan

This document is the operational design for the ablation pass that
fills Tables T2 and T3 in the paper. Read it end-to-end before anyone
hits "go" on the server. Nothing about the paper is asserted here — the
goal is to spell out exactly what trials run, in what order, on what
videos, and how the run survives an overnight session without
intervention.

## 1. Goals

For each named primitive in the integrated pipeline, produce a measured
IoU effect on the eleven-sample labeled subset (55 frames). The
benchmark answers two questions per primitive:

1. What is the per-video best setting?
2. How much does the integrated configuration's choice cost on each
   video relative to that video's own best?

The output is a per-stage CSV plus a per-stage "winner" JSON that the
paper figures and tables read directly.

The benchmark explicitly does **not**: tune defaults to win on the
labeled subset (we report what defaults give vs what each video's best
gives), search the joint space (the per-stage chain is the explicit
honesty of the design), or measure runtime — runtime is a separate
exercise documented elsewhere.

## 2. Phase architecture

Ten phases. **Phases are sequential** because each phase reads the
previous phase's per-video winner JSON as its baseline configuration.
**Within a phase, all (video, trial) combinations run in parallel**
through a worker pool. The 112-core server (`comech-2422`) is the
assumed host; the worker pool is sized at **10 concurrent vrifa-rs
processes**. Each vrifa process uses Rayon's default thread count
(~98 cores). At N=10 the box reaches ~97 cores total in use without
oversubscription, and a measured 9.5x speedup vs sequential. N=20 was
tested and is **catastrophically worse** (1.3x speedup, 7x worse than
N=10) because of cache thrashing and CPU oversubscription, so do not
push the worker count up without re-measuring.

```
Phase 1   rough per-video baseline              (gating)
   |
   v
Phase 2   colorspace + channel-weights deep     (per-video, uses Phase-1 winners)
   |
   v
Phase 3   reference selection                   (per-video, uses Phase-1+2 winners)
   |
   v
Phase 4   threshold mode (finer grids)          (per-video, uses Phase-1..3 winners)
   |
   v
Phase 5   pre-delta blur (finer grids)          (per-video, uses Phase-1..4 winners)
   |
   v
Phase 6   post-delta blur (finer grids)         (per-video, uses Phase-1..5 winners)
   |
   v
Phase 7   morphology (finer kernel grid)        (per-video, uses Phase-1..6 winners)
   |
   v
Phase 8   lock fine sweep                       (per-video, uses Phase-1..7 winners)
   |
   v
Phase 10  joint perturbation around chained     (per-video, uses Phase-1..8 winners)
   |       best (sanity check on chained-greedy)
   v
Phase 9   stabilization                         (input_1 only, uses Phase-1..8 winners)
```

(Phase 9 is numbered last in the order it executes; the digit is
preserved from the original design for traceability.)

Phase 9 is technically independent of Phases 2-8 (it only needs Phase-1
winner for input_1) but is run after Phase 10 so the entire integrated chain is
honest end-to-end. If Phase 9 needs to be moved earlier for time
reasons, the script supports a `--start-phase` flag.

## 3. Per-trial protocol

Every trial in every phase follows the same recipe:

1. Resolve the trial's full config: take this video's **prior-stage
   winner** as the baseline, then change exactly the knobs this phase
   sweeps.
2. Set `--roi-margin 0` for input_2 through input_11 (pre-cropped),
   keep the default 0.15 for input_1.
3. Run the vrifa binary on `data/ablation_data/<sample>.mp4` (trimmed
   stride-15 + label-preserving copy, ~600-1000 frames per video) with
   **mask PNGs only** -- no overlays, no heatmaps, no videos, no
   annotation export. (`--write-mask-pngs true` plus the others false.)
4. Output to `$TMPDIR/vrifa_ablation/runs/<trial_id>/<sample>/masks/`.
5. Run `agreement.py` against `data/ablation_data/labels.json` (remapped
   to the trimmed videos' frame indices), restricted to the matching
   sample's frames.
6. Persist the trial's metrics into a per-trial JSON in
   `$TMPDIR/vrifa_ablation/results/phase<N>/<trial_id>.json`.
7. Delete the trial's mask directory.
8. Append a one-line CSV row to `$TMPDIR/vrifa_ablation/all_trials.csv`
   for monitoring.

`trial_id` format: `phase<N>__<sample>__<knob1>=<val1>__<knob2>=<val2>...`.
This is the durable identity used by checkpointing.

## 4. Phase-by-phase trial matrix

All counts assume 11 videos unless otherwise noted. The "Knob sweep"
column lists the discrete values varied; everything else is held at the
prior-stage per-video winner (or the integrated default for Phase 1).

### Phase 1 — Rough per-video baseline

| Knob | Default | Sweep values |
|---|---|---|
| `--threshold-offset` | -30 | -50, -30, -10 |
| `--min-area` | 400 | 100, 400, 1600 |
| `--morph-kernel` | 13 | 7, 13, 21 |
| `--lock-frames` | 3 | 0, 3, 10 |

3⁴ = **81 trials per video × 11 videos = 891 trials.** Output:
`data/ablation/phase1_per_video_best.json`.

### Phase 2 — Colorspace + channel weights (deep, per-Option-C)

At Phase-1 best per video. Substantially expanded grid for the 3-channel
colorspaces so the per-mold colorspace recommendation has weight-vector
detail.

| `--colorspace` | `--channel-weights` (parsed as `w1,w2,w3`) |
|---|---|
| CIELAB | (1,0,0) (0,1,0) (0,0,1) (1,0.5,0) (1,0,0.5) (1,0.5,0.5) (1,1,0) (1,0,1) (0,1,1) (1,1,1) (0.5,1,1) (0.7,0.7,0.3) (0.3,0.7,0.7) (0.7,0.3,0.7) (0.5,0.7,0.3) (0.3,0.5,0.7) (0.6,0.2,0.2) |
| RGB | (1,1,1) (1,0,0) (0,1,0) (0,0,1) (0.299,0.587,0.114) (0.5,0.5,0) (0.5,0,0.5) (0,0.5,0.5) (0.7,0.2,0.1) (0.2,0.7,0.1) (0.1,0.2,0.7) (0.4,0.4,0.2) (0.6,0.3,0.1) (0.3,0.6,0.1) (0.1,0.3,0.6) (0.33,0.34,0.33) (0.5,0.3,0.2) |
| HSV | (0,0,1) (0,1,0) (1,0,0) (0,1,1) (1,0,1) (1,1,0) (1,1,1) (0.3,0.3,0.4) (0.2,0.4,0.4) |
| GRAYSCALE | (1) |

× `--darken-only` ∈ {on, off} = (17 + 17 + 9 + 1) × 2 = **88 trials × 11
videos = 968 trials.**

### Phase 3 — Reference selection

At Phase-2 best per video.

| `--ref-mode` | per-mode params |
|---|---|
| first | — |
| running | `--ref-running-alpha` ∈ {0.01, 0.02, 0.05, 0.1, 0.2} |
| prev 1 | — |
| prev 3 | — |
| prev 10 | — |
| prev 30 | — |
| dynamic | `--dynamic-target-fraction` × `--dynamic-lag-scale` ∈ {0.1, 0.2, 0.3, 0.5} × {0.5, 1.0, 2.0} |

= 1 + 5 + 4 + 12 = **22 trials × 11 = 242 trials.**

### Phase 4 — Threshold mode (finer grids, per-Option-C)

At Phase-3 best per video.

| `--threshold` | sweep |
|---|---|
| `otsu` | `--threshold-offset` ∈ {-60, -55, -50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, +5, +10} |
| `triangle` | `--threshold-offset` ∈ {-30, -25, -20, -15, -10, -5, 0, +5, +10, +15, +20, +25, +30} |
| `manual:V` | V ∈ {20, 40, 60, 80, 100, 120, 140, 160, 180} |
| `percentile:P` | P ∈ {40, 50, 60, 70, 80, 85, 90, 95, 98} |
| `adaptive-mean:B:C` | B ∈ {7, 11, 21, 31, 51, 71} × C ∈ {-10, -5, 0, 5, 10, 15} |
| `adaptive-gaussian:B:C` | B ∈ {7, 11, 21, 31, 51, 71} × C ∈ {-10, -5, 0, 5, 10, 15} |

= 15 + 13 + 9 + 9 + 36 + 36 = **118 trials × 11 = 1,298 trials.**

### Phase 5 — Pre-delta blur (finer grids + all kinds, per-Option-C)

At Phase-4 best per video. Adds median and bilateral kinds that the
blur module supports.

| `--pre-delta-blur` | sweep |
|---|---|
| `none` | (1 trial) |
| `flat:S` | S ∈ {3, 5, 7, 9, 11, 13, 15, 17, 21} |
| `gaussian:S` | same |
| `triangle:S` | same |
| `median:S` | S ∈ {3, 5} (OpenCV restricts f32 median to these) |
| `bilateral:S` | S ∈ {5, 9, 15} |

= 1 + 3 × 9 + 2 + 3 = **33 trials × 11 = 363 trials.**

### Phase 6 — Post-delta blur (finer grids + all kinds, per-Option-C)

At Phase-5 best per video. Same kind menu as Phase 5.

| `--blur` | sweep |
|---|---|
| `none` | (1 trial) |
| `flat:S` | S ∈ {3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 25} |
| `gaussian:S` | same |
| `triangle:S` | same |
| `median:S` | S ∈ {3, 5} |
| `bilateral:S` | S ∈ {5, 9, 15} |

= 1 + 3 × 11 + 2 + 3 = **39 trials × 11 = 429 trials.**

### Phase 7 — Morphology (finer kernel grid, per-Option-C)

At Phase-6 best per video. Two sub-sweeps to keep the matrix small.

7a. Kernel × shape, iters fixed at (1, 1):
- `--morph-kernel` ∈ {3, 5, 7, 9, 11, 13, 15, 17, 21, 25, 31, 41, 51, 71, 101, 151}
- `--morph-shape` ∈ {ellipse, rect, cross}

= 16 × 3 = 48 trials.

7b. Iterations × kernel, shape fixed at the 7a winner:
- `--morph-close-iterations` × `--morph-open-iterations` ∈ {1, 2, 3} × {1, 2, 3}
- × top 5 kernels from 7a (per video)

= 9 × 5 = 45 trials.

= 93 trials × 11 = **1,023 trials.**

### Phase 8 — Lock fine sweep (per-Option-C)

At Phase-7 best per video.

| `--lock-frames` | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 60, 90 |

= **19 trials × 11 = 209 trials.**

### Phase 10 — Joint perturbation (Option-C addition, expanded)

At each video's chained best from Phases 1-8, perturb the three
highest-leverage knobs simultaneously by ±2 steps on each axis. Tests
whether the chained-greedy winner is actually a local optimum in the
joint space, or whether knob-i and knob-j interact in a way the
single-axis sweeps missed. Knobs perturbed: `threshold-offset`,
`morph-kernel`, `lock-frames`. Each axis takes 5 values: {-2 step,
-1 step, 0 (the chained best), +1 step, +2 step}.

= 5³ = 125 trials per video × 11 = **1,375 trials.**

### Phase 9 — Stabilization (input_1 only)

At Phase-8 best for input_1.

| Knob | sweep |
|---|---|
| `--camera-stable` | off, on (2 values) |
| `--motion-model` | translation, affine |
| `--motion-per-frame-threshold` | 0.5, 1.0, 1.5, 2.5, 5.0 |
| `--cumulative-motion-threshold` | 1, 2, 3, 5, 10 |

`--camera-stable=off` is one trial with the other knobs ignored. `=on`
is the cross-product of motion-model × per-frame × cumulative = 2 × 5 × 5 = 50.

= 1 + 50 = **51 trials, input_1 only.**

## 5. Aggregate trial count and runtime budget

| Phase | Trials | Notes |
|---|---:|---|
| 1 | 891 | All 11 videos |
| 2 | 968 | All 11 videos (Option-C deep weight grid: 17/17/9/1 colorspace combos × 2 darken) |
| 3 | 242 | All 11 videos |
| 4 | 1,298 | All 11 videos (Option-C finer threshold grid; all 6 modes) |
| 5 | 363 | All 11 videos (Option-C finer pre-blur grid; all 5 kinds + none) |
| 6 | 429 | All 11 videos (Option-C finer post-blur grid; all 5 kinds + none) |
| 7 | 1,023 | All 11 videos (Option-C finer morph grid) |
| 8 | 209 | All 11 videos (Option-C finer lock grid) |
| 10 | 1,375 | All 11 videos (Option-C joint perturbation, 5³ expansion) |
| 9 | 51 | input_1 only (run last) |
| **Total** | **~6,849** | (drop the original §5 row totals; superseded by this row) |

| 3 | 242 | All 11 videos |
| 4 | 649 | All 11 videos |
| 5 | 209 | All 11 videos |
| 6 | 275 | All 11 videos |
| 7 | 957 | All 11 videos |
| 8 | 187 | All 11 videos |
| 9 | 51 | input_1 only |
| **Total** | **3,791** | |

Runtime model: each trial is one vrifa-rs subprocess + one
agreement.py invocation. Mean per-trial wall-clock at 8 threads on the
RTX-host server:
- input_1 / input_2 / input_3 (small): ~5–8 s
- input_4–11 (large, 8k–15k frames): ~30–60 s

Mean across 11 videos ≈ 30 s. With 14 concurrent workers:
Per-trial wall on the trimmed videos was measured at ~4 s effective
under N=10 (input_11 trimmed batch of 10 finished in 40.6 s wall-clock,
matching ideal scaling on the 112-core box). The full ablation budget
under Option C and N=10 workers projects to:

- 6,849 trials × ~4 s effective / 10 workers ≈ **~46 min pure compute**
- With process spawn overhead, agreement.py serialization, and
  occasional retry: **estimate 2-4 hours wall-clock end-to-end.**

The wall budget is now generous enough that the run is no longer the
constraint; we'd rather report fewer trials with high CIs than push
to N=20 and lose 7x in throughput from oversubscription.

Disk peak: 10 concurrent × ~3 MB of trimmed-video mask PNGs ≈ 30 MB
transient. Cleaned per trial; total persistent on-disk after the run
≈ 30 MB of JSON plus 5 MB of CSV.

## 6. Where things live on disk

Two concerns: temporary work that must NOT enter git, and final
artifacts that must.

**Transient (gitignored, on the run host's local disk):**
```
$TMPDIR/vrifa_ablation/
  state.json                  checkpoint + resume index
  all_trials.csv              every trial result, append-only
  log/                        per-phase stdout/stderr capture
    phase1.log
    phase2.log
    ...
  results/
    phase1/<trial_id>.json    one file per trial, full metrics
    phase2/<trial_id>.json
    ...
  runs/                       trial mask PNGs, deleted after agreement
    <trial_id>/<sample>/masks/
```

**Persistent (committed under `data/ablation/`):**
```
data/ablation/
  README.md                   how to read these files
  phase1_per_video_best.json  per-sample winner config + metrics
  phase1_all_trials.csv       all trials with metrics
  phase2_per_video_best.json
  phase2_all_trials.csv
  ...
  phase9_per_video_best.json
  phase9_all_trials.csv
  summary.md                  consolidated report, one row per phase
```

`data/ablation/` is added to a fresh `data/.gitignore`
exception so only the JSON/CSV/MD land in git, NOT the transient mask
PNGs (which never leave `$TMPDIR` anyway).

## 7. Run procedure

Three commands. Designed for SSH-then-disconnect.

```bash
# 0. one-time setup on comech-2422
cd ~/bench_vrifa
source ~/miniforge3/etc/profile.d/conda.sh && conda activate fastvrifa
export LD_LIBRARY_PATH=$HOME/cuda-12.4/lib64:$HOME/miniforge3/envs/fastvrifa/lib

# 1. start the run inside a tmux session so SSH disconnect doesn't kill it
tmux new -d -s ablation \
  "python3 _dev/validation/ablation_v2/run.py --workers 10 \
   2>&1 | tee /tmp/vrifa_ablation/log/run.log"

# 2. detach (Ctrl-b d if attached). Then reconnect any time:
tmux attach -t ablation

# 3. resume after a crash / kill / power loss
tmux new -d -s ablation \
  "python3 _dev/validation/ablation_v2/run.py --workers 10 --resume \
   2>&1 | tee -a /tmp/vrifa_ablation/log/run.log"
```

Monitor remotely without attaching:

```bash
ssh comech-2422 "tail -F /tmp/vrifa_ablation/log/run.log"
ssh comech-2422 "wc -l /tmp/vrifa_ablation/all_trials.csv"
ssh comech-2422 "jq '.phases_done, .trials_done | length' /tmp/vrifa_ablation/state.json"
```

When the run finishes, the final summary lands at
`$TMPDIR/vrifa_ablation/summary.md` and the JSON/CSV are copied into
`data/ablation/`. The user pulls them home with one rsync.

## 8. Overnight-robustness checklist

Each row is a concrete failure scenario and the concrete mitigation
that prevents it.

| Failure | Mitigation |
|---|---|
| SSH disconnects mid-run | Run lives inside `tmux`. Survives the disconnect. |
| Server reboots | Resume reads `state.json` and skips trial_ids in `trials_done`. |
| Single trial crashes (vrifa segfault, OpenCV exception) | Caught per-worker; trial is logged as `failed` with stderr captured. Pool keeps going. |
| Single trial hangs forever | Each subprocess has a per-trial timeout (default **1200 s**). On timeout the worker SIGKILLs the subprocess and marks the trial failed. The 1200 s budget is conservative versus the measured input_11 trimmed cost of ~40 s under N=10. |
| Failed trial loses its diagnostic | Every failed trial dumps `$TMPDIR/vrifa_ablation/log/failed/<trial_id>/{cli.txt, stderr.txt, stdout.txt, env.txt, exit_code}`. The harness's `--resume` flag reruns failed trials only when `--retry-failed` is also passed, so the diagnostic stays intact for review even after a successful retry. |
| Disk fills with mask PNGs | Mask cleanup is the last step inside the worker, in a `try`/`finally` that always runs. Worker fails closed (cleans even on exception). |
| Two workers race on the same `runs/` directory | Each worker creates a UUID-suffixed runs subdir; no two workers ever share a path. |
| Process pool dies | Driver detects worker pool exit; if `state.json` shows < 100% complete, exits with error 1 so the user sees a non-zero shell prompt instead of a silent stop. |
| Power loss during state.json write | State writes are atomic (write to `state.json.tmp`, fsync, rename). |
| Git pollution | `data/.gitignore` already excludes `data/`; ablation outputs that we WANT committed live in `data/ablation/` with an explicit `!ablation/` exception added. Transient masks live in `/tmp` which is never near `git status`. |
| User wants to stop early | `Ctrl-C` (or `kill -TERM` from outside tmux) traps cleanly: workers finish current trial, state.json is updated, partial results are still queryable. |
| Re-run after a flag-set change | `state.json` records the trial_id, which encodes the full config. Changing a sweep value generates a new trial_id and forces a re-run for that trial only. |
| Repository changes mid-run (e.g. someone pushes) | Run uses the binary at the path it started with. Source changes don't affect the in-flight run. The driver records `git rev-parse HEAD` at start and refuses to merge results into a different rev unless `--force` is passed. |

## 9. Definition of done

The run is complete when:

1. `state.json` reports `phases_done == [1, 2, 3, 4, 5, 6, 7, 8, 10, 9]`
   (note ordering: Phase 10 runs after 8, Phase 9 last) and
   `trials_failed.length` is within the **failure budget (default 50)**.
2. `data/ablation/summary.md` exists with one row per phase.
3. Each `phase<N>_per_video_best.json` contains 11 entries (or 1 for
   Phase 9), each with `mean_iou`, `ci_low`, `ci_high`, and the winning
   config.
4. Each `phase<N>_all_trials.csv` row count matches the phase's
   declared trial count to within the failure budget.
5. `$TMPDIR/vrifa_ablation/log/failed/` is reviewed and either
   `--retry-failed` cleared the failures or each failure has a comment
   from the operator marking it understood (a `*.understood` sibling
   file).

The default budget of **50** is intentionally tolerant: the harness is
designed to keep going through pathological-config crashes (e.g. some
combinations of huge morph kernel × small ROI on a particular video).
Each failure is logged with full diagnostic so it can be reviewed
post-hoc, but a single failure does not block the run.

## 10. Final experiments table

The full experiment matrix the harness will execute, ordered by phase.
Read this top-to-bottom and check that every cell matches what you
asked for.

| Phase | Subject | Sweep dims | Per-video trials | Videos | Total trials | Output |
|---:|---|---|---:|---:|---:|---|
| 1 | Rough baseline | thr-offset × min-area × morph-kernel × lock | 81 | 11 | 891 | data/ablation/phase1_*.{json,csv} |
| 2 | Colorspace + channel weights (deep, Option-C) | colorspace × weights × darken | 88 | 11 | 968 | data/ablation/phase2_*.{json,csv} |
| 3 | Reference selection | mode + mode-specific params | 22 | 11 | 242 | data/ablation/phase3_*.{json,csv} |
| 4 | Threshold mode (finer grids, Option-C) | otsu/triangle offsets, manual, percentile, adaptive-mean B×C, adaptive-gaussian B×C | 118 | 11 | 1,298 | data/ablation/phase4_*.{json,csv} |
| 5 | Pre-delta blur (finer grids + all 5 kinds, Option-C) | kind × size, plus none; flat/gaussian/triangle/median/bilateral | 33 | 11 | 363 | data/ablation/phase5_*.{json,csv} |
| 6 | Post-delta blur (finer grids + all 5 kinds, Option-C) | kind × size, plus none; same kinds as Phase 5 | 39 | 11 | 429 | data/ablation/phase6_*.{json,csv} |
| 7a | Morphology kernel × shape (Option-C) | 16 kernels × 3 shapes | 48 | 11 | 528 | data/ablation/phase7a_*.{json,csv} |
| 7b | Morphology iterations × top-5 kernel | 9 iters × 5 kernels | 45 | 11 | 495 | data/ablation/phase7b_*.{json,csv} |
| 8 | Lock fine sweep (Option-C) | 19 lock-frame values | 19 | 11 | 209 | data/ablation/phase8_*.{json,csv} |
| 10 | Joint perturbation (Option-C, 5³) | 3 axes × 5 values around chained best | 125 | 11 | 1,375 | data/ablation/phase10_*.{json,csv} |
| 9 | Stabilization | camera-stable + motion-model × per-frame × cumulative | 51 | 1 (input_1) | 51 | data/ablation/phase9_*.{json,csv} |
| **all** | | | | | **6,849** | data/ablation/summary.md |

## 11. What lands in the paper

After the run finishes, the paper sections that get filled in are:

| Paper element | Source | What gets read |
|---|---|---|
| Table T2 (Results §6) | `phase8_per_video_best.json` | Per-sample IoU at the integrated configuration's chain of bests |
| Component-removal ablation (§5) | `phase{N}_per_video_best.json` for relevant N | "remove this primitive, IoU drops by X" |
| Per-mold colorspace recommendation (§Discussion) | `phase2_per_video_best.json` | Which colorspace + weights wins per video |
| Reference-mode comparison panel | `phase3_all_trials.csv` | One panel per ref-mode showing IoU spread |
| Threshold-mode comparison panel | `phase4_all_trials.csv` | Otsu vs Triangle vs Adaptive curves |
| Pre-blur cross-video table | `phase5_all_trials.csv` | Whether pre-blur helps per video |
| Stabilization panel | `phase9_all_trials.csv` | input_1 IoU with stabilization on/off |

## 12. What this benchmark does NOT do

- **Does not measure runtime.** Wall-clock numbers go in a separate
  three-binary comparison (python / vrifa-rs / fast-vrifa) at a single
  chosen config, not in this ablation. The two are kept apart so neither
  contaminates the other.
- **Does not search the full joint config space.** Each phase varies one
  primitive's knobs while holding everything else at the prior winner;
  Phase 10 perturbs three knobs jointly around the chained best as a
  sanity check, but a full 3⁸ joint grid (6,561 per video) is out of
  scope.
- **Does not retune defaults.** The integrated configuration is the
  shipped defaults, frozen for the paper. The ablation reports what the
  per-video best is for each primitive without changing the defaults.
- **Does not regenerate paper figures automatically.** The CeTZ figures
  read these JSON / CSV files; the figures themselves recompile when
  the paper compiles. No image generation is part of this run.

## 13. Locked decisions

| Decision | Value | Rationale |
|---|---|---|
| Failure budget | **50** | High tolerance so pathological configs do not block the run. Each failure is logged with full diagnostic. |
| Per-trial timeout | **1200 s** | Conservative versus the measured input_11 trimmed cost of ~40 s under N=10. Catches genuine hangs without false-killing slow-but-progressing trials. |
| Resume granularity | **per-trial** | `state.json` records every completed trial_id. Restart skips done trials and re-runs only what is missing. |
| Phase 9 timing | **last (after Phase 10)** | Keeps the chained-honest narrative; input_1 stabilization data is the final stage. |
| Workers | **10** | Measured 9.5x speedup vs sequential at N=10 (input_11 trimmed batch finished in 40.6 s). N=20 was tested and is 7x worse than N=10 due to oversubscription on the 112-core box. |

Harness implementation follows. Ready to write when this document is
signed off.
