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

Nine phases, one per primitive. **Phases are sequential** because each
phase reads the previous phase's per-video winner JSON as its baseline
configuration. **Within a phase, all (video, trial) combinations run in
parallel** through a worker pool. The 112-core server is the assumed
host; with 14 concurrent vrifa-rs processes (each ~8 threads), wall-clock
per phase is dominated by the longest video × the per-video trial count.

```
Phase 1  rough per-video baseline               (gating)
   |
   v
Phase 2  colorspace + channel weights           (per-video, uses Phase-1 winners)
   |
   v
Phase 3  reference selection                    (per-video, uses Phase-1+2 winners)
   |
   v
Phase 4  threshold mode                         (per-video, uses Phase-1+2+3 winners)
   |
   v
Phase 5  pre-delta blur                         (per-video, uses Phase-1..4 winners)
   |
   v
Phase 6  post-delta blur                        (per-video, uses Phase-1..5 winners)
   |
   v
Phase 7  morphology (kernel/shape/iters)        (per-video, uses Phase-1..6 winners)
   |
   v
Phase 8  lock fine sweep                        (per-video, uses Phase-1..7 winners)
   |
   v
Phase 9  stabilization                          (input_1 only, uses Phase-1..8 winners for input_1)
```

Phase 9 is technically independent of Phases 2-8 (it only needs Phase-1
winner for input_1) but is run last so the entire integrated chain is
honest end-to-end. If Phase 9 needs to be moved earlier for time
reasons, the script supports a `--start-phase` flag.

## 3. Per-trial protocol

Every trial in every phase follows the same recipe:

1. Resolve the trial's full config: take this video's **prior-stage
   winner** as the baseline, then change exactly the knobs this phase
   sweeps.
2. Set `--roi-margin 0` for input_2 through input_11 (pre-cropped),
   keep the default 0.15 for input_1.
3. Run the vrifa binary with **mask PNGs only** — no overlays, no
   heatmaps, no videos, no annotation export. (`--write-mask-pngs true`
   plus the others false.)
4. Output to `$TMPDIR/vrifa_ablation/runs/<trial_id>/<sample>/masks/`.
5. Run `agreement.py` against `data/labels_55.json`, restricted
   to the matching sample's frames.
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

### Phase 2 — Colorspace + channel weights

At Phase-1 best per video.

| `--colorspace` | `--channel-weights` (parsed as `w1,w2,w3`) |
|---|---|
| CIELAB | (1,0,0) (1,0.5,0.5) (1,1,1) (0.5,1,1) (0,1,1) |
| RGB | (1,1,1) (1,0,0) (0,1,0) (0,0,1) (0.299,0.587,0.114) |
| HSV | (0,0,1) (0,1,0) (0,1,1) (1,1,1) |
| GRAYSCALE | (1) |

× `--darken-only` ∈ {on, off} = (5 + 5 + 4 + 1) × 2 = **30 trials × 11
videos = 330 trials.**

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

### Phase 4 — Threshold mode

At Phase-3 best per video.

| `--threshold` | sweep |
|---|---|
| `otsu` | `--threshold-offset` ∈ {-60, -50, -40, -30, -20, -10, 0, +10} |
| `triangle` | `--threshold-offset` ∈ {-30, -20, -10, 0, +10, +20, +30} |
| `manual:V` | V ∈ {30, 50, 70, 90, 110, 130, 150} |
| `percentile:P` | P ∈ {50, 70, 80, 90, 95} |
| `adaptive-mean:B:C` | B ∈ {11, 21, 31, 51} × C ∈ {-5, 0, 5, 10} |
| `adaptive-gaussian:B:C` | B ∈ {11, 21, 31, 51} × C ∈ {-5, 0, 5, 10} |

= 8 + 7 + 7 + 5 + 16 + 16 = **59 trials × 11 = 649 trials.**

### Phase 5 — Pre-delta blur

At Phase-4 best per video.

| `--pre-delta-blur` | sweep |
|---|---|
| `none` | (1 trial) |
| `flat:S` | S ∈ {3, 5, 7, 9, 11, 15} |
| `gaussian:S` | same |
| `triangle:S` | same |

= 1 + 3 × 6 = **19 trials × 11 = 209 trials.**

### Phase 6 — Post-delta blur

At Phase-5 best per video.

| `--blur` | sweep |
|---|---|
| `none` | (1 trial) |
| `flat:S` | S ∈ {3, 5, 7, 9, 11, 13, 15, 19} |
| `gaussian:S` | same |
| `triangle:S` | same |

= 1 + 3 × 8 = **25 trials × 11 = 275 trials.**

### Phase 7 — Morphology

At Phase-6 best per video. Two sub-sweeps to keep the matrix small.

7a. Kernel × shape, iters fixed at (1, 1):
- `--morph-kernel` ∈ {3, 5, 7, 9, 11, 13, 17, 21, 25, 31, 41, 51, 71, 101}
- `--morph-shape` ∈ {ellipse, rect, cross}

= 14 × 3 = 42 trials.

7b. Iterations × kernel, shape fixed at the 7a winner:
- `--morph-close-iterations` × `--morph-open-iterations` ∈ {1, 2, 3} × {1, 2, 3}
- × top 5 kernels from 7a (per video)

= 9 × 5 = 45 trials.

= 87 trials × 11 = **957 trials.**

### Phase 8 — Lock fine sweep

At Phase-7 best per video.

| `--lock-frames` | 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 60 |

= **17 trials × 11 = 187 trials.**

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
| 2 | 330 | All 11 videos |
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
- 3,791 trials × 30 s / 14 ≈ **2.3 hours pure compute**
- With process spawn overhead, agreement.py serialization, and
  occasional retry: **estimate 4–6 hours wall-clock end-to-end.**

Disk peak: 14 concurrent × ~30 MB of mask PNGs = ~500 MB transient.
Cleaned per trial; total persistent on-disk after run ≈ 30 MB of JSON
plus 5 MB of CSV.

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
  "python3 _dev/validation/ablation_v2/run.py --workers 14 \
   2>&1 | tee /tmp/vrifa_ablation/log/run.log"

# 2. detach (Ctrl-b d if attached). Then reconnect any time:
tmux attach -t ablation

# 3. resume after a crash / kill / power loss
tmux new -d -s ablation \
  "python3 _dev/validation/ablation_v2/run.py --workers 14 --resume \
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
| Single trial hangs forever | Each subprocess has a per-trial timeout (default 600 s). On timeout the worker SIGKILLs the subprocess and marks the trial failed. |
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

1. `state.json` reports `phases_done == [1, 2, 3, 4, 5, 6, 7, 8, 9]` and
   `trials_failed.length == 0` (or the user has reviewed the failures).
2. `data/ablation/summary.md` exists with one row per phase.
3. Each `phase<N>_per_video_best.json` contains 11 entries (or 1 for
   Phase 9), each with `mean_iou`, `ci_low`, `ci_high`, and the winning
   config.
4. Each `phase<N>_all_trials.csv` row count matches the phase's
   declared trial count to within the failure budget (default 0).

## 10. Final experiments table

The full experiment matrix the harness will execute, ordered by phase.
Read this top-to-bottom and check that every cell matches what you
asked for.

| Phase | Subject | Sweep dims | Per-video trials | Videos | Total trials | Output (relative to paper/) |
|---:|---|---|---:|---:|---:|---|
| 1 | Rough baseline | thr-offset × min-area × morph-kernel × lock | 81 | 11 | 891 | data/ablation/phase1_*.{json,csv} |
| 2 | Colorspace + channel weights | colorspace × weights × darken | 30 | 11 | 330 | data/ablation/phase2_*.{json,csv} |
| 3 | Reference selection | mode + mode-specific params | 22 | 11 | 242 | data/ablation/phase3_*.{json,csv} |
| 4 | Threshold mode | otsu/triangle offsets, manual, percentile, adaptive-mean B×C, adaptive-gaussian B×C | 59 | 11 | 649 | data/ablation/phase4_*.{json,csv} |
| 5 | Pre-delta blur | kind × size, plus none | 19 | 11 | 209 | data/ablation/phase5_*.{json,csv} |
| 6 | Post-delta blur | kind × size, plus none | 25 | 11 | 275 | data/ablation/phase6_*.{json,csv} |
| 7a | Morphology kernel × shape | 14 kernels × 3 shapes | 42 | 11 | 462 | data/ablation/phase7a_*.{json,csv} |
| 7b | Morphology iterations × top-5 kernel | 9 iters × 5 kernels | 45 | 11 | 495 | data/ablation/phase7b_*.{json,csv} |
| 8 | Lock fine sweep | 17 lock-frame values | 17 | 11 | 187 | data/ablation/phase8_*.{json,csv} |
| 9 | Stabilization | camera-stable + motion-model × per-frame × cumulative | 51 | 1 (input_1) | 51 | data/ablation/phase9_*.{json,csv} |
| **all** | | | | | **3,791** | data/ablation/summary.md |

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
- **Does not search the joint config space.** Each phase varies one
  primitive's knobs while holding everything else at the prior winner.
  Joint optima are not chased; the cost of chained-greedy is reported in
  the discussion.
- **Does not retune defaults.** The integrated configuration is the
  shipped defaults, frozen for the paper. The ablation reports what the
  per-video best is for each primitive without changing the defaults.
- **Does not regenerate paper figures automatically.** The CeTZ figures
  read these JSON / CSV files; the figures themselves recompile when
  the paper compiles. No image generation is part of this run.

## 13. Open decisions before kickoff

1. **Failure budget.** Default is 0 (any failed trial blocks the
   "complete" check). Acceptable to set to e.g. 5 (skip up to 5
   pathological configs)?
2. **Per-trial timeout.** Default 600 s. Big videos at extreme morph
   kernels could plausibly approach this. Raise to 1200 s for safety?
3. **Resume-from-where checkpoint granularity.** Per-trial (current
   design) vs per-phase. Per-trial is more conservative and recommended.
4. **Phase 9 timing.** Run last (current design) or earlier so input_1
   stabilization data is available before the long-running phases?
5. **Workers.** 14 is conservative for a 112-core box. Push to 20 or
   24? The constraint is OpenCV's thread pool overlap.

Once these are answered the harness gets written and we kick off.
