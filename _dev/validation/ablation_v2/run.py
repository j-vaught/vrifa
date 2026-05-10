#!/usr/bin/env python3
"""Ablation v2 orchestrator -- per BENCHMARK.md.

Runs the 10 phases in order, with per-trial worker-pool parallelism.
Resumable, failure-tolerant up to the configured budget. Final outputs
land at data/ablation/.

Typical use on the server:
  cd ~/bench_vrifa
  source ~/miniforge3/etc/profile.d/conda.sh && conda activate fastvrifa
  export LD_LIBRARY_PATH=$HOME/cuda-12.4/lib64:$HOME/miniforge3/envs/fastvrifa/lib
  tmux new -d -s ablation \\
    "python3 -m _dev.validation.ablation_v2.run --workers 10 \\
     2>&1 | tee /tmp/vrifa_ablation/log/run.log"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from multiprocessing import Value
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_BINARY,
    DEFAULT_LABELS,
    DEFAULT_RESULTS_DIR,
    DEFAULT_STATE_DIR,
    DEFAULT_VIDEOS_DIR,
    PHASE_ORDER,
    SAMPLES_FULL,
)
from . import phases as phase_defs_full
from . import phases_lean as phase_defs_lean
phase_defs = phase_defs_full  # default; overridden by --lean
from .state import load_state, save_state
from .trial import TrialConfig, TrialResult, run_trial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=1200.0,
                        help="per-trial timeout in seconds")
    parser.add_argument("--failure-budget", type=int, default=50)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS_DIR)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--start-phase", type=str, default=None,
                        help="Skip earlier phases; useful for resume + reorder")
    parser.add_argument("--only-phase", type=str, default=None,
                        help="Run a single phase and stop")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Re-run trials that failed previously")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: run only 3 trials per phase across 2 samples")
    parser.add_argument("--lean", action="store_true",
                        help="Use lean grids from phases_lean (~1.4k trials post-Phase-1).")
    parser.add_argument("--gpus", type=int, default=0,
                        help="Number of GPUs to round-robin workers across "
                             "via CUDA_VISIBLE_DEVICES. 0 = let the binary "
                             "pick its own GPU (default).")
    parser.add_argument("--phase-order", type=str, default=None,
                        help="Comma-separated phase IDs to run in order, "
                             "overriding the default PHASE_ORDER. "
                             "Example: --phase-order 7a,7b,8,10,3,2,5,6,4,9")
    return parser.parse_args()


def _gpu_init(counter, num_gpus: int):
    """Process-pool initializer: each worker gets pinned to a GPU
    via CUDA_VISIBLE_DEVICES. Counter is a multiprocessing.Value
    that hands out sequential worker IDs."""
    if num_gpus <= 0:
        return
    with counter.get_lock():
        my_id = counter.value
        counter.value += 1
    gpu = my_id % num_gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


# Worker entry: ProcessPoolExecutor pickles the call. Keep arguments
# JSON-friendly where possible; pass labels as a path so each worker
# loads it once.
def _worker_run_trial(
    trial_payload: dict[str, Any],
    binary: str,
    labels_path: str,
    runs_root: str,
    log_root: str,
    timeout_s: float,
) -> dict[str, Any]:
    import json as _json
    from pathlib import Path as _P
    labels = _json.loads(_P(labels_path).read_text())
    trial = TrialConfig(
        trial_id=trial_payload["trial_id"],
        phase=trial_payload["phase"],
        sample=trial_payload["sample"],
        video_path=_P(trial_payload["video_path"]),
        flags=trial_payload["flags"],
    )
    result = run_trial(
        trial,
        _P(binary),
        labels,
        _P(runs_root),
        _P(log_root),
        timeout_s,
    )
    return asdict(result)


def trial_to_payload(trial: TrialConfig) -> dict[str, Any]:
    return {
        "trial_id": trial.trial_id,
        "phase": trial.phase,
        "sample": trial.sample,
        "video_path": str(trial.video_path),
        "flags": trial.flags,
    }


def all_trials_csv_init(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="") as fh:
            csv.writer(fh).writerow([
                "trial_id", "phase", "sample", "ok", "iou_mean",
                "dice_mean", "boundary_f1_mean", "box_iou_mean",
                "runtime_s", "error",
            ])


def all_trials_csv_append(path: Path, result: dict[str, Any]) -> None:
    with path.open("a", newline="") as fh:
        csv.writer(fh).writerow([
            result["trial_id"], result["phase"], result["sample"],
            result["ok"],
            result.get("iou_mean", ""), result.get("dice_mean", ""),
            result.get("boundary_f1_mean", ""), result.get("box_iou_mean", ""),
            result.get("runtime_s", ""), result.get("error", "") or "",
        ])


def video_path_for(videos_dir: Path, sample: str) -> Path:
    return videos_dir / f"{sample}.mp4"


def overrides_for_phase(
    phase: str,
    sample: str,
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return the list of override dicts for one (phase, sample)."""
    if phase in phase_defs.PHASE_OVERRIDE_FNS:
        return phase_defs.PHASE_OVERRIDE_FNS[phase]()

    if phase == "7b":
        # Top 5 kernels from this sample's phase 7a winners.
        winners_7a = state["winners"].get("7a", {}).get(sample, {})
        top_kernels = winners_7a.get("__top5_kernels__")
        shape = winners_7a.get("morph_shape", "ellipse")
        if not top_kernels:
            return []
        return phase_defs.phase7b_overrides_for_sample(top_kernels, shape)

    if phase == "10":
        prior = prior_winner_for(state, "10", sample)
        if prior is None:
            return []
        return phase_defs.phase10_overrides_for_sample(prior)

    raise ValueError(f"unknown phase {phase!r}")


def prior_winner_for(
    state: dict[str, Any],
    phase: str,
    sample: str,
) -> dict[str, Any] | None:
    """Walk back through PHASE_ORDER and return the most recent
    per-sample winner that exists in state['winners']. None for Phase 1."""
    if phase == "1":
        return None
    idx = PHASE_ORDER.index(phase)
    for prev in PHASE_ORDER[idx - 1:: -1]:
        winners = state["winners"].get(prev, {})
        if sample in winners:
            return winners[sample]
    return None


def best_trial_per_sample(
    state: dict[str, Any],
    phase: str,
    sample: str,
) -> dict[str, Any] | None:
    """Across this phase's done trials for this sample, return the trial
    with the highest IoU mean. Ties broken by lowest runtime."""
    candidates = [
        v for v in state["trials_done"].values()
        if v["phase"] == phase and v["sample"] == sample and v.get("iou_mean") is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda v: (v.get("iou_mean", -1.0), -v.get("runtime_s", 1e9)))


def winner_flags_for_sample(
    state: dict[str, Any],
    phase: str,
    sample: str,
) -> dict[str, Any] | None:
    """Look up the winning trial for a (phase, sample) and return the
    full flag dict (not just the overrides)."""
    best = best_trial_per_sample(state, phase, sample)
    if best is None:
        return None
    return best["flags"]


def update_winners(
    state: dict[str, Any],
    phase: str,
    samples: list[str],
) -> None:
    state["winners"].setdefault(phase, {})
    for sample in samples:
        winner = winner_flags_for_sample(state, phase, sample)
        if winner is not None:
            state["winners"][phase][sample] = winner

    if phase == "7a":
        # Compute the top-5 kernels per sample for Phase 7b.
        for sample in samples:
            sample_done = [
                v for v in state["trials_done"].values()
                if v["phase"] == phase and v["sample"] == sample and v.get("iou_mean") is not None
            ]
            ranked = sorted(sample_done, key=lambda v: -v.get("iou_mean", 0.0))
            kernels: list[int] = []
            for entry in ranked:
                k = entry["flags"].get("morph_kernel")
                if isinstance(k, int) and k not in kernels:
                    kernels.append(k)
                if len(kernels) >= 5:
                    break
            shape = ranked[0]["flags"].get("morph_shape", "ellipse") if ranked else "ellipse"
            state["winners"][phase].setdefault(sample, {})
            state["winners"][phase][sample]["__top5_kernels__"] = kernels
            state["winners"][phase][sample]["morph_shape"] = shape


def write_phase_outputs(
    results_dir: Path,
    phase: str,
    state: dict[str, Any],
    samples: list[str],
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    # Per-trial CSV for this phase (subset of all_trials.csv).
    csv_path = results_dir / f"phase{phase}_all_trials.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "trial_id", "sample", "ok", "iou_mean", "dice_mean",
            "boundary_f1_mean", "box_iou_mean", "runtime_s", "error",
            "flags_json",
        ])
        for v in state["trials_done"].values():
            if v["phase"] != phase:
                continue
            writer.writerow([
                v["trial_id"], v["sample"], v["ok"],
                v.get("iou_mean", ""), v.get("dice_mean", ""),
                v.get("boundary_f1_mean", ""), v.get("box_iou_mean", ""),
                v.get("runtime_s", ""), v.get("error", "") or "",
                json.dumps(v.get("flags", {})),
            ])
        for v in state["trials_failed"]:
            if v["phase"] != phase:
                continue
            writer.writerow([
                v["trial_id"], v["sample"], False,
                "", "", "", "", v.get("runtime_s", ""),
                v.get("error", "") or "",
                json.dumps(v.get("flags", {})),
            ])

    # Per-sample winner JSON.
    winners_payload = {
        sample: state["winners"].get(phase, {}).get(sample)
        for sample in samples
    }
    out_path = results_dir / f"phase{phase}_per_video_best.json"
    out_path.write_text(json.dumps(winners_payload, indent=2) + "\n")


def run_phase(
    phase: str,
    state: dict[str, Any],
    args: argparse.Namespace,
    state_path: Path,
    log_root: Path,
    runs_root: Path,
    csv_path: Path,
) -> bool:
    """Execute one phase's trials. Returns True on success, False if the
    failure budget was exceeded."""
    samples = list(phase_defs.PHASE_SAMPLES[phase])
    if args.smoke:
        samples = samples[:2]

    trials: list[TrialConfig] = []
    for sample in samples:
        prior = prior_winner_for(state, phase, sample)
        overrides_list = overrides_for_phase(phase, sample, state)
        if args.smoke:
            overrides_list = overrides_list[:3]
        video_path = video_path_for(args.videos_dir, sample)
        sample_trials = phase_defs.make_trials(
            phase, sample, video_path, prior, overrides_list,
        )
        trials.extend(sample_trials)

    pending: list[TrialConfig] = []
    for trial in trials:
        if trial.trial_id in state["trials_done"]:
            continue
        if not args.retry_failed and any(
            t["trial_id"] == trial.trial_id for t in state["trials_failed"]
        ):
            continue
        pending.append(trial)

    if not pending:
        print(f"  phase {phase}: nothing to do (all trials already in state)",
              flush=True)
        update_winners(state, phase, samples)
        write_phase_outputs(args.results_dir, phase, state, samples)
        return True

    print(f"  phase {phase}: {len(pending)} pending trials across {len(samples)} samples",
          flush=True)

    completed = 0
    last_save = time.monotonic()
    gpu_counter = Value("i", 0)
    pool_kwargs: dict[str, Any] = {"max_workers": args.workers}
    if args.gpus > 0:
        pool_kwargs["initializer"] = _gpu_init
        pool_kwargs["initargs"] = (gpu_counter, args.gpus)
    with ProcessPoolExecutor(**pool_kwargs) as pool:
        futures = {
            pool.submit(
                _worker_run_trial,
                trial_to_payload(trial),
                str(args.binary),
                str(args.labels),
                str(runs_root),
                str(log_root),
                args.timeout,
            ): trial
            for trial in pending
        }
        for future in as_completed(futures):
            result = future.result()
            completed += 1

            # Persist into the canonical state.
            if result["ok"]:
                state["trials_done"][result["trial_id"]] = result
            else:
                state["trials_failed"].append(result)

            all_trials_csv_append(csv_path, result)

            # Save state every 30 s of wall time to bound recovery loss.
            if time.monotonic() - last_save > 30.0:
                save_state(state_path, state)
                last_save = time.monotonic()

            iou = result.get("iou_mean")
            iou_str = f"IoU={iou:.4f}" if iou is not None else f"FAIL ({result.get('error','?')})"
            print(f"    [{completed:>4}/{len(pending)}] {result['trial_id']}: {iou_str}  "
                  f"({result.get('runtime_s', 0):.1f}s)", flush=True)

            if len(state["trials_failed"]) > args.failure_budget:
                print(f"  phase {phase}: failure budget {args.failure_budget} exceeded "
                      f"({len(state['trials_failed'])} failures). Aborting.", flush=True)
                save_state(state_path, state)
                return False

    save_state(state_path, state)
    update_winners(state, phase, samples)
    write_phase_outputs(args.results_dir, phase, state, samples)
    save_state(state_path, state)
    return True


def write_summary(results_dir: Path, state: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# VRIFA ablation v2 summary")
    lines.append("")
    lines.append(f"- host: {state.get('host', '?')}")
    lines.append(f"- git rev: {state.get('git_rev', '?')}")
    lines.append(f"- started: {state.get('started_at', '?')}")
    lines.append(f"- phases done: {state['phases_done']}")
    lines.append(f"- trials done: {len(state['trials_done'])}")
    lines.append(f"- trials failed: {len(state['trials_failed'])}")
    lines.append("")
    lines.append("## Per-phase per-sample IoU at chained best")
    lines.append("")
    lines.append("| phase | sample | iou_mean | runtime_s |")
    lines.append("|---|---|---:|---:|")
    for phase in PHASE_ORDER:
        for sample, flags in (state["winners"].get(phase, {})).items():
            best_done = None
            for v in state["trials_done"].values():
                if v["phase"] == phase and v["sample"] == sample:
                    if best_done is None or v.get("iou_mean", -1) > best_done.get("iou_mean", -1):
                        best_done = v
            if best_done is None:
                continue
            lines.append(
                f"| {phase} | {sample} | "
                f"{best_done.get('iou_mean', float('nan')):.4f} | "
                f"{best_done.get('runtime_s', 0):.1f} |"
            )
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    state_path = args.state_dir / "state.json"
    log_root = args.state_dir / "log" / "failed"
    runs_root = args.state_dir / "runs"
    csv_path = args.state_dir / "all_trials.csv"

    args.state_dir.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    all_trials_csv_init(csv_path)

    if args.lean:
        global phase_defs
        phase_defs = phase_defs_lean
        print("ablation v2: LEAN grids active (phases_lean)", flush=True)

    if not args.binary.exists():
        print(f"FATAL: vrifa binary missing at {args.binary}", file=sys.stderr)
        return 1
    if not args.labels.exists():
        print(f"FATAL: labels missing at {args.labels}", file=sys.stderr)
        return 1

    state = load_state(state_path)
    print(f"loaded state: {len(state['trials_done'])} done, "
          f"{len(state['trials_failed'])} failed, "
          f"{len(state['phases_done'])} phases done", flush=True)

    # Precedence: --only-phase > --phase-order > --start-phase > default
    if args.only_phase:
        if args.only_phase not in PHASE_ORDER:
            print(f"FATAL: unknown --only-phase {args.only_phase!r}", file=sys.stderr)
            return 1
        phase_iter = [args.only_phase]
    elif args.phase_order:
        custom = [p.strip() for p in args.phase_order.split(",") if p.strip()]
        unknown = [p for p in custom if p not in PHASE_ORDER]
        if unknown:
            print(f"FATAL: unknown phases in --phase-order: {unknown}", file=sys.stderr)
            return 1
        phase_iter = custom
        if args.start_phase:
            if args.start_phase not in custom:
                print(f"FATAL: --start-phase {args.start_phase!r} not in --phase-order",
                      file=sys.stderr)
                return 1
            phase_iter = phase_iter[phase_iter.index(args.start_phase):]
        print(f"phase order overridden to: {phase_iter}", flush=True)
    elif args.start_phase:
        if args.start_phase not in PHASE_ORDER:
            print(f"FATAL: unknown --start-phase {args.start_phase!r}", file=sys.stderr)
            return 1
        phase_iter = PHASE_ORDER[PHASE_ORDER.index(args.start_phase):]
    else:
        phase_iter = PHASE_ORDER

    overall_t0 = time.monotonic()
    for phase in phase_iter:
        if phase in state["phases_done"]:
            print(f"SKIP phase {phase} (already in state)", flush=True)
            continue
        print(f"\n=== phase {phase} ===", flush=True)
        ok = run_phase(phase, state, args, state_path, log_root, runs_root, csv_path)
        if not ok:
            print(f"\nphase {phase} aborted -- failure budget exceeded", flush=True)
            write_summary(args.results_dir, state)
            return 2
        state["phases_done"].append(phase)
        save_state(state_path, state)
        elapsed = time.monotonic() - overall_t0
        print(f"  phase {phase} complete  (cumulative wall: {elapsed/60:.1f} min)",
              flush=True)

    write_summary(args.results_dir, state)
    print("\nablation complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
