"""Atomic checkpoint and resume index for the ablation harness.

state.json shape:
{
  "started_at": ISO timestamp,
  "host": str,
  "git_rev": str,
  "phases_done": ["1", "2", ...],
  "trials_done": {trial_id: {phase, sample, iou_mean, runtime_s}},
  "trials_failed": [{trial_id, phase, sample, error, runtime_s, flags}],
  "winners": {phase: {sample: {flags...}}}
}
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def initial_state() -> dict[str, Any]:
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
        ).strip()
    except Exception:
        rev = "unknown"
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "git_rev": rev,
        "phases_done": [],
        "trials_done": {},
        "trials_failed": [],
        "winners": {},
    }


def load_state(state_path: Path) -> dict[str, Any]:
    if state_path.exists():
        return json.loads(state_path.read_text())
    return initial_state()


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    _atomic_write_json(state_path, state)
