"""The live-telemetry seam: a job appends progress to ``progress.json`` in its artifact dir.

A training script calls `write_progress(artifact_dir, episode=..., reward=...)` as it runs;
`status()` (and the dashboard) read the same flat file. A file, not a socket — resilient to
crashes, pollable across hosts after rsync, and trivially decoupled (the job depends only on
this tiny module, not on the orchestrator).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROGRESS_FILE = "progress.json"


def write_progress(artifact_dir: Path | str, *, history_key: str | None = None,
                   **fields: Any) -> None:
    """Update the progress file with `fields`.

    If `history_key` is given, the call's fields are also appended to a list under that key
    (e.g. ``history_key="curve"`` to accumulate a reward curve), while the top level keeps
    the latest scalar values.
    """
    path = Path(artifact_dir) / PROGRESS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    data = read_progress(artifact_dir) or {}
    data.update(fields)
    if history_key is not None:
        data.setdefault(history_key, []).append(dict(fields))
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_progress(artifact_dir: Path | str) -> dict | None:
    path = Path(artifact_dir) / PROGRESS_FILE
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
