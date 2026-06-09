"""The run store + submit/status — the fire-and-poll surface the MCP tools wrap.

A run is a directory under the store: ``<store>/<run_id>/`` holding ``spec.json``,
``status.json``, ``run.log``, and an ``artifacts/`` bundle the job writes. State lives on
disk (not in memory) so ``status`` works across processes — the MCP server can poll a run a
different process launched.

This first cut runs `submit` **synchronously** (it returns the final status): right for the
seconds-long export demo and for CI. Long RL runs will add a background variant that returns
immediately and relies on the same on-disk status/`progress.json` for polling.
"""

from __future__ import annotations

import json
from pathlib import Path

from remote_train.executor import LocalExecutor
from remote_train.progress import read_progress
from remote_train.spec import JobSpec, RunStatus

DEFAULT_STORE = Path("runs")


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _next_run_id(store: Path, name: str) -> str:
    existing = sorted(store.glob(f"{name}-*")) if store.exists() else []
    return f"{name}-{len(existing) + 1:03d}"


def submit(spec: JobSpec, executor=None, store: Path | str = DEFAULT_STORE,
           run_id: str | None = None) -> RunStatus:
    """Launch a job and return its final `RunStatus` (synchronous).

    Creates the run dir, records the spec, runs the command via `executor` (default
    `LocalExecutor`), and writes a terminal status. Artifacts land in
    ``<run_dir>/<artifact_subdir>/``.
    """
    executor = executor or LocalExecutor()
    store = Path(store)
    run_id = run_id or _next_run_id(store, spec.name)
    run_dir = store / run_id
    artifact_dir = run_dir / spec.artifact_subdir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    _write_json(run_dir / "spec.json", spec.to_dict())
    _write_json(run_dir / "status.json",
                {"run_id": run_id, "state": "running", "executor": executor.name})
    try:
        rc = executor.run(spec, run_dir, artifact_dir, log_path)
        state = "succeeded" if rc == 0 else "failed"
        _write_json(run_dir / "status.json",
                    {"run_id": run_id, "state": state, "executor": executor.name,
                     "returncode": rc})
    except Exception as exc:  # noqa: BLE001 — record any launch failure as a terminal state
        _write_json(run_dir / "status.json",
                    {"run_id": run_id, "state": "failed", "executor": executor.name,
                     "error": str(exc)})
        raise
    return status(run_id, store)


def status(run_id: str, store: Path | str = DEFAULT_STORE) -> RunStatus:
    """Read a run's current status, including the latest `progress.json` it has emitted."""
    store = Path(store)
    run_dir = store / run_id
    raw = _read_json(run_dir / "status.json") or {"run_id": run_id, "state": "unknown"}
    artifact_dir = run_dir / (_read_json(run_dir / "spec.json") or {}).get("artifact_subdir", "artifacts")
    return RunStatus(
        run_id=run_id,
        state=raw.get("state", "unknown"),
        executor=raw.get("executor"),
        returncode=raw.get("returncode"),
        progress=read_progress(artifact_dir),
        artifact_dir=str(artifact_dir) if artifact_dir.exists() else None,
        log_path=str(run_dir / "run.log"),
        error=raw.get("error"),
    )


def list_runs(store: Path | str = DEFAULT_STORE) -> list[str]:
    store = Path(store)
    if not store.exists():
        return []
    return sorted(p.name for p in store.iterdir() if (p / "status.json").exists())


def submit_background(spec: JobSpec, executor=None, store: Path | str = DEFAULT_STORE,
                      run_id: str | None = None) -> RunStatus:
    """Launch a job **detached** and return immediately (fire-and-poll for long RL runs).

    The job self-reports via ``progress.json`` (and self-publishes its bundle); `poll` reads
    that + process liveness. Pass the same `executor` to `poll` to read remote progress.
    """
    executor = executor or LocalExecutor()
    store = Path(store)
    run_id = run_id or _next_run_id(store, spec.name)
    run_dir = store / run_id
    artifact_dir = run_dir / spec.artifact_subdir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    _write_json(run_dir / "spec.json", spec.to_dict())
    handle = executor.launch(spec, run_dir, artifact_dir, run_dir / "run.log")
    _write_json(run_dir / "handle.json", handle)
    _write_json(run_dir / "status.json",
                {"run_id": run_id, "state": "running", "executor": executor.name})
    return poll(run_id, store=store, executor=executor)


def poll(run_id: str, store: Path | str = DEFAULT_STORE, executor=None) -> RunStatus:
    """Current status of a backgrounded run: terminal `progress.state` wins, else liveness."""
    store = Path(store)
    run_dir = store / run_id
    raw = _read_json(run_dir / "status.json") or {"run_id": run_id, "state": "unknown"}
    handle = _read_json(run_dir / "handle.json")
    artifact_dir = run_dir / (_read_json(run_dir / "spec.json") or {}).get("artifact_subdir", "artifacts")

    state = raw.get("state", "unknown")
    progress = read_progress(artifact_dir)          # local fallback
    if handle is not None and executor is not None:
        progress = executor.read_progress(handle) or progress
        pstate = (progress or {}).get("state")
        if pstate in ("complete", "done", "succeeded"):
            state = "succeeded"
        elif pstate in ("failed", "error"):
            state = "failed"
        elif executor.is_alive(handle):
            state = "running"
        else:
            state = "failed"                        # process gone without a terminal state
        _write_json(run_dir / "status.json", {**raw, "state": state})

    return RunStatus(run_id=run_id, state=state, executor=raw.get("executor"),
                     returncode=raw.get("returncode"), progress=progress,
                     artifact_dir=str(artifact_dir) if artifact_dir.exists() else None,
                     log_path=str(run_dir / "run.log"), error=raw.get("error"))
