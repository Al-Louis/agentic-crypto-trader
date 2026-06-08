"""Job + run data shapes for the remote-train orchestrator.

Deliberately generic: a `JobSpec` is "run this command on a host and capture an output
directory" — it knows nothing about ML, RL, or trading. That decoupling is what lets this
package be lifted into its own repo later (see the vault note "Remote Capabilities").
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Placeholders substituted into entrypoint argv and env values at launch time, so a job
# learns where to write its artifacts without the orchestrator knowing what they are.
RUN_DIR = "{run_dir}"
ARTIFACT_DIR = "{artifact_dir}"


@dataclass
class JobSpec:
    """A host-agnostic unit of work.

    Args:
        name: short label; run ids are derived from it (`<name>-001`).
        entrypoint: argv to execute (e.g. ``["python", "train.py", "--out", "{artifact_dir}"]``).
            ``{run_dir}`` / ``{artifact_dir}`` placeholders are filled at launch.
        workdir: working directory on the host (usually the repo root).
        repo_ref: optional git ref to check out before running (None → run as-is).
        env: extra environment variables (values support the same placeholders).
        artifact_subdir: where the job writes its output bundle, relative to the run dir.
        resources: advisory hints (e.g. ``{"gpu": 1}``) for schedulers; not enforced here.
    """

    name: str
    entrypoint: list[str]
    workdir: str = "."
    repo_ref: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    artifact_subdir: str = "artifacts"
    resources: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunStatus:
    """A snapshot of one run, assembled from its on-disk status + progress files."""

    run_id: str
    state: str                       # "running" | "succeeded" | "failed"
    executor: str | None = None
    returncode: int | None = None
    progress: dict[str, Any] | None = None   # latest progress.json the job emitted
    artifact_dir: str | None = None
    log_path: str | None = None
    error: str | None = None

    @property
    def done(self) -> bool:
        return self.state in ("succeeded", "failed")

    @property
    def ok(self) -> bool:
        return self.state == "succeeded"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
