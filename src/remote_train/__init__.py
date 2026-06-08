"""remote_train — a small, generic remote-job orchestrator.

Dispatch a command to a host (local or SSH/Tailscale), capture its artifact directory,
poll progress, and publish the result. It knows nothing about ML or trading — that is the
point: it is designed to be lifted into a standalone, reusable project once a second use-case
validates the interface (vault "Remote Capabilities"). The trading-specific bundle format
lives in `trader.report`, never here.

Hard rule: this package must never import `trader.*` (enforced by tests).
"""

from __future__ import annotations

from remote_train.executor import LocalExecutor, SSHExecutor
from remote_train.progress import read_progress, write_progress
from remote_train.publish import (
    get_bytes,
    invalidate_cloudfront,
    join,
    publish,
    put_bytes,
)
from remote_train.runner import list_runs, status, submit
from remote_train.spec import JobSpec, RunStatus

__all__ = [
    "JobSpec", "RunStatus",
    "LocalExecutor", "SSHExecutor",
    "submit", "status", "list_runs",
    "publish", "get_bytes", "put_bytes", "join", "invalidate_cloudfront",
    "write_progress", "read_progress",
]
