"""Publish a local bundle directory to a target the frontend can read.

Targets:
  - a local path / ``file://`` → recursive merge-copy (the dev + CI path, used now)
  - ``s3://bucket/prefix`` (also serves Cloudflare R2 via its S3-compatible API) → upload

The S3/R2 path imports boto3 lazily so the dependency is optional — `pip install
'agentic-crypto-trader[remote]'`. R2 credentials/endpoint come from env
(``REMOTE_TRAIN_S3_ENDPOINT`` for the R2 account endpoint, plus AWS-style keys); never
hard-coded (vault "Remote Capabilities" / CLAUDE.md secrets rule).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from urllib.parse import urlparse


def publish(local_dir: Path | str, target: str) -> str:
    """Sync every file under `local_dir` to `target`. Returns the resolved target URI."""
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise NotADirectoryError(f"bundle dir not found: {local_dir}")
    scheme = urlparse(target).scheme
    if scheme in ("s3", "r2"):
        return _publish_s3(local_dir, target)
    return _publish_local(local_dir, target)


def _publish_local(local_dir: Path, target: str) -> str:
    dest = Path(target[len("file://"):] if target.startswith("file://") else target)
    dest.mkdir(parents=True, exist_ok=True)
    for src in local_dir.rglob("*"):
        if src.is_file():
            out = dest / src.relative_to(local_dir)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
    return str(dest)


def _publish_s3(local_dir: Path, target: str) -> str:
    try:
        import boto3  # noqa: PLC0415 — optional dependency, imported on demand
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "publishing to s3/r2 needs boto3 — install the 'remote' extra "
            "(pip install 'agentic-crypto-trader[remote]')"
        ) from exc

    parsed = urlparse(target)
    bucket, prefix = parsed.netloc, parsed.path.lstrip("/")
    client = boto3.client("s3", endpoint_url=os.environ.get("REMOTE_TRAIN_S3_ENDPOINT") or None)
    for src in local_dir.rglob("*"):
        if src.is_file():
            key = f"{prefix.rstrip('/')}/{src.relative_to(local_dir).as_posix()}".lstrip("/")
            client.upload_file(str(src), bucket, key,
                               ExtraArgs={"ContentType": "application/json"}
                               if src.suffix == ".json" else None)
    return f"s3://{bucket}/{prefix}"
