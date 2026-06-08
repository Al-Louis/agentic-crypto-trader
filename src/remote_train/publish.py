"""Publish bundles + read/write single objects to a local dir or S3/Cloudflare R2.

Targets are URIs: a local path / ``file://`` → the filesystem; ``s3://bucket/key`` (also
serves R2 via its S3-compatible API) → object storage. The S3 path imports boto3 lazily so
the dependency stays optional — ``pip install 'agentic-crypto-trader[remote]'``. R2
credentials/endpoint come from env (``REMOTE_TRAIN_S3_ENDPOINT`` + AWS-style keys), never
hard-coded (vault "Remote Capabilities" / CLAUDE.md secrets rule).

`get_bytes`/`put_bytes` are single-object I/O (used for the shared manifest's read-merge-write);
`publish` uploads a whole directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

S3_SCHEMES = ("s3", "r2")


def _is_s3(uri: str) -> bool:
    return urlparse(uri).scheme in S3_SCHEMES


def _local_path(uri: str) -> Path:
    return Path(uri[len("file://"):] if uri.startswith("file://") else uri)


def _s3_split(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.lstrip("/")


def _s3_client():
    try:
        import boto3  # noqa: PLC0415 — optional dependency, imported on demand
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "s3/r2 access needs boto3 — install the 'remote' extra "
            "(pip install 'agentic-crypto-trader[remote]')"
        ) from exc
    return boto3.client("s3", endpoint_url=os.environ.get("REMOTE_TRAIN_S3_ENDPOINT") or None)


def join(base: str, rel: str) -> str:
    """Join a target URI with a relative key (forward slashes for s3, OS path for local)."""
    if _is_s3(base):
        return f"{base.rstrip('/')}/{rel.lstrip('/')}"
    return str(_local_path(base) / rel)


def get_bytes(uri: str) -> bytes | None:
    """Read an object's bytes, or None if it doesn't exist."""
    if _is_s3(uri):
        from botocore.exceptions import ClientError  # noqa: PLC0415
        bucket, key = _s3_split(uri)
        try:
            return _s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("NoSuchKey", "404", "NotFound"):
                return None
            raise
    path = _local_path(uri)
    return path.read_bytes() if path.is_file() else None


def put_bytes(uri: str, data: bytes, content_type: str | None = None) -> str:
    """Write bytes to an object (local file or s3/r2 key)."""
    if _is_s3(uri):
        bucket, key = _s3_split(uri)
        extra = {"ContentType": content_type} if content_type else {}
        _s3_client().put_object(Bucket=bucket, Key=key, Body=data, **extra)
        return uri
    path = _local_path(uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def publish(local_dir: Path | str, target: str) -> str:
    """Upload every file under `local_dir` into `target` (local dir or s3/r2 prefix)."""
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise NotADirectoryError(f"bundle dir not found: {local_dir}")
    for src in sorted(local_dir.rglob("*")):
        if src.is_file():
            rel = src.relative_to(local_dir).as_posix()
            put_bytes(join(target, rel), src.read_bytes(),
                      content_type="application/json" if src.suffix == ".json" else None)
    return target
