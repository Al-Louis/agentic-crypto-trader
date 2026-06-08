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


def put_bytes(uri: str, data: bytes, content_type: str | None = None,
              cache_control: str | None = None) -> str:
    """Write bytes to an object (local file or s3/r2 key)."""
    if _is_s3(uri):
        bucket, key = _s3_split(uri)
        extra: dict[str, str] = {}
        if content_type:
            extra["ContentType"] = content_type
        if cache_control:
            extra["CacheControl"] = cache_control
        _s3_client().put_object(Bucket=bucket, Key=key, Body=data, **extra)
        return uri
    path = _local_path(uri)                          # cache_control is a CDN concern; n/a locally
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def publish(local_dir: Path | str, target: str, cache_control: str | None = None) -> str:
    """Upload every file under `local_dir` into `target` (local dir or s3/r2 prefix)."""
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise NotADirectoryError(f"bundle dir not found: {local_dir}")
    for src in sorted(local_dir.rglob("*")):
        if src.is_file():
            rel = src.relative_to(local_dir).as_posix()
            put_bytes(join(target, rel), src.read_bytes(),
                      content_type="application/json" if src.suffix == ".json" else None,
                      cache_control=cache_control)
    return target


def invalidate_cloudfront(distribution_id: str, paths: list[str],
                          caller_reference: str | None = None) -> str:
    """Invalidate CloudFront `paths` so freshly-published objects are served immediately.

    AWS-specific (the CDN in front of the S3 publish target). boto3 is lazy/optional. Returns
    the invalidation id. `caller_reference` must be unique per request — defaults to a clock
    stamp (this runs as a normal script, not a workflow, so the clock is available).
    """
    try:
        import boto3  # noqa: PLC0415 — optional dependency
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("CloudFront invalidation needs boto3 — install the 'remote' extra") from exc
    import time  # noqa: PLC0415
    ref = caller_reference or f"apentic-{time.time_ns()}"
    resp = boto3.client("cloudfront").create_invalidation(
        DistributionId=distribution_id,
        InvalidationBatch={"Paths": {"Quantity": len(paths), "Items": list(paths)},
                           "CallerReference": ref})
    return resp["Invalidation"]["Id"]
