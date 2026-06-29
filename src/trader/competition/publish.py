"""Write the snapshot to a target — a local dir (dry run) or `s3://…/competition` (CDN).

Reuses the project publisher (`remote_train.publish`). The target is the bucket/prefix ROOT; this
module writes under a `competition/` sub-prefix so it can NEVER collide with the live agent's
`trading/*` files. No-cache so the CDN always serves fresh (like the live `trading/*` feeds).
"""

from __future__ import annotations

import json
import os

from remote_train.publish import join, put_bytes

PREFIX = "competition"
CLOUDFRONT_DIST = "E14F268NIY6WLZ"   # data.alexlouis.dev distribution
# Rolling files that change every capture -> invalidate so the CDN serves them fresh. Immutable
# per-hour snapshot archives (snapshots/<id>/) are never invalidated.
ROLLING_PATHS = [f"/{PREFIX}/leaderboard.json", f"/{PREFIX}/manifest.json",
                 f"/{PREFIX}/series.json", f"/{PREFIX}/snapshots/index.json",
                 f"/{PREFIX}/wallets/*"]


def _put(target: str, rel: str, obj) -> str:
    return put_bytes(join(target, f"{PREFIX}/{rel}"),
                     json.dumps(obj, separators=(",", ":")).encode("utf-8"),
                     content_type="application/json", cache_control="no-cache")


def write_outputs(leaderboard: dict, payloads: dict[str, dict], target: str, *, log=print) -> str:
    """Publish `competition/leaderboard.json`, `competition/manifest.json`, and a
    `competition/wallets/<addr>.json` per participant. Returns the target."""
    _put(target, "leaderboard.json", leaderboard)
    _put(target, "manifest.json", {
        "generated": leaderboard["generated"],
        "metric": leaderboard.get("metric"),
        "n_participants": leaderboard["n_participants"],
        "total_equity_usd": leaderboard.get("total_equity_usd"),
        "wallets": [r["wallet"] for r in leaderboard["rows"]],
    })
    for w, payload in payloads.items():
        _put(target, f"wallets/{w}.json", payload)
    log(f"  wrote leaderboard + manifest + {len(payloads)} wallet files -> {target}/{PREFIX}/")
    return target


def mirror_to_cdn(comp_dir: str, s3_root: str, rel_paths: list[str], *, log=print) -> None:
    """Upload the given files (relative to the local `competition/` dir) to `s3_root/competition/`,
    no-cache. `rel_paths` should be the latest board + wallets + the new snapshot + the rolling
    indexes — old immutable snapshots already on the CDN are not re-pushed."""
    n = 0
    for rel in rel_paths:
        local = os.path.join(comp_dir, rel.replace("/", os.sep))
        if not os.path.isfile(local):
            continue
        with open(local, "rb") as f:
            put_bytes(join(s3_root, f"{PREFIX}/{rel}"), f.read(),
                      content_type="application/json", cache_control="no-cache")
        n += 1
    log(f"  published {n} files -> {s3_root}/{PREFIX}/")


def invalidate_cdn(dist_id: str = CLOUDFRONT_DIST, paths: list[str] | None = None, *, log=print) -> None:
    """Best-effort CloudFront invalidation of the rolling paths (immutable snapshots untouched).
    A perms/quirk failure warns but never breaks the capture."""
    from remote_train.publish import invalidate_cloudfront  # noqa: PLC0415
    try:
        inv = invalidate_cloudfront(dist_id, paths or ROLLING_PATHS)
        log(f"  cloudfront invalidation {inv} for {len(paths or ROLLING_PATHS)} rolling paths")
    except Exception as e:  # noqa: BLE001
        log(f"  cloudfront invalidation skipped: {type(e).__name__} {str(e)[:80]}")
