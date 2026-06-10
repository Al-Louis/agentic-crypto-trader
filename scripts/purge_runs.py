"""Remove specific runs from the published Apentic manifest by id-prefix (run on the DESKTOP —
needs the S3 write creds in .env). build_ledger / the frontend read manifest.json, so dropping a
run's manifest entry removes it from the leaderboard and the dashboard (the S3 object bytes are
left orphaned-but-unlisted, which is harmless).

Unlike the stale scripts/clean_manifest.py — whose hardcoded keep-list would now DROP every
ppo-event-* / ppo-event-sel-* sweep — this drops ONLY the prefixes you explicitly name, so it is
safe to run against the live ledger. Dry-run by default; output kept tiny (a count + the dropped
ids) so it is safe over the tailnet.

    python scripts/purge_runs.py ppo-shakedown            # show what would drop (dry-run)
    python scripts/purge_runs.py ppo-shakedown --apply    # publish the pruned manifest + invalidate
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trader import config  # noqa: E402

publish = importlib.import_module("remote_train.publish")  # submodule, not the re-exported fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prefixes", nargs="+", help="run-id prefixes to drop (e.g. ppo-shakedown)")
    p.add_argument("--apply", action="store_true", help="actually publish (default: dry-run)")
    p.add_argument("--target", default=None, help="default: env APENTIC_PUBLISH_TARGET")
    args = p.parse_args()
    config.load_dotenv()
    target = args.target or config.get("APENTIC_PUBLISH_TARGET")
    if not target:
        raise SystemExit("no publish target (APENTIC_PUBLISH_TARGET)")
    uri = f"{target}/manifest.json"

    man = json.loads(publish.get_bytes(uri))
    dropped = [e.get("id", "") for e in man
               if any(e.get("id", "").startswith(pfx) for pfx in args.prefixes)]
    kept = [e for e in man if e.get("id", "") not in dropped]

    print(f"manifest {len(man)} -> keep {len(kept)}, drop {len(dropped)}: {', '.join(dropped) or '(none)'}")
    if not args.apply:
        print("[dry-run] manifest untouched")
        return
    if not dropped:
        print("nothing to drop")
        return
    publish.put_bytes(uri, json.dumps(kept, indent=2).encode(), "application/json",
                      "no-cache, max-age=0")
    dist = config.get("APENTIC_CLOUDFRONT_DIST_ID")
    if dist:
        publish.invalidate_cloudfront(dist, ["/manifest.json"], caller_reference="purge-runs")
    print(f"published pruned manifest ({len(kept)} entries) + invalidated /manifest.json")


if __name__ == "__main__":
    main()
