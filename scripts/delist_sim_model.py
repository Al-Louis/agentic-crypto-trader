"""De-list one run from the published `simulated_models.json` index (the Simulations page's model
picker). The S3 publisher can PUT but not byte-delete ([[apentic-publisher-no-delete]]); this rewrites
the INDEX without the entry, so the frontend stops loading the bundle (the bytes stay, just unlisted).

Use it to pull a throwaway / closed-branch run off the dashboard (e.g. a fixed-universe experiment
whose bundle has dataless-token assets that crash the page). DESKTOP-side (needs the S3 write creds +
APENTIC_PUBLISH_TARGET in .env).

    python scripts/delist_sim_model.py --id ppo-event-rdLe4-eff-2345fd6-s1 [--dry-run]
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from trader import config  # noqa: E402

publish = importlib.import_module("remote_train.publish")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True, help="run id to remove from simulated_models.json")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    config.load_dotenv()
    target = config.get("APENTIC_PUBLISH_TARGET")
    if not target:
        raise SystemExit("no APENTIC_PUBLISH_TARGET")
    uri = f"{target}/simulated_models.json"

    idx = json.loads(publish.get_bytes(uri))
    kept = [e for e in idx if e.get("id") != args.id]
    dropped = len(idx) - len(kept)
    print(f"drop {dropped} ({args.id}) | before {len(idx)} -> after {len(kept)}")
    for e in sorted(kept, key=lambda x: x.get("generated", ""), reverse=True):
        print(f"  keep {e.get('id')}")
    if dropped == 0:
        print("NOT FOUND — nothing changed"); return
    if args.dry_run:
        print("[dry-run] index untouched"); return

    publish.put_bytes(uri, json.dumps(kept, indent=2).encode(), "application/json", "no-cache, max-age=0")
    dist = config.get("APENTIC_CLOUDFRONT_DIST_ID")
    if dist:
        publish.invalidate_cloudfront(dist, ["/simulated_models.json"], caller_reference="delist-sim")
    print(f"published de-listed index ({len(kept)} entries) + invalidated /simulated_models.json")


if __name__ == "__main__":
    main()
