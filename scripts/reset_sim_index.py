"""Curate the published `simulated_models.json` index to KEEP ONLY a given set of run ids (de-list the
rest). The S3 publisher can PUT but not byte-delete ([[apentic-publisher-no-delete]]); this rewrites the
INDEX so the Simulations page lists only the kept runs — the dropped bundles' bytes stay in the bucket,
just unreferenced/invisible. One shot vs calling delist_sim_model.py per id. DESKTOP-side (needs
APENTIC_PUBLISH_TARGET + the S3 write creds in .env).

    python scripts/reset_sim_index.py --keep id1,id2,id3 [--dry-run]
    python scripts/reset_sim_index.py --keep "" --dry-run        # would clear the index entirely
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
    p.add_argument("--keep", required=True,
                   help="comma-separated run ids to KEEP; everything else is de-listed (empty = clear all)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    config.load_dotenv()
    target = config.get("APENTIC_PUBLISH_TARGET")
    if not target:
        raise SystemExit("no APENTIC_PUBLISH_TARGET")
    keep = {s.strip() for s in args.keep.split(",") if s.strip()}
    uri = f"{target}/simulated_models.json"

    idx = json.loads(publish.get_bytes(uri))
    kept = [e for e in idx if e.get("id") in keep]
    dropped = [e.get("id") for e in idx if e.get("id") not in keep]
    print(f"index {len(idx)} -> keep {len(kept)} / drop {len(dropped)}")
    for e in kept:
        print(f"  KEEP {e.get('id')}")
    for i in dropped:
        print(f"  drop {i}")
    missing = keep - {e.get("id") for e in kept}
    if missing:
        print(f"  WARN requested-keep not in index (will appear once republished): {sorted(missing)}")
    if args.dry_run:
        print("[dry-run] index untouched")
        return

    publish.put_bytes(uri, json.dumps(kept, indent=2).encode(), "application/json", "no-cache, max-age=0")
    dist = config.get("APENTIC_CLOUDFRONT_DIST_ID")
    if dist:
        publish.invalidate_cloudfront(dist, ["/simulated_models.json"], caller_reference="reset-sim-index")
    print(f"published curated index ({len(kept)} entries) + invalidated /simulated_models.json")


if __name__ == "__main__":
    main()
