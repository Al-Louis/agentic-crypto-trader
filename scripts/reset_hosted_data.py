"""One-shot RESET of the hosted Apentic data to ONLY the live-leaderboard run set.

Keeps the runs the current leaderboard.json references (renamed with an `lb_` prefix so a future
purge can't touch them), and FULL-DELETES every other run object from S3 + drops it from the
manifest. Regenerates manifest.json + leaderboard.json to point at the renamed runs and
invalidates the two index files at the CDN.

Run on the DESKTOP (needs the S3 write creds in .env). **DRY-RUN by default** — prints a tiny
summary (counts + the top-level prefixes it found, so a surprise non-run folder is caught) and
touches nothing. `--apply` executes in a FAIL-SAFE order so the live page is never broken midway:

  1. copy each KEEP run's objects   <id>/*  ->  lb_<id>/*        (non-destructive)
  2. publish new manifest.json (lb_ ids) + leaderboard.json (lb_ labels) + invalidate
     -> the live page now points at lb_ objects, which already exist
  3. delete the old KEEP originals <id>/* AND every other run prefix <id>/*  (the purge)

Stop after (1): page still serves the old manifest. After (2): serves the new one. (3) is pure
orphan cleanup. The keep set is derived from leaderboard.json; the delete set from the LIVE S3
listing (so orphaned objects not in the manifest are swept too).

    python scripts/reset_hosted_data.py                # dry-run (review this first)
    python scripts/reset_hosted_data.py --apply        # execute the reset
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trader import config  # noqa: E402

publish = importlib.import_module("remote_train.publish")

ROOT_FILES = {"manifest.json", "leaderboard.json", "champion.json", "ledger.jsonl"}


def _bucket(target: str) -> str:
    return target.split("://", 1)[1].split("/", 1)[0]


def list_top_level_prefixes(client, bucket: str) -> tuple[list[str], list[str]]:
    """All top-level "folders" (run ids) + the root-level object keys."""
    prefixes: list[str] = []
    roots: list[str] = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Delimiter="/"):
        prefixes += [cp["Prefix"].rstrip("/") for cp in page.get("CommonPrefixes", [])]
        roots += [o["Key"] for o in page.get("Contents", []) if "/" not in o["Key"]]
    return prefixes, roots


def list_keys(client, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys += [o["Key"] for o in page.get("Contents", [])]
    return keys


def transform_leaderboard(lb: dict, generated: str) -> dict:
    """Prefix every config_label + run_id with lb_; refresh totals + generated. Content unchanged."""
    nlb = copy.deepcopy(lb)
    for c in nlb.get("configs", []):
        c["config_label"] = "lb_" + c["config_label"]
        for d in c.get("seeds_detail", []):
            d["run_id"] = "lb_" + d["run_id"]
    nlb["generated"] = generated
    nlb["totals"] = {"runs": sum(c.get("n", 0) for c in nlb.get("configs", [])),
                     "configs": len(nlb.get("configs", []))}
    return nlb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="execute (default: dry-run)")
    ap.add_argument("--target", default=None, help="default: env APENTIC_PUBLISH_TARGET")
    args = ap.parse_args()
    config.load_dotenv()
    target = args.target or config.get("APENTIC_PUBLISH_TARGET")
    if not target:
        raise SystemExit("no publish target (APENTIC_PUBLISH_TARGET)")
    dist = config.get("APENTIC_CLOUDFRONT_DIST_ID")
    bucket = _bucket(target)
    client = publish._s3_client()

    man = json.loads(publish.get_bytes(f"{target}/manifest.json") or b"[]")
    lb = json.loads(publish.get_bytes(f"{target}/leaderboard.json") or b"{}")
    keep = {d["run_id"] for c in lb.get("configs", []) for d in c.get("seeds_detail", [])}

    prefixes, roots = list_top_level_prefixes(client, bucket)
    rename = sorted(p for p in prefixes if p in keep)
    already_lb = sorted(p for p in prefixes if p.startswith("lb_"))
    delete = sorted(p for p in prefixes if p not in keep and not p.startswith("lb_"))
    missing = sorted(keep - set(rename))

    print(f"bucket={bucket}  top-level-prefixes={len(prefixes)}  root-files={sorted(roots)}")
    print(f"KEEP->lb_: {len(rename)} (leaderboard wants {len(keep)})  DELETE: {len(delete)}  "
          f"already-lb_: {len(already_lb)}")
    if missing:
        print(f"!! {len(missing)} leaderboard run(s) have NO S3 prefix: {missing}")
    unexpected = [r for r in roots if r not in ROOT_FILES]
    if unexpected:
        print(f"!! unexpected root objects (review): {unexpected}")
    print(f"sample DELETE: {delete[:6]}{' …' if len(delete) > 6 else ''}")

    if not args.apply:
        print("[dry-run] nothing changed — re-run with --apply to execute")
        return
    if missing:
        raise SystemExit("ABORT: leaderboard runs missing from S3 — refusing to proceed")

    # 1) copy KEEP -> lb_ (non-destructive)
    copied = 0
    for rid in rename:
        for k in list_keys(client, bucket, f"{rid}/"):
            client.copy_object(Bucket=bucket, CopySource={"Bucket": bucket, "Key": k},
                               Key="lb_" + k)
            copied += 1

    # 2) new manifest (lb_ ids only) + leaderboard (lb_ labels) + invalidate the index files
    new_man = sorted(({**e, "id": "lb_" + e["id"]} for e in man if e.get("id") in keep),
                     key=lambda e: e["id"])
    new_lb = transform_leaderboard(lb, datetime.now(timezone.utc).isoformat())
    publish.put_bytes(f"{target}/manifest.json", json.dumps(new_man, indent=2).encode(),
                      "application/json", "no-cache, max-age=0")
    publish.put_bytes(f"{target}/leaderboard.json", json.dumps(new_lb, indent=2).encode(),
                      "application/json", "no-cache, max-age=0")
    if dist:
        publish.invalidate_cloudfront(dist, ["/manifest.json", "/leaderboard.json"],
                                      caller_reference="reset-hosted-data")

    # 3) delete the old KEEP originals + every purged run prefix
    victims = rename + delete
    keys = [k for rid in victims for k in list_keys(client, bucket, f"{rid}/")]
    for i in range(0, len(keys), 1000):
        client.delete_objects(Bucket=bucket,
                              Delete={"Objects": [{"Key": k} for k in keys[i:i + 1000]]})

    print(f"DONE: renamed {len(rename)} runs ({copied} objs copied), deleted {len(delete)} runs + "
          f"{len(rename)} old originals ({len(keys)} objs), manifest now {len(new_man)} entries")


if __name__ == "__main__":
    main()
