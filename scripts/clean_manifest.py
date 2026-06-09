"""Tidy the published Apentic manifest (run on the DESKTOP — needs the S3 write creds in .env).

Default action: keep the reward-sweep runs (`ppo-<mode>-s<seed>`) and any non-`ppo-` entries
(the demo heuristics), drop every other `ppo-*` (throwaway exploration runs), and rewrite each
sweep entry's `model_name` to include its seed so the frontend list is legible.

    python scripts/clean_manifest.py            # apply
    python scripts/clean_manifest.py --dry-run  # show what would change, touch nothing
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from remote_train import publish  # noqa: E402
from trader import config  # noqa: E402

SWEEP = re.compile(r"^ppo-(sharpe|giveback|realized|turnover)-s(\d+)$")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--target", default=None, help="default: env APENTIC_PUBLISH_TARGET")
    args = p.parse_args()
    config.load_dotenv()
    target = args.target or config.get("APENTIC_PUBLISH_TARGET")
    if not target:
        raise SystemExit("no publish target (APENTIC_PUBLISH_TARGET)")
    uri = f"{target}/manifest.json"

    man = json.loads(publish.get_bytes(uri))
    kept, dropped = [], []
    for e in man:
        rid = e.get("id", "")
        m = SWEEP.match(rid)
        if m:                                              # a sweep run — keep + label with the seed
            mode, seed = m.group(1), m.group(2)
            name = e.get("model_name", "")
            if f"s{seed}" not in name:
                e["model_name"] = (re.sub(r"\s*\(", f" s{seed} (", name, count=1)
                                   if "(" in name else f"{name} s{seed}")
            kept.append(e)
        elif rid.startswith("ppo-"):                       # throwaway exploration run — drop
            dropped.append(rid)
        else:                                              # demos / anything else — keep as-is
            kept.append(e)

    kept.sort(key=lambda e: e.get("id", ""))
    print(f"keep {len(kept)}, drop {len(dropped)}: {', '.join(dropped) or '(none)'}")
    for e in kept:
        print(f"  {e.get('id'):20} {e.get('model_name')}")
    if args.dry_run:
        print("\n[dry-run] manifest untouched")
        return

    publish.put_bytes(uri, json.dumps(kept, indent=2).encode(), "application/json",
                      "no-cache, max-age=0")
    dist = config.get("APENTIC_CLOUDFRONT_DIST_ID")
    if dist:
        publish.invalidate_cloudfront(dist, ["/manifest.json"], caller_reference="clean-manifest")
    print(f"\npublished cleaned manifest ({len(kept)} entries) + invalidated /manifest.json")


if __name__ == "__main__":
    main()
