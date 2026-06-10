"""Repair the published Apentic manifest after the run-id collision (laptop-side; needs AWS creds).

The pre-`ec1e487` trainer published every event run with the same display name
("PPO event-rung sN (1,000,000 steps)"), so the frontend dropdown shows ~22 indistinguishable
entries — and the fixed-env g2b re-run (which OVERWROTE the broken-env bundles at the same ids)
can't be told apart from the invalid vintage. This retro-fits ec1e487's self-describing
`model_name` onto every published portfolio run, rebuilt from each bundle's own `provenance`
block, and prefixes runs whose provenance commit predates the `8ccad69` env fix with
"[INVALID env-bug]". `lb_*` (protected leaderboard set) and runs without provenance are untouched.

Also patches each renamed run's `run_info.json` (the in-page header) to match, then issues one
CloudFront invalidation. The publisher IAM cannot delete — this only puts objects; with
`--delist-invalid` the invalid entries are removed from the MANIFEST ONLY (bundles stay on S3).

  python scripts/repair_manifest.py                 # dry-run: print the rename plan
  python scripts/repair_manifest.py --apply         # write manifest + run_info + invalidate CDN
  python scripts/repair_manifest.py --apply --delist-invalid
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from remote_train import get_bytes, invalidate_cloudfront, join, put_bytes  # noqa: E402
from trader.report.apentic import MANIFEST_CACHE_CONTROL, RUN_CACHE_CONTROL  # noqa: E402

HOST = "https://data.alexlouis.dev"
FIX_SHA = "8ccad69"          # the env exit-stop fix — provenance commits BEFORE this are invalid
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fetch_json(url):
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.load(r)
    except Exception:  # noqa: BLE001 — missing bundle file
        return None


def predates_fix(sha: str) -> bool | None:
    """True if `sha` is an ancestor of the env fix's parent (trained in the broken env);
    None if the commit is unknown to this clone (leave the entry unjudged)."""
    if not sha or sha == "unknown":
        return None
    if subprocess.run(["git", "cat-file", "-t", sha], capture_output=True, cwd=REPO).returncode:
        return None
    r = subprocess.run(["git", "merge-base", "--is-ancestor", sha, f"{FIX_SHA}^"],
                       capture_output=True, cwd=REPO)
    return r.returncode == 0


def self_describing(run_id: str, prov: dict) -> str:
    """ec1e487's model_name format, rebuilt from a published bundle's provenance block."""
    bits = [prov.get("reward_mode", "?")]
    if prov.get("k") is not None and prov.get("universe_mode"):
        bits.append(f"k{prov['k']}/{prov['universe_mode']}")
    if prov.get("dd_lambda") is not None:
        bits.append(f"dd{prov['dd_lambda']}")
    flags = " ".join(bits)
    flags += (" +harvest" if prov.get("harvest_obs") else "") + (" +crash" if prov.get("crash_eval") else "")
    steps = f"{prov.get('timesteps', 0) // 1000}k"
    return f"{run_id} @{prov.get('git_commit', '?')} | {flags} | s{prov.get('seed', '?')} {steps}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="s3://alexlouis-apentic-data")
    p.add_argument("--dist-id", default="E14F268NIY6WLZ")
    p.add_argument("--apply", action="store_true", help="write changes (default: dry-run plan)")
    p.add_argument("--delist-invalid", action="store_true",
                   help="drop pre-fix entries from the manifest (bundles stay on S3)")
    args = p.parse_args()

    manifest = fetch_json(f"{HOST}/manifest.json")
    if not manifest:
        raise SystemExit("could not fetch the manifest")

    out, patched_infos, renamed, delisted = [], [], 0, 0
    for e in manifest:
        rid = e["id"]
        if rid.startswith("lb_"):                      # protected leaderboard set — never touch
            out.append(e)
            continue
        m = fetch_json(f"{HOST}/{rid}/metrics.json") or {}
        prov = m.get("provenance")
        if not prov:
            out.append(e)
            print(f"  keep   {rid:28} (no provenance)")
            continue
        name = self_describing(rid, prov)
        old_vintage = predates_fix(prov.get("git_commit", ""))
        if old_vintage:
            if args.delist_invalid:
                delisted += 1
                print(f"  DELIST {rid:28} @{prov.get('git_commit')} (pre-{FIX_SHA} env bug)")
                continue
            name = f"[INVALID env-bug] {name}"
        if name != e.get("model_name"):
            renamed += 1
            print(f"  rename {rid:28} -> {name}")
            e = {**e, "model_name": name}
            info = fetch_json(f"{HOST}/{rid}/run_info.json")
            if info is not None:
                patched_infos.append((rid, {**info, "model_name": name}))
        out.append(e)

    print(f"\nplan: {renamed} renamed, {delisted} delisted, {len(out)} entries remain"
          + ("" if args.apply else "   (dry-run — pass --apply to write)"))
    if not args.apply:
        return

    for rid, info in patched_infos:
        put_bytes(join(args.target, f"{rid}/run_info.json"), json.dumps(info, indent=2).encode(),
                  content_type="application/json", cache_control=RUN_CACHE_CONTROL)
    put_bytes(join(args.target, "manifest.json"), json.dumps(out, indent=2).encode(),
              content_type="application/json", cache_control=MANIFEST_CACHE_CONTROL)
    inv = invalidate_cloudfront(args.dist_id, ["/*"])
    print(f"wrote {len(patched_infos)} run_info.json + manifest.json; CloudFront invalidation {inv}")


if __name__ == "__main__":
    main()
