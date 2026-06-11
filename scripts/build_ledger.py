"""Build the committed experiment ledger from the published bundles.

The rigid, reproducible record of every training iteration (exact config -> performance), so we
can always find and RETURN TO the best known formula when a tweak degrades things. This is the
TradeSim lesson made structural: never tweak without a permanent, version-controlled performance
trail.

    python scripts/build_ledger.py            # rebuild experiments/ledger.jsonl + champion.json

Thin CLI over `trader.experiment.champion` (same logic the experiment_record / experiment_champion
MCP tools use). Source of truth = the immutable published metrics.json per run (which carries a
`provenance` block). Champion = highest MEAN return among configs that PASSED the frozen test.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trader.experiment.champion import DEFAULT_OUT, rebuild_ledger, write_ledger  # noqa: E402

HOST = "https://data.alexlouis.dev"
# public infra IDs (not secrets — protected by IAM creds, not obscurity); override via CLI / env
DEFAULT_TARGET = "s3://alexlouis-apentic-data"
DEFAULT_CF_DIST = "E14F268NIY6WLZ"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=HOST)
    p.add_argument("--dd-gate", type=float, default=0.30)
    p.add_argument("--all", action="store_true",
                   help="include pre-sha-convention runs (default: sha-stamped run-ids only — "
                        "everything before ec1e487 is the invalid era)")
    p.add_argument("--publish", action="store_true",
                   help="publish leaderboard.json to the Apentic data host for the frontend overview")
    p.add_argument("--publish-target", default=None, help=f"default: env or {DEFAULT_TARGET}")
    p.add_argument("--cloudfront-dist", default=None, help=f"default: env or {DEFAULT_CF_DIST}")
    args = p.parse_args()

    res = rebuild_ledger(host=args.host, dd_gate=args.dd_gate,
                         generated=datetime.now(timezone.utc).isoformat(),
                         sha_only=not args.all)
    write_ledger(res, DEFAULT_OUT)
    rows, summary, champion, leaderboard = (res["rows"], res["summary"],
                                            res["champion"], res["leaderboard"])

    print(f"ledger: {len(rows)} runs -> {DEFAULT_OUT}/ledger.jsonl\n")
    print(f"{'config':20}{'split':>6}{'n':>3}{'mean ret':>10}{'mean DD':>9}{'worst DD':>9}{'vs base':>9}")
    for label, v in sorted(summary.items(), key=lambda x: (x[1].get("split", "val"), -x[1]["mean_return"])):
        base = v.get("baseline")
        vs = f"{(v['mean_return'] - base) * 100:+.0f}pt" if base is not None else "?"
        print(f"{label:20}{v.get('split', 'val'):>6}{v['n']:>3}{v['mean_return']*100:>+9.1f}%"
              f"{v['mean_maxdd']*100:>8.1f}%{(v['worst_maxdd'] or 0)*100:>8.1f}%{vs:>9}")
    if champion:
        print(f"\nCHAMPION (passed frozen-test OOS): {champion['config_label']}  "
              f"+{champion['mean_return']*100:.1f}% @ worst-seed {(champion['worst_maxdd'] or 0)*100:.1f}% DD")
    else:
        print("\nCHAMPION: none — no config has passed frozen-test OOS "
              "(beat its test baseline + worst-seed under the gate)")

    print(f"\nleaderboard -> {DEFAULT_OUT}/leaderboard.json ({len(leaderboard['configs'])} configs)")
    if args.publish:
        import importlib  # noqa: PLC0415
        pub = importlib.import_module("remote_train.publish")
        from trader import config  # noqa: PLC0415
        config.load_dotenv()
        target = args.publish_target or config.get("APENTIC_PUBLISH_TARGET") or DEFAULT_TARGET
        dist = args.cloudfront_dist or config.get("APENTIC_CLOUDFRONT_DIST_ID") or DEFAULT_CF_DIST
        data = json.dumps(leaderboard, indent=2).encode()
        pub.put_bytes(f"{target}/leaderboard.json", data, "application/json", "no-cache, max-age=0")
        inv = pub.invalidate_cloudfront(dist, ["/leaderboard.json"]) if dist else None
        print(f"published leaderboard.json -> {target}"
              + (f" (+ CloudFront invalidation {inv})" if inv else ""))


if __name__ == "__main__":
    main()
