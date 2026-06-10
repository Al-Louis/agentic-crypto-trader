"""Compute the volatility / correlation dashboard metrics and publish ``market_metrics.json``.

Top-level Apentic artifact (sibling of ``leaderboard.json``) for the frontend's market-structure
view: per-token realized vol, each token's corr/beta to BTC, and the token x token correlation
matrix (see trader.report.market_metrics + [[Apentic Data Contract]]).

  # write locally (laptop, inspect the JSON):
  python scripts/publish_market_metrics.py --out data/apentic
  # publish to the data host + invalidate the CDN (desktop, where the OHLCV lives):
  python scripts/publish_market_metrics.py --publish [--window last --last-hours 720]

`--window` picks the slice: full (default) | train | val | test | last (the most recent
`--last-hours`). For a near-real-time view, run the `last` window on a schedule on the desktop.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, "scripts")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train_rl import load_data, time_split  # noqa: E402
from trader import config  # noqa: E402
from trader.report.market_metrics import compute_market_metrics  # noqa: E402

DEFAULT_TARGET = "s3://alexlouis-apentic-data"
DEFAULT_CF_DIST = "E14F268NIY6WLZ"


def _slice(returns, window: str, last_hours: int):
    if window == "last":
        return returns.iloc[-last_hours:]
    if window in ("train", "val", "test"):
        tr, val, te = time_split(returns)
        return {"train": tr, "val": val, "test": te}[window]
    return returns                                              # full


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/apentic", help="local dir to write market_metrics.json")
    p.add_argument("--window", default="full", choices=["full", "train", "val", "test", "last"])
    p.add_argument("--last-hours", type=int, default=720, help="--window last: trailing hours (30d)")
    p.add_argument("--vol-window", type=int, default=168, help="rolling-vol sparkline lookback (bars)")
    p.add_argument("--publish", action="store_true", help="publish market_metrics.json + invalidate CDN")
    p.add_argument("--publish-target", default=None, help=f"default: env or {DEFAULT_TARGET}")
    p.add_argument("--cloudfront-dist", default=None, help=f"default: env or {DEFAULT_CF_DIST}")
    args = p.parse_args()

    returns, btc_close, _anchor, _liq = load_data()
    r = _slice(returns, args.window, args.last_hours)
    metrics = compute_market_metrics(r, btc_close, vol_spark_window=args.vol_window,
                                     generated=datetime.now(timezone.utc).isoformat())
    metrics["window"]["kind"] = args.window

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "market_metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    s = metrics["summary"]
    print(f"market_metrics ({args.window}, {metrics['window']['bars']} bars, {s['n_tokens']} tokens) "
          f"-> {path}")
    print(f"  regime {s['regime_label']}  universe-EW {s['universe_ew_return']:+.1%}  "
          f"BTC {metrics['btc']['ret_window']:+.1%}  avg-corr {s['avg_pairwise_corr']:+.3f}")
    top = metrics["tokens"][0]
    print(f"  most volatile: {top['symbol']} ann_vol {top['ann_vol']:.1f}  corr_BTC {top['corr_btc']:+.2f}")

    if args.publish:
        import importlib  # noqa: PLC0415
        pub = importlib.import_module("remote_train.publish")
        config.load_dotenv()
        target = args.publish_target or config.get("APENTIC_PUBLISH_TARGET") or DEFAULT_TARGET
        dist = args.cloudfront_dist or config.get("APENTIC_CLOUDFRONT_DIST_ID") or DEFAULT_CF_DIST
        data = json.dumps(metrics, indent=2).encode()
        pub.put_bytes(f"{target}/market_metrics.json", data, "application/json", "no-cache, max-age=0")
        inv = pub.invalidate_cloudfront(dist, ["/market_metrics.json"]) if dist else None
        print(f"published market_metrics.json -> {target}"
              + (f" (+ CloudFront invalidation {inv})" if inv else ""))


if __name__ == "__main__":
    main()
