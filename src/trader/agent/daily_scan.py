"""Daily market-volatility scan -> `market_metrics.json` (informational) + today's traded set.

Runs on the EC2 box at the start of each trading day (systemd timer). Two parts, both reusing
existing, validated code:
  * the per-token vol / correlation dashboard — `trader.report.market_metrics.compute_market_metrics`
    (pure, unchanged);
  * a **`selected`** block = the model's ACTUAL current vol-top-8, read from the SAME env path the
    live harness trades (`train_event.eval_universe_and_caps` over the current cold-week window).
    ef-s2 selects WEEKLY (causal trailing-168h vol at the Monday open, fixed for the week — how it
    was trained), so `selected` changes weekly while the metrics refresh daily. This surfaces the
    real traded set transparently WITHOUT changing the frozen model. See [[Live Forward-Run Harness]].

The selection is torch-free (only `model.predict` needs torch), so this is laptop-testable against
recorded data. Publishes the top-level `market_metrics.json` via the instance role (needs the
`market_metrics.json` PutObject grant — `deploy/iam/market-metrics-put-policy.json`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

from trader import config

DEFAULT_TARGET = "s3://alexlouis-apentic-data"


def _slice(returns, window: str, last_hours: int):
    """Window the panel for the dashboard metrics (the SELECTED set always uses the full panel so
    the week-open warmup is intact)."""
    if window == "last":
        return returns.iloc[-last_hours:]
    if window in ("train", "val", "test"):
        from train_rl import time_split  # noqa: PLC0415
        tr, val, te = time_split(returns)
        return {"train": tr, "val": val, "test": te}[window]
    return returns


def build_selected(returns, btc, liq, vol, env_kwargs: dict, now_ts: int, run_id: str) -> dict:
    """The model's CURRENT vol-top-8 for the cold week containing `now_ts`, read from the exact env
    the harness trades (`eval_universe_and_caps`) — authoritative, not a re-derivation. Torch-free."""
    from train_event import eval_universe_and_caps  # noqa: PLC0415
    from trader.agent.event_live import cold_week_window  # noqa: PLC0415
    win, ws, _i0 = cold_week_window(returns, now_ts)
    uni, caps = eval_universe_and_caps(win, btc, liq, vol, env_kwargs)
    return {
        "method": "EventRungEnv voltopk k=8 — causal trailing-168h vol at the week open (weekly cadence)",
        "run_id": run_id,
        "week_start": int(ws),
        "as_of": int(now_ts),
        "tokens": list(uni),
        "caps": {t: round(float(caps.get(t, 0.0)), 4) for t in uni},
        "alloc_usd": {t: round(float(caps.get(t, 0.0)) * 10_000.0, 2) for t in uni},
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trader.agent.daily_scan",
                                description="Daily market_metrics.json scan + today's traded set")
    p.add_argument("--run-dir", required=True, help="deployed checkpoint dir (for the env config)")
    p.add_argument("--run-id", default=None, help="defaults to the run-dir basename")
    p.add_argument("--window", default="last", choices=["full", "train", "val", "test", "last"],
                   help="dashboard-metrics window (the SELECTED set always uses the full panel)")
    p.add_argument("--last-hours", type=int, default=720, help="--window last: trailing hours (30d)")
    p.add_argument("--out", default="data/apentic", help="local dir to write market_metrics.json")
    p.add_argument("--publish", action="store_true", help="publish to the CDN (instance role)")
    p.add_argument("--now", type=int, default=None, help="override wall-clock now (unix sec; tests)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config.load_dotenv()
    run_id = args.run_id or os.path.basename(os.path.normpath(args.run_dir))
    now_ts = int(args.now if args.now is not None else time.time())

    from train_rl import build_volume_panel, load_data
    from trader.agent.event_agent import load_provenance
    from trader.agent.event_live import LiveEventTrader
    from trader.report.market_metrics import compute_market_metrics

    returns, btc, _anchor, liq = load_data()
    vol = build_volume_panel(list(returns.columns), returns.index)

    metrics = compute_market_metrics(_slice(returns, args.window, args.last_hours), btc,
                                     generated=datetime.now(timezone.utc).isoformat())
    metrics["window"]["kind"] = args.window

    prov = load_provenance(args.run_dir, run_id)
    env_kwargs = LiveEventTrader(prov).env_kwargs(returns)
    metrics["selected"] = build_selected(returns, btc, liq, vol, env_kwargs, now_ts, run_id)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "market_metrics.json")
    data = json.dumps(metrics, indent=2).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(data)
    sel = metrics["selected"]
    print(f"market_metrics: {metrics['summary']['n_tokens']} tokens, regime "
          f"{metrics['summary']['regime_label']} -> {path}", file=sys.stderr)
    print(f"selected (week {datetime.fromtimestamp(sel['week_start'], timezone.utc).date()}): "
          f"{sel['tokens']}", file=sys.stderr)

    if args.publish:
        from remote_train.publish import put_bytes  # noqa: PLC0415
        target = config.get("APENTIC_PUBLISH_TARGET") or DEFAULT_TARGET
        # top-level artifact (NOT under trading/) — needs the market_metrics PutObject grant.
        # no-cache so the daily refresh is seen promptly (belt-and-braces for the CDN behavior).
        uri = f"{target.rstrip('/')}/market_metrics.json"
        put_bytes(uri, data, content_type="application/json", cache_control="no-cache, max-age=0")
        print(f"published -> {uri}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
