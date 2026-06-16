"""Export a backtest/eval into the Apentic dashboard bundle.

This is the *contract seam* between our Python sim and the frontend (`alexlouis-site`,
`src/apentic/`). The dashboard reads static JSON from ``PUBLIC_APENTIC_DATA``: a top-level
``manifest.json`` plus, per run, ``trades.json`` / ``metrics.json`` / ``candles.json`` /
``equity_curve.json`` / ``run_info.json``. The TypeScript shapes are
`RoundTrip` / `MetricsReport` / `CandleData` / `EquityPoint`.

Generic over the strategy: `roundtrips_from_position` folds any single-asset exposure series
(a heuristic policy now, an RL policy later) into round-trips. The portfolio path can emit
metrics + equity + candles with an empty trade list until the single-asset-vs-portfolio fork
is decided (vault "Remote Capabilities").
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trader.sim.broker import DEFAULT_GAS_USD, DEFAULT_LP_FEE_BPS, amm_cost_usd
from trader.sim.metrics import MetricsReport, Trade

BUNDLE_FILES = ("trades.json", "metrics.json", "candles.json", "equity_curve.json", "run_info.json")


def _to_secs(ts: Any) -> int:
    """Normalize an epoch timestamp (s or ms) to integer seconds for lightweight-charts."""
    v = int(pd.Timestamp(ts).value // 1_000_000_000) if isinstance(ts, pd.Timestamp) else int(ts)
    return v // 1000 if v > 10_000_000_000 else v   # 13-digit ms → seconds


def _iso(ts: Any) -> str:
    return datetime.fromtimestamp(_to_secs(ts), tz=timezone.utc).isoformat()


def equity_points(equity: pd.Series, episode: int = 0) -> list[dict]:
    return [{"time": _to_secs(t), "value": float(v), "episode": episode}
            for t, v in equity.items()]


def candles_from_ohlcv(df: pd.DataFrame, episode: int = 0, time_col: str = "timestamp") -> list[dict]:
    """OHLCV frame (`timestamp,open,high,low,close,volume`) → `CandleData[]`, time-ascending."""
    df = df.sort_values(time_col)
    return [{"time": _to_secs(r[time_col]), "open": float(r["open"]), "high": float(r["high"]),
             "low": float(r["low"]), "close": float(r["close"]), "volume": float(r["volume"]),
             "episode": episode} for _, r in df.iterrows()]


def metrics_to_frontend(m: MetricsReport, *, episodes_evaluated: int = 1,
                        episodes_profitable: int | None = None,
                        avg_episode_return: float | None = None) -> dict:
    """`MetricsReport` → the dashboard's `MetricsReport` (adds the three episode fields).

    Non-finite floats (inf Calmar on a degenerate curve, etc.) become null — JSON has no inf.
    """
    d = {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
         for k, v in m.to_dict().items()}
    d["episodes_evaluated"] = episodes_evaluated
    d["episodes_profitable"] = (episodes_profitable if episodes_profitable is not None
                                else (1 if m.total_return_pct > 0 else 0))
    d["avg_episode_return"] = (avg_episode_return if avg_episode_return is not None
                               else m.total_return_pct)
    return d


def roundtrips_from_position(prices: pd.Series, position: pd.Series, *, capital: float = 10_000.0,
                             liquidity_usd: float = 0.0, lp_fee_bps: float = DEFAULT_LP_FEE_BPS,
                             gas_usd: float = DEFAULT_GAS_USD, rebal_threshold: float = 0.02,
                             min_trade_usd: float = 1.0, episode: int = 0
                             ) -> tuple[list[dict], pd.Series, list[Trade]]:
    """Fold a single-asset exposure series into round-trips, an equity curve, and Trade objects.

    `position[i]` ∈ [0,1] is the target fraction of equity to hold *during* bar i (caller makes
    it causal). A held position rides — we only trade when the target exposure moves by more
    than `rebal_threshold`, so a 0/1 policy yields exactly one buy + one sell per spell (no
    fee-induced churn). A round-trip is one contiguous in-position spell (exposure leaves 0 →
    returns to 0). AMM cost is charged on every traded notional, so the equity curve and
    round-trip PnL are cost-honest.
    """
    prices = prices.astype(float)
    position = position.reindex(prices.index).fillna(0.0).clip(0.0, 1.0)
    idx = list(prices.index)

    cash, hold, applied = float(capital), 0.0, 0.0
    equity = np.empty(len(idx))
    trade_objs: list[Trade] = []
    trips: list[dict] = []
    spell: dict | None = None       # open round-trip accumulator

    prev_price = float(prices.iloc[0])
    for i, t in enumerate(idx):
        p = float(prices.iloc[i])
        hold *= (1.0 + (p / prev_price - 1.0)) if prev_price else 1.0
        prev_price = p
        eq = cash + hold

        desired = float(position.iloc[i])
        if abs(desired - applied) >= rebal_threshold and eq > 1.0:
            trade = desired * eq - hold
            if abs(trade) >= min_trade_usd:
                fee = amm_cost_usd(trade, liquidity_usd, lp_fee_bps, gas_usd)
                cash -= trade + fee
                hold += trade
                eq = cash + hold
                applied = desired
                trade_objs.append(Trade(side="buy" if trade > 0 else "sell",
                                        quantity=abs(trade) / p, price=p, fee=fee, step=i))
                if trade > 0 and spell is None:        # 0 → in: open a round-trip
                    spell = {"i": i, "t": t, "price": p, "qty": trade / p, "fee": fee, "pv": eq}
                elif desired <= rebal_threshold and spell is not None:  # back to flat: close it
                    trips.append(_close_trip(spell, len(trips), i, t, p, fee, eq, episode))
                    spell = None
        equity[i] = eq

    if spell is not None:                          # still in position at series end → mark out
        trips.append(_close_trip(spell, len(trips), len(idx) - 1, idx[-1], float(prices.iloc[-1]),
                                 0.0, float(equity[-1]), episode, reason="end of sample"))
    return trips, pd.Series(equity, index=prices.index), trade_objs


def _close_trip(spell: dict, n: int, i: int, t: Any, exit_price: float, exit_fee: float,
                exit_pv: float, episode: int, reason: str = "flat signal") -> dict:
    pnl = exit_pv - spell["pv"]
    return {
        "id": f"rt-{n:04d}", "episode": episode,
        "entry_datetime": _iso(spell["t"]), "entry_price": spell["price"],
        "entry_quantity": spell["qty"], "entry_fee": spell["fee"],
        "entry_close": spell["price"], "entry_portfolio_value": spell["pv"],
        "exit_datetime": _iso(t), "exit_price": exit_price, "exit_fee": exit_fee,
        "exit_close": exit_price, "exit_portfolio_value": exit_pv, "exit_reason": reason,
        "duration_steps": i - spell["i"],
        "pnl_usdt": pnl, "pnl_pct": pnl / spell["pv"] if spell["pv"] else 0.0,
        "total_fees": spell["fee"] + exit_fee,
    }


def export_run(out_dir: Path | str, run_id: str, *, equity: pd.Series, metrics: dict,
               symbol: str, model_name: str, timestamp: str, trades: list[dict] | None = None,
               candles: list[dict] | None = None, regime: str = "", n_episodes: int = 1,
               indicators_used: list[str] | None = None,
               available_indicators: list[str] | None = None, ta_time: float | None = None,
               simulation: bool = False) -> dict:
    """Write a run's bundle under ``<out_dir>/<run_id>/`` and upsert ``<out_dir>/manifest.json``.

    Returns the manifest entry. `metrics` is the dict from `metrics_to_frontend`.
    """
    out_dir = Path(out_dir)
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    run_info = {"model_name": model_name,
                "indicators_used": indicators_used or [],
                "available_indicators": available_indicators or (indicators_used or []),
                "ta_time": ta_time, "regime": regime, "n_episodes": n_episodes}
    _dump(run_dir / "trades.json", trades or [])
    _dump(run_dir / "metrics.json", metrics)
    _dump(run_dir / "candles.json", candles or [])
    _dump(run_dir / "equity_curve.json", equity_points(equity))
    _dump(run_dir / "run_info.json", run_info)

    entry = {"id": run_id, "model_name": model_name, "timestamp": timestamp,
             "n_episodes": n_episodes, "regime": regime, "symbol": symbol,
             "simulation": simulation}
    upsert_manifest(out_dir / "manifest.json", entry)
    return entry


def _slug(token: str) -> str:
    """URL/file-safe token name (symbols can contain odd chars)."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(token)) or "tok"


def export_portfolio_run(out_dir: Path | str, run_id: str, *, equity: pd.Series, metrics: dict,
                         weights: list[dict], token_candles: dict[str, list],
                         token_trades: dict[str, list], universe: list[str], model_name: str,
                         timestamp: str, token_pnl: dict | None = None,
                         action_mode: str = "weights", regime: str = "",
                         simulation: bool = False) -> dict:
    """Write a **portfolio** run bundle (vs the single-asset `export_run`).

    Layout under ``<out_dir>/<run_id>/``:
      - metrics.json / equity_curve.json     — portfolio NAV + risk panel
      - weights.json                         — allocation over time: ``[{time, weights:{sym:w}}]``
      - run_info.json                        — model + ``universe`` ([{symbol, slug}], action_mode)
      - token_pnl.json                       — ``{symbol: $pnl}``, the env's EXACT per-token ledger
        (realized cash flow + any open position marked at the LAST bar — i.e. open lots treated as
        closed at the final price). The frontend reads PnL from THIS, never reconstructed from the
        markers (whose ``price`` is a display-basis index, not a clean per-unit price).
      - tk_<slug>_candles.json / _trades.json — per held token: its candles + buy/sell markers
    The frontend renders the allocation view from the first group and the per-token candle+marker
    drill-down from the last. Manifest entry carries ``kind:"portfolio"`` + the universe.
    """
    out_dir = Path(out_dir)
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    uni = [{"symbol": t, "slug": _slug(t)} for t in universe]

    _dump(run_dir / "metrics.json", metrics)
    _dump(run_dir / "equity_curve.json", equity_points(equity))
    _dump(run_dir / "weights.json", weights)
    _dump(run_dir / "token_pnl.json", {t: round(float(v), 2) for t, v in (token_pnl or {}).items()})
    _dump(run_dir / "run_info.json",
          {"model_name": model_name, "kind": "portfolio", "action_mode": action_mode,
           "universe": uni, "regime": regime, "n_episodes": 1, "indicators_used": [],
           "available_indicators": []})
    for u in uni:
        _dump(run_dir / f"tk_{u['slug']}_candles.json", token_candles.get(u["symbol"], []))
        _dump(run_dir / f"tk_{u['slug']}_trades.json", token_trades.get(u["symbol"], []))

    entry = {"id": run_id, "model_name": model_name, "timestamp": timestamp, "n_episodes": 1,
             "regime": regime, "symbol": "PORTFOLIO", "simulation": simulation,
             "kind": "portfolio", "universe": uni}
    upsert_manifest(out_dir / "manifest.json", entry)
    return entry


def upsert_manifest(manifest_path: Path | str, entry: dict) -> list[dict]:
    """Read-merge-write the dashboard manifest, replacing any entry with the same id."""
    manifest_path = Path(manifest_path)
    try:
        items = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        items = []
    items = [e for e in items if e.get("id") != entry["id"]] + [entry]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(items, indent=2), encoding="utf-8")
    return items


def _merge_entry(items: list[dict], entry: dict) -> list[dict]:
    return [e for e in items if e.get("id") != entry["id"]] + [entry]


# Per-run files never change (new run id each time) → cache forever; the manifest changes
# every publish → short cache, and CloudFront is invalidated so it refreshes immediately.
RUN_CACHE_CONTROL = "public, max-age=31536000, immutable"
MANIFEST_CACHE_CONTROL = "public, max-age=60"


def upsert_manifest_at(manifest_uri: str, entry: dict, cache_control: str | None = None) -> list[dict]:
    """Read-merge-write the manifest at a URI (local path or ``s3://…``).

    The same read-merge-write as `upsert_manifest`, but over `remote_train`'s object store so
    it works against S3/R2. Single-writer assumption (one run publishes at a time).
    """
    from remote_train import get_bytes, put_bytes  # local import: trader may depend on remote_train

    raw = get_bytes(manifest_uri)
    try:
        items = json.loads(raw) if raw else []
    except json.JSONDecodeError:
        items = []
    items = _merge_entry(items, entry)
    put_bytes(manifest_uri, json.dumps(items, indent=2).encode("utf-8"),
              content_type="application/json", cache_control=cache_control)
    return items


def publish_run(run_bundle_dir: Path | str, run_id: str, entry: dict, target: str,
                cloudfront_dist_id: str | None = None) -> str:
    """Publish one run to `target` (local dir or ``s3://…``) and merge its manifest entry.

    Uploads ``<run_bundle_dir>`` → ``<target>/<run_id>/`` and upserts ``<target>/manifest.json``.
    This is what the *job* calls (on the desktop) so the bundle goes straight to S3/R2 over the
    desktop's own internet — nothing large traverses the tailnet back to the laptop. When
    `cloudfront_dist_id` is set and `target` is S3, invalidates the served prefix so the CDN
    serves the new run immediately.
    """
    from urllib.parse import urlparse

    from remote_train import invalidate_cloudfront, join, publish  # trader may depend on remote_train

    dest = join(target, run_id)
    publish(run_bundle_dir, dest, cache_control=RUN_CACHE_CONTROL)
    upsert_manifest_at(join(target, "manifest.json"), entry, cache_control=MANIFEST_CACHE_CONTROL)

    if cloudfront_dist_id and urlparse(target).scheme in ("s3", "r2"):
        # One wildcard invalidation covers the manifest + the new run.
        invalidate_cloudfront(cloudfront_dist_id, [_invalidation_path(target)])
    return dest


def _invalidation_path(target: str) -> str:
    """CloudFront path to invalidate for an S3 target — mirrors the served key prefix.

    Root-hosted (no key prefix, e.g. a dedicated `data.alexlouis.dev` distribution) → ``/*``;
    prefix-hosted (e.g. ``s3://bucket/apentic/data`` behind a path behavior) → ``/apentic/data/*``.
    """
    from urllib.parse import urlparse

    key_prefix = urlparse(target).path.strip("/")
    return f"/{key_prefix}/*" if key_prefix else "/*"


def _dump(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
