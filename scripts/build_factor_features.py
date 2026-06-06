"""Build the "Bitcoin-is-King" factor features for the 20 alts vs the BTC/BNB anchor.

For each selected alt, aligns its hourly returns to the BTC+BNB anchor, fits the causal
rolling two-factor model, and emits the residual + residual-momentum. Then standardizes
residual-momentum cross-sectionally to rank current idiosyncratic strength.

Run:  .venv/Scripts/python.exe scripts/build_factor_features.py
Out:  data/features/<symbol>_factor.parquet  (+ a printed summary)
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trader.data.anchor import load_anchor  # noqa: E402
from trader.data.downloader import load_ohlcv  # noqa: E402
from trader.features.factor import compute_factor_features, cross_sectional_zscore  # noqa: E402

WINDOW = 168   # 1-week trailing beta (hourly)
MOM_SPAN = 24  # 1-day EWMA of the residual
OUT = "data/features"


def _anchor_seconds(symbol: str) -> pd.DataFrame:
    """Anchor hourly with ms timestamps normalized to seconds (to match the alt grid)."""
    df = load_anchor(symbol, "1h")
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df["timestamp"] = df["timestamp"] // 1000
    return df


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    os.makedirs(OUT, exist_ok=True)

    btc = _anchor_seconds("BTC/USDT")
    bnb = _anchor_seconds("BNB/USDT")
    sel = json.load(open("data/selection.json", encoding="utf-8"))
    print(f"factor build: {len(sel)} alts vs BTC+BNB anchor (hourly, window={WINDOW}h)\n")

    panel: dict[str, pd.Series] = {}
    rows = []
    for s in sel:
        sym, pool = s["symbol"], s["pair_address"]
        alt = load_ohlcv(sym, pool, "hour", 1)
        if alt.empty:
            print(f"  {sym:10} (no hourly data)")
            continue
        fac = compute_factor_features(alt, btc, bnb, window=WINDOW, mom_span=MOM_SPAN)
        fac.to_parquet(os.path.join(OUT, f"{sym}_factor.parquet"), index=False)
        valid = fac.dropna(subset=["residual"])
        if valid.empty:
            print(f"  {sym:10} (history too short for window)")
            continue
        panel[sym] = fac.set_index("timestamp")["resid_mom"]
        last = fac.iloc[-1]
        rows.append({
            "symbol": sym, "tier": s.get("tier"), "n": len(valid),
            "avg_r2": valid["r2"].mean(), "beta_btc": last["beta_btc"],
            "beta_bnb": last["beta_bnb"], "resid_mom": last["resid_mom"],
        })

    # alts' hourly series end at slightly different bars (downloaded at different moments),
    # so ffill a few bars to a common recent timestamp before the cross-sectional snapshot.
    wide = pd.DataFrame(panel).sort_index().ffill(limit=12)
    z = cross_sectional_zscore({c: wide[c] for c in wide.columns})
    latest_z = z.iloc[-1] if len(z) else pd.Series(dtype=float)
    for r in rows:
        r["resid_z"] = float(latest_z.get(r["symbol"], float("nan")))

    rows.sort(key=lambda r: (r["resid_z"] if pd.notna(r["resid_z"]) else -9), reverse=True)
    print(f"  {'symbol':10} {'tier':7} {'n':>5} {'avgR2':>6} {'beta_btc':>9} {'beta_bnb':>9} "
          f"{'resid_z':>8}")
    for r in rows:
        print(f"  {r['symbol']:10} {str(r['tier']):7} {r['n']:>5} {r['avg_r2']:>6.2f} "
              f"{r['beta_btc']:>9.2f} {r['beta_bnb']:>9.2f} {r['resid_z']:>8.2f}")

    print("\n  resid_z > 0  = idiosyncratic strength vs the universe right now (selection signal)")
    print(f"  low avgR2    = dev/idiosyncratic-driven (not following BTC/BNB)")
    print(f"  -> {OUT}/<symbol>_factor.parquet")


if __name__ == "__main__":
    main()
