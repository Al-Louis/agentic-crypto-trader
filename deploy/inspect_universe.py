"""Print the vol-top-8 universe the event agent selected for the CURRENT cold week, plus the
trailing-vol ranking of the whole pool — to verify the causal selection on the live box.

  cd /srv/trader/agentic-crypto-trader && /srv/trader/venv/bin/python deploy/inspect_universe.py

The universe is `EventRungEnv._pick_universe` (voltopk, k=8): the k tokens with the highest
trailing-`warmup` realized vol at the week open. This reads the EXACT env the agent uses
(`eval_universe_and_caps`) so the printed set is authoritative, not a re-derivation.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

from train_rl import build_volume_panel, load_data  # noqa: E402
from train_event import eval_universe_and_caps  # noqa: E402
from trader.agent.event_live import LiveEventTrader, cold_week_window  # noqa: E402


def main() -> None:
    rid = sys.argv[1] if len(sys.argv) > 1 else "ppo-event-rdLe4-sbq-3c84b4a-s1"
    prov = json.load(open(f"/srv/trader/models/{rid}/{rid}/metrics.json", encoding="utf-8"))["provenance"]
    returns, btc, _a, liq = load_data()
    vol = build_volume_panel(list(returns.columns), returns.index)
    win, ws, i0 = cold_week_window(returns, int(time.time()))
    ek = LiveEventTrader(prov).env_kwargs(returns)
    uni, caps = eval_universe_and_caps(win, btc, liq, vol, ek)

    wlb = ek.get("universe_lookback") or ek["warmup"]
    rank = returns.iloc[i0 - wlb:i0].std().sort_values(ascending=False)
    print(f"week open : {dt.datetime.utcfromtimestamp(ws).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"pool size : {returns.shape[1]} tokens   |   vol lookback: {wlb}h   |   k=8 voltopk")
    print(f"SELECTED  : {uni}")
    print("trailing-vol ranking (whole pool, causal at week open):")
    for i, (tok, v) in enumerate(rank.items(), 1):
        mark = f"   <-- SELECTED  (alloc ${caps.get(tok, 0.0) * 10_000:,.0f})" if tok in uni else ""
        print(f"  {i:2}. {tok:12} {v:.5f}{mark}")


if __name__ == "__main__":
    main()
