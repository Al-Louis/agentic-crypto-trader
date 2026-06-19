"""Replay a saved policy over ONE cold week with the env trace on, and print a token's fills with
the TRIGGER enum (ignition / ema_break / trailing_stop / profit_take / loss_floor / intrabar_stop /
rotation_out). Answers "what exactly sold this position" mechanically. DESKTOP (torch).

    python scripts/trace_exit_trigger.py --run-id <id> --token FF --week-start 2026-04-06 [--vol-mult 2.0]
"""
from __future__ import annotations
import argparse, json, os, pickle, sys
from datetime import datetime, timezone
sys.path.insert(0, "scripts"); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from trader import config
from trader.train import weekly_eval as we
from train_event import evaluate_event_policy
from simulate import env_kwargs_from_provenance, make_predict

def dts(t): return datetime.fromtimestamp(int(t), timezone.utc).strftime("%m-%d %H:%M")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True); p.add_argument("--token", required=True)
    p.add_argument("--week-start", required=True); p.add_argument("--vol-mult", type=float, default=None)
    args = p.parse_args(); config.load_dotenv()
    base = os.path.join("runs-rl", args.run_id)
    prov = json.load(open(os.path.join(base, args.run_id, "metrics.json")))
    prov = prov.get("provenance", prov)
    recurrent = bool(prov.get("recurrent"))
    from train_rl import build_ohlc_frac_panels, build_volume_panel, load_data
    returns, btc, _a, liq = load_data()
    vol = build_volume_panel(list(returns.columns), returns.index)
    env_kwargs = env_kwargs_from_provenance(prov, returns, build_ohlc_frac_panels)
    if args.vol_mult is not None:
        env_kwargs["vol_mult"] = args.vol_mult
    ws = int(pd.Timestamp(args.week_start, tz="UTC").timestamp())
    win = next((w for s, w in we.cold_week_windows(returns) if s <= ws < s + 7 * 24 * 3600), None)
    if win is None:
        raise SystemExit(f"no cold week covering {args.week_start}")
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO
    model = (RecurrentPPO if recurrent else PPO).load(os.path.join(base, "policy.zip"), device="cpu")
    vn = pickle.load(open(os.path.join(base, "vecnormalize.pkl"), "rb"))
    eq, records, universe, fees, raw, tok_pnl = evaluate_event_policy(
        make_predict(model, vn, recurrent), win, btc, liq, vol, env_kwargs)
    print(f"{args.run_id} | {args.token} | week {args.week_start} | in-universe={args.token in universe}"
          f" | token PnL ${tok_pnl.get(args.token, 0.0):+.0f}")
    for rec in records:
        for f in rec.get("fills", []):
            if f["token"] == args.token:
                side = "BUY " if f["usd"] > 0 else "SELL"
                print(f"  {dts(f['time'])} {side} usd {f['usd']:+8.0f} px {f['px']:.4g} reason={f['reason']}")

if __name__ == "__main__":
    main()
