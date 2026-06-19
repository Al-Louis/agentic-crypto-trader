"""CALIBRATE the anti-chase rotation brake (user, 2026-06-19): run the ACTUAL policy over all cold-weekly
windows and, for every ENTRY, record (a) the candidate's run-up over the prior 24h/48h, (b) whether the
entry was ROTATION-FUNDED (a ROTATION_OUT co-fired in the same step => a holding was sold to fund it), and
(c) the position's REALIZED net return (net of both-side AMM/gas fees). Then bin by run-up and report mean
net return + win + summed $PnL, SPLIT by rotation-funded vs cash-funded — so we read the threshold straight
off the data: "rotation-funded entries above X% run-up have negative net return." DESKTOP (torch).

  python scripts/calib_rotation_runup.py --run-id ppo-event-rdLe4-fxsbq-62800ff-s1 [--win 24]

Run-up window defaults to 24h (the brake's rotate_pump_win). The trigger already forces rising>0 over 24h,
so every fresh ignition has run-up>0; the question is WHERE net return crosses zero for the rotation subset.
"""
from __future__ import annotations
import argparse, json, os, pickle, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
from trader import config
from trader.train import weekly_eval as we
from trader.train.event_env import (EventRungEnv, ROTATION_OUT, IGNITION, SCALE_IN, PROFIT_TAKE,
                                     TRAILING_STOP, EMA_BREAK, LOSS_FLOOR)
from train_event import WARMUP
from simulate import env_kwargs_from_provenance, make_predict

FULL_CLOSE = {ROTATION_OUT, TRAILING_STOP, EMA_BREAK, LOSS_FLOOR}   # PROFIT_TAKE may be a partial trim
BINS = [(-9, 0), (0, .05), (.05, .10), (.10, .15), (.15, .20), (.20, .30), (.30, .50), (.50, 9)]


def collect_week(predict_fn, win, btc, liq, vol, ek, runup_win):
    """One cold-week episode; yield dicts per CLOSED position: runup, rotation_funded, net_ret, in_usd, split."""
    kw = {k: v for k, v in ek.items() if k != "episode_bars"}
    env = EventRungEnv(win, btc, liq, volume=vol, episode_bars=len(win) - WARMUP - 1,
                       record_trace=True, **kw)
    obs = env.reset(start=WARMUP)
    px, cix, idx = env._px, env.col_ix, win.index
    tmap = {int(t): b for b, t in enumerate(idx)}
    open_pos, out = {}, []
    done = False
    while not done:
        obs, _, done, info = env.step(predict_fn(obs))
        fills = info.get("trades")
        if not fills:
            continue
        rot_here = any(f[5] == ROTATION_OUT for f in fills)              # f = (tok,usd,fee,time,px,reason,obs)
        for tok, usd, fee, ft, fpx, reason, _ in fills:
            j = cix[tok]; b = tmap[int(ft)]
            if usd > 0:                                                  # BUY (entry or scale-in add)
                p = open_pos.get(tok)
                if p is None:
                    ru = (px[b, j] / px[b - runup_win, j] - 1.0) if b - runup_win >= 0 and px[b - runup_win, j] > 0 else np.nan
                    open_pos[tok] = {"in": usd, "fee": fee, "ru": ru, "rot": rot_here and reason in (IGNITION, SCALE_IN)}
                else:
                    p["in"] += usd; p["fee"] += fee                     # add: keep the entry's run-up/rot flag
            elif usd < 0:                                                # SELL (proceeds gross negative)
                p = open_pos.get(tok)
                if p is None:
                    continue
                p["out"] = p.get("out", 0.0) + (-usd); p["fee"] += fee
                if reason != PROFIT_TAKE:                                # full close -> realize
                    net = (p["out"] - p["fee"]) / max(p["in"], 1e-9) - 1.0
                    out.append({"ru": p["ru"], "rot": p["rot"], "net": net, "in": p["in"]})
                    open_pos.pop(tok, None)
    # MARK-TO-MARKET still-open positions at week-end px (the cold-week liquidation) — WITHOUT this the
    # closed-only sample is biased to LOSERS (cut mid-week) and drops let-run WINNERS held to the boundary.
    for tok, p in open_pos.items():
        rem = env._pos_value(tok) if tok in env.pos else 0.0
        net = (p.get("out", 0.0) + rem - p["fee"]) / max(p["in"], 1e-9) - 1.0
        out.append({"ru": p["ru"], "rot": p["rot"], "net": net, "in": p["in"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True); ap.add_argument("--win", type=int, default=24)
    a = ap.parse_args(); config.load_dotenv()
    base = os.path.join("runs-rl", a.run_id)
    prov = json.load(open(os.path.join(base, a.run_id, "metrics.json"))); prov = prov.get("provenance", prov)
    rec_ = bool(prov.get("recurrent"))
    from train_rl import build_ohlc_frac_panels, build_volume_panel, load_data
    returns, btc, _anchor, liq = load_data(); vol = build_volume_panel(list(returns.columns), returns.index)
    ek = env_kwargs_from_provenance(prov, returns, build_ohlc_frac_panels)
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO
    model = (RecurrentPPO if rec_ else PPO).load(os.path.join(base, "policy.zip"), device="cpu")
    vn = pickle.load(open(os.path.join(base, "vecnormalize.pkl"), "rb"))
    predict = make_predict(model, vn, rec_)

    rows = []
    for wi, (ws, win) in enumerate(we.cold_week_windows(returns)):
        rows += collect_week(predict, win, btc, liq, vol, ek, a.win)
    rows = [r for r in rows if not np.isnan(r["ru"])]
    recon = sum(r["in"] * r["net"] for r in rows)
    print(f"{a.run_id} | run-up win {a.win}h | positions w/ run-up: {len(rows)} | "
          f"recon sum$PnL={recon:+.0f} (expect ~10000*ret_sum; mark-to-market on)")

    def table(name, subset):
        print(f"\n=== {name}  (n={len(subset)}) ===")
        print(f"  {'runup bin':>12} | {'n':>4} {'mean_net':>9} {'win':>5} {'sum$PnL':>9} {'cum$ from here':>14}")
        # cumulative $PnL of all entries with run-up >= bin-low (the 'what blocking >=X removes' lens)
        for lo, hi in BINS:
            b = [r for r in subset if lo <= r["ru"] < hi]
            ge = [r for r in subset if r["ru"] >= lo]
            if not b and not ge:
                continue
            mn = np.mean([r["net"] for r in b]) if b else 0.0
            win = np.mean([r["net"] > 0 for r in b]) if b else 0.0
            spnl = sum(r["in"] * r["net"] for r in b)
            cum = sum(r["in"] * r["net"] for r in ge)
            lab = f"{lo:+.0%}..{hi:+.0%}" if hi < 9 else f"{lo:+.0%}+    "
            print(f"  {lab:>12} | {len(b):4d} {mn:+8.2%} {win:4.0%} {spnl:+8.0f} {cum:+13.0f}")

    table("ROTATION-FUNDED entries", [r for r in rows if r["rot"]])
    table("CASH-FUNDED entries", [r for r in rows if not r["rot"]])
    table("ALL entries", rows)
    print("\nREAD: pick rotate_pump_block = the run-up bin where ROTATION-FUNDED 'mean_net' turns and stays "
          "negative AND 'cum$ from here' is negative (blocking >=that run-up removes net-losing rotation chases).")


if __name__ == "__main__":
    main()
