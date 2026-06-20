"""Forensic: how does the env value an OPEN position held to week-end vs the displayed OHLCV candle?
Dumps a token's fills (env exec _px), the env's exact token_pnl, the env _px vs OHLCV close at entry +
last bar, and the open-remainder mark at episode end. DESKTOP (torch). Run from the repo dir:

  cd ~/agentic-crypto-trader && .venv/bin/python /tmp/trace_open_mark.py --run-id <id> --week-start 2026-04-28 --token TAC
"""
from __future__ import annotations
import argparse, json, os, pickle, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import pandas as pd
from datetime import datetime, timezone
from trader import config
from trader.train import weekly_eval as we
from trader.train.event_env import EventRungEnv
from train_event import WARMUP
from simulate import env_kwargs_from_provenance, make_predict


def dt(t): return datetime.fromtimestamp(int(t), timezone.utc).strftime("%m-%d %H:%M")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True); ap.add_argument("--week-start", required=True)
    ap.add_argument("--token", required=True)
    a = ap.parse_args(); config.load_dotenv()
    base = os.path.join("runs-rl", a.run_id)
    prov = json.load(open(os.path.join(base, a.run_id, "metrics.json"))); prov = prov.get("provenance", prov)
    rec_ = bool(prov.get("recurrent"))
    from train_rl import build_ohlc_frac_panels, build_volume_panel, load_data, _load_token_ohlcv
    returns, btc, _a, liq = load_data(); vol = build_volume_panel(list(returns.columns), returns.index)
    ek = env_kwargs_from_provenance(prov, returns, build_ohlc_frac_panels)
    ws = int(pd.Timestamp(a.week_start, tz="UTC").timestamp())
    win = next((w for s, w in we.cold_week_windows(returns) if s <= ws < s + 7 * 24 * 3600), None)
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO
    model = (RecurrentPPO if rec_ else PPO).load(os.path.join(base, "policy.zip"), device="cpu")
    vn = pickle.load(open(os.path.join(base, "vecnormalize.pkl"), "rb"))
    predict = make_predict(model, vn, rec_)
    kw = {k: v for k, v in ek.items() if k != "episode_bars"}
    env = EventRungEnv(win, btc, liq, volume=vol, episode_bars=len(win) - WARMUP - 1, record_trace=True, **kw)
    obs = env.reset(start=WARMUP)
    tok = a.token; j = env.col_ix[tok]
    fills = []
    done = False
    while not done:
        obs, _, done, info = env.step(predict(obs))
        for f in info.get("trades", []):
            if f[0] == tok:
                fills.append(f)                              # (token,usd,fee,time,px,reason,obs)
    last_bar = min(env.bar, env.n_bars - 1)
    tp = env.token_pnls().get(tok, 0.0)
    openpos = env.pos.get(tok)
    print(f"{a.run_id} | {tok} | week {dt(win.index[WARMUP])}..{dt(win.index[last_bar])}")
    print(f"env token_pnl[{tok}] = {tp:+.2f}  (realized cash flow + open-position mark at last bar)")
    print("FILLS (env exec _px = returns-index, NOT OHLCV):")
    for f in fills:
        print(f"  {dt(f[3])} {'BUY ' if f[1] > 0 else 'SELL'} usd={f[1]:+8.0f} env_px={f[4]:.6f} {f[5]}")
    oh = _load_token_ohlcv(tok)
    def ohlc_at(t):
        if oh is None or oh.empty:
            return float("nan")
        i = (oh["timestamp"] - t).abs().idxmin()
        return float(oh.loc[i, "close"])
    entry_t = int(fills[0][3]) if fills else int(win.index[WARMUP])
    last_t = int(win.index[last_bar])
    eb = env.returns.index.get_loc(entry_t)
    print(f"env _px : entry({dt(entry_t)})={env._px[eb, j]:.6f}   last({dt(last_t)})={env._px[last_bar, j]:.6f}   ratio={env._px[last_bar, j]/env._px[eb, j]:.2f}x")
    print(f"OHLCV   : entry={ohlc_at(entry_t):.6f}   last={ohlc_at(last_t):.6f}   ratio={ohlc_at(last_t)/ohlc_at(entry_t):.2f}x")
    if openpos:
        print(f"OPEN remainder @ end: usd(cost)={openpos['usd']:.0f} cost_px={openpos['cost_px']:.6f} "
              f"-> env mark_value={env._pos_value(tok):.0f} (gain {env._pos_value(tok)-openpos['usd']:+.0f})")
    else:
        print("no open position at end (env recorded a closing fill for the remainder)")


if __name__ == "__main__":
    main()
