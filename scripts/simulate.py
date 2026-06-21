"""Simulate a SAVED event-rung policy over arbitrary trailing windows and publish one Apentic
portfolio bundle per timeframe. The diagnostic that shows WHERE a checkpoint's behavior holds or
breaks across horizons — the input to curriculum design (so we tune the gaps, not design blind).

DESKTOP-ONLY (needs torch to load the policy + the OHLCV panel). The windowing / grading / export
core is the TRAINER'S OWN eval (`scripts/train_event.evaluate_and_gate`), reused verbatim, so a
simulation's numbers are identical to a training-time eval over the same window.

  python scripts/simulate.py --run-id ppo-event-rdLe4r-68b268f-s0 \
      [--timeframes 6mo,3mo,1mo,1wk,1d] [--no-publish] [--push-checkpoint]

It reads the checkpoint's OWN provenance (its published metrics.json) to rebuild the exact env
config it trained with — so the obs shape / action space match the policy. Per window:
  * trailing N+WARMUP bars from the data end (warmup served contiguously -> tradeable from bar 0);
  * its own volatile-k universe, picked at the window start -> the universe EVOLVES across horizons;
  * labeled in-sample / OOS by overlap with the train split. 6mo/3mo reach back into training and
    will look optimistic; 1mo/1wk/1d are fully held-out — read those as the honest signal.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from remote_train import join, put_bytes  # noqa: E402
from trader import config  # noqa: E402
from trader.report import apentic as ap  # noqa: E402

# the trainer's eval core, reused verbatim so sim numbers == training-eval numbers
from train_event import WARMUP, evaluate_and_gate  # noqa: E402

# trailing-window sizes in hourly bars (1yr omitted: only ~5123 bars / ~7mo of data exist)
TIMEFRAMES = {"6mo": 4320, "3mo": 2160, "1mo": 720, "1wk": 168, "1d": 24}
CKPT_CACHE = "public, max-age=31536000, immutable"


def env_kwargs_from_provenance(prov: dict, returns, build_ohlc_frac_panels) -> dict:
    """Rebuild the EXACT EventRungEnv kwargs the checkpoint trained with (mirrors train_event)."""
    tp = prov.get("tp_rungs") or ""
    kw = dict(k=prov["k"], warmup=WARMUP, max_entry_frac=prov["max_entry_frac"], stop_k=prov["stop_k"],
              cooldown=prov["cooldown"], dd_lambda=prov["dd_lambda"], dd_soft=prov["dd_soft"],
              reward_mode=prov["reward_mode"], r4_beta=prov["r4_beta"], res_gamma=prov["res_gamma"],
              fwd_horizon=prov["fwd_horizon"], ungate=prov["ungate"],
              action_mode=prov["action_mode"], n_action_levels=prov["n_action_levels"],
              universe_mode=prov["universe_mode"], vol_target=prov["vol_target"],
              vol_mult=prov.get("vol_mult", 2.5),                 # provenance lost vol_mult pre-2026-06-19;
              fixed_universe=prov.get("fixed_universe"),          # 2.5 default kept for old runs (record it now)
              cap_floor=prov["cap_floor"], harvest_obs=prov["harvest_obs"],
              rule_default=prov["rule_default"], basket_default=prov.get("basket_default", False),
              exit_commit=prov["exit_commit"],
              dust_usd=prov["dust_usd"], tp_rungs=[float(x) for x in tp.split(",") if x],
              loss_floor=prov["loss_floor"], det_blacklist=prov["det_blacklist"],
              scale_in=prov.get("scale_in", False),
              shallow_break_max=prov.get("shallow_break_max", 0.0),
              consol_vol_max=prov.get("consol_vol_max", 0.0),
              rotate_pump_block=prov.get("rotate_pump_block", 0.0),
              rotate_pump_win=prov.get("rotate_pump_win", 24),
              candle_exit=prov.get("candle_exit", False),
              candle_uw_min=prov.get("candle_uw_min", 0.5),
              candle_lw_max=prov.get("candle_lw_max", 0.25),
              candle_doji_max=prov.get("candle_doji_max", 0.10),
              cycle_obs=prov.get("cycle_obs", False), no_btc_obs=prov.get("no_btc_obs", False),
              universe_lookback=prov.get("universe_lookback", 0), seed=prov.get("seed", 0))
    _candle = bool(prov.get("candle_exit"))                   # candle_exit needs BOTH frac panels
    if prov.get("intrabar_floor") or (prov.get("wick_reject") or 0) > 0 or _candle:
        lowf, highf = build_ohlc_frac_panels(list(returns.columns), returns.index)
        kw.update(low_frac=lowf if (prov.get("intrabar_floor") or _candle) else None,
                  intrabar_floor=prov.get("intrabar_floor", False),
                  high_frac=highf if ((prov.get("wick_reject") or 0) > 0 or _candle) else None,
                  wick_reject=prov.get("wick_reject", 0.0))
    return kw


def make_predict(model, vn, recurrent):
    """Stateful (LSTM) predictor over one window, normalizing obs with the loaded VecNormalize stats."""
    st = {"s": None, "start": np.ones(1, dtype=bool)}

    def predict_fn(obs):
        norm = vn.normalize_obs(obs.reshape(1, -1))
        if recurrent:
            a, st["s"] = model.predict(norm, state=st["s"], episode_start=st["start"], deterministic=True)
            st["start"] = np.zeros(1, dtype=bool)
        else:
            a, _ = model.predict(norm, deterministic=True)
        return np.asarray(a).reshape(-1)
    return predict_fn


def oos_split(win, train_end, val_end):
    """Fraction of the window's TRADEABLE bars (after warmup) in train / val / test, + an OOS frac.
    The model trained only on the train split, so anything after train_end is out-of-sample."""
    ts = np.asarray([int(t) for t in win.index[WARMUP:]])
    n = max(len(ts), 1)
    in_train = int(np.sum(ts <= train_end))
    in_val = int(np.sum((ts > train_end) & (ts <= val_end)))
    return {"train_frac": round(in_train / n, 3),
            "val_frac": round(in_val / n, 3),
            "test_frac": round((n - in_train - in_val) / n, 3),
            "oos_frac": round((n - in_train) / n, 3)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True, help="checkpoint run-id (runs-rl/<run-id>/policy.zip)")
    p.add_argument("--timeframes", default="6mo,3mo,1mo,1wk,1d")
    p.add_argument("--provenance", default=None, help="override path to the provenance metrics.json")
    p.add_argument("--no-publish", action="store_true", help="build bundles locally, don't push to the CDN")
    p.add_argument("--push-checkpoint", action="store_true",
                   help="also upload policy.zip + vecnormalize.pkl to the S3 bundle prefix (durability)")
    args = p.parse_args()
    config.load_dotenv()

    base = os.path.join("runs-rl", args.run_id)
    policy_path = os.path.join(base, "policy.zip")
    vecnorm_path = os.path.join(base, "vecnormalize.pkl")
    prov_path = args.provenance or os.path.join(base, args.run_id, "metrics.json")
    for path in (policy_path, vecnorm_path, prov_path):
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} - is this a saved (post-2026-06-12) checkpoint?")

    prov = json.loads(open(prov_path, encoding="utf-8").read())
    prov = prov.get("provenance", prov)                # metrics.json wraps provenance under that key
    recurrent = bool(prov.get("recurrent"))
    seed, sha = int(prov.get("seed", 0)), prov.get("git_commit", "unknown")

    from train_rl import (build_ohlc_frac_panels, build_portfolio_artifacts, build_volume_panel,
                          load_data, time_split, trade_stats)

    returns, btc, anchor, liq = load_data()
    train_r, val_r, _test_r = time_split(returns)
    train_end, val_end = int(train_r.index[-1]), int(val_r.index[-1])
    vol = build_volume_panel(list(returns.columns), returns.index)
    env_kwargs = env_kwargs_from_provenance(prov, returns, build_ohlc_frac_panels)

    if recurrent:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(policy_path, device="cpu")
    else:
        from stable_baselines3 import PPO
        model = PPO.load(policy_path, device="cpu")
    with open(vecnorm_path, "rb") as f:
        vn = pickle.load(f)                             # VecNormalize (venv=None); normalize_obs uses obs_rms

    target = None if args.no_publish else config.get("APENTIC_PUBLISH_TARGET")
    cf = config.get("APENTIC_CLOUDFRONT_DIST_ID")

    summary = []
    for tf in [t.strip() for t in args.timeframes.split(",") if t.strip()]:
        if tf not in TIMEFRAMES:
            print(f"[sim] skip unknown timeframe {tf!r} (known: {','.join(TIMEFRAMES)})")
            continue
        n_bars = TIMEFRAMES[tf]
        win = returns.tail(n_bars + WARMUP)
        if len(win) < WARMUP + 2:
            print(f"[sim] skip {tf}: not enough data ({len(win)} bars)")
            continue
        res = evaluate_and_gate(tf, win, btc, liq, vol, env_kwargs, make_predict(model, vn, recurrent), seed)
        d0, d1 = int(win.index[WARMUP]), int(win.index[-1])
        weights, candles, trades = build_portfolio_artifacts(res["records"], res["universe"], d0, d1)
        metrics = ap.metrics_to_frontend(res["report"])
        metrics["total_fees_paid"] = res["fees"]
        metrics.update(trade_stats(trades))
        oos = oos_split(win, train_end, val_end)
        metrics.update({"baseline_return": res["base"], "buyhold_return": res["bh"],
                        "random_return": res["rnd"], "regime": res["regime"],
                        "gate_pass": res["gate_pass"], "gate_binding": res["binding"],
                        "simulation": {"source_run": args.run_id, "timeframe": tf,
                                       "window_bars": int(n_bars), "window_start": d0, "window_end": d1,
                                       "git_commit": sha, **oos}})
        sim_id = f"{args.run_id}-sim-{tf}"
        oos_lbl = "OOS" if oos["oos_frac"] >= 0.999 else f"{int(oos['oos_frac'] * 100)}%OOS"
        model_name = f"{args.run_id} | sim {tf} | {res['regime']['label']} | {oos_lbl}"
        entry = ap.export_portfolio_run(base, sim_id, equity=res["eq"].iloc[::6], metrics=metrics,
                                        weights=weights, token_candles=candles, token_trades=trades,
                                        universe=res["universe"], model_name=model_name,
                                        token_pnl=res["token_pnl"],
                                        action_mode="event", regime=res["regime"]["label"],
                                        simulation=True,
                                        timestamp=datetime.now(timezone.utc).isoformat())
        # surface the selector keys directly on the manifest entry so the frontend can build the
        # model + timeframe dropdowns (and the OOS badge) without fetching every bundle's metrics.json
        entry.update({"source_run": args.run_id, "timeframe": tf, "oos_frac": oos["oos_frac"]})
        if target:
            ap.publish_run(os.path.join(base, sim_id), sim_id, entry, target, cloudfront_dist_id=cf)
        print(f"[sim] {tf}: policy {res['pol']:+.1%} vs B&H {res['bh']:+.1%} rung0 {res['base']:+.1%} "
              f"| {res['regime']['label']} | OOS {oos['oos_frac']:.0%} | trades {metrics['total_trades']} "
              f"| {'published ' + sim_id if target else 'local only'}")
        summary.append({"timeframe": tf, "policy": res["pol"], "oos_frac": oos["oos_frac"]})

    if args.push_checkpoint and target:
        for fn, ct in (("policy.zip", "application/zip"), ("vecnormalize.pkl", "application/octet-stream")):
            put_bytes(join(target, f"{args.run_id}/{fn}"), open(os.path.join(base, fn), "rb").read(),
                      content_type=ct, cache_control=CKPT_CACHE)
        print(f"[sim] pushed checkpoint -> {join(target, args.run_id)}/ (policy.zip + vecnormalize.pkl)")
    elif args.push_checkpoint:
        print("[sim] --push-checkpoint ignored (no publish target / --no-publish)")

    print("[sim] DONE " + args.run_id + ": "
          + " | ".join(f"{s['timeframe']} {s['policy']:+.0%}(OOS{s['oos_frac']:.0%})" for s in summary))


if __name__ == "__main__":
    main()
