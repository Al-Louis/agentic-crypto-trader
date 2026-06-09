"""Train a PPO exposure-overlay policy and publish its eval bundle. DESKTOP-ONLY.

Needs the `training` extra (gymnasium + stable-baselines3 + CPU torch) — the laptop's Py3.14
has no torch wheel, so this runs on the trainer. Loads the returns panel + BTC anchor,
time-splits into train / val / frozen-test, trains PPO on vectorized envs, evaluates the
policy on a held-out window, and self-publishes the Apentic bundle (so the loop's diagnose_run
scores it vs the vol-tilt baseline). Writes progress.json throughout for fire-and-poll status.

  python scripts/train_rl.py --timesteps 300000 --n-envs 8 --run-id ppo-exposure-001

NOTE: composes tested modules (env, gym_env, metrics, trader.report, remote_train.progress);
the PPO/VecNormalize glue is validated on the desktop, not the laptop.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from remote_train.progress import write_progress  # noqa: E402
from trader import config  # noqa: E402
from trader.report import apentic as ap  # noqa: E402
from trader.sim.metrics import PerformanceMetrics  # noqa: E402

HOURS_PER_YEAR = 24 * 365


def load_data():
    ret = {}
    for f in sorted(glob.glob(os.path.join("data", "features", "*_factor.parquet"))):
        sym = os.path.basename(f)[:-len("_factor.parquet")]
        ret[sym] = pd.read_parquet(f).set_index("timestamp")["r_alt"]
    returns = np.expm1(pd.DataFrame(ret).sort_index())              # log → simple
    anchor = pd.read_parquet(os.path.join("data", "anchor", "BTC_USDT", "1h.parquet"))
    anchor = anchor.set_index("timestamp").sort_index()
    if anchor.index.max() > 1e12:                  # anchor is ms; factor returns are seconds — align
        anchor.index = (anchor.index // 1000).astype("int64")
    btc_close = anchor["close"]
    liq = {s["symbol"]: (s.get("liq_usd") or 0.0)
           for s in json.load(open(os.path.join("data", "selection.json"), encoding="utf-8"))}
    return returns, btc_close, anchor, liq


def time_split(returns, train=0.6, val=0.2):
    n = len(returns)
    a, b = int(n * train), int(n * (train + val))
    return returns.iloc[:a], returns.iloc[a:b], returns.iloc[b:]


def evaluate_policy(model, vecnorm, returns_win, btc_close, liq, env_kwargs):
    """One deterministic episode spanning the held-out window → equity curve + cost/trade tally."""
    from trader.train.env import PortfolioEnv
    steps = max((len(returns_win) - env_kwargs["warmup"]) // env_kwargs["step_bars"] - 1, 1)
    env = PortfolioEnv(returns_win, btc_close, liq, **{**env_kwargs, "episode_steps": steps})
    obs = env.reset(start=env._min_start)
    fees = trades = 0
    raw_actions = []
    done = False
    while not done:
        norm = vecnorm.normalize_obs(obs.reshape(1, -1)) if vecnorm is not None else obs.reshape(1, -1)
        action, _ = model.predict(norm, deterministic=True)
        raw_actions.append(float(np.asarray(action).reshape(-1).sum()))  # total allocation weight
        obs, _, done, info = env.step(np.asarray(action).reshape(-1))
        fees += info["cost"]
        trades += 1 if info["cost"] > 0 else 0
    return np.asarray(env.equity_curve, dtype=float), fees, trades, raw_actions


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--run-id", default="ppo-exposure")
    p.add_argument("--out", default=None, help="artifact dir (default: ./runs-rl/<run-id>)")
    p.add_argument("--publish-target", default=None, help="default: env APENTIC_PUBLISH_TARGET")
    p.add_argument("--action-mode", default="exposure", choices=["exposure", "weights"],
                   help="exposure=scalar dial on vol-top8 (C); weights=per-token allocation (B)")
    p.add_argument("--step-bars", type=int, default=24)
    p.add_argument("--episode-steps", type=int, default=30)
    p.add_argument("--ent-coef", type=float, default=0.2,    # post-mortem: low ent_coef → "always-wait" collapse
                   help="PPO entropy coefficient (exploration)")
    p.add_argument("--lr", type=float, default=3e-4, help="PPO learning rate")
    p.add_argument("--eval-split", default="val", choices=["val", "test"],
                   help="held-out split to evaluate on (test = the frozen, honest verdict)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    config.load_dotenv()
    out = args.out or os.path.join("runs-rl", args.run_id)
    os.makedirs(out, exist_ok=True)

    # sb3/torch imported here so --help works without the training extra
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    from trader.train.gym_env import GymPortfolioEnv

    returns, btc_close, anchor, liq = load_data()
    train_r, val_r, test_r = time_split(returns)
    eval_r = test_r if args.eval_split == "test" else val_r    # tune on val; final verdict on test
    env_kwargs = dict(step_bars=args.step_bars, episode_steps=args.episode_steps,
                      warmup=168, action_mode=args.action_mode, seed=args.seed)

    write_progress(out, state="running", phase="setup", run_id=args.run_id,
                   timesteps=0, total=args.timesteps)

    def make_env(rank):
        def _f():
            return Monitor(GymPortfolioEnv(train_r, btc_close, liq,
                                           **{**env_kwargs, "seed": args.seed + rank}))
        return _f

    venv = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    class ProgressCb(BaseCallback):
        def _on_step(self) -> bool:
            if self.num_timesteps % 2048 < venv.num_envs:
                rews = [e["r"] for e in self.model.ep_info_buffer] if self.model.ep_info_buffer else []
                write_progress(out, state="running", phase="train", timesteps=self.num_timesteps,
                               total=args.timesteps,
                               mean_reward=float(np.mean(rews)) if rews else None,
                               history_key="curve")
            return True

    model = PPO("MlpPolicy", venv, verbose=0, seed=args.seed, n_steps=1024, batch_size=256,
                ent_coef=args.ent_coef, learning_rate=args.lr)
    model.learn(total_timesteps=args.timesteps, callback=ProgressCb())

    # ---- evaluate on the held-out split, build + publish the bundle ----
    write_progress(out, state="running", phase="evaluate")
    equity, fees, trades, raw_actions = evaluate_policy(model, venv, eval_r, btc_close, liq, env_kwargs)
    print(f"[eval] total allocation weight: min={min(raw_actions):+.3f} "
          f"mean={float(np.mean(raw_actions)):+.3f} max={max(raw_actions):+.3f} (0 ⇒ all cash)")

    # the eval equity curve is one point per *step* (step_bars hours each), so annualize per step
    steps_per_year = HOURS_PER_YEAR / args.step_bars
    report = PerformanceMetrics.compute_all(equity, steps_per_year=steps_per_year)
    metrics = ap.metrics_to_frontend(report)
    metrics["total_trades"] = trades
    metrics["total_fees_paid"] = fees
    metrics["fees_as_pct_of_pnl"] = fees / (abs(equity[-1] - equity[0]) + 1e-9)

    # equity points sit at step boundaries (warmup + k·step_bars), not consecutive bars
    sb, wu = args.step_bars, env_kwargs["warmup"]
    positions = [min(wu + k * sb, len(eval_r) - 1) for k in range(len(equity))]
    eq_series = pd.Series(equity, index=eval_r.index[positions])
    candles = ap.candles_from_ohlcv(anchor.loc[eval_r.index[0]:eval_r.index[-1]].reset_index())

    # ---- honest head-to-head: the validated vol-tilt on the SAME window, same backtester ----
    from trader.sim.backtest import run_xs_backtest
    from trader.strategy.candidate import build_candidate
    base_fn = build_candidate(eval_r, btc_close=btc_close, k=8, overlay="trend50")
    base_eq = run_xs_backtest(eval_r, base_fn, liq, rebalance_every=args.step_bars,
                              warmup=wu)["equity"].to_numpy()
    base_daily = base_eq[::sb]                              # match the policy's daily resolution
    base_report = PerformanceMetrics.compute_all(base_daily, steps_per_year=steps_per_year)
    base_ret = base_eq[-1] / base_eq[0] - 1.0 if base_eq[0] else 0.0
    metrics["baseline_return"] = base_ret
    print(f"[baseline] vol-tilt(trend50) on {args.eval_split}: return {base_ret:+.1%}, "
          f"Sharpe {base_report.sharpe_ratio:.2f}, maxDD {base_report.max_drawdown_pct:.1%}")
    print(f"[verdict] policy {report.total_return_pct:+.1%} (Sh {report.sharpe_ratio:.2f}) vs "
          f"vol-tilt {base_ret:+.1%} (Sh {base_report.sharpe_ratio:.2f}) on {args.eval_split} → "
          f"{'BEATS' if report.total_return_pct > base_ret else 'loses to'} baseline")

    entry = ap.export_run(
        out, args.run_id, equity=eq_series, metrics=metrics, trades=[], candles=candles,
        symbol="PORTFOLIO", model_name=f"PPO {args.action_mode} ({args.timesteps:,} steps)",
        regime="val", n_episodes=1, indicators_used=["exposure"],
        timestamp=datetime.now(timezone.utc).isoformat())

    target = args.publish_target or config.get("APENTIC_PUBLISH_TARGET")
    if target:
        ap.publish_run(os.path.join(out, args.run_id), args.run_id, entry, target,
                       cloudfront_dist_id=config.get("APENTIC_CLOUDFRONT_DIST_ID"))

    write_progress(out, state="complete", run_id=args.run_id,
                   total_return=report.total_return_pct, sharpe=report.sharpe_ratio,
                   max_drawdown=report.max_drawdown_pct, trades=trades)
    print(f"[train_rl] {args.run_id}: return {report.total_return_pct:+.1%}, "
          f"Sharpe {report.sharpe_ratio:.2f}, maxDD {report.max_drawdown_pct:.1%}, trades {trades}")


if __name__ == "__main__":
    main()
