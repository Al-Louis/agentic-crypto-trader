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
    # the opening bar of each token has no valid prior price, so any return computed there is a data
    # artifact (e.g. ZEC's spurious +253.8% first bar — it desynced r_alt from the candle prices by
    # +173pt). Zero each token's first valid return → r_alt reconciles to the candles (audit #2).
    for col in returns.columns:
        first = returns[col].first_valid_index()
        if first is not None:
            returns.loc[first, col] = 0.0
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
    raw_actions, records = [], []
    shaping = {"giveback": 0.0, "realized": 0.0, "turnover": 0.0}
    done = False
    while not done:
        norm = vecnorm.normalize_obs(obs.reshape(1, -1)) if vecnorm is not None else obs.reshape(1, -1)
        action, _ = model.predict(norm, deterministic=True)
        raw_actions.append(float(np.asarray(action).reshape(-1).sum()))  # total allocation weight
        obs, _, done, info = env.step(np.asarray(action).reshape(-1))
        records.append({"time": info["time"], "weights": info["weights"],
                        "trades_usd": info["trades_usd"], "trade_fees": info["trade_fees"]})
        fees += info["cost"]
        trades += 1 if info["cost"] > 0 else 0
        for key in shaping:
            shaping[key] += info[key]
    # union of every token traded (the universe rotates when rerank_every>0), final leaders first,
    # so the bundle's per-token candles/markers cover all names the agent touched
    seen = {t for r in records for t in r["weights"]}
    universe = list(env.tokens) + [t for t in sorted(seen) if t not in env.tokens]
    return (np.asarray(env.equity_curve, dtype=float), fees, trades, raw_actions, records,
            universe, shaping)


def _load_token_ohlcv(token):
    dirs = glob.glob(os.path.join("data", "ohlcv", "hour_1", f"{token}_*"))
    if not dirs:
        return None
    files = sorted(glob.glob(os.path.join(dirs[0], "*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def build_volume_panel(tokens, index):
    """Per-token hourly volume aligned to `index` (the returns timestamps) — for volume-spike signals."""
    cols = {}
    for t in tokens:
        oh = _load_token_ohlcv(t)
        if oh is None:
            continue
        ts = oh["timestamp"].to_numpy()
        ts = (ts // 1000) if ts.max() > 1e12 else ts
        cols[t] = pd.Series(oh["volume"].to_numpy(), index=ts).reindex(index).fillna(0.0)
    return pd.DataFrame(cols, index=index)


def build_ohlc_frac_panels(tokens, index):
    """Per-token intra-bar shape panels aligned to `index`: `low_frac = low/close` (recovers the
    bar's true LOW from the close-indexed env prices, so a resting stop can fill where the price
    PATH crossed it — the Q −53%-in-one-bar hole) and `high_frac = close/high` (1.0 = closed at
    the high; small = a big upper rejection wick — the dump started inside the trigger bar).
    Missing bars -> 1.0 (no intra-bar information => neither guard can fire spuriously)."""
    lows, highs = {}, {}
    for t in tokens:
        oh = _load_token_ohlcv(t)
        if oh is None:
            continue
        ts = oh["timestamp"].to_numpy()
        ts = (ts // 1000) if ts.max() > 1e12 else ts
        close = oh["close"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            lf = np.where(close > 0, oh["low"].to_numpy() / close, 1.0)
            hi = oh["high"].to_numpy()
            hf = np.where(hi > 0, close / hi, 1.0)
        lows[t] = pd.Series(lf, index=ts).reindex(index).fillna(1.0).clip(0.01, 1.0)
        highs[t] = pd.Series(hf, index=ts).reindex(index).fillna(1.0).clip(0.0, 1.0)
    return pd.DataFrame(lows, index=index), pd.DataFrame(highs, index=index)


def _price_at(win, tm):
    if win.empty:
        return None
    idx = (win["timestamp"] - tm).abs().idxmin()
    return float(win.loc[idx, "close"])


def build_portfolio_artifacts(records, universe, t0, t1):
    """From the per-step eval records → weights-over-time + per-token candles & buy/sell markers."""
    from trader.report import apentic as ap
    weights = [{"time": r["time"], "weights": r["weights"]} for r in records]
    token_candles, token_trades = {}, {}
    for t in universe:
        ohlcv = _load_token_ohlcv(t)
        if ohlcv is None:
            token_candles[t], token_trades[t] = [], []
            continue
        win = ohlcv[(ohlcv["timestamp"] >= t0) & (ohlcv["timestamp"] <= t1)]
        token_candles[t] = ap.candles_from_ohlcv(win)
        # the env trades on a returns-index price `_px` (base 1.0 at the series start); scale it to the
        # real display basis by k_t = the token's first close, so qty/PnL use the SAME basis the env
        # executed in (immune to any returns-vs-OHLCV divergence) while markers still sit on the candles.
        k_t = float(ohlcv["close"].iloc[0]) if len(ohlcv) else 1.0
        marks = []
        for r in records:
            if "fills" in r:                                  # event-env: TRUE per-fill time + _px-basis price
                for f in r["fills"]:
                    if f["token"] != t:
                        continue
                    marks.append({"time": f["time"], "price": f["px"] * k_t,
                                  "side": "buy" if f["usd"] > 0 else "sell", "usd": abs(f["usd"]),
                                  "fee": f["fee"], "weight": r["weights"].get(t, 0.0)})
            else:                                             # legacy path (rung-0 / PortfolioEnv records)
                tr = r["trades_usd"].get(t)
                if tr is None:
                    continue
                marks.append({"time": r["time"], "price": _price_at(win, r["time"]),
                              "side": "buy" if tr > 0 else "sell", "usd": abs(tr),
                              "fee": r.get("trade_fees", {}).get(t, 0.0),
                              "weight": r["weights"].get(t, 0.0)})
        token_trades[t] = marks
    return weights, token_candles, token_trades


def trade_stats(token_trades):
    """Per-token FIFO round-trips → trade-level metrics.

    `avg_win_pct`/`avg_loss_pct` are **return fractions** (round-trip PnL ÷ entry notional,
    consistent with `total_return_pct`), NOT dollars — fixes the misnamed $-as-% field.
    """
    usd, pct, n = [], [], 0
    for marks in token_trades.values():
        n += len(marks)
        lots = []                                          # FIFO open buys: [price, qty, fee]
        for m in marks:
            price = m.get("price") or 0.0
            if not price:
                continue
            qty, fee = m["usd"] / price, m.get("fee", 0.0)
            if m["side"] == "buy":
                lots.append([price, qty, fee])
            elif lots:                                     # sell closes the oldest open buy
                bp, bq, bfee = lots[0]
                q = min(qty, bq)
                pnl = (price - bp) * q - fee - bfee
                notional = bp * q
                usd.append(pnl)                            # real $ PnL (for win_rate / profit_factor)
                rt = pnl / notional if notional > 1e-6 else 0.0
                pct.append(max(-1.0, min(rt, 10.0)))       # clip dust/extreme-price fee artifacts
                lots.pop(0) if qty >= bq else lots[0].__setitem__(1, bq - qty)
    wins = [p for p in usd if p > 0]
    losses = [p for p in usd if p <= 0]
    return {
        "total_trades": n,
        "win_rate": len(wins) / max(len(usd), 1),
        "profit_factor": float(sum(wins) / (abs(sum(losses)) + 1e-10)),
        "avg_win_pct": float(np.mean([p for p in pct if p > 0])) if any(p > 0 for p in pct) else 0.0,
        "avg_loss_pct": float(np.mean([p for p in pct if p <= 0])) if any(p <= 0 for p in pct) else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--run-id", default="ppo-exposure")
    p.add_argument("--out", default=None, help="artifact dir (default: ./runs-rl/<run-id>)")
    p.add_argument("--publish-target", default=None, help="default: env APENTIC_PUBLISH_TARGET")
    p.add_argument("--action-mode", default="exposure", choices=["exposure", "weights"],
                   help="exposure=scalar dial on vol-top8 (C); weights=per-token allocation (B)")
    p.add_argument("--reward-mode", default="sharpe",
                   choices=["sharpe", "giveback", "realized", "turnover", "composite"],
                   help="reward shaping: sharpe=control (MTM Sharpe); giveback=penalize surrendered "
                        "unrealized gains; realized=reward locked-in profit; turnover=penalize churn; "
                        "composite=stack all three by their lambdas (set a lambda to 0 to disable)")
    p.add_argument("--rich-obs", action="store_true",
                   help="add per-token unrealized-gain + distance-below-recent-high observations")
    p.add_argument("--rung0-obs", action="store_true",
                   help="add rung-0's per-token signals (ignite flag, volume-surge ratio, price/EMA "
                        "cushion) to the observation (weights mode only) — train the policy WITH the "
                        "rung-0 rules as inputs. rung-0 params are fixed; only the random window varies")
    p.add_argument("--gb-lambda", type=float, default=10.0, help="giveback penalty weight")
    p.add_argument("--turn-lambda", type=float, default=0.5, help="turnover penalty weight")
    p.add_argument("--realized-lambda", type=float, default=10.0, help="realized-profit reward weight")
    p.add_argument("--dd-lambda", type=float, default=2.0,
                   help="drawdown-proximity penalty weight (the brake toward the DQ gate)")
    p.add_argument("--rerank-every", type=int, default=0,
                   help="re-pick the vol-top-k every N rebalances (0=once at episode start; 1=daily). "
                        "Microcap vol is bursty/rotational so a fixed-for-the-episode universe goes stale")
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
    volume = None
    if args.rung0_obs:                                         # per-token volume panel for rung-0 signals
        volume = build_volume_panel(list(returns.columns), returns.index)
        print(f"[rung0-obs] built volume panel: {volume.shape[1]} tokens x {volume.shape[0]} bars")
    env_kwargs = dict(step_bars=args.step_bars, episode_steps=args.episode_steps,
                      warmup=168, action_mode=args.action_mode, seed=args.seed,
                      reward_mode=args.reward_mode, rich_obs=args.rich_obs,
                      gb_lambda=args.gb_lambda, turn_lambda=args.turn_lambda,
                      realized_lambda=args.realized_lambda, dd_lambda=args.dd_lambda,
                      rerank_every=args.rerank_every, volume=volume, rung0_obs=args.rung0_obs)

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
    equity, fees, trades, raw_actions, records, universe, shaping = evaluate_policy(
        model, venv, eval_r, btc_close, liq, env_kwargs)
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

    # ---- portfolio bundle: allocation-over-time + per-token candles & buy/sell markers ----
    weights, token_candles, token_trades = build_portfolio_artifacts(
        records, universe, int(eval_r.index[0]), int(eval_r.index[-1]))

    # accurate trade-level stats from the real per-token markers: the panel was counting rebalance
    # *days* as "trades" (and win/loss were empty / in $). FIFO round-trips per token, return-%.
    metrics.update(trade_stats(token_trades))
    metrics["reward_mode"] = args.reward_mode
    metrics["eval_giveback"] = shaping["giveback"]        # mode diagnostics for the cross-run compare
    metrics["eval_realized_usd"] = shaping["realized"]
    metrics["eval_turnover_usd"] = shaping["turnover"]
    # full reproducibility provenance baked into every bundle — the experiment ledger reads this so
    # we can always reconstruct the exact (code + config) that produced a result (the TradeSim lesson)
    import subprocess
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).stdout.strip()
    except Exception:  # noqa: BLE001
        sha = "unknown"
    metrics["provenance"] = {
        "git_commit": sha, "action_mode": args.action_mode, "reward_mode": args.reward_mode,
        "rich_obs": args.rich_obs, "timesteps": args.timesteps, "seed": args.seed,
        "n_envs": args.n_envs, "step_bars": args.step_bars, "episode_steps": args.episode_steps,
        "ent_coef": args.ent_coef, "lr": args.lr, "eval_split": args.eval_split,
        "gb_lambda": args.gb_lambda, "turn_lambda": args.turn_lambda,
        "realized_lambda": args.realized_lambda, "dd_lambda": args.dd_lambda,
        "rerank_every": args.rerank_every, "rung0_obs": args.rung0_obs,
    }
    entry = ap.export_portfolio_run(
        out, args.run_id, equity=eq_series, metrics=metrics, weights=weights,
        token_candles=token_candles, token_trades=token_trades, universe=universe,
        model_name=f"PPO {args.action_mode}/{args.reward_mode}{'+rung0' if args.rung0_obs else ''} "
                   f"s{args.seed} ({args.timesteps:,} steps)",
        action_mode=args.action_mode, regime=args.eval_split,
        timestamp=datetime.now(timezone.utc).isoformat())

    target = args.publish_target or config.get("APENTIC_PUBLISH_TARGET")
    if target:
        ap.publish_run(os.path.join(out, args.run_id), args.run_id, entry, target,
                       cloudfront_dist_id=config.get("APENTIC_CLOUDFRONT_DIST_ID"))

    write_progress(out, state="complete", run_id=args.run_id,
                   total_return=report.total_return_pct, sharpe=report.sharpe_ratio,
                   max_drawdown=report.max_drawdown_pct, trades=trades)
    print(f"[train_rl] {args.run_id} [{args.reward_mode}]: return {report.total_return_pct:+.1%}, "
          f"Sharpe {report.sharpe_ratio:.2f}, maxDD {report.max_drawdown_pct:.1%}, "
          f"trades {metrics['total_trades']}, win {metrics['win_rate']:.0%}, "
          f"PF {metrics['profit_factor']:.2f}, avg win {metrics['avg_win_pct']:+.1%}, "
          f"turnover ${shaping['turnover']:,.0f}, realized ${shaping['realized']:,.0f}")


if __name__ == "__main__":
    main()
