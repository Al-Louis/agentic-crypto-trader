"""Train a PPO policy on the event-driven rung-1 env and publish its eval bundle. DESKTOP-ONLY
for training (needs torch); the eval/publish core is torch-free and laptop-testable.

The agent learns rung-0's DISCRETION (entry sizing, exit override) on top of rung-0's event timing
(see trader.train.event_env). Trains on random WEEKLY windows of the train split; evaluates one
long episode on the held-out split; self-publishes the Apentic bundle with the real intra-day
markers, and records the rung-0 RULE's return on the same window as the baseline (does learned
discretion beat the hand-coded version?).

  python scripts/train_event.py --timesteps 1000000 --n-envs 8 --seed 0 --run-id ppo-event-s0
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from remote_train.progress import write_progress  # noqa: E402
from trader import config  # noqa: E402
from trader.report import apentic as ap  # noqa: E402
from trader.sim.metrics import PerformanceMetrics  # noqa: E402

HOURS_PER_YEAR = 24 * 365
WARMUP = 168


def evaluate_event_policy(predict_fn, eval_r, btc, liq, vol, env_kwargs):
    """Run one long episode over `eval_r` with `predict_fn(obs)->action`; collect per-event markers
    and the per-bar equity trace. Torch-free — works with a PPO policy or a heuristic (for tests)."""
    from trader.train.event_env import EventRungEnv
    kw = {k: v for k, v in env_kwargs.items() if k != "episode_bars"}
    env = EventRungEnv(eval_r, btc, liq, volume=vol, episode_bars=len(eval_r) - WARMUP - 1,
                       record_trace=True, **kw)
    obs = env.reset(start=WARMUP)
    records, fees, raw = [], 0.0, []
    done = False
    while not done:
        a = predict_fn(obs)
        raw.append(float(np.asarray(a).reshape(-1)[0]))
        obs, _, done, info = env.step(a)
        if info.get("trades"):
            records.append({"time": info["trade_time"], "weights": info["weights"],
                            "trades_usd": {t: u for t, u, _ in info["trades"]},
                            "trade_fees": {t: c for t, _, c in info["trades"]}})
            fees += sum(c for _, _, c in info["trades"])
    eq = pd.Series([e for _, e in env._eq_trace], index=[t for t, _ in env._eq_trace])
    universe = sorted({t for rec in records for t in rec["trades_usd"]} | set(env.universe))
    return eq, records, universe, fees, raw


def rung0_baseline_return(eval_r, liq, vol):
    """The rung-0 RULE (hand-coded discretion) on the same window — the bar the RL must clear."""
    from trader.strategy.candidate import select_vol_tokens
    from trader.strategy.rung0 import build_rung0, run_rung0
    uni = select_vol_tokens(eval_r, 8)
    eq, _, _ = run_rung0(eval_r, build_rung0(eval_r, tokens=uni, volume=vol), liq, warmup=WARMUP)
    return float(eq.iloc[-1] / eq.iloc[0] - 1.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--run-id", default="ppo-event")
    p.add_argument("--out", default=None)
    p.add_argument("--publish-target", default=None)
    p.add_argument("--episode-bars", type=int, default=168, help="weekly episodes by default")
    p.add_argument("--max-entry-frac", type=float, default=0.34)
    p.add_argument("--stop-k", type=float, default=0.25)
    p.add_argument("--cooldown", type=int, default=48)
    p.add_argument("--reward-mode", default="absolute", choices=["absolute", "relative"],
                   help="relative = reward vs the rung-0 RULE's interval return (only BEATING it scores)")
    p.add_argument("--dd-lambda", type=float, default=2.0)
    p.add_argument("--dd-soft", type=float, default=0.15, help="drawdown penalty soft knee")
    p.add_argument("--ent-coef", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-end", type=float, default=None, help="if set, linearly anneal lr -> lr-end")
    p.add_argument("--eval-split", default="val", choices=["val", "test"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    config.load_dotenv()
    out = args.out or os.path.join("runs-rl", args.run_id)
    os.makedirs(out, exist_ok=True)

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    from train_rl import build_portfolio_artifacts, build_volume_panel, load_data, time_split, trade_stats
    from trader.train.gym_env import GymEventRungEnv

    returns, btc, anchor, liq = load_data()
    train_r, val_r, test_r = time_split(returns)
    eval_r = test_r if args.eval_split == "test" else val_r
    vol = build_volume_panel(list(returns.columns), returns.index)
    env_kwargs = dict(k=8, warmup=WARMUP, max_entry_frac=args.max_entry_frac, stop_k=args.stop_k,
                      cooldown=args.cooldown, dd_lambda=args.dd_lambda, dd_soft=args.dd_soft,
                      reward_mode=args.reward_mode, seed=args.seed)

    write_progress(out, state="running", phase="setup", run_id=args.run_id, timesteps=0,
                   total=args.timesteps)

    def make_env(rank):
        def _f():
            return Monitor(GymEventRungEnv(train_r, btc, liq, volume=vol,
                                           episode_bars=args.episode_bars,
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
                               mean_reward=float(np.mean(rews)) if rews else None, history_key="curve")
            return True

    lr = args.lr
    if args.lr_end is not None:                            # linear anneal lr -> lr_end (progress: 1->0)
        lr0, lr1 = args.lr, args.lr_end
        lr = lambda pr: lr1 + (lr0 - lr1) * pr  # noqa: E731
    model = PPO("MlpPolicy", venv, verbose=0, seed=args.seed, n_steps=1024, batch_size=256,
                ent_coef=args.ent_coef, learning_rate=lr)
    model.learn(total_timesteps=args.timesteps, callback=ProgressCb())

    write_progress(out, state="running", phase="evaluate")

    def predict_fn(obs):
        norm = venv.normalize_obs(obs.reshape(1, -1))
        a, _ = model.predict(norm, deterministic=True)
        return np.asarray(a).reshape(-1)

    eq, records, universe, fees, raw = evaluate_event_policy(predict_fn, eval_r, btc, liq, vol, env_kwargs)
    print(f"[eval] events={len(raw)} action mean={np.mean(raw):.3f} min={min(raw):.3f} max={max(raw):.3f}")

    report = PerformanceMetrics.compute_all(eq.to_numpy(), steps_per_year=HOURS_PER_YEAR)
    metrics = ap.metrics_to_frontend(report)
    metrics["total_fees_paid"] = fees
    d0, d1 = int(eval_r.index[0]), int(eval_r.index[-1])
    weights, candles, trades = build_portfolio_artifacts(records, universe, d0, d1)
    metrics.update(trade_stats(trades))
    base = rung0_baseline_return(eval_r, liq, vol)
    metrics["baseline_return"] = base
    print(f"[verdict] policy {report.total_return_pct:+.1%} (Sh {report.sharpe_ratio:.2f}, "
          f"DD {report.max_drawdown_pct:.1%}) vs rung-0 rule {base:+.1%} on {args.eval_split} -> "
          f"{'BEATS' if report.total_return_pct > base else 'loses to'} the rule")

    import subprocess
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).stdout.strip()
    except Exception:  # noqa: BLE001
        sha = "unknown"
    metrics["provenance"] = {"git_commit": sha, "env": "event_rung", "timesteps": args.timesteps,
                             "seed": args.seed, "n_envs": args.n_envs, "episode_bars": args.episode_bars,
                             "max_entry_frac": args.max_entry_frac, "stop_k": args.stop_k,
                             "cooldown": args.cooldown, "reward_mode": args.reward_mode,
                             "dd_lambda": args.dd_lambda, "dd_soft": args.dd_soft,
                             "ent_coef": args.ent_coef, "lr": args.lr, "lr_end": args.lr_end,
                             "eval_split": args.eval_split}
    eq_pub = eq.iloc[::6]                                   # ~6-bar resolution for the chart
    entry = ap.export_portfolio_run(out, args.run_id, equity=eq_pub, metrics=metrics, weights=weights,
                                    token_candles=candles, token_trades=trades, universe=universe,
                                    model_name=f"PPO event-rung s{args.seed} ({args.timesteps:,} steps)",
                                    action_mode="event", regime=args.eval_split,
                                    timestamp=datetime.now(timezone.utc).isoformat())
    target = args.publish_target or config.get("APENTIC_PUBLISH_TARGET")
    if target:
        ap.publish_run(os.path.join(out, args.run_id), args.run_id, entry, target,
                       cloudfront_dist_id=config.get("APENTIC_CLOUDFRONT_DIST_ID"))
    write_progress(out, state="complete", run_id=args.run_id, total_return=report.total_return_pct,
                   sharpe=report.sharpe_ratio, max_drawdown=report.max_drawdown_pct, trades=len(trades))
    print(f"[train_event] {args.run_id}: return {report.total_return_pct:+.1%}, "
          f"Sharpe {report.sharpe_ratio:.2f}, maxDD {report.max_drawdown_pct:.1%}, "
          f"events {len(raw)}, trades {metrics['total_trades']}")


if __name__ == "__main__":
    main()
