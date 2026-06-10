"""Make-or-break probe (exp4 blocker, Q3): is there alpha to discriminate AMONG rung-0's
SELECTED entries — not just across all ignitions?

`probe_obs_alpha` measures IC over ALL ignitions (`env._ignite`). But the agent only ever sizes
the rung-0-SELECTED subset: in-universe (vol-top-k), `ignite & cooled & reclaimed & flat & armed`.
And `_ignite` itself already gates `cush>0 & rising>0 & surge & ema_up`. If rung-0's trend-gate
already harvested the cush-alpha, the IC *within* the selected subset is ~0 and entry-sizing has no
headroom — the lever must move to exit/selection. This probe measures IC at THREE nested levels so
we know exactly where (if anywhere) the discrimination headroom lives.

  L0  all ignitions (cush>0 already, by _ignite)            <- what probe_obs_alpha scored (+0.246)
  L1  + in causal vol-top-k universe                        <- the agent's candidate pool
  L2  + cooled & reclaimed & replayed-as-flat (entry gate)  <- what the agent actually sizes

OOS temporal holdout within train (fit early, test late), per level. Causal: walks the env's event
engine to reproduce the L2 entry set exactly (cooldown/dead-zone are path-dependent). No training.

  python scripts/probe_subset_ic.py [--horizon 24] [--holdout 0.3]
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

import numpy as np  # noqa: E402

from train_rl import build_volume_panel, load_data, time_split  # noqa: E402
from trader.train.event_env import EventRungEnv  # noqa: E402

WARMUP = 168
FEAT_NAMES = ["cush", "surge", "btc_trend"]


def _oos_ic(X, y, t, holdout):
    """Temporal-OOS univariate IC for each feature, plus the OLS-combined IC. Fit on the earliest
    (1-holdout) by event time, test on the latest holdout."""
    if len(y) < 20:
        return None
    order = np.argsort(t)
    X, y = X[order], y[order]
    cut = int(len(y) * (1 - holdout))
    Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]
    out = {}
    for k, nm in enumerate(FEAT_NAMES):
        if np.std(Xte[:, k]) < 1e-12 or np.std(yte) < 1e-12:
            out[nm] = 0.0
        else:
            out[nm] = float(np.corrcoef(Xte[:, k], yte)[0, 1])
    A = np.column_stack([np.ones(len(Xtr)), Xtr])
    coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    pred = np.column_stack([np.ones(len(Xte)), Xte]) @ coef
    combined = (0.0 if np.std(pred) < 1e-12 or np.std(yte) < 1e-12
                else float(np.corrcoef(pred, yte)[0, 1]))
    top, bot = yte[pred >= np.median(pred)], yte[pred < np.median(pred)]
    return {"n": len(y), "n_test": len(yte), "uni": out, "combined": combined,
            "mean_fwd": float(y.mean()), "spread": float(top.mean() - bot.mean())}


def collect_L2_entries(env, H):
    """Replay the env's event engine on the full train window with the rule's sizing (so cooldown +
    dead-zone advance exactly as in training), recording each (bar, tok) the agent is actually
    PROMPTED to size — the realized entry set, throttled by cash/rotation. Returns rows/fwd/times.
    NOTE: this is funding-constrained (Q4); for the pure alpha-headroom question use L2gate below."""
    env.reset(start=env._min_start)
    bt = (env.btc.to_numpy() / env.btc_ema.to_numpy() - 1.0)
    rows, ys, ts = [], [], []
    seen = set()
    done = False
    while not done:
        etype, tok = env._pending
        bar = env.bar
        if etype == "entry" and tok is not None and (bar, tok) not in seen and bar < env.n_bars - H:
            seen.add((bar, tok))
            j = env.col_ix[tok]
            if env._px[bar, j] > 0:
                rows.append([env._cush[bar, j], env._surge[bar, j], float(bt[bar])])
                ys.append(env._px[bar + H, j] / env._px[bar, j] - 1.0)
                ts.append(bar)
        # size as the rule would (m s.t. entry_frac == rule's 0.20) to keep the path realistic
        a = 2.0 * (env.rule_entry_frac / env.max_entry_frac) - 1.0 if etype == "entry" else 1.0
        _, _, done, _ = env.step(a)
    return np.array(rows, float), np.array(ys, float), np.array(ts, float)


def collect_L2gate(env, H):
    """The entry-GATE candidate pool, decoupled from cash/rotation: every (bar, tok) where the
    rung-0 entry predicate fires for an IN-UNIVERSE token — `ignite & cooled & reclaimed & flat &
    armed` — tracking cooldown/dead-zone/armed per token causally, but NOT throttling by cash. This
    is the discrimination-headroom set: what the agent COULD size if funding were unconstrained
    (Q4). Isolates the Q3 alpha question from the Q4 funding bottleneck."""
    bt = (env.btc.to_numpy() / env.btc_ema.to_numpy() - 1.0)
    cool = {t: -10 ** 9 for t in env.universe}
    prior = {t: None for t in env.universe}
    armed = {t: True for t in env.universe}
    held = {t: None for t in env.universe}        # entry_bar while "in pos" (rule's fixed hold proxy)
    rows, ys, ts = [], [], []
    for bar in range(env.warmup, env.n_bars - H):
        for t in env.universe:
            j = env.col_ix[t]
            # rule-like exit bookkeeping so cooldown/dead-zone/armed advance realistically
            if held[t] is not None:
                ref = env._px[held[t], j]
                if env._px[bar, j] < ref * (1.0 - env.stop_k) or env._cush[bar, j] < 0.0:
                    cool[t], prior[t], held[t] = bar, env._px[held[t], j], None
            if not env._ignite[bar, j]:
                armed[t] = True
                continue
            if held[t] is None and armed[t]:
                cooled = (bar - cool[t]) >= env.cooldown
                po = prior[t]
                reclaimed = po is None or env._px[bar, j] > po
                if cooled and reclaimed and env._px[bar, j] > 0:
                    rows.append([env._cush[bar, j], env._surge[bar, j], float(bt[bar])])
                    ys.append(env._px[bar + H, j] / env._px[bar, j] - 1.0)
                    ts.append(bar)
                    armed[t], held[t] = False, bar      # rule enters -> consume edge, mark held
    return np.array(rows, float), np.array(ys, float), np.array(ts, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--holdout", type=float, default=0.3)
    args = ap.parse_args()
    H = args.horizon

    returns, btc, _, liq = load_data()
    train_r, _, _ = time_split(returns)
    vol = build_volume_panel(list(returns.columns), returns.index)
    env = EventRungEnv(train_r, btc, liq, volume=vol, k=8, warmup=WARMUP,
                       episode_bars=len(train_r) - WARMUP - 1, reward_mode="absolute",
                       fwd_horizon=H, seed=0)
    bt = (env.btc.to_numpy() / env.btc_ema.to_numpy() - 1.0)
    n = env.n_bars

    # L0: every ignition (any token). L1: ignition AND token in the episode's causal universe.
    env.reset(start=env._min_start)                  # sets env.universe (causal vol-top-k at start)
    uni = set(env.universe)  # fixed for the full-window single episode (picked at start)
    L0, L1 = ([], [], []), ([], [], [])
    for j in range(len(env.cols)):
        tok = env.cols[j]
        for bar in range(WARMUP, n - H):
            if env._ignite[bar, j] and env._px[bar, j] > 0:
                fwd = env._px[bar + H, j] / env._px[bar, j] - 1.0
                row = [env._cush[bar, j], env._surge[bar, j], float(bt[bar])]
                L0[0].append(row); L0[1].append(fwd); L0[2].append(bar)
                if tok in uni:
                    L1[0].append(row); L1[1].append(fwd); L1[2].append(bar)

    # L2gate: the entry-gate candidate pool (cooldown/dead-zone tracked, cash NOT throttled) -> the
    # pure alpha-headroom set. L2real: the funding-constrained realized entries (Q4 bottleneck).
    Xg, yg, tg = collect_L2gate(env, H)
    X2, y2, t2 = collect_L2_entries(env, H)

    levels = [
        ("L0 all ignitions", np.array(L0[0], float), np.array(L0[1], float), np.array(L0[2], float)),
        ("L1 +in-universe", np.array(L1[0], float), np.array(L1[1], float), np.array(L1[2], float)),
        ("L2gate +cooled&reclaimed", Xg, yg, tg),
        ("L2real +cash-throttled", X2, y2, t2),
    ]
    print(f"horizon {H}b | holdout {args.holdout:.0%} | universe(k=8): {sorted(uni)}\n")
    print(f"{'level':<28}{'n':>6}{'nTest':>7}{'meanFwd':>9}{'cush':>8}{'surge':>8}"
          f"{'btcT':>8}{'comb':>8}{'spread':>9}")
    for name, X, y, t in levels:
        r = _oos_ic(X, y, t, args.holdout)
        if r is None:
            print(f"{name:<28}{len(y):>6}   (too few events)")
            continue
        print(f"{name:<28}{r['n']:>6}{r['n_test']:>7}{r['mean_fwd']*100:>+8.2f}%"
              f"{r['uni']['cush']:>+8.3f}{r['uni']['surge']:>+8.3f}{r['uni']['btc_trend']:>+8.3f}"
              f"{r['combined']:>+8.3f}{r['spread']*100:>+8.2f}%")
    print("\nREAD: if L2 cush-IC / combined-IC stays clearly >0 (and spread>0) -> headroom WITHIN")
    print("rung-0's selected entries is real -> fix demean+gate, proceed with entry-sizing.")
    print("If L2 IC ~0 while L0 was +0.25 -> rung-0's cush>0 gate already harvested it -> entry-sizing")
    print("has no headroom -> move the RL lever to exit-override or selection.")


if __name__ == "__main__":
    main()
