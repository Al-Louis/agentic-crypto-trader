"""P-EMA-DEEPDIP — chase the INVERSE of the consolidation idea: do DEEP + HIGH-VOL EMA-breaks (sharp
dips far below the EMA) RECOVER (a V-bounce the rule wrongly cuts), and is that bounce capturable
WITHOUT eating a DQ-sized drawdown to get there?

P-EMA-CONSOLIDATION found the FF/consolidation cell is the WORST (keeps falling), but HIGH-vol/DEEP
breaks showed +EV terminal return (weakly, CI-lo~0, n=23). This strengthens that: more events (add the
TRAIN weeks for the price-dynamics characterization), monotonic-trend check (not one cherry-picked cell),
VAL = decision / TEST = confirmation, and the MAX ADVERSE EXCURSION (how far it drops before bouncing =
the drawdown you must survive — the real DQ constraint). Terminal return only. Torch-free.

DELIVERABLE: is "deeper+higher-vol => more +EV" a MONOTONIC, CI-supported trend on VAL (confirmed TEST),
AND is the deep/high-vol cell's drawdown bounded (mae > ~ -25%, inside stop_k) so holding is DQ-survivable?
If yes -> a DQ-aware cold-weekly counterfactual (Part B) is warranted. If the trend is non-monotonic /
n-limited / the MAE is DQ-sized, the inverse is a mirage and we say so (with the cells, not a blanket null).
"""
from __future__ import annotations
import sys; sys.path[:0] = ["scripts", "src"]
import numpy as np
from train_rl import load_data, build_volume_panel, time_split
from trader.train import weekly_eval as we
from trader.train.event_env import EventRungEnv

WARMUP, VOLWIN, H = 168, 24, 48
CORE = dict(k=10, vol_mult=2.0, universe_mode="voltopk", action_mode="discrete", n_action_levels=4,
            rule_default=True, stop_k=0.25, cooldown=48, exit_commit=12, dust_usd=10.0,
            tp_rungs=[0.25, 0.5, 1.0, 2.0], loss_floor=0.2, vol_target=0.005, cap_floor=0.02,
            max_entry_frac=0.34, harvest_obs=True, det_blacklist=672, reward_mode="entry_forward",
            fwd_horizon=24)


def ci(v, n=2000, seed=0):
    v = np.asarray([x for x in v if x == x], float)
    if v.size < 2:
        return (float("nan"), float("nan"), float("nan"), v.size)
    r = np.random.default_rng(seed)
    m = v[r.integers(0, v.size, (n, v.size))].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), v.size


def main():
    returns, btc, _a, liq = load_data()
    vol = build_volume_panel(list(returns.columns), returns.index)
    tr, va, te = time_split(returns)
    val_start, test_start = int(va.index[0]), int(te.index[0])

    ev = []  # (split, cushion(depth), trail_vol, fwd48, mae48)
    for ws, win in we.cold_week_windows(returns):
        sp = we.split_label(ws, val_start, test_start)
        kw = {k: v for k, v in CORE.items()}
        env = EventRungEnv(win, btc, liq, volume=vol, episode_bars=len(win) - WARMUP - 1,
                           warmup=WARMUP, record_trace=True, **kw)
        env.reset(start=WARMUP)
        px, cush = env._px, env._cush
        rstd = win.rolling(VOLWIN, min_periods=8).std().to_numpy()
        pix = {int(t): i for i, t in enumerate(win.index)}
        done = False
        while not done:
            _o, _r, done, info = env.step([0])
            for (tok, usd, _c, ft, _p, rsn, _ob) in info.get("trades", []):
                if usd < 0 and rsn == "EMA_BREAK":
                    b = pix.get(int(ft)); j = env.col_ix[tok]
                    if b is None or b < VOLWIN or b + H >= len(win):
                        continue
                    fwd = px[b + H, j] / px[b, j] - 1.0
                    mae = px[b:b + H, j].min() / px[b, j] - 1.0           # worst drop before b+H
                    ev.append((sp, float(cush[b, j]), float(rstd[b, j]), float(fwd), float(mae)))

    n_by = {s: sum(1 for e in ev if e[0] == s) for s in ("train", "val", "test")}
    print(f"EMA-break SELLs (b+{H}h on-window): {len(ev)}  by split {n_by}")
    if len(ev) < 20:
        print("too few"); return

    cush = np.array([e[1] for e in ev]); vols = np.array([e[2] for e in ev])
    # depth quartiles: most-negative cushion = deepest (Q4=deepest); vol quartiles: Q4=highest
    cq = np.percentile(cush, [25, 50, 75]); vq = np.percentile(vols, [25, 50, 75])

    def depth_q(c): return 4 - sum(c <= x for x in cq)        # deepest (most neg) -> 4
    def vol_q(v): return 1 + sum(v > x for x in vq)           # highest -> 4

    print(f"\n--- MONOTONIC TREND: fwd{H} terminal return by BREAK-DEPTH quartile (Q4=deepest) ---")
    print(f"{'split':6}{'Q1(shallow)':>14}{'Q2':>14}{'Q3':>14}{'Q4(deep)':>14}")
    for sp in ("train", "val", "test", "all"):
        sub = ev if sp == "all" else [e for e in ev if e[0] == sp]
        row = f"  {sp:5}"
        for q in (1, 2, 3, 4):
            m, lo, hi, n = ci([e[3] for e in sub if depth_q(e[1]) == q])
            row += f"{(f'{m*100:+.1f}(n{n})' if n else '-'):>14}"
        print(row)

    print(f"\n--- the DEEP & HIGH-VOL cell (depth Q4 AND vol Q4) — fwd{H} + drawdown-to-survive (MAE) ---")
    print(f"{'split':6}{'n':>4}{'fwd48 mean[CI]':>24}{'win':>6}{'MAE mean':>10}{'MAE worst':>11}")
    ok = True
    for sp in ("train", "val", "test", "all"):
        sub = [e for e in ev if (sp == "all" or e[0] == sp) and depth_q(e[1]) == 4 and vol_q(e[2]) == 4]
        if not sub:
            print(f"  {sp:5}{0:>4}  (none)"); continue
        m, lo, hi, n = ci([e[3] for e in sub])
        win = np.mean([e[3] > 0 for e in sub])
        mae_mean = np.mean([e[4] for e in sub]); mae_worst = min(e[4] for e in sub)
        print(f"  {sp:5}{n:>4}{f'{m*100:+.1f} [{lo*100:+.0f},{hi*100:+.0f}]':>24}{win*100:>5.0f}%"
              f"{mae_mean*100:>9.0f}%{mae_worst*100:>10.0f}%")
        if sp == "val" and not (lo > 0):
            ok = False

    print(f"\nVERDICT: VAL deep&high-vol CI-lo>0 = {ok}. Need: monotonic deeper->+EV trend on VAL (and not "
          f"contradicted on TEST), AND MAE bounded (worst > ~-25%, inside stop_k) so the bounce is "
          f"DQ-survivable. A big negative MAE means you eat a DQ-sized drop to reach the +EV terminal -> "
          f"not capturable. n-floor 30/cell for conclusive.")


if __name__ == "__main__":
    main()
