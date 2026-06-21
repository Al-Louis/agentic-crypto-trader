"""P-EMA-CONSOLIDATION — is the rung-0 EMA-break exit a FALSE break specifically when the token is in
TIGHT CONSOLIDATION? (user hypothesis 2026-06-19: sideways-for-X => tight EMA => shallow noise-dips trip
the break => shaken out before the breakout, the FF case). DISTINCT from P-EMA-COND (that gated on P&L);
here the conditioning variable is the token's RECENT VOLATILITY (consolidation) and the DEPTH of the break.

Characterizes every EMA-break SELL the rung-0 rule makes over the OOS (val+test) cold weeks and asks:
do EMA-breaks in LOW recent-vol (consolidation) recover (positive forward TERMINAL return => should have
held) while EMA-breaks in HIGH vol keep falling (real break => keep cutting)? If consolidation separates
them, the conditional "ignore EMA-break while consolidating" has alpha to capture. TERMINAL return only
(NOT max-run-up — the P-CAPACITY selection catch). Torch-free, laptop-local; rung-0 via the env (idx-0).
"""
from __future__ import annotations
import os, sys; sys.path[:0] = ["scripts", "src"]
import numpy as np, pandas as pd
from train_rl import load_data, build_volume_panel, time_split
from trader.train import weekly_eval as we
from trader.train.event_env import EventRungEnv

WARMUP, VOLWIN = 168, 24
CORE = dict(k=10, vol_mult=2.0, warmup=WARMUP, universe_mode="voltopk", action_mode="discrete",
            n_action_levels=4, rule_default=True, stop_k=0.25, cooldown=48, exit_commit=12,
            dust_usd=10.0, tp_rungs=[0.25, 0.5, 1.0, 2.0], loss_floor=0.2, vol_target=0.005,
            cap_floor=0.02, max_entry_frac=0.34, harvest_obs=True, det_blacklist=672,
            reward_mode="entry_forward", fwd_horizon=24)


def boot_mean_ci(v, n=2000, seed=0):
    v = np.asarray([x for x in v if x == x], float)
    if v.size < 2:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, v.size, (n, v.size))].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    returns, btc, _a, liq = load_data()
    vol = build_volume_panel(list(returns.columns), returns.index)
    tr, va, te = time_split(returns)
    val_start, test_start = int(va.index[0]), int(te.index[0])

    ev = []  # (split, token, trail_vol, cushion_at_break, fwd24, fwd48)
    for ws, win in we.cold_week_windows(returns):
        sp = we.split_label(ws, val_start, test_start)
        if sp == "train":
            continue
        kw = {k: v for k, v in CORE.items() if k != "warmup"}
        env = EventRungEnv(win, btc, liq, volume=vol, episode_bars=len(win) - WARMUP - 1,
                           warmup=WARMUP, record_trace=True, **kw)
        env.reset(start=WARMUP)
        px, cush = env._px, env._cush
        rstd = win.rolling(VOLWIN, min_periods=8).std().to_numpy()
        posix = {int(t): i for i, t in enumerate(win.index)}
        done = False
        while not done:
            _o, _r, done, info = env.step([0])
            for (tok, usd, _c, ft, _px, rsn, _ob) in info.get("trades", []):
                if usd < 0 and rsn == "EMA_BREAK":
                    b = posix.get(int(ft)); j = env.col_ix[tok]
                    if b is None or b < VOLWIN:
                        continue
                    f24 = px[b + 24, j] / px[b, j] - 1.0 if b + 24 < len(win) else np.nan
                    f48 = px[b + 48, j] / px[b, j] - 1.0 if b + 48 < len(win) else np.nan
                    ev.append((sp, tok, float(rstd[b, j]), float(cush[b, j]), f24, f48))

    print(f"EMA-break SELLs over OOS cold weeks: {len(ev)} (val {sum(1 for e in ev if e[0]=='val')}, "
          f"test {sum(1 for e in ev if e[0]=='test')})")
    if len(ev) < 10:
        print("too few events"); return
    vols = np.array([e[2] for e in ev])
    q1, q2 = np.percentile(vols, [33, 67])
    print(f"\ntrailing-{VOLWIN}h vol terciles: low<{q1*100:.2f}%  mid  high>{q2*100:.2f}%")
    print(f"\n{'bucket':18}{'n':>4}{'fwd24 mean[CI]':>26}{'win24':>7}{'fwd48 mean[CI]':>26}{'win48':>7}")

    def show(name, sub):
        f24 = [e[4] for e in sub]; f48 = [e[5] for e in sub]
        m24, lo24, hi24 = boot_mean_ci(f24); m48, lo48, hi48 = boot_mean_ci(f48)
        w24 = np.mean([x > 0 for x in f24 if x == x]) if any(x == x for x in f24) else float("nan")
        w48 = np.mean([x > 0 for x in f48 if x == x]) if any(x == x for x in f48) else float("nan")
        print(f"  {name:16}{len(sub):>4}{f'{m24*100:+.1f} [{lo24*100:+.0f},{hi24*100:+.0f}]':>26}"
              f"{w24*100:>6.0f}%{f'{m48*100:+.1f} [{lo48*100:+.0f},{hi48*100:+.0f}]':>26}{w48*100:>6.0f}%")

    show("LOW-vol (consol)", [e for e in ev if e[2] < q1])
    show("MID-vol", [e for e in ev if q1 <= e[2] <= q2])
    show("HIGH-vol", [e for e in ev if e[2] > q2])
    print("  -- by break DEPTH (cushion at break) --")
    cu = np.array([e[3] for e in ev]); cmed = np.percentile(cu, 50)
    show("SHALLOW break", [e for e in ev if e[3] >= cmed])   # cushion close to 0 (barely below EMA)
    show("DEEP break", [e for e in ev if e[3] < cmed])
    print("  -- the user's FF cell: LOW-vol AND SHALLOW --")
    show("consol+shallow", [e for e in ev if e[2] < q1 and e[3] >= cmed])
    print("\nREAD: if LOW-vol (and/or consol+shallow) fwd48 mean>0 with CI-lo>0 while HIGH-vol<0, the "
          "consolidation conditional separates false breaks (hold) from real breaks (cut). If all buckets "
          "<=0 / CIs straddle 0, suppressing EMA-break in consolidation does NOT add terminal EV.")


if __name__ == "__main__":
    main()
