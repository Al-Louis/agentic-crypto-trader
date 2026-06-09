"""STALE — predates the event-driven rung0 (uses the old daily weights-fn + no volume signal). Update
to `run_rung0` + `build_volume_panel` before re-running.

Rung-0.5 threshold sweep: rung-0 is dialed too conservative (uses only ~12% of a 30% DD budget).
Grid-search the four discipline knobs on VAL, pick the most aggressive config that stays under the
gate, then VERIFY on the frozen TEST split (selection happens on val only — test is the honest read).

    python scripts/sweep_rung0.py
"""
from __future__ import annotations

import itertools
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

from train_rl import load_data, time_split  # noqa: E402
from trader.sim.backtest import run_xs_backtest  # noqa: E402
from trader.strategy.candidate import build_candidate  # noqa: E402
from trader.strategy.rung0 import build_rung0  # noqa: E402

WARMUP, REBAL = 168, 24
VAL_CEILING = 0.28          # select configs under this val maxDD (buffer below the 30% gate)


def backtest(returns, liq, fn):
    res = run_xs_backtest(returns, fn, liq, rebalance_every=REBAL, warmup=WARMUP)
    eq = res["equity"]
    ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    dd = float(((eq.cummax() - eq) / eq.cummax()).max())
    return ret, dd, res["total_turnover"]


def main():
    returns, btc, anchor, liq = load_data()
    _, val_r, test_r = time_split(returns)

    grid = {
        "stop_k": [0.10, 0.15, 0.20, 0.30],       # trailing stop: wider = let winners run further
        "max_weight": [0.25, 0.40, 0.60, 1.0],    # per-token cap: higher = more concentration
        "cooldown": [0, 1, 2],                    # rebalances out after an exit
        "breakout_n": [24, 48, 72],               # new-high window for entry
    }
    keys = list(grid)
    rows = []
    for combo in itertools.product(*grid.values()):
        kw = dict(zip(keys, combo))
        ret, dd, to = backtest(val_r, liq, build_rung0(val_r, **kw))
        rows.append((kw, ret, dd, to))

    legal = sorted([r for r in rows if r[2] < VAL_CEILING], key=lambda r: -r[1])
    print(f"{len(rows)} configs swept on VAL; {len(legal)} under {VAL_CEILING:.0%} val maxDD\n")
    print(f"  top by VAL return (under the DD ceiling):")
    print(f"  {'stop_k':>7}{'maxW':>6}{'cool':>5}{'brkN':>5}{'val ret':>9}{'val DD':>8}{'turn$':>9}")
    for kw, ret, dd, to in legal[:8]:
        print(f"  {kw['stop_k']:>7.2f}{kw['max_weight']:>6.2f}{kw['cooldown']:>5}{kw['breakout_n']:>5}"
              f"{ret*100:>+8.1f}%{dd*100:>7.1f}%{to:>9,.0f}")

    pick = legal[0][0]
    print(f"\nPICK (best val return, val maxDD < {VAL_CEILING:.0%}): {pick}")
    print(f"  {'split':6}{'return':>9}{'maxDD':>8}{'turnover$':>11}")
    for nm, r in [("VAL", val_r), ("TEST", test_r)]:
        ret, dd, to = backtest(r, liq, build_rung0(r, **pick))
        flag = "  <-- frozen OOS verdict" if nm == "TEST" else ""
        print(f"  {nm:6}{ret*100:>+8.1f}%{dd*100:>7.1f}%{to:>11,.0f}{flag}")

    print(f"\n  reference on TEST:")
    for nm, fn in [("rung0 default", build_rung0(test_r)),
                   ("vol-top8 trend50", build_candidate(test_r, btc, overlay="trend50"))]:
        ret, dd, to = backtest(test_r, liq, fn)
        print(f"  {nm:18}{ret*100:>+8.1f}%{dd*100:>7.1f}%{to:>11,.0f}")


if __name__ == "__main__":
    main()
