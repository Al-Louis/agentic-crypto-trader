"""Probe a CANDLE-STRUCTURE EXIT (user idea, 2026-06-20): if the agent HOLDS an in-profit position and
the current bar is an INVERTED HAMMER (long upper wick, small body near the low — price spiked then got
rejected) or a DOJI (open~close, indecision), SELL. Both are short-term bearish. Q examples: W19 Mar-28
inverted hammer -> next-bar dump (-20% floor); W16 Mar-4 doji.

This is the EXIT complement to probe_wick.py (the ignition-bar entry filter). The honest question: for a
HELD-IN-PROFIT position, do these candles PRECEDE drops (exiting avoids the dump) or are they noise
(exiting just sells winners early)? Compares signal vs non-signal forward returns + forward worst-draw,
PER SPLIT (train/val/test) so a one-window pattern can't masquerade as edge.

Candle shape from the env's own arrays: close=_px[b], open=_px[b-1], high=_px[b]/_highf[b],
low=_px[b]*_lowf[b]. Held-in-profit proxy: an ignition fired within `--hold-win` bars AND price is up
since it (the agent could be holding in profit).

  python scripts/probe_candle_exit.py [--hold-win 48] [--uw-min 0.5] [--lw-max 0.25] [--doji-max 0.10] [--cost 0.01]
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

from datetime import datetime, timezone

WARMUP = 168


def dt(t):
    return datetime.fromtimestamp(int(t), timezone.utc).strftime("%Y-%m-%d %H:%M")
HZS = (1, 3, 6)            # short-term horizons (h bars) — these are SHORT-term bearish signals
FWD_MIN_H = 6              # worst path over the next 6h (the drawdown an exit would dodge)


def classify(env, uw_min, lw_max, doji_max):
    """[bar x token] bool masks for inverted-hammer and doji, from the env's _px/_highf/_lowf."""
    px = env._px
    hi = px / np.where(env._highf > 0, env._highf, np.nan)        # _highf = close/high  -> high
    lo = px * env._lowf                                           # _lowf  = low/close   -> low
    op = np.empty_like(px); op[0] = px[0]; op[1:] = px[:-1]       # bar open ~ prior close
    rng = np.maximum(hi - lo, 1e-12)
    body = np.abs(px - op)
    upper = hi - np.maximum(op, px)
    lower = np.minimum(op, px) - lo
    inv_hammer = (upper / rng >= uw_min) & (lower / rng <= lw_max) & (body / rng <= 0.5)
    doji = body / rng <= doji_max
    return inv_hammer, doji


def held_in_profit(env, hold_win, min_gain=0.0):
    """[bar x token] bool: a recent (<= hold_win bars) ignition fired AND price is up >= min_gain since
    it (min_gain>0 => only EXTENDED positions, the post-pump exhaustion scenario)."""
    px, ig = env._px, env._ignite
    n, m = px.shape
    out = np.zeros_like(px, dtype=bool)
    last_ig = np.full(px.shape, -1, dtype=int)                   # most-recent ignition bar per (bar,token)
    for j in range(m):
        last = -1
        for b in range(n):
            if ig[b, j]:
                last = b
            last_ig[b, j] = last
            if last >= 0 and (b - last) <= hold_win and px[b, j] >= px[last, j] * (1.0 + min_gain):
                out[b, j] = True
    return out, last_ig


_OH_CACHE = {}


def ohlcv_row(tok, ts):
    """Real OHLCV (open,high,low,close) at the bar nearest `ts` for `tok` — what the chart shows."""
    if tok not in _OH_CACHE:
        from train_rl import _load_token_ohlcv
        o = _load_token_ohlcv(tok)
        if o is not None:
            o = o.copy()
            t = o["timestamp"].to_numpy()
            o["ts"] = (t // 1000) if t.max() > 1e12 else t
        _OH_CACHE[tok] = o
    o = _OH_CACHE[tok]
    if o is None:
        return None
    i = (o["ts"] - ts).abs().idxmin()
    rr = o.loc[i]
    return float(rr["open"]), float(rr["high"]), float(rr["low"]), float(rr["close"])


def run_split(name, r, btc, liq, vol, lowf, highf, args):
    from trader.train.event_env import EventRungEnv
    env = EventRungEnv(r, btc, liq, volume=vol, episode_bars=len(r) - WARMUP - 1, k=8, warmup=WARMUP,
                       universe_mode="voltopk", seed=0, low_frac=lowf, high_frac=highf,
                       intrabar_floor=False, wick_reject=0.0)
    env.reset(start=WARMUP)
    px = env._px
    n = env.n_bars
    inv, doji = classify(env, args.uw_min, args.lw_max, args.doji_max)
    hip, last_ig = held_in_profit(env, args.hold_win, args.min_gain)
    sig = inv | doji                                             # the exit trigger
    cols = list(r.columns)

    rows_sig, rows_non, sig_bars = [], [], []
    for j in range(px.shape[1]):
        for b in range(WARMUP, n - FWD_MIN_H):
            if not hip[b, j] or px[b, j] <= 0:
                continue
            fwd = [px[b + h, j] / px[b, j] - 1.0 for h in HZS]
            fmin = px[b + 1: b + 1 + FWD_MIN_H, j].min() / px[b, j] - 1.0
            if sig[b, j]:
                rows_sig.append(fwd + [fmin])
                gain = (px[b, j] / px[last_ig[b, j], j] - 1.0) if last_ig[b, j] >= 0 else float("nan")
                typ = ("inv-hammer" if inv[b, j] else "") + ("+doji" if inv[b, j] and doji[b, j]
                                                             else ("doji" if doji[b, j] else ""))
                sig_bars.append({"tok": cols[j], "ts": int(r.index[b]), "gain": gain, "typ": typ,
                                 "fwd3": fwd[1], "fwd6": fwd[2], "fmin": fmin})
            else:
                rows_non.append(fwd + [fmin])

    def stat(rows):
        if not rows:
            return None
        a = np.clip(np.array(rows), -5, 5)
        return a

    s, nn = stat(rows_sig), stat(rows_non)
    print(f"\n=== {name} ===  held-in-profit bars: signal={len(rows_sig)} non-signal={len(rows_non)}")
    if s is None or nn is None:
        print("  (insufficient samples)"); return
    hdr = "  " + " ".join(f"fwd{h}h" for h in HZS) + "   fwd_min6   win3h"
    print(hdr)
    for nm, a in (("SIGNAL (inv-hammer/doji)", s), ("non-signal (held, no signal)", nn)):
        means = " ".join(f"{a[:, i].mean():+6.2%}" for i in range(len(HZS)))
        print(f"  {nm:30} {means}   {a[:, -1].mean():+6.2%}   {np.mean(a[:, 1] > 0):4.0%}")
    # the lever: exiting on signal vs holding. Edge if signal fwd << non-signal AND signal fwd3 net-neg.
    edge3 = s[:, 1].mean() - nn[:, 1].mean()
    print(f"  >> signal fwd3h {s[:,1].mean():+.2%} vs non-signal {nn[:,1].mean():+.2%}  (gap {edge3:+.2%}); "
          f"exit is +EV if signal fwd3h < -cost ({-args.cost:+.2%}) AND clearly below non-signal")
    if args.list and sig_bars:
        print(f"  --- the {len(sig_bars)} SIGNAL bars (held, up>={args.min_gain:.0%}; OHLC = the real chart candle) ---")
        for sb in sig_bars[: args.list_max]:
            oh = ohlcv_row(sb["tok"], sb["ts"])
            if oh:
                o, h, l, c = oh
                rng = max(h - l, 1e-12)
                shape = (f"o{o:.5g} h{h:.5g} l{l:.5g} c{c:.5g}  upperW {(h-max(o,c))/rng:3.0%} "
                         f"lowerW {(min(o,c)-l)/rng:3.0%} body {abs(c-o)/rng:3.0%}")
            else:
                shape = "OHLC n/a"
            print(f"    {sb['tok']:8} {dt(sb['ts'])}  gain {sb['gain']:+5.0%}  {sb['typ']:12} | {shape}"
                  f" | fwd3h {sb['fwd3']:+5.1%} fwd6h {sb['fwd6']:+5.1%} worst6 {sb['fmin']:+5.1%}")


def run(args):
    from train_rl import build_ohlc_frac_panels, build_volume_panel, load_data, time_split
    returns, btc, anchor, liq = load_data()
    tr, va, te = time_split(returns)
    vol = build_volume_panel(list(returns.columns), returns.index)
    lowf, highf = build_ohlc_frac_panels(list(returns.columns), returns.index)
    print(f"hold_win={args.hold_win} min_gain={args.min_gain} uw_min={args.uw_min} lw_max={args.lw_max} "
          f"doji_max={args.doji_max} | cost {args.cost:.0%} | population = held bars up >= min_gain")
    for name, r in (("TRAIN", tr), ("VAL", va), ("TEST", te)):
        run_split(name, r, btc, liq, vol, lowf, highf, args)
    print("\nREAD: a real exit needs SIGNAL fwd3h/fwd6h clearly NEGATIVE and below non-signal, with a worse "
          "fwd_min6 (it dodges the dump), PERSISTING across TRAIN+VAL+TEST. Else it just sells winners early.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hold-win", type=int, default=48)
    p.add_argument("--min-gain", type=float, default=0.0,
                   help="only count held positions already up >= this since entry (EXTENDED/post-pump)")
    p.add_argument("--uw-min", type=float, default=0.5)
    p.add_argument("--lw-max", type=float, default=0.25)
    p.add_argument("--doji-max", type=float, default=0.10)
    p.add_argument("--cost", type=float, default=0.01)
    p.add_argument("--list", action="store_true", help="print the exact signal bars (token/ts/OHLC/fwd)")
    p.add_argument("--list-max", type=int, default=80)
    run(p.parse_args())


if __name__ == "__main__":
    main()
