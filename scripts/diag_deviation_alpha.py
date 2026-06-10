"""Deviation-alpha diagnostic — is the event-rung RL reward-bound or capacity-bound?

For every entry the agent EXECUTED (across the published seed bundles), measure how much it
over/under-sized vs the rung-0 rule's fixed 0.20, and correlate that deviation with the token's
forward-24h return. If the agent's bigger bets don't land on bigger moves (corr ~0), it is
deviating WITHOUT skill -> the REWARD isn't teaching discrimination (reward-bound, fix the reward).
A clearly positive correlation would mean it discriminates but is capped (capacity-bound, justify
an LSTM). Reads only the published bundles (no model replay needed).

Usage:  python scripts/diag_deviation_alpha.py --prefix ppo-event-rel-test --seeds "0 1 2 3"
"""
from __future__ import annotations

import argparse
import json
import urllib.request

import numpy as np

RULE_ENTRY_FRAC = 0.20
H = 24 * 3600          # forward horizon: 24 bars (1 day)


def g(host, rid, p):
    with urllib.request.urlopen(f"{host}/{rid}/{p}", timeout=20) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="https://data.alexlouis.dev")
    ap.add_argument("--prefix", default="ppo-event-rel-test")
    ap.add_argument("--seeds", default="0 1 2 3")
    args = ap.parse_args()

    devs, frets = [], []
    for s in args.seeds.split():
        rid = f"{args.prefix}-s{s}"
        try:
            info, eqc = g(args.host, rid, "run_info.json"), g(args.host, rid, "equity_curve.json")
        except Exception as e:  # noqa: BLE001
            print(f"  skip {rid}: {e}")
            continue
        uni = [u["slug"] for u in info["universe"]]
        eqt = np.array([e["time"] for e in eqc], float)
        eqv = np.array([e["value"] for e in eqc], float)
        for slug in uni:
            try:
                tr, cd = g(args.host, rid, f"tk_{slug}_trades.json"), g(args.host, rid, f"tk_{slug}_candles.json")
            except Exception:  # noqa: BLE001
                continue
            ct = np.array([c["time"] for c in cd], float)
            cc = np.array([c["close"] for c in cd], float)
            if len(ct) < 2:
                continue
            for m in tr:
                if m.get("side") != "buy":
                    continue
                t, usd = float(m["time"]), float(m["usd"])
                eq = float(np.interp(t, eqt, eqv)) if len(eqt) else 0.0
                if eq <= 0:
                    continue
                i0, i1 = int(np.searchsorted(ct, t)), int(np.searchsorted(ct, t + H))
                i0, i1 = min(i0, len(cc) - 1), min(i1, len(cc) - 1)
                if i1 <= i0 or cc[i0] <= 0:
                    continue
                devs.append(usd / eq - RULE_ENTRY_FRAC)
                frets.append(cc[i1] / cc[i0] - 1.0)

    devs, frets = np.array(devs), np.array(frets)
    print(f"\nexecuted entries analyzed: {len(devs)}")
    if len(devs) > 3:
        r = float(np.corrcoef(devs, frets)[0, 1])
        over, under = frets[devs > 0], frets[devs <= 0]
        print(f"corr(entry over/under-size vs rule {RULE_ENTRY_FRAC}, fwd-24h return) = {r:+.3f}")
        om = f"{over.mean() * 100:+.2f}% (n={len(over)})" if len(over) else "- (n=0)"
        um = f"{under.mean() * 100:+.2f}% (n={len(under)})" if len(under) else "- (n=0)"
        print(f"mean fwd-24h return: OVERSIZED {om}  |  UNDERSIZED {um}")
        print(f"entry-size range: {devs.min() + RULE_ENTRY_FRAC:.2f}..{devs.max() + RULE_ENTRY_FRAC:.2f}")
        print("=> REWARD-BOUND if corr ~0 (deviates without skill); CAPACITY-BOUND if clearly positive")


if __name__ == "__main__":
    main()
