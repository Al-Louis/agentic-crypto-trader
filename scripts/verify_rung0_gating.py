"""Verify, from PUBLISHED run bundles, that rung-0's event skeleton actually gated the agent's
trades — the proof the vault claims but never demonstrated on real run data.

For each seed's bundle (per-token trade markers on the eval split), rebuild the EXACT env signals
locally (same data, same params from the bundle's provenance) and check:
  - every BUY lands on a bar where rung-0's ignition condition is True for that token
    (the only entry path in `EventRungEnv._scan_bar` when ungate=False);
  - every SELL is explainable as a rung-0 exit prompt (cushion<0 / trailing-stop region),
    a loser-funded rotation (same-bar buy of another token), or an agent partial-trim of a
    prompted exit.

  python scripts/verify_rung0_gating.py --prefix ppo-event-g2b --seeds 0 1 2 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

HOST = "https://data.alexlouis.dev"
WARMUP = 168


def fetch(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="ppo-event-g2b")
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3])
    args = p.parse_args()

    from train_rl import build_volume_panel, load_data, time_split
    from trader.train.event_env import EventRungEnv

    prov = fetch(f"{HOST}/{args.prefix}-s{args.seeds[0]}/metrics.json")["provenance"]
    split = prov["eval_split"]
    print(f"[cfg] {args.prefix}: split={split} ungate={prov['ungate']} universe={prov['universe_mode']} "
          f"k={prov['k']} commit={prov['git_commit']}")

    returns, btc, anchor, liq = load_data()
    train_r, val_r, test_r = time_split(returns)
    eval_r = test_r if split == "test" else val_r
    vol = build_volume_panel(list(returns.columns), returns.index)
    env = EventRungEnv(
        eval_r, btc, liq, volume=vol, episode_bars=len(eval_r) - WARMUP - 1,
        k=prov["k"], warmup=WARMUP, max_entry_frac=prov["max_entry_frac"],
        stop_k=prov["stop_k"], cooldown=prov["cooldown"],
        reward_mode=prov["reward_mode"], ungate=prov["ungate"],
        action_mode=prov["action_mode"], n_action_levels=prov["n_action_levels"],
        universe_mode=prov["universe_mode"], vol_target=prov["vol_target"],
        cap_floor=prov["cap_floor"], harvest_obs=prov.get("harvest_obs", False), seed=prov["seed"])
    env.reset(start=WARMUP)
    cix = env.col_ix
    ig, px, cush = env._ignite, env._px, env._cush
    stop_k = prov["stop_k"]

    # eval index -> bar; markers carry epoch seconds (apentic _to_secs)
    def to_secs(v):
        v = int(v)
        return v // 1000 if v > 10_000_000_000 else v
    t2bar = {to_secs(t): i for i, t in enumerate(eval_r.index)}
    n_ig = int(ig[WARMUP:, [cix[t] for t in env.universe]].sum())
    print(f"[env] universe={env.universe}")
    print(f"[env] ignition events on this window (agent's universe, post-warmup): {n_ig}")

    for s in args.seeds:
        rid = f"{args.prefix}-s{s}"
        try:
            info = fetch(f"{HOST}/{rid}/run_info.json")
        except Exception as e:  # noqa: BLE001
            print(f"[{rid}] SKIP ({e})")
            continue
        buys, sells = [], []   # (tok, bar, usd)
        for u in info.get("universe", []):
            for m in fetch(f"{HOST}/{rid}/tk_{u['slug']}_trades.json"):
                bar = t2bar.get(to_secs(m.get("time")))
                rec = (u["symbol"], bar, m)
                (buys if str(m.get("side", "")).lower().startswith("b") else sells).append(rec)

        bad_buys, ok_buys = [], 0
        for tok, bar, m in buys:
            if bar is None or tok not in cix:
                bad_buys.append((tok, bar, "unmapped"))
            elif ig[bar, cix[tok]]:
                ok_buys += 1
            else:
                bad_buys.append((tok, bar, "NO ignition"))

        buy_bars = {(b) for _, b, _ in buys}
        ok_sell, rot_sell, bad_sells = 0, 0, []
        entry_bar = {}
        for tok, bar, m in sorted(buys, key=lambda r: (r[1] or 0)):
            entry_bar.setdefault(tok, bar)            # first entry per token (approx for peak calc)
        for tok, bar, m in sells:
            if bar is None or tok not in cix:
                bad_sells.append((tok, bar, "unmapped"))
                continue
            j = cix[tok]
            e = entry_bar.get(tok, WARMUP)
            peak = px[e:bar + 1, j].max() if bar > e else px[bar, j]
            stop_zone = px[bar, j] < peak * (1.0 - stop_k) or cush[bar, j] < 0.0
            if stop_zone:
                ok_sell += 1
            elif bar in buy_bars:
                rot_sell += 1                          # loser-funded rotation on an entry bar
            else:
                # overrides re-anchor the trailing peak lower, so a later stop can fire ABOVE the
                # naive entry-peak threshold only if peaks were re-anchored by trims — flag for review
                bad_sells.append((tok, bar, f"px/peak={px[bar, j] / peak:.3f} cush={cush[bar, j]:+.3f}"))

        print(f"[{rid}] buys: {ok_buys}/{len(buys)} on rung-0 ignition bars"
              + (f"  VIOLATIONS: {bad_buys}" if bad_buys else "  (ALL gated OK)"))
        print(f"[{rid}] sells: {ok_sell} stop/EMA-zone, {rot_sell} rotation, "
              f"{len(bad_sells)} needing review" + (f": {bad_sells[:6]}" if bad_sells else ""))


if __name__ == "__main__":
    main()
