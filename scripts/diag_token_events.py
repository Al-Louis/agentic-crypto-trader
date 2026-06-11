"""Why did/didn't the agent trade <token> at <times>? The forensic answer, from the env's own
signal math — not the chart's visual impression.

For one published run and one token, prints:
  1. the eval window + where the untradeable 168-bar warmup ends (markers can't exist before it);
  2. the published buy/sell markers (what the user sees in the frontend);
  3. EVERY bar where the env's ignition fired for the token (the agent's entry prompts), with the
     four components (surge>=2.5x, rising, cushion>0, ema_up) — plus which bars the agent held /
     was in cooldown, so a fired-but-no-buy bar = the AGENT chose to skip (or couldn't fund);
  4. the component breakdown at each user-supplied timestamp (what failed and by how much);
  5. what the rung-0 RULE (the env's shadow mirror, same universe) did with the token.

  python scripts/diag_token_events.py --run-id ppo-event-g2b-s3 --token SIREN \
      --times 2026-03-22T06:00 2026-03-25T00:00 2026-03-28T00:00 2026-04-04T00:00 2026-04-17T08:00
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

HOST = "https://data.alexlouis.dev"
WARMUP = 168
VOL_MULT, VOL_SPK, VOL_BASE, VOL_FAST, EMA_SPAN = 2.5, 24, 168, 4, 72


def fetch(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def to_secs(v):
    v = int(v)
    return v // 1000 if v > 10_000_000_000 else v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default="ppo-event-g2b-s3")
    p.add_argument("--token", default="SIREN")
    p.add_argument("--times", nargs="*", default=[])
    args = p.parse_args()

    from train_rl import build_volume_panel, load_data, time_split
    from trader.train.event_env import EventRungEnv

    prov = fetch(f"{HOST}/{args.run_id}/metrics.json")["provenance"]
    returns, btc, anchor, liq = load_data()
    train_r, val_r, test_r = time_split(returns)
    eval_r = test_r if prov["eval_split"] == "test" else val_r
    if prov.get("eval_prepad"):                       # warmup served from the prior split's tail
        import pandas as _pd
        prev = train_r if prov["eval_split"] == "val" else val_r
        eval_r = _pd.concat([prev.tail(WARMUP), eval_r])
    vol = build_volume_panel(list(returns.columns), returns.index)
    env = EventRungEnv(
        eval_r, btc, liq, volume=vol, episode_bars=len(eval_r) - WARMUP - 1,
        k=prov["k"], warmup=WARMUP, max_entry_frac=prov["max_entry_frac"], stop_k=prov["stop_k"],
        cooldown=prov["cooldown"], reward_mode=prov["reward_mode"], ungate=prov["ungate"],
        action_mode=prov["action_mode"], n_action_levels=prov["n_action_levels"],
        universe_mode=prov["universe_mode"], vol_target=prov["vol_target"],
        cap_floor=prov["cap_floor"], harvest_obs=prov.get("harvest_obs", False), seed=prov["seed"])
    env.reset(start=WARMUP)
    tok, j = args.token, env.col_ix[args.token]
    secs = np.array([to_secs(t) for t in eval_r.index])
    dt = pd.to_datetime(secs, unit="s", utc=True)

    # recompute the ignition components exactly as EventRungEnv.__init__ does
    v = vol.reindex(eval_r.index).fillna(0.0)
    px = (1.0 + eval_r.fillna(0.0)).cumprod()
    ema = px.ewm(span=EMA_SPAN, adjust=False).mean()
    vrec = v.rolling(VOL_FAST, min_periods=1).mean()
    vbase = v.shift(VOL_FAST).rolling(max(VOL_BASE - VOL_FAST, 1), min_periods=1).mean()
    surge = (vrec / vbase.replace(0.0, np.nan)).fillna(0.0)
    cush = px / ema - 1.0
    rising = px / px.shift(VOL_SPK) - 1.0
    ema_up = ema >= ema.shift(VOL_FAST)
    S, C, R, E = surge[tok].to_numpy(), cush[tok].to_numpy(), rising[tok].to_numpy(), ema_up[tok].to_numpy()
    ig = env._ignite[:, j]
    print(f"[window] {dt[0]:%Y-%m-%d %H:%M} .. {dt[-1]:%Y-%m-%d %H:%M} UTC  "
          f"| warmup ends (first tradeable bar): {dt[WARMUP]:%Y-%m-%d %H:%M}")
    print(f"[cfg] in-universe: {tok in env.universe}  cap={env._tok_cap.get(tok, 0):.3f}  "
          f"ungate={prov['ungate']}  cooldown={prov['cooldown']}h")

    # what the user sees: the published markers
    try:
        marks = fetch(f"{HOST}/{args.run_id}/tk_{tok}_trades.json")
    except Exception:  # noqa: BLE001
        marks = []
    buys = [m for m in marks if m["side"] == "buy"]
    sells = [m for m in marks if m["side"] == "sell"]
    print(f"\n[published markers] {len(buys)} buys, {len(sells)} sells")
    for m in (buys + sells[:3] + sells[-2:] if len(sells) > 5 else buys + sells):
        print(f"  {m['side']:4} {pd.Timestamp(to_secs(m['time']), unit='s', tz='UTC'):%Y-%m-%d %H:%M} "
              f"${m['usd']:8.2f} @ {m['price']:.4f}")
    if len(sells) > 5:
        print(f"  ... ({len(sells)} sells total — geometric partial-trim tail)")

    buy_bars = {int(np.searchsorted(secs, to_secs(m["time"]))) for m in buys}
    sell_bars = {int(np.searchsorted(secs, to_secs(m["time"]))) for m in sells}

    # every ignition prompt for this token in the tradeable window
    print(f"\n[ignition bars for {tok}] (the agent's entry prompts; BUY = it bought, "
          f"skip = prompted but no trade, cool = inside 48h cooldown of its own last exit)")
    last_exit = -10**9
    n_fired = 0
    for b in range(WARMUP, len(eval_r)):
        if b in sell_bars:
            last_exit = b
        if not ig[b]:
            continue
        n_fired += 1
        tag = ("BUY " if b in buy_bars else
               "cool" if (b - last_exit) < prov["cooldown"] else "skip")
        print(f"  {dt[b]:%Y-%m-%d %H:%M}  surge {S[b]:5.1f}x  rising {R[b]:+6.1%}  "
              f"cush {C[b]:+6.1%}  emaUp {str(bool(E[b])):5}  -> {tag}")
    if not n_fired:
        print("  (none — the ignition never fired for this token in the tradeable window)")

    # the user's timestamps: component forensics
    if args.times:
        print(f"\n[your timestamps] ignition needs: surge>={VOL_MULT}x AND rising>0 AND cush>0 AND emaUp")
        for ts in args.times:
            t = int(pd.Timestamp(ts, tz="UTC").timestamp())
            b = int(np.searchsorted(secs, t))
            if b >= len(secs):
                print(f"  {ts}: outside the window")
                continue
            fails = []
            if S[b] < VOL_MULT:
                fails.append(f"surge {S[b]:.2f}x < {VOL_MULT}")
            if R[b] <= 0:
                fails.append(f"rising {R[b]:+.1%} <= 0")
            if C[b] <= 0:
                fails.append(f"cush {C[b]:+.1%} <= 0 (below EMA)")
            if not E[b]:
                fails.append("EMA falling")
            warm = " [IN WARMUP — untradeable]" if b < WARMUP else ""
            print(f"  {dt[b]:%Y-%m-%d %H:%M}  surge {S[b]:5.1f}x rising {R[b]:+6.1%} cush {C[b]:+6.1%} "
                  f"emaUp {str(bool(E[b])):5} -> {'FIRES' if ig[b] else 'no: ' + '; '.join(fails)}{warm}")

    # what the rung-0 RULE did with this token (the env's shadow mirror, same universe)
    eq, w = env._rule_equity_curve(WARMUP, env.end)
    ui = env.universe.index(tok) if tok in env.universe else None
    if ui is not None:
        held = w[:, ui] > 0
        print(f"\n[rung-0 RULE on {tok}] (shadow mirror, same {len(env.universe)}-token universe)")
        prev = False
        for kbar, h in enumerate(held):
            if h != prev:
                print(f"  {'ENTER' if h else 'EXIT '} {dt[WARMUP + kbar]:%Y-%m-%d %H:%M}")
                prev = h
        if not held.any():
            print("  (the rule never held it)")


if __name__ == "__main__":
    main()
