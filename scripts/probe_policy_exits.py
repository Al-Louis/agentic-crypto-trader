"""P-POLICY-EXITS — re-do the exit/giveback analysis on the TRAINED POLICY (not the bare rule). The rule
only sells on weakness (trailing/EMA-cross) so it gives back pumps; the policy has tp_rungs + learned
discretion (override the EMA-break / hold / take profit). Question: how does the POLICY actually capture
its gains, and does ITS EMA-break give back winners? DESKTOP (torch).

  python scripts/probe_policy_exits.py --run-id ppo-event-rdLe4-ef2-0913df8-s0 [--vol-mult 2.0]
"""
from __future__ import annotations
import argparse, json, os, pickle, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from trader import config
from trader.train import weekly_eval as we
from train_event import evaluate_event_policy
from simulate import env_kwargs_from_provenance, make_predict

H = 48


def main():
    p = argparse.ArgumentParser(); p.add_argument("--run-id", required=True)
    p.add_argument("--vol-mult", type=float, default=None); args = p.parse_args(); config.load_dotenv()
    base = os.path.join("runs-rl", args.run_id)
    prov = json.load(open(os.path.join(base, args.run_id, "metrics.json"))); prov = prov.get("provenance", prov)
    recurrent = bool(prov.get("recurrent"))
    from train_rl import build_ohlc_frac_panels, build_volume_panel, load_data, time_split
    returns, btc, _a, liq = load_data()
    vol = build_volume_panel(list(returns.columns), returns.index)
    tr, va, te = time_split(returns); val_start, test_start = int(va.index[0]), int(te.index[0])
    env_kwargs = env_kwargs_from_provenance(prov, returns, build_ohlc_frac_panels)
    if args.vol_mult is not None:
        env_kwargs["vol_mult"] = args.vol_mult
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO
    model = (RecurrentPPO if recurrent else PPO).load(os.path.join(base, "policy.zip"), device="cpu")
    vn = pickle.load(open(os.path.join(base, "vecnormalize.pkl"), "rb"))

    reason_ct = {}                      # reason -> count of sells
    emabrk_fwd = []                     # fwd48 terminal return after each EMA_BREAK sell (giveback check)
    capture = {"held_to_end": [], "exited": []}   # token-week PnL by how the winner ended
    for ws, win in we.cold_week_windows(returns):
        sp = we.split_label(ws, val_start, test_start)
        if sp == "train":
            continue
        eq, records, universe, fees, raw, tok_pnl = evaluate_event_policy(
            make_predict(model, vn, recurrent), win, btc, liq, vol, env_kwargs)
        # per-token: did it ever SELL (and on what), or was it held to the end?
        sold = {}                       # token -> list of reasons
        pix = {int(t): i for i, t in enumerate(win.index)}
        from train_event import WARMUP
        # rebuild px for the fwd-return-after-EMA-break (use a torch-free env clone for arrays)
        from trader.train.event_env import EventRungEnv
        kwc = {k: v for k, v in env_kwargs.items() if k != "episode_bars"}
        clone = EventRungEnv(win, btc, liq, volume=vol, episode_bars=len(win) - WARMUP - 1, **kwc)
        clone.reset(start=WARMUP); px = clone._px
        for rec in records:
            for f in rec.get("fills", []):
                if f["usd"] < 0:        # a SELL
                    r = f["reason"]; reason_ct[r] = reason_ct.get(r, 0) + 1
                    sold.setdefault(f["token"], []).append(r)
                    if r == "EMA_BREAK":
                        b = pix.get(int(f["time"])); j = clone.col_ix.get(f["token"])
                        if b is not None and j is not None and b + H < len(win):
                            emabrk_fwd.append(px[b + H, j] / px[b, j] - 1.0)
        for tk, pnl in tok_pnl.items():
            if abs(pnl) < 1.0:
                continue
            (capture["exited"] if tk in sold else capture["held_to_end"]).append(pnl)

    print(f"{args.run_id} | OOS cold weeks | vol_mult={env_kwargs.get('vol_mult')}")
    tot = sum(reason_ct.values()) or 1
    print("\nSELL reason distribution (in-episode exits):")
    for r, c in sorted(reason_ct.items(), key=lambda x: -x[1]):
        print(f"  {r:14} {c:4}  ({c/tot*100:.0f}%)")
    if emabrk_fwd:
        a = np.array(emabrk_fwd)
        print(f"\nPOLICY's EMA_BREAK sells: n={len(a)}  fwd{H}h terminal after the sell: "
              f"mean {a.mean()*100:+.1f}%  median {np.median(a)*100:+.1f}%  %that-recovered(>0) {(a>0).mean()*100:.0f}%")
        print("  (positive => the policy's OWN EMA-break gives back; negative => it cuts real weakness)")
    for k in ("held_to_end", "exited"):
        v = np.array(capture[k]) if capture[k] else np.array([0.0])
        print(f"\ncaptured-by={k:11} token-weeks n={len(capture[k]):3}  total PnL ${v.sum():+.0f}  "
              f"mean ${v.mean():+.0f}  best ${v.max():+.0f}")
    print("\nREAD: if the big gains are 'held_to_end' and the policy's EMA_BREAK fwd is NEGATIVE, the agent "
          "already overrides the EMA-break on winners + cuts real weakness -> the exit is NOT the leak, and "
          "my rule-based giveback probes were the wrong substrate. If EMA_BREAK fwd is POSITIVE, the policy "
          "DOES give back and a sell-into-strength fix is live.")


if __name__ == "__main__":
    main()
