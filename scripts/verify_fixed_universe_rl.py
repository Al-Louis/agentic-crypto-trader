"""VERIFY the user's 2026-06-19 fixed-universe proposal with the REAL ef2 RL policy (not just the
rung-0 rule). DESKTOP-ONLY (torch). Grades ppo-event-rdLe4-ef2-0913df8-s1 the deployment-honest way
(cold weekly sessions, fresh $10k, OOS val+test) under three policy-universe configs:

  A1 voltopk vm2.0  -- the CORRECT training config (k=10, vol_mult=2.0): the true honest baseline.
  A2 voltopk vm2.5  -- what the PUBLISHED simulate_weekly actually ran (provenance never records
                       vol_mult -> env_kwargs_from_provenance defaults to 2.5): quantifies that bug.
  C  fixed13 vm2.0  -- the proposal: a FIXED 13-token universe (BTC-majors dropped), no weekly re-pick.

Per week it records each config's cold-week return / maxDD AND whether the policy actually held/traded
FF -- the token whose Apr-9/10 roundtrip the causal re-pick misses. Caveat: A1/A2 are in-distribution
(trained on voltopk k=10); C is OFF-distribution for the policy (universe size 10->13 shifts the
len(pos)/k and breadth obs slots), so a WEAK C is confounded with off-distribution and a retrain is the
definitive test -- but a STRONG C (catches FF, beats A1) would be a clean signal to retrain on fixed.

  python3 scripts/verify_fixed_universe_rl.py            # full table -> runs-rl/verify_fixed.out, [SUM] to stdout
"""
from __future__ import annotations
import os, sys, json, pickle
sys.path.insert(0, "scripts")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from datetime import datetime, timezone

from trader import config
from trader.train import weekly_eval as we
from train_event import evaluate_event_policy, WARMUP
from simulate import env_kwargs_from_provenance, make_predict

RUN_ID = "ppo-event-rdLe4-ef2-0913df8-s1"
KEEP13 = ['ASTER', 'B', 'BANANAS31', 'COAI', 'FF', 'HUMA', 'Q', 'SIREN', 'SKYAI', 'TAC', 'TAG', 'UB', 'ZEC']
BASE_K, VT, CF = 10, 0.005, 0.02          # rung-0 reference bar: constant causal top-10 across all configs
OUT = os.path.join("runs-rl", "verify_fixed.out")
def d(ts): return datetime.fromtimestamp(int(ts), timezone.utc).strftime("%Y-%m-%d")


def main():
    config.load_dotenv()
    base = os.path.join("runs-rl", RUN_ID)
    prov = json.load(open(os.path.join(base, RUN_ID, "metrics.json"), encoding="utf-8"))
    prov = prov.get("provenance", prov)
    recurrent = bool(prov.get("recurrent"))

    from train_rl import build_ohlc_frac_panels, build_volume_panel, load_data, time_split
    returns, btc, _anchor, liq = load_data()
    vol = build_volume_panel(list(returns.columns), returns.index)
    tr, va, te = time_split(returns)
    val_start, test_start = int(va.index[0]), int(te.index[0])
    base_kwargs = env_kwargs_from_provenance(prov, returns, build_ohlc_frac_panels)

    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO
    model = (RecurrentPPO if recurrent else PPO).load(os.path.join(base, "policy.zip"), device="cpu")
    vn = pickle.load(open(os.path.join(base, "vecnormalize.pkl"), "rb"))
    mp = lambda: make_predict(model, vn, recurrent)
    cap = float(base_kwargs.get("capital", we.START_CAPITAL))

    cfgs = [("A1_voltopk_vm2.0", dict(universe_mode="voltopk", vol_mult=2.0)),
            ("A2_voltopk_vm2.5", dict(universe_mode="voltopk", vol_mult=2.5)),
            ("C_fixed13_vm2.0",  dict(universe_mode="fixed", fixed_universe=list(KEEP13), vol_mult=2.0))]

    log = open(OUT, "w", encoding="utf-8")
    def pr(*a):
        s = " ".join(str(x) for x in a); log.write(s + "\n"); print(s)

    pr(f"# verify fixed-universe RL | {RUN_ID} | recurrent={recurrent} | cap={cap:.0f}")
    pr(f"# VAL {d(val_start)} TEST {d(test_start)} | KEEP13={sorted(KEEP13)}")
    agg = {name: {"ret": [], "dd": [], "ff_uni": 0, "ff_trade": 0, "ff_pnl": 0.0} for name, _ in cfgs}
    header = f"{'week':11}{'split':5}{'regime':7}" + "".join(f"{n.split('_')[0]:>9}" for n, _ in cfgs) + "  FFuni(A1/C) FFpnlC"
    pr(header)

    for ws, win in we.cold_week_windows(returns):
        split = we.split_label(ws, val_start, test_start)
        if split == "train":
            continue
        base_g = we.grade_week_baselines(ws, win, liq, vol, k=BASE_K, vol_target=VT, cap_floor=CF, vol_mult=2.0)
        line = f"{d(ws):11}{split:5}{base_g.regime:7}"
        ff_uni = {}
        for name, ov in cfgs:
            kw = dict(base_kwargs); kw.update(ov)
            eq, recs, universe, _f, _r, tok_pnl = evaluate_event_policy(mp(), win, btc, liq, vol, kw)
            ret = float(eq.iloc[-1] / cap - 1.0)
            dd = abs(float((eq / eq.cummax() - 1.0).min()))
            ff_in = "FF" in universe
            ff_tr = any("FF" in (rec.get("trades_usd") or {}) for rec in recs)
            a = agg[name]; a["ret"].append(ret); a["dd"].append(dd)
            a["ff_uni"] += int(ff_in); a["ff_trade"] += int(ff_tr); a["ff_pnl"] += float(tok_pnl.get("FF", 0.0))
            ff_uni[name] = ff_in
            line += f"{ret * 100:>+8.1f}%"
        ffc = agg["C_fixed13_vm2.0"]["ret"]  # last appended is this week's C
        line += f"   {('Y' if ff_uni['A1_voltopk_vm2.0'] else '-')}/{('Y' if ff_uni['C_fixed13_vm2.0'] else '-')}"
        pr(line)

    pr("\n[SUM] config            n   mean  median winrate worstDD DQwk  FFuniWk FFtradeWk  FFpnl$")
    for name, _ in cfgs:
        a = agg[name]; r = np.array(a["ret"]); dd = np.array(a["dd"])
        pr(f"[SUM] {name:18}{len(r):3}{r.mean() * 100:+6.1f}%{np.median(r) * 100:+7.1f}%"
           f"{(r > 0).mean() * 100:6.0f}% {dd.max() * 100:6.1f}% {int((dd > 0.30).sum()):4}"
           f"   {a['ff_uni']:4}    {a['ff_trade']:4}   {a['ff_pnl']:+8.0f}")
    log.close()
    print(f"\n[done] full table -> {OUT}")


if __name__ == "__main__":
    main()
