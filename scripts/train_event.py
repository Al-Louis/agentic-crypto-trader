"""Train a PPO policy on the event-driven rung-1 env and publish its eval bundle. DESKTOP-ONLY
for training (needs torch); the eval/publish core is torch-free and laptop-testable.

The agent learns rung-0's DISCRETION (entry sizing, exit override) on top of rung-0's event timing
(see trader.train.event_env). Trains on random WEEKLY windows of the train split; evaluates one
long episode on the held-out split; self-publishes the Apentic bundle with the real intra-day
markers, and records the rung-0 RULE's return on the same window as the baseline (does learned
discretion beat the hand-coded version?).

  python scripts/train_event.py --timesteps 1000000 --n-envs 8 --seed 0 --run-id ppo-event-s0
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from remote_train.progress import write_progress  # noqa: E402
from trader import config  # noqa: E402
from trader.report import apentic as ap  # noqa: E402
from trader.sim.broker import DEFAULT_GAS_USD, DEFAULT_LP_FEE_BPS, amm_cost_usd  # noqa: E402
from trader.sim.metrics import PerformanceMetrics  # noqa: E402
from trader.train.curriculum import (horizon_at, max_horizon, parse_horizon_schedule,  # noqa: E402
                                     parse_universe_schedule, universe_at)

HOURS_PER_YEAR = 24 * 365
WARMUP = 168


def evaluate_event_policy(predict_fn, eval_r, btc, liq, vol, env_kwargs):
    """Run one long episode over `eval_r` with `predict_fn(obs)->action`; collect per-event markers
    and the per-bar equity trace. Torch-free — works with a PPO policy or a heuristic (for tests)."""
    from trader.train.event_env import EventRungEnv
    kw = {k: v for k, v in env_kwargs.items() if k != "episode_bars"}
    env = EventRungEnv(eval_r, btc, liq, volume=vol, episode_bars=len(eval_r) - WARMUP - 1,
                       record_trace=True, **kw)
    obs = env.reset(start=WARMUP)
    records, fees, raw = [], 0.0, []
    if env._trades:                                    # basket_default buys the WHOLE basket in reset();
        ofills = [{"token": t, "usd": u, "fee": c, "time": ft, "px": px, "reason": rsn, "obs": ob}
                  for t, u, c, ft, px, rsn, ob in env._trades]                  # step() clears _trades each
        #                                                                       # call, so the opening buy is
        records.append({"time": int(eval_r.index[WARMUP]), "fills": ofills,    # lost unless emitted HERE —
                        "weights": {t: env._pos_value(t) / max(env._equity(), 1.0) for t in env.pos},
                        "trades_usd": {f["token"]: f["usd"] for f in ofills},   # the missing buy that made
                        "trade_fees": {f["token"]: f["fee"] for f in ofills}})  # the chart show sells-first
        fees += sum(f["fee"] for f in ofills)
    done = False
    while not done:
        a = predict_fn(obs)
        raw.append(float(np.asarray(a).reshape(-1)[0]))
        obs, _, done, info = env.step(a)
        if info.get("trades"):
            fills = [{"token": t, "usd": u, "fee": c, "time": ft, "px": px, "reason": rsn, "obs": ob}
                     for t, u, c, ft, px, rsn, ob in info["trades"]]   # per-fill TRUE time + _px exec px
            #                                                          # + trigger reason + state-acted-on
            records.append({"time": info["trade_time"], "weights": info["weights"], "fills": fills,
                            "trades_usd": {f["token"]: f["usd"] for f in fills},   # legacy dicts (other consumers)
                            "trade_fees": {f["token"]: f["fee"] for f in fills}})
            fees += sum(f["fee"] for f in fills)
    eq = pd.Series([e for _, e in env._eq_trace], index=[t for t, _ in env._eq_trace])
    universe = sorted({t for rec in records for t in rec["trades_usd"]} | set(env.universe))
    # token_pnls() = the EXACT per-token realized+open PnL ledger; the weekly export snaps each asset's
    # positions to it so the dashboard's derived PnL equals the sim's by construction (no inference).
    return eq, records, universe, fees, raw, env.token_pnls()


def rung0_baseline(eval_r, liq, vol):
    """The rung-0 RULE (hand-coded discretion, canonical vol-top-8) on the same window — the bar the RL
    must clear. Returns (return, maxDD>=0). A rung-0 that is ITSELF DQ'd (maxDD >= the 30% gate) is not
    a valid live strategy, so the honest gate stops forcing the agent to match its (unsurvivable) return."""
    from trader.strategy.rung0 import build_rung0, run_rung0
    # CAUSAL universe (trailing-warmup std at WARMUP-1, matching EventRungEnv._pick_universe). NOT
    # candidate.select_vol_tokens, which ranks by FULL-window std — lookahead that peeks at the late
    # pumpers and inflated the rung-0 bar (e.g. test +29% lookahead vs +18% causal).
    std = eval_r.rolling(WARMUP, min_periods=8).std().to_numpy()
    order = np.argsort(np.nan_to_num(std[WARMUP - 1], nan=-1.0))[::-1][:8]
    uni = [eval_r.columns[j] for j in order]
    eq, _, _ = run_rung0(eval_r, build_rung0(eval_r, tokens=uni, volume=vol), liq, warmup=WARMUP)
    ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    maxdd = abs(float((eq / eq.cummax() - 1.0).min()))
    return ret, maxdd


def eval_universe_and_caps(eval_r, btc, liq, vol, env_kwargs):
    """The EXACT universe + per-token weight caps the agent trades on this window — instantiate the
    env and read `env.universe` / `env._tok_cap`, so the Buy&Hold benchmark and the regime are over
    the SAME basket the policy uses (mirrors universe_mode + vol_target; no duplicated pick logic)."""
    from trader.train.event_env import EventRungEnv
    kw = {k: v for k, v in env_kwargs.items() if k != "episode_bars"}
    env = EventRungEnv(eval_r, btc, liq, volume=vol, episode_bars=len(eval_r) - WARMUP - 1, **kw)
    env.reset(start=WARMUP)
    return list(env.universe), dict(env._tok_cap)


def buy_and_hold_return(eval_r, liq, universe, caps, warmup=WARMUP, capital=10_000.0):
    """BUY & HOLD of the AGENT'S OWN universe, weighted by its risk-parity caps (weight proportional
    to cap = capped inverse-vol), fully invested at warmup, entry AMM cost once via the same broker.
    The honest 'passive version of the same strategy' bar — NOT a different basket ([[AI Training]])."""
    px = (1.0 + eval_r.fillna(0.0)).cumprod().to_numpy()
    cix = {t: i for i, t in enumerate(eval_r.columns)}
    wsum = sum(caps[t] for t in universe) or 1.0
    eq_end = 0.0
    for t in universe:
        alloc = (caps[t] / wsum) * capital
        j = cix[t]
        p0 = px[warmup, j]
        if p0 <= 0:
            eq_end += alloc
            continue
        invested = alloc - amm_cost_usd(alloc, liq.get(t, 0.0), DEFAULT_LP_FEE_BPS, DEFAULT_GAS_USD)
        eq_end += invested * px[-1, j] / p0
    return eq_end / capital - 1.0


def random_baseline_return(eval_r, btc, liq, vol, env_kwargs, n=3, seed=0):
    """RANDOM discretion through the SAME event env — the floor a learned policy must clear. Mean of
    `n` seeded random-action passes (the env's event timing is fixed; only the discretion is random)."""
    discrete = env_kwargs.get("action_mode") == "discrete"
    n_lvl = env_kwargs.get("n_action_levels", 4)
    rets = []
    for s in range(n):
        rng = np.random.default_rng(seed + 100 + s)
        sample = ((lambda o: np.array([rng.integers(0, n_lvl)])) if discrete
                  else (lambda o: np.array([rng.uniform(-1.0, 1.0)])))
        eq, *_ = evaluate_event_policy(sample, eval_r, btc, liq, vol, env_kwargs)
        rets.append(float(eq.iloc[-1] / eq.iloc[0] - 1.0))
    return float(np.mean(rets))


def honest_gate(pol, rung0, buyhold, random_, pol_maxdd=0.0, rung0_maxdd=0.0, dq_gate=0.30):
    """The structural gate ([[AI Training]]; DIRECTION RESET 2026-06-15, 6eda1d5): a model earns a
    version only if it (1) survives the DQ gate AND (2) beats a SURVIVING rung-0 RULE. Returns
    (passed, binding).

    Competition reality (PnL scored under a hard ~30% max-drawdown DQ): (1) the POLICY itself must keep
    maxDD < dq_gate — a higher return that breaches the gate is worthless (DQ'd). (2) The rung-0 RULE is
    the only binding return bar, and only when it itself SURVIVES the DQ gate — a baseline that is itself
    DQ'd is not a valid live competitor, so beating its (unsurvivable) return is NOT required (this is what
    stops a concentrated, DQ-prone rung-0 from setting an unbeatable bar that only risk *avoidance* could
    clear). `buyhold` and `random_` are still ACCEPTED (callers pass them; they are computed and REPORTED
    everywhere — numbers/bars/dashboards) but are NEVER binding gate checks: requiring "beat Buy&Hold"
    rewards holding-everything (the buy-everything basket overlay the user rejected). The selective
    event-driven agent sits in cash between ignitions and structurally cannot out-return B&H in a bull, by
    design. The rung-0 RULE is the real bar."""
    if pol_maxdd > dq_gate:
        return False, f"DQ: policy maxDD {pol_maxdd:.0%} > {dq_gate:.0%}"
    beats = {}
    if rung0_maxdd <= dq_gate:                              # only a SURVIVING rung-0 is a bar to beat
        beats["rung-0"] = pol > rung0
    passed = all(beats.values())
    binding = None if passed else next(n for n in beats if not beats[n])
    return passed, binding


def eval_regime(eval_r, btc, universe, warmup=WARMUP):
    """The eval window's regime over the AGENT'S universe so 'beats the rule' can never hide 'lost to
    the market': BTC return and the universe equal-weight return over warmup->end, bull/bear/flat."""
    px = (1.0 + eval_r.fillna(0.0)).cumprod()
    uni_ew = float(np.mean([px[t].iloc[-1] / px[t].iloc[warmup] - 1.0 for t in universe]))
    b = btc.reindex(eval_r.index).ffill().bfill().to_numpy()
    btc_ret = float(b[-1] / b[warmup] - 1.0) if b[warmup] else 0.0
    label = "bull" if uni_ew > 0.10 else "bear" if uni_ew < -0.10 else "flat"
    return {"btc_return": btc_ret, "universe_ew_return": uni_ew, "label": label}


def evaluate_and_gate(name, eval_r, btc, liq, vol, env_kwargs, predict_fn, seed):
    """Run the policy on one split and grade it through the full honest gate (universe-matched
    Buy&Hold, Random-through-env, rung-0, regime). Returns everything needed to publish + report."""
    eq, records, universe, fees, raw, token_pnl = evaluate_event_policy(predict_fn, eval_r, btc, liq, vol, env_kwargs)
    report = PerformanceMetrics.compute_all(eq.to_numpy(), steps_per_year=HOURS_PER_YEAR)
    pol, pol_dd = report.total_return_pct, report.max_drawdown_pct
    uni, caps = eval_universe_and_caps(eval_r, btc, liq, vol, env_kwargs)
    base, base_dd = rung0_baseline(eval_r, liq, vol)
    bh = buy_and_hold_return(eval_r, liq, uni, caps)
    rnd = random_baseline_return(eval_r, btc, liq, vol, env_kwargs, seed=seed)
    regime = eval_regime(eval_r, btc, uni)
    gate_pass, binding = honest_gate(pol, base, bh, rnd, pol_maxdd=pol_dd, rung0_maxdd=base_dd)
    return {"name": name, "eq": eq, "records": records, "universe": universe, "fees": fees, "raw": raw,
            "report": report, "pol": pol, "base": base, "base_dd": base_dd, "bh": bh, "rnd": rnd,
            "regime": regime, "gate_pass": gate_pass, "binding": binding, "token_pnl": token_pnl}


def evaluate_weekly_gate(returns, btc, liq, vol, env_kwargs, make_predict, val_start, test_start,
                         seed, k, vol_target, cap_floor, vol_mult=2.5):
    """Grade the policy the way it DEPLOYS: independent COLD weekly sessions (fresh $10k, no cross-week
    holds) over the OOS weeks (val+test), vs rung-0 + Buy&Hold graded the same way, then apply the
    random-week distribution gate ([[AI Training]] §the-fork). A fresh predictor per week = a cold LSTM
    start each session (in-distribution: every training episode also starts cold). Returns (verdict, rows)."""
    from trader.train import weekly_eval as we
    pol_rets, pol_dds, bh_rets, rule_rets, active, rows = [], [], [], [], [], []
    for ws, win in we.cold_week_windows(returns):
        split = we.split_label(ws, val_start, test_start)
        if split == "train":
            continue                                           # gate on OOS weeks only (val+test)
        base = we.grade_week_baselines(ws, win, liq, vol, k=k, vol_target=vol_target, cap_floor=cap_floor,
                                       vol_mult=vol_mult)
        eq, recs, *_ = evaluate_event_policy(make_predict(), win, btc, liq, vol, env_kwargs)
        # return from the $10k DEPOSIT (capital), NOT eq[0]: a basket_default policy's eq[0] is already
        # post-entry-cost, so dividing by it would manufacture a spurious edge over B&H (which measures
        # from capital). The competition scores final/$10k; a flat-start policy has eq[0]==capital anyway.
        cap = float(env_kwargs.get("capital", we.START_CAPITAL))
        pol_ret = float(eq.iloc[-1] / cap - 1.0)               # eq_trace spans the cold week only
        pol_dd = abs(float((eq / eq.cummax() - 1.0).min()))
        tdays = we._trade_days(recs, ws)
        pol_rets.append(pol_ret); pol_dds.append(pol_dd)
        bh_rets.append(base.buyhold_ret); rule_rets.append(base.rung0_ret)
        active.append(len(tdays) >= 7)
        rows.append((ws, split, base.regime, pol_ret, pol_dd, len(tdays), base.buyhold_ret, base.rung0_ret))
    verdict = we.weekly_gate(pol_rets, pol_dds, bh_rets, rule_rets, active, seed=seed)
    return verdict, rows


def print_weekly_verdict(verdict, rows):
    """Per-week table + the random-week distribution gate (the sweep's primary read in weekly mode)."""
    import datetime as dt
    print(f"[weekly] {'week':>10} {'split':>5} {'regime':>6} {'policy':>8} {'maxDD':>6} {'days':>4} "
          f"{'B&H':>8} {'rung0':>8}")
    for ws, split, regime, pr, dd, td, bh, r0 in rows:
        d = dt.datetime.fromtimestamp(ws, dt.timezone.utc).date()
        print(f"[weekly] {str(d):>10} {split:>5} {regime:>6} {pr:>+7.1%} {dd:>5.0%} {td:>2}/7 "
              f"{bh:>+7.1%} {r0:>+7.1%}")
    blo, bhi = verdict["edge_buyhold_ci"]
    rlo, rhi = verdict["edge_rung0_ci"]
    print(f"[weekly] policy mean {verdict['policy_mean']:+.2%}  B&H {verdict['buyhold_mean']:+.2%}  "
          f"rung-0 {verdict['rung0_mean']:+.2%}  worst-week DD {verdict['worst_week_dd']:.0%}  "
          f"activity-miss {verdict['activity_fail_weeks']}wk")
    print(f"[weekly] PAIRED edge vs B&H {verdict['edge_vs_buyhold']:+.2%} (95%% CI [{blo:+.2%},{bhi:+.2%}])"
          f"  vs rung-0 {verdict['edge_vs_rung0']:+.2%} (95%% CI [{rlo:+.2%},{rhi:+.2%}])")
    print(f"[weekly] gate: {'PASS' if verdict['pass'] else 'FAIL'}"
          + ("" if verdict["pass"] else f" (binding: {verdict['binding']})")
          + f"  checks={verdict['checks']}")


def print_verdict(r):
    """Print the per-split [regime]/[baselines]/[verdict]/[gate] block for one split's result."""
    rg, rep = r["regime"], r["report"]
    dqd = " [DQ'd >30%]" if r.get("base_dd", 0.0) > 0.30 else ""
    print(f"[{r['name']}] regime: BTC {rg['btc_return']:+.1%}  universe-EW {rg['universe_ew_return']:+.1%}  "
          f"({rg['label']})  |  Buy&Hold {r['bh']:+.1%}  Random {r['rnd']:+.1%}  "
          f"rung-0 {r['base']:+.1%} (DD {r.get('base_dd', 0.0):.0%}{dqd})")
    print(f"[{r['name']}] policy {r['pol']:+.1%} (Sh {rep.sharpe_ratio:.2f}, DD {rep.max_drawdown_pct:.1%}) | "
          f"vs Buy&Hold {'BEATS' if r['pol'] > r['bh'] else 'LOSES'} ({r['pol'] - r['bh']:+.1%}) | "
          f"vs rung-0 {'BEATS' if r['pol'] > r['base'] else 'LOSES'} ({r['pol'] - r['base']:+.1%}) | "
          f"vs Random {'BEATS' if r['pol'] > r['rnd'] else 'LOSES'}")
    print(f"[{r['name']}] gate: {'PASS' if r['gate_pass'] else 'FAIL'}"
          + ("" if r["gate_pass"] else f" (binding: {r['binding']})"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--run-id", default="ppo-event")
    p.add_argument("--out", default=None)
    p.add_argument("--publish-target", default=None)
    p.add_argument("--episode-bars", type=int, default=168, help="weekly episodes by default")
    p.add_argument("--curriculum-horizon", default="", help="horizon curriculum (2026-06-14): ramp "
                   "episode_bars DOWN over training as 'bars:progress' pairs, e.g. "
                   "'672:0.0,336:0.40,168:0.70' (long episodes first teach holding the bull where the "
                   "missed-run cost is in-episode/creditable; anneal to the 1wk deploy shape). '' = OFF "
                   "(constant --episode-bars). The env is built at the LARGEST horizon; it only shrinks.")
    p.add_argument("--curriculum-universe", default="", help="universe-regime curriculum (2026-06-15): "
                   "ramp the TRAINING universe through volatility regimes as 'mode:progress' pairs, e.g. "
                   "'lowvol:0.0,broad:0.35,voltopk:0.65' (the k calmest tokens first to learn basics on "
                   "tractable dynamics, anneal to the voltopk deploy distribution). '' = OFF (constant "
                   "--universe-mode). Must END at --universe-mode; EVAL always runs on --universe-mode.")
    p.add_argument("--max-entry-frac", type=float, default=0.34)
    p.add_argument("--stop-k", type=float, default=0.25)
    p.add_argument("--cooldown", type=int, default=48)
    p.add_argument("--reward-mode", default="absolute",
                   choices=["absolute", "relative", "residual", "residual_ranked", "entry_forward"],
                   help="relative = beat the rule's portfolio return; residual = per-decision weight "
                        "deviations x returns; residual_ranked = demeaned residual + budget; "
                        "entry_forward = per-entry dev x (fwd_ret - typical-ignition) (the corr metric)")
    p.add_argument("--fwd-horizon", type=int, default=24, help="entry_forward forward-return window (bars)")
    p.add_argument("--ungate", action="store_true", help="exp5 selector: decide over every in-universe "
                   "ignition (drop rung-0's cooled&reclaimed gate -> ~960 vs 39 decisions)")
    p.add_argument("--action-mode", default="continuous", choices=["continuous", "discrete"],
                   help="discrete = categorical size/keep levels (no continuous-head boundary collapse)")
    p.add_argument("--n-action-levels", type=int, default=4, help="discrete: # of size/keep levels")
    p.add_argument("--universe-mode", default="voltopk", choices=["voltopk", "broad", "lowvol", "fixed"],
                   help="universe axis: voltopk (chaos) | broad (stratified) | lowvol (calm) | fixed (a "
                        "hand-set token list, NO causal re-pick — requires --fixed-universe)")
    p.add_argument("--fixed-universe", default="", help="comma-separated token set for --universe-mode "
                   "fixed (e.g. 'FF,HUMA,Q'); identical basket every episode, no causal vol re-pick")
    p.add_argument("--shallow-break-max", type=float, default=0.0, help="suppress the EMA-break exit when "
                   "the break is SHALLOW (cushion > -this) AND the token is QUIET (--consol-vol-max): a "
                   "sideways noise-dip that shakes the agent out before a pump. 0 = off. loss_floor/trail stay")
    p.add_argument("--consol-vol-max", type=float, default=0.0, help="the QUIET threshold (24h realized vol "
                   "< this) for --shallow-break-max sideways EMA-break suppression. 0 = off")
    p.add_argument("--rotate-pump-block", type=float, default=0.0, help="ANTI-CHASE rotation brake: skip "
                   "loser-funded rotation when the candidate has run up > this over --rotate-pump-win bars "
                   "(don't SELL a holding to chase an already-pumped token). 0 = off")
    p.add_argument("--rotate-pump-win", type=int, default=24, help="lookback bars for --rotate-pump-block "
                   "run-up (default 24h)")
    p.add_argument("--vol-target", type=float, default=0.0, help="risk-parity: >0 caps each token's "
                   "weight at vol_target/trailing_vol (clip [cap-floor, max-entry-frac]); 0 = flat cap")
    p.add_argument("--cap-floor", type=float, default=0.02, help="risk-parity: min per-token weight cap")
    p.add_argument("--universe-lookback", type=int, default=0, help="trailing bars for universe vol "
                   "ranking (0 = warmup/168h default; 24=1d 720=1mo 2160=3mo 4320=6mo)")
    p.add_argument("--harvest-obs", action="store_true", help="lever-2: append the event token's r24/r3d/r7d "
                   "momentum slots (OBS_DIM 13->16) so the policy can size UP on bull-harvest setups")
    p.add_argument("--cycle-obs", action="store_true", help="SPENT-MOVE knowledge: 2 obs slots - "
                   "the event token's ret-since / bars-since its PRIOR ignition (probe: "
                   "prior-paid>10%% ignitions return -6..-7%% fwd-24h vs -1..-2%% fresh, train AND val)")
    p.add_argument("--rule-default", action="store_true", help="rung-1b: discrete action idx 0 EXECUTES "
                   "rung-0's decision (entry at rule sizing / exit full cut); deviations are earned")
    p.add_argument("--basket-default", action="store_true", help="long-default OVERLAY (2026-06-14): "
                   "start fully long the risk-parity basket (= Buy&Hold); events TILT names off it and "
                   "the default action HOLDS, so doing nothing ~= B&H — closes the +13%% bull-gap the "
                   "event-only skeleton bleeds on cold weekly sessions. Builds on --rule-default.")
    p.add_argument("--exit-commit", type=int, default=0, help="rung-1b: bars a non-cut exit decision "
                   "commits for (no re-prompt drip); 0 = legacy per-bar re-prompting")
    p.add_argument("--dust-usd", type=float, default=0.0, help="rung-1b: partial keeps below this USD "
                   "force a full close (kills the sub-$1 gas-bleeding trim tail); 0 = legacy")
    p.add_argument("--rule-prior", type=float, default=0.0, help="rung-1b: +logit bias on action idx 0 at "
                   "init so the untrained policy ~= the rule and PPO must learn to deviate")
    p.add_argument("--tp-rungs", default="", help="profit-take prompts at these unrealized-gain levels "
                   "(comma list, e.g. 0.25,0.5,1,2) — lets the agent SELL INTO STRENGTH; '' = off")
    p.add_argument("--recurrent", action="store_true", help="RecurrentPPO (sb3-contrib MlpLstmPolicy): "
                   "give the policy MEMORY across the episode's events — the sequence skills the "
                   "forensics demand (trade-the-pump-then-walk-away, hold-the-winner, don't-rebuy-"
                   "the-bleed) are inexpressible for a stateless MLP")
    p.add_argument("--lstm-size", type=int, default=256, help="LSTM hidden size (TradeSim converged 256)")
    p.add_argument("--det-blacklist", type=int, default=0, help="detonation blacklist: after a massive "
                   "surge WHILE price collapses (the Q pattern), zero the token's ignitions for N bars "
                   "(probe-calibrated 672 = 4wk; post-det ignitions are poison); 0 = off")
    p.add_argument("--loss-floor", type=float, default=0.0, help="disaster floor: a position below "
                   "entry*(1-floor) cannot be overridden/trimmed — forced full cut, punctures the "
                   "exit-commit window (closes the override-down-a-crash loss path); 0 = off")
    p.add_argument("--intrabar-floor", action="store_true", help="the floor is a RESTING STOP: "
                   "filled where the bar's LOW crossed entry*(1-floor), not at the next close — "
                   "closes the Q hole (a -53%% bar blowing through the floor); needs --loss-floor")
    p.add_argument("--wick-reject", type=float, default=0.0, help="kill ignitions whose trigger bar "
                   "closed below (1-X)*high (extreme upper-wick rejection = the dump is mid-flight); "
                   "probe-calibrated 0.30; the mild 0.10 version is REFUTED — 0 = off")
    p.add_argument("--scale-in", action="store_true", help="let the agent ADD to a HELD winner on a "
                   "fresh ignition (a held token's re-ignition is otherwise invisible — it missed a "
                   "+16%% ZEC re-ignition runup); fenced: in-profit + under the per-token cap ONLY, so "
                   "it cannot average down into the disaster floor or pyramid past the risk-parity cap")
    p.add_argument("--eval-prepad", action="store_true", help="serve each eval window's 168-bar signal "
                   "warmup from the TAIL OF THE PRIOR SPLIT (contiguous time), so the published window "
                   "is tradeable from bar 0 — no dead first week on the charts; mirrors live trading, "
                   "where full history always exists behind the current bar")
    p.add_argument("--k", type=int, default=8, help="universe size (# tokens the agent trades); broaden "
                   "beyond rung-0's 8 to diversify the risk-parity drawdown (the alts are ~uncorrelated)")
    p.add_argument("--vol-mult", type=float, default=2.5, help="ignition volume-surge threshold "
                   "(4h-avg vol / prior-164h-avg >= this). Lower (e.g. 2.0) fires earlier + lets the "
                   "policy LEARN the cutoff from the surge obs instead of hard-coding 2.5")
    p.add_argument("--crash-train", type=int, default=0, help="inject N synthetic alt-crashes into the "
                   "TRAINING data so the agent sees crashes and learns to de-risk into low breadth")
    p.add_argument("--crash-eval", action="store_true", help="add a held-out CRASH regime (a crash spliced "
                   "into the test window) to the per-regime gate — where de-risking finally pays")
    p.add_argument("--crash-depth", type=float, default=-0.6, help="systemic drop of an injected crash")
    p.add_argument("--crash-beta", type=float, default=1.4, help="alt stress beta in a crash (realized "
                   "alt drop ~ beta*crash-depth; 1.4*-0.6 ~ -84%%, SIREN-scale)")
    p.add_argument("--norm-reward", action="store_true", help="VecNormalize norm_reward (for the small "
                   "zero-centered relative/residual rewards)")
    p.add_argument("--r4-beta", type=float, default=0.0, help="residual R4 foregone-opportunity penalty: "
                   "charge beta x surrendered upside when the agent under-sizes a token that rose")
    p.add_argument("--res-gamma", type=float, default=0.0,
                   help="residual_ranked quadratic deviation-budget weight (interior optimum; set via "
                        "scripts/preflight_residual.py)")
    p.add_argument("--dd-lambda", type=float, default=2.0)
    p.add_argument("--dd-soft", type=float, default=0.15, help="drawdown penalty soft knee")
    p.add_argument("--ent-coef", type=float, default=0.1)
    p.add_argument("--n-epochs", type=int, default=10, help="PPO epochs per rollout (update conservatism; semi-MDP decisions are few)")
    p.add_argument("--target-kl", type=float, default=None, help="PPO early-stop KL per update (None = SB3 default, unconstrained)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-end", type=float, default=None, help="if set, linearly anneal lr -> lr-end")
    p.add_argument("--eval-split", default="val", choices=["val", "test"])
    p.add_argument("--eval-mode", default="continuous", choices=["continuous", "weekly"],
                   help="continuous = one held episode per split (legacy, flattering); weekly = the "
                        "COLD weekly-session DISTRIBUTION gate (the deployment structure, 2026-06-14 "
                        "fork) — bootstrap-CI beat B&H + rung-0 over random OOS weeks, worst-week DD<30%%")
    p.add_argument("--no-btc-obs", action="store_true", help="neutralize the btc_trend obs slot — the "
                   "tokens are BTC-decorrelated by selection, so a BTC-anchored regime signal is near-noise")
    p.add_argument("--reexport", action="store_true", help="regenerate + republish the bundle from the "
                   "SAVED policy.zip (NO retrain) — e.g. after a marker/export fix. Config is read from "
                   "the saved provenance so the eval is byte-identical; only the published artifacts change.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    config.load_dotenv()
    out = args.out or os.path.join("runs-rl", args.run_id)
    os.makedirs(out, exist_ok=True)
    if args.reexport:                                       # rebuild config from the saved provenance so
        import json                                         # env_kwargs / recurrent / eval match the run
        prov = json.load(open(os.path.join(out, args.run_id, "metrics.json"),
                              encoding="utf-8"))["provenance"]
        for k, v in prov.items():
            if hasattr(args, k) and k not in ("git_commit", "env"):
                setattr(args, k, v)
    hsched = parse_horizon_schedule(args.curriculum_horizon)        # [] when OFF
    construct_bars = max_horizon(hsched) if hsched else args.episode_bars   # build env at the LARGEST horizon
    usched = parse_universe_schedule(args.curriculum_universe)      # [] when OFF
    construct_universe = usched[0][1] if usched else args.universe_mode   # phase-0 regime for the TRAINING start
    if usched and usched[-1][1] != args.universe_mode:             # eval runs on --universe-mode, so the
        raise ValueError(f"--curriculum-universe must END at the deploy --universe-mode "  # ramp must land there
                         f"({args.universe_mode!r}); schedule ends at {usched[-1][1]!r}")

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    from train_rl import build_portfolio_artifacts, build_volume_panel, load_data, time_split, trade_stats
    from trader.train.gym_env import GymEventRungEnv

    returns, btc, anchor, liq = load_data()
    train_r, val_r, test_r = time_split(returns)
    train_pre = train_r                                     # pristine (pre-crash-injection) for eval prepad
    if args.crash_train > 0:                                # augment TRAINING data so the agent sees crashes
        from trader.sim.crash import inject_random_crashes
        train_r, placed = inject_random_crashes(train_r, n_crashes=args.crash_train,
                                                rng=np.random.default_rng(args.seed),
                                                total_drop=args.crash_depth, beta=args.crash_beta)
        print(f"[crash] injected {len(placed)} training crashes at bars {placed}")
    vol = build_volume_panel(list(returns.columns), returns.index)
    ohlc_kwargs = {}
    if args.intrabar_floor or args.wick_reject > 0:
        from train_rl import build_ohlc_frac_panels
        lowf, highf = build_ohlc_frac_panels(list(returns.columns), returns.index)
        ohlc_kwargs = {"low_frac": lowf if args.intrabar_floor else None,
                       "intrabar_floor": args.intrabar_floor,
                       "high_frac": highf if args.wick_reject > 0 else None,
                       "wick_reject": args.wick_reject}
    env_kwargs = dict(k=args.k, vol_mult=args.vol_mult, warmup=WARMUP, max_entry_frac=args.max_entry_frac, stop_k=args.stop_k,
                      cooldown=args.cooldown, dd_lambda=args.dd_lambda, dd_soft=args.dd_soft,
                      reward_mode=args.reward_mode, r4_beta=args.r4_beta, res_gamma=args.res_gamma,
                      fwd_horizon=args.fwd_horizon, ungate=args.ungate,
                      action_mode=args.action_mode, n_action_levels=args.n_action_levels,
                      universe_mode=args.universe_mode, vol_target=args.vol_target,
                      cap_floor=args.cap_floor, harvest_obs=args.harvest_obs,
                      rule_default=args.rule_default, basket_default=args.basket_default,
                      exit_commit=args.exit_commit, dust_usd=args.dust_usd,
                      tp_rungs=[float(x) for x in args.tp_rungs.split(",") if x],
                      loss_floor=args.loss_floor, det_blacklist=args.det_blacklist,
                      scale_in=args.scale_in,
                      shallow_break_max=args.shallow_break_max, consol_vol_max=args.consol_vol_max,
                      rotate_pump_block=args.rotate_pump_block, rotate_pump_win=args.rotate_pump_win,
                      cycle_obs=args.cycle_obs, universe_lookback=args.universe_lookback,
                      no_btc_obs=args.no_btc_obs,
                      fixed_universe=[t.strip() for t in args.fixed_universe.split(",") if t.strip()] or None,
                      **ohlc_kwargs, seed=args.seed)

    write_progress(out, state="running", phase="setup", run_id=args.run_id, timesteps=0,
                   total=args.timesteps)

    def make_env(rank):
        def _f():
            return Monitor(GymEventRungEnv(train_r, btc, liq, volume=vol,
                                           episode_bars=construct_bars,   # max horizon; curriculum shrinks it
                                           **{**env_kwargs, "seed": args.seed + rank,
                                              "universe_mode": construct_universe}))  # phase-0; curriculum ramps it
        return _f

    if args.reexport:                                     # regenerate the bundle from the SAVED policy
        import pickle                                      # (no retrain): load the EXACT policy.zip +
        if args.recurrent:                                # vecnormalize.pkl, so the verdict is byte-
            from sb3_contrib import RecurrentPPO           # identical and only the published artifacts
            model = RecurrentPPO.load(os.path.join(out, "policy.zip"), device="cpu")   # change.
        else:
            model = PPO.load(os.path.join(out, "policy.zip"), device="cpu")
        with open(os.path.join(out, "vecnormalize.pkl"), "rb") as f:
            venv = pickle.load(f)                          # VecNormalize.normalize_obs works standalone
        print(f"[reexport] loaded {args.run_id} policy.zip + vecnormalize.pkl (no retrain)")
    else:
        venv = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
        venv = VecNormalize(venv, norm_obs=True, norm_reward=args.norm_reward, clip_obs=10.0)

        class ProgressCb(BaseCallback):
            def _on_step(self) -> bool:
                if self.num_timesteps % 2048 < venv.num_envs:
                    rews = [e["r"] for e in self.model.ep_info_buffer] if self.model.ep_info_buffer else []
                    write_progress(out, state="running", phase="train", timesteps=self.num_timesteps,
                                   total=args.timesteps,
                                   mean_reward=float(np.mean(rews)) if rews else None, history_key="curve")
                return True

        class HorizonCurriculumCallback(BaseCallback):
            """Push the scheduled episode horizon into EVERY sub-env between episodes — the REAL,
            non-cosmetic curriculum (it mutates the sampler via `EventRungEnv.set_episode_bars`; the
            TradeSim post-mortem's #1 lesson was a curriculum that only logged phase names). Only fires
            when the target changes (next reset() picks it up; never mid-episode)."""
            def __init__(self, schedule, total):
                super().__init__()
                self._sched, self._total, self._cur = schedule, max(int(total), 1), None
            def _on_step(self) -> bool:
                target = horizon_at(self._sched, self.num_timesteps / self._total)
                if target != self._cur:
                    self.training_env.env_method("set_episode_bars", target)
                    self._cur = target
                return True

        class UniverseCurriculumCallback(BaseCallback):
            """Push the scheduled universe REGIME into every sub-env between episodes — the volatility-
            axis analog of HorizonCurriculumCallback (mutates which k tokens `reset()` samples via
            `EventRungEnv.set_universe_mode`; non-cosmetic). Fires only when the target changes; the
            next reset() picks it up (never mid-episode)."""
            def __init__(self, schedule, total):
                super().__init__()
                self._sched, self._total, self._cur = schedule, max(int(total), 1), None
            def _on_step(self) -> bool:
                target = universe_at(self._sched, self.num_timesteps / self._total)
                if target != self._cur:
                    self.training_env.env_method("set_universe_mode", target)
                    self._cur = target
                return True

        lr = args.lr
        if args.lr_end is not None:                        # linear anneal lr -> lr_end (progress: 1->0)
            lr0, lr1 = args.lr, args.lr_end
            lr = lambda pr: lr1 + (lr0 - lr1) * pr  # noqa: E731
        if args.recurrent:                                 # memory across the episode's events
            from sb3_contrib import RecurrentPPO
            model = RecurrentPPO("MlpLstmPolicy", venv, verbose=0, seed=args.seed, n_steps=1024,
                                 batch_size=256, ent_coef=args.ent_coef, learning_rate=lr,
                                 n_epochs=args.n_epochs, target_kl=args.target_kl,
                                 policy_kwargs=dict(lstm_hidden_size=args.lstm_size))
        else:
            model = PPO("MlpPolicy", venv, verbose=0, seed=args.seed, n_steps=1024, batch_size=256,
                        ent_coef=args.ent_coef, learning_rate=lr,
                        n_epochs=args.n_epochs, target_kl=args.target_kl)
        if args.rule_default and args.rule_prior > 0:      # default-executes-the-rule prior: bias the
            import torch                                   # categorical head toward idx 0 at init, so the
            with torch.no_grad():                          # untrained policy ~= rung-0 and deviation is learned
                model.policy.action_net.bias[0] += args.rule_prior
        callbacks = [ProgressCb()]
        if hsched:                                         # ramp episode_bars down across training
            callbacks.append(HorizonCurriculumCallback(hsched, args.timesteps))
            print(f"[curriculum] horizon schedule {hsched} (env built at {construct_bars} bars)")
        if usched:                                         # ramp the training universe across regimes
            callbacks.append(UniverseCurriculumCallback(usched, args.timesteps))
            print(f"[curriculum] universe schedule {usched} (deploy/eval mode {args.universe_mode})")
        model.learn(total_timesteps=args.timesteps, callback=callbacks)
        model.save(os.path.join(out, "policy.zip"))        # persist the trained policy + the obs-
        venv.save(os.path.join(out, "vecnormalize.pkl"))   # normalization stats: re-loadable for the
        #   simulator/deployment (every pre-2026-06-12 policy was lost on process exit). Stays on the
        #   training box (outside the published bundle subdir).

    write_progress(out, state="running", phase="evaluate")

    def make_predict():
        """Fresh per-episode predictor. Recurrent: thread the LSTM state across the episode's
        events (one state per split eval — memory is the point); stateless MLP path unchanged."""
        st = {"s": None, "start": np.ones(1, dtype=bool)}

        def predict_fn(obs):
            norm = venv.normalize_obs(obs.reshape(1, -1))
            if args.recurrent:
                a, st["s"] = model.predict(norm, state=st["s"], episode_start=st["start"],
                                           deterministic=True)
                st["start"] = np.zeros(1, dtype=bool)
            else:
                a, _ = model.predict(norm, deterministic=True)
            return np.asarray(a).reshape(-1)
        return predict_fn

    # Per-regime held-out eval: grade the policy on BOTH val and test (the reversal pocket AND the
    # BTC-bear/alt-flat window) so a pass can't hide in the friendlier regime. Overall gate = all pass.
    held = {"val": val_r, "test": test_r}
    if args.crash_eval:                                    # held-out CRASH regime: a crash spliced into test
        from trader.sim.crash import inject_crash
        held["crash"] = inject_crash(test_r, at=len(test_r) // 2, duration=8,
                                     total_drop=args.crash_depth, beta=args.crash_beta, seed=args.seed)
    if args.eval_prepad:                                   # serve the warmup from the PRIOR split's tail
        prev = {"val": train_pre, "test": val_r, "crash": val_r}   # (contiguous time; pristine train)
        held = {nm: pd.concat([prev[nm].tail(WARMUP), r]) for nm, r in held.items()}
    weekly_verdict = None
    if args.eval_mode == "weekly":                         # the DEPLOYMENT-structure gate (2026-06-14 fork)
        weekly_verdict, weekly_rows = evaluate_weekly_gate(
            returns, btc, liq, vol, env_kwargs, make_predict, int(val_r.index[0]), int(test_r.index[0]),
            args.seed, args.k, args.vol_target, args.cap_floor, args.vol_mult)
        print_weekly_verdict(weekly_verdict, weekly_rows)
        overall_gate = weekly_verdict["pass"]
        pr = evaluate_and_gate(args.eval_split, held[args.eval_split], btc, liq, vol, env_kwargs,
                               make_predict(), args.seed)    # continuous replay -> dashboard CHART only
        results = {args.eval_split: pr}                    # the [eval] line keeps the continuous format
        print(f"[eval] primary={args.eval_split} events={len(pr['raw'])} action "   # (smoke-gate parses
              f"mean={np.mean(pr['raw']):.3f} min={min(pr['raw']):.3f} max={max(pr['raw']):.3f}")  # it)
    else:                                                  # legacy: one held episode per split
        results = {nm: evaluate_and_gate(nm, r, btc, liq, vol, env_kwargs, make_predict(), args.seed)
                   for nm, r in held.items()}
        pr = results[args.eval_split]                      # primary split -> the published bundle
        print(f"[eval] primary={args.eval_split} events={len(pr['raw'])} action "
              f"mean={np.mean(pr['raw']):.3f} min={min(pr['raw']):.3f} max={max(pr['raw']):.3f}")
        for nm in results:
            print_verdict(results[nm])
        overall_gate = all(r["gate_pass"] for r in results.values())
        print(f"[gate] OVERALL: {'PASS - beats the rung-0 RULE + survives DQ on EVERY held-out regime' if overall_gate else 'FAIL'}"
              + ("" if overall_gate else " - must beat the rung-0 RULE + survive DQ on val AND test (Buy&Hold/Random reported)"))

    eq, records, universe, fees, raw, report = (pr["eq"], pr["records"], pr["universe"], pr["fees"],
                                                pr["raw"], pr["report"])
    token_pnl = pr["token_pnl"]                            # exact per-token PnL ledger (realized + open
    #   marked at the LAST bar) — the frontend reads this instead of reconstructing from markers
    metrics = ap.metrics_to_frontend(report)
    metrics["total_fees_paid"] = fees
    pub_r = held[args.eval_split]                          # published window starts at the first
    d0 = int(pub_r.index[WARMUP if args.eval_prepad else 0])   # TRADEABLE bar when prepadded —
    d1 = int(pub_r.index[-1])                              # no dead warmup week on the charts
    weights, candles, trades = build_portfolio_artifacts(records, universe, d0, d1)
    metrics.update(trade_stats(trades))
    metrics.update({"baseline_return": pr["base"], "buyhold_return": pr["bh"], "random_return": pr["rnd"],
                    "regime": pr["regime"], "gate_pass": overall_gate, "eval_mode": args.eval_mode,
                    "gate_binding": (weekly_verdict["binding"] if weekly_verdict else pr["binding"]),
                    "regimes": {nm: {"return": r["pol"], "baseline_return": r["base"],
                                     "buyhold_return": r["bh"], "random_return": r["rnd"],
                                     "regime": r["regime"], "maxdd": r["report"].max_drawdown_pct,
                                     "gate_pass": r["gate_pass"], "gate_binding": r["binding"]}
                                for nm, r in results.items()}})
    if weekly_verdict is not None:                         # the DEPLOYMENT-structure verdict (the real gate)
        metrics["weekly"] = {k: v for k, v in weekly_verdict.items() if k != "checks"}
        metrics["weekly"]["checks"] = weekly_verdict["checks"]
        metrics["weekly"]["weeks"] = [
            {"ws": ws, "split": sp, "regime": rg, "return": prr, "maxdd": dd,
             "trade_days": td, "buyhold": bh, "rung0": r0}
            for ws, sp, rg, prr, dd, td, bh, r0 in weekly_rows]

    import subprocess
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).stdout.strip()
    except Exception:  # noqa: BLE001
        sha = "unknown"
    metrics["provenance"] = {"git_commit": sha, "env": "event_rung", "timesteps": args.timesteps,
                             "seed": args.seed, "n_envs": args.n_envs, "episode_bars": args.episode_bars,
                             "max_entry_frac": args.max_entry_frac, "stop_k": args.stop_k,
                             "cooldown": args.cooldown, "reward_mode": args.reward_mode,
                             "norm_reward": args.norm_reward, "r4_beta": args.r4_beta,
                             "res_gamma": args.res_gamma, "fwd_horizon": args.fwd_horizon,
                             "ungate": args.ungate, "action_mode": args.action_mode,
                             "n_action_levels": args.n_action_levels, "universe_mode": args.universe_mode,
                             "vol_mult": args.vol_mult,                  # was UNRECORDED -> sims defaulted to 2.5
                             "fixed_universe": [t.strip() for t in args.fixed_universe.split(",") if t.strip()] or None,
                             "vol_target": args.vol_target, "cap_floor": args.cap_floor, "k": args.k, "universe_lookback": args.universe_lookback,
                             "harvest_obs": args.harvest_obs, "cycle_obs": args.cycle_obs,
                             "rule_default": args.rule_default, "basket_default": args.basket_default,
                             "exit_commit": args.exit_commit, "dust_usd": args.dust_usd,
                             "rule_prior": args.rule_prior, "tp_rungs": args.tp_rungs,
                             "eval_prepad": args.eval_prepad, "loss_floor": args.loss_floor,
                             "intrabar_floor": args.intrabar_floor, "wick_reject": args.wick_reject,
                             "scale_in": args.scale_in,
                             "shallow_break_max": args.shallow_break_max, "consol_vol_max": args.consol_vol_max,
                             "rotate_pump_block": args.rotate_pump_block, "rotate_pump_win": args.rotate_pump_win,
                             "det_blacklist": args.det_blacklist, "recurrent": args.recurrent,
                             "lstm_size": args.lstm_size if args.recurrent else None,
                             "crash_train": args.crash_train, "crash_eval": args.crash_eval,
                             "crash_depth": args.crash_depth, "crash_beta": args.crash_beta,
                             "dd_lambda": args.dd_lambda, "dd_soft": args.dd_soft,
                             "ent_coef": args.ent_coef, "n_epochs": args.n_epochs, "target_kl": args.target_kl, "lr": args.lr, "lr_end": args.lr_end,
                             "no_btc_obs": args.no_btc_obs, "eval_mode": args.eval_mode,
                             "curriculum_horizon": args.curriculum_horizon,
                             "curriculum_universe": args.curriculum_universe,
                             "eval_split": args.eval_split}
    eq_pub = eq.iloc[::6]                                   # ~6-bar resolution for the chart
    # self-describing display name: the frontend should never be ambiguous about which run/config it shows
    flags = (f"{args.reward_mode} k{args.k}/{args.universe_mode} dd{args.dd_lambda}"
             + (f" +lstm{args.lstm_size}" if args.recurrent else "")
             + (" +rd" if args.rule_default else "") + (" +basket" if args.basket_default else "")
             + (" +tp" if args.tp_rungs else "")
             + (" +harvest" if args.harvest_obs else "") + (" +cyc" if args.cycle_obs else "")
             + (" +crash" if args.crash_eval else ""))
    model_name = f"{args.run_id} @{sha} | {flags} | s{args.seed} {args.timesteps // 1000}k"
    entry = ap.export_portfolio_run(out, args.run_id, equity=eq_pub, metrics=metrics, weights=weights,
                                    token_candles=candles, token_trades=trades, universe=universe,
                                    model_name=model_name, token_pnl=token_pnl,
                                    action_mode="event", regime=args.eval_split,
                                    timestamp=datetime.now(timezone.utc).isoformat())
    target = args.publish_target or config.get("APENTIC_PUBLISH_TARGET")
    if target:
        ap.publish_run(os.path.join(out, args.run_id), args.run_id, entry, target,
                       cloudfront_dist_id=config.get("APENTIC_CLOUDFRONT_DIST_ID"))
    write_progress(out, state="complete", run_id=args.run_id, total_return=report.total_return_pct,
                   sharpe=report.sharpe_ratio, max_drawdown=report.max_drawdown_pct, trades=len(trades))
    print(f"[train_event] {args.run_id}: return {report.total_return_pct:+.1%}, "
          f"Sharpe {report.sharpe_ratio:.2f}, maxDD {report.max_drawdown_pct:.1%}, "
          f"events {len(raw)}, trades {metrics['total_trades']}")


if __name__ == "__main__":
    main()
