"""Build the committed experiment ledger from the published bundles.

The rigid, reproducible record of every training iteration (exact config -> performance), so we
can always find and RETURN TO the best known formula when a tweak degrades things. This is the
TradeSim lesson made structural: never tweak without a permanent, version-controlled performance
trail.

    python scripts/build_ledger.py            # rebuild experiments/ledger.jsonl + champion.json

Source of truth = the immutable published metrics.json per run (which now carries a `provenance`
block: git commit + every hyperparameter). Deterministically rebuildable, so the ledger is just
an aggregation we commit to git — the bundles are the canonical record.

Champion rule: highest MEAN return among reward configs whose MEAN max-drawdown stays under the
~30% DQ gate (single-seed RL is unstable, so we judge on the seed mean, and also report the worst
seed's drawdown so we know the margin to the gate).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from collections import defaultdict

HOST = "https://data.alexlouis.dev"
SWEEP = re.compile(r"^ppo-(sharpe|giveback|realized|turnover)-s(\d+)$")
OUT_DIR = "experiments"


def fetch(u):
    with urllib.request.urlopen(u, timeout=30) as r:
        return json.load(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=HOST)
    p.add_argument("--dd-gate", type=float, default=0.30)
    args = p.parse_args()

    man = fetch(f"{args.host}/manifest.json")
    rows = []
    for e in man:
        if e.get("kind") != "portfolio":
            continue
        rid = e["id"]
        try:
            m = fetch(f"{args.host}/{rid}/metrics.json")
        except Exception:  # noqa: BLE001
            continue
        prov = m.get("provenance", {})
        sm = SWEEP.match(rid)
        steps = prov.get("timesteps")
        if steps is None:
            mt = re.search(r"([\d,]+)\s*steps", e.get("model_name", ""))
            steps = int(mt.group(1).replace(",", "")) if mt else None
        rows.append({
            "run_id": rid,
            "mode": prov.get("reward_mode") or (sm.group(1) if sm else m.get("reward_mode", "?")),
            "seed": prov.get("seed", int(sm.group(2)) if sm else None),
            "timesteps": steps,
            "git": prov.get("git_commit"),
            "return": m.get("total_return_pct"), "sharpe": m.get("sharpe_ratio"),
            "maxdd": m.get("max_drawdown_pct"), "pf": m.get("profit_factor"),
            "win": m.get("win_rate"), "trades": m.get("total_trades"),
            "turnover_usd": m.get("eval_turnover_usd"), "realized_usd": m.get("eval_realized_usd"),
            "giveback": m.get("eval_giveback"), "baseline": m.get("baseline_return"),
            "beats_baseline": (m.get("total_return_pct") or 0) > (m.get("baseline_return") or 0),
            "legal_dd": (m.get("max_drawdown_pct") is not None
                         and m.get("max_drawdown_pct") < args.dd_gate),
            "config": prov or "(pre-provenance: see model_name / run_id)",
        })

    rows.sort(key=lambda r: (r["mode"] or "", r["seed"] if r["seed"] is not None else -1))
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "ledger.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # per-mode summary over the formal sweep runs only (ppo-<mode>-s<seed>)
    by_mode = defaultdict(list)
    for r in rows:
        if SWEEP.match(r["run_id"]) and r["return"] is not None:
            by_mode[r["mode"]].append(r)
    summary = {}
    for mode, rs in by_mode.items():
        n = len(rs)
        mean = lambda k: sum(x[k] for x in rs if x[k] is not None) / max(n, 1)
        summary[mode] = {
            "n": n, "seeds": sorted(x["seed"] for x in rs),
            "mean_return": mean("return"), "mean_maxdd": mean("maxdd"),
            "mean_sharpe": mean("sharpe"), "mean_pf": mean("pf"),
            "worst_maxdd": max((x["maxdd"] for x in rs if x["maxdd"] is not None), default=None),
            "timesteps": rs[0].get("timesteps"), "git": rs[0].get("git"),
            "legal_mean": mean("maxdd") < args.dd_gate,
        }

    legal = {k: v for k, v in summary.items() if v["legal_mean"]}
    champ = max((legal or summary), key=lambda k: summary[k]["mean_return"]) if summary else None
    champion = None
    if champ:
        s = summary[champ]
        champion = {
            "mode": champ, "by_legal_mean": bool(legal), "dd_gate": args.dd_gate,
            "mean_return": s["mean_return"], "mean_maxdd": s["mean_maxdd"],
            "worst_maxdd": s["worst_maxdd"], "timesteps": s["timesteps"], "git": s["git"],
            "reproduce": (f"python scripts/train_rl.py --action-mode weights --reward-mode {champ} "
                          f"--rich-obs --eval-split val --timesteps {s['timesteps']} --n-envs 6 "
                          f"--seed <0|1|2>   (defaults: ent_coef 0.2, lr 3e-4, step_bars 24)"),
        }
    with open(os.path.join(OUT_DIR, "champion.json"), "w", encoding="utf-8") as f:
        json.dump({"champion": champion, "modes": summary, "dd_gate": args.dd_gate}, f, indent=2)

    print(f"ledger: {len(rows)} runs -> {OUT_DIR}/ledger.jsonl\n")
    print(f"{'mode':10}{'n':>3}{'mean ret':>10}{'mean DD':>9}{'worst DD':>9}{'Sharpe':>8}{'legal?':>8}")
    for mode, v in sorted(summary.items(), key=lambda x: -x[1]["mean_return"]):
        print(f"{mode:10}{v['n']:>3}{v['mean_return']*100:>+9.1f}%{v['mean_maxdd']*100:>8.1f}%"
              f"{(v['worst_maxdd'] or 0)*100:>8.1f}%{v['mean_sharpe']:>8.2f}"
              f"{('YES' if v['legal_mean'] else 'NO'):>8}")
    if champion:
        print(f"\nCHAMPION (best mean return under {args.dd_gate:.0%} DD gate): "
              f"{champ}  +{champion['mean_return']*100:.1f}% @ {champion['mean_maxdd']*100:.1f}% "
              f"mean DD (worst seed {champion['worst_maxdd']*100:.1f}%)")


if __name__ == "__main__":
    main()
