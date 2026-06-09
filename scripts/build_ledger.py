"""Build the committed experiment ledger from the published bundles.

The rigid, reproducible record of every training iteration (exact config -> performance), so we
can always find and RETURN TO the best known formula when a tweak degrades things. This is the
TradeSim lesson made structural: never tweak without a permanent, version-controlled performance
trail.

    python scripts/build_ledger.py            # rebuild experiments/ledger.jsonl + champion.json

Source of truth = the immutable published metrics.json per run (which carries a `provenance`
block: git commit + every hyperparameter). Runs are grouped by **config label** = the run_id with
its `-s<seed>` suffix stripped (so `ppo2-real-turn1-s{0,1,2}` form one config). Champion = highest
MEAN return among configs whose MEAN max-drawdown clears the ~30% DQ gate; we also report the worst
seed's drawdown so the margin to the gate is visible.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from collections import defaultdict

HOST = "https://data.alexlouis.dev"
SEEDED = re.compile(r"^(.*)-s(\d+)$")   # any seeded run: <config-label>-s<seed>
OUT_DIR = "experiments"


def fetch(u):
    with urllib.request.urlopen(u, timeout=30) as r:
        return json.load(r)


def reproduce_cmd(prov, label):
    """Exact command to re-run a config, reconstructed from its provenance block."""
    if not isinstance(prov, dict) or not prov:
        return f"(pre-provenance) re-run with run-id pattern {label}-s<seed>"
    rich = "--rich-obs " if prov.get("rich_obs") else ""
    return (f"python scripts/train_rl.py --action-mode {prov.get('action_mode', 'weights')} "
            f"--reward-mode {prov.get('reward_mode')} {rich}"
            f"--eval-split {prov.get('eval_split', 'val')} --timesteps {prov.get('timesteps')} "
            f"--n-envs {prov.get('n_envs', 6)} --gb-lambda {prov.get('gb_lambda')} "
            f"--turn-lambda {prov.get('turn_lambda')} --realized-lambda {prov.get('realized_lambda')} "
            f"--dd-lambda {prov.get('dd_lambda', 2.0)} --seed <0|1|2>")


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
        sm = SEEDED.match(rid)
        steps = prov.get("timesteps")
        if steps is None:
            mt = re.search(r"([\d,]+)\s*steps", e.get("model_name", ""))
            steps = int(mt.group(1).replace(",", "")) if mt else None
        rows.append({
            "run_id": rid,
            "config_label": sm.group(1) if sm else rid,
            "mode": prov.get("reward_mode") or m.get("reward_mode", "?"),
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

    rows.sort(key=lambda r: (r["config_label"], r["seed"] if r["seed"] is not None else -1))
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "ledger.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # per-config summary over seeded runs (>=2 seeds), grouped by config label
    by_cfg = defaultdict(list)
    for r in rows:
        if SEEDED.match(r["run_id"]) and r["return"] is not None:
            by_cfg[r["config_label"]].append(r)
    summary = {}
    for label, rs in by_cfg.items():
        if len(rs) < 2:                                  # need a couple seeds to judge a config
            continue
        n = len(rs)
        mean = lambda k: sum(x[k] for x in rs if x[k] is not None) / max(n, 1)
        summary[label] = {
            "n": n, "seeds": sorted(x["seed"] for x in rs),
            "mean_return": mean("return"), "mean_maxdd": mean("maxdd"),
            "mean_sharpe": mean("sharpe"), "mean_pf": mean("pf"),
            "worst_maxdd": max((x["maxdd"] for x in rs if x["maxdd"] is not None), default=None),
            "timesteps": rs[0].get("timesteps"), "git": rs[0].get("git"),
            "reproduce": reproduce_cmd(rs[0].get("config"), label),
            "legal_mean": mean("maxdd") < args.dd_gate,
        }

    legal = {k: v for k, v in summary.items() if v["legal_mean"]}
    champ = max((legal or summary), key=lambda k: summary[k]["mean_return"]) if summary else None
    champion = dict(summary[champ], config_label=champ) if champ else None
    with open(os.path.join(OUT_DIR, "champion.json"), "w", encoding="utf-8") as f:
        json.dump({"champion": champion, "configs": summary, "dd_gate": args.dd_gate}, f, indent=2)

    print(f"ledger: {len(rows)} runs -> {OUT_DIR}/ledger.jsonl\n")
    print(f"{'config':22}{'n':>3}{'mean ret':>10}{'mean DD':>9}{'worst DD':>9}{'Sharpe':>8}{'legal?':>8}")
    for label, v in sorted(summary.items(), key=lambda x: -x[1]["mean_return"]):
        print(f"{label:22}{v['n']:>3}{v['mean_return']*100:>+9.1f}%{v['mean_maxdd']*100:>8.1f}%"
              f"{(v['worst_maxdd'] or 0)*100:>8.1f}%{v['mean_sharpe']:>8.2f}"
              f"{('YES' if v['legal_mean'] else 'NO'):>8}")
    if champion:
        print(f"\nCHAMPION (best mean return under {args.dd_gate:.0%} DD gate): "
              f"{champ}  +{champion['mean_return']*100:.1f}% @ {champion['mean_maxdd']*100:.1f}% "
              f"mean DD (worst seed {(champion['worst_maxdd'] or 0)*100:.1f}%)")


if __name__ == "__main__":
    main()
