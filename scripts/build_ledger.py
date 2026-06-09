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
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

HOST = "https://data.alexlouis.dev"
SEEDED = re.compile(r"^(.*)-s(\d+)$")   # any seeded run: <config-label>-s<seed>
OUT_DIR = "experiments"
# public infra IDs (not secrets — protected by IAM creds, not obscurity); override via CLI / env
DEFAULT_TARGET = "s3://alexlouis-apentic-data"
DEFAULT_CF_DIST = "E14F268NIY6WLZ"


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
    p.add_argument("--publish", action="store_true",
                   help="publish leaderboard.json to the Apentic data host for the frontend overview")
    p.add_argument("--publish-target", default=None, help=f"default: env or {DEFAULT_TARGET}")
    p.add_argument("--cloudfront-dist", default=None, help=f"default: env or {DEFAULT_CF_DIST}")
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
            "split": prov.get("eval_split", "val"),       # val = tuning; test = frozen OOS verdict
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
            "split": rs[0].get("split", "val"),
            "baseline": next((x["baseline"] for x in rs if x["baseline"] is not None), None),
            "reproduce": reproduce_cmd(rs[0].get("config"), label),
            "legal_mean": mean("maxdd") < args.dd_gate,
        }

    # Champion must have PASSED the frozen test: split=test, beats its test baseline, AND worst-seed
    # drawdown under the gate. None ⇒ nothing has generalized out-of-sample yet (the honest state).
    def _passed_oos(v):
        return (v.get("split") == "test" and v["worst_maxdd"] is not None
                and v["worst_maxdd"] < args.dd_gate
                and v["baseline"] is not None and v["mean_return"] > v["baseline"])

    oos_ok = {k: v for k, v in summary.items() if _passed_oos(v)}
    champ = max(oos_ok, key=lambda k: summary[k]["mean_return"]) if oos_ok else None
    champion = dict(summary[champ], config_label=champ) if champ else None
    with open(os.path.join(OUT_DIR, "champion.json"), "w", encoding="utf-8") as f:
        json.dump({"champion": champion, "configs": summary, "dd_gate": args.dd_gate}, f, indent=2)

    # ---- leaderboard.json: a self-contained training-progress overview for the frontend ----
    baseline_ret = next((r["baseline"] for r in rows if r["baseline"] is not None), None)

    def cfg_card(label, v):
        runs_d = sorted(by_cfg[label], key=lambda r: r["seed"] if r["seed"] is not None else -1)
        cfg_base = next((r["baseline"] for r in runs_d if r["baseline"] is not None), None)
        return {
            "config_label": label, "timesteps": v["timesteps"], "n": v["n"], "seeds": v["seeds"],
            "split": runs_d[0]["split"] if runs_d else "val",   # val=tuning, test=frozen OOS verdict
            "baseline": cfg_base,                               # this config's own split baseline
            "mean_return": v["mean_return"], "mean_maxdd": v["mean_maxdd"],
            "worst_maxdd": v["worst_maxdd"], "mean_sharpe": v["mean_sharpe"], "mean_pf": v["mean_pf"],
            "legal_mean": v["legal_mean"],
            "gate_safe_worst": v["worst_maxdd"] is not None and v["worst_maxdd"] < args.dd_gate,
            "beats_baseline": cfg_base is not None and v["mean_return"] > cfg_base,
            "git": v["git"], "reproduce": v["reproduce"],
            "seeds_detail": [{"seed": r["seed"], "return": r["return"], "maxdd": r["maxdd"],
                              "sharpe": r["sharpe"], "pf": r["pf"], "run_id": r["run_id"]}
                             for r in runs_d],
        }

    leaderboard = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "dd_gate": args.dd_gate,
        "totals": {"runs": len(rows), "configs": len(summary)},
        "baseline": {"name": "vol-tilt(trend50)", "return_pct": baseline_ret, "window": "val"},
        "champion": champion,
        "champion_criterion": ("PASSED frozen-test OOS: split=test, beats its test baseline, AND "
                               "worst-seed maxDD under the gate. null = nothing generalized yet."),
        "configs": [cfg_card(label, v) for label, v in
                    sorted(summary.items(), key=lambda x: -x[1]["mean_return"])],
    }
    with open(os.path.join(OUT_DIR, "leaderboard.json"), "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2)

    print(f"ledger: {len(rows)} runs -> {OUT_DIR}/ledger.jsonl\n")
    print(f"{'config':20}{'split':>6}{'n':>3}{'mean ret':>10}{'mean DD':>9}{'worst DD':>9}{'vs base':>9}")
    for label, v in sorted(summary.items(), key=lambda x: (x[1].get("split", "val"), -x[1]["mean_return"])):
        base = v.get("baseline")
        vs = f"{(v['mean_return'] - base) * 100:+.0f}pt" if base is not None else "?"
        print(f"{label:20}{v.get('split', 'val'):>6}{v['n']:>3}{v['mean_return']*100:>+9.1f}%"
              f"{v['mean_maxdd']*100:>8.1f}%{(v['worst_maxdd'] or 0)*100:>8.1f}%{vs:>9}")
    if champion:
        print(f"\nCHAMPION (passed frozen-test OOS): {champ}  +{champion['mean_return']*100:.1f}% "
              f"@ worst-seed {(champion['worst_maxdd'] or 0)*100:.1f}% DD")
    else:
        print("\nCHAMPION: none — no config has passed frozen-test OOS "
              "(beat its test baseline + worst-seed under the gate)")

    print(f"\nleaderboard -> {OUT_DIR}/leaderboard.json ({len(leaderboard['configs'])} configs)")
    if args.publish:
        import importlib  # noqa: PLC0415
        pub = importlib.import_module("remote_train.publish")  # the submodule, not the re-exported fn
        from trader import config  # noqa: PLC0415
        config.load_dotenv()
        target = args.publish_target or config.get("APENTIC_PUBLISH_TARGET") or DEFAULT_TARGET
        dist = args.cloudfront_dist or config.get("APENTIC_CLOUDFRONT_DIST_ID") or DEFAULT_CF_DIST
        data = json.dumps(leaderboard, indent=2).encode()
        pub.put_bytes(f"{target}/leaderboard.json", data, "application/json", "no-cache, max-age=0")
        inv = pub.invalidate_cloudfront(dist, ["/leaderboard.json"]) if dist else None
        print(f"published leaderboard.json -> {target}"
              + (f" (+ CloudFront invalidation {inv})" if inv else ""))


if __name__ == "__main__":
    main()
