"""The committed experiment ledger + champion — importable core behind experiment_champion.

The rigid, version-controlled record of every training iteration (exact config -> performance)
so we can always RETURN TO the best known formula when a tweak degrades things (the TradeSim
lesson made structural). Source of truth = the immutable published `metrics.json` per run (which
carries a `provenance` block: git commit + every hyperparameter).

Runs group by **config label** = the run_id with its `-s<seed>` suffix stripped. The **champion**
is the highest mean-return config that has PASSED the **honest gate** on the frozen test (vault
"Agent Communication Contract"; DIRECTION RESET 2026-06-15): `split == test`, worst-seed maxDD under
the DQ gate, AND the seed-mean beats the rung-0 RULE baseline (if present). Buy&Hold and Random are
COMPUTED and REPORTED but are NEVER binding — requiring "beat Buy&Hold" rewards holding-everything.
`None` ⇒ nothing has generalized out-of-sample yet (the honest state). `read_champion` reads the
committed artifact instantly (no network); `rebuild_ledger` re-derives everything from the CDN
(`fetch` injected).

scripts/build_ledger.py is now a thin CLI over these functions.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

DEFAULT_HOST = "https://data.alexlouis.dev"
DEFAULT_OUT = Path("experiments")
DD_GATE = 0.30
SEEDED = re.compile(r"^(.*)-s(\d+)$")   # any seeded run: <config-label>-s<seed>
# the sha naming convention (ec1e487): run-ids carry the git short-hash so a re-run on different
# code can never overwrite/alias an old name. Everything published BEFORE it is the invalid era
# (broken env / ambiguous vintage) — `sha_only` keeps the board to runs that carry the stamp.
SHA_NAMED = re.compile(r"-[0-9a-f]{7}(-test)?-s\d+$")

Fetch = Callable[[str], Any]


def _http_fetch(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=30) as r:  # noqa: S310 — fixed CDN host
        return json.load(r)


def reproduce_cmd(prov: Any, label: str) -> str:
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


def _row(rid: str, m: dict, model_name: str = "") -> dict:
    prov = m.get("provenance", {})
    sm = SEEDED.match(rid)
    steps = prov.get("timesteps")
    if steps is None:
        mt = re.search(r"([\d,]+)\s*steps", model_name)
        steps = int(mt.group(1).replace(",", "")) if mt else None
    ret, base = m.get("total_return_pct"), m.get("baseline_return")
    dd = m.get("max_drawdown_pct")
    return {
        "run_id": rid, "config_label": sm.group(1) if sm else rid,
        "mode": prov.get("reward_mode") or m.get("reward_mode", "?"),
        "split": prov.get("eval_split", "val"),     # val = tuning; test = frozen OOS verdict
        "seed": prov.get("seed", int(sm.group(2)) if sm else None),
        "timesteps": steps, "git": prov.get("git_commit"),
        "return": ret, "sharpe": m.get("sharpe_ratio"), "maxdd": dd,
        "pf": m.get("profit_factor"), "win": m.get("win_rate"), "trades": m.get("total_trades"),
        "turnover_usd": m.get("eval_turnover_usd"), "realized_usd": m.get("eval_realized_usd"),
        "giveback": m.get("eval_giveback"), "baseline": base,
        "buyhold": m.get("buyhold_return"), "random": m.get("random_return"),
        "beats_baseline": (ret or 0) > (base or 0),
        "legal_dd": (dd is not None and dd < DD_GATE),
        "config": prov or "(pre-provenance: see model_name / run_id)",
    }


def _summarize(rows: list[dict], dd_gate: float) -> tuple[dict, dict]:
    """Per-config summary over seeded runs (>=2 seeds) + the grouping. Returns (summary, by_cfg)."""
    by_cfg: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if SEEDED.match(r["run_id"]) and r["return"] is not None:
            by_cfg[r["config_label"]].append(r)
    summary = {}
    for label, rs in by_cfg.items():
        if len(rs) < 2:                              # need a couple seeds to judge a config
            continue
        n = len(rs)
        def mean(k, rs=rs, n=n):
            return sum(x[k] for x in rs if x[k] is not None) / max(n, 1)
        # Buy&Hold is a fixed per-window baseline (same for every seed); Random discretion varies
        # per seed, so it is meaned. Both feed the honest gate.
        bh = next((x["buyhold"] for x in rs if x["buyhold"] is not None), None)
        rnd_vals = [x["random"] for x in rs if x["random"] is not None]
        summary[label] = {
            "n": n, "seeds": sorted(x["seed"] for x in rs),
            "mean_return": mean("return"), "mean_maxdd": mean("maxdd"),
            "mean_sharpe": mean("sharpe"), "mean_pf": mean("pf"),
            "worst_maxdd": max((x["maxdd"] for x in rs if x["maxdd"] is not None), default=None),
            "timesteps": rs[0].get("timesteps"), "git": rs[0].get("git"),
            "split": rs[0].get("split", "val"),
            "baseline": next((x["baseline"] for x in rs if x["baseline"] is not None), None),
            "buyhold": bh, "random": (sum(rnd_vals) / len(rnd_vals)) if rnd_vals else None,
            "reproduce": reproduce_cmd(rs[0].get("config"), label),
            "legal_mean": mean("maxdd") < dd_gate,
        }
    return summary, by_cfg


def _honest_gate(v: dict, dd_gate: float) -> tuple[bool, str | None]:
    """The honest gate on a config's seed-mean (vault "Agent Communication Contract"; DIRECTION
    RESET 2026-06-15).

    Pass = frozen test + worst-seed DD under the gate + seed-mean beats the rung-0 RULE baseline
    (if present). Buy&Hold and Random are computed + reported but NEVER binding — requiring "beat
    Buy&Hold" rewards holding-everything (the rejected basket overlay); the rung-0 RULE is the real
    bar. Returns (passed, binding) where `binding` is the first failing check ("not-test" /
    "drawdown" / "rung-0").

    Caveat (pre-existing): unlike weekly_gate / train_event.honest_gate, this cannot exempt a DQ'd
    rung-0 — the ledger carries `baseline_return` but not rung-0's maxDD — so a rung-0 that itself
    breaches the DQ on the test window is still treated as a return bar here. In practice rung-0
    rarely DQs on the frozen test; plumb baseline_maxdd through `_row`/`_summarize` for full parity.
    """
    if v.get("split") != "test":
        return False, "not-test"
    if v.get("worst_maxdd") is None or v["worst_maxdd"] >= dd_gate:
        return False, "drawdown"
    base = v.get("baseline")
    if base is not None and not v["mean_return"] > base:
        return False, "rung-0"
    return True, None


def _passed_oos(v: dict, dd_gate: float) -> bool:
    return _honest_gate(v, dd_gate)[0]


def pick_champion(summary: dict, dd_gate: float = DD_GATE) -> dict | None:
    """The highest mean-return config that passed the frozen-test OOS gate (or None)."""
    oos_ok = {k: v for k, v in summary.items() if _passed_oos(v, dd_gate)}
    if not oos_ok:
        return None
    champ = max(oos_ok, key=lambda k: summary[k]["mean_return"])
    return dict(summary[champ], config_label=champ)


def rebuild_ledger(*, host: str = DEFAULT_HOST, dd_gate: float = DD_GATE,
                   fetch: Fetch = _http_fetch, generated: str | None = None,
                   sha_only: bool = False) -> dict:
    """Re-derive ledger rows + per-config summary + champion + leaderboard from published bundles.

    Network reads are injected via `fetch` (tests pass fixtures). `generated` is the leaderboard
    timestamp, injected to keep this pure (callers stamp `datetime.now(...)`). `sha_only` keeps
    only sha-stamped run-ids (the post-ec1e487 valid era).
    """
    man = fetch(f"{host.rstrip('/')}/manifest.json")
    rows: list[dict] = []
    for e in man:
        if e.get("kind") != "portfolio":
            continue
        rid = e["id"]
        if sha_only and not SHA_NAMED.search(rid):
            continue
        try:
            m = fetch(f"{host.rstrip('/')}/{rid}/metrics.json")
        except Exception:  # noqa: BLE001
            continue
        rows.append(_row(rid, m, e.get("model_name", "")))
    rows.sort(key=lambda r: (r["config_label"], r["seed"] if r["seed"] is not None else -1))

    summary, by_cfg = _summarize(rows, dd_gate)
    champion = pick_champion(summary, dd_gate)
    baseline_ret = next((r["baseline"] for r in rows if r["baseline"] is not None), None)

    def cfg_card(label, v):
        runs_d = sorted(by_cfg[label], key=lambda r: r["seed"] if r["seed"] is not None else -1)
        cfg_base = next((r["baseline"] for r in runs_d if r["baseline"] is not None), None)
        passed, binding = _honest_gate(v, dd_gate)
        return {
            "config_label": label, "timesteps": v["timesteps"], "n": v["n"], "seeds": v["seeds"],
            "split": runs_d[0]["split"] if runs_d else "val",
            "baseline": cfg_base, "buyhold": v.get("buyhold"), "random": v.get("random"),
            "mean_return": v["mean_return"], "mean_maxdd": v["mean_maxdd"],
            "worst_maxdd": v["worst_maxdd"], "mean_sharpe": v["mean_sharpe"], "mean_pf": v["mean_pf"],
            "legal_mean": v["legal_mean"],
            "gate_safe_worst": v["worst_maxdd"] is not None and v["worst_maxdd"] < dd_gate,
            "beats_baseline": cfg_base is not None and v["mean_return"] > cfg_base,
            "honest_gate_pass": passed, "honest_gate_binding": binding,   # beats the rung-0 RULE + survives DQ
            "git": v["git"], "reproduce": v["reproduce"],
            "seeds_detail": [{"seed": r["seed"], "return": r["return"], "maxdd": r["maxdd"],
                              "sharpe": r["sharpe"], "pf": r["pf"], "run_id": r["run_id"]}
                             for r in runs_d],
        }

    leaderboard = {
        "generated": generated,
        "dd_gate": dd_gate,
        "totals": {"runs": len(rows), "configs": len(summary)},
        "baseline": {"name": "vol-tilt(trend50)", "return_pct": baseline_ret, "window": "val"},
        "champion": champion,
        "champion_criterion": ("PASSED the honest gate on frozen test: split=test, worst-seed maxDD "
                               "under the gate, AND seed-mean beats the rung-0 RULE (DIRECTION RESET "
                               "2026-06-15; Buy&Hold/Random reported but never binding). null = "
                               "nothing generalized yet."),
        "configs": [cfg_card(label, v) for label, v in
                    sorted(summary.items(), key=lambda x: -x[1]["mean_return"])],
    }
    return {"rows": rows, "summary": summary, "champion": champion, "leaderboard": leaderboard,
            "dd_gate": dd_gate}


def write_ledger(result: dict, out_dir: Path | str = DEFAULT_OUT) -> None:
    """Write ledger.jsonl + champion.json + leaderboard.json from a `rebuild_ledger` result."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ledger.jsonl", "w", encoding="utf-8") as f:
        for r in result["rows"]:
            f.write(json.dumps(r) + "\n")
    (out / "champion.json").write_text(
        json.dumps({"champion": result["champion"], "configs": result["summary"],
                    "dd_gate": result["dd_gate"]}, indent=2), encoding="utf-8")
    (out / "leaderboard.json").write_text(
        json.dumps(result["leaderboard"], indent=2), encoding="utf-8")


def read_champion(out_dir: Path | str = DEFAULT_OUT) -> dict:
    """Read the committed champion.json (instant, no network) — the current best + repro command."""
    p = Path(out_dir) / "champion.json"
    if not p.exists():
        return {"champion": None, "configs": {}, "note": "no champion.json — run experiment_record"}
    return json.loads(p.read_text(encoding="utf-8"))
