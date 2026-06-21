"""Run one iteration of the training loop against the demo, and record it.

Dispatches a demo config (token/ema/band), fetches the published results, diagnoses them
against the honest gates, and records the experiment + lineage. This is the Level-B loop's
mechanical half — a driver (a Claude session now, a scheduled workflow later) calls this,
reads the diagnosis, and decides the next config.

  python scripts/train_loop.py --token ZEC --ema 168 --band 0.04          # → desktop → data.alexlouis.dev
  python scripts/train_loop.py --local --token HUMA --ema 120             # → laptop → local dashboard dir
  python scripts/train_loop.py --token ZEC --parent exp-001               # descend from a prior experiment
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from remote_train import LocalExecutor, SSHExecutor  # noqa: E402
from trader.train import demo_config, run_iteration  # noqa: E402
from trader.train.registry import Registry  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LOCAL_DASHBOARD = REPO.parent / "alexlouis-site" / "public" / "apentic" / "data"
DATA_CDN = "https://data.alexlouis.dev"

# Set TRAINER_SSH_HOST in your local .env (gitignored); the placeholder default is non-routable.
REMOTE_HOST = os.environ.get("TRAINER_SSH_HOST", "root@<TRAINER_TAILNET_IP>")
REMOTE_WORKDIR = "/root/agentic-crypto-trader"
REMOTE_PYTHON = "/root/agentic-crypto-trader/.venv/bin/python"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--token", default="HUMA")
    p.add_argument("--ema", type=int, default=168)
    p.add_argument("--band", type=float, default=0.04)
    p.add_argument("--parent", default=None, help="parent experiment id (lineage)")
    p.add_argument("--local", action="store_true", help="run on this machine, not the desktop")
    p.add_argument("--experiments", default=str(REPO / "experiments"))
    p.add_argument("--store", default=str(REPO / "runs"))
    p.add_argument("--host", default=REMOTE_HOST)
    p.add_argument("--remote-workdir", default=REMOTE_WORKDIR)
    p.add_argument("--remote-python", default=REMOTE_PYTHON)
    args = p.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    if args.local:
        executor, python, workdir = LocalExecutor(), sys.executable, str(REPO)
        publish_target, results_base = str(LOCAL_DASHBOARD), str(LOCAL_DASHBOARD)
    else:
        executor = SSHExecutor(host=args.host, remote_workdir=args.remote_workdir)
        python, workdir = args.remote_python, args.remote_workdir
        publish_target, results_base = None, DATA_CDN     # desktop .env target; read from the CDN

    config = demo_config(args.token, args.ema, args.band)
    registry = Registry(args.experiments)

    print(f"[loop] dispatching {config} ({'local' if args.local else 'desktop'})…")
    exp, status = run_iteration(
        config, registry, executor=executor, store=args.store, results_base=results_base,
        python=python, workdir=workdir, publish_target=publish_target, parent_id=args.parent,
        created=datetime.now(timezone.utc).isoformat())

    print(f"[loop] {exp.id} run={exp.run_id} dispatch={status.state}")
    if not status.ok:
        print(f"[loop] dispatch failed — see {status.log_path}")
        raise SystemExit(1)
    dx = exp.diagnosis or {}
    print(f"[loop] VERDICT: {dx.get('verdict', '?').upper()}  failed={dx.get('failed', [])}")
    for g in dx.get("gates", []):
        mark = "PASS" if g["passed"] else "FAIL"
        print(f"    [{mark}] {g['name']:16} value={g['value']:+.4g} vs {g['threshold']:+.4g}  ({g['note']})")
    print(f"\n[loop] recorded → {Path(args.experiments) / (exp.id + '.json')}")
    if exp.metrics:
        print(f"[loop] metrics: return {exp.metrics.get('total_return_pct'):+.1%}, "
              f"Sharpe {exp.metrics.get('sharpe_ratio'):.2f}, "
              f"maxDD {exp.metrics.get('max_drawdown_pct'):.1%}")


if __name__ == "__main__":
    main()
