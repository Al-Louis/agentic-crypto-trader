"""End-to-end pipeline proof: dispatch the demo run through `remote_train`, then publish.

By default this dispatches to the **training desktop** over SSH (`act-trainer`); the job
runs there, its bundle is rsynced back, and this (laptop) process publishes it. Pass
`--local` to run the whole loop on this machine instead (the original local proof).

  submit → job writes bundle + progress.json (on the desktop) → rsync back → publish.

Run:  python scripts/dispatch_demo.py            # → desktop over SSH
      python scripts/dispatch_demo.py --local    # → this machine
      python scripts/dispatch_demo.py --target s3://<bucket>/apentic   # publish to R2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from remote_train import JobSpec, LocalExecutor, SSHExecutor, publish, submit  # noqa: E402
from trader.report import upsert_manifest  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = REPO.parent / "alexlouis-site" / "public" / "apentic" / "data"

# Training desktop (CPU-parallel host, no keys). Reachable via Tailscale; key-based SSH.
REMOTE_HOST = "root@act-trainer"
REMOTE_WORKDIR = "/root/agentic-crypto-trader"
REMOTE_PYTHON = "/root/agentic-crypto-trader/.venv/bin/python"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--token", default="HUMA")
    p.add_argument("--ema", type=int, default=168)   # ~1 week on hourly bars → low churn
    p.add_argument("--target", default=str(DEFAULT_TARGET),
                   help="dashboard data dir, or an s3://bucket/prefix (R2)")
    p.add_argument("--store", default=str(REPO / "runs"))
    p.add_argument("--local", action="store_true", help="run on this machine, not the desktop")
    p.add_argument("--host", default=REMOTE_HOST, help="SSH target for the training desktop")
    p.add_argument("--remote-workdir", default=REMOTE_WORKDIR, help="repo path on the desktop")
    p.add_argument("--remote-python", default=REMOTE_PYTHON, help="python on the desktop")
    args = p.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    run_id = f"{args.token.lower()}-trend-ema{args.ema}"
    if args.local:
        executor, py, workdir, label = LocalExecutor(), sys.executable, str(REPO), "LocalExecutor"
    else:
        executor = SSHExecutor(host=args.host, remote_workdir=args.remote_workdir)
        py, workdir, label = args.remote_python, args.remote_workdir, f"SSHExecutor({args.host})"

    spec = JobSpec(
        name="apentic-demo",
        entrypoint=[py, "scripts/export_demo_run.py", "--out", "{artifact_dir}",
                    "--token", args.token, "--ema", str(args.ema), "--run-id", run_id],
        workdir=workdir,
    )

    print(f"[dispatch] submitting {spec.name} ({label})…")
    st = submit(spec, executor=executor, store=args.store)
    print(f"[dispatch] {st.run_id}: {st.state} (rc={st.returncode})")
    if not st.ok:
        print(f"[dispatch] job failed — see {st.log_path}")
        Path(st.log_path).exists() and print(Path(st.log_path).read_text(encoding='utf-8')[-1500:])
        raise SystemExit(1)
    if st.progress:
        print(f"[dispatch] progress: {json.dumps(st.progress, indent=2)}")

    artifact_dir = Path(st.artifact_dir)
    bundle = artifact_dir / run_id
    is_s3 = args.target.startswith(("s3://", "r2://"))

    # Publish the run's bundle, then merge the manifest entry at the destination.
    target_run = args.target.rstrip("/") + f"/{run_id}" if is_s3 else str(Path(args.target) / run_id)
    uri = publish(bundle, target_run)
    print(f"[dispatch] published bundle → {uri}")

    entry = next(e for e in json.loads((artifact_dir / "manifest.json").read_text("utf-8"))
                 if e["id"] == run_id)
    if is_s3:
        print("[dispatch] NOTE: merge this entry into the dashboard manifest.json on R2 "
              f"(read-modify-write) when wiring the bucket:\n  {json.dumps(entry)}")
    else:
        items = upsert_manifest(Path(args.target) / "manifest.json", entry)
        print(f"[dispatch] manifest upserted → {Path(args.target) / 'manifest.json'} "
              f"({len(items)} run(s))")
        print(f"\n  View: Apentic → /apentic/training  (run '{entry['model_name']}')")


if __name__ == "__main__":
    main()
