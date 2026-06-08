"""Dispatch the demo run through `remote_train`; the job self-publishes its bundle.

By default this dispatches to the **training desktop** over SSH (`act-trainer`): the job runs
there, exports the bundle, and uploads it **straight to R2 over the desktop's own internet** —
nothing large traverses the tailnet back to the laptop (which is why this avoids the path-MTU
problem). The laptop only sends the tiny trigger and reads tiny status/progress.

`--local` runs the whole loop here and publishes to the local dashboard dir instead.

Run:  python scripts/dispatch_demo.py                 # → desktop over SSH, publishes to R2
      python scripts/dispatch_demo.py --local         # → this machine, publishes locally
      python scripts/dispatch_demo.py --target s3://apentic   # override the publish target
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from remote_train import JobSpec, LocalExecutor, SSHExecutor, submit  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_TARGET = REPO.parent / "alexlouis-site" / "public" / "apentic" / "data"

# Training desktop (CPU-parallel host, no keys). Reachable via Tailscale; key-based SSH.
# Use the Tailscale IP, not the MagicDNS name `act-trainer`: the name didn't resolve inside
# the ssh *subprocess* (works interactively). The 100.x tailnet IP is stable per device.
REMOTE_HOST = "root@100.97.195.65"   # act-trainer (Tailscale IP)
REMOTE_WORKDIR = "/root/agentic-crypto-trader"
REMOTE_PYTHON = "/root/agentic-crypto-trader/.venv/bin/python"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--token", default="HUMA")
    p.add_argument("--ema", type=int, default=168)   # ~1 week on hourly bars → low churn
    p.add_argument("--target", default=None,
                   help="publish target (local dir or s3://…); remote default = the desktop's "
                        "APENTIC_PUBLISH_TARGET from its .env; local default = the dashboard dir")
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
        publish_target = args.target or str(DEFAULT_LOCAL_TARGET)
    else:
        executor = SSHExecutor(host=args.host, remote_workdir=args.remote_workdir)
        py, workdir, label = args.remote_python, args.remote_workdir, f"SSHExecutor({args.host})"
        publish_target = args.target            # None → desktop job reads APENTIC_PUBLISH_TARGET

    entrypoint = [py, "scripts/export_demo_run.py", "--out", "{artifact_dir}",
                  "--token", args.token, "--ema", str(args.ema), "--run-id", run_id]
    if publish_target:
        entrypoint += ["--publish-target", publish_target]

    # The job publishes its own output → no artifact haul-back across the link.
    spec = JobSpec(name="apentic-demo", entrypoint=entrypoint, workdir=workdir,
                   fetch_artifacts=False)

    print(f"[dispatch] submitting {spec.name} ({label})…")
    st = submit(spec, executor=executor, store=args.store)
    print(f"[dispatch] {st.run_id}: {st.state} (rc={st.returncode})")

    log = (Path(st.log_path).read_text(encoding="utf-8")
           if st.log_path and Path(st.log_path).exists() else "")
    if not st.ok:
        print(f"[dispatch] job failed — see {st.log_path}\n{log[-1500:]}")
        raise SystemExit(1)
    if st.progress:
        print(f"[dispatch] progress: {json.dumps(st.progress, indent=2)}")
    for line in log.splitlines():
        if "published" in line or "[export_demo_run]" in line:
            print(line)
    print("\n  Done — the job published its own bundle. View: Apentic → /apentic/training")


if __name__ == "__main__":
    main()
