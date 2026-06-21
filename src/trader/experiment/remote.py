"""Central remote-training config + the one disciplined SSH helper the RL tools share.

Every laptop->desktop call routes through here so the runbook scars (vault "Remote
Capabilities") are enforced in ONE place instead of being re-learned the hard way:

  1. The **Windows OpenSSH** binary is pinned. `where ssh` finds Git's MSYS ssh FIRST, and
     that one cannot route the tailnet — it hangs forever. We pick `System32\\OpenSSH\\ssh.exe`
     explicitly on Windows (PATH ssh elsewhere, e.g. the Linux box / CI).
  2. Replies are kept **tiny** (the path-MTU black hole: a reply >~2 KB stalls and kills the
     session). `run_ssh` hard-guards the response size — status one-liners must return COUNTS,
     never full `ps`/`pgrep` output.
  3. Connects **fail fast** and a stalled established session is dropped in ~15 s (keepalives),
     mirroring `remote_train.executor.SSHExecutor.BASE_OPTS`.

The training desktop holds **no keys and never touches mainnet** — pure compute on recorded
data. Host/workdir/python/CDN live here (previously duplicated in scripts/train_loop.py).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

# The keyless training desktop (act-trainer; vault "Remote Capabilities"). Set TRAINER_SSH_HOST in
# your local .env (gitignored) to your own host; the placeholder default is intentionally non-routable.
HOST = os.environ.get("TRAINER_SSH_HOST", "root@<TRAINER_TAILNET_IP>")
REMOTE_WORKDIR = "/root/agentic-crypto-trader"
REMOTE_PYTHON = "/root/agentic-crypto-trader/.venv/bin/python"
# Results are read over normal internet from the CDN, never hauled back over the tailnet.
DATA_CDN = "https://data.alexlouis.dev"

# Pinned Windows OpenSSH — the MSYS/Git ssh hangs on the tailnet (runbook rule #1).
WINDOWS_OPENSSH = r"C:\Windows\System32\OpenSSH\ssh.exe"

# Non-interactive, fail-fast, drop a stalled session in ~15 s (same posture as SSHExecutor).
SSH_OPTS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3")

# The path-MTU ceiling: a reply bigger than this stalls the tailnet, so we refuse to issue a
# command whose output exceeds it rather than wedge the session.
MAX_REPLY_BYTES = 2048


def ssh_binary() -> str:
    """The ssh executable to use: pinned Windows OpenSSH if present, else PATH ssh."""
    if Path(WINDOWS_OPENSSH).exists():
        return WINDOWS_OPENSSH
    return shutil.which("ssh") or "ssh"


def run_ssh(remote_cmd: str, *, host: str = HOST, timeout: float = 20.0) -> str:
    """Run ONE remote command over the pinned, disciplined SSH and return stdout (stripped).

    Raises on a connection/command failure (rc != 0 — design the one-liners to exit 0) or if
    the reply would exceed `MAX_REPLY_BYTES` (the MTU guard: keep replies tiny). `timeout` is a
    hard backstop in case keepalives don't fire.
    """
    proc = subprocess.run([ssh_binary(), *SSH_OPTS, host, remote_cmd],  # noqa: S603 — fixed host
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, timeout=timeout)
    if len(proc.stdout.encode("utf-8", "replace")) > MAX_REPLY_BYTES:
        raise RuntimeError(
            f"ssh reply too large ({len(proc.stdout)} chars) — keep replies tiny (MTU black hole)")
    if proc.returncode != 0:
        raise RuntimeError(f"ssh failed (rc={proc.returncode}): {proc.stderr.strip()[:200]}")
    return proc.stdout.strip()


# One-liner: process counts + load, kept to a single tiny line (e.g. "drv=1 tr=1 load=6.22").
# `pgrep -fc PAT` prints the count even when 0 (it just exits 1 then — `|| true` swallows that
# without adding output). Counts only: never list PIDs/argv here (MTU).
# The `[r]un_…` / `[t]rain_…` bracket trick is load-bearing: with `-f`, pgrep matches the FULL
# cmdline of every process — including this very status command's wrapper shell, which contains
# the literal patterns. A bracketed class matches the real processes but NOT the pattern string
# itself, so the matcher can't count itself (otherwise an idle box reports phantom runners).
_STATUS_ONELINER = (
    "printf 'drv=%s tr=%s load=%s\\n' "
    "\"$(pgrep -fc '[r]un_eventrung_sweep' || true)\" "
    "\"$(pgrep -fc '[t]rain_event.py' || true)\" "
    "\"$(cut -d' ' -f1 /proc/loadavg)\""
)


def parse_status(line: str) -> dict:
    """Parse the `drv=N tr=N load=F` status line into a dict (pure, testable)."""
    out: dict[str, float] = {}
    for tok in line.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                out[k] = float(v)
            except ValueError:
                continue
    drv, tr = int(out.get("drv", 0)), int(out.get("tr", 0))
    return {"driver": drv, "trainers": tr, "load": out.get("load"),
            "running": (drv > 0 or tr > 0)}


def sweep_status(*, host: str = HOST) -> dict:
    """Liveness of any running event-rung sweep: driver/trainer counts + 1-min load (one tiny ssh)."""
    return parse_status(run_ssh(_STATUS_ONELINER, host=host))


def ssh_executor(host: str = HOST, workdir: str = REMOTE_WORKDIR):
    """A `remote_train.SSHExecutor` wired to the desktop with the pinned ssh binary.

    The launch tier (rl_train/rl_kill, vault "MCP Server") rides this; centralizing the pin here
    means those tools can't accidentally grab the MSYS ssh.
    """
    from remote_train import SSHExecutor
    return SSHExecutor(host=host, remote_workdir=workdir, ssh=ssh_binary())
