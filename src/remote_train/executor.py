"""Executors — *where* a job's command runs. The orchestrator stays identical across them.

`LocalExecutor` runs on this machine (used now for the pipeline-first proof and CI).
`SSHExecutor` runs on a remote host over SSH (the desktop training box — chosen for CPU cores
+ RAM, since this RL workload is env-stepping-bound, not GPU-bound — reachable via Tailscale)
and syncs the remote artifact directory back. Both implement the same `run` contract, so
swapping local→remote is a one-line change at the call site.
"""

from __future__ import annotations

import base64
import io
import json
import os
import shlex
import subprocess
import tarfile
from pathlib import Path

from remote_train.progress import read_progress as _read_progress_file
from remote_train.spec import JobSpec


def _substitute(spec: JobSpec, run_dir: Path, artifact_dir: Path) -> tuple[list[str], dict[str, str]]:
    sub = {"run_dir": str(run_dir), "artifact_dir": str(artifact_dir)}
    argv = [a.format(**sub) for a in spec.entrypoint]
    env = {k: v.format(**sub) for k, v in spec.env.items()}
    return argv, env


class LocalExecutor:
    """Run the job as a local subprocess, streaming output to ``run.log``."""

    name = "local"

    def run(self, spec: JobSpec, run_dir: Path, artifact_dir: Path, log_path: Path) -> int:
        argv, extra_env = _substitute(spec, run_dir, artifact_dir)
        env = {**os.environ, **extra_env}
        with open(log_path, "w", encoding="utf-8", errors="replace") as log:
            proc = subprocess.run(  # noqa: S603 — argv list, not shell
                argv, cwd=spec.workdir, env=env,
                stdout=log, stderr=subprocess.STDOUT, text=True,
            )
        return proc.returncode

    # -- background (fire-and-poll) -----------------------------------------
    def launch(self, spec: JobSpec, run_dir: Path, artifact_dir: Path, log_path: Path) -> dict:
        """Start the job detached and return a handle (the job self-reports via progress.json)."""
        argv, extra_env = _substitute(spec, run_dir, artifact_dir)
        env = {**os.environ, **extra_env}
        flags = ({"start_new_session": True} if os.name == "posix"
                 else {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)})
        log = open(log_path, "w", encoding="utf-8", errors="replace")  # noqa: SIM115 - child owns it
        proc = subprocess.Popen(argv, cwd=spec.workdir, env=env,  # noqa: S603
                                stdout=log, stderr=subprocess.STDOUT, **flags)
        return {"executor": "local", "pid": proc.pid, "artifact_dir": str(artifact_dir)}

    def read_progress(self, handle: dict) -> dict | None:
        return _read_progress_file(Path(handle["artifact_dir"]))

    def is_alive(self, handle: dict) -> bool:
        pid = handle.get("pid")
        if not pid:
            return False
        if os.name != "posix":
            return True   # os.kill(pid,0) *terminates* on Windows — rely on progress.json state
        try:
            os.kill(int(pid), 0)
            return True
        except OSError:
            return False


class SSHExecutor:
    """Run the job on a remote host over SSH, then stream its artifacts back.

    Assumes key-based SSH to ``host`` (e.g. a Tailscale name) and that the repo already
    lives at ``remote_workdir`` on that host. The training box holds **no keys and never
    touches mainnet** (vault "Remote Capabilities") — it is pure compute on recorded data.

    Artifacts come back as a **tar stream over ssh** (remote ``tar`` → local `tarfile`),
    not rsync/scp: it needs only ``ssh`` locally (Windows OpenSSH has no rsync, and scp
    mis-parses ``C:\\`` drive-letter targets), and ``tar`` on the Linux host.
    """

    name = "ssh"
    # Always non-interactive: never hang on a password prompt (key-only), fail fast on a
    # dead route (ConnectTimeout), and drop a *stalled* established session within ~15s via
    # keepalives — ConnectTimeout only covers the initial connect, not a mid-stream stall.
    BASE_OPTS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                 "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3")

    def __init__(self, host: str, remote_workdir: str,
                 ssh: str = "ssh", ssh_opts: tuple[str, ...] = ()):
        self.host = host
        self.remote_workdir = remote_workdir
        self.ssh = ssh
        self.ssh_opts = list(ssh_opts)

    def _remote_command(self, spec: JobSpec, remote_run: str, remote_artifacts: str) -> str:
        sub = {"run_dir": remote_run, "artifact_dir": remote_artifacts}
        argv = " ".join(shlex.quote(a.format(**sub)) for a in spec.entrypoint)
        envs = " ".join(f"{k}={shlex.quote(v.format(**sub))}" for k, v in spec.env.items())
        parts = [f"cd {shlex.quote(self.remote_workdir)}"]
        if spec.repo_ref:
            parts.append(f"git fetch --quiet && git checkout --quiet {shlex.quote(spec.repo_ref)}")
        parts.append(f"mkdir -p {shlex.quote(remote_artifacts)}")
        parts.append((f"{envs} " if envs else "") + argv)
        return " && ".join(parts)

    def run(self, spec: JobSpec, run_dir: Path, artifact_dir: Path, log_path: Path) -> int:
        remote_run = f"{self.remote_workdir}/.runs/{run_dir.name}"
        remote_artifacts = f"{remote_run}/{spec.artifact_subdir}"
        remote_cmd = self._remote_command(spec, remote_run, remote_artifacts)
        with open(log_path, "w", encoding="utf-8", errors="replace") as log:
            rc = subprocess.run([self.ssh, *self.BASE_OPTS, *self.ssh_opts, self.host, remote_cmd],  # noqa: S603
                                stdout=log, stderr=subprocess.STDOUT, text=True).returncode
            if rc != 0 or not spec.fetch_artifacts:
                if not spec.fetch_artifacts:
                    log.write("\n[ssh] fetch_artifacts=False — job published its own output\n")
                return rc
            log.write("\n[ssh] fetching artifacts via tar stream\n")
            return self._fetch_artifacts(remote_run, spec.artifact_subdir, run_dir, log)

    def _fetch_artifacts(self, remote_run: str, subdir: str, run_dir: Path, log,
                         attempts: int = 3, timeout: float = 120.0) -> int:
        """`ssh host 'cd run && tar czf - subdir | base64'` → decode + extract locally.

        The archive is gzipped *and* base64-encoded so it travels as **text**: raw binary
        over a freshly-revived tailnet path stalls and the connection dies (the job's text
        stdout returns fine — only binary fails), whereas base64 text comes back cleanly.
        Retries: keepalives (BASE_OPTS) drop a stalled session in ~15s; `timeout` is a hard
        backstop; a corrupt/truncated payload also triggers a retry.
        """
        tar_cmd = [self.ssh, *self.BASE_OPTS, *self.ssh_opts, self.host,
                   f"cd {shlex.quote(remote_run)} && tar czf - {shlex.quote(subdir)} | base64"]
        last = "no attempt made"
        for attempt in range(1, attempts + 1):
            try:
                proc = subprocess.run(tar_cmd, stdout=subprocess.PIPE,  # noqa: S603
                                      stderr=subprocess.PIPE, timeout=timeout)
            except subprocess.TimeoutExpired:
                last = f"fetch exceeded {timeout:.0f}s"
                log.write(f"[ssh] artifact fetch attempt {attempt}/{attempts}: {last}\n")
                continue
            if proc.returncode != 0:
                last = proc.stderr.decode("utf-8", "replace").strip() or f"rc={proc.returncode}"
                log.write(f"[ssh] artifact fetch attempt {attempt}/{attempts} failed: {last}\n")
                continue
            try:
                raw = base64.b64decode(proc.stdout)         # ignores the wrapping newlines
                with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tf:
                    tf.extractall(run_dir, filter="data")   # filter guards path traversal (py3.12+)
                return 0
            except (tarfile.TarError, ValueError) as exc:   # corrupt/truncated payload → retry
                last = f"corrupt payload: {exc}"
                log.write(f"[ssh] artifact fetch attempt {attempt}/{attempts}: {last}\n")
        log.write(f"[ssh] artifact fetch failed after {attempts} attempts ({last})\n")
        return 1

    # -- background (fire-and-poll) -----------------------------------------
    def launch(self, spec: JobSpec, run_dir: Path, artifact_dir: Path, log_path: Path) -> dict:
        """`nohup` the job detached on the host; return a handle. Status comes from progress.json."""
        remote_run = f"{self.remote_workdir}/.runs/{run_dir.name}"
        remote_artifacts = f"{remote_run}/{spec.artifact_subdir}"
        sub = {"run_dir": remote_run, "artifact_dir": remote_artifacts}
        argv = " ".join(shlex.quote(a.format(**sub)) for a in spec.entrypoint)
        envs = " ".join(f"{k}={shlex.quote(v.format(**sub))}" for k, v in spec.env.items())
        parts = [f"cd {shlex.quote(self.remote_workdir)}"]
        if spec.repo_ref:
            parts.append(f"git fetch --quiet && git checkout --quiet {shlex.quote(spec.repo_ref)}")
        parts.append((f"{envs} " if envs else "") + argv)
        inner = " && ".join(parts)
        remote_cmd = (f"mkdir -p {shlex.quote(remote_artifacts)}; "
                      f"nohup sh -c {shlex.quote(inner)} > {shlex.quote(remote_run)}/run.log 2>&1 "
                      f"& echo $!")
        proc = subprocess.run([self.ssh, *self.BASE_OPTS, *self.ssh_opts, self.host, remote_cmd],  # noqa: S603
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        pid = proc.stdout.strip().splitlines()[-1].strip() if proc.stdout.strip() else ""
        Path(log_path).write_text(f"[ssh] launched detached pid={pid}\n{proc.stderr}",
                                  encoding="utf-8")
        return {"executor": "ssh", "host": self.host, "remote_run": remote_run,
                "remote_artifacts": remote_artifacts, "pid": pid}

    def read_progress(self, handle: dict) -> dict | None:
        cmd = f"cat {shlex.quote(handle['remote_artifacts'])}/progress.json 2>/dev/null"
        proc = subprocess.run([self.ssh, *self.BASE_OPTS, *self.ssh_opts, handle["host"], cmd],  # noqa: S603
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            return json.loads(proc.stdout) if proc.stdout.strip() else None
        except json.JSONDecodeError:
            return None

    def is_alive(self, handle: dict) -> bool:
        pid = handle.get("pid")
        if not pid:
            return False
        proc = subprocess.run(  # noqa: S603
            [self.ssh, *self.BASE_OPTS, *self.ssh_opts, handle["host"],
             f"kill -0 {shlex.quote(str(pid))} 2>/dev/null && echo 1 || echo 0"],
            stdout=subprocess.PIPE, text=True)
        return proc.stdout.strip() == "1"
