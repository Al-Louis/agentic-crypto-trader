"""Executors — *where* a job's command runs. The orchestrator stays identical across them.

`LocalExecutor` runs on this machine (used now for the pipeline-first proof and CI).
`SSHExecutor` runs on a remote host over SSH (the desktop training box — chosen for CPU cores
+ RAM, since this RL workload is env-stepping-bound, not GPU-bound — reachable via Tailscale)
and syncs the remote artifact directory back. Both implement the same `run` contract, so
swapping local→remote is a one-line change at the call site.
"""

from __future__ import annotations

import io
import os
import shlex
import subprocess
import tarfile
from pathlib import Path

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
            rc = subprocess.run([self.ssh, *self.ssh_opts, self.host, remote_cmd],  # noqa: S603
                                stdout=log, stderr=subprocess.STDOUT, text=True).returncode
            if rc != 0:
                return rc
            log.write("\n[ssh] fetching artifacts via tar stream\n")
            return self._fetch_artifacts(remote_run, spec.artifact_subdir, run_dir, log)

    def _fetch_artifacts(self, remote_run: str, subdir: str, run_dir: Path, log) -> int:
        """`ssh host 'cd run && tar cf - subdir'` → extract into the local run dir."""
        tar_cmd = [self.ssh, *self.ssh_opts, self.host,
                   f"cd {shlex.quote(remote_run)} && tar cf - {shlex.quote(subdir)}"]
        proc = subprocess.run(tar_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # noqa: S603
        if proc.returncode != 0:
            log.write(proc.stderr.decode("utf-8", "replace"))
            return proc.returncode
        with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tf:
            tf.extractall(run_dir, filter="data")   # filter guards against path traversal (py3.12+)
        return 0
