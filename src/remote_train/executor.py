"""Executors — *where* a job's command runs. The orchestrator stays identical across them.

`LocalExecutor` runs on this machine (used now for the pipeline-first proof and CI).
`SSHExecutor` runs on a remote host over SSH (the desktop GPU box, reachable via Tailscale)
and syncs the remote artifact directory back. Both implement the same `run` contract, so
swapping local→remote is a one-line change at the call site.
"""

from __future__ import annotations

import os
import shlex
import subprocess
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
    """Run the job on a remote host over SSH, then rsync its artifacts back.

    Assumes key-based SSH to ``host`` (e.g. a Tailscale name) and that the repo already
    lives at ``remote_workdir`` on that host. The training box holds **no keys and never
    touches mainnet** (vault "Remote Capabilities") — it is pure compute on recorded data.
    """

    name = "ssh"

    def __init__(self, host: str, remote_workdir: str,
                 ssh: str = "ssh", rsync: str = "rsync", ssh_opts: tuple[str, ...] = ()):
        self.host = host
        self.remote_workdir = remote_workdir
        self.ssh = ssh
        self.rsync = rsync
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
        ssh_cmd = [self.ssh, *self.ssh_opts, self.host, remote_cmd]
        with open(log_path, "w", encoding="utf-8", errors="replace") as log:
            rc = subprocess.run(ssh_cmd, stdout=log, stderr=subprocess.STDOUT, text=True).returncode  # noqa: S603
            if rc != 0:
                return rc
            # Pull artifacts back so downstream (publish) is host-agnostic.
            pull = [self.rsync, "-az", f"{self.host}:{remote_artifacts}/", f"{artifact_dir}/"]
            log.write(f"\n[ssh] rsync artifacts: {' '.join(pull)}\n")
            return subprocess.run(pull, stdout=log, stderr=subprocess.STDOUT, text=True).returncode  # noqa: S603
