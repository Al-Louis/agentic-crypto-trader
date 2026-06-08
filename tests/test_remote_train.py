"""Tests for the generic remote-train orchestrator (no trading knowledge here)."""

from __future__ import annotations

import base64
import io
import re
import sys
import tarfile
from pathlib import Path

import remote_train as rt
from remote_train.executor import LocalExecutor, SSHExecutor, _substitute
from remote_train.spec import JobSpec


def test_jobspec_placeholder_substitution():
    spec = JobSpec(name="j", entrypoint=["python", "-c", "{artifact_dir}"],
                   env={"OUT": "{run_dir}/x"})
    argv, env = _substitute(spec, Path("/runs/j-001"), Path("/runs/j-001/artifacts"))
    assert argv[-1].replace("\\", "/") == "/runs/j-001/artifacts"
    assert env["OUT"].replace("\\", "/") == "/runs/j-001/x"


def test_submit_runs_job_and_captures_artifacts(tmp_path):
    # A job that writes a file into its artifact dir, the generic "output bundle".
    spec = JobSpec(
        name="writer",
        entrypoint=[sys.executable, "-c",
                    "import sys,pathlib;"
                    "p=pathlib.Path(sys.argv[1]);p.mkdir(parents=True,exist_ok=True);"
                    "(p/'out.txt').write_text('hi')",
                    "{artifact_dir}"],
    )
    st = submit_in(spec, tmp_path)
    assert st.ok and st.returncode == 0
    assert (Path(st.artifact_dir) / "out.txt").read_text() == "hi"


def test_submit_records_failure(tmp_path):
    spec = JobSpec(name="boom", entrypoint=[sys.executable, "-c", "import sys;sys.exit(3)"])
    st = submit_in(spec, tmp_path)
    assert st.state == "failed" and st.returncode == 3
    # status() re-reads from disk → cross-process visibility.
    assert rt.status(st.run_id, store=tmp_path / "runs").state == "failed"


def test_run_ids_increment(tmp_path):
    spec = JobSpec(name="seq", entrypoint=[sys.executable, "-c", "pass"])
    a = submit_in(spec, tmp_path)
    b = submit_in(spec, tmp_path)
    assert a.run_id == "seq-001" and b.run_id == "seq-002"
    assert set(rt.list_runs(store=tmp_path / "runs")) == {"seq-001", "seq-002"}


def test_progress_roundtrip_and_history(tmp_path):
    rt.write_progress(tmp_path, episode=1, reward=0.5, history_key="curve")
    rt.write_progress(tmp_path, episode=2, reward=0.9, history_key="curve")
    p = rt.read_progress(tmp_path)
    assert p["episode"] == 2 and p["reward"] == 0.9        # latest scalars
    assert [c["reward"] for c in p["curve"]] == [0.5, 0.9]  # accumulated history


def test_publish_local_merge_copy(tmp_path):
    src = tmp_path / "bundle"
    (src / "sub").mkdir(parents=True)
    (src / "a.json").write_text("1")
    (src / "sub" / "b.json").write_text("2")
    dest = tmp_path / "dash"
    rt.publish(src, str(dest))
    assert (dest / "a.json").read_text() == "1"
    assert (dest / "sub" / "b.json").read_text() == "2"


def test_ssh_remote_command_builds_shell():
    ex = SSHExecutor(host="root@act-trainer", remote_workdir="/root/app")
    spec = JobSpec(name="j", entrypoint=["/root/app/.venv/bin/python", "s.py", "--out",
                                         "{artifact_dir}"], env={"K": "v"})
    cmd = ex._remote_command(spec, "/root/app/.runs/j-001", "/root/app/.runs/j-001/artifacts")
    assert cmd.startswith("cd /root/app")
    assert "mkdir -p /root/app/.runs/j-001/artifacts" in cmd
    assert "K=v /root/app/.venv/bin/python s.py --out /root/app/.runs/j-001/artifacts" in cmd


def test_ssh_fetch_artifacts_extracts_tar_stream(tmp_path, monkeypatch):
    # Emulate the remote `tar czf - artifacts | base64` stdout the fetch decodes + extracts.
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        data = b"[]"
        info = tarfile.TarInfo("artifacts/manifest.json")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))

    class FakeProc:
        returncode, stdout, stderr = 0, base64.b64encode(buf.getvalue()), b""

    monkeypatch.setattr("remote_train.executor.subprocess.run", lambda *a, **k: FakeProc())
    ex = SSHExecutor(host="h", remote_workdir="/root/app")
    run_dir = tmp_path / "j-001"
    run_dir.mkdir()
    rc = ex._fetch_artifacts("/root/app/.runs/j-001", "artifacts", run_dir, log=io.StringIO())
    assert rc == 0
    assert (run_dir / "artifacts" / "manifest.json").read_text() == "[]"


def test_get_put_bytes_local_roundtrip(tmp_path):
    uri = str(tmp_path / "sub" / "a.json")
    assert rt.get_bytes(uri) is None
    rt.put_bytes(uri, b'{"x":1}', content_type="application/json")
    assert rt.get_bytes(uri) == b'{"x":1}'


def test_ssh_skips_fetch_when_disabled(tmp_path, monkeypatch):
    ex = SSHExecutor(host="h", remote_workdir="/root/app")

    class P:
        returncode = 0

    monkeypatch.setattr("remote_train.executor.subprocess.run", lambda *a, **k: P())
    called = {"fetch": False}
    monkeypatch.setattr(ex, "_fetch_artifacts",
                        lambda *a, **k: called.__setitem__("fetch", True) or 99)
    run_dir = tmp_path / "j"
    (run_dir / "artifacts").mkdir(parents=True)
    spec = JobSpec(name="j", entrypoint=["echo"], fetch_artifacts=False)
    rc = ex.run(spec, run_dir, run_dir / "artifacts", run_dir / "log.txt")
    assert rc == 0 and called["fetch"] is False     # job self-published → no haul-back


def test_invalidate_cloudfront_calls_boto3(monkeypatch):
    import types
    calls = {}
    fake_client = types.SimpleNamespace(
        create_invalidation=lambda **kw: calls.update(kw) or {"Invalidation": {"Id": "I1"}})
    monkeypatch.setitem(sys.modules, "boto3",
                        types.SimpleNamespace(client=lambda svc, **kw: fake_client))
    iid = rt.invalidate_cloudfront("DIST", ["/apentic/data/*"], caller_reference="ref1")
    assert iid == "I1"
    assert calls["DistributionId"] == "DIST"
    assert calls["InvalidationBatch"]["Paths"]["Items"] == ["/apentic/data/*"]
    assert calls["InvalidationBatch"]["CallerReference"] == "ref1"


def test_remote_train_never_imports_trader():
    """The whole point of the package: it stays liftable into its own repo."""
    pkg = Path(rt.__file__).parent
    offenders = [f.name for f in pkg.glob("*.py")
                 if re.search(r"^\s*(from|import)\s+trader\b", f.read_text(encoding="utf-8"), re.M)]
    assert offenders == [], f"remote_train must not import trader: {offenders}"


def submit_in(spec, tmp_path):
    return rt.submit(spec, executor=LocalExecutor(), store=tmp_path / "runs")
