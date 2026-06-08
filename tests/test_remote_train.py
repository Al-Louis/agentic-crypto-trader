"""Tests for the generic remote-train orchestrator (no trading knowledge here)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import remote_train as rt
from remote_train.executor import LocalExecutor, _substitute
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


def test_remote_train_never_imports_trader():
    """The whole point of the package: it stays liftable into its own repo."""
    pkg = Path(rt.__file__).parent
    offenders = [f.name for f in pkg.glob("*.py")
                 if re.search(r"^\s*(from|import)\s+trader\b", f.read_text(encoding="utf-8"), re.M)]
    assert offenders == [], f"remote_train must not import trader: {offenders}"


def submit_in(spec, tmp_path):
    return rt.submit(spec, executor=LocalExecutor(), store=tmp_path / "runs")
