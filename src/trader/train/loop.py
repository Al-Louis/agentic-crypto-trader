"""One iteration of the train → evaluate → diagnose loop.

Ties the pieces together: register the config as an experiment → dispatch the job (via the
generic `remote_train` substrate) → fetch the *published* results (over normal internet from
`data.alexlouis.dev`, never the tailnet) → diagnose against the honest gates → record. The
self-publishing job means we read results from the CDN, not haul them back.

Scaffolded on the demo heuristic; when the RL env lands, only the entrypoint + config change —
the loop is identical (vault "MCP Server" / "AI Training").
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

from remote_train import JobSpec, submit
from trader.train.diagnose import diagnose
from trader.train.registry import Registry


def demo_run_id(config: dict[str, Any]) -> str:
    """A descriptive, config-unique run id (each distinct config is its own dashboard run)."""
    return f"{config['token'].lower()}-ema{config['ema']}-b{config['band']}"


def _fetch_json(uri: str) -> Any:
    """Read JSON from an http(s) URL (the published CDN) or a local path."""
    if urlparse(uri).scheme in ("http", "https"):
        with urlopen(uri, timeout=30) as resp:    # noqa: S310 - fixed CDN host, not user input
            return json.loads(resp.read().decode("utf-8"))
    return json.loads(Path(uri).read_text(encoding="utf-8"))


def fetch_artifact(results_base: str, run_id: str, name: str) -> Any:
    return _fetch_json(f"{results_base.rstrip('/')}/{run_id}/{name}")


def derive_baseline_and_days(results_base: str, run_id: str) -> tuple[float | None, float | None]:
    """From the published bundle: buy&hold return (single-asset baseline) + day-span.

    Lets `diagnose` run its beats-baseline and activity gates without the job emitting them.
    """
    baseline = days = None
    try:
        candles = fetch_artifact(results_base, run_id, "candles.json")
        if len(candles) >= 2 and candles[0]["close"]:
            baseline = candles[-1]["close"] / candles[0]["close"] - 1.0
    except (OSError, ValueError, KeyError, ZeroDivisionError):
        pass
    try:
        eq = fetch_artifact(results_base, run_id, "equity_curve.json")
        if len(eq) >= 2:
            days = (eq[-1]["time"] - eq[0]["time"]) / 86400.0
    except (OSError, ValueError, KeyError):
        pass
    return baseline, days


def run_iteration(config: dict[str, Any], registry: Registry, *, executor, store: Path | str,
                  results_base: str, python: str = "python", workdir: str = ".",
                  publish_target: str | None = None, parent_id: str | None = None,
                  created: str | None = None, derive_gates: bool = True):
    """Run one loop iteration: register → dispatch → fetch results → diagnose → record.

    Returns ``(experiment, run_status)``. On a failed dispatch the experiment is recorded with
    an ``error`` verdict (and the log path) so the loop can see what broke.
    """
    exp = registry.register(config, parent_id=parent_id, created=created)
    run_id = demo_run_id(config)

    entrypoint = [python, "scripts/export_demo_run.py", "--out", "{artifact_dir}",
                  "--token", config["token"], "--ema", str(config["ema"]),
                  "--band", str(config["band"]), "--run-id", run_id]
    if publish_target:
        entrypoint += ["--publish-target", publish_target]
    spec = JobSpec(name="apentic-train", entrypoint=entrypoint, workdir=workdir,
                   fetch_artifacts=False)

    status = submit(spec, executor=executor, store=store)
    if not status.ok:
        registry.record(exp.id, run_id=run_id,
                        diagnosis={"verdict": "error", "log_path": status.log_path})
        return registry.get(exp.id), status

    metrics = fetch_artifact(results_base, run_id, "metrics.json")
    baseline, days = derive_baseline_and_days(results_base, run_id) if derive_gates else (None, None)
    dx = diagnose(metrics, baseline_return=baseline, days=days)
    registry.record(exp.id, run_id=run_id, metrics=metrics, diagnosis=dx)
    return registry.get(exp.id), status
