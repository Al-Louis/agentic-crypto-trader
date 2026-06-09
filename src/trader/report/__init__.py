"""trader.report — turn sim/eval results into shareable artifacts.

`apentic` exports the Apentic dashboard bundle (the static-JSON contract the frontend at
`alexlouis-site/src/apentic/` reads). This is the project-specific half of the telemetry
pipeline; the generic job orchestration lives in the standalone `remote_train` package.
"""

from __future__ import annotations

from trader.report.apentic import (
    candles_from_ohlcv,
    equity_points,
    export_portfolio_run,
    export_run,
    metrics_to_frontend,
    publish_run,
    roundtrips_from_position,
    upsert_manifest,
    upsert_manifest_at,
)

__all__ = [
    "export_run", "export_portfolio_run", "roundtrips_from_position", "metrics_to_frontend",
    "candles_from_ohlcv", "equity_points",
    "upsert_manifest", "upsert_manifest_at", "publish_run",
]
