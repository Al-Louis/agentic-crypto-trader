"""trader.train — the training loop's domain logic (config → train → diagnose → iterate).

Pure, testable core: `config` (run configs + stable keys), `registry` (experiment lineage),
`diagnose` (the honest gates a run must clear before it counts as progress). The dispatch/fetch
orchestration wraps the generic `remote_train` substrate; the MCP server exposes these as the
loop tools (vault "MCP Server" / "AI Training").
"""

from __future__ import annotations

from trader.train.config import config_key, demo_config
from trader.train.diagnose import Gate, diagnose
from trader.train.registry import Experiment, Registry

__all__ = ["demo_config", "config_key", "diagnose", "Gate", "Registry", "Experiment"]
