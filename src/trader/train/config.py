"""Training-run configs for the loop.

A config is a plain dict so RL params (reward weights, curriculum, network) stay open-ended;
these helpers build the demo-heuristic config the loop is first scaffolded against and a stable
key (for run ids / dedup / "have we tried this before?"). See vault "AI Training" / "MCP Server".
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def demo_config(token: str = "HUMA", ema: int = 168, band: float = 0.04) -> dict[str, Any]:
    """The config the loop tunes while scaffolded on the demo heuristic (pre-RL)."""
    return {"kind": "demo-heuristic", "token": token, "ema": int(ema), "band": float(band)}


def config_key(config: dict[str, Any]) -> str:
    """Stable short hash of a config — order-independent, for dedup and run ids."""
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10]  # noqa: S324 - dedup key, not security
