"""Experiment registry — config → run → result lineage for the training loop.

A flat JSON store (one file per experiment) so "what did we change, and did it help?" is
answerable across iterations, and so the loop can build on a parent instead of starting blind.
This is the memory the train → evaluate → diagnose loop iterates against (vault "MCP Server").

Timestamps are injected (callers pass `created`), keeping the registry pure/testable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class Experiment:
    id: str
    config: dict[str, Any]
    parent_id: str | None = None
    created: str | None = None
    run_id: str | None = None
    metrics: dict[str, Any] | None = None
    diagnosis: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Registry:
    """JSON-backed experiment store under `path` (one ``exp-NNN.json`` per experiment)."""

    def __init__(self, path: Path | str = "experiments"):
        self.path = Path(path)

    def _file(self, exp_id: str) -> Path:
        return self.path / f"{exp_id}.json"

    def _next_id(self) -> str:
        n = len(list(self.path.glob("exp-*.json"))) if self.path.exists() else 0
        return f"exp-{n + 1:03d}"

    def register(self, config: dict[str, Any], *, parent_id: str | None = None,
                 created: str | None = None, exp_id: str | None = None) -> Experiment:
        self.path.mkdir(parents=True, exist_ok=True)
        exp = Experiment(id=exp_id or self._next_id(), config=config,
                         parent_id=parent_id, created=created)
        self._write(exp)
        return exp

    def record(self, exp_id: str, *, run_id: str | None = None,
               metrics: dict[str, Any] | None = None,
               diagnosis: dict[str, Any] | None = None) -> Experiment:
        exp = self.get(exp_id)
        if exp is None:
            raise KeyError(exp_id)
        if run_id is not None:
            exp.run_id = run_id
        if metrics is not None:
            exp.metrics = metrics
        if diagnosis is not None:
            exp.diagnosis = diagnosis
        self._write(exp)
        return exp

    def get(self, exp_id: str) -> Experiment | None:
        try:
            return Experiment(**json.loads(self._file(exp_id).read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError):
            return None

    def list(self) -> list[Experiment]:
        if not self.path.exists():
            return []
        return [e for p in sorted(self.path.glob("exp-*.json")) if (e := self.get(p.stem))]

    def lineage(self, exp_id: str) -> list[Experiment]:
        """The chain from the root ancestor down to `exp_id` (root first)."""
        chain: list[Experiment] = []
        seen: set[str] = set()
        cur = self.get(exp_id)
        while cur is not None and cur.id not in seen:
            chain.append(cur)
            seen.add(cur.id)
            cur = self.get(cur.parent_id) if cur.parent_id else None
        return list(reversed(chain))

    def _write(self, exp: Experiment) -> None:
        self._file(exp.id).write_text(json.dumps(exp.to_dict(), indent=2), encoding="utf-8")
