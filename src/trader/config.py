"""Local config — load the git-ignored .env without a hard python-dotenv dep.

Secrets (CMC / BscScan / TWAK keys) live in `.env` (see `.env.example`) and never
in the repo. `require()` is the gate keyed clients call; it fails loud with a
pointer to `.env.example` rather than making a half-configured network call.
"""

from __future__ import annotations

import os

_loaded = False


def load_dotenv(path: str = ".env") -> dict:
    """Parse `.env` into os.environ (without overriding already-set vars)."""
    global _loaded
    vals: dict[str, str] = {}
    if not os.path.exists(path):
        _loaded = True
        return vals
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            vals[k] = v
            os.environ.setdefault(k, v)
    _loaded = True
    return vals


def get(name: str, default: str | None = None) -> str | None:
    if not _loaded:
        load_dotenv()
    return os.environ.get(name, default)


def require(name: str) -> str:
    v = get(name)
    if not v:
        raise RuntimeError(
            f"{name} is not set — copy .env.example to .env and fill it in."
        )
    return v
