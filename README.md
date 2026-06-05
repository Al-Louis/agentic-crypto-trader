# agentic-crypto-trader

An autonomous, self-custody crypto trading agent for the BNB Chain "AI Trading Agent
Edition" hackathon (Track 1). The trading strategy is an open design space; the
infrastructure (execution, custody, data, guardrails) is the same regardless of strategy.

- **Orientation:** see `CLAUDE.md` (auto-loaded), then run `/orient`.
- **Knowledge base:** `.obsidian-vault/` — start at `BNB Hackathon/Index.md` →
  `Project Overview`, `Tech Stack`, `MCP Server`.
- **SDK docs:** mirrored under `.obsidian-vault/References/`.

## Quickstart (Windows PowerShell)

```powershell
python -m venv .venv               # create virtual env
.venv\Scripts\Activate.ps1         # activate it
pip install -e ".[dev]"            # install package + dev tools (editable)
Copy-Item .env.example .env        # fill in keys (never commit)
python -m trader.mcp_server        # run the project MCP server (skeleton)
pytest                             # smoke test
```

On macOS/Linux: `source .venv/bin/activate`, `cp .env.example .env`. The `.mcp.json`
launcher points at `.venv\Scripts\python.exe` (Windows); change it to `.venv/bin/python`
on POSIX.

The `trader` MCP server is registered in `.mcp.json`. Tools are added per phase — see
`.obsidian-vault/BNB Hackathon/MCP Server.md`.

## Layout

```
src/trader/   execution · data · strategy · risk · agent · monitoring · mcp_server
tests/        smoke + unit tests
```

> Secrets never committed. Self-custody signing stays local. See `Security and Encryption`.
