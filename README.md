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
pip install -e ".[dev,data]"       # package + dev + data/features/sim deps (pandas, pyarrow, ta, ccxt)
Copy-Item .env.example .env        # fill in keys (never commit)
python -m trader.mcp_server        # run the project MCP server (skeleton)
pytest                             # smoke test
```

On macOS/Linux: `source .venv/bin/activate`, `cp .env.example .env`. The `.mcp.json`
launcher points at `.venv\Scripts\python.exe` (Windows); change it to `.venv/bin/python`
on POSIX.

The `trader` MCP server is registered in `.mcp.json`. Tools are added per phase — see
`.obsidian-vault/BNB Hackathon/MCP Server.md`.

## Layout (as built)

```
src/trader/
  config.py    .env loader
  data/        universe + OHLCV: dexscreener · geckoterminal · cmc · goplus ·
               select · downloader (resumable Parquet) · anchor (ccxt BTC/BNB)   [built]
  features/    indicators (71-col + leakage guard) · factor (BTC/BNB residual)   [built]
  sim/         metrics · broker (AMM cost) · backtest · strategies · resample · ic [built]
  {execution,strategy,risk,agent,monitoring}/  stubs (Phase 2+)
  mcp_server/  skeleton (health + eligible_tokens stub)
scripts/       research CLIs (see pipeline below)
data/          generated caches — git-ignored (ohlcv/ anchor/ features/ *.json)
tests/         ~89 pytest functions
```

## Research pipeline (offline; keyless except CMC)

```powershell
python scripts/screen_universe.py        # 149 eligible -> DexScreener on-chain screen
python scripts/resolve_contracts.py      # -> canonical BSC contracts (needs CMC_API_KEY)
python scripts/select_universe.py --exclude SHIB,BAS,FORM --pin LTC:anchor
python scripts/forensics.py              # GoPlus rug/honeypot gate
python scripts/download_ohlcv.py --selection data/selection.json --timeframes day,hour
python scripts/download_anchor.py        # BTC/BNB factor anchor (ccxt / Binance.US)
python scripts/build_factor_features.py  # two-factor "Bitcoin-is-King" residual model
python scripts/run_backtest.py           # cost-aware baseline backtest
python scripts/tail_sweep.py             # tournament-objective sweep (upper tail vs DQ)
python scripts/oos_validate.py           # out-of-sample validation of the vol tilt
```

Findings and the current strategy candidate live in `.obsidian-vault/BNB Hackathon/`
(`Build Log`, `Trading Strategies`, `Token Universe`).

> Secrets never committed. Self-custody signing stays local. See `Security and Encryption`.
