# CLAUDE.md — agentic-crypto-trader

Orientation for any Claude agent working in this project. **Read this first**, then run
`/orient` to load the current objective and phase. (`/goal` is a *different*, built-in
command — see the pipeline section.)

## What this project is

A from-scratch build of an **autonomous, self-custody crypto trading agent** for the BNB
Chain "AI Trading Agent Edition" hackathon (Track 1). The agent reads market and on-chain
data, decides, and signs and executes its own transactions on BSC, hands-off, inside hard
guardrails. The trading strategy is an **open design space** — not yet committed.

The single source of project knowledge is the Obsidian vault at `.obsidian-vault/`. Start
from its map of content: **[[Index]]**, and the neutral **[[Project Overview]]**.

## The two halves of this repo

| Path | What it is | Posture |
|------|-----------|---------|
| `.obsidian-vault/` | Living project documentation — the MOC (`Index`), topic notes (`BNB Hackathon/`), prior work (`Past Work/`), and mirrored SDK docs (`References/`). | **Read-write.** Agents develop these notes. See conventions below. |
| `src/` *(to be created)* | The agent codebase — execution loop, strategy modules, guardrails, MCP server. | Read-write. |

> **Note — vault posture differs from the `claude-vault` project.** There, the vault is a
> read-only personal source of truth. **Here, the vault is a working knowledge base that
> agents are expected to write to and grow.** Keep it tidy: prefer developing the existing
> topic stubs over creating sprawl, and do not restructure the `Index` MOC without reason —
> it is the human-maintained entry point.

## The stack — four surfaces

Detail lives in [[Tech Stack]]; the shape (the "three SDKs" framing undercounts):

- **Trust Wallet Agent Kit (TWAK)** — execution + self-custody signing (the `twak` CLI/MCP). Sole execution layer; competition registration runs through it.
- **CoinMarketCap AI Agent Hub (MCP)** — market/on-chain/social/news data **and x402** (x402 is in the Agent Hub MCP, not the standalone `cmc` CLI).
- **BNB AI Agent SDK** (Python) — agent runtime + on-chain identity (ERC-8004). **Not an execution layer** — trades route through TWAK.
- **BscScan API** — on-chain analytics for wallet/transfer monitoring and on-chain signals.

Runtime is **Python**. SDK reference docs are mirrored under `.obsidian-vault/References/`.

## Objective & key dates

- **Scored on live PnL** (June 22–28), with a **hard max-drawdown DQ gate** (~30%), ≥1
  trade/day, and a fixed eligible-token list. Discretionary **special prizes** per SDK are a
  lower-variance path alongside the leaderboard.
- **June 16 — Track 1 proof-of-concept gate.** The PoC must show the **live execution loop
  end-to-end on-chain** (a real trade signed and landed via TWAK, guardrails active) — not
  just an offline backtest. **If the live loop isn't real by June 16, switch to Track 2.**
- **Register on-chain before June 22.** Full rules: [[BNB Hack - AI Trading Agent Edition]].

## How we work — the pipeline

This project is built around a small set of harness features. They compose: **`/goal`**
(built-in) drives work to a verifiable end state, **`/orient`** aligns a fresh session,
**agents** own domains, **workflows** orchestrate them, and the **project MCP server** gives
them shared, deterministic operations.

### `/goal` — built-in completion condition
A built-in Claude Code command. `/goal <condition>` sets a verifiable end state and
auto-continues across turns until it's met (a fast model checks after each turn), then clears
itself. Use it to drive a phase artifact to done, e.g.
`/goal a dust trade lands on BSC via TWAK with a confirmed tx hash`. Clear with `/goal clear`.

### `/orient` — session orientation
A project slash command (`.claude/commands/orient.md`) that loads the current north-star,
active phase, priorities, and blockers so a fresh chat starts aligned. Run it first.

### Agents — domain owners (`.claude/agents/`)
Specialized subagents, each owning a slice of the work and a set of vault notes. Spawn them
for focused tasks or fan them out via a workflow. **Proposed roster** (confirm/adjust before
building the full set):

| Agent | Domain | Owns (notes) |
|-------|--------|--------------|
| `principal-engineer` | Architecture, execution loop, MCP server, integration, code review — the technical lead | [[Tech Stack]], [[Remote Capabilities]] |
| `rl-ml-trainer` | RL/ML training, reward design, curriculum | [[AI Training]], [[Simulated Market]] |
| `quant-analyst` | Quantitative market analysis, backtest methodology, risk metrics, honest evaluation | [[Market Conditions]], [[Simulated Market]] |
| `market-indicator-expert` | Technical indicators, signal design, strategy logic | [[Trading Strategies]] |
| `onchain-custody-engineer` | TWAK self-custody, key management, wallet/tx monitoring (BscScan) | [[Security and Encryption]], [[Real-time Monitoring]] |
| `sentiment-scanner` *(optional)* | X.com / news scanning for breaking events | [[Social Media Scanner]] |

### Workflows — orchestration (`/workflows`)
Use the built-in Workflow tool to run agents deterministically — fan out across domains,
pipeline phases, verify before committing. Reserve for genuinely multi-agent work; one
focused task is a single agent, not a workflow.

### Project MCP server *(to be built — `mcp-server/`, registered in `.mcp.json`)*
An **isolated** MCP server exposing this project's operations as tools so agents and
workflows drive them deterministically — mirroring the train→evaluate→diagnose loop pattern
from [[TradeSim]]. **The full command-set design lives in [[MCP Server]]** (tool catalog,
safety tiers, and build phasing); it is built incrementally from Phase 2 onward.

## Remote training — deployment (READ before launching ANY training run)

Full runbook: **[[Remote Capabilities]] §"Remote training — deploy runbook"**. This process took
a full day to nail down and was once lost to context compression — **do not improvise it.** The
training desktop is **`root@<TRAINER_TAILNET_IP>`** (WSL2 `act-trainer`, FQDN
`act-trainer.<TAILNET>.ts.net`, 8c/16t, 32 GB; keyless, no mainnet; jobs self-publish to
`data.alexlouis.dev`). The five rules that, if forgotten, break things:

1. **SSH only via the PowerShell tool (Windows OpenSSH), never the Bash tool's ssh.** The
   Bash/MSYS ssh can't route to the tailnet and hangs forever. From PowerShell:
   `ssh root@<TRAINER_TAILNET_IP> '<cmd>'` (pass multi-line remote cmds via a single-quoted here-string).
2. **Keep every SSH response tiny (< ~512 B).** The tailnet has a **path-MTU black hole** — replies
   ≥ ~4 KB stall and kill the session. Status checks return *counts*, never full `ps` / `pgrep -fa`.
3. **Launch the sweep ONCE, then WAIT 60–90 s before verifying.** Torch import + the volume-panel
   build mean *no process or log appears for ~30–60 s*. A fast empty check is **not** failure —
   relaunching stacks **parallel** sweeps that oversubscribe the WSL2 VM (Vmmem), throttle the
   Windows host, and force a reboot. The sweep script already **sequences** seeds internally; the
   word is **sequence, never parallel**.
4. **`mkdir -p runs-rl runs-rl/<sweep>-logs` first.** It's gitignored (absent on a fresh checkout),
   and a missing dir makes the `> runs-rl/<sweep>.log` redirect fail *silently* — looks like a dead
   launch, tempting a (fatal) relaunch.
5. **Sync + preflight before launching:** on the desktop `git fetch && git checkout <sha>`, confirm
   HEAD == the pushed sha, and that `data/ohlcv/hour_1/` + `build_volume_panel` work (market data is
   gitignored — it lives only on the box). Then `nohup bash scripts/<sweep>.sh TS "SEEDS" >
   runs-rl/<sweep>.log 2>&1 < /dev/null &`. Aggregate via `compare_seeds.py` / `compare_sweep.py`
   (pull `metrics.json` from `data.alexlouis.dev`). Pattern script: `scripts/run_reward_sweep.sh`.

**Stopping a run:** kill the **specific PIDs** (driver `bash` + `train_rl` python main) — **never**
`kill -- -<PGID>`. A `nohup`'d job shares the tailscaled SSH session's process group, so a
group-kill takes **tailscaled** down too and drops the box off the tailnet (happened 2026-06-09;
needed a host-side `tailscale up --ssh` to recover).

## Conventions

- **Neutral, factual docs.** Topic notes describe options and decisions on their merits;
  avoid locking the project into one strategy prematurely.
- **Strategy is modular.** Keep the decision core behind a clean interface; execution,
  custody, and guardrails stay strategy-agnostic.
- **Guardrails are hard, external limits** in code around the TWAK signing call (allowlist,
  per-trade/daily caps, slippage, drawdown stop) — never prompt suggestions.
- **Validate offline before live capital.** Strategy logic is pure and testable against
  recorded/simulated data ([[Simulated Market]]) before it touches mainnet.
- **Secrets never committed.** Keys, mnemonics, API keys, wallet files → local `.env`
  (git-ignored). Self-custody signing stays local; treat the hosting/key story as a blocker
  ([[Security and Encryption]]).
- **Cite the vault.** When agents act, ground decisions in the relevant topic note and keep
  that note updated as understanding grows.

## Phased plan (working draft — refine with `/goal`)

1. **Foundation** — this CLAUDE.md, agent roster, `/goal`, repo/git init, Python project skeleton.
2. **Stack spike** — stand up all four surfaces; land a real dust trade on BSC via TWAK.
3. **Execution loop + guardrails** — autonomous read→decide(stub)→sign→confirm with caps enforced.
4. **Strategy + offline validation** — implement the chosen decision core; validate vs simulated market.
5. **June 16 PoC gate** — live loop proven on-chain, or pivot to Track 2.
6. **Harden + forward-run**, **register + submit**, then **live window** (June 22–28).

The MCP server is built incrementally from Phase 2 onward, exposing each operation the
agents need as a tool.

## Relationship to the personal-context vault

Prior work referenced here (e.g. [[TradeSim]]) originates from a separate personal vault
project. This project is independent and self-contained; do not assume access to that vault.
