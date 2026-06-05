# Remote Capabilities

How and where the agent runs off the developer's laptop. The live window demands an
always-on host; that host holds self-custody signing keys, which is the central tension —
key-safety detail lives in [[Security and Encryption]]. See [[Project Overview]] for scope,
[[Tech Stack]] for layout, and [[MCP Server]] for the tools referenced here.

## Why always-on is non-negotiable

The contest requires **≥1 trade/day for 7 days (June 22–28), hands-off** (see
[[BNB Hack - AI Trading Agent Edition]]). A laptop that sleeps, drops Wi-Fi, or reboots
during the window misses a day and forfeits ranking — an hour starting at ≤ $1 scores 0%, and
a missed daily trade fails the activity minimum. So the agent loop must run on a reliable,
unattended host for the full week.

The cost of that reliability: **self-custody signing keys end up on a remote box.** This is a
flagged blocker in [[Project Overview]]. This note owns *where and how the runtime lives*;
how the key is stored and unlocked safely there is deferred to [[Security and Encryption]].

## Two distinct remote needs — keep them separate

| | **Live trading runtime** | **Offline RL/ML training** |
|---|---|---|
| When | Continuous, June 22–28 (and forward-run before) | Burst, any time before the live window |
| Resource shape | Low CPU/RAM, network-reliable, must not miss a tick | GPU-friendly, high burst compute, interruptible |
| Custody posture | **Holds signing keys** — custody-sensitive | **No keys** — pure compute on recorded data |
| Failure mode | Missed day → forfeit; key exposure → loss of funds | Wasted compute; re-runnable |

These have opposite security and uptime profiles. **Run them on different hosts.** The
training box never sees a key; the trading box never runs untrusted training code or opens
GPU-driver attack surface. Conflating them puts keys next to the most experimental code in
the project — avoid it.

## Hosting options for the live runtime

Neutral tradeoffs; decide against custody exposure first, then uptime, then cost.

| Option | Uptime | Cost | Custody exposure | Notes |
|--------|--------|------|------------------|-------|
| **VPS** (Hetzner, DigitalOcean, Vultr) | High, SLA-backed | Low ($5–20/mo) | Key on a multi-tenant provider box | Simplest always-on path; pick a region near BSC RPC/data endpoints for latency. |
| **Cloud VM** (AWS EC2, GCP) | High | Medium; managed extras | Provider box, but KMS/secret-manager available | Heavier ops; worth it only if its secret tooling is actually used (see [[Security and Encryption]]). |
| **Home server / mini-PC** | Depends on home power + ISP | One-time hardware | Key stays on hardware you physically control | Best custody story, weakest uptime guarantee — needs a UPS and ISP failover to be week-reliable. |
| **Container on any of the above** | Inherits host | — | Inherits host | Reproducible deploy unit; does **not** add custody safety by itself. |

No option removes the core tension — an unattended host must be able to *unlock and sign*,
which is precisely the threat model in [[Security and Encryption]]. Region matters for both
RPC latency and any data-residency constraints.

## Deploying the runtime

Three processes run on the trading host:

1. **`twak serve --watch`** — the background runner. `serve` exposes the TWAK MCP (stdio) or
   a local REST API (`--rest --port <n>`, Bearer-auth with the HMAC secret). `--watch` is
   what actually executes saved `automate` automations and signing in the background;
   `--watch-interval <dur>` tunes the poll cadence (default 60s), and `--auto-lock <minutes>`
   re-locks the wallet after inactivity. Without `--watch`, automations are saved but never
   fire. The runner uses the **local agent wallet**; a WalletConnect session stays idle since
   it needs manual approval — so unattended operation requires the local-key path.
2. **The agent loop** (`trader.agent`) — read → decide → sign → confirm, calling TWAK to sign
   and the data surfaces to read.
3. **The `trader` MCP server** — `.venv\Scripts\python.exe -m trader.mcp_server` (stdio),
   exposing the project tools so workflows and the loop drive operations deterministically.

### Keeping it alive

- **Process supervision** with auto-restart: `systemd` (Linux) or a container restart policy
  (`restart: unless-stopped`). On a home Windows box, a Scheduled Task or `nssm` service.
- **Crash recovery must be idempotent.** On restart the loop re-reads on-chain state
  (`wallet_status`, `portfolio_pnl`) as truth — never replays a pending trade blindly, or a
  restart could double-submit. Guardrails (`guardrails_get`) re-load and re-arm on boot.
- **`--auto-lock` vs. unattended signing** is a real tension: a short auto-lock limits key
  exposure but means the host must re-unlock to trade, which needs a credential present on
  the box anyway (env / OS keychain). Resolve the unlock mechanism in
  [[Security and Encryption]]; this note just notes that supervision and auto-lock must be
  tuned together so a daily trade is never blocked by a locked wallet.

## Remote training orchestration

Lineage from [[TradeSim]]: training is launched as **background subprocesses driven by MCP
tools**, not interactive jobs — so it runs unattended on the GPU host and is workflow-drivable
from the principal host.

- `start_training` (config → run id) spawns the run; `training_status` (run id → metrics)
  polls progress. The host fires-and-polls rather than blocking.
- The full **train → evaluate → diagnose** loop runs remotely: `evaluate_model` against
  held-out data, `diagnose_run` for rule-based failure-mode checks (under-random,
  over-trading, fee drag, drawdown). Internals and reward design → [[AI Training]]; backtest
  methodology → [[Simulated Market]].
- **Surfacing results:** dashboards and exports (also a [[TradeSim]] pattern) make remote runs
  legible — metrics tables, equity curves, baseline comparisons pulled via `backtest_report`
  / `model_info`. The training host produces artifacts; the developer reviews them without
  SSHing into a live job.

The training host holds **no keys and touches no mainnet** — its only outputs are model
artifacts and reports that later flow into the strategy core.

## CI/CD and validation gates

- **Offline-first gate:** tests + lint + the strategy core's pure-logic checks must pass
  before any deploy. Strategy logic is validated against recorded/simulated data
  ([[Simulated Market]]) — nothing reaches mainnet on red.
- **Execution-tier separation in CI:** READ/SIMULATE tools (the 🟢/🟡 tiers in
  [[MCP Server]]) run freely in CI; 🔴 EXECUTE tools (`execute_trade`,
  `competition_register`) are never invoked by automation — they stay behind the explicit
  enable flag and run only on the deliberately provisioned host.
- **Deploy = reproducible unit** (container image or pinned `pip install -e`), config via
  git-ignored `.env`; secrets injected from OS keychain / host env, **never baked into an
  image or committed** (CLAUDE.md, [[Tech Stack]]).

## Observability and remote access

- **Health:** liveness check on `twak serve` and the loop heartbeat; alert if a daily trade
  hasn't landed by a cutoff (mirrors the activity minimum). Runtime watch targets and PnL →
  [[Real-time Monitoring]].
- **Logs:** every tx hash logged (per [[MCP Server]] execute-tier rule); structured loop logs
  shippable off-host so a crash leaves a trail.
- **Access:** key-based SSH only, firewalled to the RPC/data egress the agent needs; the
  local REST `serve` port bound to loopback, not exposed.

## Open questions

- **Auto-lock vs. uptime:** can a short `--auto-lock` coexist with truly unattended week-long
  signing, or does it force a persistent unlock credential on the host? (Gated by
  [[Security and Encryption]].)
- **`twak serve` flag set:** the mirrored docs describe `--watch` / `--watch-interval` in
  prose but list only `--rest`/`--port`/`--host`/`--auto-lock`/`--password` in the flag
  table — confirm `--watch` is accepted by `serve` directly (vs. a separate command) against
  the live CLI.
- **MCP vs. REST for the runner:** does the loop drive TWAK via the `serve` MCP (stdio) or the
  local `--rest` API? Decide alongside whether `execute_trade` wraps the `twak` CLI or the
  TWAK MCP ([[MCP Server]] open item).
- **Home vs. VPS for custody:** is the custody benefit of a physically-controlled home box
  worth its weaker uptime, given a UPS + ISP-failover budget? Decide with
  [[onchain-custody-engineer]].
