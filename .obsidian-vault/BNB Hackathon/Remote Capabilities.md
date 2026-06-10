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
| Resource shape | Low CPU/RAM, network-reliable, must not miss a tick | **Many CPU cores + RAM** (env-stepping-bound, not GPU-bound), interruptible |
| Custody posture | **Holds signing keys** — custody-sensitive | **No keys** — pure compute on recorded data |
| Failure mode | Missed day → forfeit; key exposure → loss of funds | Wasted compute; re-runnable |

These have opposite security and uptime profiles. **Run them on different hosts.** The
training box never sees a key; the trading box never runs the experimental training code.
Conflating them puts keys next to the most volatile code in the project — avoid it.

> **Training is CPU-bound, not GPU-bound.** This RL workload (small MLP/attention policies on
> engineered features) is bottlenecked on **environment stepping**, not the policy's
> forward/backward pass — confirmed across the prior [[TradeSim]] runs. The win comes from
> **env parallelism across CPU cores** (vectorized / `SubprocVecEnv`, `n_envs ≈ physical
> cores`), so the desktop is chosen for **core count + RAM**, and torch installs **CPU-only**
> (no CUDA toolkit/driver matching — a simpler, faster setup). GPU is revisited *only* if
> profiling ever shows the policy pass dominating (a large extractor with big minibatches) —
> not the default here.

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
tools**, not interactive jobs — so it runs unattended on the training host and is
workflow-drivable from the principal host.

- `start_training` (config → run id) spawns the run; `training_status` (run id → metrics)
  polls progress. The host fires-and-polls rather than blocking. **As-built:**
  `remote_train.submit_background` launches the job **detached** (`nohup` over SSH) and returns
  immediately; `poll` reads the job's `progress.json` (terminal `state` wins) with a `kill -0`
  liveness fallback — so an hours-long RL run never blocks the orchestrator, and the desktop
  self-publishes (no haul-back).
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

### Apentic pipeline — as-built (2026-06-08)

The training→telemetry pipeline is built **pipeline-first** and proven end-to-end *locally*
before the desktop exists. Three tiers: **laptop** (dev + orchestration, all dev stays here),
**desktop** (CPU/core-parallel training host — no keys; env-stepping-bound, torch CPU-only),
**Apentic frontend** (`alexlouis-site`, reads static JSON). Two cleanly separated code layers:

- **`remote_train/`** — a **generic, trading-agnostic** job orchestrator (its own package,
  `src/remote_train/`, in the wheel separately). `JobSpec` → `submit` → `status` (fire-and-poll
  via on-disk `status.json` + `progress.json`) → `publish`. Pluggable executors: **`LocalExecutor`**
  (now / CI) and **`SSHExecutor`** (the desktop over Tailscale — runs the command, streams the
  artifact dir back as a **tar over ssh**, since Windows OpenSSH has no rsync and scp mis-parses
  `C:\` targets). The desktop is **`root@act-trainer`**, repo at `/root/agentic-crypto-trader`;
  `scripts/dispatch_demo.py` dispatches there by default (`--local` to run on the laptop).
  **Hard rule, test-enforced: `remote_train` must never `import trader`** — so it lifts
  into its own repo after the hackathon (decouple-now, extract-after-a-second-use — *not* a
  premature separate repo today).
- **`trader.report.export_run`** — the **trading-specific** bridge to the dashboard contract.
  The frontend reads, per run, a `manifest.json` + `trades.json` (`RoundTrip[]`) / `metrics.json`
  (`MetricsReport`) / `candles.json` (`CandleData[]`) / `equity_curve.json` (`EquityPoint[]`) /
  `run_info.json`, from `PUBLIC_APENTIC_DATA`. `roundtrips_from_position` folds any single-asset
  exposure series (heuristic now, RL later) into cost-honest round-trips.

**Decisions locked:** dispatch = **SSH over Tailscale**; sequencing = **pipeline-first**;
**Telemetry seam:** the job appends `progress.json` (reward/return curve); `status()` and the
dashboard poll the same flat file.

**Publishing — revised to AWS, self-from-desktop (2026-06-08).** Two findings reshaped this:
1. **The bundle haul-back is fragile** — a **path-MTU black hole** on the freshly-revived
   tailnet (≤~512 B returns work, ≥4 KB stall and the ssh session dies). So we **stopped
   pulling artifacts back**: `JobSpec.fetch_artifacts=False` and the **job self-publishes**
   (`trader.report.publish_run`) straight to the target. For remote runs the desktop uploads
   over **its own internet**, so nothing large crosses the tailnet — the laptop only sends the
   tiny ssh trigger + reads tiny `progress`/`status`.
2. **Drop Cloudflare R2 → reuse the site's AWS infra.** `alexlouis.dev` is already **static →
   S3 (`alexlouis-site-web`, private/OAC) + CloudFront `ESSV4WVWKTQ9F`**, deployed by GitHub
   OIDC. Decision: a **separate data bucket** (`alexlouis-apentic-data`) served at **root** by
   its **own CloudFront distribution** on a dedicated subdomain **`data.alexlouis.dev`**
   (`PUBLIC_APENTIC_DATA=https://data.alexlouis.dev`). Chosen over the apex-path
   (`alexlouis.dev/apentic/data`) for **clean error semantics** — the site distribution's
   custom error responses (`403/404 → /index.html`) are distribution-wide, so on the apex a
   *missing* data object returns the SPA HTML with 200 instead of a clean 404; a dedicated
   distribution avoids that and isolates caching/WAF/logging. Cross-origin is handled by
   CloudFront's managed **`SimpleCORS`** response-headers policy (the browser only talks to
   CloudFront; S3 needs no CORS). `api.` is reserved for a future *dynamic* API; static
   bundles live on `data.`. Naturally **isolated from the site's `s3 sync --delete`** (separate
   bucket). Freshness: **CloudFront invalidation of `/*` on every publish** (the `remote`
   extra's boto3 path; immutable cache on per-run files, short on `manifest.json`). Desktop
   creds = a **scoped IAM user** (`PutObject`/`GetObject` on the data bucket + `CreateInvalidation`
   on the data distribution) in the desktop `.env` — *not* the GitHub OIDC role. The publish
   code is cloud-agnostic (plain S3 API), so AWS works unchanged from the R2 design.

**Open fork (deferred):** the frontend contract is **single-asset** (entry/exit round-trips on
one symbol). Our live strategy is **cross-sectional portfolio**. The demo uses a single-asset
trend heuristic to exercise every panel; the trained agent's shape (single-asset entry/exit RL
that fits the frontend vs. portfolio allocator that needs a new view) is decided with
[[rl-ml-trainer]] — the exporter and pipeline are identical either way. The MCP `start_training`
/ `training_status` / `export_run` tools ([[MCP Server]]) become thin wrappers over `remote_train`.

### Desktop host — as-provisioned (2026-06-08)

The desktop is now stood up. **8 physical / 16 logical cores, 32 GB RAM** — sized for
`n_envs ≈ 8` on this env-stepping-bound workload.

**Decision: the training host runs inside WSL2 (Ubuntu-24.04), not native Windows.** Two
reasons the original "just SSH into the desktop" framing missed: (1) `SSHExecutor` builds a
**POSIX** remote command (`mkdir -p`, `shlex.quote` single-quoting, `git fetch && checkout`)
and pulls artifacts with **`rsync`** — native Windows OpenSSH defaults to a `cmd.exe` shell
where that quoting and `mkdir -p` break and `rsync` is absent; (2) the Windows-side Python is
3.14, which has **no torch wheel** yet. WSL2 resolves both: inside Ubuntu-24.04 there is
**systemd (pid 1), Python 3.12, rsync, and tailscaled** already present, so `SSHExecutor`
runs unmodified and CPU-only torch installs cleanly.

**Setup shape (keyless host):**
- Repo cloned into the **Linux FS** at `/root/agentic-crypto-trader` (= `~`; *not* under
  `/mnt/...` — cross-OS file access throttles git/rsync/env-stepping). This is
  `SSHExecutor.remote_workdir`. **Update gotcha:** this clone's git `origin` is a stale-prone
  **P:-drive mirror** (`/mnt/p/Development/Projects/agentic-crypto-trader`), not GitHub — to
  update it: pull the P: mirror from GitHub (it has auth), then `git -C /root/… pull origin main`.
- venv: `pip install -e ".[data,dev,remote,training]"` + the **CPU torch wheel**
  (`--index-url https://download.pytorch.org/whl/cpu`). `remote` (boto3) self-publishes from
  here; `training` (gymnasium + stable-baselines3 + sb3-contrib) trains PPO here. Desktop
  `.env`: scoped AWS creds + `APENTIC_PUBLISH_TARGET=s3://alexlouis-apentic-data` +
  `APENTIC_CLOUDFRONT_DIST_ID=E14F268NIY6WLZ` (the dedicated `data.alexlouis.dev` distribution).
- Reachability: **Tailscale SSH** (`tailscale up --ssh`) chosen over installing
  `openssh-server` — tailscaled terminates the session (identity-based, no key files), works
  under WSL2 userspace networking, and the artifact bundle streams back as a **tar over the
  same `ssh` transport** (no rsync — see the as-built note above; this is also why the
  orchestrator can be plain Windows). The classic `sshd` + key-auth path would additionally
  have to solve WSL2 inbound forwarding.
- Dispatch: `scripts/dispatch_demo.py` now **defaults to SSH** dispatch to the desktop
  (Tailscale IP `root@100.97.195.65`, `/root/agentic-crypto-trader`); pass `--local` to run on
  the laptop. The job self-publishes; remaining step is the **AWS data bucket + CloudFront
  behavior + IAM user** (publishing revision above) and the desktop `.env`.

Custody posture holds: **no `.env` mnemonic, no `twak serve`, no execute-tier tools on this
box** — its only outputs are model artifacts and reports ([[Security and Encryption]]).

**Verified end-to-end (2026-06-08):** torch 2.12.0+cpu (cuda False), sb3/sb3_contrib 2.8.0,
gymnasium 1.2.3; **122 tests pass**; `remote_train` imports pull in zero `trader` modules
(decoupling holds at runtime); the exact dispatch entrypoint runs on the trainer and emits the
full dashboard bundle against real data.

### Operational gotchas — learned standing it up (2026-06-08)

The host is `100.97.195.65` / `act-trainer.tail7214b2.ts.net` (tailnet `al-louis.github`),
user `root`, repo `/root/agentic-crypto-trader`. Five things bit us; record them so they don't again:

- **WSL idles the distro out → tailscaled dies → node drops off the tailnet.** A distro that
  nothing holds open is shut down after the idle timeout, killing its systemd services. Fix: a
  keep-alive pin (`wsl -d act-trainer -u root -e sleep infinity`) registered as a **logon
  scheduled task `act-trainer-keepalive`** so the trainer auto-starts and stays reachable
  across reboots. Host sleep still pauses it (acceptable for a burst training box).
- **Tailnet name doesn't resolve from the Windows laptop.** Even with MagicDNS on, Windows
  skips the tailnet search domain, so bare `act-trainer` fails (`could not resolve hostname`).
  Use the **IP `100.97.195.65`** or the **FQDN** `act-trainer.tail7214b2.ts.net`.
- **The repo is private + the headless distro has no GitHub creds** → an HTTPS `git clone`
  hangs forever on a credential prompt. Cloned instead from the **local Windows working copy
  at `/mnt/p/...`** (origin points there; auth-free). `GIT_TERMINAL_PROMPT=0` set so git fails
  fast instead of hanging. Update path: `git pull` the Windows clone, then pull on the trainer.
- **Market data is gitignored** (the repo is ~1 MB) and lives only on the laptop, so it must be
  copied over: `scp -r data root@100.97.195.65:/root/agentic-crypto-trader/` (102 MB; the job
  hard-requires hourly OHLCV under `data/ohlcv/hour_1/`).
- **WSL2 clock skew was the root cause of Tailscale login not holding** ("appears in the admin
  console but `tailscale status` says Logged out"). `systemd-timesyncd` (enabled via
  `/etc/wsl.conf [boot] systemd=true`) keeps the clock synced and the session stable.

### Remote training — deploy runbook (2026-06-09)

The end-to-end procedure for launching a training sweep on the desktop, with the failure modes
that cost a full day (and a forced reboot) to learn. **CLAUDE.md carries the five-rule short form;
this is the complete version.** Host facts: `root@100.97.195.65`
(`act-trainer.tail7214b2.ts.net`), repo `/root/agentic-crypto-trader`, `.venv/bin/python`,
self-publishes to `data.alexlouis.dev` (`APENTIC_PUBLISH_TARGET=s3://alexlouis-apentic-data` +
`APENTIC_CLOUDFRONT_DIST_ID=E14F268NIY6WLZ` in the desktop `.env`). 8 physical / 16 logical cores →
`--n-envs 8`, runs **sequential**.

**Step by step:**

1. **Build + validate locally**, then **commit + push to GitHub** — the pure-numpy env runs on the
   laptop, so validate obs width / causality / tests before spending desktop hours.
2. **SSH via the PowerShell tool only** (failure mode #1 below). Sync + preflight in one
   small-output command: `cd /root/agentic-crypto-trader && git fetch -q && git checkout -q <sha>
   && git rev-parse --short HEAD && ls data/ohlcv/hour_1 | wc -l`, plus a one-line
   `build_volume_panel` smoke. Confirm HEAD == the pushed sha.
3. **Sweep script** follows `scripts/run_reward_sweep.sh`: loop seeds/configs → `train_rl.py`, one
   bundle per run, per-run logs in `runs-rl/<sweep>-logs/`, run-ids `ppo-<tag>-s<seed>`. The script
   **sequences** the runs (each finishes before the next starts) — that IS the design. **Never
   launch it more than once, and never background the seeds in parallel.**
4. **Launch ONCE, detached:** `cd /root/agentic-crypto-trader; mkdir -p runs-rl
   runs-rl/<sweep>-logs; nohup bash scripts/<sweep>.sh <TIMESTEPS> "<SEEDS>" >
   runs-rl/<sweep>.log 2>&1 < /dev/null &`.
5. **WAIT 60–90 s, then verify with tiny output:** exactly one driver bash + one `train_rl` main,
   the log's last line, and `cut -d' ' -f1 /proc/loadavg` (load should sit near `--n-envs` ≈ 8; if
   it's ~16+, duplicates are stacked — stop and kill the extras by **specific PID**, not broad pkill).
6. **Monitor** via each run's `progress.json` (terminal `state` wins) or the published bundles;
   **aggregate** with `scripts/compare_seeds.py` (seed average) / `scripts/compare_sweep.py` (reward
   modes), pulling `metrics.json` from `data.alexlouis.dev`.

**The five failure modes (each cost real time):**

- **The Bash-tool `ssh` can't reach the tailnet — it hangs, never times out cleanly.** Windows
  OpenSSH (the PowerShell tool) uses the Windows network stack, which routes. *Always drive SSH from
  PowerShell.* Diagnose reachability with `Test-NetConnection -ComputerName 100.97.195.65 -Port 22`
  (`TcpTestSucceeded:True` = up; `PingSucceeded:False` is just ICMP being firewalled — ignore it).
- **Path-MTU black hole on the tailnet:** replies ≥ ~4 KB stall and the session dies (same root
  cause as why artifacts aren't hauled back). Keep every status command's *output* tiny — counts and
  a single `tail -1`, never `pgrep -fa` / full `ps` listings.
- **Slow startup masquerades as a dead launch:** torch import + `build_volume_panel` (`--rung0-obs`)
  take ~30–60 s before any process or log line exists. **Do not conclude failure from a fast check
  and relaunch** — stacked parallel sweeps oversubscribe the WSL2 VM, spike **Vmmem** on the Windows
  host, throttle the whole machine, and force a reboot. (This bit us 2026-06-09.)
- **`runs-rl/` is gitignored** → absent on a fresh checkout → the `> runs-rl/<sweep>.log` redirect
  fails silently and the launch looks dead. `mkdir -p runs-rl runs-rl/<sweep>-logs` first.
- **Self-matching pkill/pgrep:** your SSH command line contains the very patterns you search for, so
  a naive `pgrep -f`/`pkill -f` matches your own probe (and the lingering tailscaled ssh wrappers).
  Use the bracket trick (`'[r]un_…'`) so a count/kill can't match itself, and prefer killing by
  **specific PID** over a broad `pkill -9` (correctly blocked as too blunt on a shared host).

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
