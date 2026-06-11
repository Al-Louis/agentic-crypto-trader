# TWAK Spike Runbook

The ordered, checkpointed procedure for the Phase-2 custody slice: **a guardrail-checked dust
trade signed via TWAK with local self-custody keys, confirmed on BSC with a real tx hash**
(the June 16 Track-1 PoC gate — see [[Tech Stack]] §Phase 2). Custody discipline per
[[Security and Encryption]]: the agent never sees the mnemonic, wallet password, or HMAC
secret — every secret-bearing step is a **USER-ACTION** checkpoint; the agent verifies state
afterward with read-only calls.

**Hard spike ceiling: $10 total (gas + dust), throwaway wallet, never the competition wallet.**

## Verified keylessly on the dev laptop (2026-06-11)

| Fact | Evidence |
|---|---|
| Host = **native Windows 11** (decided) | WSL is **not installed** (`wsl.exe -l -v` → "not installed"); TWAK runs natively; installing WSL adds time/risk and no custody gain |
| Node v22.20.0 / npm 10.9.3 | preinstalled |
| `@trustwallet/cli` **v0.19.0** installed globally | `twak --version` → `0.19.0` |
| `twak compete register` / `compete status` exist | both take `--password` (keychain/`TWAK_WALLET_PASSWORD` fallback), `--json`; status also reports the registration deadline |
| **`twak erc8004` is native in the CLI** (register/set-uri/set-metadata/show; `--chain bsc` default, `bsctestnet` known) | wallet-unification implication → [[Security and Encryption]] |
| BSC supported: chain key `bsc`, coinId `20000714` | `twak chains --json` |
| **OS keychain works on Windows** — backend is `@napi-rs/keyring` → **Windows Credential Manager** | dummy round-trip verified: `wallet keychain save` → `check` ("stored") → `delete` → `check` ("no password"); dummy removed |
| `wallet create` **saves the password to the keychain by default** (`--no-keychain` to opt out) | `twak wallet create --help` |
| Everything fails **closed** without credentials | `price`, `compete status`, `serve` all exit with `No API credentials found. Run \`twak setup\` (or set TWAK_ACCESS_ID and TWAK_HMAC_SECRET env vars).` |
| API-credential env names are **`TWAK_ACCESS_ID` / `TWAK_HMAC_SECRET`** | CLI error text + `twak init --help` ("prefer ... env var"); `.env.example` corrected to match |
| Telemetry **off by default** | `twak telemetry` → "disabled (default)" |
| `twak swap` has `--usd <amount>` (swap a USD-equivalent) + `--quote-only` + `--slippage` | `twak swap --help` — `--usd 1` is the natural dust-trade shape |
| Standalone `twak watch` exists (alerts + automations watcher, `--dry-run`) alongside `serve --watch` | `twak watch --help` |

No wallet, no credentials, no `~/.twak` exist yet on this machine — clean slate.

## The runbook

Order is deliberate: **signing proof first** (sign-message needs zero funds and is the step
that can reveal an uncodeable blocker), **guardrails before any funded trade**.

> **Status 2026-06-11 — Steps 0–4 DONE/VERIFIED.** Signing proof landed keychain-resolved and
> unattended: `twak wallet sign-message --chain bsc --message "act-spike-2026-06-11"` →
> signature `0x3be15c12a46cb5dd…1b`, digest
> `0xbac00e5ae7f96a1c883cffbfb7345b56b37bbe125ed92769ea3502f26633196b`. Spike wallet
> `0x2C195BaCbE0d6333da9CD8aa3BeEc7340FE7D32E` (in `.env` as `AGENT_WALLET_ADDRESS`; address
> not secret), balance 0 BNB as designed. Guardrail skeleton implemented + tested (Step 4
> notes below). **Next: Step 5 — USER-ACTION funding (≤ $8).**

### Step 0 — done (agent)
CLI installed and surface verified (table above).

### Step 1 — USER-ACTION: portal credentials
1. Sign in / sign up at **https://portal.trustwallet.com** (dashboard → apps) and create an
   API key. You get an **Access ID** and an **HMAC secret** (shown once — store it in your
   password manager, nowhere else).
2. In a terminal **you** run (interactive prompts, so the secret never enters shell history):
   ```
   twak setup
   ```
   Do the **credentials** phase. You may skip the harness phase; if it offers the **wallet**
   phase you can do Step 3 in the same sitting (see below). Alternative non-interactive form:
   set `TWAK_ACCESS_ID` / `TWAK_HMAC_SECRET` for the session and run `twak init`.
   Credentials land in `~/.twak/credentials.json` (keep it `0600`-equivalent; it's outside
   the repo).

**Agent verifies (read-only):** `twak auth status` → `Status: configured`;
`twak price BNB --json` returns a price (proves HMAC auth works end-to-end).

### Step 2 — USER-ACTION: create the throwaway spike wallet
You run (interactive via `twak setup --wallet`, or directly — note a `--password` flag enters
shell history, so prefer the setup prompt):
```
twak setup --wallet
```
- Use a **fresh password** (≥8 chars, upper+lower+digit), not one you use elsewhere.
- **Accept the default keychain save** — on this machine that is Windows Credential Manager
  (verified). This is what lets the agent drive signing later without ever seeing the password.
- The mnemonic shown at creation: this is a **throwaway** wallet (≤$10 ever), so a password-
  manager note is sufficient backup. Never paste it anywhere else.

**Agent verifies (read-only):** `twak wallet status` → `Wallet: configured`,
`Keychain: password stored`; `twak wallet keychain check`.

### Step 3 — signing proof, zero funds (agent, password via keychain)
The first signature — before any money exists:
```
twak wallet address --chain bsc --json     # password resolves from keychain (derivation is gated)
twak wallet sign-message --chain bsc --message "act-spike-2026-06-11" --json
```
Expected: a `0x…` BSC address and a signature hex. **This closes blocker #1 in miniature** —
unattended local signing with custody on this box, no human in the signing path. If either
fails with an auth error, the keychain save didn't take; re-run Step 2's keychain phase.

Record the address (it is not a secret) in `.env` as `AGENT_WALLET_ADDRESS=` for the
guardrail/monitoring code.

### Step 4 — guardrails in code (agent) — BEFORE funding — **DONE 2026-06-11**
Implement and test `src/trader/risk/` + the `execute_trade` wrapper per the spec below.
Done = pytest green on the refusal matrix (every cap refuses; in-policy intent passes) with
the **spike policy** values pinned: allowlist `{BNB, USDT-BSC}`, per-trade ≤ $2, daily ≤ $6,
slippage ≤ 1%, **lifetime spike ceiling $10**, drawdown stop 30%.

**Done:** all modules landed per spec; full suite **324 passed** (43 new: refusal matrix,
ledger persistence, quote parsing vs captured fixtures, mocked-swap end-to-end). Spec
deviations, all minor (rationale inline in code):
- **Allowlist stores bare symbols `{BNB, USDT}`**, not `USDT-BSC` — the CLI and its quotes
  speak symbols; the "-BSC" qualifier is carried by the pinned `chain="bsc"` (any other
  chain refuses `CHAIN_NOT_BSC`, so a non-BSC USDT cannot slip through on symbol alone).
- **Quote JSON has NO USD field** (observed live): even with `--json`, stdout is a human
  line (`$1 USD ≈ 0.001652… BNB (@ $605.13…)`) *then* JSON
  `{input, output, minReceived ("<amt> <SYM>" strings), provider, priceImpact}`. USD is
  derived as the conservative **max** of (input leg × the `(@ $price)` prefix, stable legs
  at $1); unvaluable ⇒ refuse `STATE_UNAVAILABLE`. Implied slippage =
  `1 − minReceived/output` (verified: exactly the requested tolerance at 1% and 0.5%).
- **Return contract gained a third arm**: `{tx_hash}` on success, `{refused, detail, phase}`
  on policy refusal, and `{error, tx_hash: None}` when a trade *passed* checks but the
  swap/confirm failed — that is a failure, not a refusal, and the attempt's spend stays
  counted (conservative: an unknown outcome may have moved money).
- Refusal rows are appended to the ledger for the audit trail (count zero spend); the
  attempt row must land on disk **before** signing or the trade refuses.
- `data/*.jsonl` added to `.gitignore` (the ledger wasn't covered by existing patterns).
- The drawdown stop halts **at** ≥30% below high-water (the DQ gate line); spend caps are
  inclusive ("spend up to": exactly $2.00 passes).
- The swap-execution and `tx` output shapes are **unknown until Step 6** (unfunded wallet);
  `extract_tx_hash`/`parse_tx_status` are tolerant across common key/status shapes and fail
  closed (no hash ⇒ error + `status: unknown` row, never a silent pass).

### Step 5 — USER-ACTION: fund the spike wallet (≤ $8) — **DONE 2026-06-11**
Send **at most $8 worth of BNB** (e.g. from your main wallet/exchange) to the Step-3 BSC
address. $8 keeps headroom under the $10 ceiling for gas drift. Double-check the address
byte-for-byte; BSC = BEP-20/`BNB Smart Chain` network.

**Agent verifies (read-only):** `twak wallet balance --chain bsc --json` shows the BNB;
cross-check on `https://bscscan.com/address/<addr>` and compute USD via `twak price BNB`.
Confirm total received ≤ $8 — if more arrived, **stop** and flag.

**Done:** user funded **$2.00** (0.003317 BNB) from Coinbase — under the planned $8 because
of a Coinbase transfer timelock; ample for the $1 dust trade + gas. Verified ≤ $8 ✓.

### Step 6 — dust trade through the guardrails (agent) — **DONE 2026-06-11 — THE GATE ARTIFACT**

**The June-16 PoC gate is satisfied.** A guardrail-checked $1 BNB→USDT swap was signed via
TWAK (password keychain-resolved, unattended) through `execute_trade` and confirmed on BSC:

- **tx hash:** `0x739bb1516c99e56237c7585a449455d90a7f0b027ef9f252a5275b67e4757c96`
  (`twak tx … --json` → `"confirmed": true, "failed": false`; explicit user go-ahead given)
- Realized quote: 0.001658678 BNB → 0.9933 USDT, provider `Native`, implied slippage 1.0%,
  price impact 0; post-trade balance shows **0.993388 USDT** received, ~$1 BNB remaining.
- Rug-gate read passed first: USDT-BSC canonical contract verified via `twak asset`, `twak
  risk` → `riskLevel: low`, Blockaid-audited, `supportsSwap: true`.
- **Negative proof:** a $5 intent against the live ledger state refuses
  `PER_TRADE_CAP: $5.00 > $2.00` (pure `check_trade`; the harness's permission layer blocked
  re-running the full `execute_trade` path with an over-cap amount, which is itself
  defense-in-depth). The ledger shows `spent_today_usd = spent_lifetime_usd = $0.9933` —
  caps are reading persisted real spend.
- **Bug found & fixed by the live run:** `twak tx --json` returns *boolean* fields
  (`{"confirmed": true, "pending": false, "failed": false}`), not a `status` string —
  `parse_tx_status` polled "pending" forever on an already-confirmed tx. Fixed (booleans
  checked first, `failed` wins conservatively) + regression test; suite **325 passed**.

Original procedure (as executed):
1. **Resolve the pair**: `twak search USDT --json` / `twak asset c20000714_t0x55d398326f99059fF775485246999027B3197955 --json`
   (USDT-BSC; verify the contract via the asset lookup, never trust a hardcode blindly), and
   `twak risk <assetId> --json` as the rug-gate read.
2. **Quote first** (read-only): `twak swap --usd 1 BNB USDT --chain bsc --quote-only --json`.
   `execute_trade` re-checks guardrails against the *quote* (realized USD, route, slippage).
3. **Execute** via `execute_trade` (which shells `twak swap --usd 1 BNB USDT --chain bsc --slippage 1 --json`,
   password from keychain). Capture the **tx hash**.
4. **Confirm**: `twak tx <hash> --chain bsc --json` until success; record hash + gas in the
   risk ledger; check on bscscan.com. **This is the June-16 gate artifact.**
5. Negative proof: attempt an out-of-policy trade (e.g. $5 > per-trade cap) and record the
   refusal — the demo that guardrails are code, not prompts.

### Step 7 — registration dry-run (agent, read-only on the spike wallet) — **DONE 2026-06-11**

**Done:** `twak compete status --json` → `{"registered": false, "open": true, "chain": "bsc",
"opensAt": "2026-06-02T21:15Z", "deadline": "2026-06-25T00:00Z"}`. **The on-chain
registration deadline is June 25** — three days later than the June 22 the vault assumed
(June 22 remains our working deadline since the scored live window starts then). `register
--help` surface confirmed (`--password`/`--json` only). `compete register` NOT run — deferred
to the final unified wallet per below.

```
twak compete status --json
```
Reads registration state + the deadline for *this* wallet. **Do NOT run `twak compete register`
on the throwaway wallet** — registration binds an agent wallet address to the competition and
the registered address must be the final, unified competition wallet
([[Security and Encryption]] §wallet unification). The dry-run deliverable is: `status` output
captured, `register --help` surface confirmed (done — only `--password`/`--json`), and the
unification probe (`twak erc8004 show/register` path) scheduled before June 22.

**Unification probe DONE 2026-06-11** ([[Security and Encryption]] §wallet unification):
ERC-8004 **agentId 1369** minted on `bsctestnet` from this wallet (faucet tBNB, keychain
signing, tx `0xb03cdd…48db7`); `erc8004 show` confirms `owner`/`agentWallet` == the trading
address. One `~/.twak` store covers trading + registration + identity, zero key export.
Gotcha: `--uri` is **required** on `erc8004 register` — prepare the agent-card URI before the
mainnet mint on the competition wallet.

### Step 8 — `--auto-lock` empirical check (agent, free) — **DONE 2026-06-11**

**Done:** ran `twak serve --auto-lock 1` for >2 minutes (idle, no signing), then
`twak wallet sign-message` from a fresh process → signature returned with **no human step**
(password re-resolved from Windows Credential Manager). Honest scope of the proof: each
`twak` CLI invocation is its own process that starts *locked* and unlocks from the keychain —
so our trade path (own loop shelling the CLI per call, per [[Security and Encryption]])
re-unlocks transparently on **every** call, demonstrated repeatedly today (dust trade,
erc8004 mint, this check). The serve process's *internal* relock→re-unlock wasn't observable
from outside (it logs nothing on lock), but `serve` is not in the trade path, and its
auto-lock did not interfere with other processes' signing.
With the wallet idle-locked (`twak serve --auto-lock 1` for a couple of minutes, or just a
later session), run `twak wallet sign-message …` again: confirm re-unlock is transparent via
keychain resolution with no human step. Feeds the open question in
[[Security and Encryption]] §open questions.

## Guardrail skeleton spec — `src/trader/risk/` + `src/trader/execution/`

Follows the repo conventions (pure, unit-testable cores; thin wrappers; refusal dicts like the
MCP tools' `{"refused": …}`; see `src/trader/mcp_server/server.py`). Out-of-policy calls are
**refused with a coded reason, never negotiated, never auto-adjusted**. Any error computing
state (ledger unreadable, quote missing fields) ⇒ refuse (fail closed).

```
src/trader/risk/
├── __init__.py      # exports: Policy, TradeIntent, Verdict, check_trade, SPIKE_POLICY
├── policy.py        # frozen dataclass Policy: allowlist (set of asset ids), per_trade_usd,
│                    #   daily_usd, max_slippage_pct, drawdown_stop_pct, lifetime_usd_ceiling,
│                    #   chain ("bsc" pinned). SPIKE_POLICY = Policy(allowlist={BNB, USDT-BSC},
│                    #   per_trade_usd=2, daily_usd=6, max_slippage_pct=1.0,
│                    #   drawdown_stop_pct=30.0, lifetime_usd_ceiling=10.0)
├── ledger.py        # append-only JSONL at data/risk_ledger.jsonl (git-ignored): one row per
│                    #   ATTEMPT and per RESULT (ts, intent, verdict, tx_hash, usd, gas_usd).
│                    #   Derives: spent_today(), spent_lifetime() (gas + notional, conservative
│                    #   — both legs counted), equity high-water mark for the drawdown stop.
│                    #   Survives restarts; the caps are meaningless if state lives in memory.
└── checks.py        # pure: check_trade(policy, intent, state) -> Verdict.
                     #   Refusal codes: CHAIN_NOT_BSC, NOT_ALLOWLISTED, PER_TRADE_CAP,
                     #   DAILY_CAP, LIFETIME_CEILING, SLIPPAGE_BOUND, DRAWDOWN_STOP,
                     #   STATE_UNAVAILABLE. Verdict = {allowed: bool, refusals: [codes+detail]}.

src/trader/execution/
├── twak_cli.py      # subprocess wrapper over the twak CLI with --json: quote(), swap(),
│                    #   tx_status(), wallet_balance(). NEVER passes --password (resolution =
│                    #   OS keychain / TWAK_WALLET_PASSWORD on the host); never logs argv or
│                    #   env; timeouts on every call.
└── execute.py       # execute_trade(intent, policy=SPIKE_POLICY) -> result dict:
                     #   1) check_trade on the intent            -> refuse early
                     #   2) quote (--quote-only)                 -> re-check on REALIZED quote
                     #      (usd value, route tokens, implied slippage) — the quote is the
                     #      truth, the intent is a wish
                     #   3) ledger.append(attempt)
                     #   4) twak swap (TWAK natives layered on top: --slippage from policy;
                     #      for transfers: --max-usd, --confirm-to)
                     #   5) poll tx_status -> confirmed/failed; ledger.append(result)
                     #   returns {"tx_hash": …} or {"refused": [codes], "detail": …}
```

Two-phase check (intent, then quote) is the load-bearing design: slippage and true USD size
are only knowable from the quote, and TWAK's own flags are belt-and-suspenders *under* our
checks, never instead of them. The drawdown stop and daily cap read the ledger, so a crashed
and restarted loop stays capped. `simulate_trade`/`execute_trade` MCP tools ([[MCP Server]])
are thin wrappers over `check_trade` + `execute_trade` later — coordinate the exact intent
shape with `principal-engineer` before the agent loop consumes it.

## Links

Owned by [[Security and Encryption]] (custody model, host design, open questions);
gate definition in [[Tech Stack]] §Phase 2; tool surface in [[MCP Server]].
