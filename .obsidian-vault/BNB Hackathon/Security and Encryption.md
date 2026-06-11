# Security and Encryption

How the agent holds keys, signs unattended, and stays self-custodial through the whole trade
loop. This is the project's **#1 blocker** (autonomous signing while custody stays local) and
a directly scored criterion. See [[Project Overview]] for scope, [[Tech Stack]] for the four
surfaces, [[Remote Capabilities]] for *where* the host runs, [[Trading Strategies]] for the
guardrail *values*, and [[Real-time Monitoring]] for live tx/PnL watching.

## The self-custody model (TWAK)

TWAK generates and stores keys locally; **keys never leave the device** — not to Trust Wallet
servers, not to the model provider. On `twak wallet create`, a BIP39 HD mnemonic is generated
via Trust Wallet Core, encrypted immediately, and written to `~/.twak/wallet.json`. The
mnemonic is never stored in plaintext.

| Field in `wallet.json` | What it is |
|---|---|
| `encryptedMnemonic` | **AES-256-GCM** ciphertext of the mnemonic |
| `iv` | random initialization vector |
| `authTag` | GCM authentication tag (detects tampering on decrypt) |
| `salt` | random **PBKDF2** salt; the AES key is PBKDF2-derived from the wallet password |
| `createdAt`, `chains` | timestamp, supported chain keys |

No plaintext key, mnemonic, or password ever touches disk. The encryption key is reconstructed
in memory from the password each time a signing operation runs. A single wallet derives
addresses for 25+ chains via BIP-44; the EVM address is shared across all EVM networks
(including BSC) — relevant to wallet unification below.

## Read vs. sign — the capability boundary

TWAK enforces a hard split between read-only queries and signing. **Without** the password the
agent can: query any balance, search tokens, fetch prices, get swap quotes, validate addresses,
check token rug-risk, and view history — everything the decision core needs to *read*. The
password is **required** to: derive/reveal addresses, sign or send transactions, execute swaps,
ERC-20 `approve`, and sign messages.

The model never has direct access to the mnemonic or private keys — it interacts through TWAK's
**action layer**, which gates all signing behind password auth. This maps cleanly onto our risk
posture: the decision core runs unprivileged on read tools; only the guarded `execute_trade`
code path supplies the credential that unlocks signing.

## Password resolution & autonomous signing (resolves blocker #1)

When a signing op is requested, TWAK tries each source in order and uses the first available:

| Order | Source | Posture |
|---|---|---|
| 1 | `--password` flag | **Avoid** — leaks into shell history / process args. |
| 2 | `TWAK_WALLET_PASSWORD` env var | For CI/containers; acceptable on a hardened host. |
| 3 | **OS keychain** | Most secure; set via `twak wallet keychain save` (and `wallet create` saves there **by default**; `--no-keychain` opts out). |

None available → authentication error (signing simply fails closed — **empirically confirmed
2026-06-11**: every credential-needing command, including `serve`, hard-errors without setup).

**The keychain is cross-platform, including Windows.** The CLI's backend is
`@napi-rs/keyring` (Rust keyring bindings), which targets macOS Keychain, Linux Secret
Service, **and Windows Credential Manager**. Verified on the Windows 11 dev laptop with CLI
v0.19.0 via a dummy save→check→delete round-trip — the docs' "macOS/Linux" framing
undersells it. On a Linux always-on host the Secret Service path is preferred; the env-var
path is the container fallback. Either way the password unlocks the *local* `wallet.json` —
custody never leaves the box.

**The unattended-signer mechanism.** A keychain- or env-resolved password plus
`twak serve --watch` runs the signing/automation loop in the background, so the agent can sign
for the full June 22–28 window hands-off. Relevant knobs: `--watch-interval` (poll cadence,
default 60s), `--auto-lock <minutes>` (re-lock the wallet after inactivity). The background
runner uses the *local* agent wallet; if you connected via WalletConnect instead it stays idle
(those need manual approval) — so the local-wallet path is the one that makes hands-off real.
This is the design that closes blocker #1. The host/key-on-remote-box logistics are owned by
[[Remote Capabilities]].

## Guardrails wrap the signing call

Per [[Project Overview]] and CLAUDE.md, **guardrails are hard, external code limits around the
TWAK signing call — not model suggestions**: token allowlist, per-trade and daily caps, slippage
bound, drawdown stop. They live in `src/trader/risk/` and gate `execute_trade`; out-of-policy
calls are *refused, not negotiated*. TWAK offers some native belt-and-suspenders checks we layer
on top, never instead of, our own: `transfer --max-usd` / `--confirm-to` (pin the resolved
payee), `swap --slippage` (max 50, default 1) and `--quote-only`, and `erc20 approve
--confirm-unlimited`. Limit *values* and the risk module design are owned by
[[Trading Strategies]] / risk.

**Implemented 2026-06-11** ([[TWAK Spike Runbook]] Step 4): `trader.risk` (frozen
`SPIKE_POLICY`, pure `check_trade` with 8 refusal codes, append-only JSONL ledger so caps
survive restarts) + `trader.execution` (`twak_cli` wrapper that **structurally refuses
`--password`** and never logs argv/env; `execute_trade` two-phase intent→quote re-check —
the quote's realized USD/route/implied-slippage are re-judged under the same caps before
signing). Fail closed throughout: unreadable ledger, unvaluable quote, or twak error ⇒
`STATE_UNAVAILABLE` refusal. 43 tests pin the refusal matrix.

## x402 signing safety

x402 (pay-per-request) is the one place an external server influences what we sign, so it gets
explicit bounds. Two paths exist:

**TWAK native (`twak x402 request`).** Signs an on-chain payment authorization and retries with
a `PAYMENT-SIGNATURE` header. EIP-3009 gasless is preferred over Permit2; `--max-payment`
(atomic units) is a required hard cap; only `https://` URLs are accepted and loopback/private
IPs are rejected before any network call. `--prefer-network bsc` keeps payments on-chain with
the trading wallet.

**BNB SDK (`X402Signer` + `SigningPolicy`).** `EVMWalletProvider.sign_typed_data` is
policy-gated by default. `SigningPolicy.strict_default()`:

- **Denylists** ERC-2612 `Permit` and Permit2 `PermitSingle`/`PermitBatch` **unconditionally** —
  the denylist wins even if your own code mistakenly allowlists them. These are unbounded
  allowance grants and the core drain vector.
- Allows only EIP-3009 `TransferWithAuthorization` / `ReceiveWithAuthorization` against the
  registered U-token deployments; other tokens/chains require an explicit `domain_allowlist`
  extension keyed on `(chain_id, verifyingContract)`.
- Enforces a validity window (≤600s window / ≤900s future by default).

Tool functions must receive a **scoped `X402Signer`, never a raw `WalletProvider`**. The signer
enforces per-call `max_value`, a cumulative `session_budget`, `message['from'] == wallet.address`,
and a caller-committed **`expected_to`** that MUST come from config/registry — *not* from the 402
challenge body. **Threat:** a malicious 402 server talks the LLM into signing an unbounded
Permit, or redirects payment to an attacker payee. The denylist + `expected_to` commitment +
budget caps neutralize both. (Test-only escapes `SigningPolicy.permissive()` and
`_DANGEROUS_sign_typed_data_no_policy()` MUST NOT appear in agent-reachable code.)

## Secrets hygiene

- **Nothing secret in git, code, logs, or the vault.** Wallet passwords, mnemonics, and API keys
  live in a git-ignored local `.env` or a secure store, never committed.
- **All interactive secret-bearing steps are run by the user, never an agent** (portal signup,
  `twak setup`/`init`, `wallet create`, keychain save, funding). Agents verify the resulting
  state with read-only calls only — the checkpointed procedure is [[TWAK Spike Runbook]].
- **TWAK API auth is separate from wallet signing.** API requests use an Access ID +
  **HMAC-SHA256** secret; `twak init` writes them to `~/.twak/credentials.json` with `0600`.
  The env-var names the CLI actually reads are **`TWAK_ACCESS_ID` / `TWAK_HMAC_SECRET`**
  (v0.19.0 verified; `.env.example` matches these, not the older `TWAK_API_KEY` guess).
  The HMAC secret is shown once — never commit it, never add it to `~/.bashrc`/`~/.zshrc` (env
  exports are for ephemeral CI only). `twak serve --rest` uses the raw HMAC secret as a local
  bearer token — a reason to prefer the **stdio MCP** server over the REST surface on the host.
- Rotate keys from the portal; use separate dev/prod keys.

## Self-custody scoring — what keeps the 25 points

Self-custody integrity is worth **25 pts** on the Best-Use-of-TWAK rubric, with a penalty
ladder (not a hard DQ):

| Custody posture | Points |
|---|---|
| Fully self-custodial, clean local signing end-to-end | **20–25** |
| A custodial component at one step (third-party co-sign / custody) | 8–15 |
| Core trade loop depends on custody | 0–7 |

The whole design above (local `wallet.json`, local password resolution, `twak serve --watch`
signing on the local wallet) sits in the top band. **Any custodial shortcut in the trade loop is
a hard no** — if a hosting or convenience choice introduces one, flag it loudly and propose the
custodial-clean alternative. The tie-breaker also favors cleanest self-custody first.

## Wallet unification — a key design decision

The two SDKs keep **separate** key stores:

| Store | Path | Encryption | Perms |
|---|---|---|---|
| TWAK | `~/.twak/wallet.json` | AES-256-GCM + PBKDF2 | — |
| BNB SDK `EVMWalletProvider` | `~/.bnbagent/wallets/` | Keystore V3 (scrypt + AES-128-CTR) | `0o600` / dir `0o700` |

The competition registers **one** agent wallet address on-chain (`twak compete register` /
`competition_register`), and the ERC-8004 identity is minted to *its* wallet address. For the
registered trading address and the ERC-8004 identity to be the **same address**, the **same
mnemonic/private key must back both stores** — i.e. import the TWAK-derived EVM private key into
`EVMWalletProvider` (via `PRIVATE_KEY`, encrypted on first run), or derive both from one seed.

**RESOLVED — probed on `bsctestnet` 2026-06-11.** TWAK CLI v0.19.0 ships native `twak erc8004`
(`register` / `set-uri` / `set-metadata` / `show`, default `--chain bsc`, `bsctestnet` also a
known deployment), and the probe proved unification end-to-end with the spike wallet:

- `twak wallet address --chain bsctestnet` == `--chain bsc` == the trading address
  (`0x2C19…D32E`) — one BIP-44 EVM derivation across chains, as designed.
- `twak erc8004 register --chain bsctestnet --uri data:…` (keychain-resolved, faucet tBNB for
  gas) minted **agentId 1369**, tx
  `0xb03cdd129a86b880b53da89fe9eb4b07ce51b86023342316da646a340c248db7`.
- On-chain read-back `erc8004 show 1369`: **`owner` AND `agentWallet` == the wallet's own
  address.** Identity, competition registration, and trading all sign from the one `~/.twak`
  store — **zero key export**; the BNB SDK only *reads* the identity and never holds keys.

Gotcha for the real mint: `--uri` is **required** on `register` (the help reads as optional;
it is not) — have the competition agent-card URI ready (e.g. hosted on data.alexlouis.dev)
before minting the identity on mainnet.

## Always-on host design — the live-week key story (design, not yet stood up)

The June 22–28 window needs unattended signing on an always-on box. The spike runs on the
**Windows 11 dev laptop** ([[TWAK Spike Runbook]] — WSL is not installed there, and the
keychain works natively, so native Windows is the spike host). For the live week, the
candidates and their key stories:

| Host option | Password at rest | Unattended after reboot? | Notes |
|---|---|---|---|
| **Dev laptop, kept always-on (Windows 11)** | Windows Credential Manager (DPAPI, per-user) — **verified working** with TWAK | After *user logon* only — Credential Manager needs the logon session. Auto-logon closes the gap but weakens at-rest protection (anyone at the machine is the user). | Lowest new-surface risk; the box we already trust. Residential power/network are the real failure mode — pair with [[Real-time Monitoring]] dead-man alerting. |
| **Small Linux VPS (systemd service)** | Either (a) Secret Service via a keyring daemon — needs a **manual unlock per boot** (one SSH after reboot, unattended thereafter), or (b) `TWAK_WALLET_PASSWORD` in a root-owned `0600` systemd `EnvironmentFile`, never in a dotfile | (a) no — one manual step per boot; (b) yes | Cloud reliability; but the provider/hypervisor joins the trust base. Env-file is acceptable for a competition wallet sized to the live-week bankroll — the wallet holds only what the week needs. |
| **act-trainer desktop** | — | — | **Out of scope by policy**: keyless, no-mainnet, shared. Never a signing host. |

Custody integrity is identical in all options — `wallet.json` + password on the box we
control, keys never third-party. The decision axis is *password-at-rest vs. unattended-reboot
recovery*, plus uptime. **Recommended shape:** a small Linux host running the loop as a
systemd unit with the env-file pattern (hardened: dedicated user, `ProtectHome`, no
world-readable paths), manual-unlock Secret Service as the stretch goal; the laptop is the
fallback if VPS setup threatens the timeline. Decide after the spike proves the loop;
provisioning steps live in [[Remote Capabilities]] when scheduled.

**`twak serve --watch` vs. our own loop — resolved (design).** TWAK's watcher executes only
its own DCA/limit *automations* — it cannot run our decision core. So the agent runs **our
`execute_trade` loop driving `twak swap` via the CLI with `--json`** (subprocess wrapper,
deterministic, unit-testable, no extra server process); `serve`/`watch` are not in the trade
path. The TWAK MCP surface stays available for interactive/debug use, but the signing path
the guardrails wrap is the CLI call.

**`--auto-lock` mechanics.** Re-locking discards the in-memory key; the *next* signing op
re-resolves the password (keychain/env) and re-unlocks — with either source present this is
transparent and costs nothing, so run with a short auto-lock. Empirical confirmation is a
runbook step ([[TWAK Spike Runbook]] step 8).

## Threat model — always-on host holding signing authority

| Threat | Mitigation |
|---|---|
| Host compromise reads `wallet.json` | Ciphertext only; useless without the password. Keep the password in the OS keychain, not in `wallet.json`'s directory or a dotfile. |
| Password exfiltration | Never `--password` (shell history/args). Prefer keychain; env-var only inside a hardened container. Scope OS perms to the agent user. |
| Malicious 402 server drains wallet | TWAK `--max-payment` cap + https-only; BNB SDK Permit denylist + `expected_to` + per-call/session budgets. Never expose a raw `WalletProvider`. |
| LLM tricked into an unbounded `approve` | `approve` requires the password and is behind `execute_trade` guardrails; `--confirm-unlimited` gates unbounded grants; allowlist constrains spenders. |
| Wrong-payee transfer / ENS swap | `transfer --confirm-to` pins the resolved address; `--max-usd` caps value. |
| Runaway/looping signer | `--auto-lock` re-locks on inactivity; daily-cap + drawdown-stop guardrails fail closed; tx hashes logged for audit ([[Real-time Monitoring]]). |
| Secret leaks to repo/logs | `.env` git-ignored; `credentials.json` at `0600`; no secret in logs or the vault. |

The residual single point of failure is the password's at-rest protection on the host — which is
exactly the key-on-remote-box question owned by [[Remote Capabilities]].

## Open questions

*(updated 2026-06-11 after the keyless CLI verification — see [[TWAK Spike Runbook]])*

- **Live-week host pick.** Design done (§always-on host design): Linux VPS env-file pattern
  vs. always-on laptop with Credential Manager. Decide after the spike proves the loop;
  provisioning in [[Remote Capabilities]].
- **x402 on BSC.** Confirm the live BSC x402 routes (USDC/USDT) and whether our data/inference
  spend uses the TWAK native path or the BNB SDK `X402Signer`. (`twak x402 quote` is read-only
  and needs no wallet — cheap to probe once credentials exist.)
- **`--auto-lock` re-unlock (mechanism understood, empirically unconfirmed).** Resolution
  should make re-unlock transparent from keychain/env; confirm in runbook step 8.

**Resolved 2026-06-11:**
- ~~**Wallet unification (was gating)**~~ → **proven on `bsctestnet`**: ERC-8004 agentId 1369
  minted from the spike wallet; `owner`/`agentWallet` == the trading address; one `~/.twak`
  store, zero key export (§wallet unification above).
- ~~`twak serve --watch` vs. our own loop~~ → **our own loop wrapping the `twak` CLI
  (`--json`)**; TWAK's watcher only runs its DCA/limit automations, not a custom decision core
  (§always-on host design; mirrors the [[MCP Server]] open item).
- ~~Keychain availability on Windows~~ → **works** (`@napi-rs/keyring` → Windows Credential
  Manager; verified round-trip). The spike host is native Windows (WSL not installed).
- ~~`twak compete register` surface~~ → real in v0.19.0: `compete register|status`, flags just
  `--password`/`--json`; `status` reports the deadline. Register only the final unified wallet,
  never the throwaway ([[TWAK Spike Runbook]] step 7).
