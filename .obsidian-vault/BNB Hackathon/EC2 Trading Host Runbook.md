# EC2 Trading Host Runbook

The ordered, checkpointed procedure to stand up the **live-week trading host** on AWS EC2 and
perform the **competition-wallet key ceremony** on it — the build called for by
[[Security and Encryption]] §always-on host design and the [[Remote Capabilities]] hosting
decision. Same discipline as [[TWAK Spike Runbook]]: **every secret-bearing step is a
USER-ACTION**; the agent verifies the resulting state with read-only calls only; every command
is copy-pasteable. Deployable templates live in **`deploy/`** at the repo root.

**Hard rules carried in from the custody design:**
- The competition wallet is **created ON this box** — keys are born where signing happens and
  never transit. The laptop spike wallet (`0x2C19…D32E`) is a throwaway and is never reused.
- The mnemonic backup is **paper, user-held, offline** — never cloud, never a password manager,
  never a screenshot (this differs from the spike wallet, where a password-manager note was
  fine for a ≤$10 throwaway).
- **No `--password` flag, ever.** The unlock path on this headless box is the root-owned
  `0600` env-file (§the headless keychain answer, below).
- **Never snapshot/AMI this instance** once the wallet exists — a snapshot copies
  `wallet.json` + the env-file into account-level artifacts.
- Registration must land **before the trading window opens June 22** — the contract rejects
  entries after the window opens ([[BNB Hack - AI Trading Agent Edition]]). `compete status`
  reads a June 25 deadline; do not lean on that slack. The wallet must also **hold in-scope
  assets at window start** to be ranked, so bankroll funding must also land before June 22.

**Timeline fit** ([[Build Log]] 2026-06-11 plan): provision now→Jun 16 in parallel with the
loop build; deploy paper mode Jun 16; ceremony + registration + dust-from-the-box in the
Jun 16–21 window; bankroll before Jun 22.

## Decisions baked into this runbook (rationale one-liners)

| Decision | Choice | Why |
|---|---|---|
| Instance | **t3.small (2 vCPU, 2 GiB), x86_64** | I/O-bound loop + Node CLI subprocesses; 1 GiB risks OOM mid-trade; x86_64 because every TWAK native binding was verified on x64 — no arch surprises. ~$15/mo, ~3 weeks ≈ $12. |
| Region | **us-east-1** | Same region as the data bucket + the account's existing ops. RPC latency is not load-bearing here (decision cadence is minutes–hours; the 1% slippage bound dominates; TWAK quotes route via its API anyway). |
| OS | **Ubuntu 24.04 LTS** | systemd + Python 3.12 native; matches the desktop's proven toolchain. |
| Disk | **gp3 16 GiB, EBS encryption ON** | Encryption protects the physical layer (decommissioned disks), not the AWS API layer — see the threat note. |
| IMDS | **IMDSv2 required, hop limit 1** | Blocks SSRF-style credential theft of the instance role. |
| Security group | **Inbound: TCP 22 from the user's IP /32 only. Outbound: default allow** | The loop needs 443 egress to many endpoints (TWAK API, RPC, S3, BscScan, CMC, GitHub, npm); tightening egress buys little here. No other inbound port exists — the REST `serve` surface is never used. |
| IP | **Elastic IP (recommended)** | Stable SSH target + stable SG story across stop/start; free while attached to a running instance. |
| Wallet unlock | **Root-owned 0600 `EnvironmentFile`** (`TWAK_WALLET_PASSWORD`) | No Secret Service on a headless box — see §the headless keychain answer. |
| Telemetry | **Instance role, `s3:PutObject` on `trading/*` ONLY** | Put-only, no delete/list/get — consistent with the no-delete publisher posture ([[Apentic Data Contract]]). |
| Freshness | **CloudFront cache behavior `trading/*` → CachingDisabled; NO invalidation rights on the box** | `cloudfront:CreateInvalidation` can't be path-scoped and would have to sit on the instance role; a one-time cache behavior keeps the role pure put-only and the heartbeat fresh in seconds. |
| Repo access | **Read-only GitHub deploy key generated on the box** | Repo is private; HTTPS clone hangs on a credential prompt (learned on the desktop); a deploy key is least-privilege and revocable. |

Placeholders used throughout: `<USER_IP>` (your current public IP), `<ACCOUNT_ID>`,
`<GITHUB_OWNER>`, `<EIP_ALLOC_ID>`, `<INSTANCE_ID>`, `<SG_ID>`, `<AMI_ID>`.

---

## Phase A — Provision (USER-ACTION: AWS console or CLI from the laptop)

All AWS API work is run **by the user from the laptop** (admin credentials live there; the
box itself only ever gets the put-only role). CLI form below; console is equivalent.

### A1 — USER-ACTION: SSH keypair (import, don't generate in AWS)

Generate locally so the private key never transits AWS:

```powershell
ssh-keygen -t ed25519 -f $env:USERPROFILE\.ssh\act-trading-host -C "act-trading-host"
aws ec2 import-key-pair --region us-east-1 --key-name act-trading-host `
  --public-key-material fileb://$env:USERPROFILE\.ssh\act-trading-host.pub
```

### A2 — USER-ACTION: IAM role (put-only telemetry) + instance profile

Policy JSON is `deploy/iam/trading-put-only-policy.json`; trust policy is
`deploy/iam/ec2-trust-policy.json`. From the repo root:

```powershell
aws iam create-role --role-name act-trading-host-role `
  --assume-role-policy-document file://deploy/iam/ec2-trust-policy.json
aws iam put-role-policy --role-name act-trading-host-role `
  --policy-name act-trading-put-only `
  --policy-document file://deploy/iam/trading-put-only-policy.json
aws iam create-instance-profile --instance-profile-name act-trading-host-profile
aws iam add-role-to-instance-profile --instance-profile-name act-trading-host-profile `
  --role-name act-trading-host-role
```

The policy is exactly: `s3:PutObject` on `arn:aws:s3:::alexlouis-apentic-data/trading/*`.
No Get, no List, no Delete, no CloudFront — the box can append telemetry and nothing else.

### A3 — USER-ACTION: security group

```powershell
aws ec2 create-security-group --region us-east-1 --group-name act-trading-host-sg `
  --description "act trading host: ssh from user ip only"
aws ec2 authorize-security-group-ingress --region us-east-1 --group-id <SG_ID> `
  --protocol tcp --port 22 --cidr <USER_IP>/32
```

(Outbound stays the default allow-all. If your home IP changes mid-week, update this one
ingress rule — the dead-man heartbeat does not depend on SSH.)

### A4 — USER-ACTION: launch the instance

```powershell
# Current Ubuntu 24.04 LTS amd64 AMI:
aws ssm get-parameter --region us-east-1 `
  --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id `
  --query Parameter.Value --output text

aws ec2 run-instances --region us-east-1 `
  --image-id <AMI_ID> --instance-type t3.small `
  --key-name act-trading-host --security-group-ids <SG_ID> `
  --iam-instance-profile Name=act-trading-host-profile `
  --metadata-options "HttpTokens=required,HttpPutResponseHopLimit=1,HttpEndpoint=enabled" `
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":16,"VolumeType":"gp3","Encrypted":true,"DeleteOnTermination":true}}]' `
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=act-trading-host}]'

# Elastic IP (recommended):
aws ec2 allocate-address --region us-east-1 --query AllocationId --output text
aws ec2 associate-address --region us-east-1 --instance-id <INSTANCE_ID> --allocation-id <EIP_ALLOC_ID>
```

**Checkpoint A (agent-verifiable once the user shares the IP):**
`ssh -i ~/.ssh/act-trading-host ubuntu@<EIP>` connects; on the box,
`curl -s -o /dev/null -w '%{http_code}' http://169.254.169.254/latest/meta-data/` → **401**
(IMDSv2 enforced). `deploy/verify-host.sh` items 1 and 10 pass.

---

## Phase B — Base hardening + toolchain (agent-OK over SSH; no secrets involved)

As `ubuntu` on the box:

```bash
sudo apt-get update && sudo apt-get -y upgrade
sudo apt-get install -y git python3-venv python3-pip unattended-upgrades curl

# Security-only unattended upgrades, NO automatic reboot during the live window:
sudo dpkg-reconfigure -f noninteractive unattended-upgrades
echo 'Unattended-Upgrade::Automatic-Reboot "false";' | sudo tee /etc/apt/apt.conf.d/52act-no-reboot

# Node 22 + the pinned TWAK CLI (0.19.0 is the verified surface — re-verify before bumping):
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo npm install -g @trustwallet/cli@0.19.0

# 2 GiB swap — OOM-kill mid-trade is the failure mode this buys out on a 2 GiB box:
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Dedicated non-root user; home OUTSIDE /home so ProtectHome=yes still works in the unit:
sudo useradd --create-home --home-dir /srv/trader --shell /bin/bash trader
sudo chmod 750 /srv/trader

# sshd: keys only, ubuntu only (trader is never SSH-reachable):
printf 'PasswordAuthentication no\nKbdInteractiveAuthentication no\nAllowUsers ubuntu\n' | sudo tee /etc/ssh/sshd_config.d/60-act.conf
sudo systemctl reload ssh

# journald bounds (the execution wrapper already guarantees no argv/env in logs):
printf '[Journal]\nSystemMaxUse=200M\n' | sudo tee /etc/systemd/journald.conf.d/act.conf >/dev/null 2>&1 || \
  (sudo mkdir -p /etc/systemd/journald.conf.d && printf '[Journal]\nSystemMaxUse=200M\n' | sudo tee /etc/systemd/journald.conf.d/act.conf)
sudo systemctl restart systemd-journald
```

**Checkpoint B:** `bash deploy/verify-host.sh` (once the repo is cloned — or run the checks
manually) → node v22.x, twak 0.19.x, trader user, swap, sshd, clock all PASS.

---

## Phase C — App deploy (agent-OK; read-only deploy key is the one USER-ACTION)

```bash
sudo -u trader -H bash -c 'ssh-keygen -t ed25519 -f ~/.ssh/act_deploy -N "" -C "act-ec2-deploy" && cat ~/.ssh/act_deploy.pub'
```

**USER-ACTION C1:** add the printed public key as a **read-only deploy key** on the GitHub
repo (Settings → Deploy keys; do NOT tick write access).

```bash
sudo -u trader -H bash <<'EOF'
printf 'Host github.com\n  IdentityFile ~/.ssh/act_deploy\n  IdentitiesOnly yes\n' >> ~/.ssh/config
GIT_TERMINAL_PROMPT=0 git clone git@github.com:<GITHUB_OWNER>/agentic-crypto-trader.git /srv/trader/agentic-crypto-trader
python3 -m venv /srv/trader/venv
/srv/trader/venv/bin/pip install -e "/srv/trader/agentic-crypto-trader[data,remote]"
EOF
```

(Extras: `data` + `remote` cover the reads + boto3 publish today; if principal-engineer adds
a `live`/loop extra, install that instead — confirm before Phase F.)

**Checkpoint C:** `/srv/trader/venv/bin/python -c "import trader"` exits 0; `git -C
/srv/trader/agentic-crypto-trader rev-parse --short HEAD` matches the pushed sha.

---

## Phase D — The env-file (USER-ACTION: secrets enter here, via sudoedit only)

```bash
cd /srv/trader/agentic-crypto-trader
sudo install -d -m 0755 /etc/trader
sudo install -m 0600 -o root -g root deploy/trader.env.template /etc/trader/trader.env
sudo install -m 0700 -o root -g root deploy/trader-twak.sh /usr/local/sbin/trader-twak
```

**USER-ACTION D1:** `sudoedit /etc/trader/trader.env` and fill `CMC_API_KEY`,
`BSCSCAN_API_KEY` (and later, Phase E, `TWAK_WALLET_PASSWORD` + `AGENT_WALLET_ADDRESS`).
`sudoedit` keeps values out of shell history; **never `cat` this file in a session an agent
can read.** No AWS keys go in it — S3 comes from the instance role.

**Checkpoint D (agent-verifiable, names/perms only — never values):**
`deploy/verify-host.sh` item 3 → root:root 0600, required keys non-empty, no static AWS keys.

---

## Phase E — The key ceremony (USER-ACTION — the heart of this runbook)

All of E runs **by the user**, over SSH, as the trader user: `sudo -u trader -H bash` first.
The agent's role is the verify blocks only.

### E1 — USER-ACTION: production TWAK API credentials

Create a **second, production API key** at https://portal.trustwallet.com (separate dev/prod
keys per [[Security and Encryption]] §secrets hygiene — the laptop keeps the dev key). The
HMAC secret is shown once: password manager only. Then, in the `trader` shell (interactive
prompts — the secret never enters history):

```bash
twak setup        # credentials phase only; skip harness; wallet phase comes next as E2
```

**Agent verifies (read-only):** `sudo -u trader -H twak auth status` → `configured`;
`sudo -u trader -H twak price BNB --json` returns a price (HMAC auth proven end-to-end);
`sudo -n stat -c '%U %a' /srv/trader/.twak/credentials.json` → `trader 600`.

### E2 — USER-ACTION: create the competition wallet ON the box

In the `trader` shell:

```bash
twak setup --wallet
```

- **Password:** fresh and strong, never used elsewhere. It goes in exactly two places: your
  password manager, and (E3) the env-file. Nowhere else, ever.
- **Keychain:** on this headless box the default keychain save has **no Secret Service to
  talk to** — expect it to fail or be skipped; that is by design here (§headless answer
  below). If `setup --wallet` aborts on the keychain step, fall back to
  `twak wallet create --no-keychain` (it prompts for the password interactively, same as the
  spike). Contingency only if some path *demands* a `--password` flag: `set +o history`
  first, run it, `history -c`, and treat that shell as burned — but prefer the prompts.
- **The mnemonic** (shown once): copy it to **paper, by hand, two copies**, stored where you
  control them. Read it back from paper to verify before continuing. NEVER: screenshot,
  clipboard, cloud notes, password manager — this is the competition wallet, not the spike
  throwaway. Don't run this step inside any terminal that logs scrollback to disk.

**Agent verifies (read-only):** `sudo -u trader -H twak wallet status` → `Wallet: configured`
(keychain line expected to read *no password stored* — correct on this host);
`sudo -n stat -c '%U %a' /srv/trader/.twak/wallet.json` → `trader 600`-class perms.

### E3 — USER-ACTION: wire the unlock path

`sudoedit /etc/trader/trader.env` → set `TWAK_WALLET_PASSWORD=` to the E2 password.

### E4 — Record the address + the zero-cost signing proof (agent-OK, before any funds)

Address derivation and message signing are password-gated; the root helper resolves the
password from the env-file **without it ever appearing in argv or history**:

```bash
sudo trader-twak wallet address --chain bsc --json
sudo trader-twak wallet sign-message --chain bsc --message "act-live-2026-06" --json
```

Expected: a `0x…` address and a signature hex — **unattended signing proven on the
production host with zero funds at risk**, the same blocker-#1-in-miniature gate the spike
used. Then **USER-ACTION E4a:** `sudoedit /etc/trader/trader.env` → set
`AGENT_WALLET_ADDRESS=<the address>` (public, not secret; also record it in the laptop `.env`
for monitoring). If either command auth-errors, the env-file password is wrong — fix E3
before anything else.

**Checkpoint E (the ceremony gate):** signature returned; address recorded; mnemonic on
paper and verified; password in exactly password-manager + env-file; `verify-host.sh` all
green through item 5.

---

## Phase F — systemd + paper mode (agent-OK)

```bash
cd /srv/trader/agentic-crypto-trader
sudo install -m 0644 deploy/trader-agent.service /etc/systemd/system/trader-agent.service
sudo systemctl daemon-reload
sudo systemctl enable --now trader-agent
```

The unit (template: `deploy/trader-agent.service`): `User=trader`, root-owned `0600`
`EnvironmentFile`, `ExecStart=/srv/trader/venv/bin/python -m trader.agent` (**confirmed** —
the loop's entry as built; with no `--mode` argv it reads `TRADER_MODE` from the env-file,
invalid values refuse loudly), `Restart=always` (safe: the risk
ledger persists caps and the loop re-reads on-chain truth on boot — a restart cannot
double-spend), `NoNewPrivileges` + `ProtectSystem=strict` + `ProtectHome=yes` (trader's home
is `/srv/trader`, deliberately outside `/home`) + `ReadWritePaths=/srv/trader`, capability
set empty. `TRADER_MODE=paper` until Phase H.

**Checkpoint F:** `systemctl is-active trader-agent` → `active`; `journalctl -u trader-agent
-n 50` shows loop ticks and **no secret material** (the wrapper never logs argv/env — spot
check anyway); within minutes `https://data.alexlouis.dev/trading/heartbeat.json` (exact
filenames per the loop's publisher — [[Apentic Data Contract]] §trading/) is fresh, proving
the put-only role works.

**One-time, from the laptop (USER-ACTION F1):** add a CloudFront cache behavior on the data
distribution for path pattern `trading/*` using the managed **CachingDisabled** policy (keep
`SimpleCORS` response headers). This is the freshness answer: no invalidations from the box,
no extra IAM, heartbeat staleness measured in seconds. (Console: CloudFront → the
`data.alexlouis.dev` distribution → Behaviors → Create.)

---

## Phase G — Identity + registration (ordering matters; before June 22)

### G1 — Host the agent card (USER-ACTION, laptop)

Fill `deploy/agent-card.template.json` (name, description, the E4 address; leave `agentId`
placeholder for now) and publish it from the **laptop** (existing publisher creds) to
`trading/agent-card.json` on the data bucket. Verify
`https://data.alexlouis.dev/trading/agent-card.json` returns 200 JSON. **`--uri` is REQUIRED
on the erc8004 mint** ([[TWAK Spike Runbook]] step 7 gotcha) — this must exist first.

### G2 — USER-ACTION: gas + smoke funding (~$10 of BNB)

Send ~$10 of BNB to the E4 address (BEP-20 / BNB Smart Chain network; byte-for-byte address
check). This covers the dust-trade smoke + two registration transactions, nothing more.

**Agent verifies:** `sudo trader-twak wallet balance --chain bsc --json` shows it; cross-check
`https://bscscan.com/address/<addr>`; confirm amount ≈ intended, else stop and flag.

### G3 — Dust trade from the production host (agent runs, explicit user go-ahead)

The final smoke gate before anything irreversible: a ~$1 BNB→USDT swap through
`execute_trade` (guardrails active, spike-sized caps; LIVE policy values are
[[Trading Strategies]]/quant's call and need not be final here), confirmed via
`twak tx … --json` and on BscScan. Plus the negative proof: an over-cap intent refuses.
This re-proves, on this box: env-file unlock, quote→re-check→sign→confirm, ledger
persistence, journald hygiene.

### G4 — Mint the ERC-8004 identity on BSC mainnet (USER-ACTION go-ahead; agent may drive)

```bash
sudo trader-twak erc8004 register --chain bsc --uri https://data.alexlouis.dev/trading/agent-card.json --json
sudo trader-twak erc8004 show <AGENT_ID> --chain bsc --json
```

Expected (mirrors the bsctestnet probe, agentId 1369): `owner` AND `agentWallet` == the E4
address. Then update the hosted card's `agentId` field and re-publish (G1 path).

### G5 — Competition registration (USER-ACTION go-ahead; the point of no return)

```bash
sudo trader-twak compete register --json
sudo trader-twak compete status --json
```

`status` must show `registered: true` for this address. This binds the competition to the
box's wallet — exactly once, never the spike wallet.

**Checkpoint G:** registered:true + erc8004 owner/agentWallet match + dust tx hash on
BscScan + agent card live. **All before June 22.**

### G6 — USER-ACTION: bankroll funding (shortly before June 22)

Fund the live-week bankroll (sizing is the user's call with quant — open question below).
Constraints: must be **on the box before the window opens** (non-zero in-scope balance at
start or you're unranked), and the wallet should hold **only** the week's bankroll —
bankroll sizing is the only protection that survives outright key theft (see threat note).

**Agent verifies:** balance matches intent; the live flip is a deliberate, separate
USER-ACTION at window open — sudoedit sets **both** `TRADER_MODE=live` **and**
`AGENT_ALLOW_LIVE=1` (the loop's double-gate; either alone refuses), then
`sudo systemctl restart trader-agent`.

---

## The headless keychain answer (asked honestly)

**There is no OS-keychain path on this box, and that's fine.** TWAK's keychain backend
(`@napi-rs/keyring`) targets Linux **Secret Service** — a D-Bus session service that exists
on desktop Linux. A headless server has no D-Bus session and no keyring daemon. You *can*
bolt on `gnome-keyring` headless, but its keyring must itself be unlocked at boot by a
secret that lives in plaintext on the same disk — which reduces to the env-file with extra
moving parts and a daemon added to the trust base. So the **root-owned 0600
`EnvironmentFile` is the unlock path**, exactly option (b) in the host-design table
([[Security and Encryption]]), already accepted there for a bankroll-sized wallet.

What it actually defends: a **non-root** compromise cannot read the password at rest (file
is root:root 0600; systemd pid-1 injects it). What nothing on this box defends: a
**trader-level** compromise reads the running process's environment and `wallet.json` and
therefore signs — but that is the price of *unattended signing itself*, identical under a
keychain (an unattended box must hold an unlockable credential). The honest mitigations are
the ones this design already has: zero inbound surface (SSH-only, one IP), no REST `serve`,
a sandboxed service, the abnormal-transfer alert ([[Real-time Monitoring]]), and **a wallet
that only ever holds the week's bankroll**.

**The trade-off to say out loud:** the AWS control plane joins the trust base. Anyone with
EC2/EBS permissions in the account can snapshot the volume and read env-file + wallet.json —
EBS encryption does not stop API-level access. And **guardrails do not protect stolen keys**
(a thief signs directly, bypassing our code). So: AWS account hygiene (MFA, no stray IAM
users/keys), never snapshot the box, and bankroll-only sizing. Custody integrity itself is
unchanged — keys live on a box we control, no third party ever holds them; this stays in the
top self-custody band.

---

## Phase H — Live-week ops (June 22–28)

**Daily user checklist (~2 minutes):**
1. `/apentic/trading` — heartbeat fresh (stale > ~15 min = investigate now), drawdown vs the
   ~30% DQ gate, **daily trade count ≥ 1** before the day closes.
2. `https://bscscan.com/address/<addr>` — no unexpected transfers (an outbound tx the ledger
   doesn't know about = possible key compromise: safe-stop immediately, sweep funds out).
3. Only if something looks off: SSH in, `systemctl status trader-agent`,
   `journalctl -u trader-agent --since -1h`.

**Safe stop / restart:** `sudo systemctl stop trader-agent` is always safe — the risk ledger
(`data/risk_ledger.jsonl`) persists spend/caps/high-water across restarts and the loop
re-reads on-chain state as truth on boot, so stop→start can never double-spend. Remember the
≥1-trade/day floor: don't leave it stopped past a day boundary.

**Reboots:** the unit is `enabled`; the env-file path means the box is **fully unattended
through reboots** (the one advantage bought by accepting the env-file posture).

**Host changes freeze:** no apt upgrades of node/twak, no repo pulls onto the box during the
window unless fixing a live defect (then: pin, test in paper mode first if at all possible).

## Phase I — Teardown + key retirement (after June 28)

Order matters — funds first, then secrets, then the box:

1. **USER-ACTION:** wait until results/judging no longer read the wallet, then sweep funds
   to your main wallet: `sudo trader-twak transfer … --max-usd <amount> --confirm-to <your
   address>` (TWAK's payee-pin + cap flags, per [[Security and Encryption]]). Verify on
   BscScan the wallet is empty (dust remainder ok).
2. `sudo systemctl disable --now trader-agent`.
3. Destroy secrets on-box: `sudo shred -u /etc/trader/trader.env
   /srv/trader/.twak/wallet.json /srv/trader/.twak/credentials.json`.
4. **USER-ACTION:** revoke the production TWAK API key in the portal; remove the GitHub
   deploy key; destroy the paper mnemonic copies (the wallet is empty and retired — the
   address's history stays public, which is fine and expected).
5. **USER-ACTION (laptop):** terminate the instance (EBS `DeleteOnTermination` wipes the
   volume), release the Elastic IP, delete the SG/role/profile/keypair if not reused.

---

## Open questions (for the user)

- **SSH key:** new dedicated `act-trading-host` ed25519 key (as written) or reuse an
  existing personal key? Dedicated recommended (revocable independently).
- **Elastic IP:** accept the (tiny) cost/step, or live with the IP changing on stop/start?
  Runbook assumes yes.
- **Bankroll size + funding date:** how much, and funded which day (must be pre-June 22;
  later = less time at rest on the box)? Decide with quant — it is also the key-theft blast
  radius.
- **Home-IP churn:** if your ISP rotates your IP, the SSH ingress rule needs a manual
  update — acceptable, or pre-authorize a second CIDR?

## Links

Decision + host design: [[Security and Encryption]] §always-on host design ·
[[Remote Capabilities]] (hosting decision). Proven ceremony pattern: [[TWAK Spike Runbook]].
Telemetry contract: [[Apentic Data Contract]] §trading/ · [[Real-time Monitoring]].
Templates: `deploy/` (repo root).
