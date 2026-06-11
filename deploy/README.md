# deploy/ — EC2 trading-host templates

Deployable templates for the live-week host. **The procedure is the vault note
`EC2 Trading Host Runbook`** (`.obsidian-vault/BNB Hackathon/`) — these files are its
artifacts; do not apply them out of order. Templates only: placeholders, **no secrets**.

| File | What |
|---|---|
| `trader-agent.service` | systemd unit for the agent loop (hardened sandbox, root-owned env-file) |
| `trader.env.template` | `/etc/trader/trader.env` skeleton (root:root 0600; filled via `sudoedit`) |
| `trader-twak.sh` | root-only ceremony/verify helper — runs `twak` as `trader` with the password from the env-file, never in argv |
| `verify-host.sh` | read-only hardening/state checks (agent-safe; never prints secret values) |
| `iam/trading-put-only-policy.json` | instance-role policy: `s3:PutObject` on `trading/*` only |
| `iam/ec2-trust-policy.json` | EC2 assume-role trust policy for the instance role |
| `agent-card.template.json` | ERC-8004 agent card skeleton (hosted on data.alexlouis.dev before the mainnet mint — `--uri` is REQUIRED) |
