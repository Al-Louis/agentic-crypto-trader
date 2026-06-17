# Private model store — getting trained weights to the trading host privately

The trained policy (`policy.zip` + `vecnormalize.pkl`) is the project's core IP and **must not
be public**. The training/results bucket (`s3://alexlouis-apentic-data`, fronted by the public
`data.alexlouis.dev` CloudFront) is public-read on its served prefixes, so weights never go there.

Instead: a **dedicated private bucket** — Block Public Access ON, default SSE, **no CloudFront
behavior** (unreachable from the CDN). The desktop (or an admin) PUTs the checkpoint; the EC2
trading host's instance role GETs it with a tightly-scoped grant. Doubles as an offline-style
private backup of the deployed policy.

Bucket: **`alexlouis-act-private`**, region **us-east-1**, key prefix **`models/<run-id>/`**.

## USER-ACTION (admin creds — the laptop's `malexy-deploy` profile cannot create buckets/IAM)

### 1. Create the private bucket (locked down)

```powershell
aws s3api create-bucket --bucket alexlouis-act-private --region us-east-1
aws s3api put-public-access-block --bucket alexlouis-act-private `
  --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws s3api put-bucket-encryption --bucket alexlouis-act-private `
  --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
```

(us-east-1 needs no `LocationConstraint`. Use SSE-KMS instead of AES256 if you want a CMK.)

### 2. Grant the EC2 trading host read access (scoped to `models/*`)

The autonomous box uses its instance role (not admin creds), so this grant is required:

```powershell
aws iam put-role-policy --role-name act-trading-host-role `
  --policy-name act-private-models-get `
  --policy-document file://deploy/iam/private-models-get-policy.json
```

### 3. Upload the checkpoint (whoever holds the files + AWS creds)

Either the desktop pushes directly (attach `deploy/iam/private-models-put-policy.json` to the
desktop's publisher IAM user), or an admin uploads. The three files per run:

```bash
RID=ppo-event-rdLe4-ef-503b784-s2
aws s3 cp runs-rl/$RID/policy.zip        s3://alexlouis-act-private/models/$RID/
aws s3 cp runs-rl/$RID/vecnormalize.pkl  s3://alexlouis-act-private/models/$RID/
aws s3 cp runs-rl/$RID/$RID/metrics.json s3://alexlouis-act-private/models/$RID/
```

### 4. EC2 pulls (the agent verifies, then deploys)

On the box, into the run-dir the event-agent expects (`<run-dir>/policy.zip`,
`vecnormalize.pkl`, `<run-id>/metrics.json`):

```bash
RID=ppo-event-rdLe4-ef-503b784-s2
mkdir -p /srv/trader/models/$RID/$RID
aws s3 cp s3://alexlouis-act-private/models/$RID/policy.zip       /srv/trader/models/$RID/
aws s3 cp s3://alexlouis-act-private/models/$RID/vecnormalize.pkl /srv/trader/models/$RID/
aws s3 cp s3://alexlouis-act-private/models/$RID/metrics.json     /srv/trader/models/$RID/$RID/metrics.json
```

Then `python -m trader.agent.event_agent --run-dir /srv/trader/models/$RID --once --no-refresh`
is the dry-run gate (model loads, LSTM threads, a tick produces fills) before enabling the unit.

## Notes
- Nothing here is served by `data.alexlouis.dev` — no CloudFront behavior points at this bucket.
- The EC2 role's existing grant stays put-only on the public `trading/*`; this only ADDS
  `s3:GetObject` on the private `models/*`. No write to the private bucket from the box.
- Never snapshot/AMI the box (the runbook rule) — weights + env-file would land in account-level
  artifacts. The private bucket (SSE, BPA) is the sanctioned copy.
