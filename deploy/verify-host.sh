#!/usr/bin/env bash
# verify-host.sh — read-only hardening + state checks for the EC2 trading host.
# Safe for an agent to run (never prints a secret value; checks names/permissions only).
# Run on the box: bash deploy/verify-host.sh   (sudo not required for most checks;
# the env-file checks need sudo and degrade to SKIP without it.)
set -u

pass=0; fail=0; skip=0
ok()   { echo "PASS  $1"; pass=$((pass+1)); }
bad()  { echo "FAIL  $1"; fail=$((fail+1)); }
skp()  { echo "SKIP  $1"; skip=$((skip+1)); }

# --- 1. IMDSv2 enforced (v1 request must be rejected) ---
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://169.254.169.254/latest/meta-data/ || echo 000)
if [[ "$code" == "401" ]]; then ok "IMDSv2 required (IMDSv1 gets 401)"; else bad "IMDSv1 not rejected (got HTTP $code; want 401)"; fi

# --- 2. trader user exists, correct home, no sudo ---
if home=$(getent passwd trader | cut -d: -f6) && [[ "$home" == "/srv/trader" ]]; then
  ok "trader user exists with home /srv/trader"
else
  bad "trader user missing or home != /srv/trader"
fi
if id -nG trader 2>/dev/null | grep -qwE 'sudo|admin|wheel'; then bad "trader is in a sudo group"; else ok "trader has no sudo group"; fi

# --- 3. env file: exists, root:root, 0600, required keys non-empty (names only) ---
if sudo -n test -f /etc/trader/trader.env 2>/dev/null; then
  st=$(sudo -n stat -c '%U:%G %a' /etc/trader/trader.env)
  [[ "$st" == "root:root 600" ]] && ok "trader.env is root:root 0600" || bad "trader.env perms are '$st' (want root:root 600)"
  for k in TWAK_WALLET_PASSWORD CMC_API_KEY BSCSCAN_API_KEY AGENT_WALLET_ADDRESS; do
    if sudo -n grep -qE "^${k}=.+" /etc/trader/trader.env; then ok "trader.env has ${k} (non-empty)"; else bad "trader.env missing/empty ${k}"; fi
  done
  if sudo -n grep -qE '^AWS_(ACCESS_KEY_ID|SECRET_ACCESS_KEY)=.+' /etc/trader/trader.env; then
    bad "trader.env contains static AWS keys (instance role should cover S3)"
  else
    ok "no static AWS keys in trader.env"
  fi
else
  skp "env-file checks (need sudo, or file absent)"
fi

# --- 4. wallet store: exists, owned by trader, tight perms ---
if sudo -n test -f /srv/trader/.twak/wallet.json 2>/dev/null; then
  st=$(sudo -n stat -c '%U %a' /srv/trader/.twak/wallet.json)
  [[ "$st" == trader\ 6* ]] && ok "wallet.json owned by trader, perms $st" || bad "wallet.json owner/perms '$st'"
elif sudo -n true 2>/dev/null; then
  skp "wallet.json not present yet (pre-ceremony)"
else
  skp "wallet checks (need sudo)"
fi

# --- 5. toolchain versions ---
nodev=$(node --version 2>/dev/null || echo none); [[ "$nodev" == v22* ]] && ok "node $nodev" || bad "node is '$nodev' (want v22.x)"
twakv=$(twak --version 2>/dev/null || echo none); [[ "$twakv" == 0.19.* ]] && ok "twak $twakv (pinned line)" || bad "twak is '$twakv' (verified surface is 0.19.x — re-verify before bumping)"
pyv=$(/srv/trader/venv/bin/python --version 2>/dev/null || echo none); [[ "$pyv" == Python\ 3.1* ]] && ok "venv $pyv" || bad "venv python is '$pyv'"

# --- 6. sshd: no password auth ---
pa=$(sudo -n sshd -T 2>/dev/null | grep -i '^passwordauthentication' | awk '{print $2}')
if [[ "$pa" == "no" ]]; then ok "sshd PasswordAuthentication no"; elif [[ -z "$pa" ]]; then skp "sshd -T (need sudo)"; else bad "sshd PasswordAuthentication=$pa"; fi

# --- 7. swap active (OOM mid-trade is the named failure mode on a small instance) ---
if [[ $(swapon --noheadings 2>/dev/null | wc -l) -ge 1 ]]; then ok "swap active"; else bad "no swap configured"; fi

# --- 8. clock sync (HMAC auth + hourly scoring both care) ---
if timedatectl show -p NTPSynchronized --value 2>/dev/null | grep -q yes; then ok "clock NTP-synchronized"; else bad "clock not NTP-synchronized"; fi

# --- 9. service state (post-deploy phases) ---
if systemctl list-unit-files trader-agent.service --no-legend 2>/dev/null | grep -q trader-agent; then
  en=$(systemctl is-enabled trader-agent 2>/dev/null); ac=$(systemctl is-active trader-agent 2>/dev/null)
  [[ "$en" == "enabled" ]] && ok "trader-agent enabled" || bad "trader-agent is-enabled=$en"
  [[ "$ac" == "active" ]] && ok "trader-agent active" || bad "trader-agent is-active=$ac"
else
  skp "trader-agent.service not installed yet"
fi

# --- 10. instance role visible (put-only telemetry path) ---
tok=$(curl -s --max-time 2 -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')
role=$(curl -s --max-time 2 -H "X-aws-ec2-metadata-token: $tok" http://169.254.169.254/latest/meta-data/iam/security-credentials/ || true)
[[ -n "$role" ]] && ok "instance role attached: $role" || bad "no instance role visible via IMDS"

echo "----"
echo "pass=$pass fail=$fail skip=$skip"
[[ $fail -eq 0 ]]
