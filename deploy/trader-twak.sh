#!/usr/bin/env bash
# trader-twak — root-only helper: run `twak` as the trader user with the wallet
# password resolved from /etc/trader/trader.env, with the secret NEVER appearing
# in argv (so `ps` / journald / shell history stay clean — only process *env*
# carries it, same as the systemd unit).
#
# Install: sudo install -m 0700 -o root -g root deploy/trader-twak.sh /usr/local/sbin/trader-twak
# Use:     sudo trader-twak wallet address --chain bsc --json
#          sudo trader-twak wallet sign-message --chain bsc --message "act-live" --json
#
# This is the ceremony/verify path only. The live loop resolves the password via the
# systemd EnvironmentFile — it does not use this helper.
set -euo pipefail

ENV_FILE=/etc/trader/trader.env

if [[ ${EUID} -ne 0 ]]; then
  echo "trader-twak: must run as root (sudo trader-twak ...)" >&2
  exit 1
fi
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "trader-twak: ${ENV_FILE} missing" >&2
  exit 1
fi

# Pull only the password line; tolerate other vars. No value is ever echoed.
pw="$(grep -E '^TWAK_WALLET_PASSWORD=' "${ENV_FILE}" | head -1 | cut -d= -f2-)"
if [[ -z "${pw}" ]]; then
  echo "trader-twak: TWAK_WALLET_PASSWORD not set in ${ENV_FILE}" >&2
  exit 1
fi

export TWAK_WALLET_PASSWORD="${pw}"
export HOME=/srv/trader
# setpriv inherits the environment (secret stays out of argv) and drops to trader.
exec setpriv --reuid=trader --regid=trader --init-groups twak "$@"
