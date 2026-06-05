# Source / provenance — Trust Wallet Agent Kit (TWAK)

- **Source:** https://github.com/trustwallet/developer (the GitBook-synced docs repo —
  authoritative source, replaces the earlier scraped `.md` pages)
- **Pulled:** 2026-06-05 (shallow clone; agent-relevant slice copied, temp clone removed)
- **CLI package:** `@trustwallet/cli` (the `twak` command)
- **Developer portal (API keys):** https://portal.trustwallet.com

## What's mirrored here (the agent-relevant slice, not the whole repo)

| Path | Content |
|------|---------|
| `agent-sdk/agent-sdk.md` | Agent SDK overview |
| `agent-sdk/quickstart.md` | 5-minute CLI setup |
| `agent-sdk/cli-reference.md` | Full `twak` command set |
| `agent-sdk/authentication.md` | API keys + HMAC-SHA256 signing |
| `agent-sdk/key-management.md` | Local key storage, encryption, signing permissions — relevant to self-custody-on-a-host |
| `mcp/mcp.md`, `mcp/docs-mcp.md`, `mcp/api-gateway.md` | TWAK **MCP servers** — the MCP surface (cf. the hackathon's `competition_register` MCP action) |
| `claude-code-skills/claude-code-skills.md` | TWAK agent Skills |
| `SUMMARY.md` | Full repo nav (lists sections NOT mirrored) |
| `_get-started.md` | Repo landing README |

**Deliberately not mirrored** (in the repo, not needed for this build): `trustconnect/`
(dApp wallet-connect UI), `wallet-core/`, `barz-smart-wallet/`, `dapps/`, `assets/`,
`develop-for-trust/`. See `SUMMARY.md` for the full tree.

## Refresh / grab more sections

```bash
git clone --depth 1 https://github.com/trustwallet/developer _tw_tmp
cp -r _tw_tmp/agent-sdk _tw_tmp/mcp _tw_tmp/claude-code-skills .   # or any section from SUMMARY.md
rm -rf _tw_tmp
```

> Snapshot only — verify against the live repo before relying on specifics. Note the
> hackathon's `twak compete register` / `competition_register` may be competition-specific
> and not in the general CLI/MCP reference; confirm in the official hackathon channels.
