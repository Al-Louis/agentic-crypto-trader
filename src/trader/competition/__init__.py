"""Competition leaderboard — an ISOLATED, read-only side process.

Computes an equity + PnL% snapshot for every BNB-hackathon participant wallet and publishes static
JSON to the `competition/` CDN prefix. It is deliberately separate from the live trading agent
(`trader.agent.*`): it imports only PURE helpers (`trader.agent.wallet_recon.build_wallet_payload`,
`trader.report`/`remote_train.publish`), reads chain state read-only (no private key, no signer, no
TWAK), and never writes anything the live loop reads. See the vault note on the participant feed and
[[Apentic Data Contract]].

Why hand-rolled (not the bnbagent SDK): the SDK has no at-block/archive reads (the baseline needs
them), can't enumerate participants (its registry != our CSV's contract), and drags in web3. Reviewed
2026-06-22 — verdict: keep stdlib-only. The one piece we own that the SDK lacks is reading balances at
the June-22-00:00-UTC start block against an archive RPC.
"""
