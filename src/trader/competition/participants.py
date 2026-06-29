"""Parse the participant roster from the registration-contract CSV.

`data/bnb_hackathon_participants.csv` is a BscScan tx export of the BNB-hackathon registration
contract. Every row's `to` is that one contract; the register call is `methodID 0x1aa3a008`; the
**`from` column is each participant's registered agent wallet**. We keep the register-call rows,
dedupe by wallet (first registration kept), and drop the contract-deploy / setup rows.

Pure + stdlib-only: `parse_participants(csv_text)` is testable without the file or the network.
"""

from __future__ import annotations

import csv
import io

# The registration contract the CSV was exported from, and the register() selector.
REGISTRY_CONTRACT = "0x212c61b9b72c95d95bf29cf032f5e5635629aed5"
REGISTER_METHOD_ID = "0x1aa3a008"


class Participant(dict):
    """A registered agent: ``{wallet, registered_ts, registered_block, tx_hash}``."""


def parse_participants(csv_text: str, *, registry: str = REGISTRY_CONTRACT,
                       method_id: str = REGISTER_METHOD_ID) -> list[Participant]:
    """Registered agent wallets from the CSV text, deduped (earliest registration kept), sorted by
    wallet. Keeps only rows that are register() calls to the registration contract — this drops the
    contract-deploy (`0x60806040`) and setup (`0x386c1866`) rows the deployer made.

    Each wallet appears once even if it registered multiple times; `registered_*` is the earliest."""
    reg = (registry or "").lower()
    by_wallet: dict[str, Participant] = {}
    for row in csv.DictReader(io.StringIO(csv_text)):
        to = (row.get("to") or "").lower()
        mid = (row.get("methodID") or row.get("methodId") or "").lower()
        wallet = (row.get("from") or "").lower()
        if not wallet or (reg and to != reg) or (method_id and mid != method_id.lower()):
            continue
        ts = (row.get("timestamp") or "").strip()
        try:
            block = int(row.get("block") or 0)
        except ValueError:
            block = 0
        cur = by_wallet.get(wallet)
        # keep the earliest registration (smallest block) as the canonical record
        if cur is None or (block and block < (cur.get("registered_block") or 1 << 62)):
            by_wallet[wallet] = Participant(
                wallet=wallet, registered_ts=ts, registered_block=block,
                tx_hash=row.get("txn hash") or row.get("txhash") or "")
    return sorted(by_wallet.values(), key=lambda p: p["wallet"])


def load_participants(path: str = "data/bnb_hackathon_participants.csv", **kw) -> list[Participant]:
    """`parse_participants` over the CSV file (UTF-8)."""
    with open(path, encoding="utf-8") as f:
        return parse_participants(f.read(), **kw)
