"""Resolve eligible symbols to canonical BSC contracts (CMC) + pools (DexScreener).

Replaces the symbol-search heuristic (35% ambiguous) with CMC contract-address
resolution, then screens each token's *correct* contract via DexScreener
token-pairs. Output supersedes data/universe_screen.json for selection.

Needs CMC_API_KEY in .env. Run:
  .venv/Scripts/python.exe scripts/resolve_contracts.py
Out: data/resolved.json  (symbol -> {cmc_id, name, bsc_contract, pool, liq, vol, ...})
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trader import config  # noqa: E402
from trader.data import cmc, dexscreener as ds  # noqa: E402
from trader.data.eligible import ELIGIBLE_SYMBOLS, STABLES  # noqa: E402

OUT = "data/resolved.json"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    api_key = config.require("CMC_API_KEY")
    symbols = list(ELIGIBLE_SYMBOLS)
    print(f"resolving {len(symbols)} symbols via CMC -> BSC contract -> DexScreener pool\n")

    resolved = cmc.resolve_bsc_contracts(symbols, api_key)

    rows = []
    for i, sym in enumerate(symbols, 1):
        r = resolved[sym]
        contract = r.get("bsc_contract")
        if contract:
            try:
                row = ds.summarize_token_pairs(sym, ds.token_pairs(contract))
            except Exception as e:  # noqa: BLE001
                row = {"symbol": sym, "status": "pool_error", "error": repr(e)}
            time.sleep(0.3)  # DexScreener politeness
        else:
            row = {"symbol": sym, "status": "no_bsc_contract"}
        row.update(cmc_id=r.get("cmc_id"), cmc_name=r.get("name"),
                   cmc_rank=r.get("rank"), n_cmc_candidates=r.get("n_candidates"),
                   bsc_contract=contract, is_stable=sym in STABLES,
                   vol_proxy=ds.vol_proxy(row))
        rows.append(row)
        print(f"[{i:3}/{len(symbols)}] {sym:12} "
              f"cmc={'Y' if r.get('cmc_id') else '-'} "
              f"bsc={'Y' if contract else '-'} "
              f"pool={'Y' if row.get('pair_address') else '-'} "
              f"liq={row.get('liq_usd')}", flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    on_cmc = [r for r in rows if r.get("cmc_id")]
    with_contract = [r for r in rows if r.get("bsc_contract")]
    with_pool = [r for r in rows if r.get("pair_address")]
    print("\n=== resolution summary ===")
    print(f"  on CMC            : {len(on_cmc)}/{len(symbols)}")
    print(f"  BSC contract found: {len(with_contract)}")
    print(f"  DexScreener pool  : {len(with_pool)}")
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
