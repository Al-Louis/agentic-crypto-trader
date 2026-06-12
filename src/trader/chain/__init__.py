"""Pool-event data layer — read-only PancakeSwap pair/pool event collection.

Collects Sync/Swap/Mint/Burn(/Collect) logs for the 20-token universe's pools
via ``eth_getLogs`` on a public BSC RPC, stores them as Parquet under
``data/chain/``, and derives hourly panels aligned to the recorded OHLCV
returns index (liquidity-over-time, net swap-flow imbalance, LP add/remove,
wallet-attributed flow).

ISOLATION CONTRACT: this package is research instrumentation only. It reads
the chain and writes ``data/chain/``; it imports nothing from and is imported
by nothing in ``trader.train``, the env, the rewards, or the live trading
path. Integration happens only if a probe passes, later, through the training
loop's process. See the vault note [[Pool-Event Data Layer]].
"""
