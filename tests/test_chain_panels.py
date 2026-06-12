"""Panel aggregation tests — synthetic decoded rows through build_panel."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from trader.chain import panels


@pytest.fixture()
def chain_root(tmp_path):
    root = str(tmp_path)
    pool = "0x" + "aa" * 20
    reg = [{"symbol": "TOK", "pool": pool, "version": "v3", "token_side": 1,
            "token0": "0x" + "01" * 20, "token1": "0x" + "02" * 20,
            "dec0": 18, "dec1": 18, "fee_ppm": 100, "quote": "USDT",
            "quote_anchor": "USD", "tier": "meme"}]
    with open(os.path.join(root, "_pools.json"), "w") as f:
        json.dump(reg, f)
    d = os.path.join(root, "logs", f"TOK_{pool[:10]}")
    os.makedirs(d)
    # block 1000 -> ts 3600 (hour 1), block 2000 -> ts 7200 (hour 2)
    bd = os.path.join(root, "blockindex")
    os.makedirs(bd)
    pd.DataFrame({"block": [1000, 2000, 3000],
                  "ts": [3600, 7200, 10800]}).to_parquet(
        os.path.join(bd, "samples.parquet"), index=False)

    seq = iter(range(100))

    def row(block, event, a0=None, a1=None, **kw):
        base = {c: None for c in
                ["block", "log_index", "tx_hash", "event", "a0", "a1",
                 "liquidity", "price1per0", "tick", "r0", "r1", "amount_l",
                 "sender", "recipient"]}
        base.update(block=block, log_index=next(seq), tx_hash="0x00",
                    event=event, a0=a0, a1=a1, **kw)
        return base

    rows = [
        # hour 1: a buy (quote/token0 in, token/token1 out) and a sell
        row(1000, "swap", a0=100.0, a1=-50.0, liquidity=10.0, price1per0=2.0,
            recipient="0xw1"),
        row(1100, "swap", a0=-20.0, a1=10.5, liquidity=10.0, price1per0=1.9,
            recipient="0xw2"),
        # hour 2: an LP pull
        row(2000, "burn", a0=-30.0, a1=-15.0, amount_l=-5e18, sender="0xlp"),
        row(2000, "collect", a0=-31.0, a1=-15.5, sender="0xlp", recipient="0xlp"),
    ]
    pd.DataFrame(rows).to_parquet(os.path.join(d, "p_1000_2000.parquet"), index=False)
    return root


def test_build_panel_token_side_1(chain_root):
    g = panels.build_panel("TOK", root=chain_root)
    h1, h2 = 3600, 7200
    assert list(g.index) == [h1, h2]
    # token is side 1: vol_token from |a1|, net from a1
    assert g.loc[h1, "n_swaps"] == 2
    assert g.loc[h1, "vol_token"] == pytest.approx(60.5)
    assert g.loc[h1, "vol_quote"] == pytest.approx(120.0)
    assert g.loc[h1, "net_token_in"] == pytest.approx(-39.5)   # net token OUT (net buying)
    assert g.loc[h1, "net_quote_in"] == pytest.approx(80.0)
    assert g.loc[h1, "unique_swappers"] == 2
    # hour 2: LP removal magnitudes positive, token side from a1
    assert g.loc[h2, "n_burns"] == 1 and g.loc[h2, "n_collects"] == 1
    assert g.loc[h2, "lp_remove_token"] == pytest.approx(15.0)
    assert g.loc[h2, "lp_remove_quote"] == pytest.approx(30.0)
    # price: token is side 1 -> 1/price1per0 of the last hour-1 swap
    assert g.loc[h1, "price_end"] == pytest.approx(1 / 1.9)
    # state forward-fills into the no-swap hour
    assert g.loc[h2, "price_end"] == pytest.approx(1 / 1.9)
    assert g.loc[h2, "liquidity_end"] == pytest.approx(10.0)


def test_interpolate_ts():
    bix = pd.DataFrame({"block": [0, 100], "ts": [0, 1000]})
    out = panels.interpolate_ts(np.array([50, 100]), bix)
    assert out[0] == pytest.approx(500.0) and out[1] == pytest.approx(1000.0)
