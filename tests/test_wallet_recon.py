"""On-chain wallet reconciliation -> trading/wallet.json: the real equity/PnL math (pure), the
balanceOf/decimals decoding (fake RPC), and the local put path. No network."""

import json

from trader.agent.wallet_recon import (build_wallet_payload, publish_wallet, read_holdings_onchain,
                                       read_live_equity_usd)


class _FakeRpc:
    """Scripted responses: balances {contract: raw_wei}, decimals {contract: int}, native raw_wei."""

    def __init__(self, balances, decimals, native):
        self._bal, self._dec, self._native = balances, decimals, native

    def eth_call(self, to, data):
        if data.startswith("0x70a08231"):       # balanceOf(address)
            return hex(self._bal.get(to, 0))
        if data == "0x313ce567":                 # decimals()
            return hex(self._dec.get(to, 18))
        return "0x0"

    def call(self, method, params):
        assert method == "eth_getBalance"
        return hex(self._native)


def test_build_payload_equity_and_pnl():
    holdings = {"USDT": 50.0, "UB": 100.0, "BNB": 0.01}
    prices = {"UB": 2.0, "BNB": 600.0}           # USDT is priced 1.0 internally
    p = build_wallet_payload(holdings, prices, baseline_usd=100.0, address="0xabc")
    assert p["equity_usd"] == 256.0              # 50*1 + 100*2 + 0.01*600
    assert p["pnl_usd"] == 156.0 and p["pnl_pct"] == 156.0
    assert p["source"] == "onchain" and p["stale"] is False
    assert p["holdings"][0]["token"] == "UB"     # sorted by value desc (200 > 50 > 6)


def test_build_payload_missing_price_flags_stale_and_excludes():
    p = build_wallet_payload({"USDT": 10.0, "ZZZ": 5.0}, {}, baseline_usd=10.0, address="0xabc")
    assert p["stale"] is True                    # ZZZ unpriced -> incomplete
    assert p["equity_usd"] == 10.0               # only USDT counted
    assert [h for h in p["holdings"] if h["token"] == "ZZZ"][0]["value_usd"] is None


def test_build_payload_no_baseline_omits_pnl():
    p = build_wallet_payload({"USDT": 10.0}, {}, baseline_usd=None, address="0xabc")
    assert p["pnl_usd"] is None and p["pnl_pct"] is None and p["equity_usd"] == 10.0


def test_read_holdings_decodes_balances_and_native():
    UB, USDT = "0xUB", "0xUSDT"
    rpc = _FakeRpc(balances={UB: 100 * 10**18, USDT: 50 * 10**18},
                   decimals={UB: 18, USDT: 18}, native=int(0.01 * 10**18))
    h = read_holdings_onchain("0xWALLET", [{"symbol": "UB", "contract": UB},
                                           {"symbol": "USDT", "contract": USDT}], rpc=rpc)
    assert h["UB"] == 100.0 and h["USDT"] == 50.0 and abs(h["BNB"] - 0.01) < 1e-9


def test_read_holdings_honors_nondefault_decimals():
    rpc = _FakeRpc(balances={"0xT": 5 * 10**6}, decimals={"0xT": 6}, native=0)
    h = read_holdings_onchain("0xW", [{"symbol": "T", "contract": "0xT"}], rpc=rpc)
    assert h["T"] == 5.0                          # 5e6 / 10^6 = 5 (decimals() honored, not assumed 18)


def test_publish_wallet_writes_local_target(tmp_path):
    target = str(tmp_path / "trading")

    def fake_holdings(addr, assets):
        return {"USDT": 90.0, "UB": 4.0, "BNB": 0.005}

    payload = publish_wallet(target, address="0xabc",
                             assets=[{"symbol": "UB", "contract": "0xUB"}],
                             prices={"UB": 2.5, "BNB": 600.0}, baseline_usd=100.0,
                             holdings_fn=fake_holdings)
    f = json.loads((tmp_path / "trading" / "wallet.json").read_text(encoding="utf-8"))
    assert f["equity_usd"] == payload["equity_usd"] == 103.0   # 90 + 4*2.5 + 0.005*600
    assert f["pnl_usd"] == 3.0 and f["address"] == "0xabc" and f["source"] == "onchain"


def test_read_live_equity_usd_sums_total_not_just_usdt():
    """The bankroll anchor captures capital PARKED IN TOKENS + BNB, not just USDT (the $67.83 trap)."""
    sel = [{"symbol": "B", "pair_address": "0xp", "token_address": "0xB"}]

    def holdings(addr, assets):
        return {"USDT": 67.83, "B": 120.0, "BNB": 0.011}

    eq = read_live_equity_usd("0xW", sel, holdings_fn=holdings, prices={"B": 0.25, "BNB": 600.0})
    assert abs(eq - 104.43) < 0.01     # 67.83 + 120*0.25 + 0.011*600 -- NOT 67.83 (USDT-only)
