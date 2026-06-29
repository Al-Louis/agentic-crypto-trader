"""Competition leaderboard — pure parts: participant CSV parsing, Multicall3 aggregate3 encode/decode,
and balance scaling. No network (the eth_call transport is faked)."""

from trader.competition import flows
from trader.competition import multicall as mc
from trader.competition.participants import parse_participants

_HEADER = ('"category","txn hash","methodID","block","timestamp","from","to","value","asset"\n')
_REG = "0x212c61b9b72c95d95bf29cf032f5e5635629aed5"


def _row(h, mid, block, ts, frm, to):
    return f'"external","{h}","{mid}",{block},"{ts}","{frm}","{to}","0","BNB"\n'


def test_parse_participants_filters_dedupes_and_drops_deploy():
    csv = _HEADER + "".join([
        _row("0xa", "0x1aa3a008", 200, "t2", "0xWALLET1", _REG),       # later dup of wallet1
        _row("0xb", "0x1aa3a008", 100, "t1", "0xwallet1", _REG),       # earlier -> kept
        _row("0xc", "0x1aa3a008", 150, "t3", "0xWALLET2", _REG),
        _row("0xd", "0x60806040", 50, "t0", "0xDEPLOYER", "0x0000000000000000000000000000000000000000"),
        _row("0xe", "0x1aa3a008", 160, "t4", "0xOUTSIDER", "0xsomeothercontract"),  # wrong contract
    ])
    ps = parse_participants(csv)
    assert [p["wallet"] for p in ps] == ["0xwallet1", "0xwallet2"]   # lowercased, sorted, deduped
    w1 = next(p for p in ps if p["wallet"] == "0xwallet1")
    assert w1["registered_block"] == 100 and w1["registered_ts"] == "t1"  # earliest kept


def _enc_results(values):
    """Encode a Multicall3 `Result[]` of uint256 returns. `values` = [(success, int|None)]."""
    def u(n):
        return f"{int(n):064x}"
    n = len(values)
    heads, tails, running = [], [], n * 32
    for ok, val in values:
        tup = u(1 if ok else 0) + u(0x40) + (u(0) if val is None else u(32) + u(val))
        heads.append(u(running))
        tails.append(tup)
        running += len(tup) // 2
    return "0x" + u(0x20) + u(n) + "".join(heads) + "".join(tails)


def test_aggregate3_decode_roundtrips_results():
    decoded = mc.decode_aggregate3(_enc_results([(True, 123), (False, None), (True, 0)]))
    assert decoded[0][0] is True and int(decoded[0][1], 16) == 123
    assert decoded[1][0] is False and decoded[1][1] == "0x"
    assert decoded[2][0] is True and int(decoded[2][1], 16) == 0


def test_encode_aggregate3_shape():
    usdt = "0x55d398326f99059ff775485246999027b3197955"
    data = mc.encode_aggregate3([(usdt, mc.calldata_balance_of("0xabc")), (mc.MULTICALL3, "0xdeadbeef")])
    assert data.startswith("0x" + mc.AGGREGATE3)
    # the array length word (3rd 32-byte word: selector, head-offset 0x20, then len) == 2
    body = data[2 + 8:]                       # strip 0x + selector
    assert int(body[64:128], 16) == 2        # word[1] after the 0x20 head == array length
    assert mc.BALANCE_OF in data and usdt[2:] in data.lower()


def test_read_holdings_scales_by_decimals_and_adds_bnb():
    usdt = "0x55d398326f99059fF775485246999027B3197955"
    six = "0x000000000000000000000000000000000000c01c"   # a pretend 6-decimals token
    tokens = [{"symbol": "USDT", "contract": usdt}, {"symbol": "SIX", "contract": six}]
    decimals = {usdt.lower(): 18, six.lower(): 6}

    # one multicall: balanceOf(USDT)=2e18, balanceOf(SIX)=5e6, getEthBalance=3e17
    scripted = _enc_results([(True, 2 * 10**18), (True, 5 * 10**6), (True, 3 * 10**17)])

    def fake_call(method, params):
        assert method == "eth_call" and params[0]["to"] == mc.MULTICALL3
        return scripted

    h = mc.read_holdings(fake_call, "0xabc", tokens, decimals, block="latest")
    assert h["USDT"] == 2.0 and h["SIX"] == 5.0 and h["BNB"] == 0.3


class _FakeNR:
    """Returns scripted inbound (to_address query) / outbound (from_address query) transfers."""

    def __init__(self, inbound, outbound):
        self._in, self._out = inbound, outbound

    def asset_transfers(self, *, from_block, to_block, from_address=None, to_address=None,
                        category=None):
        return list(self._out) if from_address else list(self._in)


def _t(asset, qty, h, *, cat="20", ts=1000):
    return {"category": cat, "asset": asset, "qty": qty, "hash": h, "ts": ts,
            "from": "0xext", "to": "0xwallet", "block": 1, "contract": ""}


def test_cost_basis_classifies_deposits_swaps_withdrawals_airdrops():
    inbound = [
        _t("USDT", 100.0, "h1"),                 # deposit
        _t("BNB", 1.0, "h2", cat="external"),     # deposit, priced at bnb_price_now
        _t("USDT", 50.0, "h3"),                   # swap leg (h3 also has an outbound) -> excluded
        _t("SHIB", 1_000_000.0, "h4"),            # airdrop (non-fundable) -> ignored, but recorded
    ]
    outbound = [
        _t("TOKEN", 5.0, "h3"),                   # swap leg pairing h3 -> whole tx excluded
        _t("USDT", 20.0, "h5"),                   # withdrawal
    ]
    cb = flows.wallet_cost_basis(_FakeNR(inbound, outbound), "0xwallet",
                                 from_block=0, to_block=9, bnb_price_now=600.0)
    assert cb["gross_deposits"] == 700.0          # 100 USDT + 1 BNB*600 (h3's 50 USDT excluded)
    assert cb["gross_withdrawals"] == 20.0
    assert cb["net_deposited"] == 680.0
    assert cb["n_deposits"] == 2 and cb["n_withdrawals"] == 1
    assert cb["nonfundable_deposit_assets"] == ["SHIB"]


def test_universe_canonicalizes_and_dedupes_by_contract(tmp_path):
    from trader.competition.universe import CANONICAL, load_universe
    # resolved.json with a ticker collision (USDC mis-resolved onto UB's contract) + a USDF/USDf dup
    bad = '0x40b8129b786d766267a7a118cf8c07e31cdb6fde'
    resolved = [
        {"symbol": "USDC", "status": "resolved", "token_address": bad},          # mis-resolved
        {"symbol": "UB", "status": "resolved", "token_address": bad},             # real UB
        {"symbol": "USDf", "status": "resolved", "token_address": "0xdup"},
        {"symbol": "USDF", "status": "resolved", "token_address": "0xdup"},       # dup contract
        {"symbol": "ZEC", "status": "resolved", "token_address": "0xzec"},
    ]
    p = tmp_path / "resolved.json"
    p.write_text(__import__("json").dumps(resolved), encoding="utf-8")
    uni = load_universe(str(p))
    by = {u["symbol"]: u["contract"].lower() for u in uni}
    assert by["USDC"] == CANONICAL["USDC"].lower()        # forced canonical, off UB's contract
    assert by["UB"] == bad                                # UB keeps its real contract
    contracts = [u["contract"].lower() for u in uni]
    assert len(contracts) == len(set(contracts))          # NO contract counted twice
    assert ("USDf" in by) ^ ("USDF" in by)                # exactly one of the dup pair survives


def test_completed_window_days_excludes_in_progress_day():
    from datetime import datetime, timezone
    from trader.competition.snapshot import completed_window_days
    start = int(datetime(2026, 6, 22, 0, 0, tzinfo=timezone.utc).timestamp())  # window open
    # 2.5 days in (Jun 24 12:00) -> Jun 22 and Jun 23 are complete; Jun 24 (in progress) is NOT.
    now = int(datetime(2026, 6, 24, 12, 0, tzinfo=timezone.utc).timestamp())
    assert completed_window_days(start, now) == ["2026-06-22", "2026-06-23"]
    # just after open: no completed days yet (no DQ possible)
    assert completed_window_days(start, start + 3600) == []
    # past the 7-day window end: all 7 days complete, capped at the window
    assert len(completed_window_days(start, start + 99 * 86400)) == 7


def _mini_lb(generated, equity=100.0):
    return {"generated": generated, "window": {"start_ts": 1782086400},
            "n_participants": 1, "n_ranked": 1, "n_disqualified": 0, "n_dq_risk": 0,
            "total_equity_usd": equity,
            "rows": [{"wallet": "0xa", "rank": 1, "equity_usd": equity, "pnl_pct": 5.0,
                      "capital_basis_usd": 95.0, "ranked": True, "disqualified": False,
                      "traded_in_window": True}]}


def test_history_idempotent_per_hour_and_appends_new_hours(tmp_path):
    import json
    from trader.competition import history
    comp = str(tmp_path / "competition")
    history.update_history(_mini_lb("2026-06-22T23:05:00+00:00"), comp)
    history.update_history(_mini_lb("2026-06-22T23:40:00+00:00", equity=110.0), comp)  # same hour -> replace
    ser = json.load(open(tmp_path / "competition" / "series.json", encoding="utf-8"))
    assert [s["id"] for s in ser["snapshots"]] == ["2026-06-22T23Z"]       # one hour
    assert len(ser["wallets"]["0xa"]) == 1 and ser["wallets"]["0xa"][0]["equity_usd"] == 110.0  # replaced
    history.update_history(_mini_lb("2026-06-23T00:05:00+00:00", equity=120.0), comp)  # new hour -> append
    ser = json.load(open(tmp_path / "competition" / "series.json", encoding="utf-8"))
    assert [s["id"] for s in ser["snapshots"]] == ["2026-06-22T23Z", "2026-06-23T00Z"]
    assert len(ser["wallets"]["0xa"]) == 2
    assert (tmp_path / "competition" / "snapshots" / "2026-06-23T00Z" / "leaderboard.json").exists()


def test_daily_trade_rule_counts_bnb_and_stable_swaps():
    # A BNB<->USDT keepalive swap (no non-stable alt) MUST satisfy the >=1-trade/day rule,
    # even though it is NOT an eligible-alt trade for ranking. Regression for the false-DQ bug.
    ts = 1782129600  # 2026-06-22T12:00Z
    inbound = [_t("USDT", 30.0, "h1", ts=ts)]              # swap output (USDT in)
    outbound = [_t("BNB", 0.05, "h1", cat="external", ts=ts)]  # swap input (BNB out), same tx
    cb = flows.wallet_cost_basis(_FakeNR(inbound, outbound), "0xwallet",
                                 from_block=0, to_block=9, bnb_price_now=600.0,
                                 eligible_contracts=set())
    assert cb["trade_days"] == ["2026-06-22"]    # the keepalive day IS a trade day
    assert cb["n_swaps"] == 1
    assert cb["n_eligible_buys"] == 0 and cb["traded_eligible"] is False  # but NOT an alt trade (ranking)


def test_boundary_flow_treats_uncounted_conversion_as_capital():
    usdt = "0x55d398326f99059ff775485246999027b3197955"
    btcb = "0x7130d2a12b9bcbfae4f2634d864a1ee1ce3ead9c"   # NOT counted
    counted = {usdt}
    # swap: sell uncounted BTCB for counted USDT (same tx) -> the USDT is CAPITAL, not PnL
    inbound = [{**_t("USDT", 10.0, "h1"), "contract": usdt}]
    outbound = [{**_t("BTCB", 0.001, "h1"), "contract": btcb}]
    cb = flows.wallet_cost_basis(_FakeNR(inbound, outbound), "0xw", from_block=0, to_block=9,
                                 bnb_price_now=600.0, counted_contracts=counted, prices={})
    assert cb["boundary_flow"] == 10.0 and cb["net_capital_in"] == 10.0
    # counted<->counted swap (BNB<->USDT) is a pure trade -> NO boundary flow
    inb2 = [{**_t("BNB", 0.01, "h2", cat="external"), "contract": "0x0"}]
    out2 = [{**_t("USDT", 6.0, "h2"), "contract": usdt}]
    cb2 = flows.wallet_cost_basis(_FakeNR(inb2, out2), "0xw", from_block=0, to_block=9,
                                  bnb_price_now=600.0, counted_contracts=counted, prices={})
    assert cb2["boundary_flow"] == 0.0


def test_deepest_price_requires_base_token_match():
    # Regression for the 2026-06-26 16Z phantom -45% dip: TRX's deepest BSC pool is HTX-BASED, whose
    # priceUsd (~$1.7e-6) is HTX's, not TRX's. The base-token guard must reject it and use the (shallower)
    # TRX-based pool, else a TRX-heavy wallet's equity collapses to ~$0 for that token.
    from trader.competition import pricing
    trx = "0xce7de646e7208a4ef112cb6ed5038fa6cc6b12e3"
    pairs = [
        {"chainId": "bsc", "liquidity": {"usd": 3_400_000}, "priceUsd": "0.0000017",
         "baseToken": {"address": "0xHTX", "symbol": "HTX"}},          # deepest, but our token is QUOTE
        {"chainId": "bsc", "liquidity": {"usd": 278_000}, "priceUsd": "0.32",
         "baseToken": {"address": trx.upper(), "symbol": "TRX"}},      # shallower, base IS our token
    ]
    assert pricing._deepest_price_usd(pairs, trx) == 0.32              # base-matched price wins
    assert pricing._deepest_price_usd(pairs[:1], trx) is None          # no base match -> unpriced, not garbage
    assert pricing._deepest_price_usd(pairs) == 0.0000017             # no addr -> back-compat (deepest wins)


def test_stable_peg_only_when_cmc_confirms():
    # The is_stable flag is unreliable (resolved.json flagged STABLE, a ~$0.04 token, as a stablecoin ->
    # priced $1 -> a wallet showed +2145%). Honor the $1 peg only when CMC agrees it's ~$1, or has no data.
    from trader.competition import pricing
    assert pricing._peg_to_dollar(True, 0.999) is True       # genuine stable, CMC confirms
    assert pricing._peg_to_dollar(True, None) is True        # genuine stable, CMC has no data (e.g. DAI)
    assert pricing._peg_to_dollar(True, 0.0386) is False     # STABLE: flagged stable but ~$0.04 -> use CMC
    assert pricing._peg_to_dollar(True, 1.136) is False      # EURI: EUR-pegged ~$1.14 -> use CMC
    assert pricing._peg_to_dollar(False, 0.999) is False     # not flagged -> never pegged


def test_cmc_bar_selection_for_historical_pricing():
    # CMC k-line gives per-hour history by address; _cmc_at must pick the bar COVERING ts (open <= ts),
    # so a wallet's equity at any past hour is marked at that hour's price (not the latest).
    from trader.competition import pricing
    addr = "0xtoken"
    pricing._CMC_SERIES[addr] = {1000: 10.0, 4600: 11.0, 8200: 12.0}   # three hourly bars
    try:
        assert pricing._cmc_at(addr, 5000) == 11.0      # within the 4600 bar
        assert pricing._cmc_at(addr, 8200) == 12.0      # exactly on a bar open
        assert pricing._cmc_at(addr, 9999) == 12.0      # after the last bar -> last close
        assert pricing._cmc_at(addr, 500) == 10.0       # before the series -> earliest bar
        assert pricing._cmc_now(addr) == 12.0           # current = latest bar
    finally:
        del pricing._CMC_SERIES[addr]


class _BlockNR:
    """Fake NodeReal that serves block-stamped transfers and records every live fetch's range, so a
    test can prove the cache only fetches NEW blocks on the second run (and dedups the overlap)."""

    def __init__(self, head, transfers):
        self.head = head
        self.transfers = transfers          # list of {id, block, from, to, ...}
        self.fetches = []                   # (from_block, to_block, direction)

    def block_number(self):
        return self.head

    def asset_transfers(self, *, from_block, to_block, from_address=None, to_address=None,
                        category=None):
        direction = "out" if from_address else "in"
        self.fetches.append((from_block, to_block, direction))
        key = "from" if from_address else "to"
        addr = (from_address or to_address or "").lower()
        return [t for t in self.transfers
                if t.get(key) == addr and from_block <= t["block"] <= to_block]


def test_cached_nodereal_fetches_only_new_blocks_and_dedups(tmp_path):
    from trader.competition.nodereal import CachedNodeReal
    cache = str(tmp_path / "nr.json")
    win = 100
    transfers = [
        {"id": "a", "block": 110, "to": "0xw", "from": "0xext", "qty": 1.0},   # in
        {"id": "b", "block": 150, "to": "0xw", "from": "0xext", "qty": 2.0},   # in (boundary block)
        {"id": "c", "block": 180, "to": "0xw", "from": "0xext", "qty": 3.0},   # in (after first run)
    ]

    # --- run 1: cache empty, head=150 -> scans [100..150], sees a + b ---
    nr1 = _BlockNR(150, transfers)
    c1 = CachedNodeReal(nr1, cache, win)
    c1.LOOKBACK = 10                                    # small, so the boundary math stays observable
    latest1 = c1.block_number()
    got1 = c1.asset_transfers(from_block=win, to_block=latest1, to_address="0xw")
    assert {t["id"] for t in got1} == {"a", "b"}
    assert nr1.fetches == [(100, 150, "in")]          # full window scanned once (scanned=0)
    c1.save()

    # --- run 2: fresh instance loads the cache, head=180 -> scans only the tail+lookback ---
    nr2 = _BlockNR(180, transfers)
    c2 = CachedNodeReal(nr2, cache, win)
    c2.LOOKBACK = 10
    assert c2.scanned == 150                            # persisted boundary
    latest2 = c2.block_number()
    got2 = c2.asset_transfers(from_block=win, to_block=latest2, to_address="0xw")
    assert {t["id"] for t in got2} == {"a", "b", "c"}   # cached a,b + newly-fetched c
    assert nr2.fetches == [(140, 180, "in")]            # only the NEW tail (scanned-lookback), not [100..180]
    # block 150 ("b") sits in both ranges but is deduped by id -> appears exactly once
    assert sum(1 for t in got2 if t["id"] == "b") == 1
