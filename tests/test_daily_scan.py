"""The daily market-scan's `selected` block (the model's current vol-top-8) + windowing.

Torch-free — `eval_universe_and_caps` builds the pure EventRungEnv (selection is vol-ranking
only; no model). Run against the recorded panel with `--now` inside its span."""

import pytest

from trader.agent.daily_scan import _slice, build_selected
from trader.agent.event_live import LiveEventTrader

from train_rl import build_volume_panel, load_data  # noqa: E402 (event_live set the path)


def _prov() -> dict:
    # intrabar_floor/wick OFF (no frac panels needed) — selection is vol-ranking only, unaffected
    return {"k": 8, "max_entry_frac": 0.34, "stop_k": 0.25, "cooldown": 48, "dd_lambda": 2.0,
            "dd_soft": 0.15, "reward_mode": "absolute", "r4_beta": 0.0, "res_gamma": 0.0,
            "fwd_horizon": 24, "ungate": False, "action_mode": "discrete", "n_action_levels": 4,
            "universe_mode": "voltopk", "vol_target": 0.005, "cap_floor": 0.02,
            "harvest_obs": False, "rule_default": True, "basket_default": False, "exit_commit": 12,
            "dust_usd": 0.0, "tp_rungs": "0.25,0.5,1,2", "loss_floor": 0.2, "det_blacklist": 0,
            "scale_in": False, "cycle_obs": False, "no_btc_obs": False, "universe_lookback": 0,
            "intrabar_floor": False, "wick_reject": 0.0, "recurrent": False, "seed": 0}


@pytest.fixture(scope="module")
def recorded():
    returns, btc, _a, liq = load_data()
    vol = build_volume_panel(list(returns.columns), returns.index)
    return returns, btc, liq, vol


def test_build_selected_is_the_models_vol_top8(recorded):
    returns, btc, liq, vol = recorded
    ek = LiveEventTrader(_prov()).env_kwargs(returns)
    now_ts = int(returns.index.max())
    sel = build_selected(returns, btc, liq, vol, ek, now_ts, "ppo-event-rdLe4-sbq-3c84b4a-s1")
    assert len(sel["tokens"]) == 8                                   # k=8
    assert set(sel["caps"]) == set(sel["tokens"])
    # risk-parity caps within [cap_floor, max_entry_frac]
    assert all(0.02 - 1e-9 <= sel["caps"][t] <= 0.34 + 1e-9 for t in sel["tokens"])
    t0 = sel["tokens"][0]
    assert sel["alloc_usd"][t0] == pytest.approx(sel["caps"][t0] * 10_000.0, rel=1e-3)
    assert sel["week_start"] <= now_ts and sel["run_id"].startswith("ppo-event")


def test_slice_windows(recorded):
    returns, *_ = recorded
    assert len(_slice(returns, "last", 100)) == 100
    assert len(_slice(returns, "full", 0)) == len(returns)
    assert len(_slice(returns, "test", 0)) < len(returns)
