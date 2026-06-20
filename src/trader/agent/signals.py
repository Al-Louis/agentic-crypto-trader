"""Decision-tape tally for the dashboard — `trading/signals.json`.

How many ignition SIGNALS the agent saw, EXECUTED, and IGNORED per UTC day this cold week, so you
can watch ef-s2's participation rate without an ad-hoc replay. INFORMATIONAL: a separate
instrumented replay of the current cold week (reuses `EventRungEnv` + a cold predictor, same
determinism as the harness) — it does NOT touch the trading path (which uses
`evaluate_event_policy` verbatim). A "signal the agent saw" is an ignition that passed the rung-0
gates (flat + cooled + reclaimed) and became an entry decision; most raw ignitions are gated before
the agent ever sees them. See [[Live Forward-Run Harness]] / [[Apentic Data Contract]] §trading/.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from trader.agent import event_live as _el  # noqa: F401 — its import puts scripts/ on sys.path


def _utc(ts) -> str:
    return datetime.fromtimestamp(int(ts), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def replay_decisions(trader, returns, btc, liq, vol, now_ts: int, env_kwargs: dict, *,
                     predict_fn=None) -> tuple[int, list[dict]]:
    """Instrumented replay of the current cold week: one record per decision EVENT
    (entry/exit/profit) — type, token, bar time, action idx, and whether it executed. `predict_fn`
    injectable for tests; production uses a cold-LSTM predictor (matches the harness)."""
    from trader.agent.event_live import WARMUP, cold_week_window  # noqa: PLC0415
    from trader.train.event_env import EventRungEnv  # noqa: PLC0415
    win, ws, _i0 = cold_week_window(returns, int(now_ts))
    kw = {k: v for k, v in env_kwargs.items() if k != "episode_bars"}
    env = EventRungEnv(win, btc, liq, volume=vol, episode_bars=len(win) - WARMUP - 1, **kw)
    obs = env.reset(start=WARMUP)
    pf = predict_fn if predict_fn is not None else trader._predict_fn()
    out: list[dict] = []
    done = False
    while not done:
        etype, tok = env._pending
        bar_ts = int(env.returns.index[env.bar])
        a = pf(obs)
        idx = int(round(float(np.asarray(a).reshape(-1)[0])))
        obs, _r, done, info = env.step(a)
        if etype in ("entry", "exit", "profit"):
            out.append({"type": etype, "token": tok, "time": bar_ts, "time_utc": _utc(bar_ts),
                        "action_idx": idx, "executed": bool(info.get("trades"))})
    return int(ws), out


def tally(ws: int, events: list[dict], *, generated: str | None = None) -> dict:
    """Bucket the decision tape into per-UTC-day seen/executed/ignored (entries) + exits, with
    week totals and the participation rate. Pure."""
    generated = generated or datetime.now(timezone.utc).isoformat()
    days: dict[str, dict] = {}
    for e in events:
        d = e["time_utc"][:10]
        b = days.setdefault(d, {"date": d, "signals_seen": 0, "executed": 0, "ignored": 0, "exits": 0})
        if e["type"] == "entry":
            b["signals_seen"] += 1
            b["executed" if e["executed"] else "ignored"] += 1
        elif e["type"] == "exit":
            b["exits"] += 1
    seen = sum(d["signals_seen"] for d in days.values())
    ex = sum(d["executed"] for d in days.values())
    return {"generated": generated, "week_start": int(ws),
            "totals": {"signals_seen": seen, "executed": ex, "ignored": seen - ex,
                       "exits": sum(d["exits"] for d in days.values()),
                       "participation": round(ex / seen, 3) if seen else None},
            "days": [days[k] for k in sorted(days)], "events": events}


def publish_signals_tally(trader, target: str, now_ts: int, *, predict_fn=None) -> dict:
    """Load panels, replay the week's decision tape, tally, and PUT `<target>/signals.json`
    (target already ends in the `trading` prefix). Returns the tally."""
    from remote_train.publish import join, put_bytes  # noqa: PLC0415
    from train_rl import build_volume_panel, load_data  # noqa: PLC0415
    returns, btc, _a, liq = load_data()
    vol = build_volume_panel(list(returns.columns), returns.index)
    ek = trader.env_kwargs(returns)
    ws, events = replay_decisions(trader, returns, btc, liq, vol, int(now_ts), ek, predict_fn=predict_fn)
    t = tally(ws, events)
    put_bytes(join(target, "signals.json"),
              json.dumps(t, separators=(",", ":")).encode("utf-8"),
              content_type="application/json", cache_control="no-cache")
    return t
