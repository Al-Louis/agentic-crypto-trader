"""Published-bundle diagnostics — the importable cores behind rl_compare / rl_diagnose.

These read only the **published** seed bundles from the CDN (no model replay, no tailnet) and
return JSON-able dicts. The CLI scripts (scripts/compare_seeds.py, scripts/diag_deviation_alpha.py)
are thin wrappers that pretty-print these, so the MCP tool and the terminal command can never
diverge. Network reads are injected via `fetch` so tests pass fixtures instead of hitting the CDN.

`compare_seeds`  — per-seed + across-seed mean return / maxDD / Sharpe vs the rung-0 baseline
                   (single-seed RL is unstable; the mean and its spread are the read).
`deviation_alpha` — for every EXECUTED entry, correlate over/under-sizing-vs-rule with the
                    token's forward-24h return: corr ~0 ⇒ deviating without skill ⇒ REWARD-bound;
                    clearly positive ⇒ discriminates but capped ⇒ CAPACITY-bound (vault "AI Training").
"""

from __future__ import annotations

import json
import statistics
import urllib.request
from typing import Any, Callable

import numpy as np

from trader.experiment.remote import DATA_CDN

RULE_ENTRY_FRAC = 0.20      # rung-0's fixed entry size — the deviation is measured against this
FWD_HORIZON_S = 24 * 3600   # forward-return window: 24 bars (1 day)

Fetch = Callable[[str], Any]


def _http_fetch(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=30) as r:  # noqa: S310 — fixed CDN host
        return json.load(r)


def compare_seeds(prefix: str, seeds: list[int] | list[str], *, host: str = DATA_CDN,
                  fetch: Fetch = _http_fetch) -> dict:
    """Average a seed sweep's published metrics. Returns per-seed rows + the across-seed summary.

    Skipped (not-yet-published / missing) seeds appear in `per_seed` with a `skip` note and are
    excluded from the mean. `baseline` is the config's own split baseline (e.g. vol-tilt on val).
    """
    per_seed: list[dict] = []
    rets: list[float] = []
    dds: list[float] = []
    baseline: float | None = None
    buyhold: float | None = None
    randoms: list[float] = []
    regime: dict | None = None
    gate_flags: list[bool] = []
    for s in seeds:
        rid = f"{prefix}-s{s}"
        try:
            m = fetch(f"{host.rstrip('/')}/{rid}/metrics.json")
        except Exception as e:  # noqa: BLE001 — not-yet-published / missing run
            per_seed.append({"run_id": rid, "skip": str(e)})
            continue
        r = m.get("total_return_pct")
        d = m.get("max_drawdown_pct")
        b = m.get("baseline_return")
        bh = m.get("buyhold_return")               # the honest market bar (None on pre-gate bundles)
        rnd = m.get("random_return")
        if b is not None:
            baseline = b
        if bh is not None:
            buyhold = bh
        if rnd is not None:
            randoms.append(rnd)
        if m.get("regime") is not None:
            regime = m.get("regime")
        if m.get("gate_pass") is not None:
            gate_flags.append(bool(m.get("gate_pass")))
        if r is not None:
            rets.append(r)
        if d is not None:
            dds.append(d)
        per_seed.append({
            "run_id": rid, "return": r, "maxdd": d, "sharpe": m.get("sharpe_ratio"),
            "trades": m.get("total_trades"), "baseline": b, "buyhold": bh, "random": rnd,
            "gate_pass": m.get("gate_pass"), "gate_binding": m.get("gate_binding"),
            "vs_baseline": (r - b) if (r is not None and b is not None) else None,
            "vs_buyhold": (r - bh) if (r is not None and bh is not None) else None,
        })

    out: dict[str, Any] = {"prefix": prefix, "per_seed": per_seed, "n": len(rets),
                           "baseline": baseline, "buyhold": buyhold,
                           "random": (sum(randoms) / len(randoms)) if randoms else None,
                           "regime": regime}
    if rets:
        mean = sum(rets) / len(rets)
        mean_rnd = (sum(randoms) / len(randoms)) if randoms else None
        # the honest gate at the sweep level: the seed-mean must beat rung-0 AND Buy&Hold AND Random.
        beats_bh = (buyhold is not None and mean > buyhold)
        beats_rnd = (mean_rnd is not None and mean > mean_rnd)
        beats_base = (baseline is not None and mean > baseline)
        # Buy&Hold (the market) is NON-NEGOTIABLE: a bundle without it predates the honest gate and
        # CANNOT pass — silently checking only rung-0 is exactly the drift the gate exists to kill.
        if buyhold is None:
            gate_pass_mean, binding = False, "Buy&Hold (not computed - re-run on the gated build)"
        else:
            checks = {"Buy&Hold": beats_bh, "Random": beats_rnd, "rung-0": beats_base}
            # only enforce baselines that are present, but Buy&Hold is guaranteed present here
            present = {k: ok for k, ok in checks.items()
                       if k == "Buy&Hold" or (k == "Random" and mean_rnd is not None)
                       or (k == "rung-0" and baseline is not None)}
            gate_pass_mean = all(present.values())
            binding = None if gate_pass_mean else next(k for k, ok in present.items() if not ok)
        out.update({
            "mean_return": mean,
            "spread": statistics.pstdev(rets) if len(rets) > 1 else 0.0,
            "worst_return": min(rets), "best_return": max(rets),
            "mean_maxdd": (sum(dds) / len(dds)) if dds else None,
            "worst_maxdd": max(dds) if dds else None,
            "beats_baseline": beats_base,
            "beats_buyhold": beats_bh,
            "gate_pass_mean": gate_pass_mean, "gate_binding": binding,
            "gate_pass_all_seeds": (bool(gate_flags) and all(gate_flags)),
        })
    return out


def deviation_alpha(prefix: str, seeds: list[int] | list[str], *, host: str = DATA_CDN,
                    rule_frac: float = RULE_ENTRY_FRAC, horizon_s: int = FWD_HORIZON_S,
                    fetch: Fetch = _http_fetch) -> dict:
    """Reward-bound vs capacity-bound diagnostic over executed entries (vault "AI Training").

    For each buy the agent executed, `dev = usd/equity − rule_frac` (how much it over/under-sized
    vs the rung-0 rule), paired with the token's forward-`horizon_s` return. Returns the
    correlation, the over/under-sized means, the executed entry-size range, and a verdict.
    """
    devs: list[float] = []
    frets: list[float] = []
    for s in seeds:
        rid = f"{prefix}-s{s}"
        try:
            info = fetch(f"{host.rstrip('/')}/{rid}/run_info.json")
            eqc = fetch(f"{host.rstrip('/')}/{rid}/equity_curve.json")
        except Exception:  # noqa: BLE001 — seed not published yet
            continue
        eqt = np.array([e["time"] for e in eqc], float)
        eqv = np.array([e["value"] for e in eqc], float)
        for u in info.get("universe", []):
            slug = u["slug"]
            try:
                tr = fetch(f"{host.rstrip('/')}/{rid}/tk_{slug}_trades.json")
                cd = fetch(f"{host.rstrip('/')}/{rid}/tk_{slug}_candles.json")
            except Exception:  # noqa: BLE001
                continue
            ct = np.array([c["time"] for c in cd], float)
            cc = np.array([c["close"] for c in cd], float)
            if len(ct) < 2:
                continue
            for m in tr:
                if m.get("side") != "buy":
                    continue
                t, usd = float(m["time"]), float(m["usd"])
                eq = float(np.interp(t, eqt, eqv)) if len(eqt) else 0.0
                if eq <= 0:
                    continue
                i0 = min(int(np.searchsorted(ct, t)), len(cc) - 1)
                i1 = min(int(np.searchsorted(ct, t + horizon_s)), len(cc) - 1)
                if i1 <= i0 or cc[i0] <= 0:
                    continue
                devs.append(usd / eq - rule_frac)
                frets.append(cc[i1] / cc[i0] - 1.0)

    out: dict[str, Any] = {"prefix": prefix, "n_entries": len(devs)}
    if len(devs) <= 3:
        out["verdict"] = "inconclusive (too few executed entries)"
        return out
    dv, fr = np.array(devs), np.array(frets)
    if np.std(dv) < 1e-12 or np.std(fr) < 1e-12:   # degenerate: no variance to correlate
        out["verdict"] = "inconclusive (no variance in deviation or forward return)"
        return out
    corr = float(np.corrcoef(dv, fr)[0, 1])
    over, under = fr[dv > 0], fr[dv <= 0]
    out.update({
        "corr": corr,
        "over_mean": float(over.mean()) if len(over) else None, "over_n": int(len(over)),
        "under_mean": float(under.mean()) if len(under) else None, "under_n": int(len(under)),
        "entry_size_min": float(dv.min() + rule_frac),
        "entry_size_max": float(dv.max() + rule_frac),
        # corr ~0 ⇒ deviates without skill ⇒ the REWARD isn't teaching discrimination.
        "verdict": ("capacity-bound (discriminates but capped)" if corr > 0.10
                    else "reward-bound (deviates without skill)" if abs(corr) <= 0.10
                    else "inverse (over-sizes losers)"),
    })
    return out
