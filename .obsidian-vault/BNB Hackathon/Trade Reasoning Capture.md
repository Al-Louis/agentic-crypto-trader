# Trade Reasoning Capture — spec

A per-trade "why" annotation surfaced in the frontend (hover/select a marker → see what triggered it and the state the agent acted on). Spec'd 2026-06-17. Companion to [[Apentic Data Contract]] (the `token_trades` schema this extends) and [[Simulated Market]] (the dashboard). Build posture: **additive, eval/reporting-only, byte-identical to training** — the same low-risk class as the loss_floor-marker / token_pnl.json / marker-on-candle fixes.

## Goal & honest framing

Each trade gets a structured explanation with **two layers** — only the first is an enum:

1. **The TRIGGER (deterministic — the enum).** The rung-0 rule generated the decision point. Fully enumerable and accurate. This is the user's idea ("BUY — volume-increase ignition signal"), and for *sells* it is exactly what we keep reverse-engineering by hand (the ZEC Apr-3 forensic established those were discretionary `EMA_BREAK`s — this makes that a hover, not an investigation).
2. **The agent's DISCRETION (the learned part — context, not a reason).** On that trigger the LSTM chose a size (entries) or hold-vs-sell (exits), conditioned on the obs. Not symbolic — but we record the **state it saw** + the **action it took** + a **FORCED/DISCRETIONARY** flag.

> **Do NOT label this "the agent's reasoning."** The policy has no chain-of-thought to extract. The honest, and stronger, framing is *"trigger + state + action."* The FORCED/DISCRETIONARY split is what gives it explanatory teeth (where the agent exercised judgment vs. where a hard rail fired) without fabricating intent.

## Enum taxonomy (canonical string codes + display labels)

Use a **string code** as the reference (self-documenting, no index table to drift); a human label maps for display. (The user's "number in a list" works too, but the string is more robust to reordering.)

**Entry triggers**
| code | label | fires when |
|------|-------|-----------|
| `IGNITION` | Volume-ignition signal | fresh rung-0 ignition (surge ≥2.5× base, rising>0, cushion>0, uptrend) on a flat, cooled, reclaimed token |
| `SCALE_IN` | Add to held winner (re-ignition) | fresh ignition on an already-held in-profit token, under cap (`scale_in` flag; currently default-OFF/refuted) |
| `BASKET_OPEN` | Basket open | `basket_default` overlay open at reset (shelved) |

**Exit triggers**
| code | label | fires when | forced? |
|------|-------|-----------|---------|
| `EMA_BREAK` | Trend break (price below EMA) | `cush<0` (`_scan_bar` L396) | discretionary |
| `TRAILING_STOP` | Trailing stop (gave back 25% from peak) | `px < peak·(1−stop_k)` (L394) | discretionary |
| `PROFIT_TAKE` | Took profit (hit +X% rung) | `unreal ≥ tp_rungs[tp_i]` (L405) | discretionary |
| `LOSS_FLOOR` | Disaster floor (−20% from cost) | `unreal < −loss_floor` (L385/L474) | **FORCED** |
| `INTRABAR_STOP` | Resting-stop fill (intrabar) | intrabar low crossed the floor (`_stop_fill`) | **FORCED** |
| `ROTATION_OUT` | Rotated out to fund a stronger ignition | weakest-cushion holding sold by `_rotate_for` | **FORCED** |

**FORCED** = `{LOSS_FLOOR, INTRABAR_STOP, ROTATION_OUT}` (executes regardless of the agent). **DISCRETIONARY** = `{EMA_BREAK, TRAILING_STOP, PROFIT_TAKE}` (the agent *took the prompt — it could have held*) and all entries (the agent *sized* it). Derive the flag from the code; don't store it twice.

## Where each trigger is decided (code map, current `event_env.py`)

- **`_scan_bar` L385–398** — the exit trigger is determined here:
  - `floored` (L385/391) → `LOSS_FLOOR` (punctures the `exit_commit` window)
  - `stop_hit = px < peak·(1−stop_k)` (L394) → `TRAILING_STOP`
  - `ema_hit = cush<0` (L396) → `EMA_BREAK`
  - **precedence when more than one is true:** `LOSS_FLOOR > TRAILING_STOP > EMA_BREAK` (and set a `both_stop_ema` boolean on the obs snapshot when stop+ema co-fire, so the forensics aren't lossy).
- **`_scan_bar` L405–406** → `PROFIT_TAKE`. **L419–420** → `SCALE_IN`; **L425–426** → `IGNITION`.
- **`_do_exit` L474–477** — the internal disaster-floor re-check force-sells via `_sell_down(tok,0.0)` regardless of the prompt; this **overrides** the carried reason to `LOSS_FLOOR`.
- **`_rotate_for`** — calls `_do_exit(weak, 0.0)`; tag this sell `ROTATION_OUT` (a funding side-effect, not a deliberate exit). *Edge case:* if `weak` is itself past the floor, `LOSS_FLOOR` takes precedence (it's a forced cut either way).
- **`_stop_fill`** — `INTRABAR_STOP`.

## Marker sites (the 4 `_trades.append` calls — current tuple = `(tok, usd, fee, time, px)`)

| line | path | trade |
|------|------|-------|
| L462 | `_do_entry` | BUY (IGNITION or SCALE_IN) |
| L512 | `_stop_fill` | SELL (INTRABAR_STOP) |
| L531 | `_sell_down` | SELL — the **shared** path (exit cut, trim, loss-floor force, rotation) |
| L661 | `_buy_basket` | BASKET_OPEN |

## Mechanism — thread the reason from trigger → marker

The trigger is known in `_scan_bar`; the marker is written in `_sell_down`/`_stop_fill`/`_do_entry`. Thread it:

1. **`_scan_bar`** emits the reason with the event: `("exit", t, reason)`, `("entry", t, reason)`, `("profit", t)` (profit ⇒ always `PROFIT_TAKE`). (Today it emits 2-tuples; widen to carry the reason.)
2. **The step dispatch** (the loop consuming `self._queue`) passes the reason into `_do_exit(tok, a, reason)` / `_do_entry(tok, a, reason)` (`_do_profit` is implicitly `PROFIT_TAKE`).
3. **`_do_exit`**: the floor-force branch overrides to `LOSS_FLOOR`; otherwise pass the carried reason to `_sell_down(tok, keep, reason)`. **`_rotate_for`** calls `_do_exit(weak, 0.0, ROTATION_OUT)`. **`_stop_fill`** uses `INTRABAR_STOP`. **`_do_entry`** uses `SCALE_IN` if `held` else `IGNITION`.
4. **`_sell_down` / `_stop_fill` / `_do_entry`**: append the extended tuple `(tok, usd, fee, time, px, reason, obs)`.

`obs` = a small dict captured at the marker bar from slots the env already has:
`{surge, cush, giveback (px/peak−1), unreal (px/cost_px−1), held_frac, surge_decay, action}` — where `action` is the agent's chosen size-multiple (entries) or keep-fraction (exits). This is the "state the agent acted on."

> **Stage 1 status (shipped 2026-06-17, UNCOMMITTED).** Env trace landed in `src/trader/train/event_env.py`: module-level reason enums + `FORCED_REASONS`, `_scan_bar` widened to `(etype, tok, reason[, both_stop_ema])`, `_set_pending` normalizes back to a 2-tuple `_pending` + `_pending_reason`/`_pending_both` (so every `etype,tok=self._pending` consumer is untouched), and the reason is threaded dispatch → `_do_exit`/`_do_entry` → `_sell_down`/`_stop_fill`. The 4 marker sites now append `(tok, usd, fee, time, px, reason, obs)`. `obs` carries `{surge, cush, giveback, unreal, held_frac, action, both_stop_ema}` — **`surge_decay` is DEFERRED** (it needs new per-position state tracking the surge at entry vs now; recorded here as a follow-up, not in Stage 1). `both_stop_ema` is passed explicitly by the stop/ema exit path (False on every other marker, so a stale co-fire can't leak into an intrabar/rotation fill). The single positional unpacking consumer (`scripts/train_event.py evaluate_event_policy`, both the opening-buy emit and the per-step path) now unpacks the 2 new fields into the fill dict as `reason`/`obs`. Tests in `tests/test_event_env.py` cover every path; numbers are byte-identical (golden return/DD/trade-count test). Stage 2 (export schema) + Stage 3 (frontend) still pending.

## Data contract (env → eval → export → frontend)

- **env** (`event_env.py`): `_trades` tuple → `(tok, usd, fee, time, px, reason, obs)`. (6 edits: the 3 marker sites + `_scan_bar` event widening + the dispatch + `_rotate_for`.)
- **eval** (`train_event.evaluate_event_policy`): the fill dict already unpacks the tuple — extend `{"token","usd","fee","time","px"}` with `"reason"`, `"obs"`. The opening-buy emit (right after `reset()`) and the per-step `info["trades"]` path both need the new fields.
- **export** (`apentic.export_portfolio_run` → `token_trades`, via `train_rl.build_portfolio_artifacts`): add `reason` (string code), `forced` (bool, derived), and `obs` (object) to each trade record. Also surface the human `label` (or let the frontend map it).
- **frontend**: on hover/select, render `label` + the FORCED/DISCRETIONARY badge + the `obs` context (e.g. *"Trend break · DISCRETIONARY · surge 1.8×, cushion −1.5%, +0% vs cost, held 14h, agent: full exit"*). Keep a code→label map client-side so labels can be reworded without a re-export.
- **backward-compat:** pre-spec bundles lack the fields → frontend shows "—" / "(legacy)". Don't hard-require them.

## Optional: action conviction (richer discretion layer)

Capture the policy's action *distribution* at the decision (discrete 4-level → the level probs) to show conviction ("strongly favored full size" vs "marginal"). Needs the policy at eval — extend `make_predict`/`predict_fn` to also return the distribution (today it returns only the argmax action). Modest add; do it after the core lands.

## Risk & scope

- **Byte-identical to training.** `_trades` is recording-only; the reward is equity-based. Adding fields to the markers changes nothing the policy optimizes. Eval/reporting-only.
- **It operationalizes the forensics.** EMA_BREAK vs TRAILING_STOP vs LOSS_FLOOR vs ROTATION_OUT is exactly the disambiguation the ZEC and wsi forensics did by hand — this bakes it in (and would have made that dig a glance).
- **Orthogonal** to the alpha hunt and the frozen-test decision — a showcase + diagnostics asset, not a PnL lever. For an all-or-nothing showcase, an honest "trigger + state + action + forced/discretionary" panel is a strong narrative.

## Build phasing

1. **Env trace** — thread `reason` + `obs` through `_scan_bar`/dispatch/`_do_*`/`_sell_down`/`_stop_fill`/`_rotate_for` into the marker tuple; tests asserting each path tags the right code (ignition→IGNITION, ema-only exit→EMA_BREAK, stop→TRAILING_STOP, floor→LOSS_FLOOR, intrabar→INTRABAR_STOP, rotation→ROTATION_OUT, tp→PROFIT_TAKE) and the FORCED flag.
2. **Eval + export** — thread the fields through `evaluate_event_policy` + `export_portfolio_run` + the `token_trades` schema; re-export one bundle and eyeball the markers.
3. **Frontend** — the hover/select panel + the code→label map + the FORCED badge + the obs context; legacy fallback.
4. **(optional)** action-conviction via the policy distribution.
