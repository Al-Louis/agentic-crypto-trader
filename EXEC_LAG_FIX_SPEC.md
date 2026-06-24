# Fix Spec: Make strategy signals act on the just-closed bar at the same tick

## 0. Verdict on the hypothesis (reconciling the four traces)

The hypothesis — `+1` (bar must close to enter panel) and `+1` ("act on next bar i+1") — is **HALF right, and the four traces conflict on the second half. The code resolves it.**

- **Bar A (+1, finalization): CONFIRMED by all four traces and by code.** `live_data.finalized_bars` admits a bar only when `ts + 3600 <= now` (open-stamped candles per `geckoterminal.py:59`), and `cold_week_window` (`event_live.py:78-85`) slices up to `cur` = latest bar `<= now_ts`. At an `HH:10` tick the newest finalized bar is the one that opened `(HH-1):00` / closed `HH:00`. Irreducible without lookahead.

- **Bar B: the hypothesis's wording ("execute a bar-i signal only when it steps i+1") is WRONG, but there IS a real second env bar.** Traces 2 & 3 claim the second bar is a runner/tick-cadence artifact and the env has no second `+1`; traces 1 & 4 locate it in the env's window-edge math. **The code proves traces 1 & 4 are correct.** Two distinct facts, both true:
  - **Fill PRICE is same-bar (no i+1 price shift).** `step` fills at `self._px[self.bar, j]` and stamps `int(self.returns.index[self.bar])` (`event_env.py:395, 572-573, 651-652`), where `self.bar` is the detection bar. There is no next-open execution. This is what traces 2 & 3 verified, and it is correct — but it is *not* the question.
  - **The env never SCANS its last window bar.** `evaluate_event_policy` sets `episode_bars = len(eval_r) - WARMUP - 1` (`train_event.py:42`), so `reset(start=WARMUP)` gives `self.end = WARMUP + (len-WARMUP-1) = len-1` (`event_env.py:319`). In `_advance_to_event` (`event_env.py:444-475`) the done-check `if self.bar >= self.end` (line 464) **returns at line 466-467 BEFORE `_scan_bar(self.bar)` at line 471.** So the last bar ever scanned is `end-1 = len-2`. The window's final bar `len-1` (= `cur` = the just-closed bar) is computed in the signal arrays but never turned into a pending event, never stepped, never filled.
  - First-bar handling confirms the asymmetry: `reset` calls `_advance_to_event(first=True)`, which does `self.bar += 0` then *does* scan `start` (line 452→471). So the env scans `[start, end-1]` inclusive and skips exactly the terminal bar `end`.

**Net mechanism, reconciled with +132min vs +2min:**
- A signal on the bar closing at `HH:00` enters the panel at the first tick `>= HH:00` that also clears Gecko's finalize lag — `HH:10` tick (Bar A).
- At that tick it is the window's *terminal* bar `len-1` → unscanned (Bar B). It only becomes interior (and thus scanned/filled/signed) on the *next* tick, `(HH+1):10`, once a later bar enters the window.
- Plus the Gecko per-pool settle race (P90 376s, slowest active 465s; `event_agent.py:36-42`) can push a thin-pool bar's finalization past the `HH:10` window into the `(HH+1):10` tick. That is the extra hour that makes the observed worst case `(HH+2):11` rather than `(HH+1):11`.
- **COMPLIANCE (+2min)** bypasses the env entirely: `_run_compliance` → `compliance_action(now_ts)` with `bar_ts = now_ts` (`event_runner.py:374-459`), computed at the tick wall-clock — no finalization wait, no terminal-bar exclusion.

**The eliminable bar is B (the env's terminal-bar exclusion). A is intrinsic; the fast CMC feed only lets us move the tick offset earlier so A's wall-clock cost shrinks toward ~HH:02.**

---

## 1. The minimal fix

Two independent, composable changes. The risk lives entirely in change (1).

### Change (1) — scan the window's last bar (eliminate Bar B), LIVE-PATH ONLY

The clean lever is `episode_bars`: make `end = len(eval_r) - WARMUP` (one larger) so the real last bar `len-1` becomes `end-1` (the last *scanned* bar) and the done-check fires at `len` (one past the array, never reached on a real bar). **No lookahead is introduced** — the env still only ever reads `self._px[self.bar, j]` (the just-closed bar's close) and `_scan_bar` reads `[bar, j]`; it never indexes `bar+1`.

Do NOT edit `train_event.py:42` directly (that path is shared by training, the offline `simulate_weekly` cold-weekly metric, AND the consumed frozen-TEST cert of sbq-s1 — see §3 train/serve). Add a keyword that the LIVE caller sets and the training/eval caller does not:

`scripts/train_event.py` — `evaluate_event_policy`:
```python
def evaluate_event_policy(predict_fn, eval_r, btc, liq, vol, env_kwargs, *, act_last_bar=False):
    ...
    headroom = 0 if act_last_bar else 1          # was hard-coded -1
    env = EventRungEnv(eval_r, btc, liq, volume=vol,
                       episode_bars=len(eval_r) - WARMUP - headroom,
                       record_trace=True, **kw)
```
Note `EventRungEnv.__init__` line 283 computes `_max_start = n_bars - episode_bars - 1`; with `episode_bars = len - WARMUP` and the eval `reset(start=WARMUP)`, `_max_start = WARMUP - 1 < _min_start = WARMUP` would trip the guard at line 284. So the live path must construct with this larger `episode_bars` AND only ever `reset(start=WARMUP)` (it already does — single full-window episode). Add `episode_bars` to the live env build so the `_max_start` guard is satisfied: since eval uses an explicit `start`, relax the guard to allow `_max_start == _min_start - 1` only on the act-last-bar eval construction, OR (cleaner) keep `episode_bars = len - WARMUP - 1` for the `_max_start` math and instead bump `self.end` after reset. The lowest-blast-radius form is a reset-time flag:

`src/trader/train/event_env.py` — `reset` (after line 319):
```python
self.end = self.start + self.episode_bars
if getattr(self, "act_last_bar", False):
    self.end += 1                         # scan the terminal window bar (live: the just-closed bar)
```
and set `self.act_last_bar` in `__init__` from a new kwarg (default `False`). This leaves `_max_start`/`episode_bars` and ALL training math byte-identical; it only extends the done boundary by one at reset when the live flag is on. **This is the preferred shape** — it cannot affect `_max_start`, curriculum, or any training reset.

`event_live.py` — `LiveEventTrader.evaluate_week` passes the flag through to `evaluate_event_policy`/the env build (one-line plumb), defaulting OFF so offline `simulate_weekly` is untouched.

### Change (2) — move the tick offset to ~HH:02, GATED on the CMC feed cutover

The offset is already CLI-configurable (`event_agent.py:82 --tick-offset-secs`). Do NOT lower it on the current GeckoTerminal path — `DEFAULT_TICK_OFFSET=600` exists precisely because Gecko's slowest active pool finalizes at 465s; `HH:02` would re-introduce the 180s→600s miss-the-bar-for-an-hour race (`event_agent.py:36-42`). The substantive change is in the live-data source (`live_data.fetch_alt_latest`/`update_live`) to the fast CMC feed (~30s finalize), THEN set `--tick-offset-secs 120`. Ship both in the same deploy; the offset move is inert without the feed swap.

With both changes: signal bar closes at `HH:00` → CMC finalizes it by ~`HH:00:30` → `HH:02` tick sees it as the window's terminal bar → `act_last_bar` scans it → dedup-signs it → tx ~`HH:02`. Matches compliance.

---

## 2. Safety (adversarial)

**LOOKAHEAD — proven clean.** The just-closed bar is *fully finalized* (`live_data` admits it only after `ts+3600<=now`; it is a closed candle, identical content to what the env consumes one tick later today). `_scan_bar(bar)` reads only `self._ignite[bar,j]`, `self._cush[bar,j]`, `self._px[bar,j]`, `self._lowf[bar,j]` — all `[bar, ...]`, never `[bar+1, ...]` (verified `event_env.py:484-535`). The fill prices at `self._px[self.bar]` (close of the just-closed bar). `act_last_bar` only changes *which bar is the terminal boundary*, not which array indices are read. **The single lookahead trap:** forward-maturation reward modes. `_mature_entries` (gated `reward_mode=="entry_forward"`, line 469-470) and `_ignition_base_rate`/`_mu_base` (line 280) read forward windows; `entry_forward`/`residual_ranked` could touch a non-existent `bar+horizon` if the terminal bar is scanned. **sbq-s1 is `reward_mode="absolute"`** (no forward reads), so it is safe — but the flag MUST assert/guard against forward modes:
```python
if self.act_last_bar and self.reward_mode in ("entry_forward",):
    raise ValueError("act_last_bar unsafe with forward-maturation reward modes")
```

**DOUBLE-SIGN / DEDUP — unchanged and proven safe.** The runner records via identity-dedup `(bar_ts, token, side)` (`event_runner.py:526-534`), reading `_recorded_fill_keys(ws)` from the ledger each tick. A just-closed-bar fill at `bar_ts=(HH-?):00` is first-seen → not in `recorded_keys` → passes dedup → signed (line 553). On every later tick the deterministic replay re-emits the identical fill, but its key is now in the ledger → skipped at line 532. This is the exact post-deploy-drop-incident design (the forward cursor that dropped sbq-s1's back-dated week-open ignition was *replaced* by this key set). **Acting one bar earlier cannot duplicate** (key is recorded the first time it's signed) **and cannot drop** (the key set is rebuilt from disk every tick; nothing is skipped by a cursor). The sell-side `_onchain_held` guard (lines 549-551) still fires identically.

**DQ / COMPLIANCE — no impact.** Compliance is a separate sleeve computed at the tick (`_run_compliance`), untouched by env changes; the `>=1-trade/day` floor still fires. Drawdown gate reads `self._week_peak_eq`/`equity` from the full replay equity curve (`event_runner.py:498-503`) — acting on the terminal bar *adds* its equity mark to the curve (it was already recorded by `record_trace` at line 462-463 even when unscanned), so the DD anchor is unchanged or strictly more current. No new DQ exposure; if anything the agent now exits a falling position one bar sooner.

**TRAIN/SERVE.** Today backtest==live==cert because all three use `evaluate_event_policy` with `episode_bars = len-WARMUP-1` (the terminal bar is excluded *everywhere*). The fix deliberately makes LIVE diverge by one terminal bar. This is acceptable and the **fill-price convention does NOT change** (still bar-close), so it is "act on the just-closed bar at its close price" — exactly the convention the env already validated, just applied to the one bar it used to drop. The divergence is one bar at the very end of each window. Two postures:
- **(Recommended, minimal) Live-only flag**, leave the consumed sbq-s1 cert untouched (no re-cert of a spent test). The behavioral delta is in-distribution: live acts on a bar whose obs the cert's model already consumed one tick later. Document it as a known, bounded train/serve delta.
- **(Rigorous, optional)** also run the offline `simulate_weekly`/cert harness WITH `act_last_bar=True` to confirm sbq-s1's frozen-TEST numbers move within a one-final-bar tolerance — for confidence only; do not re-tune to it (test is consumed).

---

## 3. Test plan

Existing coverage: `tests/test_event_env.py` (env emission/sequencing), `tests/test_event_runner.py` + `tests/test_live_event_agent.py` (dedup/sign loop, `_recorded_fill_keys`), `tests/test_event_live.py` (`cold_week_window`, `evaluate_week` determinism), `tests/test_simulate_weekly.py` (the cold-weekly metric — must stay byte-identical with flag OFF).

New/changed tests:
1. **`test_event_env.py::test_act_last_bar_scans_terminal`** — synthetic panel with an ignition placed on the LAST window bar (`len-1`). Assert: default env (flag OFF) emits NO fill at `index[len-1]`; flag ON emits exactly one entry at `time==index[len-1]`, price `==_px[len-1]` (close, not `_px[len-1+1]` which doesn't exist). This directly asserts same-tick action on the just-closed bar with no lookahead.
2. **`test_event_live.py::test_evaluate_week_act_last_bar_default_off`** — assert `evaluate_week` with default args reproduces current fills byte-for-byte (regression guard on the cert path).
3. **`test_event_runner.py::test_just_closed_bar_signs_once_no_dup`** — two consecutive ticks (`now=HH:02` then `(HH+1):02`) over a deterministic stub where the terminal bar carries a signal with `act_last_bar` ON. Assert: the fill signs on the FIRST tick; on the second tick the same `(bar_ts, token, side)` is in `_recorded_fill_keys` and is NOT re-signed (recorded count unchanged). Covers the deploy-drop / double-sign class.
4. **`test_event_env.py::test_act_last_bar_rejects_forward_reward`** — assert the `entry_forward` guard raises (lookahead protection).
5. **`test_event_agent.py`** — assert `seconds_until_next_tick(now, offset=120)` targets `HH:02` (offset plumbing only; pure function already unit-tested).

Run full suite (`524+ pass` baseline per memory) plus the on-box `--once --no-refresh` dry-run gate against recorded data, confirming a terminal-bar signal signs at the same tick.

---

## 4. Deploy / rollback (live, mid-window)

**Ship as ONE tested deploy** (do not split — the offset move is inert without the feed):
1. Land changes (1) env `act_last_bar` flag + plumb, (2) CMC feed in `live_data` + `--tick-offset-secs 120`, on a branch; full suite + dry-run green.
2. On the box: `git fetch && git checkout <sha>`, confirm HEAD==pushed sha, run `--once --now <recent_ts> --no-refresh` dry-run gate (loads sbq-s1, ticks, signs the terminal-bar fill, dedup holds).
3. Repoint the service to pass `--act-last-bar` (new CLI flag, default off so the binary is safe pre-flag) and `--tick-offset-secs 120`, restart. Watch one live tick land at `HH:02`.
4. **Ledger hygiene (recall the deploy-drop incident):** do NOT hand-edit the JSONL. The dedup key set is rebuilt from disk each tick, so the change is self-consistent on restart — no manual ledger surgery needed.

**One-line rollback:** drop `--act-last-bar` (and reset `--tick-offset-secs 600`) on the service unit and restart — the env reverts to the validated terminal-exclusion convention and the GeckoTerminal-safe offset; no code revert, no ledger change. (Already-signed terminal-bar fills stay in the ledger and are simply not re-emitted.)

---

## Things to verify in code before changing (flagged conflicts)

- **Confirm sbq-s1's `reward_mode`** in its `metrics.json` provenance is `"absolute"` (memory says so) before enabling `act_last_bar` — the forward-mode guard depends on it.
- **Confirm the CMC feed actually finalizes the just-closed bar in <120s** for the thinnest *traded* pool (re-run `scripts/calibrate_gecko_lag.py`-equivalent against CMC). The `HH:02` offset is only safe if this holds; otherwise keep 600s and accept that change (1) alone still removes one full bar (~1h) of lag.
- **The two refuting traces (2,3) are wrong on Bar B's location** — they conflate "fill price is same-bar" (true) with "the last window bar is acted on" (false). Implement against traces 1 & 4 (env terminal-bar exclusion), which match the code at `event_env.py:464-471` + `train_event.py:42`.

Key files: `src/trader/train/event_env.py` (`reset` L314-356 add `act_last_bar` end-bump; `_advance_to_event` L444-475 the done-check; `__init__` L283 `_max_start`), `scripts/train_event.py:42` (`evaluate_event_policy` episode_bars), `src/trader/agent/event_live.py:175-186` (`evaluate_week` plumb) + `:61-86` (`cold_week_window`), `src/trader/agent/event_runner.py:526-534` (identity-dedup, unchanged), `src/trader/agent/event_agent.py:36-42,82` (tick offset), `src/trader/agent/live_data.py:45-50` (finalization, intrinsic Bar A; CMC feed swap point).