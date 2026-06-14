# Apentic — Simulated Trades · Handoff

A self-contained dashboard that renders a **6-month / multi-week back-test** of the
trading agent: a weekly equity hero, a per-week drill-down (hourly candles +
executions per asset), a 6-month token P&L table, and **competition-compliance
checks** for the two hard disqualifiers.

This page is **read-only over a single JSON file**. Your simulator's job is to emit
that JSON; the page computes every derived metric itself.

---

## 1. Quick start

1. Emit your back-test as a JSON file in the schema below.
2. Save it next to the HTML as **`simulated_trades.json`**.
3. Serve the folder over HTTP (the page `fetch()`es the JSON — `file://` is blocked by browsers):
   ```bash
   cd <this folder>
   python3 -m http.server 8000
   # open http://localhost:8000/Apentic%20Simulated%20Trades.html
   ```
4. The source badge (top-right of the header) flips to **“● live: simulated_trades.json”**.
   If the file is missing or invalid, the page falls back to bundled **sample data**
   and shows **“○ sample data”**.

`simulated_trades.sample.json` in this folder is a complete, valid example (the same
data the bundled mock generates). **Rename a copy to `simulated_trades.json` to see it load.**

---

## 2. File manifest

| File | Purpose |
|---|---|
| `Apentic Simulated Trades.html` | Entry point. Loads everything, fetches the JSON, renders. |
| `apentic-backtest.js` | **The data layer.** `Apentic.computeBacktest(raw)` derives the model; also holds the mock generator. This is the file to read to understand the contract. |
| `apentic-sim-charts.jsx` | Equity columns, mini candles, big hourly candle chart, session-equity sparkline. |
| `apentic-sim-panels.jsx` | Headline stats, compliance cards, week detail, trade log, token table. |
| `apentic-sim.css` | Styles specific to this page. |
| `apentic.css` | Shared Apentic design system (colors, type, nav, ticker). |
| `apentic-components.jsx` | Shared nav, ticker, logo, formatters. |
| `apentic-charts.jsx` | Shared `Sparkline` (used by the token table). |
| `apentic-data.js` | Shared ticker/nav reference data (price ticker tape). |
| `tweaks-panel.jsx` | Accent/grain tweak panel shell. |
| `assets/ape_avatar.png` | Logo. |
| `simulated_trades.sample.json` | Reference input (full 24-week example). |

React, ReactDOM and Babel load from CDN (pinned). No build step.

> **Nav note:** in this standalone export the Overview / Markets / Leaderboard tabs
> are inert (this is the only page). Re-point `navTo()` in the HTML when you integrate
> into the full app.

---

## 3. JSON schema (what your simulator emits)

Top level:

```jsonc
{
  "meta": { ... },
  "weeks": [ { ...week }, ... ]   // chronological, one entry per session
}
```

### 3.1 `meta`

| Field | Type | Req | Meaning |
|---|---|:--:|---|
| `start_capital` | number (USD) | rec | Account value at the start of the back-test. Default `10000`. |
| `window_start` | unix sec | opt | Back-test window start (header display). Defaults to first week's `start`. |
| `window_end` | unix sec | opt | Back-test window end. Defaults to last week's `end`. |
| `n_weeks` | int | opt | Informational. |
| `candle_interval_seconds` | int | opt | Bar size. `3600` = hourly (the reference). The page also infers it from candle timestamps. |
| `drawdown_limit` | number | opt | Rule-2 limit as a **negative fraction**. Default `-0.30` (−30%). |
| `universe_size` | int | opt | Total tradable universe (for the “of N universe” stat). Defaults to the count of distinct symbols seen. |
| `generated` | string | opt | ISO timestamp, informational. |

### 3.2 `weeks[]` — one per session

| Field | Type | Req | Meaning |
|---|---|:--:|---|
| `index` | int | opt | 0-based order. |
| `label` | string | opt | Column / header label. Default `"W01"`, `"W02"`, … |
| `start` | unix sec | **yes** | Session open time. Day buckets for **Rule 1** are measured from here. |
| `end` | unix sec | opt | Session close time. Default `start + 7 days`. Number of days = `round((end-start)/86400)`. |
| `portfolio_start` | number (USD) | rec | Account value at session open. If omitted, it chains from the previous week's computed close (first week uses `meta.start_capital`). **Provide it** to avoid drift. |
| `assets` | array | **yes** | The assets traded this session — **8 expected** (the grid + detail are tuned for 8; other counts still render). |

### 3.3 `assets[]` — one per traded token per session

| Field | Type | Req | Meaning |
|---|---|:--:|---|
| `symbol` | string | **yes** | Ticker, e.g. `"HUMA"`. |
| `class` | `"alt"` \| `"major"` \| `"peg"` | opt | Dot color + label (Altcoin / Major / Gold-pegged). Default `"alt"`. |
| `vol_rank` | int (1–8) | opt | Volatility rank used to order the selection (1 = most volatile). Default = array order. |
| `alloc_usd` | number (USD) | rec | Capital allocated to this asset this session (denominator for the asset's return %). Defaults to the sum of position notionals. |
| `candles` | array | **yes** | OHLCV bars, chronological, uniform interval (168 hourly bars per 7-day week is the reference). |
| `positions` | array | **yes** | Round-trip trades (entry + exit). May be empty. |

**Candle** `{ "t", "o", "h", "l", "c", "v" }`
- `t` unix sec (bar open). `o/h/l/c` prices. `v` relative volume (any positive scale — only relative heights are drawn).

**Position** `{ "entry_t", "entry_price", "exit_t", "exit_price", "qty", "kind" }`
- `entry_t` / `exit_t` — unix sec; snapped to the nearest candle for markers + shaded holding region. `entry_t < exit_t`.
- `entry_price` / `exit_price` — actual execution prices (may differ from the candle close → models slippage).
- `qty` — units of the base asset.
- `kind` — `"core"` (directional position, counted in win-rate) or `"scalp"` (light activity; counts as a trade and toward Rule 1, **excluded** from win-rate). Default `"core"`.

> A “trade/execution” for **Rule 1** = a position **entry or exit**. A position that is
> merely *held through* a day (no entry/exit that day) does **not** count as trading that day.

---

## 4. What the page computes (do **not** send these)

Given only `candles` + `positions` per asset, the page derives:

- **Per-asset realized P&L** = Σ `qty · (exit_price − entry_price)`.
- **Per-week P&L**, **portfolio close**, and the **weekly equity curve** (hero columns).
- **Intra-week equity path** — hourly mark-to-market: `portfolio_start` + Σ over positions of
  `qty · (mark − entry_price)` while open, realized after exit. → **session drawdown** (Rule 2).
- **Daily activity** — execution count per calendar day. → **Rule 1**.
- **Token roll-up** — total P&L, return %, weeks-selected, and a contribution sparkline per symbol.
- **Headline summary** — 6-mo return, total P&L, **max session drawdown**, overall (close-to-close)
  drawdown, win rate (core positions), total trades, best/worst week, Sharpe (from weekly returns),
  unique tokens, and the disqualifier roll-up.

The single source of truth is **`Apentic.computeBacktest(raw)`** in `apentic-backtest.js`.
You can call it directly in a console to validate your file:

```js
const raw = await (await fetch('simulated_trades.json')).json();
const model = window.Apentic.computeBacktest(raw);
console.log(model.summary);
```

`window.Apentic.buildMockRaw()` returns a complete example object in this exact schema —
use it as a generator reference.

---

## 5. The two disqualifiers (how they surface)

**Rule 1 — daily activity.** The agent must execute **≥ 1 trade every calendar day** of a
session. Any day in `[start, end)` with zero entries/exits flags the session.

**Rule 2 — session drawdown.** The intra-session equity must never draw down more than
**30%** (`meta.drawdown_limit`) from **any running peak within that session**.

A flagged week gets:
- a **bold, radiating orange ring + ⚠** on its equity column,
- DQ line(s) in the column **hover tooltip**,
- a red **banner** in the week detail naming the rule, the figure, and the timestamp,
- a **session-equity sparkline** marking the trough (for Rule 2),
- an entry in the **Competition compliance** cards and the **MAX SESSION DD** headline stat.

To intentionally produce a DQ for testing: leave a day with no executions (Rule 1), or let
the held positions draw the MTM equity down past the limit mid-session (Rule 2).

---

## 6. Validation checklist

- Timestamps are **unix seconds (UTC)** and strictly increasing within each `candles` / `positions` array.
- Candles use a **uniform interval**; 168 hourly bars per 7-day session is the reference shape.
- `entry_t < exit_t`; both inside the session window (or spanning it for held-through days).
- Prefer **8 assets per week**; set `vol_rank` if you want a specific order.
- Provide `portfolio_start` per week for exact chaining (otherwise it compounds from computed closes).
- Numbers are plain JSON numbers (no strings); prices may be any magnitude (the charts auto-scale,
  incl. sub-cent tokens like `BabyDoge`).
