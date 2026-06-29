# Competition Leaderboard — Frontend Handoff (REFRESHED 2026-06-22)

> Build/upgrade the **BNB-hackathon participant PnL leaderboard** page in the Astro site at `E:/projects/alexlouis-site`. A first version of this page already ships (`CompetitionClient.tsx`); this handoff covers the **new TIMELINE + DISQUALIFICATION features** and the fact that the **data is now LIVE on the CDN**. Read it end-to-end before editing.

---

## CONTEXT

- **What this page is.** An unofficial, independent leaderboard for the BNB-chain "AI Trading Agent Edition" hackathon (live window **June 22–28, 2026**). It ranks registered participant wallets by **window-scoped, deposit-proof PnL** read on-chain. It is *not* the Apentic training/back-test leaderboard (`leaderboard.json` for that is a different schema under a different route) — keep them separate.
- **Where it lives in the repo.**
  - Page: `src/pages/apentic/competition.astro`
  - Island: `src/apentic/components/competition/CompetitionClient.tsx`
  - Types: `src/apentic/types/competition.ts`
  - Page CSS: `src/apentic/competition.css` (+ shared `src/apentic/leaderboard.css`)
- **Producer is out of scope.** The Python producer (`E:/projects/agentic-crypto-trader`) emits all JSON. **Do not touch it.** The page is read-only over the published feed and computes any derived view-state itself.
- **The data is LIVE.** Every endpoint below returns HTTP 200 from `https://data.alexlouis.dev/competition/...`. The old "vendored copy until the producer publishes" stopgap is **obsolete** — flip the page to read the CDN (see DATA SOURCE).

---

## DATA SOURCE — read the LIVE CDN directly

Use the standard env seam already used by every other client (`LeaderboardClient.tsx`, `DashboardClient`):

```ts
const DATA_BASE = import.meta.env.PUBLIC_APENTIC_DATA || '/apentic/data';
const COMP = `${DATA_BASE}/competition`;
```

`E:/projects/alexlouis-site/.env.local` already sets `PUBLIC_APENTIC_DATA=https://data.alexlouis.dev` (baked at build via `import.meta.env`). So with no further config the page fetches `https://data.alexlouis.dev/competition/...` — **no local copy needed**.

> **ACTION ITEM — flip the existing page off the vendored copy.** `CompetitionClient.tsx` currently hardcodes `const COMP = '/apentic/data/competition';` (the deliberate pre-CDN stopgap; the comment above it leaves the one-line flip). Replace it with the `DATA_BASE`-derived `COMP` above.

The `|| '/apentic/data'` is purely an **offline fallback** (vendored copy under `public/apentic/data/competition/`); the live path is the default. Use **`{ cache: 'no-store' }`** on every fetch — the feed updates hourly and no-store is the established convention here.

### Endpoints (all live, HTTP 200)

| File | Fetch URL |
|---|---|
| live board | `${COMP}/leaderboard.json` |
| timeline | `${COMP}/series.json` |
| snapshot index | `${COMP}/snapshots/index.json` |
| per-wallet detail | `${COMP}/wallets/<addr>.json` (addr lowercase 0x) |
| historical board | `${COMP}/snapshots/<id>/leaderboard.json` (id e.g. `2026-06-22T23Z`) |
| roster (optional) | `${COMP}/manifest.json` |

---

## DATA CONTRACT (current — mirrors the CDN)

Snapshot used below: a mid-window state (day 3 of 7). Representative aggregates: `n_participants=123`, `n_entered=80`, `n_traded_in_window=63`, `n_ranked=53`, `n_disqualified=70`, `n_dq_risk≈8`, `total_equity_usd≈8321`, `completed_days=['2026-06-22','2026-06-23']`.

> **The three contest DQ rules (any single failure DQs a wallet).** This drives most of the page — read it first:
> 1. **Entry** — the wallet must HOLD AN ELIGIBLE TOKEN at the window open (Jun 22 00:00). BNB/WBNB are **not** eligible (USDT and the 148 listed tokens are), so a wallet holding only gas-BNB, or nothing (a late entrant), is DQ'd.
> 2. **Daily trade** — **≥1 swap on EVERY completed UTC day** of the window (incl. Jun 22). Any swap counts (a BNB↔USDT keepalive qualifies) — this is NOT limited to eligible-alt trades.
> 3. **$1 equity floor** — total wallet value must **never fall below $1** (checked at open, now, and across every hourly snapshot).
>
> There is **NO minimum-capital rule** — a $5 wallet competes. `min_capital_usd` is GONE. The old `dd_gate` (~30% drawdown) is **not** enforced.

> **Units rule (critical):** all `*_pct` fields are **ALREADY PERCENT** (`275.88` = +275.88%) → render with `formatRawPct`, **not** `formatPct`. `dd_gate` is the **lone fraction** (`0.3` = 30%). Nullables are called out per field.

### 1) `leaderboard.json` — the live board

**Top-level:**

| Field | Type | Notes |
|---|---|---|
| `generated` | string ISO (`+00:00`) | board publish time |
| `metric` | string enum | `"window_pnl_vs_capital_basis"` — PnL vs capital basis, not raw equity |
| `window` | object | window definition (below) |
| `floor_usd` | number USD | the equity-floor rule value (`1.0`) |
| `dd_gate` | number **FRACTION** | `0.3` ≈ 30% drawdown — informational, **NOT enforced** |
| `n_participants` | int | total registered wallets (`123`) |
| `n_entered` | int | wallets that held an eligible token at the window open (`80`) |
| `n_traded_in_window` | int | wallets that acquired an eligible alt via swap (ranking info, not a gate) |
| `n_ranked` | int | active (non-DQ) wallets shown on the board (`53`) |
| `n_disqualified` | int | DQ count (`70`) |
| `n_dq_risk` | int | at-risk-today wallets (haven't traded the in-progress day yet) |
| `total_equity_usd` | number USD | sum of all participant equity |
| **(removed)** `min_capital_usd` | — | **no longer emitted** — there is no min-capital rule |
| `rows` | array | the board, pre-sorted (see SORT) |

**`window` sub-object:**

| Field | Type | Notes |
|---|---|---|
| `start_block` | int | BSC block at window open (`105617727`) |
| `start_ts` | int unix s | window start epoch |
| `start_utc` | string ISO | `"2026-06-22T00:00:00+00:00"` |
| `end_utc` | string ISO | `"2026-06-29T00:00:00+00:00"` (7-day) |
| `completed_days` | string[] (`YYYY-MM-DD`) | **drives DQ.** Fully-closed UTC days. `[]` today → no DQ possible |

**`rows[]` — per-wallet row (24 fields):**

| Field | Type | Notes / Nullable |
|---|---|---|
| `wallet` | string 0x (lowercase) | join key |
| `equity_usd` | number USD | current on-chain equity |
| `equity_open_usd` | number USD | equity at window open; `0.0` if unfunded at open |
| `window_flows_usd` | number USD (signed) | **all non-trading capital in** = external deposits/withdrawals **+ boundary flows** (below). Can be negative |
| `boundary_flow_usd` | number USD (signed) **or null** | value crossing the eligible↔uncounted boundary — e.g. selling Bitcoin/spam (assets NOT in the eligible list) for USDT counts as **capital in, not profit**. Folded into `window_flows_usd`. (Null on pre-fix historical rows.) |
| `capital_basis_usd` | number USD | PnL denominator = `equity_open_usd + window_flows_usd` |
| `pnl_usd` | number USD (signed) | `equity_usd − capital_basis_usd` |
| `pnl_pct` | number PERCENT (signed) **or null** | `pnl_usd/capital_basis_usd × 100`. **null when `capital_basis_usd == 0`** (dust wallets). Already ×100 |
| `entered` | bool | held an eligible token at the window open (entry rule). `false` → DQ |
| `e0_eligible_usd` | number USD | USD value of non-BNB (eligible) holdings at open — the entry-rule basis |
| `floor_min_usd` | number USD | minimum total equity seen (open/now/series). `< 1.0` → DQ |
| `traded_in_window` | bool | acquired an eligible **alt** via swap — **ranking info only, NOT a DQ gate** |
| `n_eligible_buys` | int | count of eligible-alt acquisitions (display: `buys/swaps`) |
| `n_swaps` | int | total swaps in window (display: `buys/swaps`) |
| `trade_days` | string[] (`YYYY-MM-DD`) | UTC days with **≥1 swap of ANY kind** — the daily-trade DQ ledger |
| `ranked` | bool | `= !disqualified && pnl_pct != null`. **ALL active wallets are ranked — no capital floor** |
| `disqualified` | bool | true if **any** of the 3 rules fails |
| `dq_reason` | string **or null** | null when not DQ'd; else the failed rule(s) **joined by `"; "`** — can list **MULTIPLE** (e.g. `"No eligible token held at window open (Jun 22 00:00); No trade on 2026-06-22"`) |
| `dq_risk` | bool | at-risk: active wallet with no trade on the in-progress day yet (warning, not a DQ) |
| `n_holdings` | int | distinct priced token holdings now |
| `stale` | bool | on-chain read stale/unavailable |
| `registered_ts` | string ISO (`…Z` form) | on-chain registration time (note: `Z` suffix, unlike `generated`'s `+00:00`) |
| `rank` | int (1-based) | position in `rows[]` |

**SORT / partition order** (`rank` is assigned 1..N over the whole list):
1. **Ranked** (`ranked=true`, i.e. not DQ'd with a computable `pnl_pct`), sorted by `pnl_pct` **descending**. Includes tiny wallets (e.g. a $12 wallet at +31.5%) — there is no capital floor.
2. **Non-ranked active** (`ranked=false`, `disqualified=false`) — a non-DQ wallet whose `pnl_pct` is null (`capital_basis==0`), sorted by `equity_usd` desc.
3. **Disqualified last** (`disqualified=true`), sorted by `equity_usd` desc.

> Don't re-sort client-side for the main board — honor the producer order. (Per-section client sorting in tanstack-table is fine for optional inspector views.)

### 2) `series.json` — the TIMELINE (source for all over-time charts)

```jsonc
{
  "snapshots": [ { "id": "2026-06-22T23Z", "generated": "2026-06-22T23:34:32…+00:00" } ],
  "wallets":   { "<addr>": [ { point }, … ] },
  "generated": "<iso>"
}
```

| Field | Type | Notes |
|---|---|---|
| `snapshots[]` | array | the time axis — all captured hours, chronological |
| `snapshots[].id` | string | hour bucket, format **`YYYY-MM-DDTHHZ`** (UTC) |
| `snapshots[].generated` | string ISO | exact publish ts for that hour |
| `wallets` | object keyed by addr | value = chronological array of per-hour points |
| `wallets["<addr>"][]` | point[] | one point per wallet **per hour** (today exactly 1 each — single snapshot so far) |
| `generated` | string ISO | top-level publish ts |

**Each timeline point (7 fields):**

| Field | Type | Notes |
|---|---|---|
| `id` | string | the hour bucket (x-axis key; matches `snapshots[].id`) |
| `rank` | int | → rank-over-time |
| `equity_usd` | number | → equity-over-time / sparkline |
| `pnl_pct` | number PERCENT **or null** | null when capital basis 0 |
| `capital_basis_usd` | number | denominator context |
| `ranked` | bool | style ranked vs not |
| `disqualified` | bool | strike out DQ'd series |
| `traded_in_window` | bool | |

This is **THE** source for rank-over-time, equity-over-time, and per-wallet sparklines: read `wallets[addr]`, map each point `id → unix time`, plot `equity_usd` / `rank` / `pnl_pct`. Iterate the `wallets` dict for the multi-series overlay. Each wallet's array grows by one point per hourly capture.

### 3) `snapshots/index.json` — the snapshot/time selector

```jsonc
{
  "snapshots": [
    { "id":"2026-06-22T23Z", "generated":"…+00:00",
      "n_participants":123, "n_ranked":35, "n_disqualified":0,
      "n_dq_risk":68, "total_equity_usd":8146.58 }
  ],
  "generated": "<iso>"
}
```

| Field | Type | Notes |
|---|---|---|
| `snapshots[].id` | string `YYYY-MM-DDTHHZ` | maps to `snapshots/<id>/leaderboard.json` |
| `snapshots[].generated` | string ISO | publish ts |
| `snapshots[].n_participants` / `n_ranked` / `n_disqualified` / `n_dq_risk` | int | per-hour aggregates (drive a summary-over-time strip) |
| `snapshots[].total_equity_usd` | number | per-hour total equity |
| `generated` | string ISO | index publish ts |

Each `id` resolves to `snapshots/<id>/leaderboard.json` — a **full historical board with the identical `leaderboard.json` schema** (verified: `snapshots/2026-06-22T23Z/leaderboard.json` is byte-identical to the live board). This drives the "scrub through the week" selector: pick an id → fetch its archived board → render it through the same row components.

### 4) `manifest.json` — lightweight roster (optional)

`{ generated, metric, n_participants, total_equity_usd, wallets: string[] }` — `wallets` is the 0x address list (123), order matches `rows[]`. Not required by the UI; useful for prefetch/roster.

### 5) `wallets/<addr>.json` — per-wallet detail (drill-down)

Richer than the board row — adds `holdings[]` + a funding `cost_basis`:

| Field | Type | Notes |
|---|---|---|
| `generated`, `address`, `source` (`"onchain"`), `stale` | — | provenance |
| `equity_usd`, `pnl_usd` | number | |
| `baseline_usd` | number **or null** | **legacy, superseded by `capital_basis_usd`/`cost_basis`** — null today; don't rely on it |
| `pnl_pct` | number PERCENT **or null** | null when basis 0 |
| `holdings[]` | array | `{token, qty, price_usd, value_usd}`; **includes dust** (filter `value_usd < 0.01`) |
| `equity_open_usd`, `window_flows_usd`, `capital_basis_usd` | number | same meaning as the row |
| `cost_basis` | object | funding ledger (below) |
| `ranked`, `traded_in_window`, `disqualified`, `entered` | bool | |
| `dq_reason` | string **or null** | null when not DQ'd; else failed rule(s) joined by `"; "` |
| `dq_risk` | bool | |
| `registered_ts` | string ISO `…Z` | |

> The wallet payload does not separately repeat `e0_eligible_usd`/`floor_min_usd` — read those from the leaderboard row.

**`cost_basis`:** `net_deposited`, `gross_deposits`, `gross_withdrawals`, `n_deposits` (int), `n_withdrawals` (int), `first_funding_ts` (unix s **or null**), `nonfundable_deposit_assets` (string[], spam/airdrop names — may be non-ASCII/unicode), `n_swaps` (int), `n_eligible_buys` (int), `traded_eligible` (bool), `eligible_tokens_traded` (string[], may be empty), `trade_days` (string[] `YYYY-MM-DD` — RENAMED from `eligible_trade_days`), `last_trade_ts` (unix s — RENAMED from `last_eligible_trade_ts`).

---

## WHAT TO BUILD

The first version already ships **(a)**, **(e)** (holdings + cost basis), and **(f)** in `CompetitionClient.tsx`. This refresh **extends** that with the AT-RISK badge, the DQ section, the full header strip, and — the headline — the **timeline charts + snapshot selector**. Build everything in the one island (or split chart components under `components/competition/`).

**(a) Ranked board** (`ranked === true`), in producer `rank` order. Columns: `rank` (medal for 1–3), **wallet** (BscScan link `https://bscscan.com/address/<addr>`, truncated `0x6bcf…616b1e`), **window PnL%** (colored: `var(--up)` / `var(--down)` / muted when null), `equity_usd`, `equity_open_usd`, `window_flows_usd` (signed, colored), `n_eligible_buys / n_swaps`, and an **AT-RISK badge when `dq_risk === true`**. Row click → drill-down (e). *(Most of this exists; add the AT-RISK badge + the OPEN/FLOWS columns are already present.)*

**(b) DISQUALIFIED section** (`disqualified === true`) — a **separate block below** the ranked + non-ranked boards, visually de-emphasized. It's **large (~70 of 123)**, so make it collapsible/scrollable. Show the wallet, its final figures, and **`dq_reason` per wallet** — note `dq_reason` can contain **multiple reasons** joined by `"; "` (split on it to render as chips: e.g. "no eligible token at open", "no trade on 06-22", "fell below $1"). Empty-state ("No disqualifications yet") only applies before the first day completes.

**(c) Header strip** (KPI tiles + window context). Show: `generated`, window `start_utc` / `end_utc`, `completed_days` (count + list), `n_ranked` / `n_traded_in_window` / `n_disqualified` / `n_dq_risk` / `n_participants`, `total_equity_usd`, and the **honesty note** (see HONESTY LABELS). The existing KPI strip covers most of these — add `end_utc`, `completed_days`, `n_disqualified`, `n_dq_risk`.

**(d) TIMELINE views** from `series.json`:
- A **multi-line chart** of **rank-over-time and/or equity-over-time** for the **top N wallets** (e.g. top 10 ranked), one `LineSeries` per wallet, with the ability to **highlight/select** a wallet (dim the rest, bold the selected). For rank-over-time, invert so rank 1 sits on top.
- A **per-wallet sparkline** (equity-over-time, or rank-over-time) inside the drill-down.
- A **snapshot/time selector** built from `snapshots/index.json`: a dropdown / slider over `snapshots[].id` that, on change, fetches `snapshots/<id>/leaderboard.json` and **re-renders the board at that historical hour** ("scrub through the week"). Show the snapshot's roll-ups (`n_ranked`, `n_dq_risk`, `total_equity_usd`) in the strip while scrubbed.
  > There are now **~40+ hourly snapshots** (one per hour since Jun 22 23Z), so the charts and scrubber have real history. The series/snapshots were **backfill-corrected** after a rules fix — both `series.json` and each `snapshots/<id>/leaderboard.json` carry `"backfilled": true`; treat it as a normal flag (no special handling). Still degrade gracefully at N=1.

**(e) Per-wallet drill-down** (on row expand): fetch `wallets/<addr>.json`, show `holdings[]` (drop `value_usd < 0.01` dust) + `cost_basis` funding details + the wallet's **series sparkline** (from `series.json` `wallets[addr]`). *(Holdings + cost basis already built; add the sparkline.)*

**(f) Non-ranked toggle** — a control to reveal the non-ranked active wallets too, so all **123** are inspectable. *(Already built as the "Show N non-ranked wallets" toggle.)*

---

## HONESTY LABELS (must appear on the page)

- **Window-scoped, deposit-proof PnL.** Returns are measured **within the June 22–28 window** as `pnl_usd / capital_basis_usd`, where `capital_basis = equity_open + net window flows`. Funding mid-window is **not** profit; withdrawals aren't loss. **Eligible-only:** converting an uncounted asset (Bitcoin, spam tokens — anything off the eligible list) into a counted one is treated as capital, not gain (`boundary_flow_usd`), so PnL reflects only eligible-asset trading.
- **The three DQ rules (any one fails → DQ).** (1) **Entry** — held an eligible token at the Jun 22 00:00 open (`entered=false` → out; late entrants are removed). (2) **Daily trade** — ≥1 swap on every completed UTC day in `window.completed_days` (per `trade_days[]`); **any** swap counts. (3) **$1 floor** — total equity never below $1 (`floor_min_usd < 1`). `dq_reason` names the failed rule(s).
- **Ranked = every active (non-DQ) wallet**, sorted by PnL%. There is **no minimum-capital rule** — a $5 wallet competes (state this; it's a deliberate correction).
- **`traded_in_window` / `n_eligible_buys`** describe whether a wallet bought eligible *alts* — useful context (buys/swaps), **not** a qualification gate.
- **At-risk (`dq_risk`).** An active wallet that hasn't traded the **current still-open** day yet — flagged, **not yet** DQ'd.
- **~30% drawdown (`dd_gate=0.3`, a fraction)** — informational only, **NOT enforced**.
- **Not official standings.** Unofficial & independent; on-chain reads by a participant for reference. Not affiliated with / endorsed by the competition. *(Existing disclaimer banner already says this — keep it.)*

---

## CHARTING (lib + pattern)

**`lightweight-charts` v5 (`^5.2.0`) is the only charting lib in the repo.** No recharts/visx/chart.js/d3. (`three` is for brand 3D, not data.) Use lightweight-charts for both the multi-line timeline and the sparkline. All chart components read colors from the single `THEME` object in `src/apentic/utils/constants.ts`.

Existing chart components to copy from live in `src/apentic/components/dashboard/`: `EquityCurve.tsx`, `TokenChart.tsx`, `WeightsChart.tsx`, etc.

**(a) Multi-series line (rank/equity over time across wallets):** one `chart.addSeries(LineSeries, {...})` per wallet, colored via `tokenColor(i)` / `TOKEN_PALETTE` from `constants.ts`. Copy the create/teardown/resize shape from **`EquityCurve.tsx`** (it already adds two series, so multi-`LineSeries` is the same code multiplied):
- `createChart(containerRef.current, { ...THEME-derived options })` in a **mount-once `useEffect([])`**; store chart + series in refs.
- Attach a `ResizeObserver` that calls `chartRef.current.resize(w, h)`.
- Cleanup returns `observer.disconnect(); chart.remove();`.
- A **separate data `useEffect`** calls `series.setData(points)` then `chart.timeScale().fitContent()`.
- Point shape: `{ time: unixSeconds as UTCTimestamp, value }[]`, **sorted ascending by time**. Map each `series.json` point `id` → unix seconds (parse `YYYY-MM-DDTHHZ`), `value` = `equity_usd` or `rank`.
- For **rank-over-time**, put rank 1 on top — either `rightPriceScale.invertScale` or map `value = -rank` / a max-rank offset.
- Highlight/select: re-`applyOptions({ color, lineWidth })` per series (bold selected, dim others) on selection state change.

**(b) Sparkline (drill-down / per-row):** a small `LineSeries` (or `AreaSeries`) in a ~24px-tall container with everything stripped — hide both axes + grid, `handleScroll:false`, `handleScale:false`, `priceLineVisible:false`, `lastValueVisible:false`, crosshair off. Same mount-once + `setData` + `fitContent()` pattern, minimal options. No sparkline component exists yet — build one as a small variant of `EquityCurve.tsx` (`WeightsChart.tsx` is the closest small reference). For **markers** (e.g. buy/sell or snapshot ticks), see `TokenChart.tsx`'s `createSeriesMarkers` / `setMarkers`.

> Degrade at N=1: with a single point, `fitContent` shows a dot — guard against empty/one-length arrays so the chart never throws.

---

## FRONTEND CONVENTIONS (keep these)

- **Island pattern.** React components are `'use client'`, mounted from the `.astro` page with `client:only="react"`. One Client island per page (`competition.astro` → `<CompetitionClient client:only="react" />`).
- **Layout + nav.** Wrap the page in `src/layouts/Apentic.astro` with `active="competition"`. The page already exists. (To add a new dashboard page generally: add an `id` to the `active` union, add a `nav` entry — `desktopOnly: true` for chart-heavy pages like Training/Trades/Markets. Competition uses a separate right-side nav pill.)
- **Scoped CSS.** Per-page scoped class prefixes imported in the `.astro` frontmatter: `competition.css` (+ shared `leaderboard.css`, class prefix `.apx-lb`). Chart containers are inline-styled from `THEME`.
- **Reusable components & formatters** (all under `src/apentic/`):
  - `components/common/StatCard.tsx`, `components/common/PnLBadge.tsx`.
  - Formatters in `utils/formatters.ts`: **`formatUSD`**, **`formatRawPct`** (for the percent-unit competition fields), `formatPct` (decimal fractions only — e.g. `dd_gate`), `formatUnixTimestamp`.
  - Colors/series in `utils/constants.ts`: `THEME`, `TOKEN_PALETTE`, `tokenColor(i)`, `CASH_COLOR`.
  - Sortable tables (optional inspector views): **`@tanstack/react-table`** (`useReactTable`, `getCoreRowModel`, `getSortedRowModel`, `flexRender`, `ColumnDef`) — see `components/dashboard/TradeTable.tsx` for the import set.
- **Types.** `src/apentic/types/competition.ts` already has `CompetitionLeaderboard`, `CompetitionRow`, `CompetitionWallet`, `CompetitionCostBasis`, etc. **Update it** for the rules change: row fields **add** `entered`, `e0_eligible_usd`, `floor_min_usd` and keep `trade_days`, `disqualified`, `dq_reason`, `dq_risk`, `ranked`, `n_holdings`; top-level **add** `floor_usd`, `n_entered` and **REMOVE** `min_capital_usd` (no longer emitted); keep `metric`, `dd_gate`, `n_disqualified`, `n_dq_risk`, `window.start_ts/end_utc/completed_days`. In `CompetitionCostBasis`, **rename** `eligible_trade_days → trade_days` and `last_eligible_trade_ts → last_trade_ts`. Add the optional `backfilled?: boolean` to the leaderboard/series types. Plus the timeline types `CompetitionSeries` (`{ snapshots: {id,generated}[]; wallets: Record<string, SeriesPoint[]>; generated; backfilled? }`), `SeriesPoint`, `SnapshotIndex`.

---

## SCOPE GUARDRAILS

- **Do NOT touch the Python producer** or any file in `E:/projects/agentic-crypto-trader`. The page is read-only over the published JSON; compute all derived/view state client-side.
- **Read the CDN, don't vendor.** Default to `PUBLIC_APENTIC_DATA`; the vendored `public/apentic/data/competition/` copy is an offline-only fallback. Don't re-introduce a hardcoded local path.
- **Honor the producer's row order** for the main board (don't re-rank). Client sorting is fine only for optional inspector tables.
- **Respect the units rule:** `*_pct` → `formatRawPct`; `dd_gate` → fraction. Treat `pnl_pct`/`dq_reason`/`first_funding_ts`/`baseline_usd` as nullable everywhere.
- **One island per page; scoped CSS; theme-driven charts.** Don't pull in a second charting lib — lightweight-charts only.
- **Degrade gracefully at N=1** snapshots/points (single-dot charts, one-option selector) and on empty DQ section.

---

## DEV / ACCEPTANCE

- **Run:** `npm run dev` (Astro dev server, default **http://localhost:4321**), open `/apentic/competition`. With `.env.local` set, it reads the live CDN automatically.
- **Acceptance:** the dev server renders, reading the **LIVE CDN** —
  1. the **ranked board** (correct order, colored PnL%, BscScan links, AT-RISK badges),
  2. the **DISQUALIFIED section** (collapsible, ~70 wallets, multi-reason `dq_reason` rendered as chips),
  3. a **working timeline chart** (rank- or equity-over-time, top N, with wallet highlight/select),
  4. a **snapshot selector** that loads a historical board from `snapshots/<id>/leaderboard.json`,
  5. a **per-wallet drill-down** with holdings + cost basis + a working **series sparkline**,
  6. the **non-ranked toggle** revealing all 123 wallets, and the **header strip** + honesty labels present.

---

(Note for the calling script: there was no pre-existing `COMPETITION_FRONTEND_HANDOFF.md` in either repo — the file referenced did not exist. The CONTEXT / CONVENTIONS / SCOPE-GUARDRAILS above were reconstructed from the actually-shipped competition page at `E:/projects/alexlouis-site/src/apentic/components/competition/CompetitionClient.tsx`, `competition.astro`, `competition.ts`, and the charts/wiring report, so they reflect the real repo state. The closest existing handoff, `E:/projects/agentic-crypto-trader/.design-export-simulated/HANDOFF.md`, is for the separate standalone back-test export page, not this Astro leaderboard.)
