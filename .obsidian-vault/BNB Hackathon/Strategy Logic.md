# Strategy Logic — visual map

A visual summary of the logic developed for the [[Project Overview|agent]]: the offline
**universe pipeline**, the **research loop** that produced (and gated) the strategy, the
**runtime decision logic** of the codified candidate, and the **regime overlay**. Diagrams are
Mermaid (render in Obsidian). Detail lives in [[Trading Strategies]], [[Token Universe]],
[[Simulated Market]], and the [[Build Log]].

## 1 · End-to-end system

```mermaid
flowchart TD
  subgraph P1["1 · Universe pipeline (offline, mostly keyless)"]
    A1["149 eligible BEP-20"] --> A2["DexScreener screen<br/>liquidity · volume · turnover"]
    A2 --> A3["CMC resolve<br/>→ canonical BSC contract"]
    A3 --> A4["GoPlus forensic gate<br/>rug / honeypot veto"]
    A4 --> A5["Select<br/>turnover-rank + CMC-rank risk tiers"]
    A5 --> A6[("20-token universe")]
  end
  subgraph P2["2 · Research loop (offline, evidence-gated)"]
    B1["Factor model + IC gate<br/>→ entry alpha refuted"] --> B2["Cost-aware backtest<br/>→ high-turnover dies"]
    B2 --> B3["7-day resample + DQ gates<br/>→ buy&hold disqualified"]
    B3 --> B4["Tournament sweep<br/>→ vol-top8"]
    B4 --> B5["OOS validation<br/>→ tilt holds"]
    B5 --> B6["Regime overlay + crash test<br/>→ trend50 / severity"]
  end
  subgraph P3["3 · Codified strategy — trader.strategy.build_candidate"]
    C1["vol-top8 × regime exposure<br/>daily-rebalanced · AMM-costed"]
  end
  ANCHOR["GeckoTerminal OHLCV<br/>+ ccxt BTC/BNB anchor"] --> B1
  A6 --> B1
  B6 --> C1
  C1 -. "deferred (Phase 2)" .-> EX["TWAK self-custody signing<br/>+ on-chain execution · register by June 22"]
```

## 2 · Runtime decision logic (what the agent does each day)

```mermaid
flowchart TD
  S(["Each day · rebalance — satisfies ≥1 trade/day"]) --> U["Universe = vol-top8<br/>8 highest realized-vol tokens (forensically clean)"]
  U --> R["Read BTC regime from the anchor"]
  R --> E{"Overlay → exposure e ∈ [0,1]"}
  E -->|"none (pure bull)"| N["e = 1.0"]
  E -->|"trend50 (default)"| T["e = 1.0 if BTC > 72h EMA<br/>else 0.5"]
  E -->|"severity"| V["e ramps 1 → 0 as the trailing<br/>BTC drop deepens −5% → −20%"]
  N --> W["Target weights = equal-weight(vol-top8) × e<br/>remainder (1 − Σw) → cash / stables"]
  T --> W
  V --> W
  W --> B["Rebalance to target<br/>cost = LP fee + trade ÷ (liquidity ÷ 2) + gas"]
  B --> H["Hold ~1 day"]
  H --> S
```

### Overlay exposure — the three gates

| Overlay | Exposure rule | Keeps | Insures |
|---|---|---|---|
| `none` | always 1.0 | max upside | nothing (blows the gate in any crash) |
| **`trend50`** (default) | 1.0 above 72h EMA, else 0.5 | moderate upside | moderate / sharp crashes (DQ→0) |
| `severity` | 1 → 0 as trailing drop −5%→−20% | ~full upside (dormant in calm) | the **deep slow crash** tail (survives −50%) |

## 3 · Research loop — how each finding gated the strategy

```mermaid
flowchart TD
  A["Factor model: r_alt = α + β·BTC + β·BNB + ε<br/>R² classifies factor-driven vs dev-driven"] --> B{"IC gate: does residual-momentum<br/>predict forward returns?"}
  B -->|"NEGATIVE IC (mean-reverts)"| C["❌ entry / continuation alpha refuted"]
  C --> D{"Cost-aware backtest: does high turnover<br/>survive thin-pool AMM slippage?"}
  D -->|"NO — 100%+ cost drag"| Er["❌ momentum / reversal dead → low-turnover only"]
  Er --> F{"7-day resample: is drawdown the<br/>weekly binding constraint?"}
  F -->|"NO — but buy&hold fails ≥1-trade/day"| G["activity DQ → must rebalance daily"]
  G --> H["Tournament sweep:<br/>maximize P(big week ∧ not DQ)"]
  H --> I["✅ vol-top8 (volatility tilt ≫ beta tilt)"]
  I --> J{"OOS validation: does the tilt<br/>hold out-of-sample?"}
  J -->|"YES — vol-rank persists +0.66,<br/>doubles the contender rate"| K["Regime overlay + synthetic-crash test"]
  K --> L["✅ trend50 (default) · severity (deep-tail insurance)"]
```

> **Read it as:** every box is a question the data answered, and most answers were *negative* —
> entry alpha refuted, high-turnover killed by costs, buy&hold disqualified — which is what
> shaped the survivor: **a daily-rebalanced, volatility-tilted, regime-gated low-turnover book.**
> The honest no's mattered as much as the yes's.

## Legend

- **vol-top8** — the 8 highest-realized-volatility eligible tokens, equal-weighted.
- **TOURNEY** — P(weekly return > +15% **and** not disqualified) — the leaderboard objective.
- **DQ gates** — drawdown > 30% **or** < 1 trade/day. Both disqualify.
- **AMM cost** — constant-product price impact ≈ trade ÷ (liquidity ÷ 2) + LP fee + gas.
