/* ============================================================
   APENTIC — Simulated Trades data layer.

   PUBLIC API (window.Apentic):
     Apentic.computeBacktest(raw)  → full dashboard model the UI renders
     Apentic.buildMockRaw()        → a sample RAW backtest (the input contract)
     Apentic.buildMock()           → computeBacktest(buildMockRaw())

   The page sets window.BACKTEST to the model. By default it uses the mock;
   if a `simulated_trades.json` file is present it is loaded and run through
   computeBacktest() instead. See HANDOFF.md for the raw JSON schema.
   ============================================================ */
(function () {
  "use strict";

  const CLASS_COLOR = { alt: "#FF7512", major: "#5B9BFF", peg: "#F5C24A" };
  const DAY = 86400, HOUR = 3600;

  /* ============================================================
     computeBacktest(raw) — derive ALL analytics from raw input.
     raw = { meta, weeks: [ { index,label,start,end,portfolio_start?,
              assets:[ { symbol,class,vol_rank,alloc_usd,
                         candles:[{t,o,h,l,c,v}],
                         positions:[{entry_t,entry_price,exit_t,exit_price,qty,kind}] } ] } ] }
     ============================================================ */
  function buildEquityPath(P0, assets, H) {
    const legs = [];
    assets.forEach((a) => a.holds.forEach((h) => legs.push({ candles: a.candles, from: h.from, to: h.to, entryPx: h.entryPx, qty: h.qty })));
    const path = new Array(H);
    for (let hh = 0; hh < H; hh++) {
      let eq = P0;
      for (const lg of legs) {
        if (hh < lg.from) continue;
        const px = hh <= lg.to ? lg.candles[hh].c : lg.candles[lg.to].c;
        eq += lg.qty * (px - lg.entryPx);
      }
      path[hh] = eq;
    }
    return path;
  }

  function computeBacktest(raw) {
    const meta = raw.meta || {};
    const startCapital = meta.start_capital != null ? meta.start_capital : 10000;
    const ddLimit = meta.drawdown_limit != null ? meta.drawdown_limit : -0.30;
    const nWeeks = raw.weeks.length;

    const tokenAgg = {};
    const weeks = [];
    const equity = [startCapital];
    let portfolio = startCapital, peak = startCapital, maxDD = 0;
    let totalTrades = 0, coreTrades = 0, winTrades = 0;
    let bestWeek = { idx: -1, pct: -1e9 }, worstWeek = { idx: -1, pct: 1e9 };

    raw.weeks.forEach((rw, w) => {
      const wStart = rw.start;
      const wEnd = rw.end != null ? rw.end : wStart + 7 * DAY;
      const portfolioStart = rw.portfolio_start != null ? rw.portfolio_start : portfolio;
      let weekPnl = 0;

      const assets = rw.assets.map((ra, ai) => {
        const candles = ra.candles;
        const t0 = candles[0].t;
        const iv = candles.length > 1 ? (candles[1].t - candles[0].t) : HOUR;
        const idxAt = (t) => Math.max(0, Math.min(candles.length - 1, Math.round((t - t0) / iv)));

        // positions → holds (shading + equity) and executions (markers + day count)
        const holds = [];
        const trades = [];
        (ra.positions || []).forEach((p) => {
          const from = idxAt(p.entry_t), to = idxAt(p.exit_t);
          const core = (p.kind || "core") !== "scalp";
          const qty = p.qty;
          holds.push({ from, to, entryPx: p.entry_price, exitPx: p.exit_price, qty, usd: p.entry_price * qty, core, entryT: p.entry_t, exitT: p.exit_t });
          trades.push({ t: p.entry_t, side: "BUY", px: p.entry_price, qty, usd: p.entry_price * qty, di: from, core });
          trades.push({ t: p.exit_t, side: "SELL", px: p.exit_price, qty, usd: p.exit_price * qty, di: to, core });
        });
        trades.sort((a, b) => a.t - b.t);

        let pnl = 0;
        holds.forEach((h) => { pnl += h.qty * (h.exitPx - h.entryPx); });
        const alloc = ra.alloc_usd != null ? ra.alloc_usd : holds.reduce((s, h) => s + h.usd, 0);

        totalTrades += holds.length;
        holds.forEach((h) => { if (h.core) { coreTrades++; if (h.exitPx >= h.entryPx) winTrades++; } });
        weekPnl += pnl;

        const cls = ra.class || "alt";
        const ag = tokenAgg[ra.symbol] || (tokenAgg[ra.symbol] = { sym: ra.symbol, cls, pnl: 0, weeks: 0, contrib: new Array(nWeeks).fill(0) });
        ag.pnl += pnl; ag.weeks += 1; ag.contrib[w] = pnl;

        return { sym: ra.symbol, cls, volRank: ra.vol_rank != null ? ra.vol_rank : ai + 1, alloc, candles, trades, holds, pnl, pnlPct: alloc ? pnl / alloc : 0 };
      });

      const portfolioEnd = portfolioStart + weekPnl;
      const wPct = portfolioStart ? (portfolioEnd - portfolioStart) / portfolioStart : 0;
      portfolio = portfolioEnd;
      equity.push(portfolioEnd);
      peak = Math.max(peak, portfolioEnd);
      maxDD = Math.min(maxDD, (portfolioEnd - peak) / peak);
      if (wPct > bestWeek.pct) bestWeek = { idx: w, pct: wPct };
      if (wPct < worstWeek.pct) worstWeek = { idx: w, pct: wPct };

      // intra-week mark-to-market equity path + worst drawdown (rule 2)
      const H = assets[0] ? assets[0].candles.length : 0;
      const equityPath = buildEquityPath(portfolioStart, assets, H);
      let wkPeak = portfolioStart, intraDD = 0, troughHour = 0;
      for (let hh = 0; hh < equityPath.length; hh++) {
        wkPeak = Math.max(wkPeak, equityPath[hh]);
        const d = (equityPath[hh] - wkPeak) / wkPeak;
        if (d < intraDD) { intraDD = d; troughHour = hh; }
      }
      // daily activity coverage (rule 1): a day "trades" if it has an execution
      const nDays = Math.max(1, Math.round((wEnd - wStart) / DAY));
      const dayCounts = new Array(nDays).fill(0);
      assets.forEach((a) => a.trades.forEach((tr) => {
        const d = Math.floor((tr.t - wStart) / DAY);
        if (d >= 0 && d < nDays) dayCounts[d]++;
      }));
      const missedDays = dayCounts.map((c, i) => (c === 0 ? i : -1)).filter((i) => i >= 0);
      const dq = {
        noTrade: missedDays.length > 0, missedDays, dayCounts,
        drawdown: intraDD < ddLimit, intraDD, troughHour,
        any: missedDays.length > 0 || intraDD < ddLimit,
      };

      weeks.push({
        idx: w, label: rw.label || "W" + String(w + 1).padStart(2, "0"),
        start: wStart, end: wEnd,
        portfolioStart, portfolioEnd, pnl: weekPnl, pnlPct: wPct,
        assets, equityPath, intraDD, troughHour, dq,
      });
    });

    // 6-month token roll-up
    const tokens = Object.values(tokenAgg).filter((t) => t.weeks > 0).map((t) => {
      const notional = (startCapital * 0.86 / 8) * t.weeks; // rough avg-deployed denominator
      return { sym: t.sym, cls: t.cls, pnl: t.pnl, weeks: t.weeks, pct: notional ? t.pnl / notional : 0, contrib: t.contrib };
    }).sort((a, b) => b.pnl - a.pnl);

    // summary + disqualifier roll-up
    const finalReturnPct = (portfolio - startCapital) / startCapital;
    const maxIntraDD = weeks.length ? Math.min(...weeks.map((w) => w.intraDD)) : 0;
    const ddBreachWeeks = weeks.filter((w) => w.dq.drawdown).map((w) => ({ label: w.label, idx: w.idx, intraDD: w.intraDD }));
    const noTradeWeeks = weeks.filter((w) => w.dq.noTrade).map((w) => ({ label: w.label, idx: w.idx, missedDays: w.dq.missedDays, start: w.start }));
    const rets = weeks.map((w) => w.pnlPct);
    const mean = rets.reduce((a, b) => a + b, 0) / (rets.length || 1);
    const sd = Math.sqrt(rets.reduce((a, b) => a + (b - mean) * (b - mean), 0) / (rets.length || 1)) || 1e-9;
    const sharpe = (mean / sd) * Math.sqrt(52);
    const symbols = new Set(raw.weeks.flatMap((w) => w.assets.map((a) => a.symbol)));

    const summary = {
      start: meta.window_start != null ? meta.window_start : (raw.weeks[0] ? raw.weeks[0].start : 0),
      end: meta.window_end != null ? meta.window_end : (weeks.length ? weeks[weeks.length - 1].end : 0),
      startCapital, finalValue: portfolio, finalReturnPct, totalPnlUsd: portfolio - startCapital,
      maxDrawdown: maxDD, maxIntraDD, ddLimit,
      winRate: coreTrades ? winTrades / coreTrades : 0, totalTrades, coreTrades,
      bestWeek, worstWeek, sharpe, uniqueTokens: tokens.length,
      universeSize: meta.universe_size != null ? meta.universe_size : symbols.size,
      ddBreachWeeks, noTradeWeeks, dqWeeks: weeks.filter((w) => w.dq.any).length,
    };

    return { weeks, tokens, summary, equity, CLASS_COLOR, N_WEEKS: nWeeks, DAYS: 7, HOURS: weeks[0] ? weeks[0].equityPath.length : 0 };
  }

  /* ============================================================
     MOCK generator — produces RAW in the documented schema, then
     computeBacktest()s it. This is the reference for your simulator.
     ============================================================ */
  function mulberry(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const START = 10000, N_WEEKS = 24, DAYS = 7, HOURS = DAYS * 24, WEEK = DAY * DAYS;
  const WIN_END = Math.floor(new Date("2026-06-08T00:00:00Z").getTime() / 1000);
  const WIN_START = WIN_END - WEEK * N_WEEKS;

  const UNIVERSE = [
    ["HUMA", 0.42, 0.98, "alt"], ["SIREN", 0.073, 0.95, "alt"], ["Q", 0.018, 0.93, "alt"],
    ["SKYAI", 0.11, 0.9, "alt"], ["B", 0.34, 0.86, "alt"], ["UB", 0.0062, 0.84, "alt"],
    ["TAC", 0.021, 0.82, "alt"], ["TAG", 0.0091, 0.8, "alt"], ["BANANAS31", 0.0047, 0.78, "alt"],
    ["FF", 0.13, 0.74, "alt"], ["COAI", 0.29, 0.7, "alt"], ["ASTER", 1.18, 0.6, "alt"],
    ["SFP", 0.78, 0.5, "major"], ["BabyDoge", 0.0000012, 0.55, "alt"], ["ZEC", 38.2, 0.42, "major"],
    ["ADA", 0.61, 0.34, "major"], ["XRP", 2.18, 0.3, "major"], ["LINK", 17.4, 0.32, "major"],
    ["LTC", 88.5, 0.28, "major"], ["XAUt", 2340, 0.06, "peg"],
  ];
  const SHOCK = { week: 11, center: 88, half: 26, depth: 0.36, plateau: 0.93 };
  const NOTRADE = { 6: 3, 17: 4 };

  function driftFor(w) {
    const t = w / (N_WEEKS - 1);
    let d = 0.03 + 0.055 * Math.sin(t * Math.PI * 0.9);
    if (w >= 9 && w <= 12) d = -0.075 - (w - 9) * 0.004;
    if (w >= 13 && w <= 17) d = 0.13;
    if (w >= 18) d = 0.055;
    return d * 2.1;
  }

  function applyCrash(candles, cfg) {
    const { center, half, depth, plateau } = cfg;
    for (let i = 0; i < candles.length; i++) {
      let f = 1;
      if (i >= center - half && i <= center) f = 1 - depth * ((i - (center - half)) / half);
      else if (i > center && i <= center + half) f = (1 - depth) + (plateau - (1 - depth)) * ((i - center) / half);
      else if (i > center + half) f = plateau;
      const c = candles[i]; c.o *= f; c.h *= f; c.l *= f; c.c *= f;
    }
  }

  function buildCandles(rng, base, weekReturn, startT) {
    const out = []; let px = base;
    const targetEnd = base * (1 + weekReturn);
    for (let i = 0; i < HOURS; i++) {
      const o = px;
      const stepDrift = (targetEnd - base) / HOURS / Math.max(px, 1e-12);
      const cycle = Math.sin((i / 24) * Math.PI * 2) * 0.004;
      const noise = (rng() - 0.5) * 0.05 + stepDrift + cycle;
      let c = o * (1 + noise);
      const hi = Math.max(o, c) * (1 + rng() * 0.022);
      const lo = Math.min(o, c) * (1 - rng() * 0.022);
      const vol = (0.45 + rng() * 1.4) * (1 + Math.abs(noise) * 6) * (i % 24 < 4 ? 1.4 : 1);
      out.push({ t: startT + i * HOUR, o, h: hi, l: lo, c, v: +vol.toFixed(3) });
      px = c;
    }
    return out;
  }

  // returns position objects {entry_t,entry_price,exit_t,exit_price,qty,kind}
  function buildPositions(rng, candles, alloc, coreSpan) {
    const H = candles.length;
    const spans = [];
    if (coreSpan) {
      spans.push({ from: coreSpan.from, to: coreSpan.to, usd: alloc, kind: "core" });
    } else {
      const nTrips = 1 + Math.floor(rng() * 3);
      let lastExit = -1;
      for (let trip = 0; trip < nTrips; trip++) {
        const earliest = lastExit + 2 + Math.floor(rng() * 8);
        if (earliest >= H - 6) break;
        const ei = earliest;
        const xi = Math.min(H - 1, ei + 20 + Math.floor(rng() * 92));
        spans.push({ from: ei, to: xi, usd: alloc / nTrips, kind: "core" });
        lastExit = xi;
      }
    }
    for (let d = 0; d < DAYS; d++) {
      if (rng() < 0.3) continue;
      const h0 = d * 24 + 2 + Math.floor(rng() * 18);
      if (h0 >= H - 2) continue;
      spans.push({ from: h0, to: Math.min(H - 1, h0 + 1 + Math.floor(rng() * 5)), usd: alloc * 0.01, kind: "scalp" });
    }
    return spans.map((s) => {
      const entry = candles[s.from], exit = candles[s.to];
      return { entry_t: entry.t, entry_price: +entry.c.toPrecision(6), exit_t: exit.t, exit_price: +exit.c.toPrecision(6), qty: +(s.usd / entry.c).toPrecision(6), kind: s.kind };
    });
  }

  function buildMockRaw() {
    const rng = mulberry(20260608);
    const weeks = [];
    let portfolio = START;
    for (let w = 0; w < N_WEEKS; w++) {
      const wStart = WIN_START + w * WEEK;
      const drift = driftFor(w);
      const isShock = w === SHOCK.week;
      const skipDay = NOTRADE[w];
      const ranked = UNIVERSE
        .map(([sym, base, volt, cls]) => ({ sym, base, volt, cls, wvol: volt * (0.45 + rng() * 1.1) }))
        .sort((a, b) => b.wvol - a.wvol).slice(0, 8);
      const portfolioStart = portfolio;
      const deploy = 0.92;
      const wi = ranked.map((tk) => (1.1 - tk.volt) + rng() * 0.5);
      const wsum = wi.reduce((a, b) => a + b, 0);
      let weekPnl = 0;
      const assets = ranked.map((tk, ai) => {
        const alloc = portfolioStart * deploy * (wi[ai] / wsum);
        const r = isShock ? 0 : drift + (rng() - 0.44) * (0.18 + tk.volt * 0.42);
        const candles = buildCandles(rng, tk.base, r, wStart);
        if (isShock) applyCrash(candles, SHOCK);
        const coreSpan = isShock ? { from: Math.max(1, SHOCK.center - SHOCK.half - 3), to: Math.min(candles.length - 1, SHOCK.center + SHOCK.half + 8) } : null;
        let positions = buildPositions(rng, candles, alloc, coreSpan);
        if (skipDay != null) {
          positions = positions.filter((p) => Math.floor((p.entry_t - wStart) / DAY) !== skipDay && Math.floor((p.exit_t - wStart) / DAY) !== skipDay);
        }
        positions.forEach((p) => { weekPnl += p.qty * (p.exit_price - p.entry_price); });
        return { symbol: tk.sym, class: tk.cls, vol_rank: ai + 1, alloc_usd: +alloc.toFixed(2), candles, positions };
      });
      portfolio = portfolioStart + weekPnl;
      weeks.push({ index: w, label: "W" + String(w + 1).padStart(2, "0"), start: wStart, end: wStart + WEEK, portfolio_start: +portfolioStart.toFixed(2), assets });
    }
    return {
      meta: {
        start_capital: START, window_start: WIN_START, window_end: WIN_END,
        n_weeks: N_WEEKS, candle_interval_seconds: HOUR, drawdown_limit: -0.30,
        universe_size: UNIVERSE.length, generated: new Date().toISOString(),
      },
      weeks,
    };
  }

  function buildMock() { return computeBacktest(buildMockRaw()); }

  window.Apentic = window.Apentic || {};
  window.Apentic.computeBacktest = computeBacktest;
  window.Apentic.buildMockRaw = buildMockRaw;
  window.Apentic.buildMock = buildMock;
  window.BACKTEST = buildMock();
})();
