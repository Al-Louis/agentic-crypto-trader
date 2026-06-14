/* Apentic — mock data + live generators. Plain JS, attaches APENTIC to window. */
(function () {
  "use strict";

  const ASSETS = [
    { sym: "BTC", name: "Bitcoin", px: 71284.5 },
    { sym: "ETH", name: "Ethereum", px: 3842.1 },
    { sym: "SOL", name: "Solana", px: 188.74 },
    { sym: "DOGE", name: "Dogecoin", px: 0.1642 },
    { sym: "AVAX", name: "Avalanche", px: 41.22 },
    { sym: "LINK", name: "Chainlink", px: 18.91 },
  ];

  // --- deterministic-ish pseudo random so reloads look stable but lively ---
  function mulberry(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Headline KPIs for the run
  const KPIS = {
    startCapital: 10000,
    portfolio: 24718.42,
    pnlPct: 147.18,
    pnlUsd: 14718.42,
    winRate: 58.3,
    trades: 8412,
    maxDrawdown: -22.4,
    sharpe: 2.14,
    episode: 4213,
    totalEpisodes: 5000,
    stepsM: 41.8, // million env steps
    bestTrade: { sym: "SOL", side: "LONG", usd: 1842.6, pct: 9.4 },
    worstTrade: { sym: "ETH", side: "SHORT", usd: -1310.2, pct: -7.1 },
    sessionPnl: 3.82, // today's session %, drives ape mood
  };

  // Equity curve: 10k -> ~24.7k with volatility + a nasty drawdown mid-run
  function equityCurve(n) {
    const r = mulberry(7);
    const pts = [];
    let v = KPIS.startCapital;
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1);
      // upward drift, a deep dip around 45-58%, recovery after
      let drift = 0.012;
      if (t > 0.44 && t < 0.6) drift = -0.03;
      if (t > 0.6 && t < 0.72) drift = 0.028;
      const shock = (r() - 0.48) * 0.045;
      v = Math.max(4200, v * (1 + drift + shock));
      pts.push(v);
    }
    // pin the end near portfolio
    const k = KPIS.portfolio / pts[pts.length - 1];
    return pts.map((p) => p * k);
  }

  // Reward curve for RL training (noisy, rising, plateauing)
  function rewardCurve(n) {
    const r = mulberry(13);
    const pts = [];
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1);
      const base = -0.8 + 2.4 * (1 - Math.exp(-3.1 * t));
      pts.push(base + (r() - 0.5) * 0.42);
    }
    return pts;
  }

  // Candlesticks for the focus asset
  function candles(n, start) {
    const r = mulberry(29);
    const out = [];
    let px = start;
    for (let i = 0; i < n; i++) {
      const o = px;
      const vol = px * 0.012;
      const dir = r() > 0.46 ? 1 : -1;
      const body = dir * vol * (0.3 + r() * 1.5);
      const c = o + body;
      const hi = Math.max(o, c) + vol * r() * 1.1;
      const lo = Math.min(o, c) - vol * r() * 1.1;
      out.push({ o, h: hi, l: lo, c });
      px = c;
    }
    return out;
  }

  // Live trade feed generator
  const SIDES = ["BUY", "SELL"];
  function makeTrade(rng) {
    const a = ASSETS[Math.floor(rng() * ASSETS.length)];
    const side = SIDES[Math.floor(rng() * SIDES.length)];
    const qty = +(rng() * 4 + 0.05).toFixed(3);
    const slip = (rng() - 0.5) * a.px * 0.01;
    const px = a.px + slip;
    // pnl realized only on SELL; bias slightly positive (win rate ~58%)
    let pnl = null;
    if (side === "SELL") {
      const win = rng() < 0.58;
      const mag = a.px * qty * (0.002 + rng() * 0.03);
      pnl = win ? mag : -mag * (0.7 + rng() * 0.6);
    }
    return {
      id: Math.random().toString(36).slice(2, 9),
      t: Date.now(),
      sym: a.sym,
      side,
      qty,
      px,
      pnl,
      conf: +(0.5 + rng() * 0.49).toFixed(2),
    };
  }

  function seedTrades(count) {
    const rng = mulberry(91);
    const arr = [];
    for (let i = 0; i < count; i++) {
      const tr = makeTrade(rng);
      tr.t = Date.now() - (count - i) * 4200;
      arr.push(tr);
    }
    return arr;
  }

  // Open positions
  const POSITIONS = [
    { sym: "BTC", side: "LONG", size: 0.42, entry: 68120.0, mark: 71284.5, lev: 3 },
    { sym: "SOL", side: "LONG", size: 38.5, entry: 171.2, mark: 188.74, lev: 5 },
    { sym: "ETH", side: "SHORT", size: 2.1, entry: 3910.0, mark: 3842.1, lev: 2 },
  ].map((p) => {
    const dir = p.side === "LONG" ? 1 : -1;
    const upnl = dir * (p.mark - p.entry) * p.size;
    const pct = dir * ((p.mark - p.entry) / p.entry) * 100 * p.lev;
    return { ...p, upnl, pct };
  });

  // ---- Leaderboard: training sessions (his own runs over time) ----
  function spark(seed, end, n) {
    const r = mulberry(seed);
    const out = []; let v = 100;
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1);
      v = v * (1 + (end / 100) * 0.02 + (r() - 0.5) * 0.05);
      out.push(v);
    }
    return out;
  }

  // raw runs; rank assigned after sort. dd = max drawdown (drives ape damage)
  const RAW_SESSIONS = [
    { name: "ppo-ape", ver: "v7", algo: "PPO", pnl: 147.18, sharpe: 2.14, win: 58.3, trades: 8412, dd: -22.4, eps: 4213, status: "LIVE", days: 0 },
    { name: "ppo-ape", ver: "v6", algo: "PPO", pnl: 98.4, sharpe: 1.81, win: 56.1, trades: 7980, dd: -29.7, eps: 5000, status: "RETIRED", days: 6 },
    { name: "sac-banana", ver: "v3", algo: "SAC", pnl: 76.2, sharpe: 1.66, win: 54.8, trades: 11240, dd: -34.1, eps: 5000, status: "RETIRED", days: 11 },
    { name: "ppo-ape", ver: "v5", algo: "PPO", pnl: 53.9, sharpe: 1.42, win: 53.2, trades: 6820, dd: -38.8, eps: 5000, status: "RETIRED", days: 15 },
    { name: "td3-moonshot", ver: "v2", algo: "TD3", pnl: 31.4, sharpe: 1.18, win: 52.0, trades: 9310, dd: -41.5, eps: 5000, status: "RETIRED", days: 19 },
    { name: "ppo-ape", ver: "v4", algo: "PPO", pnl: 12.7, sharpe: 0.74, win: 50.6, trades: 5510, dd: -47.2, eps: 5000, status: "RETIRED", days: 24 },
    { name: "a2c-gigachad", ver: "v1", algo: "A2C", pnl: -4.3, sharpe: 0.31, win: 49.1, trades: 7140, dd: -53.0, eps: 5000, status: "RETIRED", days: 28 },
    { name: "dqn-degen", ver: "v2", algo: "DQN", pnl: -17.9, sharpe: -0.12, win: 47.4, trades: 4980, dd: -61.8, eps: 3100, status: "REKT", days: 33 },
    { name: "ppo-ape", ver: "v3", algo: "PPO", pnl: -23.6, sharpe: -0.28, win: 46.0, trades: 4220, dd: -66.3, eps: 5000, status: "RETIRED", days: 37 },
    { name: "rainbow-rekt", ver: "v1", algo: "Rainbow", pnl: -41.2, sharpe: -0.61, win: 43.8, trades: 6010, dd: -74.9, eps: 2400, status: "REKT", days: 41 },
    { name: "impala-ape", ver: "v1", algo: "IMPALA", pnl: -55.7, sharpe: -0.9, win: 41.2, trades: 8800, dd: -83.1, eps: 1900, status: "REKT", days: 46 },
    { name: "ppo-ape", ver: "v2", algo: "PPO", pnl: -68.4, sharpe: -1.22, win: 38.9, trades: 3050, dd: -91.0, eps: 5000, status: "REKT", days: 52 },
  ];

  function apenticScore(s) {
    // composite 0–1000: weights P&L, Sharpe, win-rate, penalize drawdown
    const v = 500 + s.pnl * 2.4 + s.sharpe * 70 + (s.win - 50) * 6 + s.dd * 1.8;
    return Math.max(0, Math.round(v));
  }

  const SESSIONS = RAW_SESSIONS.map((s, i) => ({
    ...s,
    id: s.name + "-" + s.ver,
    label: s.name + " " + s.ver,
    health: Math.max(4, Math.round(100 + s.dd)), // dd is negative; deeper dd = more damaged
    score: apenticScore(s),
    spark: spark(i * 17 + 5, s.pnl, 24),
  }));

  window.APENTIC = {
    ASSETS,
    KPIS,
    POSITIONS,
    SESSIONS,
    equityCurve,
    rewardCurve,
    candles,
    makeTrade,
    seedTrades,
    mulberry,
  };
})();
