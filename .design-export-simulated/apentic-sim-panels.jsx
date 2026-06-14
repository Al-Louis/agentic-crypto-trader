/* Apentic — Simulated Trades panels. window.{BtHeadline, BtWeekDetail, BtTokenTable} */
const { useState: useStateP, useMemo: useMemoP } = React;

const _usd0 = (v) => (v < 0 ? "-$" : "$") + Math.abs(v).toLocaleString(undefined, { maximumFractionDigits: 0 });
const _pct = (v, d = 1) => (v >= 0 ? "+" : "") + (v * 100).toFixed(d) + "%";
const CLR = { alt: "#FF7512", major: "#5B9BFF", peg: "#F5C24A" };
const CLSL = { alt: "Altcoin", major: "Major", peg: "Gold-pegged" };

/* ============ headline stats band ============ */
function BtHeadline({ summary, weeks }) {
  const s = summary;
  const bw = weeks[s.bestWeek.idx], ww = weeks[s.worstWeek.idx];
  const ddPct = Math.abs(s.maxDrawdown * 100);
  const sddPct = Math.abs(s.maxIntraDD * 100);
  const sddBreached = s.maxIntraDD < s.ddLimit;
  const cells = [
    { k: "6-MO RETURN", v: _pct(s.finalReturnPct, 1), c: s.finalReturnPct >= 0 ? "var(--up)" : "var(--down)", sub: _usd0(s.startCapital) + " → " + _usd0(s.finalValue), feat: true },
    { k: "TOTAL P&L", v: (s.totalPnlUsd >= 0 ? "+" : "") + _usd0(s.totalPnlUsd), c: s.totalPnlUsd >= 0 ? "var(--up)" : "var(--down)", sub: "net realized" },
    { k: "MAX SESSION DD", v: "−" + sddPct.toFixed(1) + "%", c: sddBreached ? "var(--down)" : "var(--up)",
      sub: sddBreached ? "⚠ breached −30% · " + s.ddBreachWeeks.map((x) => x.label).join(", ") : (30 - sddPct).toFixed(1) + "pp headroom" },
    { k: "OVERALL DRAWDOWN", v: "−" + ddPct.toFixed(1) + "%", c: "var(--text)", sub: "peak-to-trough equity" },
    { k: "WIN RATE", v: (s.winRate * 100).toFixed(1) + "%", c: "var(--text)", sub: s.coreTrades + " core positions" },
    { k: "TOTAL TRADES", v: s.totalTrades.toLocaleString(), c: "var(--text)", sub: "incl. daily scalps" },
    { k: "BEST / WORST WK", v: _pct(s.bestWeek.pct, 1), c: "var(--up)", sub: ww.label + " " + _pct(s.worstWeek.pct, 1) },
    { k: "UNIQUE TOKENS", v: s.uniqueTokens, c: "var(--text)", sub: "of " + s.universeSize + " universe" },
  ];
  return (
    <div className="bt-stats">
      {cells.map((c, i) => (
        <div key={i} className={"bt-stat" + (c.feat ? " feat" : "")}>
          <span className="bt-sk">{c.k}</span>
          <span className="bt-sv" style={{ color: c.c }}>{c.v}</span>
          <span className="bt-ss">{c.sub}</span>
        </div>
      ))}
    </div>
  );
}

/* ============ competition compliance summary ============ */
function BtCompliance({ summary }) {
  const s = summary;
  const dl = window.dayLabelBt;
  const r1Fail = s.noTradeWeeks.length > 0;
  const r2Fail = s.ddBreachWeeks.length > 0;
  return (
    <div className="compliance">
      <div className={"rule-card " + (r1Fail ? "fail" : "pass")}>
        <div className="rule-top">
          <span className="rule-id">RULE 1 · DAILY ACTIVITY</span>
          <span className="rule-flag">{r1Fail ? "⚠ " + s.noTradeWeeks.length + " SESSION" + (s.noTradeWeeks.length > 1 ? "S" : "") + " FLAGGED" : "✓ COMPLIANT"}</span>
        </div>
        <div className="rule-desc">Agent must execute at least one trade every calendar day of the session.</div>
        {r1Fail && (
          <div className="rule-detail">
            {s.noTradeWeeks.map((w) => (
              <span key={w.label} className="rule-chip">{w.label} · idle {w.missedDays.map((d) => dl(w.start + d * 86400)).join(", ")}</span>
            ))}
          </div>
        )}
      </div>
      <div className={"rule-card " + (r2Fail ? "fail" : "pass")}>
        <div className="rule-top">
          <span className="rule-id">RULE 2 · SESSION DRAWDOWN</span>
          <span className="rule-flag">{r2Fail ? "⚠ " + s.ddBreachWeeks.length + " BREACH" + (s.ddBreachWeeks.length > 1 ? "ES" : "") : "✓ COMPLIANT"}</span>
        </div>
        <div className="rule-desc">No drawdown greater than 30% from any peak within a session.</div>
        {r2Fail && (
          <div className="rule-detail">
            {s.ddBreachWeeks.map((w) => (
              <span key={w.label} className="rule-chip">{w.label} · {(w.intraDD * 100).toFixed(1)}% intra-week</span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/* ============ week detail (focus + small multiples + trades) ============ */
function BtWeekDetail({ week, locked, focusIdx, setFocusIdx }) {
  const assets = week.assets;
  const focus = assets[Math.min(focusIdx, assets.length - 1)];
  const dl = window.dayLabelBt;

  return (
    <div className="wk-detail">
      <div className="wk-bar">
        <span className="wk-tag">{week.label}</span>
        <span className={"wk-locked " + (locked ? "on" : "off")}>
          {locked ? "📌 LOCKED" : "○ hover preview"}
        </span>
        <span className="wk-dates">{dl(week.start)} – {dl(week.end)}</span>
        <div className="wk-metrics">
          <div className="wk-metric">
            <span className="k">WEEK P&L</span>
            <span className="v" style={{ color: week.pnlPct >= 0 ? "var(--up)" : "var(--down)" }}>{_pct(week.pnlPct, 1)}</span>
          </div>
          <div className="wk-metric">
            <span className="k">PORTFOLIO</span>
            <span className="v">{_usd0(week.portfolioEnd)}</span>
          </div>
          <div className="wk-metric">
            <span className="k">P&L $</span>
            <span className="v" style={{ color: week.pnl >= 0 ? "var(--up)" : "var(--down)" }}>{(week.pnl >= 0 ? "+" : "") + _usd0(week.pnl)}</span>
          </div>
        </div>
      </div>

      <div className="wk-body">
        {week.dq.any && (
          <div className="wk-dq-banner">
            <div className="dq-msgs">
              <div className="dq-title">⚠ DISQUALIFYING EVENT{(week.dq.noTrade && week.dq.drawdown) ? "S" : ""} · THIS SESSION</div>
              {week.dq.drawdown && (
                <div className="dq-line">
                  <span className="dq-rule">RULE 2</span> intra-week drawdown hit <b>{(week.intraDD * 100).toFixed(1)}%</b> (limit −30%) — trough {window.dayTimeLabelBt(week.start + week.troughHour * 3600)}
                </div>
              )}
              {week.dq.noTrade && (
                <div className="dq-line">
                  <span className="dq-rule">RULE 1</span> no trade executed on <b>{week.dq.missedDays.map((d) => window.dayLabelBt(week.start + d * 86400)).join(", ")}</b> — agent sat the day out
                </div>
              )}
            </div>
            <div className="dq-spark">
              <BtSessionEquity path={week.equityPath} intraDD={week.intraDD} troughHour={week.troughHour} breach={week.dq.drawdown} />
            </div>
          </div>
        )}
        <div className="wk-grid-side">
          <div className="sm-title">
            <span>8 ASSETS · BY VOLATILITY RANK</span>
            <span style={{ color: "var(--faint)" }}>click to inspect</span>
          </div>
          <div className="sm-grid">
            {assets.map((a, i) => (
              <div key={a.sym} className={"sm-card" + (i === focusIdx ? " on" : "")} onClick={() => setFocusIdx(i)}>
                <div className="sm-card-top">
                  <span className="sm-sym">{a.sym}</span>
                  <span className="sm-rk">#{a.volRank}</span>
                </div>
                <BtMiniCandle candles={a.candles} holds={a.holds} />
                <div className="sm-card-top">
                  <span className="sm-rk" style={{ color: CLR[a.cls] }}>●</span>
                  <span className="sm-pnl" style={{ color: a.pnl >= 0 ? "var(--up)" : "var(--down)" }}>
                    {(a.pnl >= 0 ? "+" : "") + _usd0(a.pnl)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="wk-focus">
          <div className="fc-head">
            <span className="rank-badge">VOL #{focus.volRank}</span>
            <span className="fc-sym">{focus.sym}</span>
            <span className="fc-cls">{CLSL[focus.cls]}</span>
            <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--faint)" }}>1H · {focus.candles.length} bars</span>
            <div className="fc-metrics">
              <div className="fc-metric"><span className="k">WK P&L</span>
                <span className="v" style={{ color: focus.pnl >= 0 ? "var(--up)" : "var(--down)" }}>{(focus.pnl >= 0 ? "+" : "") + _usd0(focus.pnl)}</span>
              </div>
              <div className="fc-metric"><span className="k">RETURN</span>
                <span className="v" style={{ color: focus.pnlPct >= 0 ? "var(--up)" : "var(--down)" }}>{_pct(focus.pnlPct, 1)}</span>
              </div>
              <div className="fc-metric"><span className="k">ALLOC</span>
                <span className="v">{_usd0(focus.alloc)}</span>
              </div>
            </div>
          </div>

          <BtAssetChart asset={focus} key={week.idx + "-" + focus.sym} />

          <BtTradeList asset={focus} />
        </div>
      </div>
    </div>
  );
}

function BtTradeList({ asset }) {
  const dt = window.dayTimeLabelBt;
  // round trips come straight from holds (entry/exit already paired)
  const trips = asset.holds
    .map((h) => ({ entryPx: h.entryPx, exitPx: h.exitPx, entryT: h.entryT, exitT: h.exitT, usd: h.usd, core: h.core, pnl: h.qty * (h.exitPx - h.entryPx), pct: (h.exitPx - h.entryPx) / h.entryPx }))
    .sort((a, b) => a.entryT - b.entryT);
  const fmtP = (p) => p >= 100 ? p.toFixed(2) : p >= 1 ? p.toFixed(3) : p >= 0.001 ? p.toFixed(5) : p.toExponential(2);
  return (
    <div className="tl">
      <div className="tl-title">TRADE LOG · {trips.length} round trip{trips.length !== 1 ? "s" : ""}</div>
      {trips.length === 0 ? <div className="tl-empty">no executions this week</div> : (
        <table className="tl-table">
          <thead>
            <tr>
              <th>#</th><th>ENTRY (BUY)</th><th>EXIT (SELL)</th><th>SIZE</th><th>P&amp;L</th>
            </tr>
          </thead>
          <tbody>
            {trips.map((t, i) => (
              <tr key={i}>
                <td style={{ color: "var(--faint)" }}>{i + 1}{t.core ? "" : <span style={{ color: "var(--faint)", fontSize: 9 }}> s</span>}</td>
                <td><span className="tl-side"><span className="buy-tri">▲</span> {fmtP(t.entryPx)}</span> <span style={{ color: "var(--faint)" }}>· {dt(t.entryT)}</span></td>
                <td><span className="tl-side"><span className="sell-tri">▼</span> {fmtP(t.exitPx)}</span> <span style={{ color: "var(--faint)" }}>· {dt(t.exitT)}</span></td>
                <td style={{ textAlign: "right", color: "var(--muted)" }}>{_usd0(t.usd)}</td>
                <td style={{ textAlign: "right", color: t.pnl >= 0 ? "var(--up)" : "var(--down)", fontWeight: 600 }}>
                  {(t.pnl >= 0 ? "+" : "") + _usd0(t.pnl)}<span style={{ color: "var(--faint)", fontWeight: 400 }}> ({_pct(t.pct, 1)})</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

/* ============ 6-month token P&L table ============ */
function BtTokenTable({ tokens, nWeeks }) {
  const [sortKey, setSortKey] = useStateP("pnl");
  const [dir, setDir] = useStateP(-1);
  const maxAbs = Math.max(...tokens.map((t) => Math.abs(t.pnl)));

  const rows = useMemoP(() => {
    const arr = [...tokens].sort((a, b) => {
      if (sortKey === "sym") return a.sym.localeCompare(b.sym) * dir;
      return (a[sortKey] - b[sortKey]) * dir;
    });
    return arr;
  }, [tokens, sortKey, dir]);

  const click = (k) => { if (k === sortKey) setDir((d) => -d); else { setSortKey(k); setDir(k === "sym" ? 1 : -1); } };
  const caret = (k) => sortKey === k ? (dir === 1 ? " ▲" : " ▼") : "";

  return (
    <div className="tok-tbl">
      <div className="tok-head">
        <span className="tok-th l">#</span>
        <button className="tok-th l" style={{ background: "none", border: "none", cursor: "pointer", color: sortKey === "sym" ? "var(--orange)" : "var(--muted)" }} onClick={() => click("sym")}>TOKEN{caret("sym")}</button>
        <button className="tok-th" style={{ background: "none", border: "none", cursor: "pointer", color: sortKey === "pnl" ? "var(--orange)" : "var(--muted)" }} onClick={() => click("pnl")}>TOTAL P&L{caret("pnl")}</button>
        <button className="tok-th" style={{ background: "none", border: "none", cursor: "pointer", color: sortKey === "pct" ? "var(--orange)" : "var(--muted)" }} onClick={() => click("pct")}>RETURN %{caret("pct")}</button>
        <button className="tok-th tok-col-hide" style={{ background: "none", border: "none", cursor: "pointer", color: sortKey === "weeks" ? "var(--orange)" : "var(--muted)" }} onClick={() => click("weeks")}>WEEKS IN VOL-8{caret("weeks")}</button>
        <span className="tok-th tok-col-hide">CONTRIBUTION</span>
      </div>
      {rows.map((t, i) => (
        <div key={t.sym} className="tok-row">
          <span className="tok-rank">{i + 1}</span>
          <div className="tok-name">
            <span className="dot" style={{ background: CLR[t.cls] }} />
            <span className="s">{t.sym}</span>
          </div>
          <span className="tok-num tok-pnl" style={{ color: t.pnl >= 0 ? "var(--up)" : "var(--down)" }}>
            {(t.pnl >= 0 ? "+" : "") + _usd0(t.pnl)}
          </span>
          <span className="tok-num" style={{ color: t.pct >= 0 ? "var(--up)" : "var(--down)" }}>{_pct(t.pct, 1)}</span>
          <div className="tok-weeks tok-col-hide">
            <span className="tok-weekbar"><span style={{ width: (t.weeks / nWeeks) * 100 + "%" }} /></span>
            <span style={{ fontFamily: "var(--mono)", fontSize: 12, color: "var(--muted)", minWidth: 30, textAlign: "right" }}>{t.weeks}/{nWeeks}</span>
          </div>
          <div className="tok-spark tok-col-hide">
            <Sparkline data={cumContrib(t.contrib)} color={t.pnl >= 0 ? "#16D67A" : "#FF4D4D"} w={104} h={28} fill={true} />
          </div>
        </div>
      ))}
    </div>
  );
}

function cumContrib(arr) {
  let s = 0; const out = [0];
  for (const v of arr) { s += v; out.push(s); }
  return out;
}

Object.assign(window, { BtHeadline, BtCompliance, BtWeekDetail, BtTokenTable });
