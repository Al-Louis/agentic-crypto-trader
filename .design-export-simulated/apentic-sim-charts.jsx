/* Apentic — Simulated Trades charts. window.{BtEquityColumns, BtMiniCandle, BtAssetChart} */
const { useState: useStateSC, useEffect: useEffectSC, useRef: useRefSC, useMemo: useMemoSC } = React;

function useWidth(ref, init) {
  const [w, setW] = useStateSC(init);
  useEffectSC(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => setW(el.clientWidth || el.getBoundingClientRect().width || init);
    measure(); requestAnimationFrame(measure);
    const ro = new ResizeObserver(measure); ro.observe(el);
    window.addEventListener("resize", measure);
    return () => { ro.disconnect(); window.removeEventListener("resize", measure); };
  }, []);
  return w;
}

const usd0 = (v) => (v < 0 ? "-$" : "$") + Math.abs(v).toLocaleString(undefined, { maximumFractionDigits: 0 });
const dayLabel = (unix) => new Date(unix * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric" });
const dayTimeLabel = (unix) => {
  const d = new Date(unix * 1000);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" }) + " " +
         d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", hour12: false });
};
// aggregate hourly candles into coarser buckets (e.g. 24 = daily)
function aggCandles(candles, bucket) {
  const out = [];
  for (let i = 0; i < candles.length; i += bucket) {
    const seg = candles.slice(i, i + bucket);
    if (!seg.length) break;
    out.push({
      t: seg[0].t, o: seg[0].o,
      h: Math.max(...seg.map((c) => c.h)), l: Math.min(...seg.map((c) => c.l)),
      c: seg[seg.length - 1].c, v: seg.reduce((a, c) => a + c.v, 0),
    });
  }
  return out;
}

/* ============ HERO: 24 equity columns ============ */
function BtEquityColumns({ weeks, equity, selected, locked, onHover, onLock }) {
  const ref = useRefSC(null);
  const w = useWidth(ref, 1000);
  const [tip, setTip] = useStateSC(null);
  const H = 300, m = { l: 52, r: 14, t: 18, b: 26 };
  const pw = Math.max(120, w - m.l - m.r), ph = H - m.t - m.b;

  const vals = weeks.map((wk) => wk.portfolioEnd);
  const minV = Math.min(...vals, 10000), maxV = Math.max(...vals);
  const floor = minV * 0.94, cap = maxV * 1.04;
  const y = (v) => m.t + (1 - (v - floor) / (cap - floor)) * ph;
  const n = weeks.length;
  const slot = pw / n;
  const cw = slot * 0.62;
  const cx = (i) => m.l + slot * (i + 0.5);

  const peakIdx = vals.indexOf(maxV);
  const yticks = 4;
  const tickVals = Array.from({ length: yticks + 1 }, (_, i) => floor + (cap - floor) * (i / yticks));

  const linePts = weeks.map((wk, i) => `${cx(i)},${y(wk.portfolioEnd)}`).join(" ");

  return (
    <div ref={ref} style={{ width: "100%", position: "relative" }}>
      <svg width={w} height={H} viewBox={`0 0 ${w} ${H}`} style={{ display: "block" }}
           onMouseLeave={() => { setTip(null); onHover(null); }}>
        {/* y grid + labels */}
        {tickVals.map((tv, i) => (
          <g key={i}>
            <line x1={m.l} x2={w - m.r} y1={y(tv)} y2={y(tv)} stroke="#1E2128" strokeWidth="1" />
            <text x={m.l - 8} y={y(tv) + 3} textAnchor="end" fontFamily="var(--mono)" fontSize="9.5" fill="#6b7280">{usd0(tv)}</text>
          </g>
        ))}
        {/* start capital line */}
        <line x1={m.l} x2={w - m.r} y1={y(10000)} y2={y(10000)} stroke="#3a3f4a" strokeWidth="1" strokeDasharray="5 4" />
        <text x={w - m.r} y={y(10000) - 5} textAnchor="end" fontFamily="var(--mono)" fontSize="9" fill="#6b7280">START $10K</text>

        {/* columns */}
        {weeks.map((wk, i) => {
          const up = wk.pnlPct >= 0;
          const col = up ? "#16D67A" : "#FF4D4D";
          const isSel = selected === i;
          const isLock = locked === i;
          const mag = Math.min(1, Math.abs(wk.pnlPct) / 0.12);
          const op = isSel ? 1 : 0.34 + mag * 0.4;
          const top = y(wk.portfolioEnd);
          const isDq = wk.dq && wk.dq.any;
          return (
            <g key={i}>
              <rect x={cx(i) - cw / 2} y={top} width={cw} height={Math.max(2, y(floor) - top)}
                    fill={col} fillOpacity={op} rx="1.5"
                    stroke={isSel ? col : "none"} strokeWidth={isSel ? 1.5 : 0} />
              {isLock && <rect x={cx(i) - cw / 2 - 1.5} y={top - 1.5} width={cw + 3} height={y(floor) - top + 1.5}
                    fill="none" stroke="var(--orange)" strokeWidth="1.5" rx="2" strokeDasharray="3 2" />}
              {isDq && (
                <g>
                  <rect className="dq-radiate" x={cx(i) - cw / 2 - 2.5} y={top - 2.5} width={cw + 5} height={y(floor) - top + 5}
                        rx="3" fill="none" stroke="var(--orange)" strokeWidth="2" />
                  <rect className="dq-ring" x={cx(i) - cw / 2 - 2.5} y={top - 2.5} width={cw + 5} height={y(floor) - top + 5}
                        rx="3" fill="none" stroke="var(--orange)" strokeWidth="2" />
                  <text x={cx(i)} y={top - 9} textAnchor="middle" fontSize="11" fill="var(--orange)">⚠</text>
                </g>
              )}
            </g>
          );
        })}

        {/* equity line over tops */}
        <polyline points={linePts} fill="none" stroke="#E8EAED" strokeOpacity="0.5" strokeWidth="1.4" />
        {weeks.map((wk, i) => (
          <circle key={i} cx={cx(i)} cy={y(wk.portfolioEnd)} r={selected === i ? 3.2 : 1.8}
                  fill={selected === i ? "var(--orange)" : "#E8EAED"} fillOpacity={selected === i ? 1 : 0.6} />
        ))}

        {/* peak marker */}
        <g>
          <circle cx={cx(peakIdx)} cy={y(maxV)} r="3" fill="none" stroke="#FFD24A" strokeWidth="1.5" />
          <text x={cx(peakIdx)} y={y(maxV) - 9} textAnchor="middle" fontFamily="var(--mono)" fontSize="9" fill="#FFD24A">PEAK {usd0(maxV)}</text>
        </g>

        {/* x labels every 4 weeks */}
        {weeks.map((wk, i) => (i % 4 === 0 || i === n - 1) ? (
          <text key={i} x={cx(i)} y={H - 8} textAnchor="middle" fontFamily="var(--mono)" fontSize="9" fill="#6b7280">{wk.label}</text>
        ) : null)}

        {/* hit areas */}
        {weeks.map((wk, i) => (
          <rect key={i} className="heq-col-hit" x={m.l + slot * i} y={m.t} width={slot} height={ph}
                fill="transparent"
                onMouseMove={(e) => { setTip({ wk, x: e.clientX, y: e.clientY }); onHover(i); }}
                onMouseEnter={(e) => { setTip({ wk, x: e.clientX, y: e.clientY }); onHover(i); }}
                onClick={() => onLock(i)} />
        ))}
      </svg>

      {tip && (
        <div className="heq-tip" style={{ left: Math.min(window.innerWidth - 180, tip.x + 14), top: tip.y + 14 }}>
          <div className="wk">{tip.wk.label} · {dayLabel(tip.wk.start)}–{dayLabel(tip.wk.end)}</div>
          <div className="row"><span className="k">Portfolio</span><span>{usd0(tip.wk.portfolioEnd)}</span></div>
          <div className="row"><span className="k">Week P&L</span>
            <span style={{ color: tip.wk.pnlPct >= 0 ? "var(--up)" : "var(--down)" }}>
              {(tip.wk.pnlPct >= 0 ? "+" : "") + usd0(tip.wk.pnl)} ({(tip.wk.pnlPct * 100).toFixed(1)}%)
            </span>
          </div>
          {tip.wk.dq && tip.wk.dq.drawdown && (
            <div className="row dq"><span>⚠ Session drawdown</span>
              <span>{(tip.wk.intraDD * 100).toFixed(0)}% · breaches −30%</span></div>
          )}
          {tip.wk.dq && tip.wk.dq.noTrade && (
            <div className="row dq"><span>⚠ Missed trade day</span>
              <span>{tip.wk.dq.missedDays.map((d) => dayLabel(tip.wk.start + d * 86400)).join(", ")}</span></div>
          )}
          <div className="row"><span className="k">click</span><span style={{ color: "var(--orange)" }}>to lock →</span></div>
        </div>
      )}
    </div>
  );
}

/* ============ mini candlestick (small multiple) — hourly aggregated to daily ============ */
function BtMiniCandle({ candles, holds, w = 150, h = 50 }) {
  const pad = 3;
  const bucket = candles.length > 40 ? 24 : 1; // hourly → daily thumbnail
  const cd = bucket > 1 ? aggCandles(candles, bucket) : candles;
  const hd2 = bucket > 1 && holds
    ? holds.map((hd) => ({ from: Math.floor(hd.from / bucket), to: Math.ceil(hd.to / bucket) }))
    : holds;
  const allH = cd.map((c) => c.h), allL = cd.map((c) => c.l);
  const hi = Math.max(...allH), lo = Math.min(...allL);
  const y = (v) => pad + (1 - (v - lo) / ((hi - lo) || 1)) * (h - pad * 2);
  const slot = (w - pad * 2) / cd.length;
  const cw = Math.max(2, slot * 0.6);
  const x = (i) => pad + slot * (i + 0.5);
  return (
    <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" style={{ display: "block" }}>
      {hd2 && hd2.map((hd, k) => (
        <rect key={k} x={x(hd.from) - cw / 2} y={pad} width={Math.max(cw, x(hd.to) - x(hd.from) + cw)} height={h - pad * 2}
              fill="var(--orange)" fillOpacity="0.08" />
      ))}
      {cd.map((c, i) => {
        const up = c.c >= c.o; const col = up ? "#16D67A" : "#FF4D4D";
        return (
          <g key={i}>
            <line x1={x(i)} x2={x(i)} y1={y(c.h)} y2={y(c.l)} stroke={col} strokeWidth="1" />
            <rect x={x(i) - cw / 2} y={Math.min(y(c.o), y(c.c))} width={cw} height={Math.max(1, Math.abs(y(c.o) - y(c.c)))} fill={col} rx="0.5" />
          </g>
        );
      })}
    </svg>
  );
}

/* ============ big asset chart: candles + execs + holding + volume ============ */
function BtAssetChart({ asset }) {
  const ref = useRefSC(null);
  const w = useWidth(ref, 600);
  const [tip, setTip] = useStateSC(null);
  const candles = asset.candles;
  const H = 380;
  const m = { l: 56, r: 16, t: 16, b: 22 };
  const volH = 56, gap = 10;
  const priceH = H - m.t - m.b - volH - gap;
  const pw = Math.max(120, w - m.l - m.r);

  const allH = candles.map((c) => c.h), allL = candles.map((c) => c.l);
  const hi = Math.max(...allH), lo = Math.min(...allL);
  const pad = (hi - lo) * 0.08 || hi * 0.1;
  const y = (v) => m.t + (1 - (v - (lo - pad)) / ((hi + pad) - (lo - pad))) * priceH;
  const slot = pw / candles.length;
  const cw = Math.max(1.5, slot * 0.62);
  const x = (i) => m.l + slot * (i + 0.5);
  const dayMarks = candles.map((c, i) => (i % 24 === 0 ? i : -1)).filter((i) => i >= 0);

  const maxVol = Math.max(...candles.map((c) => c.v));
  const volTop = m.t + priceH + gap;
  const vy = (v) => volTop + (1 - v / maxVol) * volH;

  const fmtP = (p) => p >= 100 ? p.toFixed(2) : p >= 1 ? p.toFixed(3) : p >= 0.001 ? p.toFixed(5) : p.toExponential(2);
  const yticks = 4;
  const tvals = Array.from({ length: yticks + 1 }, (_, i) => (lo - pad) + ((hi + pad) - (lo - pad)) * (i / yticks));

  return (
    <div className="ac-wrap" ref={ref} style={{ position: "relative" }}>
      <svg width={w} height={H} viewBox={`0 0 ${w} ${H}`} style={{ display: "block" }} onMouseLeave={() => setTip(null)}>
        {/* y grid */}
        {tvals.map((tv, i) => (
          <g key={i}>
            <line x1={m.l} x2={w - m.r} y1={y(tv)} y2={y(tv)} stroke="#1A1D23" strokeWidth="1" />
            <text x={m.l - 8} y={y(tv) + 3} textAnchor="end" fontFamily="var(--mono)" fontSize="9" fill="#6b7280">{fmtP(tv)}</text>
          </g>
        ))}

        {/* holding shaded regions */}
        {asset.holds.map((hd, k) => (
          <g key={k}>
            <rect x={x(hd.from)} y={m.t} width={x(hd.to) - x(hd.from)} height={priceH}
                  fill="var(--orange)" fillOpacity="0.07" />
            <line x1={x(hd.from)} x2={x(hd.from)} y1={m.t} y2={m.t + priceH} stroke="var(--orange)" strokeOpacity="0.3" strokeWidth="1" strokeDasharray="2 2" />
            <line x1={x(hd.to)} x2={x(hd.to)} y1={m.t} y2={m.t + priceH} stroke="var(--orange)" strokeOpacity="0.3" strokeWidth="1" strokeDasharray="2 2" />
          </g>
        ))}

        {/* day-boundary separators */}
        {dayMarks.map((i) => (
          <line key={"d" + i} x1={x(i) - slot / 2} x2={x(i) - slot / 2} y1={m.t} y2={volTop + volH}
                stroke="#1A1D23" strokeWidth="1" />
        ))}

        {/* candles */}
        {candles.map((c, i) => {
          const up = c.c >= c.o; const col = up ? "#16D67A" : "#FF4D4D";
          return (
            <g key={i}>
              <line x1={x(i)} x2={x(i)} y1={y(c.h)} y2={y(c.l)} stroke={col} strokeWidth="1" />
              <rect x={x(i) - cw / 2} y={Math.min(y(c.o), y(c.c))} width={cw} height={Math.max(1, Math.abs(y(c.o) - y(c.c)))} fill={col} rx="0.5" />
              {/* volume bar */}
              <rect x={x(i) - cw / 2} y={vy(c.v)} width={cw} height={volTop + volH - vy(c.v)} fill={col} fillOpacity="0.4" rx="0.5" />
            </g>
          );
        })}
        <line x1={m.l} x2={w - m.r} y1={volTop + volH} y2={volTop + volH} stroke="#1E2128" strokeWidth="1" />
        <text x={m.l - 8} y={volTop + 10} textAnchor="end" fontFamily="var(--mono)" fontSize="8" fill="#5b626d">VOL</text>

        {/* execution markers */}
        {asset.trades.map((tr, k) => {
          const c = candles[tr.di];
          const isBuy = tr.side === "BUY";
          const col = isBuy ? "#16D67A" : "#FF4D4D";
          const py = isBuy ? y(c.l) + 14 : y(c.h) - 14;
          const tri = isBuy ? `${x(tr.di)},${py - 6} ${x(tr.di) - 5},${py + 3} ${x(tr.di) + 5},${py + 3}`
                            : `${x(tr.di)},${py + 6} ${x(tr.di) - 5},${py - 3} ${x(tr.di) + 5},${py - 3}`;
          return (
            <g key={k}>
              <polygon points={tri} fill={col} stroke="#0A0B0D" strokeWidth="0.8" />
            </g>
          );
        })}

        {/* hover hit-rects (one per candle, full height) */}
        {candles.map((c, i) => (
          <rect key={"h" + i} x={m.l + slot * i} y={m.t} width={slot} height={priceH + gap + volH}
                fill="transparent" style={{ cursor: "crosshair" }}
                onMouseMove={(e) => setTip({ c, i, x: e.clientX, y: e.clientY })}
                onMouseEnter={(e) => setTip({ c, i, x: e.clientX, y: e.clientY })} />
        ))}

        {/* x day labels (one per day boundary) */}
        {dayMarks.map((i) => (
          <text key={"l" + i} x={x(i)} y={H - 6} textAnchor="start" fontFamily="var(--mono)" fontSize="9" fill="#6b7280">{dayLabel(candles[i].t)}</text>
        ))}
      </svg>

      <div className="ac-legend">
        <span className="lg"><span style={{ color: "var(--up)" }}>▲</span> buy execution</span>
        <span className="lg"><span style={{ color: "var(--down)" }}>▼</span> sell execution</span>
        <span className="lg"><span className="sw" style={{ width: 10, height: 10, background: "color-mix(in srgb, var(--orange) 35%, transparent)", borderRadius: 2, display: "inline-block" }}></span> in position</span>
      </div>

      {tip && (
        <div className="ac-tip" style={{ left: Math.min(window.innerWidth - 170, tip.x + 14), top: tip.y + 14 }}>
          <div style={{ marginBottom: 3, color: "var(--muted)" }}>{dayTimeLabel(tip.c.t)}</div>
          <div><span className="k">O</span> {fmtP(tip.c.o)} <span className="k">H</span> {fmtP(tip.c.h)}</div>
          <div><span className="k">L</span> {fmtP(tip.c.l)} <span className="k">C</span> {fmtP(tip.c.c)}</div>
        </div>
      )}
    </div>
  );
}

Object.assign(window, { BtEquityColumns, BtMiniCandle, BtAssetChart, BtSessionEquity, usd0bt: usd0, dayLabelBt: dayLabel, dayTimeLabelBt: dayTimeLabel });

/* ============ intra-week session equity (MTM path) ============ */
function BtSessionEquity({ path, intraDD, troughHour, breach, label }) {
  const ref = useRefSC(null);
  const w = useWidth(ref, 360);
  const h = 92, pad = 6, mb = 14;
  const lo = Math.min(...path), hi = Math.max(...path);
  const span = (hi - lo) || 1;
  const x = (i) => pad + (i / (path.length - 1)) * (w - pad * 2);
  const y = (v) => pad + (1 - (v - lo) / span) * (h - pad - mb);
  const col = breach ? "#FF4D4D" : "var(--up)";
  const d = path.map((v, i) => (i ? "L" : "M") + x(i).toFixed(1) + " " + y(v).toFixed(1)).join(" ");
  // running peak to mark the peak the trough fell from
  let pk = path[0], pkHour = 0;
  for (let i = 0; i <= troughHour; i++) { if (path[i] > pk) { pk = path[i]; pkHour = i; } }
  const gid = "se" + Math.round(path[0]);
  return (
    <div ref={ref} style={{ width: "100%" }}>
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ display: "block" }}>
        <defs>
          <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={col} stopOpacity="0.22" />
            <stop offset="100%" stopColor={col} stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d={`${d} L ${x(path.length - 1)} ${h - mb} L ${x(0)} ${h - mb} Z`} fill={`url(#${gid})`} />
        <path d={d} fill="none" stroke={col} strokeWidth="1.6" strokeLinejoin="round" />
        {/* peak → trough drop marker */}
        <circle cx={x(pkHour)} cy={y(pk)} r="2.6" fill="#FFD24A" />
        <line x1={x(troughHour)} x2={x(troughHour)} y1={y(pk)} y2={y(path[troughHour])} stroke={col} strokeWidth="1" strokeDasharray="2 2" />
        <circle cx={x(troughHour)} cy={y(path[troughHour])} r="3" fill={col} stroke="#0A0B0D" strokeWidth="1" />
        <text x={Math.min(w - 4, x(troughHour) + 6)} y={y(path[troughHour]) + 4} fontFamily="var(--mono)" fontSize="10" fill={col}
              textAnchor={x(troughHour) > w - 60 ? "end" : "start"}>
          {(intraDD * 100).toFixed(1)}% intra
        </text>
        <text x={pad} y={h - 3} fontFamily="var(--mono)" fontSize="8.5" fill="#5b626d">{label || "SESSION EQUITY (hourly MTM)"}</text>
      </svg>
    </div>
  );
}
