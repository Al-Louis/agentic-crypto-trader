/* Apentic — chart components (SVG, animated). Exposes to window. */
const { useState, useEffect, useRef, useMemo } = React;

/* ---------- tiny helpers ---------- */
function niceExtent(arr, pad = 0.06) {
  let lo = Math.min(...arr), hi = Math.max(...arr);
  const d = (hi - lo) || 1;
  return [lo - d * pad, hi + d * pad];
}
function buildPath(vals, w, h, pad) {
  const [lo, hi] = niceExtent(vals);
  const n = vals.length;
  const x = (i) => pad + (i / (n - 1)) * (w - pad * 2);
  const y = (v) => h - pad - ((v - lo) / (hi - lo)) * (h - pad * 2);
  let d = "";
  vals.forEach((v, i) => (d += (i ? "L" : "M") + x(i).toFixed(2) + " " + y(v).toFixed(2) + " "));
  return { d, x, y };
}

/* ---------- Sparkline ---------- */
function Sparkline({ data, color = "#16D67A", w = 120, h = 36, fill = true }) {
  const { d, x, y } = useMemo(() => buildPath(data, w, h, 3), [data, w, h]);
  const last = data[data.length - 1];
  const lo = Math.min(...data), hi = Math.max(...data);
  const ly = h - 3 - ((last - lo) / ((hi - lo) || 1)) * (h - 6);
  const id = "sg" + Math.round(x(0) * 1000) + color.replace("#", "");
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ display: "block" }}>
      {fill && (
        <>
          <defs>
            <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity="0.28" />
              <stop offset="100%" stopColor={color} stopOpacity="0" />
            </linearGradient>
          </defs>
          <path d={`${d} L ${w - 3} ${h} L 3 ${h} Z`} fill={`url(#${id})`} />
        </>
      )}
      <path d={d} fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx={w - 3} cy={ly} r="2.4" fill={color} />
    </svg>
  );
}

/* ---------- Equity curve (big, animated draw-on) ---------- */
function EquityChart({ data, height = 260, color = "#FF7512" }) {
  const wrapRef = useRef(null);
  const [w, setW] = useState(720);
  const [hover, setHover] = useState(null);
  const pad = 8;

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const measure = () => setW(el.clientWidth || el.getBoundingClientRect().width || 720);
    measure();
    requestAnimationFrame(measure);
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    window.addEventListener("resize", measure);
    return () => { ro.disconnect(); window.removeEventListener("resize", measure); };
  }, []);

  const h = height;
  const { d, x, y } = useMemo(() => buildPath(data, w, h, pad), [data, w, h]);
  const area = `${d} L ${x(data.length - 1)} ${h - pad} L ${x(0)} ${h - pad} Z`;

  // baseline (start capital) reference
  const lo = Math.min(...data), hi = Math.max(...data);
  const baseY = h - pad - ((10000 - lo) / ((hi - lo) || 1)) * (h - pad * 2);

  const onMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const i = Math.round(((px - pad) / (w - pad * 2)) * (data.length - 1));
    const idx = Math.max(0, Math.min(data.length - 1, i));
    setHover({ i: idx, x: x(idx), y: y(data[idx]), v: data[idx] });
  };

  return (
    <div ref={wrapRef} style={{ width: "100%", position: "relative" }}>
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}
           onMouseMove={onMove} onMouseLeave={() => setHover(null)}
           style={{ display: "block", cursor: "crosshair" }}>
        <defs>
          <linearGradient id="eqfill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.30" />
            <stop offset="70%" stopColor={color} stopOpacity="0.04" />
            <stop offset="100%" stopColor={color} stopOpacity="0" />
          </linearGradient>
        </defs>
        {/* grid */}
        {[0.25, 0.5, 0.75].map((g) => (
          <line key={g} x1={pad} x2={w - pad} y1={pad + g * (h - pad * 2)} y2={pad + g * (h - pad * 2)}
                stroke="#23262E" strokeWidth="1" strokeDasharray="2 4" />
        ))}
        <line x1={pad} x2={w - pad} y1={baseY} y2={baseY} stroke="#3a3f4a" strokeWidth="1" strokeDasharray="5 4" />
        <text x={pad + 4} y={baseY - 5} fill="#6b7280" fontSize="10" fontFamily="'JetBrains Mono',monospace">START $10K</text>

        <path d={area} fill="url(#eqfill)" className="eq-area" />
        <path d={d} fill="none" stroke={color} strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round"
              pathLength="1" className="eq-line" />
        <circle cx={x(data.length - 1)} cy={y(data[data.length - 1])} r="3.5" fill={color} className="eq-dot" />
        <circle cx={x(data.length - 1)} cy={y(data[data.length - 1])} r="6.5" fill="none" stroke={color} strokeWidth="1.2" className="eq-dot2" />

        {hover && (
          <g>
            <line x1={hover.x} x2={hover.x} y1={pad} y2={h - pad} stroke="#4b5160" strokeWidth="1" />
            <circle cx={hover.x} cy={hover.y} r="4" fill="#0B0C0E" stroke={color} strokeWidth="2" />
          </g>
        )}
      </svg>
      {hover && (
        <div style={{
          position: "absolute", left: Math.min(w - 130, Math.max(0, hover.x - 60)), top: 6,
          background: "#0B0C0E", border: "1px solid #2c2f38", borderRadius: 7, padding: "5px 9px",
          fontFamily: "'JetBrains Mono',monospace", fontSize: 12, color: "#F5F6F7", pointerEvents: "none",
          boxShadow: "0 8px 24px rgba(0,0,0,.5)", whiteSpace: "nowrap",
        }}>
          <span style={{ color: "#7a808c" }}>EQ </span>
          ${hover.v.toLocaleString(undefined, { maximumFractionDigits: 0 })}
        </div>
      )}
    </div>
  );
}

/* ---------- Candlestick (focus asset, live-forming last candle) ---------- */
function CandleChart({ seed, height = 210, n = 38 }) {
  const wrapRef = useRef(null);
  const [w, setW] = useState(420);
  const baseRef = useRef(null);
  if (!baseRef.current) baseRef.current = APENTIC.candles(n, seed);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const measure = () => setW(el.clientWidth || el.getBoundingClientRect().width || 420);
    measure();
    requestAnimationFrame(measure);
    const ro = new ResizeObserver(measure);
    ro.observe(el); 
    window.addEventListener("resize", measure);
    return () => { ro.disconnect(); window.removeEventListener("resize", measure); };
  }, []);

  // animate the last forming candle
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 900);
    return () => clearInterval(id);
  }, []);

  const candles = useMemo(() => {
    const c = baseRef.current.map((x) => ({ ...x }));
    const last = c[c.length - 1];
    const wob = Math.sin(tick / 2) * (last.o * 0.004) + (Math.random() - 0.5) * last.o * 0.002;
    last.c = last.o + (last.c - last.o) * 0.5 + wob;
    last.h = Math.max(last.h, last.c);
    last.l = Math.min(last.l, last.c);
    return c;
  }, [tick]);

  const h = height, pad = 8;
  const allH = candles.map((c) => c.h), allL = candles.map((c) => c.l);
  const hi = Math.max(...allH), lo = Math.min(...allL);
  const y = (v) => pad + (1 - (v - lo) / ((hi - lo) || 1)) * (h - pad * 2);
  const slot = (w - pad * 2) / candles.length;
  const cw = Math.max(2, slot * 0.62);

  return (
    <div ref={wrapRef} style={{ width: "100%" }}>
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ display: "block" }}>
        {candles.map((c, i) => {
          const cx = pad + slot * (i + 0.5);
          const up = c.c >= c.o;
          const col = up ? "#16D67A" : "#FF4D4D";
          const last = i === candles.length - 1;
          return (
            <g key={i} opacity={last ? 0.95 : 0.85}>
              <line x1={cx} x2={cx} y1={y(c.h)} y2={y(c.l)} stroke={col} strokeWidth="1.1" />
              <rect x={cx - cw / 2} y={Math.min(y(c.o), y(c.c))} width={cw}
                    height={Math.max(1.5, Math.abs(y(c.o) - y(c.c)))} fill={col} rx="0.5" />
              {last && <rect x={cx - cw / 2 - 1} y={Math.min(y(c.o), y(c.c)) - 1} width={cw + 2}
                    height={Math.max(1.5, Math.abs(y(c.o) - y(c.c))) + 2} fill="none" stroke={col} strokeWidth="0.8" opacity="0.5" />}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/* ---------- Reward curve (training) ---------- */
function RewardChart({ data, height = 80, color = "#8b8f9a" }) {
  const wrapRef = useRef(null);
  const [w, setW] = useState(300);
  useEffect(() => {
    const el = wrapRef.current;
    const measure = () => setW(el.clientWidth || el.getBoundingClientRect().width || 300);
    measure();
    requestAnimationFrame(measure);
    const ro = new ResizeObserver(measure);
    ro.observe(el); 
    window.addEventListener("resize", measure);
    return () => { ro.disconnect(); window.removeEventListener("resize", measure); };
  }, []);
  const { d } = useMemo(() => buildPath(data, w, height, 4), [data, w, height]);
  return (
    <div ref={wrapRef} style={{ width: "100%" }}>
      <svg width={w} height={height} viewBox={`0 0 ${w} ${height}`} style={{ display: "block" }}>
        <path d={d} fill="none" stroke={color} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  );
}

Object.assign(window, { Sparkline, EquityChart, CandleChart, RewardChart });
