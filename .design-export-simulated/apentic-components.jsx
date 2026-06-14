/* Apentic — core UI components. Exposes to window. */
const { useState: useStateC, useEffect: useEffectC, useRef: useRefC } = React;

/* ---------- formatting ---------- */
const fmtUsd = (v, d = 0) =>
  (v < 0 ? "-$" : "$") + Math.abs(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
const fmtPct = (v, d = 2) => (v >= 0 ? "+" : "") + v.toFixed(d) + "%";
const fmtPx = (v) => (v >= 100 ? v.toLocaleString(undefined, { maximumFractionDigits: 2 }) : v.toFixed(v < 1 ? 4 : 2));

/* ---------- Logo ---------- */
function Logo({ size = 34 }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 11 }}>
      <img src="assets/ape_avatar.png" alt="Apentic" width={size} height={size}
           style={{ borderRadius: "50%", border: "1.5px solid var(--line)", boxShadow: "0 0 0 2px rgba(255,117,18,.18)" }} />
      <div style={{ lineHeight: 1 }}>
        <div className="brandmark" style={{ fontSize: 21, letterSpacing: "0.01em" }}>
          <span className="bm-ape">APE</span><span className="ntic">NTIC</span>
        </div>
        <div style={{ fontFamily: "var(--mono)", fontSize: 9.5, color: "var(--muted)", letterSpacing: "0.22em", marginTop: 2 }}>
          AGENTIC · APE · ALPHA
        </div>
      </div>
    </div>
  );
}

/* ---------- Nav ---------- */
function Nav({ page, onNav, showStatus }) {
  const items = [["overview", "Overview"], ["markets", "Markets"], ["simulated", "Simulated Trades"], ["leaderboard", "Leaderboard"]];
  return (
    <nav className="nav">
      <Logo />
      <div className="nav-links">
        {items.map(([k, label]) => (
          <button key={k} className={"nav-link" + (page === k ? " active" : "")} onClick={() => onNav(k)}>
            {label}
          </button>
        ))}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
        {showStatus && (
          <div className="live-pill">
            <span className="live-dot" />
            TRAINING
            <span style={{ color: "var(--muted)", fontWeight: 400 }}>· EP {APENTIC.KPIS.episode.toLocaleString()}</span>
          </div>
        )}
      </div>
    </nav>
  );
}

/* ---------- Ticker tape ---------- */
function Ticker() {
  const items = APENTIC.ASSETS.map((a, i) => {
    const ch = (APENTIC.mulberry(i + 3)() - 0.45) * 9;
    return { ...a, ch };
  });
  const row = [...items, ...items, ...items];
  return (
    <div className="ticker">
      <div className="ticker-track">
        {row.map((a, i) => (
          <span key={i} className="ticker-item">
            <span className="t-sym">{a.sym}</span>
            <span className="t-px">${fmtPx(a.px)}</span>
            <span style={{ color: a.ch >= 0 ? "var(--up)" : "var(--down)" }}>{fmtPct(a.ch, 2)}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

/* ---------- Hero: reacting ape ---------- */
function moodOf(health) {
  if (health >= 72) return { key: "printing", label: "PRINTING", sub: "agent is cooking", color: "var(--up)" };
  if (health >= 45) return { key: "holding", label: "HOLDING THE LINE", sub: "grinding it out", color: "var(--orange)" };
  if (health >= 24) return { key: "bleeding", label: "BLEEDING", sub: "drawdown forming", color: "#FF8A3D" };
  return { key: "rekt", label: "GETTING REKT", sub: "max pain incoming", color: "var(--down)" };
}

function Hero({ health, sessionPnl, jolt, note }) {
  const mood = moodOf(health);
  const damaged = 1 - health / 100; // 0 healthy -> 1 wrecked
  return (
    <section className={"hero" + (jolt ? " jolt" : "")}>
      <div className="hero-stage" style={{ "--glow": mood.color }}>
        <div className="hero-floor" />
        <div className="ring ring-a" />
        <div className="ape-wrap">
          <img className="ape ape-base" src="assets/ape_unharmed.png" alt="" draggable="false" />
          <img className="ape ape-dmg" src="assets/ape_damaged.png" alt="" draggable="false"
               style={{ opacity: Math.max(0, Math.min(1, (damaged - 0.18) * 1.5)) }} />
        </div>
        <div className="status-tag" style={{ color: mood.color, borderColor: mood.color }}>
          <span className="status-dot" style={{ background: mood.color }} />
          {mood.label}
        </div>
      </div>

      <div className="hero-readout">
        <div className="kicker">SESSION VITALS · LIVE</div>
        <Vitals health={health} mood={mood} />
        <div className="hero-session">
          <div>
            <div className="hr-label">TODAY'S P&amp;L</div>
            <div className="hr-big" style={{ color: sessionPnl >= 0 ? "var(--up)" : "var(--down)" }}>
              {fmtPct(sessionPnl)}
            </div>
          </div>
          <div className="hr-divider" />
          <div>
            <div className="hr-label">MOOD</div>
            <div className="hr-mood" style={{ color: mood.color }}>{mood.sub}</div>
          </div>
        </div>
        <p className="hero-note">{note}</p>
      </div>
    </section>
  );
}

function Vitals({ health, mood }) {
  return (
    <div className="vitals">
      <div className="vitals-head">
        <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--muted)", letterSpacing: ".12em" }}>HP</span>
        <span style={{ fontFamily: "var(--mono)", fontSize: 13, color: mood.color, fontWeight: 600 }}>
          {Math.round(health)} / 100
        </span>
      </div>
      <div className="vitals-bar">
        <div className="vitals-fill" style={{ width: health + "%", background: mood.color }} />
        {[25, 50, 75].map((m) => <span key={m} className="vitals-notch" style={{ left: m + "%" }} />)}
      </div>
    </div>
  );
}

/* ---------- KPI big card (portfolio + pnl) ---------- */
function KpiHeadline({ portfolio, pnlPct, pnlUsd, equity }) {
  return (
    <div className="kpi-headline">
      <div className="kpi-row">
        <div>
          <div className="hr-label">PORTFOLIO VALUE</div>
          <div className="kpi-port">{fmtUsd(portfolio, 2)}</div>
        </div>
        <div className="kpi-pnl" style={{ color: pnlPct >= 0 ? "var(--up)" : "var(--down)" }}>
          <div className="kpi-pnl-pct">{fmtPct(pnlPct)}</div>
          <div className="kpi-pnl-usd">{fmtUsd(pnlUsd, 0)} all-time</div>
        </div>
      </div>
      <EquityChart data={equity} height={250} color="var(--orange)" />
    </div>
  );
}

/* ---------- Stat card ---------- */
function StatCard({ label, value, sub, color, spark, sparkColor, accent }) {
  return (
    <div className={"stat-card" + (accent ? " accent" : "")}>
      <div className="stat-label">{label}</div>
      <div className="stat-value" style={{ color: color || "var(--text)" }}>{value}</div>
      <div className="stat-foot">
        {sub && <span className="stat-sub">{sub}</span>}
        {spark && <Sparkline data={spark} color={sparkColor || "#16D67A"} w={86} h={26} />}
      </div>
    </div>
  );
}

Object.assign(window, { Logo, Nav, Ticker, Hero, Vitals, KpiHeadline, StatCard, moodOf, fmtUsd, fmtPct, fmtPx });
