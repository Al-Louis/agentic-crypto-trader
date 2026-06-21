"""Maintain the rolling top-3 `simulated_leaderboard.json` for the "Simulated Trades" dashboard.

Phase 2 of [[Dashboard Leaderboard]] (Phase 1 = `simulate_weekly`'s 3-window capture, DONE;
Phase 3 = the rl_loop verdict-phase hook, out of scope here). For a given seed run-id this:

  1. reads the seed's two ranking SCORES (below) from its already-published `simulated_trades.json`
     `meta.windows` (the cold-weekly per-window PnL Phase 1 emits) — the model's OWN performance;
  2. reads the seed's CONFIG 4-seed-mean val + DQ pass from `experiments/ledger.jsonl` (the guard);
  3. builds the leaderboard entry;
  4. upserts it into the current top-3 (sort by `weekly_score` desc, truncate to 3), identifying any
     EVICTED run-id (was listed, now isn't);
  5. PUBLISHES the updated leaderboard + the rank-1 CHAMPION (`simulated_champion.json` — the model
     to deploy, == the best `weekly_score`; auto-tracks #1 each publish) + appends the evicted file
     to `orphans.json` (the publish key can PUT but **not byte-delete** — [[apentic-publisher-no-delete]]
     — so an evicted seed is simply de-listed; its 4.7MB trades file stays orphaned in S3 for a later
     manual purge), then CloudFront-invalidates them.

  python scripts/publish_leaderboard.py --run-id ppo-event-rdLe4-wkw-ef0af8f-s3 [--no-publish]

DESKTOP-ONLY for the publish step (needs the APENTIC publish creds, like `simulate_weekly`). The
PURE ranking logic (`update_leaderboard`) is torch- and I/O-free so it is unit-tested on the laptop.

--- the two scores: model-vs-model, NO rung-0 (decided with the user 2026-06-17) ---
We deliberately do NOT rank by the gate's `edge_vs_rung0` anymore. The rung-0 RULE is not a stable
yardstick — it is a DQ-breaching, high-variance rule that rips +19.8%/wk on the recent test-window
bull, which drags every risk-managed policy's combined edge negative and ranks a *refuted* config
on top. For comparing our own trained models across iterations that is actively misleading. Instead
each entry carries two scores, both the model's OWN cold-weekly PnL straight from `meta.windows`:

  * `weekly_score`  (the PRIMARY rank) — the model's expected return in *any random week* = its mean
    per-week return over the OUT-OF-SAMPLE weeks (val + test): `(val.ret_sum + test.ret_sum) /
    (val.n_weeks + test.n_weeks)`. The live competition window (one week) is exactly a "random week",
    so this is the hackathon predictor. Ranked descending; `None` sorts last.
  * `cumulative_score` (a SECONDARY display stat) — `windows.overall.ret_sum`, the 28-week cumulative,
    the hold-across-weeks / long-run-deployment view the user weights for post-competition use.

The anti-cherry-pick guard rides ALONGSIDE the score (not baked into it): `config_seed_mean` +
`dq_pass` from the ledger expose a lucky single seed of an otherwise-weak config (e.g. a seed that
tops `weekly_score` while its config's 4-seed mean is ~0). `score_source` records how the scores
resolved, for auditability.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, "scripts")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

LEADERBOARD_KEY = "simulated_leaderboard.json"
CHAMPION_KEY = "simulated_champion.json"   # the rank-1 entry, published standalone (the deploy pick)
ORPHANS_KEY = "orphans.json"
DEFAULT_K = 3
DEFAULT_LEDGER = os.path.join("experiments", "ledger.jsonl")
CDN_BASE = "https://data.alexlouis.dev"   # read-only confirmation fetches (Phase 2 de-risking)


# ============================== PURE RANKING LOGIC (torch- and I/O-free) ==========================

def update_leaderboard(current: list[dict], new_entry: dict, k: int = DEFAULT_K
                       ) -> tuple[list[dict], list[str]]:
    """Insert/replace `new_entry`, keep the top-`k` by `cumulative_score` (6-mo return), reassign ranks.

    PURE — no torch, no network, no disk — so the eviction maths is unit-testable. Returns
    `(new_list, evicted_run_ids)` where `evicted_run_ids` are the run-ids that WERE listed in
    `current` (top-k) but are NOT in the returned list (de-listed by this update). A `new_entry`
    that fails to make the cut is itself "evicted" — it appears in neither the new list nor
    (unless it was already present) the evicted set; the caller must therefore not publish its
    trades file blindly (it checks membership first).

    Semantics:
      * UPSERT — an entry with the same `run_id` as `new_entry` is REPLACED, never duplicated.
      * SORT — descending `cumulative_score` (the 6-month total-return ranker). Ties are broken
        DETERMINISTICALLY by `run_id` (ascending), so the order is stable regardless of input
        order. A `cumulative_score` of None sorts last (treated as -inf) — an entry that couldn't
        resolve a score never out-ranks a real one.
      * TRUNCATE to `k`, then reassign `rank` = 1..k on the survivors (1 = best).
      * EVICTION = set difference on run-ids over the TRUNCATED top-k: any run-id that was in
        `current` (the prior listed set, already <=k) and isn't in the new list.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not new_entry.get("run_id"):
        raise ValueError("new_entry must carry a non-empty run_id")

    prior_ids = {e["run_id"] for e in current}

    merged = [dict(e) for e in current if e.get("run_id") != new_entry["run_id"]]
    merged.append(dict(new_entry))

    # -inf for a missing score so it never out-ranks a real one; run_id asc as the stable tiebreak.
    def sort_key(e: dict):
        s = e.get("cumulative_score")        # rank by 6-MONTH CUMULATIVE return (user, 2026-06-21);
        return (-(float("-inf") if s is None else float(s)), str(e.get("run_id")))  # weekly_score still shown

    merged.sort(key=sort_key)
    top = merged[:k]

    for i, e in enumerate(top):
        e["rank"] = i + 1

    kept_ids = {e["run_id"] for e in top}
    evicted = sorted(prior_ids - kept_ids)
    return top, evicted


def resolve_scores(windows: dict) -> tuple[float | None, float | None, str]:
    """The two leaderboard scores from a seed's published `meta.windows` (its OWN cold-weekly PnL).

    Returns `(weekly_score, cumulative_score, source)`:
      * `weekly_score`     = OOS per-week mean = `(val.ret_sum + test.ret_sum) / (val.n_weeks +
                             test.n_weeks)` — "best in a random week", the PRIMARY rank.
      * `cumulative_score` = `overall.ret_sum` — the 28-week cumulative (deployment display stat).
    `(None, None, "unavailable")` if `windows` is missing/empty — the caller still builds an entry
    (it sorts last) rather than crashing. See the module docstring for why this replaced edge_vs_rung0.
    """
    if not isinstance(windows, dict):
        return None, None, "unavailable"

    def num(d, key):
        v = (d or {}).get(key)
        return float(v) if isinstance(v, (int, float)) else None

    val, test, overall = windows.get("val") or {}, windows.get("test") or {}, windows.get("overall") or {}
    nv = int(val.get("n_weeks") or 0)
    nt = int(test.get("n_weeks") or 0)
    vs, ts = num(val, "ret_sum"), num(test, "ret_sum")

    weekly = None
    if (nv + nt) > 0 and (vs is not None or ts is not None):
        weekly = ((vs or 0.0) + (ts or 0.0)) / (nv + nt)
    cumulative = num(overall, "ret_sum")

    if weekly is None and cumulative is None:
        return None, None, "unavailable"
    return weekly, cumulative, "windows: OOS(val+test) per-week mean + overall.ret_sum"


def config_aggregate_from_ledger(run_id: str, ledger_rows: list[dict]
                                 ) -> tuple[float | None, bool, str | None]:
    """The seed's CONFIG 4-seed-mean val return + DQ pass, from `experiments/ledger.jsonl` rows.

    Matches the config by stripping the trailing ``-s<seed>`` suffix off `run_id` to get the
    `config_label`, then aggregates every ledger row for that config:
      * `config_seed_mean` = mean of the seeds' val `return` (the per-seed display return; this is
        the config's 4-seed-mean val, the anti-cherry-pick guard from [[Dashboard Leaderboard]]).
      * `dq_pass` = every matched seed is `legal_dd` True (the drawdown DQ gate held for all seeds).
    Returns `(config_seed_mean, dq_pass, config_label)`. If the config isn't found, returns
    `(None, False, <derived-label-or-None>)` GRACEFULLY (never crashes) — the entry still publishes
    with the guard fields empty.
    """
    config_label = strip_seed_suffix(run_id)
    rows = [r for r in ledger_rows if r.get("config_label") == config_label
            or strip_seed_suffix(str(r.get("run_id", ""))) == config_label]
    # prefer the val split if the ledger carries multiple (val is the gate split).
    val_rows = [r for r in rows if r.get("split") == "val"] or rows
    if not val_rows:
        return None, False, config_label

    rets = [float(r["return"]) for r in val_rows if isinstance(r.get("return"), (int, float))]
    config_seed_mean = (sum(rets) / len(rets)) if rets else None
    dq_pass = all(bool(r.get("legal_dd")) for r in val_rows)
    return config_seed_mean, dq_pass, config_label


def resolve_config_guard(run_id: str, ledger_rows: list[dict], *,
                         override_mean: float | None = None, override_dq: bool = False
                         ) -> tuple[float | None, bool, str | None]:
    """The config guard `(config_seed_mean, dq_pass, config_label)`, from an OVERRIDE or the ledger.

    The ledger is LAPTOP-authoritative (the rl_loop `record()` appends it laptop-side) but this
    publish runs on the DESKTOP (creds), whose committed ledger is stale and lacks the recent rows.
    So the caller — the laptop, or the rl_loop verdict hook — computes the guard from the FRESH
    ledger and passes it via `override_mean`/`override_dq`; only when no override is given do we read
    the local ledger (`config_aggregate_from_ledger`). PURE / I/O-free apart from that fallback.
    """
    if override_mean is not None:
        return override_mean, bool(override_dq), strip_seed_suffix(run_id)
    return config_aggregate_from_ledger(run_id, ledger_rows)


def strip_seed_suffix(run_id: str) -> str:
    """``ppo-…-ef0af8f-s3`` -> ``ppo-…-ef0af8f`` (drop a trailing ``-s<digits>``)."""
    import re
    return re.sub(r"-s\d+$", "", run_id)


def build_entry(run_id: str, weekly_score: float | None, cumulative_score: float | None,
                score_source: str, config_seed_mean: float | None, dq_pass: bool, windows: dict,
                generated: str | None = None) -> dict:
    """Assemble one leaderboard entry (the [[Dashboard Leaderboard]] §contract schema).

    `rank` is intentionally OMITTED here — `update_leaderboard` assigns it once the entry's final
    position is known. PURE / I/O-free.
    """
    return {
        "run_id": run_id,
        "weekly_score": weekly_score,
        "cumulative_score": cumulative_score,
        "score_source": score_source,
        "config_seed_mean": config_seed_mean,
        "dq_pass": dq_pass,
        "windows": windows,
        "trades_path": f"{run_id}/simulated_trades.json",
        "generated": generated or datetime.now(timezone.utc).isoformat(),
    }


# ============================== I/O + PUBLISH (desktop; needs creds) ===============================

def _load_ledger(path: str) -> list[dict]:
    """Parse the JSONL ledger into a list of dicts; missing file -> [] (graceful)."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _has_s3_creds() -> bool:
    """Best-effort: are AWS-style creds present so an `s3://…` read could actually succeed?

    Only used to decide the READ source on a `--no-publish` dry-run: with no creds we read the
    public CDN over HTTPS (laptop) instead of trying — and silently failing — an s3 GET.
    """
    return bool(os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE")
                or os.environ.get("AWS_SHARED_CREDENTIALS_FILE"))


def _fetch_json(uri: str):
    """Read a published JSON object; None if absent/unreadable/unparseable.

    `https://`/`http://` -> a plain read-only GET (the laptop confirmation path against the CDN, no
    creds; a 403/404 on a missing key -> None). Anything else (`s3://…` / a local path) goes through
    `remote_train.get_bytes` (the desktop publish path). Routing on the scheme keeps the laptop
    dry-run honest while reusing the object store for the real publish.
    """
    if uri.startswith(("http://", "https://")):
        import urllib.error
        import urllib.request
        try:
            with urllib.request.urlopen(uri, timeout=30) as resp:
                raw = resp.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            return None
    else:
        from remote_train import get_bytes
        raw = get_bytes(uri)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="Maintain the rolling top-3 simulated_leaderboard.json.")
    p.add_argument("--run-id", required=True)
    p.add_argument("--k", type=int, default=DEFAULT_K, help="leaderboard size (default 3)")
    p.add_argument("--ledger", default=DEFAULT_LEDGER, help="path to experiments/ledger.jsonl")
    p.add_argument("--no-publish", action="store_true",
                   help="compute + print the entry/eviction, do NOT PUT or invalidate (laptop-safe)")
    p.add_argument("--config-seed-mean", type=float, default=None,
                   help="OVERRIDE the ledger lookup for the config guard. The ledger is laptop-"
                        "authoritative but this publish runs on the desktop (stale ledger), so the "
                        "laptop/rl_loop computes the guard from the fresh ledger and passes it here.")
    p.add_argument("--dq-pass", action="store_true",
                   help="with --config-seed-mean, sets the entry's dq_pass True (omit => False).")
    args = p.parse_args()

    from trader import config
    from trader.report.apentic import MANIFEST_CACHE_CONTROL   # short cache + CF-invalidate, like the manifest
    config.load_dotenv()

    target = config.get("APENTIC_PUBLISH_TARGET")
    cf = config.get("APENTIC_CLOUDFRONT_DIST_ID")
    if not args.no_publish and not target:
        raise SystemExit("APENTIC_PUBLISH_TARGET unset — run on the desktop with creds, or --no-publish")

    from remote_train import invalidate_cloudfront, join, put_bytes

    # WRITES always go to the publish target (s3 + creds, desktop). READS prefer the public CDN on a
    # laptop --no-publish dry-run (no creds needed) and otherwise come from the target. `read_uri`
    # picks the source; `target_uri` is always the publish key.
    def target_uri(key: str) -> str:
        return join(target, key)

    def read_uri(key: str) -> str:
        if target and not (args.no_publish and not _has_s3_creds()):
            return join(target, key)
        return f"{CDN_BASE}/{key}"

    # --- 1. the two scores, from the seed's already-published simulated_trades.json meta.windows ---
    trades = _fetch_json(read_uri(f"{args.run_id}/simulated_trades.json"))
    if trades is None:
        raise SystemExit(f"{args.run_id}/simulated_trades.json not published yet — run "
                         f"simulate_weekly first (its meta.windows feeds the leaderboard).")
    windows = (trades.get("meta") or {}).get("windows")
    if not windows:
        raise SystemExit(f"{args.run_id}/simulated_trades.json has no meta.windows — "
                         f"re-run simulate_weekly (Phase 1).")
    weekly_score, cumulative_score, source = resolve_scores(windows)
    if weekly_score is None:
        print(f"[leaderboard] WARNING: no OOS weekly score in meta.windows ({source}); "
              f"entry will sort LAST. Inspect {args.run_id}/simulated_trades.json meta.windows.")

    # --- 2. config 4-seed-mean + dq_pass — the anti-cherry-pick guard (override or ledger) --------
    config_seed_mean, dq_pass, config_label = resolve_config_guard(
        args.run_id, _load_ledger(args.ledger),
        override_mean=args.config_seed_mean, override_dq=args.dq_pass)
    if args.config_seed_mean is not None:
        print(f"[leaderboard] config guard from args: config_seed_mean={config_seed_mean} dq_pass={dq_pass}")
    elif config_seed_mean is None:
        print(f"[leaderboard] note: config '{config_label}' not in {args.ledger} — config_seed_mean=null, "
              f"dq_pass=False. Pass --config-seed-mean (computed laptop-side) for the guard.")

    # --- 3. build the entry -----------------------------------------------------------------------
    entry = build_entry(args.run_id, weekly_score, cumulative_score, source,
                        config_seed_mean, dq_pass, windows)
    print(f"[leaderboard] entry: run_id={entry['run_id']} weekly_score={weekly_score} "
          f"cumulative_score={cumulative_score} config_seed_mean={config_seed_mean} dq_pass={dq_pass}")

    # --- 4. upsert into the current top-k ---------------------------------------------------------
    current = _fetch_json(read_uri(LEADERBOARD_KEY)) or []
    if not isinstance(current, list):
        print(f"[leaderboard] WARNING: existing {LEADERBOARD_KEY} is not a list; starting fresh.")
        current = []
    new_list, evicted = update_leaderboard(current, entry, k=args.k)

    made_cut = any(e["run_id"] == args.run_id for e in new_list)
    print(f"[leaderboard] top-{args.k} (by weekly_score): " + ", ".join(
        f"#{e['rank']} {e['run_id']} (wk={e['weekly_score']})" for e in new_list))
    if not made_cut:
        print(f"[leaderboard] {args.run_id} did NOT make the top-{args.k} (weekly_score={weekly_score}); "
              f"its trades file will NOT be listed.")
    if evicted:
        print(f"[leaderboard] EVICTED (de-listed): {evicted} — file(s) orphaned in S3 (no byte-delete).")

    # The CHAMPION is, by definition, the rank-1 entry (the best weekly_score) — the model to deploy.
    champion = new_list[0] if new_list else None
    if champion:
        print(f"[leaderboard] champion (#1) = {champion['run_id']} (weekly_score={champion['weekly_score']})")

    # --- 5. publish -------------------------------------------------------------------------------
    if args.no_publish:
        print("[leaderboard] --no-publish: not writing. Computed leaderboard JSON:")
        print(json.dumps(new_list, indent=2))
        return

    lb_bytes = json.dumps(new_list, separators=(",", ":")).encode("utf-8")
    put_bytes(target_uri(LEADERBOARD_KEY), lb_bytes,
              content_type="application/json", cache_control=MANIFEST_CACHE_CONTROL)
    invalidations = [f"/{LEADERBOARD_KEY}"]

    # The champion = the rank-1 entry, published as its own small artifact so the dashboard /
    # deployment reads "the model to deploy" directly (no re-ranking). Auto-tracks #1 every publish.
    if champion:
        put_bytes(target_uri(CHAMPION_KEY),
                  json.dumps(champion, separators=(",", ":")).encode("utf-8"),
                  content_type="application/json", cache_control=MANIFEST_CACHE_CONTROL)
        invalidations.append(f"/{CHAMPION_KEY}")
        print(f"[leaderboard] champion published -> {target_uri(CHAMPION_KEY)} ({champion['run_id']})")
    if evicted:
        orphans = _fetch_json(read_uri(ORPHANS_KEY)) or []
        if not isinstance(orphans, list):
            orphans = []
        now = datetime.now(timezone.utc).isoformat()
        orphans.extend({"run_id": rid, "path": f"{rid}/simulated_trades.json", "evicted_at": now}
                       for rid in evicted)
        put_bytes(target_uri(ORPHANS_KEY), json.dumps(orphans, separators=(",", ":")).encode("utf-8"),
                  content_type="application/json", cache_control=MANIFEST_CACHE_CONTROL)
        invalidations.append(f"/{ORPHANS_KEY}")
        print(f"[leaderboard] appended {len(evicted)} orphan(s) -> {ORPHANS_KEY}")

    if cf:
        invalidate_cloudfront(cf, invalidations)
    print(f"[leaderboard] published -> {target_uri(LEADERBOARD_KEY)} (top-{args.k}); "
          f"invalidated {invalidations}")


if __name__ == "__main__":
    main()
