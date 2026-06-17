# Dashboard Leaderboard — top-3 seeds + 3-window capture

Auto-publish the best seed of each completed sweep to a rolling **top-3 leaderboard** for the "Simulated Trades" dashboard, with each seed's **train/val/test window breakdown** captured for the overview. Spec'd 2026-06-17 (decisions locked with the user). Companion to [[Apentic Data Contract]] (the JSON contract), [[Simulated Market]] (the dashboard), and [[Remote Capabilities]] / [[MCP Server]] (the loop integration). Motivated by the ef-s2-vs-wsi-s3 reconciliation: ranking by the cumulative got the order *backwards*; ranking by val + showing the 3 windows gets it right and exposes a lucky/middling seed of a refuted config.

## Locked decisions
- **Rank by VAL cold-weekly** — the gate's honest bar (edge over the rung-0 RULE on the cold-weekly val split), the one that correctly put ef-s2 above wsi-s3. NOT the 28-week cumulative (in-sample-flattered) and NOT the frozen-test (single-seed noise).
- **Eviction = de-list** — when a new seed bumps the old #3, it's removed from the leaderboard index so the live dashboard stops loading it. The publish key can PUT but **not byte-DELETE** ([[apentic-publisher-no-delete]]), so the evicted ~4.7MB `simulated_trades.json` stays orphaned in S3 until a manual cleanup with a delete-capable credential.
- **Top-3 GLOBAL** — the 3 best individual seeds across all configs/iterations (not per-config).

## The ranking score (and the anti-cherry-pick guard)
The single ef-s2/wsi-s3 lesson is baked in so a lucky seed of a refuted config can't top the board unexamined. Each leaderboard entry carries:
- `val_score` — the seed's **val cold-weekly edge over rung-0** (`margin_vs_rung0`, from the seed's gate verdict in its published `metrics.json`). **This is the ranker.**
- `config_seed_mean` + `dq_pass` — the seed's *config's* 4-seed-mean val + whether the config passed the gate (from the loop history / `experiments/ledger.jsonl`). So a seed like wsi-s3 shows "best seed of a −2.75% config" rather than masquerading as an edge.
- `windows` — the 3-window split (below). The overview renders all three so train-loading is visible.

> The window returns are the model's **RAW per-week PnL** (for display); the `val_score` is the **edge over the rule** (the actual bar). Keep them distinct in the UI — "+25% val" is the model's return, not "+25% better than the rule."

## The 3-window capture (`simulate_weekly.py`)
`simulate_weekly` already emits all 28 weeks; add split labeling + a per-window summary in `meta`:
- Label each `week.start` by split using `train_rl.time_split` boundaries (train_end / val_end — the same split the gate, the capacity probe, and the reconciliations use). Add `"split": "train"|"val"|"test"` to each week.
- Add `meta.windows = {train|val|test: {ret_sum, ret_mean, worst_week_dd, win_rate, n_weeks}}` (+ an `overall` block). DD uses the **scale-free relative-move** mark (candle closes and the env `_px` position prices are on different per-asset scales — a naive dollar mark produces fantasy 100–350% swings; documented in [[Simulated Market]]).

## Published JSON contract
- **`simulated_trades.json`** (per seed, ~4.7 MB) — unchanged structure + the new `meta.windows` + per-week `split`. Only the top-3 seeds keep this file *listed*.
- **`simulated_leaderboard.json`** (NEW, small — the overview source): ordered top-3, each entry:
  ```
  {run_id, rank, val_score, config_seed_mean, dq_pass,
   windows: {train,val,test,overall: {ret_sum,ret_mean,worst_week_dd,win_rate,n_weeks}},
   trades_path: "<run-id>/simulated_trades.json", generated}
  ```
  The dashboard reads this for the overview (no 4.7MB fetch) and loads a `trades_path` only when a seed is selected.
- **`simulated_models.json`** (existing index): keep, but the selector is now driven by the leaderboard's 3 entries (de-listed seeds drop out).

## Eviction mechanics
`publish_leaderboard(run_id)`: (1) run `simulate_weekly` on the seed → trades + `windows`; (2) read its `val_score` from `metrics.json`; (3) load the current `simulated_leaderboard.json`; (4) insert, re-sort by `val_score` desc, truncate to 3; (5) if the new seed made the cut, PUT its `simulated_trades.json` + write the updated leaderboard (the evicted #3 simply isn't in the list → de-listed); (6) CloudFront-invalidate the leaderboard + the new file. **No byte-delete** — log the evicted run-id's orphaned path (optionally to an `orphans.json`) for a later manual purge. **Served/loaded footprint stays bounded at 3 × ~4.7MB; S3 storage grows with orphans** (the size goal is met for the *live dashboard*, not for raw S3 storage).

## Integration (auto-run post-training)
In the `rl_loop` **verdict** phase (desktop — torch + the policy + publish creds, where sweeps already self-publish): after a sweep grades, pick its **best seed by `val_score`** and call `publish_leaderboard(best_seed)`. A standalone `scripts/publish_leaderboard.py --run-id <best-seed>` does the work (also runnable by hand). Drive via the CLI, never the stale MCP ([[rl-loop-drive-via-cli-not-mcp]]); follow the [[Remote Capabilities]] runbook (PowerShell ssh, tiny output).

## Build phasing
1. **`simulate_weekly` 3-window** — split labeling + `meta.windows` + the scale-free DD; a test on a known bundle. (Self-contained; no leaderboard yet.)
2. **`publish_leaderboard.py`** — the ranking read (`val_score` from `metrics.json`), the top-3 insert/sort/truncate, the de-list write, the orphan log; tests on a synthetic leaderboard (insert, tie, eviction-de-list).
3. **Loop integration** — the verdict-phase hook to run it on the sweep's best seed.
4. **Seed the board** — publish the current gate-best (`wkw-s3`) as entry #1, then re-publish ef-s2 / wsi-s3 through the new path so the overview shows their 3-window splits + the config-mean guard (and the board ranks them correctly).

## Risks / notes
- **Orphan storage** (no byte-delete) — the live dashboard is bounded to 3, but S3 accumulates evicted files; needs a periodic manual purge if storage cost matters.
- **Single-seed display** — the dashboard shows individual seeds (you visualize one policy's trades); the `config_seed_mean` + `dq_pass` fields are the guard against reading a lucky seed as a config-level edge.
- **`val_score` source** — depends on the seed's `metrics.json` carrying the gate verdict; confirm the field name (`margin_vs_rung0` / `regimes.val`) when wiring Phase 2.
