# Dashboard Leaderboard — top-3 seeds + 3-window capture

Auto-publish the best seed of each completed sweep to a rolling **top-3 leaderboard** for the "Simulated Trades" dashboard, with each seed's **train/val/test window breakdown** captured for the overview. Spec'd 2026-06-17 (decisions locked with the user). Companion to [[Apentic Data Contract]] (the JSON contract), [[Simulated Market]] (the dashboard), and [[Remote Capabilities]] / [[MCP Server]] (the loop integration). Motivated by the ef-s2-vs-wsi-s3 reconciliation: ranking by the 28-week cumulative got the order *backwards*; ranking by the held-out per-week PnL + showing the 3 windows gets it right and exposes a lucky/middling seed of a refuted config.

## Locked decisions
- **Rank by the model's OWN cold-weekly PnL, NOT vs the rung-0 rule** (revised 2026-06-17 — see §"Why not vs_rung0"). Each entry carries **two** scores, both straight from the model's own `meta.windows`:
  - **`weekly_score`** — the **PRIMARY rank**: mean per-week return over the OUT-OF-SAMPLE weeks (val + test) = `(val.ret_sum + test.ret_sum) / (val.n_weeks + test.n_weeks)`. This is "expected return in *any random week*" — and the live competition window (one week) is exactly a random week, so it's the **hackathon** predictor.
  - **`cumulative_score`** — a **secondary display stat**: `windows.overall.ret_sum`, the 28-week cumulative — the hold-across-weeks / **post-competition deployment** view the user weights for the long run. NOT the rank (in-sample-flattered by the train weeks) but kept visible.
- **Eviction = de-list** — when a new seed bumps the old #3, it's removed from the leaderboard index so the live dashboard stops loading it. The publish key can PUT but **not byte-DELETE** ([[apentic-publisher-no-delete]]), so the evicted ~4.7MB `simulated_trades.json` stays orphaned in S3 until a manual cleanup with a delete-capable credential.
- **Top-3 GLOBAL** — the 3 best individual seeds across all configs/iterations (not per-config).

## Why not vs_rung0 (the revision)
The earlier spec ranked by the gate's `weekly.edge_vs_rung0` (the cold-weekly paired edge over the rung-0 RULE). Pulling the real numbers killed that idea: the rung-0 rule is **not a stable yardstick**. It's the same high-variance, DQ-breaching rule that — on the recent **test-window bull** — rips **+19.8%/wk**, vs every risk-managed policy's +3–8%/wk. That single window drags *every* config's val+test `edge_vs_rung0` **negative** and ranks a **refuted** config (ef-s2, −0.0205) *above* the gate-best config's seed (wkw-s3, −0.0457). Comparing our own trained models to a reckless rule that happened to win a bull is actively misleading for model-vs-model. Since this leaderboard exists to compare *our iterations to each other*, we drop the rung-0 comparison here. (The rung-0 gate still governs the research decision in the loop — it just isn't the dashboard ranker.)

## The ranking score (and the anti-cherry-pick guard)
The ef-s2/wsi-s3 lesson is baked in so a lucky seed of a refuted config can't top the board unexamined. Each leaderboard entry carries:
- `weekly_score` + `cumulative_score` — the two model-own scores above. **`weekly_score` is the ranker.**
- `config_seed_mean` + `dq_pass` — the seed's *config's* 4-seed-mean val + whether the config passed the DQ gate (from the loop history / `experiments/ledger.jsonl`). The guard rides ALONGSIDE the score, never baked into it: a lucky seed (e.g. ef-s2 tops `weekly_score` at +5.8%/wk while its config's 4-seed mean is +0.4%) is exposed, not hidden. `score_source` records how the scores resolved.

> The window returns are the model's **RAW per-week PnL** (cold-weekly, fresh $10k/week, no compounding). `weekly_score` is the OOS subset of those; `cumulative_score` is their 28-week sum. Both are the model's own PnL — there is no longer a "vs the rule" number on the board.

## The 3-window capture (`simulate_weekly.py`)
`simulate_weekly` already emits all 28 weeks; Phase 1 added split labeling + a per-window summary in `meta`:
- Label each `week.start` by split using `train_rl.time_split` boundaries (train_end / val_end — the same split the gate, the capacity probe, and the reconciliations use). Each week carries `"split": "train"|"val"|"test"`.
- `meta.windows = {train|val|test|overall: {ret_sum, ret_mean, worst_week_dd, win_rate, n_weeks}}`. DD + return come from the env's EXACT per-bar equity (a naive dollar mark over differently-scaled per-asset prices produces fantasy 100–350% swings; documented in [[Simulated Market]]). `weekly_score`/`cumulative_score` are computed from this block — no metrics.json needed.

## Published JSON contract
- **`simulated_trades.json`** (per seed, ~4.7 MB) — unchanged structure + `meta.windows` + per-week `split`. Only the top-3 seeds keep this file *listed*.
- **`simulated_leaderboard.json`** (the overview source, small): ordered top-3, each entry:
  ```
  {run_id, rank, weekly_score, cumulative_score, score_source, config_seed_mean, dq_pass,
   windows: {train,val,test,overall: {ret_sum,ret_mean,worst_week_dd,win_rate,n_weeks}},
   trades_path: "<run-id>/simulated_trades.json", generated}
  ```
  The dashboard reads this for the overview (no 4.7MB fetch) and loads a `trades_path` only when a seed is selected. Render `weekly_score` as the rank/headline and `cumulative_score` as a secondary stat; `config_seed_mean`/`dq_pass` as the guard.
- **`simulated_models.json`** (existing index): keep, but the selector is now driven by the leaderboard's 3 entries (de-listed seeds drop out).

## Eviction mechanics
`publish_leaderboard(run_id)`: (1) read the seed's already-published `simulated_trades.json` `meta.windows`; (2) compute `weekly_score`/`cumulative_score` from it (`resolve_scores`) + the config guard from the ledger; (3) load the current `simulated_leaderboard.json`; (4) upsert, re-sort by `weekly_score` desc, truncate to 3; (5) if the new seed made the cut, write the updated leaderboard (the evicted #3 simply isn't in the list → de-listed); (6) CloudFront-invalidate the leaderboard + `orphans.json`. **No byte-delete** — log the evicted run-id's orphaned path to `orphans.json` for a later manual purge. **Served/loaded footprint stays bounded at 3 × ~4.7MB; S3 storage grows with orphans.**

## Integration (auto-run post-training) — BUILT
In the `rl_loop` **verdict** phase (driver, laptop-side): after a sweep grades, the driver picks the **best seed = max val `return`** from the verdict's `per_seed` rows (a reliable proxy for the best `weekly_score` — wkw-s3 was both; running `simulate_weekly` on all seeds to rank by `weekly_score` directly would cost 4× the eval for negligible gain), computes the config guard (seed-mean + DQ ≤ 30%) **from the verdict**, and fire-and-forgets `simulate_weekly`→`publish_leaderboard` on the **desktop** (torch + publish creds) via the pinned `run_ssh` (nohup, `< /dev/null`, a tiny launch-ack reply). Best-effort: a failure records `leaderboard_error`, never halts the loop. Drive via the CLI, never the stale MCP ([[rl-loop-drive-via-cli-not-mcp]]).

**The laptop/desktop ledger split (why the guard is passed, not read on the desktop):** `config_seed_mean`/`dq_pass` come from `experiments/ledger.jsonl`, which is **laptop-authoritative** (the loop's `record()` appends it laptop-side) — but the publish runs on the **desktop**, whose committed ledger is stale, so `config_aggregate_from_ledger` finds nothing there. The guard is therefore computed laptop-side (the driver uses the verdict; a manual re-seed uses the laptop ledger) and passed via `publish_leaderboard --config-seed-mean <x> [--dq-pass]` (the Phase-2 override). The desktop never needs a fresh ledger. NOTE the publish needs the APENTIC creds, which live on the **desktop** (the laptop `.env` has none) — contra the earlier [[desktop-shared-publish-from-laptop]] note, verified false here.

## Build phasing
1. **`simulate_weekly` 3-window** — split labeling + `meta.windows` + the scale-free DD; tested on a known bundle. **DONE + committed** (@822cac6, DD fix @d395dd8).
2. **`publish_leaderboard.py`** — the two-score resolver (`resolve_scores` from `meta.windows`), the top-3 upsert/sort/truncate by `weekly_score`, the de-list write, the orphan log. **DONE 2026-06-17** (`scripts/publish_leaderboard.py` + `tests/test_publish_leaderboard.py`, 25 tests; full suite 455 pass). Pure `update_leaderboard(current, new_entry, k=3) -> (new_list, evicted_run_ids)` is torch/IO-free; the publish (PUT/CloudFront) is gated behind the read. Originally built on `edge_vs_rung0`; **revised to the two model-own scores** (above) after the vs_rung0 finding. Read path confirmed end-to-end against the live CDN (`--no-publish`): wkw-s3 → `weekly_score`=+0.0324, `cumulative_score`=+0.930, `config_seed_mean`=+0.0453, dq_pass=True.
3. **Loop integration** — **DONE 2026-06-17** (`driver.py`): the verdict phase picks the best seed (max val `return` from the verdict `per_seed`) + the config guard (seed-mean + DQ ≤ 30%) FROM THE VERDICT, then fire-and-forgets `simulate_weekly`→`publish_leaderboard` on the desktop, best-effort (`leaderboard_error` on failure, never blocks). `_best_seed_and_guard` + `_leaderboard_publish_cmd` are pure/tested (6 driver tests).
4. **Seed the board** — **LIVE 2026-06-17**: wkw-s3 / ef-s2 / wsi-s3 published to `simulated_leaderboard.json`. Order by `weekly_score`: **#1 ef-s2 +5.77%/wk (lucky seed of a refuted config — the guard flags it), #2 wkw-s3 +3.24%/wk (robust config, DQ-protective), #3 wsi-s3 +3.00%/wk**. By `cumulative_score`: wsi-s3 +109% > wkw-s3 +93% > ef-s2 +91%. NB the first publish landed with `config_seed_mean`/`dq_pass` NULL (the ledger split above); the `--config-seed-mean` override fixes it — re-publish pending the new code reaching the desktop.

## Risks / notes
- **Orphan storage** (no byte-delete) — the live dashboard is bounded to 3, but S3 accumulates evicted files; needs a periodic manual purge if storage cost matters.
- **Single-seed display** — the dashboard shows individual seeds (you visualize one policy's trades); `config_seed_mean` + `dq_pass` are the guard against reading a lucky seed as a config-level edge. NOTE the current `weekly_score` rank IS seed-level, so a lucky seed (ef-s2) can sit at #1 with a weak `config_seed_mean` beside it — by design (rank = observed random-week performance; the guard supplies the robustness context). If a config-robust rank is ever wanted instead, sort by `config_seed_mean`.
- **Two audiences, two scores** — `weekly_score` answers the hackathon ("best in a random week"); `cumulative_score` answers long-run deployment. They can disagree on the winner (ef-s2 vs wsi-s3) — that disagreement is informative, not a bug.
