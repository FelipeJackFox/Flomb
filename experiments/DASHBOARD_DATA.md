# Dashboard backend contract

`build_dashboard(root=Path('runs'))` reads files only and returns `{runs: [...], generated_at: ISO UTC}`. It never starts, stops or changes a run. `registry(root)` maps relative IDs to directories. `resolve_run(root, run_id)` rejects IDs outside that registry, absolute paths, dot segments, aliases and symlinks escaping the root.

Registered: `curriculum-001`, direct directories whose config.mode is dagger/qrdqn/hybrid, and those modes one level under a suite directory. Historical migration directories are excluded.

Each run contains:

- `id`, `config`, `progress`, `state`, `completed`, `paused`, `control`, `applied`: original JSON objects or null. `applied` comes from control-applied.json.
- `control_supported`: config.mode is a supported mode AND capabilities.json.runtime_curriculum is exactly true. A mode alone does not advertise unsupported runtime control.
- `series`: mapping exact `size x size - mines` keys (e.g. `16x16-40`) to points.
- `overall_series`: all sizes combined, for training progress and throughput.
- `bucket_size`: episode window, 1 for short pilots, around 200 near 30k episodes; adjusted to at most 150 points per curve. `metric_episodes`: unique recorded episodes.
- `evaluations`: list of `{episode, kind: before|periodic|final, results}`. Duplicate periodic episodes use the last line. Before can contain nested random/initial comparisons. Final is read separately and is never inferred from training wins.

Points contain episode (last in bucket), size/mines (null in overall), episodes, policy_episodes, wins, policy_wins, automatic_wins, win_rate. **win_rate excludes automatic wins from numerator and denominator**, and is null when no policy episode exists. Training wins may involve teacher/exploration actions; they are not independent evaluation scores.

safe_fraction, return, episode_steps, loss_qr, loss_teacher, gradient_norm are means of available observations. Missing measurements remain null. training_seconds, cumulative steps, updates, rss are latest available values. `rss` is bytes and may represent reported peak RSS, not current resident memory. Old curriculum metrics use steps for per-episode clicks and total_clicks for cumulative; both are normalized. Modern nested loss.qr/loss.teacher are adapted.

teacher_fraction is action-weighted teacher_actions / teacher_observed_steps, with both counts exposed. Missing teacher information is null, not zero. Plotting throughput should use deltas of cumulative steps and training_seconds from overall_series.

`game_metrics` is null when absent, otherwise `{measured_episodes, counters, ratios}`. Counters are summed; derived per-episode ratios are excluded from sums. Every ratio is `{numerator, denominator, value}`, with null for unmeasured or zero-denominator cases. `controller_metrics` provides the same shape separately under policy, teacher and random. This allows an actor selector without mixing teacher performance into policy diagnostics. Ratios include logical errors, safe opportunities taken, exact risk coverage, excess risk, minimum-risk selection, reveal expansion, frontier/interior/border play, guess survival and the separate final-50/50 score.

Caching uses file mtime_ns and size plus an append-only per-file offset cache. Unchanged runs reuse aggregated output; progress/control changes do not reread metrics. When metrics append, only new complete JSONL lines are parsed; historical lines are retained in memory. Partial final lines wait for the next append. Replaced/truncated files reset their file cache. Cached data is copied before returning, so caller mutation cannot corrupt subsequent responses.

Tests: `.venv/bin/python -m unittest experiments.test_dashboard_data` (temporary fixtures only).

`bucket_end` is the common upper episode boundary for each window, identical across stratum and overall points. Use it for cross-series alignment; `episode` retains the last actual observed episode of that curve. `timing_metrics_seconds`, `epsilon`, and `teacher_beta` are means. Records explicitly marked `metrics_measured: false` contribute no game/controller diagnostic counters or score denominators. Their ordinary training episode outcomes remain in win statistics.

`size_series` maps 5x5/7x7/9x9/12x12/16x16 to correctly aggregated points across mine counts (`mines: null`), using raw episode records and summed diagnostic counters. Default curves can use these five sizes; their mine-density mixture changes over training. Exact-density comparison must use `series`. Old nested baseline evaluations become separate entries such as `before/random` and `before/brain`, with flat strata under results. Final evaluations sort after periodic evaluations at the same episode.
