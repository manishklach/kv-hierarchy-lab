# Regret-Aware Ablation Artifacts

Simulator artifacts from the regret-aware eviction policy ablation study.

**These are simulator results on synthetic traces, not runtime performance claims.**

## Contents

| File | Description |
|---|---|
| `raw_results.json` | Per-run results for every seed × scenario × workload × policy combination |
| `summary.csv` | Flattened per-run results |
| `aggregated_summary.csv` | Mean ± std across seeds, grouped by scenario × workload × policy |
| `run_metadata.json` | Sweep configuration: seeds, weights, horizons, workload parameters |
| `plots/` | Ablation visualisations with error bars |

## Regeneration

```bash
python scripts/run_regret_ablation.py --output-dir artifacts/regret_ablation
```

## Sweep Configuration

- **Seeds:** 3 independent seeds per ablation point (7, 42, 91)
- **Regret weights:** 1.0, 6.0, 12.0
- **Regret horizons:** 8, 24, 64
- **Scenarios:** medium (1/3 base capacity), constrained (1/6 base capacity)
- **Workloads:** chat_continuation, periodic_reuse, adversarial_burst
- **Baselines:** LRU, cost_aware

## Plots

- `latency_vs_horizon.png` — Latency as a function of regret horizon (weight = 6.0)
- `hit_rate_vs_weight.png` — Hit rate as a function of regret weight (horizon = 24)
- `tradeoff.png` — Latency vs bytes moved (adversarial_burst, constrained)
- `failure_case.png` — Neutral case where regret offers no advantage (chat_continuation)
