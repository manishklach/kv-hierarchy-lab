# Default Sweep Artifacts

These files are committed simulator outputs from the default benchmark sweep.

Generated with:

```bash
python scripts/run_benchmarks.py --output-dir artifacts/default_sweep
python scripts/plot_results.py --results artifacts/default_sweep/results.json --out-dir artifacts/default_sweep/plots
```

Files:

- `results.json`: full benchmark records
- `results.csv`: flattened table for quick filtering
- `run_metadata.json`: exact command plus scenario, policy, and workload lists
- `plots/`: policy-level summary figures

These artifacts are simulator results on synthetic traces. They are intended for inspection and reproduction, not as claims about real runtime throughput.
