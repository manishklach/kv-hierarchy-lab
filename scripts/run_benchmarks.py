"""Run a small policy benchmark matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kv_hierarchy_lab.bench.report import make_summary_table, write_csv, write_json
from kv_hierarchy_lab.bench.runner import run_benchmark
from kv_hierarchy_lab.bench.scenarios import example_scenarios
from kv_hierarchy_lab.policies import (
    CostAwarePolicy,
    HeavyHitterPolicy,
    LRUPolicy,
    PredictivePrefetchPolicy,
    RegretAwarePolicy,
    WindowedRecencyPolicy,
)
from kv_hierarchy_lab.utils.io import ensure_dir
from kv_hierarchy_lab.workloads import (
    generate_adversarial_burst,
    generate_chat_continuation,
    generate_long_tail_mix,
    generate_mixed_locality,
    generate_periodic_reuse,
    generate_prefetch_friendly,
    generate_rag_burst,
)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/default_sweep"))
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    workload_factories = [
        generate_chat_continuation,
        generate_rag_burst,
        generate_periodic_reuse,
        generate_long_tail_mix,
        generate_prefetch_friendly,
        generate_adversarial_burst,
        generate_mixed_locality,
    ]
    policy_factories = [
        LRUPolicy,
        WindowedRecencyPolicy,
        HeavyHitterPolicy,
        CostAwarePolicy,
        PredictivePrefetchPolicy,
        RegretAwarePolicy,
    ]

    results = []
    for scenario in example_scenarios():
        for workload_factory in workload_factories:
            workload = workload_factory(quantization_scheme=scenario.quantization_scheme)
            for factory in policy_factories:
                results.append(run_benchmark(scenario, workload, factory()))

    json_path = args.output_dir / "results.json"
    csv_path = args.output_dir / "results.csv"
    metadata_path = args.output_dir / "run_metadata.json"
    write_json(results, json_path)
    write_csv(results, csv_path)
    metadata = {
        "command": f"python scripts/run_benchmarks.py --output-dir {args.output_dir.as_posix()}",
        "policies": [factory().name for factory in policy_factories],
        "workloads": [factory().name for factory in workload_factories],
        "scenarios": [scenario.name for scenario in example_scenarios()],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(make_summary_table(results))
    print(f"\nWrote {json_path}, {csv_path}, and {metadata_path}")


if __name__ == "__main__":
    main()
