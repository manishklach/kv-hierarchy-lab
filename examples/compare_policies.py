"""Compare baseline policies on one workload."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kv_hierarchy_lab.bench.report import make_summary_table
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
from kv_hierarchy_lab.workloads import generate_adversarial_burst


def main() -> None:
    """CLI entrypoint."""
    scenario = example_scenarios()[0]
    workload = generate_adversarial_burst(quantization_scheme=scenario.quantization_scheme)
    policies = [
        LRUPolicy(),
        WindowedRecencyPolicy(),
        HeavyHitterPolicy(),
        CostAwarePolicy(),
        PredictivePrefetchPolicy(),
        RegretAwarePolicy(),
    ]
    results = [run_benchmark(scenario, workload, policy) for policy in policies]
    print(make_summary_table(results))


if __name__ == "__main__":
    main()
