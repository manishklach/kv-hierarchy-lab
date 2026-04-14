"""Run one small simulation and print metrics."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kv_hierarchy_lab.bench.scenarios import example_scenarios
from kv_hierarchy_lab.policies import LRUPolicy
from kv_hierarchy_lab.simulator.engine import SimulationEngine
from kv_hierarchy_lab.simulator.tier import MemoryTier
from kv_hierarchy_lab.workloads import generate_chat_continuation


def main() -> None:
    """CLI entrypoint."""
    scenario = example_scenarios()[0]
    workload = generate_chat_continuation(quantization_scheme=scenario.quantization_scheme)
    tiers = [
        MemoryTier(
            name=cfg.name,
            capacity_bytes=cfg.capacity_bytes,
            read_latency_ms=cfg.read_latency_ms,
            write_latency_ms=cfg.write_latency_ms,
            bandwidth_bytes_per_ms=cfg.bandwidth_bytes_per_ms,
        )
        for cfg in scenario.tier_configs
    ]
    engine = SimulationEngine(
        tiers=tiers,
        policy=LRUPolicy(),
        pages=workload.pages,
        prefetch_enabled=scenario.prefetch_enabled,
    )
    result = engine.run(workload.accesses)
    print(result.metrics)


if __name__ == "__main__":
    main()
