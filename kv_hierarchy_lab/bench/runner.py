"""Benchmark runner."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from kv_hierarchy_lab.bench.scenarios import Scenario
from kv_hierarchy_lab.policies.base import BasePolicy
from kv_hierarchy_lab.simulator.engine import SimulationEngine
from kv_hierarchy_lab.simulator.tier import MemoryTier
from kv_hierarchy_lab.workloads.synthetic import SyntheticWorkload


@dataclass(slots=True)
class BenchmarkResult:
    """Result row for one policy-workload-scenario run."""

    scenario: str
    workload: str
    policy: str
    quantization_scheme: str
    prefetch_enabled: bool
    metrics: dict[str, int | float | dict[str, int | float]]


def _instantiate_tiers(scenario: Scenario) -> list[MemoryTier]:
    return [
        MemoryTier(
            name=cfg.name,
            capacity_bytes=cfg.capacity_bytes,
            read_latency_ms=cfg.read_latency_ms,
            write_latency_ms=cfg.write_latency_ms,
            bandwidth_bytes_per_ms=cfg.bandwidth_bytes_per_ms,
            concurrency_limit=cfg.concurrency_limit,
        )
        for cfg in scenario.tier_configs
    ]


def run_benchmark(
    scenario: Scenario,
    workload: SyntheticWorkload,
    policy: BasePolicy,
) -> BenchmarkResult:
    """Runs one benchmark configuration."""
    tiers = _instantiate_tiers(scenario)
    engine = SimulationEngine(
        tiers=tiers,
        policy=policy,
        pages=deepcopy(workload.pages),
        prefetch_enabled=scenario.prefetch_enabled,
    )
    result = engine.run(workload.accesses)
    return BenchmarkResult(
        scenario=scenario.name,
        workload=workload.name,
        policy=policy.name,
        quantization_scheme=scenario.quantization_scheme,
        prefetch_enabled=scenario.prefetch_enabled,
        metrics=result.metrics,
    )
