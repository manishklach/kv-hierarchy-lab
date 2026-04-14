"""Predefined benchmark scenarios."""

from dataclasses import dataclass

from kv_hierarchy_lab.config import TierConfig


@dataclass(slots=True)
class Scenario:
    """Benchmark scenario definition."""

    name: str
    tier_configs: list[TierConfig]
    quantization_scheme: str
    prefetch_enabled: bool
    notes: str = ""


def default_tier_configs() -> list[TierConfig]:
    """Returns a small default hierarchy."""
    return [
        TierConfig("tier0_gpu_fast", capacity_bytes=128_000, read_latency_ms=0.02, write_latency_ms=0.02, bandwidth_bytes_per_ms=80_000),
        TierConfig("tier1_gpu_overflow", capacity_bytes=256_000, read_latency_ms=0.06, write_latency_ms=0.06, bandwidth_bytes_per_ms=40_000),
        TierConfig("tier2_host_ram", capacity_bytes=1_000_000, read_latency_ms=0.4, write_latency_ms=0.35, bandwidth_bytes_per_ms=8_000),
        TierConfig("tier3_nvme_like", capacity_bytes=8_000_000, read_latency_ms=2.5, write_latency_ms=2.0, bandwidth_bytes_per_ms=2_000),
    ]


def example_scenarios() -> list[Scenario]:
    """Returns the default benchmark sweep used by the repo."""
    base = default_tier_configs()
    smaller = [
        TierConfig(cfg.name, max(1, cfg.capacity_bytes // 2), cfg.read_latency_ms, cfg.write_latency_ms, cfg.bandwidth_bytes_per_ms, cfg.concurrency_limit)
        for cfg in base
    ]
    constrained = [
        TierConfig(cfg.name, max(1, cfg.capacity_bytes // 4), cfg.read_latency_ms, cfg.write_latency_ms, cfg.bandwidth_bytes_per_ms, cfg.concurrency_limit)
        for cfg in base
    ]
    return [
        Scenario("small_fp16_prefetch", smaller, "fp16", True, notes="Moderate pressure with prefetch enabled."),
        Scenario("small_fp16_no_prefetch", smaller, "fp16", False, notes="Moderate pressure without prefetch."),
        Scenario("small_int4_no_prefetch", smaller, "int4", False, notes="Footprint reduction without prefetch."),
        Scenario("base_fp8_prefetch", base, "fp8", True, notes="Larger residency budget with lighter quantization."),
        Scenario("constrained_fp16_prefetch", constrained, "fp16", True, notes="Aggressive fast-tier pressure."),
    ]
