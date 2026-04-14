"""Configuration objects for simulator and benchmark scenarios."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class TierConfig:
    """Defines a memory tier in the simulated hierarchy."""

    name: str
    capacity_bytes: int
    read_latency_ms: float
    write_latency_ms: float
    bandwidth_bytes_per_ms: float
    concurrency_limit: int | None = None


@dataclass(slots=True)
class BenchmarkConfig:
    """Top-level benchmark configuration."""

    name: str
    random_seed: int = 7
    prefetch_enabled: bool = True
    max_prefetch_candidates: int = 2
    notes: dict[str, str] = field(default_factory=dict)
