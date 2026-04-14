"""Metrics collection for the simulator."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class MetricsCollector:
    """Aggregates simulator metrics during execution."""

    total_accesses: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    total_latency_ms: float = 0.0
    total_bytes_moved: int = 0
    demand_bytes_moved: int = 0
    prefetch_bytes_moved: int = 0
    prefetch_requests: int = 0
    useful_prefetches: int = 0
    wasted_prefetches: int = 0
    tier_hits: dict[str, int] = field(default_factory=dict)
    tier_peak_bytes: dict[str, int] = field(default_factory=dict)

    def record_hit(self, tier_name: str, latency_ms: float) -> None:
        """Records a hit in a given tier."""
        self.total_accesses += 1
        self.total_latency_ms += latency_ms
        self.tier_hits[tier_name] = self.tier_hits.get(tier_name, 0) + 1

    def record_miss(self) -> None:
        """Records a miss."""
        self.total_accesses += 1
        self.miss_count += 1

    def record_transfer(self, bytes_moved: int, latency_ms: float, is_prefetch: bool) -> None:
        """Records a data transfer."""
        self.total_latency_ms += latency_ms
        self.total_bytes_moved += bytes_moved
        if is_prefetch:
            self.prefetch_bytes_moved += bytes_moved
        else:
            self.demand_bytes_moved += bytes_moved

    def record_prefetch(self, useful: bool | None = None) -> None:
        """Records a prefetch request or outcome."""
        if useful is None:
            self.prefetch_requests += 1
        elif useful is True:
            self.useful_prefetches += 1
        else:
            self.wasted_prefetches += 1

    def to_dict(self) -> dict[str, float | int | dict[str, int | float]]:
        """Serializes metrics for reporting."""
        avg_latency = self.total_latency_ms / self.total_accesses if self.total_accesses else 0.0
        prefetch_usefulness = (
            self.useful_prefetches / self.prefetch_requests if self.prefetch_requests else 0.0
        )
        tier_hit_rate = {
            tier: hits / self.total_accesses if self.total_accesses else 0.0
            for tier, hits in self.tier_hits.items()
        }
        return {
            "accesses": self.total_accesses,
            "miss_count": self.miss_count,
            "eviction_count": self.eviction_count,
            "avg_latency_ms": avg_latency,
            "bytes_moved": self.total_bytes_moved,
            "demand_bytes_moved": self.demand_bytes_moved,
            "prefetch_bytes_moved": self.prefetch_bytes_moved,
            "prefetch_requests": self.prefetch_requests,
            "useful_prefetches": self.useful_prefetches,
            "wasted_prefetches": self.wasted_prefetches,
            "prefetch_usefulness": prefetch_usefulness,
            "tier_hit_rate": tier_hit_rate,
            "peak_footprint_bytes": self.tier_peak_bytes,
        }
