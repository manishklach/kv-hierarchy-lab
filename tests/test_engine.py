"""Simulator engine tests."""

from kv_hierarchy_lab.policies import LRUPolicy
from kv_hierarchy_lab.simulator.engine import SimulationEngine
from kv_hierarchy_lab.simulator.page import KVPage
from kv_hierarchy_lab.simulator.tier import MemoryTier
from kv_hierarchy_lab.simulator.trace import TraceAccess


def _make_page(page_id: str, size_bytes: int = 1024) -> KVPage:
    return KVPage(
        page_id=page_id,
        layer=0,
        token_start=0,
        token_end=16,
        head_group=0,
        size_bytes=size_bytes,
        quantization_scheme="fp16",
    )


def _tiers() -> list[MemoryTier]:
    return [
        MemoryTier("tier0", capacity_bytes=2048, read_latency_ms=0.01, write_latency_ms=0.01, bandwidth_bytes_per_ms=10_000),
        MemoryTier("tier1", capacity_bytes=8192, read_latency_ms=0.2, write_latency_ms=0.2, bandwidth_bytes_per_ms=2_000),
    ]


def test_tier_capacity_accounting() -> None:
    """Tier should update used bytes after page insert and removal."""
    tier = _tiers()[0]
    tier.add_page("a", 1024)
    assert tier.used_bytes == 1024
    tier.remove_page("a")
    assert tier.used_bytes == 0


def test_engine_records_hit_after_first_load() -> None:
    """Second access to the same page should hit the fast tier."""
    pages = {"a": _make_page("a")}
    accesses = [
        TraceAccess(step=0, page_id="a", sequence_id="s"),
        TraceAccess(step=1, page_id="a", sequence_id="s"),
    ]
    result = SimulationEngine(_tiers(), LRUPolicy(), pages, prefetch_enabled=False).run(accesses)
    assert result.metrics["miss_count"] == 1
    assert result.metrics["tier_hit_rate"]["tier0"] > 0


def test_engine_demotes_when_fast_tier_fills() -> None:
    """Accessing too many pages should trigger at least one eviction or demotion."""
    pages = {page_id: _make_page(page_id) for page_id in ["a", "b", "c"]}
    accesses = [TraceAccess(step=idx, page_id=page_id, sequence_id="s") for idx, page_id in enumerate(["a", "b", "c"])]
    result = SimulationEngine(_tiers(), LRUPolicy(), pages, prefetch_enabled=False).run(accesses)
    assert result.metrics["eviction_count"] >= 1
