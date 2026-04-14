"""Simulator engine tests."""

from kv_hierarchy_lab.policies import LRUPolicy, PredictivePrefetchPolicy
from kv_hierarchy_lab.simulator.engine import SimulationEngine
from kv_hierarchy_lab.simulator.page import KVPage
from kv_hierarchy_lab.simulator.tier import MemoryTier
from kv_hierarchy_lab.simulator.trace import TraceAccess
from kv_hierarchy_lab.workloads import generate_chat_continuation


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


def _three_tiers() -> list[MemoryTier]:
    return [
        MemoryTier("tier0", capacity_bytes=1024, read_latency_ms=0.01, write_latency_ms=0.01, bandwidth_bytes_per_ms=10_000),
        MemoryTier("tier1", capacity_bytes=1024, read_latency_ms=0.2, write_latency_ms=0.2, bandwidth_bytes_per_ms=2_000),
        MemoryTier("tier2", capacity_bytes=4096, read_latency_ms=1.0, write_latency_ms=1.0, bandwidth_bytes_per_ms=1_000),
    ]


def _single_slot_tiers() -> list[MemoryTier]:
    return [
        MemoryTier("tier0", capacity_bytes=1024, read_latency_ms=0.01, write_latency_ms=0.01, bandwidth_bytes_per_ms=10_000),
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


def test_engine_supports_multi_tier_demotion_chain() -> None:
    """Pressure should demote pages through more than one tier when needed."""
    pages = {page_id: _make_page(page_id) for page_id in ["a", "b", "c", "d"]}
    accesses = [TraceAccess(step=idx, page_id=page_id, sequence_id="s") for idx, page_id in enumerate(["a", "b", "c", "d"])]
    result = SimulationEngine(_three_tiers(), LRUPolicy(), pages, prefetch_enabled=False).run(accesses)
    demotions = [event for event in result.events if event.event_type == "demote"]
    assert len(demotions) >= 2
    assert any(event.dst_tier == "tier2" for event in demotions)


def test_prefetch_usefulness_is_counted_when_page_is_used() -> None:
    """A prefetched page consumed before eviction should count as useful."""
    pages = {page_id: _make_page(page_id) for page_id in ["a", "b", "c"]}
    accesses = [
        TraceAccess(step=0, page_id="a", sequence_id="s"),
        TraceAccess(step=1, page_id="b", sequence_id="s"),
        TraceAccess(step=2, page_id="a", sequence_id="s"),
        TraceAccess(step=3, page_id="b", sequence_id="s"),
    ]
    result = SimulationEngine(_single_slot_tiers(), PredictivePrefetchPolicy(min_confidence=1), pages).run(accesses)
    assert result.metrics["prefetch_requests"] >= 1
    assert result.metrics["useful_prefetches"] >= 1


def test_wasted_prefetch_is_counted_when_page_is_never_used() -> None:
    """A prefetched page evicted or left unused should count as wasted."""
    pages = {page_id: _make_page(page_id) for page_id in ["a", "b", "c", "d"]}
    accesses = [
        TraceAccess(step=0, page_id="a", sequence_id="s"),
        TraceAccess(step=1, page_id="b", sequence_id="s"),
        TraceAccess(step=2, page_id="a", sequence_id="s"),
        TraceAccess(step=3, page_id="c", sequence_id="s"),
        TraceAccess(step=4, page_id="d", sequence_id="s"),
    ]
    result = SimulationEngine(_single_slot_tiers(), PredictivePrefetchPolicy(min_confidence=1), pages).run(accesses)
    assert result.metrics["prefetch_requests"] >= 1
    assert result.metrics["wasted_prefetches"] >= 1


def test_quantization_footprint_changes_residency_outcome() -> None:
    """Smaller quantized footprints should reduce misses under the same tier budget."""
    tiers = [
        MemoryTier("tier0", capacity_bytes=2048, read_latency_ms=0.01, write_latency_ms=0.01, bandwidth_bytes_per_ms=10_000),
        MemoryTier("tier1", capacity_bytes=8192, read_latency_ms=0.2, write_latency_ms=0.2, bandwidth_bytes_per_ms=2_000),
    ]
    fp16_pages = {
        "a": _make_page("a", size_bytes=1536),
        "b": _make_page("b", size_bytes=1536),
    }
    int4_pages = {
        "a": KVPage(page_id="a", layer=0, token_start=0, token_end=16, head_group=0, size_bytes=384, quantization_scheme="int4"),
        "b": KVPage(page_id="b", layer=0, token_start=0, token_end=16, head_group=0, size_bytes=384, quantization_scheme="int4"),
    }
    accesses = [TraceAccess(step=idx, page_id=page_id, sequence_id="s") for idx, page_id in enumerate(["a", "b", "a", "b"])]
    fp16_result = SimulationEngine(tiers, LRUPolicy(), fp16_pages, prefetch_enabled=False).run(accesses)
    int4_result = SimulationEngine(_tiers(), LRUPolicy(), int4_pages, prefetch_enabled=False).run(accesses)
    assert int4_result.metrics["miss_count"] <= fp16_result.metrics["miss_count"]
