"""Policy behavior tests."""

from kv_hierarchy_lab.policies import HeavyHitterPolicy, LRUPolicy, PredictivePrefetchPolicy
from kv_hierarchy_lab.policies.base import PolicyContext
from kv_hierarchy_lab.simulator.page import KVPage
from kv_hierarchy_lab.simulator.trace import TraceAccess


def _page(page_id: str) -> KVPage:
    return KVPage(
        page_id=page_id,
        layer=0,
        token_start=0,
        token_end=16,
        head_group=0,
        size_bytes=1024,
        quantization_scheme="fp16",
    )


def _context(step: int) -> PolicyContext:
    return PolicyContext(
        step=step,
        access=TraceAccess(step=step, page_id="page-0", sequence_id="seq-0"),
        resident_pages={},
        tier_free_bytes={},
        pages={},
    )


def test_lru_evicts_oldest_page() -> None:
    """LRU should pick the oldest page."""
    policy = LRUPolicy()
    a, b = _page("a"), _page("b")
    policy.on_access(_context(1), a, "tier0")
    policy.on_access(_context(2), b, "tier0")
    victim = policy.select_eviction_candidate(_context(3), "tier0", [a, b], incoming_page=_page("c"))
    assert victim.page_id == "a"


def test_heavy_hitter_protects_frequent_page() -> None:
    """Frequently accessed pages should survive eviction more often."""
    policy = HeavyHitterPolicy()
    a, b = _page("a"), _page("b")
    for step in range(1, 4):
        policy.on_access(_context(step), a, "tier0")
    policy.on_access(_context(4), b, "tier0")
    victim = policy.select_eviction_candidate(_context(5), "tier0", [a, b], incoming_page=_page("c"))
    assert victim.page_id == "b"


def test_predictive_policy_learns_simple_transition() -> None:
    """A repeated transition should trigger a prefetch candidate."""
    policy = PredictivePrefetchPolicy(min_confidence=2)
    a = _page("a")
    b = _page("b")
    policy.on_access(_context(1), a, "tier0")
    policy.on_access(_context(2), b, "tier0")
    policy.on_access(_context(3), a, "tier0")
    policy.on_access(_context(4), b, "tier0")
    assert policy.maybe_prefetch(_context(5), a) == ["b"]
