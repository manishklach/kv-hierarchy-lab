"""Policy interfaces for residency, eviction, and prefetch decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from kv_hierarchy_lab.simulator.page import KVPage
from kv_hierarchy_lab.simulator.trace import TraceAccess


@dataclass(slots=True)
class PolicyContext:
    """Shared state presented to policies."""

    step: int
    access: TraceAccess
    resident_pages: dict[str, str]
    tier_free_bytes: dict[str, int]
    pages: dict[str, KVPage]
    access_history: list[str] = field(default_factory=list)


class BasePolicy(Protocol):
    """Policy protocol for eviction and optional prefetching."""

    name: str

    def on_access(self, context: PolicyContext, page: KVPage, hit_tier: str | None) -> None:
        """Updates policy state on every access."""

    def on_miss(self, context: PolicyContext, page: KVPage) -> None:
        """Updates policy state when an access misses the fast tier."""

    def on_insert(self, context: PolicyContext, page: KVPage, tier_name: str) -> None:
        """Updates policy state when a page is inserted into a tier."""

    def on_evict(self, context: PolicyContext, page: KVPage, tier_name: str) -> None:
        """Updates policy state when a page leaves a tier."""

    def on_promote(self, context: PolicyContext, page: KVPage, src_tier: str | None, dst_tier: str) -> None:
        """Updates policy state when a page is promoted between tiers."""

    def select_eviction_candidate(
        self,
        context: PolicyContext,
        tier_name: str,
        candidates: list[KVPage],
        incoming_page: KVPage,
    ) -> KVPage:
        """Selects a page to evict from the specified tier."""

    def maybe_prefetch(self, context: PolicyContext, page: KVPage) -> list[str]:
        """Returns candidate page ids to prefetch into the fast tier."""

    def score_page(self, context: PolicyContext, page: KVPage, tier_name: str) -> float:
        """Scores a page for retention. Higher means more valuable."""


@dataclass(slots=True)
class PolicyBase:
    """Convenience base class with no-op lifecycle hooks."""

    name: str

    def on_access(self, context: PolicyContext, page: KVPage, hit_tier: str | None) -> None:
        return None

    def on_miss(self, context: PolicyContext, page: KVPage) -> None:
        return None

    def on_insert(self, context: PolicyContext, page: KVPage, tier_name: str) -> None:
        return None

    def on_evict(self, context: PolicyContext, page: KVPage, tier_name: str) -> None:
        return None

    def on_promote(self, context: PolicyContext, page: KVPage, src_tier: str | None, dst_tier: str) -> None:
        return None

    def maybe_prefetch(self, context: PolicyContext, page: KVPage) -> list[str]:
        return []

    def score_page(self, context: PolicyContext, page: KVPage, tier_name: str) -> float:
        return 0.0
