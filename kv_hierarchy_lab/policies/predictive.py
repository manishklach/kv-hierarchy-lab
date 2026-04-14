"""Lightweight predictive prefetch policy."""

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from kv_hierarchy_lab.policies.base import PolicyBase, PolicyContext
from kv_hierarchy_lab.simulator.page import KVPage


@dataclass(slots=True)
class PredictivePrefetchPolicy(PolicyBase):
    """Tracks local transitions and prefetches the most likely next page."""

    name: str = "predictive"
    last_page_id: str | None = None
    transitions: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    frequency: dict[str, int] = field(default_factory=dict)
    last_access_step: dict[str, int] = field(default_factory=dict)
    min_confidence: int = 2
    max_prefetches: int = 2

    def on_access(self, context: PolicyContext, page: KVPage, hit_tier: str | None) -> None:
        if self.last_page_id is not None:
            self.transitions[self.last_page_id][page.page_id] += 1
        self.last_page_id = page.page_id
        self.frequency[page.page_id] = self.frequency.get(page.page_id, 0) + 1
        self.last_access_step[page.page_id] = context.step

    def score_page(self, context: PolicyContext, page: KVPage, tier_name: str) -> float:
        freq = self.frequency.get(page.page_id, 0)
        recency = self.last_access_step.get(page.page_id, -1)
        return freq * 100 + recency

    def select_eviction_candidate(
        self, context: PolicyContext, tier_name: str, candidates: list[KVPage], incoming_page: KVPage
    ) -> KVPage:
        return min(candidates, key=lambda candidate: self.score_page(context, candidate, tier_name))

    def maybe_prefetch(self, context: PolicyContext, page: KVPage) -> list[str]:
        transition_counts = self.transitions.get(page.page_id)
        if not transition_counts:
            return []
        ranked = transition_counts.most_common(self.max_prefetches)
        return [page_id for page_id, count in ranked if count >= self.min_confidence]
