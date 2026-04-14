"""Least-recently-used baseline."""

from dataclasses import dataclass, field

from kv_hierarchy_lab.policies.base import PolicyBase, PolicyContext
from kv_hierarchy_lab.simulator.page import KVPage


@dataclass(slots=True)
class LRUPolicy(PolicyBase):
    """Evicts the least recently accessed page."""

    name: str = "lru"
    last_access_step: dict[str, int] = field(default_factory=dict)

    def on_access(self, context: PolicyContext, page: KVPage, hit_tier: str | None) -> None:
        self.last_access_step[page.page_id] = context.step

    def on_insert(self, context: PolicyContext, page: KVPage, tier_name: str) -> None:
        self.last_access_step[page.page_id] = context.step

    def score_page(self, context: PolicyContext, page: KVPage, tier_name: str) -> float:
        return float(self.last_access_step.get(page.page_id, -1))

    def select_eviction_candidate(
        self, context: PolicyContext, tier_name: str, candidates: list[KVPage], incoming_page: KVPage
    ) -> KVPage:
        return min(candidates, key=lambda page: self.last_access_step.get(page.page_id, -1))
