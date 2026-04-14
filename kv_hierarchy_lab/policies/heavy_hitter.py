"""Frequency-biased baseline."""

from dataclasses import dataclass, field

from kv_hierarchy_lab.policies.base import PolicyBase, PolicyContext
from kv_hierarchy_lab.simulator.page import KVPage


@dataclass(slots=True)
class HeavyHitterPolicy(PolicyBase):
    """Retains frequently reused pages even if they are not the most recent."""

    name: str = "heavy_hitter"
    frequency: dict[str, int] = field(default_factory=dict)
    last_access_step: dict[str, int] = field(default_factory=dict)

    def on_access(self, context: PolicyContext, page: KVPage, hit_tier: str | None) -> None:
        self.frequency[page.page_id] = self.frequency.get(page.page_id, 0) + 1
        self.last_access_step[page.page_id] = context.step

    def score_page(self, context: PolicyContext, page: KVPage, tier_name: str) -> float:
        freq = self.frequency.get(page.page_id, 0)
        recency = self.last_access_step.get(page.page_id, -1)
        return freq * 10_000 + recency

    def select_eviction_candidate(
        self, context: PolicyContext, tier_name: str, candidates: list[KVPage], incoming_page: KVPage
    ) -> KVPage:
        return min(candidates, key=lambda candidate: self.score_page(context, candidate, tier_name))
