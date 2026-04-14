"""Windowed recency policy."""

from collections import deque
from dataclasses import dataclass, field

from kv_hierarchy_lab.policies.base import PolicyBase, PolicyContext
from kv_hierarchy_lab.simulator.page import KVPage


@dataclass(slots=True)
class WindowedRecencyPolicy(PolicyBase):
    """Counts accesses in a recent fixed-size window, then breaks ties by recency."""

    window_size: int = 32
    name: str = "windowed_recency"
    recent_window: deque[str] = field(default_factory=deque)
    last_access_step: dict[str, int] = field(default_factory=dict)

    def on_access(self, context: PolicyContext, page: KVPage, hit_tier: str | None) -> None:
        self.recent_window.append(page.page_id)
        if len(self.recent_window) > self.window_size:
            self.recent_window.popleft()
        self.last_access_step[page.page_id] = context.step

    def score_page(self, context: PolicyContext, page: KVPage, tier_name: str) -> float:
        freq = sum(1 for page_id in self.recent_window if page_id == page.page_id)
        return freq * 1_000_000 + self.last_access_step.get(page.page_id, -1)

    def select_eviction_candidate(
        self, context: PolicyContext, tier_name: str, candidates: list[KVPage], incoming_page: KVPage
    ) -> KVPage:
        return min(candidates, key=lambda candidate: self.score_page(context, candidate, tier_name))
