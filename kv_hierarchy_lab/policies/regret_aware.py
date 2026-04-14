"""Regret-aware eviction policy.

This policy treats a fast re-access after eviction as evidence that the eviction was costly.
Pages with higher accumulated regret become harder to evict in future decisions.
"""

from dataclasses import dataclass, field

from kv_hierarchy_lab.policies.base import PolicyBase, PolicyContext
from kv_hierarchy_lab.simulator.page import KVPage


@dataclass(slots=True)
class RegretAwarePolicy(PolicyBase):
    """Biases against evicting pages that historically caused painful re-accesses."""

    name: str = "regret_aware"
    regret_horizon: int = 24
    regret_weight: float = 6.0
    freq_weight: float = 1.5
    recency_weight: float = 1.0
    decay: float = 0.98
    frequency: dict[str, int] = field(default_factory=dict)
    last_access_step: dict[str, int] = field(default_factory=dict)
    eviction_step: dict[str, int] = field(default_factory=dict)
    regret_score: dict[str, float] = field(default_factory=dict)

    def on_access(self, context: PolicyContext, page: KVPage, hit_tier: str | None) -> None:
        self.frequency[page.page_id] = self.frequency.get(page.page_id, 0) + 1
        self.last_access_step[page.page_id] = context.step
        if page.page_id in self.regret_score:
            self.regret_score[page.page_id] *= self.decay
        prior_eviction = self.eviction_step.pop(page.page_id, None)
        if prior_eviction is None:
            return
        gap = context.step - prior_eviction
        if gap <= self.regret_horizon:
            penalty = (self.regret_horizon - gap + 1) / self.regret_horizon
            self.regret_score[page.page_id] = self.regret_score.get(page.page_id, 0.0) + penalty

    def on_evict(self, context: PolicyContext, page: KVPage, tier_name: str) -> None:
        self.eviction_step[page.page_id] = context.step

    def score_page(self, context: PolicyContext, page: KVPage, tier_name: str) -> float:
        frequency = self.frequency.get(page.page_id, 0)
        recency = self.last_access_step.get(page.page_id, -1)
        regret = self.regret_score.get(page.page_id, 0.0)
        return (
            self.freq_weight * frequency
            + self.recency_weight * recency / max(context.step, 1)
            + self.regret_weight * regret
        )

    def select_eviction_candidate(
        self, context: PolicyContext, tier_name: str, candidates: list[KVPage], incoming_page: KVPage
    ) -> KVPage:
        return min(candidates, key=lambda candidate: self.score_page(context, candidate, tier_name))
