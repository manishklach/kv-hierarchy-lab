"""Trace-driven simulator engine."""

from __future__ import annotations

from dataclasses import dataclass, field

from kv_hierarchy_lab.policies.base import BasePolicy, PolicyContext
from kv_hierarchy_lab.quant.schemes import QUANTIZATION_SCHEMES
from kv_hierarchy_lab.simulator.event import SimulationEvent
from kv_hierarchy_lab.simulator.metrics import MetricsCollector
from kv_hierarchy_lab.simulator.page import KVPage
from kv_hierarchy_lab.simulator.tier import MemoryTier
from kv_hierarchy_lab.simulator.trace import TraceAccess


@dataclass(slots=True)
class SimulationResult:
    """Structured result of a simulator run."""

    metrics: dict[str, int | float | dict[str, int | float]]
    events: list[SimulationEvent]


@dataclass(slots=True)
class SimulationEngine:
    """Executes a trace against a configurable tier hierarchy and policy."""

    tiers: list[MemoryTier]
    policy: BasePolicy
    pages: dict[str, KVPage]
    prefetch_enabled: bool = True
    max_prefetch_candidates: int = 2
    metrics: MetricsCollector = field(default_factory=MetricsCollector)
    events: list[SimulationEvent] = field(default_factory=list)
    access_history: list[str] = field(default_factory=list)
    pending_prefetches: set[str] = field(default_factory=set)

    def _build_context(self, access: TraceAccess) -> PolicyContext:
        resident_pages = {
            page_id: tier.name
            for tier in self.tiers
            for page_id in tier.resident_pages
        }
        tier_free_bytes = {tier.name: tier.capacity_bytes - tier.used_bytes for tier in self.tiers}
        return PolicyContext(
            step=access.step,
            access=access,
            resident_pages=resident_pages,
            tier_free_bytes=tier_free_bytes,
            pages=self.pages,
            access_history=list(self.access_history[-128:]),
        )

    def _lookup_resident_tier(self, page_id: str) -> MemoryTier | None:
        for tier in self.tiers:
            if tier.contains(page_id):
                return tier
        return None

    def _transfer_cost(self, src_tier: MemoryTier | None, dst_tier: MemoryTier, page: KVPage) -> float:
        src_read = src_tier.read_latency_ms if src_tier is not None else dst_tier.read_latency_ms * 4
        dst_write = dst_tier.write_latency_ms
        bandwidth = min(
            src_tier.bandwidth_bytes_per_ms if src_tier is not None else dst_tier.bandwidth_bytes_per_ms / 2,
            dst_tier.bandwidth_bytes_per_ms,
        )
        transfer_ms = page.size_bytes / max(bandwidth, 1.0)
        quant_penalty = QUANTIZATION_SCHEMES[page.quantization_scheme].decode_penalty_ms
        return src_read + dst_write + transfer_ms + quant_penalty

    def _ensure_capacity(self, target_tier: MemoryTier, page: KVPage, context: PolicyContext) -> None:
        while not target_tier.can_fit(page.size_bytes):
            candidates = [self.pages[page_id] for page_id in target_tier.resident_pages]
            victim = self.policy.select_eviction_candidate(
                context=context,
                tier_name=target_tier.name,
                candidates=candidates,
                incoming_page=page,
            )
            self._evict_from_tier(target_tier, victim, context)

    def _evict_from_tier(self, tier: MemoryTier, page: KVPage, context: PolicyContext) -> None:
        size_bytes = tier.remove_page(page.page_id)
        self.metrics.eviction_count += 1
        self.policy.on_evict(context, page, tier.name)
        if tier.name == self.tiers[0].name and page.page_id in self.pending_prefetches:
            self.pending_prefetches.remove(page.page_id)
            self.metrics.record_prefetch(useful=False)
        next_index = self.tiers.index(tier) + 1
        page.current_tier = None
        if next_index < len(self.tiers):
            lower_tier = self.tiers[next_index]
            self._ensure_capacity(lower_tier, page, context)
            lower_tier.add_page(page.page_id, size_bytes)
            page.current_tier = lower_tier.name
            latency_ms = self._transfer_cost(tier, lower_tier, page)
            self.metrics.record_transfer(size_bytes, latency_ms, is_prefetch=False)
            self.events.append(
                SimulationEvent(
                    step=context.step,
                    event_type="demote",
                    page_id=page.page_id,
                    src_tier=tier.name,
                    dst_tier=lower_tier.name,
                    latency_ms=latency_ms,
                    bytes_moved=size_bytes,
                )
            )
        else:
            self.events.append(
                SimulationEvent(
                    step=context.step,
                    event_type="evict",
                    page_id=page.page_id,
                    src_tier=tier.name,
                    bytes_moved=size_bytes,
                )
            )

    def _place_in_fast_tier(
        self,
        page: KVPage,
        context: PolicyContext,
        src_tier: MemoryTier | None,
        *,
        is_prefetch: bool,
    ) -> None:
        target_tier = self.tiers[0]
        if target_tier.contains(page.page_id):
            return
        self._ensure_capacity(target_tier, page, context)
        if src_tier is not None and src_tier.contains(page.page_id):
            src_tier.remove_page(page.page_id)
        target_tier.add_page(page.page_id, page.size_bytes)
        page.current_tier = target_tier.name
        latency_ms = self._transfer_cost(src_tier, target_tier, page)
        self.metrics.record_transfer(page.size_bytes, latency_ms, is_prefetch=is_prefetch)
        self.policy.on_insert(context, page, target_tier.name)
        self.policy.on_promote(
            context,
            page,
            src_tier.name if src_tier is not None else None,
            target_tier.name,
        )
        event_type = "prefetch" if is_prefetch else "promote"
        self.events.append(
            SimulationEvent(
                step=context.step,
                event_type=event_type,
                page_id=page.page_id,
                src_tier=src_tier.name if src_tier is not None else None,
                dst_tier=target_tier.name,
                latency_ms=latency_ms,
                bytes_moved=page.size_bytes,
            )
        )

    def _insert_cold(self, page: KVPage, context: PolicyContext) -> None:
        coldest_tier = self.tiers[-1]
        if not coldest_tier.contains(page.page_id):
            self._ensure_capacity(coldest_tier, page, context)
            coldest_tier.add_page(page.page_id, page.size_bytes)
            page.current_tier = coldest_tier.name
            self.policy.on_insert(context, page, coldest_tier.name)

    def run(self, accesses: list[TraceAccess]) -> SimulationResult:
        """Runs the provided access trace and returns aggregated metrics."""
        for access in accesses:
            page = self.pages[access.page_id]
            context = self._build_context(access)
            resident_tier = self._lookup_resident_tier(page.page_id)
            self.policy.on_access(context, page, resident_tier.name if resident_tier else None)

            if resident_tier is None:
                self.metrics.record_miss()
                self.policy.on_miss(context, page)
                self._insert_cold(page, context)
                resident_tier = self._lookup_resident_tier(page.page_id)
                self._place_in_fast_tier(page, context, resident_tier, is_prefetch=False)
            elif resident_tier.name != self.tiers[0].name:
                self.metrics.record_miss(resident_tier.name)
                self.policy.on_miss(context, page)
                self._place_in_fast_tier(page, context, resident_tier, is_prefetch=False)
            else:
                latency_ms = (
                    resident_tier.read_latency_ms
                    + QUANTIZATION_SCHEMES[page.quantization_scheme].decode_penalty_ms
                )
                self.metrics.record_hit(resident_tier.name, latency_ms)

            if page.page_id in self.pending_prefetches:
                self.pending_prefetches.remove(page.page_id)
                self.metrics.record_prefetch(useful=True)

            if self.prefetch_enabled:
                prefetch_candidates = self.policy.maybe_prefetch(context, page)[: self.max_prefetch_candidates]
                for candidate_id in prefetch_candidates:
                    candidate = self.pages.get(candidate_id)
                    if candidate is None:
                        continue
                    candidate_tier = self._lookup_resident_tier(candidate_id)
                    if candidate_tier is not None and candidate_tier.name == self.tiers[0].name:
                        continue
                    if candidate_tier is None:
                        self._insert_cold(candidate, context)
                        candidate_tier = self._lookup_resident_tier(candidate_id)
                    if candidate_tier is None:
                        continue
                    if candidate_id in self.pending_prefetches:
                        continue
                    self.pending_prefetches.add(candidate_id)
                    self.metrics.record_prefetch()
                    self._place_in_fast_tier(candidate, context, candidate_tier, is_prefetch=True)

            self.access_history.append(page.page_id)
            self.metrics.tier_peak_bytes = {tier.name: tier.peak_used_bytes for tier in self.tiers}

        self.metrics.wasted_prefetches += len(self.pending_prefetches)
        return SimulationResult(metrics=self.metrics.to_dict(), events=self.events)
