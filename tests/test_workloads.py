"""Synthetic workload tests."""

from kv_hierarchy_lab.workloads import (
    generate_adversarial_burst,
    generate_adversarial_prefetch,
    generate_chat_continuation,
    generate_mixed_locality,
    generate_prefetch_friendly,
)


def test_chat_workload_length_matches_request() -> None:
    """Generated trace length should match the requested length."""
    workload = generate_chat_continuation(length=50)
    assert len(workload.accesses) == 50
    assert len(workload.pages) == 64


def test_prefetch_friendly_pattern_is_regular() -> None:
    """The stride-like workload should cycle deterministically."""
    workload = generate_prefetch_friendly(num_pages=6, length=12)
    page_ids = [access.page_id for access in workload.accesses]
    assert page_ids[:6] == [f"page-{i}" for i in range(6)]
    assert page_ids[6:] == [f"page-{i}" for i in range(6)]


def test_adversarial_prefetch_visits_both_halves() -> None:
    """The adversarial pattern should span both address regions."""
    workload = generate_adversarial_prefetch(num_pages=10, length=40)
    visited = {access.page_id for access in workload.accesses}
    assert any(page_id in visited for page_id in {"page-0", "page-1", "page-2", "page-3", "page-4"})
    assert any(page_id in visited for page_id in {"page-5", "page-6", "page-7", "page-8", "page-9"})


def test_adversarial_burst_revisits_early_region() -> None:
    """Adversarial burst should return to pages after intervening pressure."""
    workload = generate_adversarial_burst(num_pages=18, length=72, burst_size=4)
    page_ids = [access.page_id for access in workload.accesses]
    assert len(set(page_ids[:4]) & set(page_ids[8:24])) > 0


def test_mixed_locality_changes_phases() -> None:
    """Mixed locality workload should show different address regions across phases."""
    workload = generate_mixed_locality(num_pages=24, length=180)
    first_phase = {access.page_id for access in workload.accesses[:60]}
    second_phase = {access.page_id for access in workload.accesses[60:120]}
    assert len(second_phase - first_phase) > 0
