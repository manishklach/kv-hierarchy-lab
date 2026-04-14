"""Synthetic workload exports."""

from kv_hierarchy_lab.workloads.synthetic import (
    SyntheticWorkload,
    generate_adversarial_prefetch,
    generate_chat_continuation,
    generate_long_tail_mix,
    generate_periodic_reuse,
    generate_prefetch_friendly,
    generate_rag_burst,
)

__all__ = [
    "SyntheticWorkload",
    "generate_adversarial_prefetch",
    "generate_chat_continuation",
    "generate_long_tail_mix",
    "generate_periodic_reuse",
    "generate_prefetch_friendly",
    "generate_rag_burst",
]
