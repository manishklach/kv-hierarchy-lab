"""Pattern metadata used by synthetic workload generators."""

PATTERN_NAMES = {
    "chat_continuation": "Recency-heavy continuation with local reuse.",
    "rag_burst": "Retrieval bursts interleaved with local decoding.",
    "periodic_reuse": "Induction-head-like periodic page revisits.",
    "long_tail_mix": "Mixed head-heavy and tail-heavy access mix.",
    "prefetch_friendly": "Predictable stride-like transition pattern.",
    "adversarial_prefetch": "Pattern that punishes naive prefetch.",
}
