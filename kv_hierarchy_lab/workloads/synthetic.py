"""Synthetic workload generators for long-context access traces."""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

from kv_hierarchy_lab.quant.footprint import quantized_size_bytes
from kv_hierarchy_lab.simulator.page import KVPage
from kv_hierarchy_lab.simulator.trace import TraceAccess


@dataclass(slots=True)
class SyntheticWorkload:
    """Bundle of pages and accesses for a simulator run."""

    name: str
    pages: dict[str, KVPage]
    accesses: list[TraceAccess]


def _build_pages(
    num_pages: int,
    base_scalars: int,
    quantization_scheme: str,
    seed: int,
) -> dict[str, KVPage]:
    rng = random.Random(seed)
    pages: dict[str, KVPage] = {}
    for index in range(num_pages):
        layer = index % 16
        head_group = index % 8
        num_scalars = base_scalars + rng.randint(-64, 64)
        size_bytes = quantized_size_bytes(max(1, num_scalars), quantization_scheme)
        page_id = f"page-{index}"
        pages[page_id] = KVPage(
            page_id=page_id,
            layer=layer,
            token_start=index * 16,
            token_end=index * 16 + 16,
            head_group=head_group,
            size_bytes=size_bytes,
            quantization_scheme=quantization_scheme,
        )
    return pages


def _wrap_trace(name: str, page_ids: Iterable[str], pages: dict[str, KVPage]) -> SyntheticWorkload:
    accesses = [
        TraceAccess(step=step, page_id=page_id, sequence_id="seq-0")
        for step, page_id in enumerate(page_ids)
    ]
    return SyntheticWorkload(name=name, pages=pages, accesses=accesses)


def generate_chat_continuation(
    num_pages: int = 64,
    length: int = 256,
    quantization_scheme: str = "fp16",
    seed: int = 7,
) -> SyntheticWorkload:
    """Generates a recency-heavy chat continuation workload."""
    rng = random.Random(seed)
    pages = _build_pages(num_pages, base_scalars=2048, quantization_scheme=quantization_scheme, seed=seed)
    working_set = [f"page-{i}" for i in range(min(12, num_pages))]
    trace: list[str] = []
    for step in range(length):
        if step % 17 == 0:
            working_set[rng.randrange(len(working_set))] = f"page-{rng.randrange(num_pages)}"
        trace.append(rng.choice(working_set[-8:]))
    return _wrap_trace("chat_continuation", trace, pages)


def generate_rag_burst(
    num_pages: int = 96,
    length: int = 320,
    quantization_scheme: str = "fp16",
    seed: int = 11,
) -> SyntheticWorkload:
    """Generates retrieval bursts with decode-locality in between."""
    rng = random.Random(seed)
    pages = _build_pages(num_pages, base_scalars=2304, quantization_scheme=quantization_scheme, seed=seed)
    hot = [f"page-{i}" for i in range(16)]
    retrieval = [f"page-{i}" for i in range(16, num_pages)]
    trace: list[str] = []
    for _ in range(length // 20):
        trace.extend(rng.choices(hot, k=14))
        trace.extend(rng.sample(retrieval, k=6))
    return _wrap_trace("rag_burst", trace[:length], pages)


def generate_periodic_reuse(
    num_pages: int = 80,
    length: int = 320,
    period: int = 9,
    quantization_scheme: str = "fp16",
    seed: int = 13,
) -> SyntheticWorkload:
    """Generates periodic reuse reminiscent of induction-style revisits."""
    pages = _build_pages(num_pages, base_scalars=2048, quantization_scheme=quantization_scheme, seed=seed)
    trace = [f"page-{(step % period) * 3 % num_pages}" for step in range(length)]
    return _wrap_trace("periodic_reuse", trace, pages)


def generate_long_tail_mix(
    num_pages: int = 128,
    length: int = 400,
    quantization_scheme: str = "fp16",
    seed: int = 17,
) -> SyntheticWorkload:
    """Generates a long-tail access mix with a small hot set and broad tail."""
    rng = random.Random(seed)
    pages = _build_pages(num_pages, base_scalars=2048, quantization_scheme=quantization_scheme, seed=seed)
    hot = [f"page-{i}" for i in range(10)]
    tail = [f"page-{i}" for i in range(10, num_pages)]
    weights = [1.0 / (idx + 1) for idx in range(len(tail))]
    trace: list[str] = []
    for _ in range(length):
        if rng.random() < 0.7:
            trace.append(rng.choice(hot))
        else:
            trace.append(rng.choices(tail, weights=weights, k=1)[0])
    return _wrap_trace("long_tail_mix", trace, pages)


def generate_prefetch_friendly(
    num_pages: int = 48,
    length: int = 240,
    quantization_scheme: str = "fp16",
    seed: int = 19,
) -> SyntheticWorkload:
    """Generates a predictable stride-like pattern."""
    pages = _build_pages(num_pages, base_scalars=2048, quantization_scheme=quantization_scheme, seed=seed)
    trace = [f"page-{step % num_pages}" for step in range(length)]
    return _wrap_trace("prefetch_friendly", trace, pages)


def generate_adversarial_prefetch(
    num_pages: int = 64,
    length: int = 256,
    quantization_scheme: str = "fp16",
    seed: int = 23,
) -> SyntheticWorkload:
    """Generates alternating jumps that confuse short-horizon prefetchers."""
    rng = random.Random(seed)
    pages = _build_pages(num_pages, base_scalars=2048, quantization_scheme=quantization_scheme, seed=seed)
    trace = []
    for step in range(length):
        if step % 2 == 0:
            trace.append(f"page-{rng.randrange(0, num_pages // 2)}")
        else:
            trace.append(f"page-{rng.randrange(num_pages // 2, num_pages)}")
    return _wrap_trace("adversarial_prefetch", trace, pages)


def generate_adversarial_burst(
    num_pages: int = 96,
    length: int = 320,
    burst_size: int = 12,
    quantization_scheme: str = "fp16",
    seed: int = 29,
) -> SyntheticWorkload:
    """Generates bursts that revisit recently evicted pages after short gaps."""
    rng = random.Random(seed)
    pages = _build_pages(num_pages, base_scalars=2112, quantization_scheme=quantization_scheme, seed=seed)
    early = [f"page-{index}" for index in range(num_pages // 3)]
    middle = [f"page-{index}" for index in range(num_pages // 3, 2 * num_pages // 3)]
    late = [f"page-{index}" for index in range(2 * num_pages // 3, num_pages)]
    trace: list[str] = []
    while len(trace) < length:
        anchor = rng.sample(early, k=min(burst_size, len(early)))
        trace.extend(anchor)
        trace.extend(rng.sample(middle, k=min(burst_size, len(middle))))
        trace.extend(anchor[: max(1, min(burst_size // 2, len(anchor)))])
        trace.extend(rng.sample(late, k=min(burst_size, len(late))))
    return _wrap_trace("adversarial_burst", trace[:length], pages)


def generate_mixed_locality(
    num_pages: int = 112,
    length: int = 360,
    quantization_scheme: str = "fp16",
    seed: int = 31,
) -> SyntheticWorkload:
    """Generates phased locality shifts across hot, tail, and periodic regions."""
    rng = random.Random(seed)
    pages = _build_pages(num_pages, base_scalars=2176, quantization_scheme=quantization_scheme, seed=seed)
    hot = [f"page-{index}" for index in range(12)]
    tail = [f"page-{index}" for index in range(12, num_pages)]
    trace: list[str] = []
    for step in range(length):
        phase = (step // 60) % 3
        if phase == 0:
            trace.append(rng.choice(hot))
        elif phase == 1:
            trace.append(rng.choice(tail[:20]))
        else:
            trace.append(f"page-{(step * 5) % num_pages}")
    return _wrap_trace("mixed_locality", trace, pages)
