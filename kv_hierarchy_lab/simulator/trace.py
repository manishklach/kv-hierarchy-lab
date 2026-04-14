"""Trace event definitions."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class TraceAccess:
    """Single page access entry for the simulator."""

    step: int
    page_id: str
    sequence_id: str
    is_prefetch_hint: bool = False
    metadata: dict[str, str | int | float] = field(default_factory=dict)
