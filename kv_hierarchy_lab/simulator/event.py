"""Internal simulator events."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class SimulationEvent:
    """Captures an important simulator state transition."""

    step: int
    event_type: str
    page_id: str
    src_tier: str | None = None
    dst_tier: str | None = None
    latency_ms: float = 0.0
    bytes_moved: int = 0
    metadata: dict[str, str | int | float] = field(default_factory=dict)
