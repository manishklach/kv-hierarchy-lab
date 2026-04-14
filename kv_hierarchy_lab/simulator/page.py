"""KV page model."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class KVPage:
    """Represents a logical KV cache page."""

    page_id: str
    layer: int
    token_start: int
    token_end: int
    head_group: int
    size_bytes: int
    quantization_scheme: str
    current_tier: str | None = None
    metadata: dict[str, str | int | float] = field(default_factory=dict)

    @property
    def token_span(self) -> tuple[int, int]:
        """Returns the inclusive-exclusive token span."""
        return self.token_start, self.token_end
