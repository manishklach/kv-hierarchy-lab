"""Memory tier model."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class MemoryTier:
    """Represents one simulated memory tier."""

    name: str
    capacity_bytes: int
    read_latency_ms: float
    write_latency_ms: float
    bandwidth_bytes_per_ms: float
    concurrency_limit: int | None = None
    resident_pages: dict[str, int] = field(default_factory=dict)
    used_bytes: int = 0
    peak_used_bytes: int = 0

    def can_fit(self, size_bytes: int) -> bool:
        """Returns true if a page can fit without further eviction."""
        return self.used_bytes + size_bytes <= self.capacity_bytes

    def contains(self, page_id: str) -> bool:
        """Returns true if the page is resident in this tier."""
        return page_id in self.resident_pages

    def add_page(self, page_id: str, size_bytes: int) -> None:
        """Adds a page to the tier and updates accounting."""
        if self.contains(page_id):
            return
        self.resident_pages[page_id] = size_bytes
        self.used_bytes += size_bytes
        self.peak_used_bytes = max(self.peak_used_bytes, self.used_bytes)

    def remove_page(self, page_id: str) -> int:
        """Removes a page from the tier and returns its size."""
        size_bytes = self.resident_pages.pop(page_id)
        self.used_bytes -= size_bytes
        return size_bytes
