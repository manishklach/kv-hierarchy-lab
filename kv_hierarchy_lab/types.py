"""Shared type aliases used across the project."""

from collections.abc import Iterable
from typing import TypeAlias

PageId: TypeAlias = str
TierName: TypeAlias = str
TraceLike: TypeAlias = Iterable["TraceAccess"]
