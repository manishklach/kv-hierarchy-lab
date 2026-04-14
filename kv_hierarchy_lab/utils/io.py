"""I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> None:
    """Creates a directory tree if needed."""
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    """Reads JSON from disk."""
    return json.loads(path.read_text(encoding="utf-8"))
